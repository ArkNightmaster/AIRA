# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose

from .config import AiraMoeConfig


class AiraMoeLayer(BaseTunerLayer):
    """
    Base AiraMoe layer that combines CoLA's collaborative adaptation with AwLoRA's three core technologies.
    """
    
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        
        # CoLA parameters (simplified to single A/B per adapter)
        
        # AwLoRA Core Technology parameters
        self.use_layer_wise_rank = {}
        self.use_awsvd_init = {}
        self.use_activation_aware = {}
        self.activation_aware_mode = {}
        self.activation_normalize = {}
        
        # For AwSVD initialization
        self.awsvd_W_prime = {}
        self.awsvd_S_diag = {}
        self.activation_stats_collected = {}
        self.awsvd_collect_steps = {}
        self.current_step = 0
        
        # For layer-wise rank allocation
        self.layer_importance = {}
        self.allocated_rank = {}
        
        # Embedding layers
        self.lora_embedding_A = nn.ModuleDict({})
        self.lora_embedding_B = nn.ModuleDict({})
        
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "Params4bit":
            # Bitsandbytes
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "EetqLinear":
            # Eetq
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "linear"):
            if hasattr(base_layer.linear, "in_features") and hasattr(base_layer.linear, "out_features"):
                in_features, out_features = base_layer.linear.in_features, base_layer.linear.out_features
            else:
                in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            in_features, out_features = None, None

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        # removed num_A/num_B in simplified AIRA
        init_lora_weights: Union[bool, str],
        use_layer_wise_rank: bool = False,
        use_awsvd_init: bool = False,
        use_activation_aware: bool = False,
        activation_aware_mode: str = "inps",
        activation_normalize: bool = True,
        awsvd_collect_steps: int = 100,
        **kwargs,
    ) -> None:
        """Update layer with AiraMoe configuration."""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = 2
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        
        # AwLoRA parameters
        self.use_layer_wise_rank[adapter_name] = use_layer_wise_rank
        self.use_awsvd_init[adapter_name] = use_awsvd_init
        self.use_activation_aware[adapter_name] = use_activation_aware
        self.activation_aware_mode[adapter_name] = activation_aware_mode
        self.activation_normalize[adapter_name] = activation_normalize
        self.awsvd_collect_steps[adapter_name] = awsvd_collect_steps
        self.activation_stats_collected[adapter_name] = False

        # Create single A and B matrices
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        # Initialize weights
        if init_lora_weights == "awsvd":
            # AwSVD initialization will be done during forward pass
            # For now, use standard initialization
            self._init_lora_weights_standard(adapter_name)
        else:
            self._init_lora_weights(adapter_name, init_lora_weights)

        # Move to device and convert to correct dtype
        self._move_adapter_to_device_of_base_layer(adapter_name)
        
        # Additional manual dtype conversion for adapter modules
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None and (weight.dtype.is_floating_point or weight.dtype.is_complex):
            self.lora_A[adapter_name] = self.lora_A[adapter_name].to(weight.dtype)
            self.lora_B[adapter_name] = self.lora_B[adapter_name].to(weight.dtype)

        self.set_adapter(self.active_adapters)

    def _init_lora_weights_standard(self, adapter_name: str) -> None:
        """Standard LoRA weight initialization."""
        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[adapter_name].weight)

    def _init_lora_weights(self, adapter_name: str, init_lora_weights: Union[bool, str]) -> None:
        """Initialize LoRA weights based on the specified method."""
        if init_lora_weights is False:
            # Random initialization
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            nn.init.normal_(self.lora_B[adapter_name].weight, std=1 / self.r[adapter_name])
        elif init_lora_weights is True:
            # Default initialization
            self._init_lora_weights_standard(adapter_name)
        elif init_lora_weights.lower() == "gaussian":
            # Gaussian initialization
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights.lower() == "pissa":
            # PiSSA initialization (similar to CoLA)
            self._pissa_init(adapter_name)
        else:
            # Default to standard initialization
            self._init_lora_weights_standard(adapter_name)

    def _pissa_init(self, adapter_name: str) -> None:
        """PiSSA initialization similar to CoLA."""
        weight = self.get_base_layer().weight.to(torch.float32)
        V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
        
        # Extract top-r components
        Vr = V[:, :self.r[adapter_name]]
        Sr = S[:self.r[adapter_name]] / self.scaling[adapter_name]
        Uhr = Uh[:self.r[adapter_name]]
        
        # Distribute among multiple A and B matrices
        lora_A = (torch.diag(torch.sqrt(Sr)) @ Uhr)
        lora_B = (Vr @ torch.diag(torch.sqrt(Sr)))
        
        # Convert back to original dtype for LoRA parameters
        original_dtype = self.lora_A[adapter_name].weight.dtype
        lora_A = lora_A.to(original_dtype)
        lora_B = lora_B.to(original_dtype)
        
        # Initialize all matrices
        self.lora_A[adapter_name].weight.data = lora_A.clone()
        self.lora_B[adapter_name].weight.data = lora_B.clone()
        
        # Update base weight (convert back to original dtype)
        delta_weight = self.scaling[adapter_name] * (lora_B.to(torch.float32) @ lora_A.to(torch.float32))
        weight.data -= delta_weight

    def _awsvd_init(self, adapter_name: str, inputs: torch.Tensor) -> None:
        """
        AwLoRA Core Technology 2: AwSVD-based initialization.
        """
        if self.activation_stats_collected[adapter_name]:
            return
            
        # Convert all tensors to float32 for SVD computation
        weight = self.get_base_layer().weight.to(torch.float32)
        inputs_f32 = inputs.to(torch.float32)
        
        # Calculate activation-aware scaling matrix S
        X = inputs_f32.view(-1, weight.size(1))  # Flatten to [batch*seq, in_features]
        S_diag = torch.sqrt(torch.mean(X**2, dim=0))  # RMS of each input feature
        S = torch.diag(S_diag)
        
        # Calculate scaled weight matrix W'
        W_prime = (weight @ S).to(torch.float32)
        
        # SVD decomposition of W' (ensure float32 for SVD)
        U, S_vals, Vh = torch.linalg.svd(W_prime, full_matrices=False)
        U = U[:, :self.r[adapter_name]]
        S_vals = S_vals[:self.r[adapter_name]]
        Vh = Vh[:self.r[adapter_name], :]
        
        # Store for initialization
        self.awsvd_W_prime[adapter_name] = W_prime
        self.awsvd_S_diag[adapter_name] = S_diag
        
        # Initialize LoRA matrices with inverse scaling
        S_inv = torch.diag(1.0 / (S_diag + 1e-8))  # Add epsilon for numerical stability
        
        # Distribute among multiple A and B matrices
        lora_A = (Vh @ S_inv)
        lora_B = (U @ torch.diag(S_vals))
        
        # Convert back to original dtype for LoRA parameters
        original_dtype = self.lora_A[adapter_name].weight.dtype
        lora_A = lora_A.to(original_dtype)
        lora_B = lora_B.to(original_dtype)
        
        # Initialize all matrices
        self.lora_A[adapter_name].weight.data = lora_A.clone()
        self.lora_B[adapter_name].weight.data = lora_B.clone()
            
        self.activation_stats_collected[adapter_name] = True

    def _compute_activation_weights(self, x: torch.Tensor, mode: str, normalize: bool = True) -> torch.Tensor:
        """
        AwLoRA Core Technology 3: Compute activation-aware weights.
        """
        with torch.no_grad():
            if mode == "inps":
                # Input-based activation weighting
                # For input x with shape [batch_size, seq_len, hidden_size]
                # We want to compute weights over the feature dimension (last dim)
                # Average over batch and sequence dimensions to get [hidden_size]
                if x.dim() == 3:  # [batch_size, seq_len, hidden_size]
                    s_i = torch.mean(torch.abs(x), dim=(0, 1))  # [hidden_size]
                elif x.dim() == 2:  # [batch_size, hidden_size]
                    s_i = torch.mean(torch.abs(x), dim=0)  # [hidden_size]
                else:
                    # For other dimensions, average over all but the last dimension
                    dims_to_avg = tuple(range(x.dim() - 1))
                    s_i = torch.mean(torch.abs(x), dim=dims_to_avg)  # [hidden_size]
            elif mode == "outps":
                # Output-based activation weighting (computed after base layer)
                base_output = F.linear(x, self.get_base_layer().weight, self.get_base_layer().bias)
                # Same dimension handling as above
                if base_output.dim() == 3:  # [batch_size, seq_len, out_features]
                    s_i = torch.mean(torch.abs(base_output), dim=(0, 1))  # [out_features]
                elif base_output.dim() == 2:  # [batch_size, out_features]
                    s_i = torch.mean(torch.abs(base_output), dim=0)  # [out_features]
                else:
                    # For other dimensions, average over all but the last dimension
                    dims_to_avg = tuple(range(base_output.dim() - 1))
                    s_i = torch.mean(torch.abs(base_output), dim=dims_to_avg)  # [out_features]
            else:
                raise ValueError(f"Unknown activation aware mode: {mode}")
            
            if normalize:
                # Normalize to [0, 1] range
                s_min, s_max = s_i.min(), s_i.max()
                if s_max > s_min:
                    s_i = (s_i - s_min) / (s_max - s_min)
                else:
                    s_i = torch.ones_like(s_i)
                    
        return s_i.detach()

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: Union[bool, str]) -> None:
        """Reset LoRA parameters."""
        if init_lora_weights is False:
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            nn.init.normal_(self.lora_B[adapter_name].weight, std=1 / self.r[adapter_name])
        elif init_lora_weights is True:
            self._init_lora_weights_standard(adapter_name)
        elif init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights.lower() == "pissa":
            self._pissa_init(adapter_name)

    def set_scale(self, adapter: str, scale: float) -> None:
        """Set absolute scale for a specific adapter following LoRA semantics."""
        if adapter not in self.scaling:
            return
        r = max(1, int(self.r.get(adapter, 1)))
        lora_alpha = float(self.lora_alpha.get(adapter, 1))
        self.scaling[adapter] = float(scale) * (lora_alpha / float(r))

    def scale_layer(self, *args, **kwargs) -> None:
        """Scale adapter(s).

        Compatible with two calling conventions:
        - scale_layer(scale: float) -> scales all active adapters
        - scale_layer(adapter_name: str, scale: float) -> scales one adapter
        """
        if len(args) == 1 and isinstance(args[0], (int, float)):
            scale = float(args[0])
            for active_adapter in self.active_adapters:
                if active_adapter in self.scaling:
                    self.scaling[active_adapter] *= scale
            return

        if len(args) >= 2:
            adapter_name, scale = args[0], float(args[1])
            if adapter_name in self.scaling:
                self.scaling[adapter_name] *= scale
            return

        # fallback to keyword usage
        adapter_name = kwargs.get("adapter_name", None)
        scale = kwargs.get("scaling", None)
        if adapter_name is not None and scale is not None and adapter_name in self.scaling:
            self.scaling[adapter_name] *= float(scale)

    def unscale_layer(self, *args, **kwargs) -> None:
        """Unscale adapter(s).

        Compatible with two calling conventions:
        - unscale_layer() -> reset all active adapters' scaling to lora_alpha / r
        - unscale_layer(scale: float) -> divide all active adapters' scaling by scale
        - unscale_layer(adapter_name: str, scale: float) -> divide one adapter's scaling by scale
        """
        # adapter-specific form
        if len(args) >= 2 and isinstance(args[0], str):
            adapter_name, scale = args[0], float(args[1])
            if adapter_name in self.scaling:
                self.scaling[adapter_name] /= scale
            return

        # all-active form
        if len(args) == 1 and isinstance(args[0], (int, float)):
            scale = float(args[0])
            for active_adapter in self.active_adapters:
                if active_adapter in self.scaling:
                    self.scaling[active_adapter] /= scale
            return

        # reset to base
        for active_adapter in self.active_adapters:
            if active_adapter not in self.scaling:
                continue
            r = max(1, int(self.r.get(active_adapter, 1)))
            lora_alpha = float(self.lora_alpha.get(active_adapter, 1))
            self.scaling[active_adapter] = lora_alpha / float(r)


class Linear(nn.Module, AiraMoeLayer):
    """
    AiraMoe Linear layer that combines CoLA's collaborative adaptation with AwLoRA's three core technologies.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        num_A: int = 1,
        num_B: int = 1,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_layer_wise_rank: bool = False,
        use_awsvd_init: bool = False,
        use_activation_aware: bool = False,
        activation_aware_mode: str = "inps",
        activation_normalize: bool = True,
        awsvd_collect_steps: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()
        AiraMoeLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_A=num_A,
            num_B=num_B,
            init_lora_weights=init_lora_weights,
            use_layer_wise_rank=use_layer_wise_rank,
            use_awsvd_init=use_awsvd_init,
            use_activation_aware=use_activation_aware,
            activation_aware_mode=activation_aware_mode,
            activation_normalize=activation_normalize,
            awsvd_collect_steps=awsvd_collect_steps,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)

        if self.merged_adapters:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.fan_in_fan_out:
                        orig_weights += delta_weight
                    else:
                        orig_weights += delta_weight.T

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.fan_in_fan_out:
                        base_layer.weight.data += delta_weight
                    else:
                        base_layer.weight.data += delta_weight.T

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged_adapters:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.fan_in_fan_out:
                    self.get_base_layer().weight.data -= delta_weight
                else:
                    self.get_base_layer().weight.data -= delta_weight.T

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.
        """
        # Support both single Linear (nn.Linear) and multi-matrix (ModuleList) implementations
        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]

        if isinstance(lora_A, nn.ModuleList):
            weight_A = torch.sum(torch.stack([layer.weight for layer in lora_A]), dim=0)
        else:
            weight_A = lora_A.weight

        if isinstance(lora_B, nn.ModuleList):
            weight_B = torch.sum(torch.stack([layer.weight for layer in lora_B]), dim=0)
        else:
            weight_B = lora_B.weight

        device = weight_B.device
        dtype = weight_B.dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        # Safety checks similar to LoRA
        if adapter_names is not None and self.merged_adapters:
            raise ValueError("Cannot pass `adapter_names` when there are merged adapters, please call `unmerge` first.")

        if self.disable_adapters:
            if self.merged_adapters:
                # Auto unmerge to mirror LoRA semantics
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            
            # Deduplicate active adapters to avoid reusing the same parameters twice per forward
            if isinstance(self.active_adapters, (list, tuple)):
                _unique_active_adapters = []
                _seen = set()
                for _a in self.active_adapters:
                    if _a not in _seen:
                        _unique_active_adapters.append(_a)
                        _seen.add(_a)
            else:
                _unique_active_adapters = [self.active_adapters]

            for active_adapter in _unique_active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                    
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                
                # AwLoRA Core Technology 2: AwSVD initialization is handled before training
                # Initialization is now controlled by trainer._initialize_aira_moe_with_activations()
                
                # AwLoRA Core Technology 3: Activation-aware weighting
                if self.use_activation_aware[active_adapter]:
                    activation_weights = self._compute_activation_weights(
                        x,
                        self.activation_aware_mode[active_adapter],
                        self.activation_normalize[active_adapter]
                    )

                    if self.activation_aware_mode[active_adapter] == "inps":
                        # Apply weights to input: compute sum over A first, then apply each B once
                        weight_shape = [1] * (x.dim() - 1) + [activation_weights.size(-1)]
                        activation_weights_expanded = activation_weights.view(*weight_shape)
                        x_weighted = dropout(x) * activation_weights_expanded

                        out_A = lora_A(x_weighted)
                        result = result + (lora_B(out_A) * scaling)
                    else:  # outps mode
                        # Compute sum over A first using dropout(x), then apply each B once; weight outputs afterward
                        # ensure dtype alignment
                        x_drop = dropout(x.to(lora_A.weight.dtype))
                        lora_output = lora_B(lora_A(x_drop))
                        weight_shape = [1] * (lora_output.dim() - 1) + [activation_weights.size(-1)]
                        activation_weights_expanded = activation_weights.view(*weight_shape)
                        result = result + ((lora_output * activation_weights_expanded) * scaling)
                else:
                    # Standard CoLA: compute sum over A first, then apply each B once
                    x_drop = dropout(x.to(lora_A.weight.dtype))
                    result = result + (lora_B(lora_A(x_drop)) * scaling)

            result = result.to(torch_result_dtype)

        return result

    def _check_forward_args(self, x, *args, **kwargs):
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return
        if len(x) != len(adapter_names):
            raise ValueError(
                "Length of `adapter_names` should be the same as the number of inputs, "
                f"but got {len(adapter_names)} and {len(x)} respectively."
            )
        if self.merged_adapters:
            raise ValueError("Cannot pass `adapter_names` when there are merged adapters, please call `unmerge` first.")

    def _mixed_batch_forward(self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, adapter_names=adapter_names, **kwargs)
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = list(set(adapter_names))
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Embedding, AiraMoeLayer):
    """
    AiraMoe Embedding layer.
    """

    def __init__(
        self,
        base_layer: nn.Embedding,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        num_A: int = 1,
        num_B: int = 1,
        init_lora_weights: Union[bool, str] = True,
        use_layer_wise_rank: bool = False,
        use_awsvd_init: bool = False,
        use_activation_aware: bool = False,
        activation_aware_mode: str = "inps",
        activation_normalize: bool = True,
        awsvd_collect_steps: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()
        AiraMoeLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_A=num_A,
            num_B=num_B,
            init_lora_weights=init_lora_weights,
            use_layer_wise_rank=use_layer_wise_rank,
            use_awsvd_init=use_awsvd_init,
            use_activation_aware=use_activation_aware,
            activation_aware_mode=activation_aware_mode,
            activation_normalize=activation_normalize,
            awsvd_collect_steps=awsvd_collect_steps,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, num_A, num_B, init_lora_weights, **kwargs):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = 2
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        
        # CoLA parameters
        self.num_A[adapter_name] = num_A
        self.num_B[adapter_name] = num_B
        
        # AwLoRA parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                getattr(self, key)[adapter_name] = value

        # Create embedding-specific LoRA layers
        self.lora_embedding_A[adapter_name] = nn.ModuleList([
            nn.Embedding(self.in_features, r) for _ in range(num_A)
        ])
        self.lora_embedding_B[adapter_name] = nn.ModuleList([
            nn.Linear(r, self.out_features, bias=False) for _ in range(num_B)
        ])

        # Initialize weights
        self.reset_lora_parameters(adapter_name, init_lora_weights)

        # Move to device and convert to correct dtype
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                for i in range(num_A):
                    self.lora_embedding_A[adapter_name][i].to(weight.device, dtype=weight.dtype)
                for i in range(num_B):
                    self.lora_embedding_B[adapter_name][i].to(weight.device, dtype=weight.dtype)

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: Union[bool, str]) -> None:
        if adapter_name in self.lora_embedding_A.keys():
            if init_lora_weights is False:
                for i in range(self.num_A[adapter_name]):
                    nn.init.normal_(self.lora_embedding_A[adapter_name][i].weight, std=1 / self.r[adapter_name])
                for i in range(self.num_B[adapter_name]):
                    nn.init.normal_(self.lora_embedding_B[adapter_name][i].weight, std=1 / self.r[adapter_name])
            elif init_lora_weights is True:
                for i in range(self.num_A[adapter_name]):
                    nn.init.zeros_(self.lora_embedding_A[adapter_name][i].weight)
                for i in range(self.num_B[adapter_name]):
                    nn.init.normal_(self.lora_embedding_B[adapter_name][i].weight, std=1 / self.r[adapter_name])
            else:
                self.reset_lora_parameters(adapter_name, True)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.disable_adapters:
            if self.merged_adapters:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_embedding_A:
                continue
                
            lora_embedding_A = self.lora_embedding_A[active_adapter]
            lora_embedding_B = self.lora_embedding_B[active_adapter]
            scaling = self.scaling[active_adapter]
            
            # CoLA collaborative strategy for embeddings
            for i in range(self.num_A[active_adapter]):
                for j in range(self.num_B[active_adapter]):
                    after_A = lora_embedding_A[i](x)
                    result += lora_embedding_B[j](after_A) * scaling

        return result

    def get_delta_weight(self, adapter) -> torch.Tensor:
        if adapter not in self.lora_embedding_A:
            return torch.zeros_like(self.get_base_layer().weight)

        weight_A_list = self.lora_embedding_A[adapter]
        weight_B_list = self.lora_embedding_B[adapter]

        # Sum all A and B then compute B @ A (embedding: treat as linear over one-hot basis)
        if isinstance(weight_A_list, nn.ModuleList):
            A = torch.sum(torch.stack([layer.weight for layer in weight_A_list]), dim=0)  # [num_embeddings, r]
        else:
            A = weight_A_list.weight  # type: ignore[attr-defined]

        if isinstance(weight_B_list, nn.ModuleList):
            B = torch.sum(torch.stack([layer.weight for layer in weight_B_list]), dim=0)  # [out_features, r]
        else:
            B = weight_B_list.weight  # type: ignore[attr-defined]

        device = B.device
        dtype = A.dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        if cast_to_fp32:
            A = A.float()
            B = B.float()

        # For embeddings, delta W has shape [num_embeddings, out_features] and transposed in LoRA merge
        delta = (B @ A.T).T * self.scaling[adapter]
        if cast_to_fp32:
            delta = delta.to(dtype=dtype)
        return delta

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        base_layer = self.get_base_layer()
        for active_adapter in adapter_names:
            if active_adapter not in self.lora_embedding_A:
                continue
            delta = self.get_delta_weight(active_adapter)
            if safe_merge:
                orig = base_layer.weight.data.clone()
                orig += delta
                if not torch.isfinite(orig).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = orig
            else:
                base_layer.weight.data += delta
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged_adapters:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_embedding_A:
                continue
            delta = self.get_delta_weight(active_adapter)
            base_layer.weight.data -= delta

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Conv2d, AiraMoeLayer):
    """
    AiraMoe Conv2d layer.
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        num_A: int = 1,
        num_B: int = 1,
        init_lora_weights: Union[bool, str] = True,
        use_layer_wise_rank: bool = False,
        use_awsvd_init: bool = False,
        use_activation_aware: bool = False,
        activation_aware_mode: str = "inps",
        activation_normalize: bool = True,
        awsvd_collect_steps: int = 100,
        **kwargs,
    ):
        super().__init__()
        AiraMoeLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_A=num_A,
            num_B=num_B,
            init_lora_weights=init_lora_weights,
            use_layer_wise_rank=use_layer_wise_rank,
            use_awsvd_init=use_awsvd_init,
            use_activation_aware=use_activation_aware,
            activation_aware_mode=activation_aware_mode,
            activation_normalize=activation_normalize,
            awsvd_collect_steps=awsvd_collect_steps,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, num_A, num_B, init_lora_weights, **kwargs):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = 2
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        
        # CoLA parameters
        self.num_A[adapter_name] = num_A
        self.num_B[adapter_name] = num_B
        
        # AwLoRA parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                getattr(self, key)[adapter_name] = value

        # Create Conv2d-specific LoRA layers
        kernel_size = self.get_base_layer().kernel_size
        stride = self.get_base_layer().stride
        padding = self.get_base_layer().padding
        
        self.lora_A[adapter_name] = nn.ModuleList([
            nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False) for _ in range(num_A)
        ])
        self.lora_B[adapter_name] = nn.ModuleList([
            nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False) for _ in range(num_B)
        ])

        # Initialize weights
        self.reset_lora_parameters(adapter_name, init_lora_weights)

        # Move to device and convert to correct dtype
        self._move_adapter_to_device_of_base_layer(adapter_name)
        
        # Additional manual dtype conversion for ModuleList items
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None and (weight.dtype.is_floating_point or weight.dtype.is_complex):
            # Convert each module in the ModuleList to the correct dtype
            self.lora_A[adapter_name] = self.lora_A[adapter_name].to(weight.dtype)
            self.lora_B[adapter_name] = self.lora_B[adapter_name].to(weight.dtype)

        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged_adapters:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
                
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            
            # CoLA collaborative strategy for Conv2d
            for i in range(self.num_A[active_adapter]):
                for j in range(self.num_B[active_adapter]):
                    result += lora_B[j](lora_A[i](dropout(x.to(lora_A[i].weight.dtype)))) * scaling

        return result

    def get_delta_weight(self, adapter) -> torch.Tensor:
        if adapter not in self.lora_A:
            return torch.zeros_like(self.get_base_layer().weight)

        weight_A_list = self.lora_A[adapter]
        weight_B_list = self.lora_B[adapter]

        if isinstance(weight_A_list, nn.ModuleList):
            weight_A = torch.sum(torch.stack([m.weight for m in weight_A_list]), dim=0)
        else:
            weight_A = weight_A_list.weight

        if isinstance(weight_B_list, nn.ModuleList):
            weight_B = torch.sum(torch.stack([m.weight for m in weight_B_list]), dim=0)
        else:
            weight_B = weight_B_list.weight

        device = weight_B.device
        dtype = weight_A.dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # 1x1 conv special case like LoRA
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            delta = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            delta = delta * self.scaling[adapter]
        else:
            delta = F.conv2d(weight_A.permute(1, 0, 2, 3), weight_B).permute(1, 0, 2, 3)
            delta = delta * self.scaling[adapter]

        if cast_to_fp32:
            delta = delta.to(dtype=dtype)
        return delta

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        base_layer = self.get_base_layer()
        for active_adapter in adapter_names:
            if active_adapter not in self.lora_A:
                continue
            delta = self.get_delta_weight(active_adapter)
            if safe_merge:
                orig = base_layer.weight.data.clone()
                orig += delta
                if not torch.isfinite(orig).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = orig
            else:
                base_layer.weight.data += delta
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged_adapters:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_A:
                continue
            delta = self.get_delta_weight(active_adapter)
            base_layer.weight.data -= delta

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep 


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: AiraMoeConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    """Dispatcher for AiraMoe default modules, mirroring LoRA behavior."""
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs.get("fan_in_fan_out", False):
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting it to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs.get("fan_in_fan_out", False):
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting it to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module