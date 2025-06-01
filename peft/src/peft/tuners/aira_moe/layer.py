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
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "num_A", "num_B")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        
        # CoLA parameters
        self.num_A = {}
        self.num_B = {}
        
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
        num_A: int,
        num_B: int,
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
        self.scaling[adapter_name] = lora_alpha / r
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        
        # CoLA parameters
        self.num_A[adapter_name] = num_A
        self.num_B[adapter_name] = num_B
        
        # AwLoRA parameters
        self.use_layer_wise_rank[adapter_name] = use_layer_wise_rank
        self.use_awsvd_init[adapter_name] = use_awsvd_init
        self.use_activation_aware[adapter_name] = use_activation_aware
        self.activation_aware_mode[adapter_name] = activation_aware_mode
        self.activation_normalize[adapter_name] = activation_normalize
        self.awsvd_collect_steps[adapter_name] = awsvd_collect_steps
        self.activation_stats_collected[adapter_name] = False

        # Create multiple A and B matrices (CoLA style)
        self.lora_A[adapter_name] = nn.ModuleList([
            nn.Linear(self.in_features, r, bias=False) for _ in range(num_A)
        ])
        self.lora_B[adapter_name] = nn.ModuleList([
            nn.Linear(r, self.out_features, bias=False) for _ in range(num_B)
        ])

        # Initialize weights
        if init_lora_weights == "awsvd":
            # AwSVD initialization will be done during forward pass
            # For now, use standard initialization
            self._init_lora_weights_standard(adapter_name)
        else:
            self._init_lora_weights(adapter_name, init_lora_weights)

        # Move to device and convert to correct dtype
        self._move_adapter_to_device_of_base_layer(adapter_name)
        
        # Additional manual dtype conversion for ModuleList items
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None and (weight.dtype.is_floating_point or weight.dtype.is_complex):
            # Convert each module in the ModuleList to the correct dtype
            for i in range(num_A):
                self.lora_A[adapter_name][i] = self.lora_A[adapter_name][i].to(weight.dtype)
            for i in range(num_B):
                self.lora_B[adapter_name][i] = self.lora_B[adapter_name][i].to(weight.dtype)

        self.set_adapter(self.active_adapters)

    def _init_lora_weights_standard(self, adapter_name: str) -> None:
        """Standard LoRA weight initialization."""
        for i in range(self.num_A[adapter_name]):
            nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
        for i in range(self.num_B[adapter_name]):
            nn.init.zeros_(self.lora_B[adapter_name][i].weight)

    def _init_lora_weights(self, adapter_name: str, init_lora_weights: Union[bool, str]) -> None:
        """Initialize LoRA weights based on the specified method."""
        if init_lora_weights is False:
            # Random initialization
            for i in range(self.num_A[adapter_name]):
                nn.init.normal_(self.lora_A[adapter_name][i].weight, std=1 / self.r[adapter_name])
            for i in range(self.num_B[adapter_name]):
                nn.init.normal_(self.lora_B[adapter_name][i].weight, std=1 / self.r[adapter_name])
        elif init_lora_weights is True:
            # Default initialization
            self._init_lora_weights_standard(adapter_name)
        elif init_lora_weights.lower() == "gaussian":
            # Gaussian initialization
            for i in range(self.num_A[adapter_name]):
                nn.init.normal_(self.lora_A[adapter_name][i].weight, std=1 / self.r[adapter_name])
            for i in range(self.num_B[adapter_name]):
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)
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
        lora_A = (torch.diag(torch.sqrt(Sr)) @ Uhr) / self.num_A[adapter_name]
        lora_B = (Vr @ torch.diag(torch.sqrt(Sr))) / self.num_B[adapter_name]
        
        # Convert back to original dtype for LoRA parameters
        original_dtype = self.lora_A[adapter_name][0].weight.dtype
        lora_A = lora_A.to(original_dtype)
        lora_B = lora_B.to(original_dtype)
        
        # Initialize all matrices
        for i in range(self.num_A[adapter_name]):
            self.lora_A[adapter_name][i].weight.data = lora_A.clone()
        for i in range(self.num_B[adapter_name]):
            self.lora_B[adapter_name][i].weight.data = lora_B.clone()
        
        # Update base weight (convert back to original dtype)
        delta_weight = self.scaling[adapter_name] * (
            (lora_B.to(torch.float32) * self.num_B[adapter_name]) @ (lora_A.to(torch.float32) * self.num_A[adapter_name])
        )
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
        lora_A = (Vh @ S_inv) / self.num_A[adapter_name]
        lora_B = (U @ torch.diag(S_vals)) / self.num_B[adapter_name]
        
        # Convert back to original dtype for LoRA parameters
        original_dtype = self.lora_A[adapter_name][0].weight.dtype
        lora_A = lora_A.to(original_dtype)
        lora_B = lora_B.to(original_dtype)
        
        # Initialize all matrices
        for i in range(self.num_A[adapter_name]):
            self.lora_A[adapter_name][i].weight.data = lora_A.clone()
        for i in range(self.num_B[adapter_name]):
            self.lora_B[adapter_name][i].weight.data = lora_B.clone()
            
        self.activation_stats_collected[adapter_name] = True

    def _compute_activation_weights(self, x: torch.Tensor, mode: str, normalize: bool = True) -> torch.Tensor:
        """
        AwLoRA Core Technology 3: Compute activation-aware weights.
        """
        with torch.no_grad():
            if mode == "inps":
                # Input-based activation weighting
                s_i = torch.mean(torch.abs(x), dim=0)  # [in_features]
            elif mode == "outps":
                # Output-based activation weighting (computed after base layer)
                base_output = F.linear(x, self.get_base_layer().weight, self.get_base_layer().bias)
                s_i = torch.mean(torch.abs(base_output), dim=0)  # [out_features]
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
            for i in range(self.num_A[adapter_name]):
                nn.init.normal_(self.lora_A[adapter_name][i].weight, std=1 / self.r[adapter_name])
            for i in range(self.num_B[adapter_name]):
                nn.init.normal_(self.lora_B[adapter_name][i].weight, std=1 / self.r[adapter_name])
        elif init_lora_weights is True:
            self._init_lora_weights_standard(adapter_name)
        elif init_lora_weights.lower() == "gaussian":
            for i in range(self.num_A[adapter_name]):
                nn.init.normal_(self.lora_A[adapter_name][i].weight, std=1 / self.r[adapter_name])
            for i in range(self.num_B[adapter_name]):
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)
        elif init_lora_weights.lower() == "pissa":
            self._pissa_init(adapter_name)

    def scale_layer(self, adapter_name: str, scaling: float) -> None:
        """Scale the adapter layer."""
        if adapter_name not in self.scaling:
            # Ignore the case where the adapter is not found in the layer. This case is not a problem as long as
            # the adapter is not present in any layer (in which case the check in the PEFT model will fail)
            return

        self.scaling[adapter_name] *= scaling

    def unscale_layer(self, adapter_name: str, scaling: float) -> None:
        """Unscale the adapter layer."""
        if adapter_name not in self.scaling:
            # Ignore the case where the adapter is not found in the layer. This case is not a problem as long as
            # the adapter is not present in any layer (in which case the check in the PEFT model will fail)
            return

        self.scaling[adapter_name] /= scaling


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
        # Sum all A matrices
        weight_A = torch.sum(torch.stack([
            layer.weight for layer in self.lora_A[adapter]
        ]), dim=0)
        
        # Sum all B matrices
        weight_B = torch.sum(torch.stack([
            layer.weight for layer in self.lora_B[adapter]
        ]), dim=0)
        
        # Compute delta weight
        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged_adapters and self.training:
                # Emit a warning if there are merged adapters and we are in training mode
                warnings.warn(
                    "Detected merged adapters in training mode. "
                    "Training with merged adapters is not recommended as it can lead to unexpected behavior."
                )
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                    
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                
                # AwLoRA Core Technology 2: AwSVD initialization
                if (self.use_awsvd_init[active_adapter] and 
                    not self.activation_stats_collected[active_adapter] and 
                    self.training):
                    self._awsvd_init(active_adapter, x)
                
                # AwLoRA Core Technology 3: Activation-aware weighting
                if self.use_activation_aware[active_adapter]:
                    activation_weights = self._compute_activation_weights(
                        x, 
                        self.activation_aware_mode[active_adapter],
                        self.activation_normalize[active_adapter]
                    )
                    
                    if self.activation_aware_mode[active_adapter] == "inps":
                        # Apply weights to input
                        x_weighted = dropout(x) * activation_weights.unsqueeze(0)
                        # CoLA collaborative strategy with activation weighting
                        for i in range(self.num_A[active_adapter]):
                            for j in range(self.num_B[active_adapter]):
                                result += lora_B[j](lora_A[i](x_weighted)) * scaling
                    else:  # outps mode
                        # Apply weights to output
                        lora_output = 0
                        for i in range(self.num_A[active_adapter]):
                            for j in range(self.num_B[active_adapter]):
                                lora_output += lora_B[j](lora_A[i](dropout(x)))
                        result += (lora_output * activation_weights.unsqueeze(0)) * scaling
                else:
                    # Standard CoLA collaborative strategy without activation weighting
                    for i in range(self.num_A[active_adapter]):
                        for j in range(self.num_B[active_adapter]):
                            result += lora_B[j](lora_A[i](dropout(x))) * scaling

            result = result.to(torch_result_dtype)

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
        self.scaling[adapter_name] = lora_alpha / r
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
        self.scaling[adapter_name] = lora_alpha / r
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
            for i in range(num_A):
                self.lora_A[adapter_name][i] = self.lora_A[adapter_name][i].to(weight.dtype)
            for i in range(num_B):
                self.lora_B[adapter_name][i] = self.lora_B[adapter_name][i].to(weight.dtype)

        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
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
                    result += lora_B[j](lora_A[i](dropout(x))) * scaling

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep 