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

import warnings
from typing import Any, Optional, Union

import torch

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose

from .layer import AiraMoeLayer


class QuantLinear(torch.nn.Module, AiraMoeLayer):
    """AiraMoe implementation for GPTQ quantized linear layers."""
    
    def __init__(
        self,
        base_layer,
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
        AiraMoeLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = False

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
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter not in self.lora_A.keys():
                continue

            warnings.warn(
                "Merge AiraMoe module to GPTQ linear may get different generations due to rounding errors."
            )
            
            # For GPTQ, we cannot directly merge weights, so we raise an error
            raise ValueError(
                "Cannot merge AiraMoe layers with GPTQ quantized layers. "
                "Please use the model without merging or use a different quantization method."
            )

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged_adapters:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        
        # For GPTQ, unmerging is not supported
        raise ValueError(
            "Unmerging is not supported for GPTQ quantized layers with AiraMoe."
        )

    def get_delta_weight(self, adapter):
        """
        Compute the delta weight for the given adapter (AiraMoe version with multiple A and B matrices).
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
        output_tensor = transpose(weight_B @ weight_A, False) * self.scaling[adapter]
        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged_adapters and self.training:
                warnings.warn(
                    "Detected merged adapters in training mode. "
                    "Training with merged adapters is not recommended as it can lead to unexpected behavior."
                )
            result = self.base_layer(x, *args, **kwargs)
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
                        # activation_weights has shape [hidden_size], need to broadcast to match x
                        weight_shape = [1] * (x.dim() - 1) + [activation_weights.size(-1)]
                        activation_weights_expanded = activation_weights.view(*weight_shape)
                        x_weighted = dropout(x) * activation_weights_expanded
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
                        # activation_weights has shape [out_features], need to broadcast to match lora_output
                        weight_shape = [1] * (lora_output.dim() - 1) + [activation_weights.size(-1)]
                        activation_weights_expanded = activation_weights.view(*weight_shape)
                        result += (lora_output * activation_weights_expanded) * scaling
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