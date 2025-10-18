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

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .layer import AiraMoeLayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, AiraMoeLayer):
        """AiraMoe implementation for 8-bit quantized linear layers."""
        
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            # simplified: single A/B (num_A/num_B removed)
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
                # simplified AIRA: single A/B
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
                    "Merge AiraMoe module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                output = dequantize_bnb_weight(weight, state=state)
                w_data = output.to(lora_data.dtype).to(lora_data.device) + lora_data

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()
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
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge AiraMoe module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                output = dequantize_bnb_weight(weight, state=state)

                w_data = output.to(lora_data.dtype).to(lora_data.device) - lora_data

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()

        def get_delta_weight(self, adapter):
            """
            Compute the delta weight for the given adapter (AiraMoe version with multiple A and B matrices).
            """
            # Single A/B weights
            weight_A = self.lora_A[adapter].weight
            weight_B = self.lora_B[adapter].weight
            
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
                
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                        
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    
                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        x = x.to(lora_A[0].weight.dtype)

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
                            # Single A/B path
                            lora_result = lora_B(lora_A(x_weighted)) * scaling
                            if requires_conversion:
                                lora_result = lora_result.to(expected_dtype)
                            result += lora_result
                        else:  # outps mode
                            # Apply weights to output
                            lora_output = lora_B(lora_A(dropout(x)))
                            # activation_weights has shape [out_features], need to broadcast to match lora_output
                            weight_shape = [1] * (lora_output.dim() - 1) + [activation_weights.size(-1)]
                            activation_weights_expanded = activation_weights.view(*weight_shape)
                            lora_result = (lora_output * activation_weights_expanded) * scaling
                            if requires_conversion:
                                lora_result = lora_result.to(expected_dtype)
                            result += lora_result
                    else:
                        # Single A/B path without activation weighting
                        lora_result = lora_B(lora_A(dropout(x))) * scaling
                        if requires_conversion:
                            lora_result = lora_result.to(expected_dtype)
                        result += lora_result

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, AiraMoeLayer):
        """AiraMoe implementation for 4-bit quantized linear layers."""
        
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            # simplified: single A/B (num_A/num_B removed)
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
                # simplified AIRA: single A/B
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
                    "Merge AiraMoe module to 4-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                w_data = dequantize_bnb_weight(weight, state=None).to(lora_data.dtype) + lora_data

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )
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
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge AiraMoe module to 4-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                w_data = dequantize_bnb_weight(weight, state=None).to(lora_data.dtype) - lora_data
                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )

        def get_delta_weight(self, adapter):
            """
            Compute the delta weight for the given adapter (AiraMoe version with multiple A and B matrices).
            """
            # Single A/B weights
            weight_A = self.lora_A[adapter].weight
            weight_B = self.lora_B[adapter].weight
            
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
                
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                        
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    
                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = lora_A[0].weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)

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
                            # Single A/B path
                            lora_result = lora_B(lora_A(x_weighted)) * scaling
                            if requires_conversion:
                                lora_result = lora_result.to(expected_dtype)
                            result += lora_result
                        else:  # outps mode
                            # Apply weights to output
                            lora_output = lora_B(lora_A(dropout(x)))
                            # activation_weights has shape [out_features], need to broadcast to match lora_output
                            weight_shape = [1] * (lora_output.dim() - 1) + [activation_weights.size(-1)]
                            activation_weights_expanded = activation_weights.view(*weight_shape)
                            lora_result = (lora_output * activation_weights_expanded) * scaling
                            if requires_conversion:
                                lora_result = lora_result.to(expected_dtype)
                            result += lora_result
                    else:
                        # Single A/B path without activation weighting
                        lora_result = lora_B(lora_A(dropout(x))) * scaling
                        if requires_conversion:
                            lora_result = lora_result.to(expected_dtype)
                        result += lora_result

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep 