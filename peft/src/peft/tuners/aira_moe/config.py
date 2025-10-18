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
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class AiraMoeConfig(PeftConfig):
    """
    Configuration class for AiraMoe (Activation-aware Improved Rank Allocation with Mixture of Experts).
    
    AiraMoe combines CoLA's collaborative low-rank adaptation with AwLoRA's three core technologies:
    1. Layer-wise LoRA Rank allocation based on LOD outlier metrics
    2. AwSVD-based LoRA matrix initialization
    3. Activation-aware weighted forward propagation
    
    Args:
        r (`int`): LoRA attention dimension (the "rank").
        target_modules (`Optional[Union[List[str], str]]`): The names of the modules to apply the adapter to.
        lora_alpha (`int`): The alpha parameter for LoRA scaling.
        lora_dropout (`float`): The dropout probability for LoRA layers.
        
        # AwLoRA Core Technology 1: Layer-wise Rank Allocation
        use_layer_wise_rank (`bool`): Whether to use layer-wise rank allocation based on LOD metrics.
        lod_threshold_M (`float`): Threshold multiplier M for LOD outlier detection.
        theta_type (`Literal["act", "lod"]`): Type of importance metric for rank allocation.
        rank_budget (`int`): Total rank budget for optimization-based allocation.
        min_rank (`int`): Minimum rank for each layer.
        max_rank (`int`): Maximum rank for each layer.
        objective_function (`Literal["log", "linear", "exp2", "cubic"]`): Objective function for rank optimization.
        
        # AwLoRA Core Technology 2: AwSVD Initialization
        use_awsvd_init (`bool`): Whether to use activation-aware SVD initialization.
        awsvd_collect_steps (`int`): Number of steps to collect activation statistics for AwSVD.
        
        # AwLoRA Core Technology 3: Activation-aware Weighting
        use_activation_aware (`bool`): Whether to use activation-aware weighted forward propagation.
        activation_aware_mode (`Literal["inps", "outps"]`): Mode for activation-aware weighting.
        activation_normalize (`bool`): Whether to normalize activation weights to [0,1].
        
        # Other inherited parameters
        fan_in_fan_out (`bool`): Set this to True if the layer stores weight like (fan_in, fan_out).
        bias (`str`): Bias type for LoRA. Can be 'none', 'all' or 'lora_only'.
        modules_to_save (`List[str]`): List of modules apart from adapter layers to be set as trainable.
        init_lora_weights (`bool` | `Literal[...]`): How to initialize the weights of the adapter layers.
        layers_to_transform (`Union[List[int], int]`): The layer indices to transform.
        layers_pattern (`str`): The layer pattern name.
        rank_pattern (`dict`): The mapping from layer names to ranks.
        alpha_pattern (`dict`): The mapping from layer names to alphas.
        megatron_config (`Optional[dict]`): The TransformerConfig arguments for Megatron.
        megatron_core (`Optional[str]`): The core module from Megatron to use.
        layer_replication (`List[Tuple[int, int]]`): Build a new stack of layers by stacking.
    """

    # Basic LoRA parameters
    r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture."
            ),
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    
    # CoLA parameters (AIRA simplified to single A/B per layer; these are removed)
    
    # AwLoRA Core Technology 1: Layer-wise Rank Allocation
    use_layer_wise_rank: bool = field(
        default=False, 
        metadata={"help": "Whether to use layer-wise rank allocation based on LOD metrics"}
    )
    lod_threshold_M: float = field(
        default=5.0, 
        metadata={"help": "Threshold multiplier M for LOD outlier detection"}
    )
    theta_type: Literal["act", "lod"] = field(
        default="lod", 
        metadata={"help": "Type of importance metric for rank allocation: 'act' for activation, 'lod' for LOD"}
    )
    rank_budget: int = field(
        default=16*32, 
        metadata={"help": "Total rank budget for optimization-based allocation"}
    )
    min_rank: int = field(
        default=8, 
        metadata={"help": "Minimum rank for each layer"}
    )
    max_rank: int = field(
        default=32, 
        metadata={"help": "Maximum rank for each layer"}
    )
    objective_function: Literal["log", "linear", "exp2", "cubic"] = field(
        default="log", 
        metadata={"help": "Objective function for rank optimization"}
    )
    
    # AwLoRA Core Technology 2: AwSVD Initialization
    use_awsvd_init: bool = field(
        default=False, 
        metadata={"help": "Whether to use activation-aware SVD initialization"}
    )
    awsvd_collect_steps: int = field(
        default=100, 
        metadata={"help": "Number of steps to collect activation statistics for AwSVD"}
    )
    
    # AwLoRA Core Technology 3: Activation-aware Weighting
    use_activation_aware: bool = field(
        default=False, 
        metadata={"help": "Whether to use activation-aware weighted forward propagation"}
    )
    activation_aware_mode: Literal["inps", "outps"] = field(
        default="inps", 
        metadata={"help": "Mode for activation-aware weighting: 'inps' for input-based, 'outps' for output-based"}
    )
    activation_normalize: bool = field(
        default=True, 
        metadata={"help": "Whether to normalize activation weights to [0,1]"}
    )
    
    # Other parameters inherited from LoRA/CoLA
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    init_lora_weights: bool | Literal["gaussian", "olora", "pissa", "pissa_niter_[number of iters]", "loftq", "awsvd"] = field(
        default="awsvd",
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. "
                "'awsvd' uses activation-aware SVD initialization (default for AiraMoe)."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, if specified, PEFT will transform only the layers indexes that are specified."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`."
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`."
            )
        },
    )
    megatron_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The TransformerConfig from Megatron. It is used to create LoRA's parallel linear layer."
            )
        },
    )
    megatron_core: Optional[str] = field(
        default="megatron.core",
        metadata={
            "help": (
                "The core module from Megatron, it is used to create LoRA's parallel linear layer."
            )
        },
    )
    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers."
            )
        },
    )

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        return rv

    def __post_init__(self):
        self.peft_type = PeftType.AIRA_MOE
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        
        # Validation checks
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
            
        # Validate AwLoRA specific parameters
        if self.use_layer_wise_rank:
            if self.min_rank >= self.max_rank:
                raise ValueError("`min_rank` must be less than `max_rank`")
            if self.rank_budget <= 0:
                raise ValueError("`rank_budget` must be positive")
                
        if self.use_awsvd_init and self.awsvd_collect_steps <= 0:
            raise ValueError("`awsvd_collect_steps` must be positive when using AwSVD initialization")

        self._custom_modules: Optional[dict[type[nn.Module], type[nn.Module]]] = None

    def _register_custom_module(self, mapping: dict[type[nn.Module], type[nn.Module]]) -> None:
        """
        Experimental API to support providing custom LoRA layers.
        """
        if self._custom_modules is None:
            self._custom_modules = {}
        self._custom_modules.update(mapping) 

