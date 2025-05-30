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

import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)

from .config import AiraMoeConfig
from .layer import AiraMoeLayer, Conv2d, Embedding, Linear

if is_bnb_available():
    import bitsandbytes as bnb

    from .bnb import Linear8bitLt, Linear4bit

if is_bnb_4bit_available():
    from .gptq import QuantLinear


class AiraMoeModel(BaseTuner):
    """
    Creates AiraMoe (Activation-aware Improved Rank Allocation with Mixture of Experts) model from a pretrained model.
    
    The AiraMoe model combines CoLA's collaborative low-rank adaptation with AwLoRA's three core technologies:
    1. Layer-wise LoRA Rank allocation based on LOD outlier metrics
    2. AwSVD-based LoRA matrix initialization  
    3. Activation-aware weighted forward propagation
    
    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`AiraMoeConfig`]): The configuration of the AiraMoe model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
    
    Returns:
        `torch.nn.Module`: The AiraMoe model.
    
    Example:
        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import AiraMoeModel, AiraMoeConfig
        
        >>> config = AiraMoeConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     num_A=2,
        ...     num_B=2,
        ...     use_layer_wise_rank=True,
        ...     use_awsvd_init=True,
        ...     use_activation_aware=True,
        ... )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> aira_moe_model = AiraMoeModel(model, config, "default")
        ```
    
    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`AiraMoeConfig`]): The configuration of the AiraMoe model.
    """

    prefix: str = "aira_moe_"
    layers_mapping = {
        nn.Linear: Linear,
        nn.Embedding: Embedding,
        nn.Conv2d: Conv2d,
        Conv1D: Linear,
    }

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _check_new_adapter_config(self, config: AiraMoeConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        Args:
            config (`AiraMoeConfig`): The config to check.
        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(aira_moe_config, key):
        return check_target_module_exists(aira_moe_config, key)

    def _create_and_replace(
        self,
        aira_moe_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find the layer name without the prefix
        pattern = aira_moe_config.rank_pattern or aira_moe_config.alpha_pattern
        layer_name = current_key.split(".")[-1] if pattern else None

        r = aira_moe_config.r
        lora_alpha = aira_moe_config.lora_alpha

        # Check if we need to use layer-specific rank/alpha
        if aira_moe_config.rank_pattern:
            for pattern_key, pattern_rank in aira_moe_config.rank_pattern.items():
                if re.fullmatch(pattern_key, current_key) or re.fullmatch(pattern_key, layer_name):
                    r = pattern_rank
                    break

        if aira_moe_config.alpha_pattern:
            for pattern_key, pattern_alpha in aira_moe_config.alpha_pattern.items():
                if re.fullmatch(pattern_key, current_key) or re.fullmatch(pattern_key, layer_name):
                    lora_alpha = pattern_alpha
                    break

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": aira_moe_config.lora_dropout,
            "num_A": aira_moe_config.num_A,
            "num_B": aira_moe_config.num_B,
            "fan_in_fan_out": aira_moe_config.fan_in_fan_out,
            "init_lora_weights": aira_moe_config.init_lora_weights,
            "use_layer_wise_rank": aira_moe_config.use_layer_wise_rank,
            "use_awsvd_init": aira_moe_config.use_awsvd_init,
            "use_activation_aware": aira_moe_config.use_activation_aware,
            "activation_aware_mode": aira_moe_config.activation_aware_mode,
            "activation_normalize": aira_moe_config.activation_normalize,
            "awsvd_collect_steps": aira_moe_config.awsvd_collect_steps,
        }
        kwargs["bias"] = bias

        # note: AdaLoRA uses quantization_config instead of 4bit_compute_dtype
        quantization_config = get_quantization_config(self.model, method="gptq")
        if quantization_config is not None:
            kwargs["gptq_quantization_config"] = quantization_config

        if isinstance(target, AiraMoeLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=aira_moe_config.lora_dropout,
                num_A=aira_moe_config.num_A,
                num_B=aira_moe_config.num_B,
                init_lora_weights=aira_moe_config.init_lora_weights,
                use_layer_wise_rank=aira_moe_config.use_layer_wise_rank,
                use_awsvd_init=aira_moe_config.use_awsvd_init,
                use_activation_aware=aira_moe_config.use_activation_aware,
                activation_aware_mode=aira_moe_config.activation_aware_mode,
                activation_normalize=aira_moe_config.activation_normalize,
                awsvd_collect_steps=aira_moe_config.awsvd_collect_steps,
            )
        else:
            new_module = self._create_new_module(aira_moe_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(self, aira_moe_config, adapter_name, target, **kwargs):
        # Collect additional keyword arguments
        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)
        elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
            new_module = QuantLinear(target, adapter_name, **kwargs)
        elif isinstance(target, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = aira_moe_config.fan_in_fan_out = False
            new_module = Linear(target, adapter_name, **kwargs)
        elif isinstance(target, torch.nn.Embedding):
            new_module = Embedding(target, adapter_name, **kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            new_module = Conv2d(target, adapter_name, **kwargs)
        elif isinstance(target, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = aira_moe_config.fan_in_fan_out = True
            new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, AiraMoeLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge GPTQ weights")

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, AiraMoeLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the AiraMoe layers into the base model. This is needed if someone wants to use the base
        model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the AiraMoe modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def subtract_embedding_weights(self, layer_name: str) -> None:
        """
        Subtract the LoRA embedding weights from the base model.
        
        This is a helper method for models that have been initialized with PiSSA, which stores the (ΔW + W) in the
        base model and (-ΔW) in the adapter. This method reverts the base model to the original W, and stores ΔW in
        the adapter, which is the more typical configuration.

        Args:
            layer_name (`str`): The name of the embedding layer
        """
        # This method is adapted from the subtract_embedding_weights method in the LoRA implementation
        # TODO: This method is not yet implemented for AiraMoe
        raise NotImplementedError("subtract_embedding_weights is not yet implemented for AiraMoe")

    # AwLoRA Core Technology 1: Layer-wise Rank Allocation
    def compute_layer_importance(self, train_loader, device="cuda", max_samples=1000):
        """
        AwLoRA Core Technology 1: Compute layer importance metrics for rank allocation.
        
        Args:
            train_loader: DataLoader for training data
            device: Device to run computation on
            max_samples: Maximum number of samples to use for computation
            
        Returns:
            dict: Layer importance metrics
        """
        self.model.eval()
        layer_importance = {}
        
        # Hook function to collect activations and compute LOD metrics
        def hook_fn(name):
            def hook(module, input, output):
                if name not in layer_importance:
                    layer_importance[name] = {
                        'activation_values': [],
                        'lod_values': [],
                        'input_norms': [],
                        'weight_abs': []
                    }
                
                # Compute activation importance
                if isinstance(output, torch.Tensor):
                    activation_importance = output.norm(p=2, dim=-1).mean().item()
                    layer_importance[name]['activation_values'].append(activation_importance)
                
                # Compute LOD metrics for Linear layers
                if isinstance(module, (nn.Linear, Linear)) and isinstance(input[0], torch.Tensor):
                    input_tensor = input[0].detach()
                    weight_tensor = module.weight.detach()
                    
                    # Compute input norms
                    input_norm = torch.linalg.vector_norm(input_tensor, ord=2, dim=0)
                    layer_importance[name]['input_norms'].append(input_norm.cpu())
                    
                    # Compute weight-activation interaction matrix A_ij
                    A_ij = torch.abs(weight_tensor) * input_norm.unsqueeze(0)
                    mean_A_l = A_ij.mean().item()
                    
                    # LOD outlier detection with threshold M
                    M = self.peft_config[self.active_adapter].lod_threshold_M
                    lod_value = (A_ij > M * mean_A_l).float().mean().item()
                    layer_importance[name]['lod_values'].append(lod_value)
                    
            return hook
        
        # Register hooks for target modules
        hooks = []
        for name, module in self.model.named_modules():
            if any(target in name for target in self.peft_config[self.active_adapter].target_modules):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Collect statistics
        sample_count = 0
        with torch.no_grad():
            for batch in train_loader:
                if sample_count >= max_samples:
                    break
                    
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    self.model(**batch)
                elif isinstance(batch, (list, tuple)):
                    batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                    self.model(*batch)
                else:
                    batch = batch.to(device)
                    self.model(batch)
                
                sample_count += batch.size(0) if hasattr(batch, 'size') else len(batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Aggregate statistics
        aggregated_importance = {}
        for layer_name, stats in layer_importance.items():
            aggregated_importance[layer_name] = {
                'activation_mean': sum(stats['activation_values']) / len(stats['activation_values']) if stats['activation_values'] else 0,
                'lod_mean': sum(stats['lod_values']) / len(stats['lod_values']) if stats['lod_values'] else 0,
            }
        
        return aggregated_importance

    def optimize_layer_ranks(self, layer_importance, objective_function="log"):
        """
        AwLoRA Core Technology 1: Optimize rank allocation based on layer importance.
        
        Args:
            layer_importance: Dictionary of layer importance metrics
            objective_function: Type of objective function ("log", "linear", "exp2", "cubic")
            
        Returns:
            dict: Optimized rank allocation for each layer
        """
        from scipy.optimize import minimize
        import numpy as np
        
        config = self.peft_config[self.active_adapter]
        theta_type = config.theta_type
        rank_budget = config.rank_budget
        min_rank = config.min_rank
        max_rank = config.max_rank
        
        # Extract importance values
        layer_names = list(layer_importance.keys())
        if theta_type == "act":
            importance_values = [layer_importance[name]['activation_mean'] for name in layer_names]
        elif theta_type == "lod":
            importance_values = [layer_importance[name]['lod_mean'] * 1000 for name in layer_names]  # Scale LOD values
        else:
            raise ValueError(f"Unknown theta_type: {theta_type}")
        
        importance_values = np.array(importance_values)
        n_layers = len(layer_names)
        
        # Define objective function
        def objective(ranks):
            if objective_function == "log":
                return -np.sum(importance_values * np.log(ranks + 1e-8))
            elif objective_function == "linear":
                return -np.sum(importance_values * ranks)
            elif objective_function == "exp2":
                return -np.sum(importance_values * (ranks ** 2))
            elif objective_function == "cubic":
                return -np.sum(importance_values * (ranks ** 3))
            else:
                raise ValueError(f"Unknown objective function: {objective_function}")
        
        # Define constraints
        def budget_constraint(ranks):
            return rank_budget - np.sum(ranks)
        
        # Bounds for each rank
        bounds = [(min_rank, max_rank) for _ in range(n_layers)]
        
        # Initial guess (uniform distribution)
        initial_ranks = np.full(n_layers, rank_budget // n_layers)
        initial_ranks = np.clip(initial_ranks, min_rank, max_rank)
        
        # Adjust initial guess to satisfy budget constraint
        current_sum = np.sum(initial_ranks)
        if current_sum != rank_budget:
            adjustment = (rank_budget - current_sum) / n_layers
            initial_ranks += adjustment
            initial_ranks = np.clip(initial_ranks, min_rank, max_rank)
        
        # Optimization
        constraints = {'type': 'eq', 'fun': budget_constraint}
        result = minimize(
            objective,
            initial_ranks,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimized_ranks = np.round(result.x).astype(int)
            # Ensure constraints are satisfied
            optimized_ranks = np.clip(optimized_ranks, min_rank, max_rank)
            
            # Final adjustment to meet budget exactly
            current_sum = np.sum(optimized_ranks)
            if current_sum != rank_budget:
                diff = rank_budget - current_sum
                # Distribute the difference
                for i in range(abs(diff)):
                    if diff > 0:
                        # Need to increase ranks
                        idx = np.argmax(importance_values * (optimized_ranks < max_rank))
                        if optimized_ranks[idx] < max_rank:
                            optimized_ranks[idx] += 1
                    else:
                        # Need to decrease ranks
                        idx = np.argmin(importance_values * (optimized_ranks > min_rank))
                        if optimized_ranks[idx] > min_rank:
                            optimized_ranks[idx] -= 1
        else:
            print(f"Optimization failed: {result.message}")
            # Fallback to uniform distribution
            optimized_ranks = np.full(n_layers, rank_budget // n_layers)
            optimized_ranks = np.clip(optimized_ranks, min_rank, max_rank)
        
        # Create rank allocation dictionary
        rank_allocation = {}
        for i, layer_name in enumerate(layer_names):
            rank_allocation[layer_name] = int(optimized_ranks[i])
        
        return rank_allocation

    def apply_layer_wise_ranks(self, rank_allocation):
        """
        Apply the optimized rank allocation to the model layers.
        
        Args:
            rank_allocation: Dictionary mapping layer names to ranks
        """
        for name, module in self.model.named_modules():
            if isinstance(module, AiraMoeLayer) and name in rank_allocation:
                new_rank = rank_allocation[name]
                adapter_name = self.active_adapter
                
                # Update the rank for this layer
                if adapter_name in module.r:
                    old_rank = module.r[adapter_name]
                    if old_rank != new_rank:
                        print(f"Updating rank for layer {name}: {old_rank} -> {new_rank}")
                        
                        # Recreate LoRA matrices with new rank
                        module.r[adapter_name] = new_rank
                        module.scaling[adapter_name] = module.lora_alpha[adapter_name] / new_rank
                        
                        # Recreate A and B matrices
                        num_A = module.num_A[adapter_name]
                        num_B = module.num_B[adapter_name]
                        
                        module.lora_A[adapter_name] = nn.ModuleList([
                            nn.Linear(module.in_features, new_rank, bias=False) for _ in range(num_A)
                        ])
                        module.lora_B[adapter_name] = nn.ModuleList([
                            nn.Linear(new_rank, module.out_features, bias=False) for _ in range(num_B)
                        ])
                        
                        # Reinitialize weights
                        module.reset_lora_parameters(adapter_name, True)
                        
                        # Move to correct device
                        device = next(module.get_base_layer().parameters()).device
                        for i in range(num_A):
                            module.lora_A[adapter_name][i].to(device)
                        for i in range(num_B):
                            module.lora_B[adapter_name][i].to(device) 