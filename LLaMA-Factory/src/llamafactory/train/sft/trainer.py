# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        # Auto-configure AdaLoRA total_step so rank allocation schedule works
        try:
            if getattr(self.finetuning_args, "finetuning_type", None) == "adalora":
                peft_wrapped = self.model
                # Try to set on PeftModel -> base_model (AdaLoraModel)
                candidate = getattr(peft_wrapped, "base_model", None)
                if candidate is None:
                    candidate = peft_wrapped
                peft_cfgs = getattr(candidate, "peft_config", None)
                active = getattr(candidate, "active_adapter", None)
                if isinstance(peft_cfgs, dict):
                    key = active if active in peft_cfgs else next(iter(peft_cfgs.keys()))
                    if key in peft_cfgs and hasattr(peft_cfgs[key], "total_step"):
                        setattr(peft_cfgs[key], "total_step", int(num_training_steps))
        except Exception:
            pass
        return super().create_scheduler(num_training_steps, optimizer)

    def _setup_aira_moe_layer_wise_ranks(self) -> None:
        """
        Setup AIRA_MoE layer-wise rank allocation if enabled.
        This method is called before training starts.
        """
        if (hasattr(self.model, 'peft_config') and 
            self.finetuning_args.finetuning_type == "aira_moe" and
            self.finetuning_args.use_layer_wise_rank):
            
            logger.info_rank0("AIRA_MoE layer-wise rank allocation enabled. Computing layer importance...")
            
            # Get the PEFT model
            peft_model = self.model
            
            # Prepare cache paths
            try:
                # try to infer max_samples used by importance collection (consistent with below)
                train_dataloader = self.get_train_dataloader()
                max_samples = min(500, len(train_dataloader.dataset))
            except Exception:
                max_samples = 500
            cache_dir = os.path.join("./outputs", "aira_moe")
            layer_importance_path = os.path.join(cache_dir, "layer_importance.json")
            rank_allocation_path = os.path.join(cache_dir, "rank_allocation.json")
            activations_path = os.path.join(cache_dir, "activation_stats.pt")
            os.makedirs(cache_dir, exist_ok=True)

            # Create a simple data loader for layer importance computation
            # Use a subset of training data
            train_dataloader = self.get_train_dataloader()
            
            # Create a simple data loader for layer importance computation
            # Convert training data to the format expected by compute_layer_importance
            device = str(next(self.model.parameters()).device)
            
            # Create a simple dataset from training data
            simple_data = []
            sample_count = 0
            max_samples = min(500, len(train_dataloader.dataset))
            
            for batch in train_dataloader:
                if sample_count >= max_samples:
                    break
                
                # Convert BatchEncoding to dict format
                if hasattr(batch, 'data'):
                    # BatchEncoding object
                    batch_dict = dict(batch.data)
                elif isinstance(batch, dict):
                    # Already a dict
                    batch_dict = batch
                else:
                    # Skip if format is not recognized
                    continue
                
                # Extract input_ids and attention_mask
                if 'input_ids' in batch_dict:
                    input_ids = batch_dict['input_ids']
                    attention_mask = batch_dict.get('attention_mask', None)
                    
                    # Add each sample individually
                    for i in range(input_ids.size(0)):
                        if sample_count >= max_samples:
                            break
                        sample_dict = {'input_ids': input_ids[i:i+1]}
                        if attention_mask is not None:
                            sample_dict['attention_mask'] = attention_mask[i:i+1]
                        simple_data.append(sample_dict)
                        sample_count += 1
            
            # Create a simple DataLoader
            from torch.utils.data import DataLoader, Dataset
            
            class SimpleDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            simple_dataset = SimpleDataset(simple_data)
            simple_dataloader = DataLoader(simple_dataset, batch_size=1, shuffle=False)
            
            # Compute and optimize ranks on rank 0 only, then broadcast
            try:
                from accelerate.utils import broadcast_object_list
            except Exception:
                broadcast_object_list = None

            rank_allocation = None
            if self.accelerator.is_main_process:
                # Try load cached importance and allocation
                layer_importance = None
                if os.path.isfile(layer_importance_path):
                    try:
                        with open(layer_importance_path, "r", encoding="utf-8") as f:
                            layer_importance = json.load(f)
                        logger.info_rank0(f"Loaded cached layer importance from {layer_importance_path}")
                    except Exception:
                        layer_importance = None

                if layer_importance is None:
                    layer_importance = peft_model.compute_layer_importance(
                        simple_dataloader,
                        device=device,
                        max_samples=max_samples,
                    )
                    logger.info_rank0(f"Layer importance computed for {len(layer_importance)} layers")
                    try:
                        with open(layer_importance_path, "w", encoding="utf-8") as f:
                            json.dump(layer_importance, f)
                        logger.info_rank0(f"Saved layer importance to {layer_importance_path}")
                    except Exception as e:
                        logger.warning_rank0(f"Failed to save layer importance: {e}")

                # Aggregate module-level importance to layer-level importance
                def _aggregate_layer_importance(raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
                    allowed_suffixes = (".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj", ".down_proj")
                    aggregated: Dict[str, Dict[str, float]] = {}
                    seen_modules: set[str] = set()
                    for mod_name, stats in raw.items():
                        # Skip LoRA-specific and dropout modules
                        if ".lora_" in mod_name or "lora_dropout" in mod_name or "base_layer" in mod_name:
                            continue
                        # Canonicalize by stripping trailing .base_layer
                        canonical = mod_name[:-11] if mod_name.endswith(".base_layer") else mod_name
                        # De-duplicate if both canonical and base_layer existed
                        if canonical in seen_modules:
                            continue
                        seen_modules.add(canonical)
                        # Only consider modules under model.layers.X.*
                        if "model.layers." not in canonical:
                            continue
                        # Filter to specific projection types
                        if not canonical.endswith(allowed_suffixes):
                            continue
                        # Extract layer prefix: model.layers.{idx}
                        try:
                            parts = canonical.split("model.layers.")[1]
                            layer_idx = parts.split(".")[0]
                            layer_key = f"model.layers.{layer_idx}"
                        except Exception:
                            continue
                        # Sum activation_mean and lod_mean
                        act = float(stats.get("activation_mean", 0.0))
                        lod = float(stats.get("lod_mean", 0.0))
                        if layer_key not in aggregated:
                            aggregated[layer_key] = {"activation_mean": 0.0, "lod_mean": 0.0}
                        aggregated[layer_key]["activation_mean"] += act
                        aggregated[layer_key]["lod_mean"] += lod
                    return aggregated

                layer_importance_layerwise = _aggregate_layer_importance(layer_importance)

                # Load or compute rank allocation (layer-wise)
                if os.path.isfile(rank_allocation_path):
                    try:
                        with open(rank_allocation_path, "r", encoding="utf-8") as f:
                            rank_allocation = json.load(f)
                        logger.info_rank0(f"Loaded cached rank allocation from {rank_allocation_path}")
                    except Exception:
                        rank_allocation = None

                if rank_allocation is None:
                    rank_allocation = peft_model.optimize_layer_ranks(
                        layer_importance_layerwise,
                        objective_function=self.finetuning_args.objective_function,
                    )
                    try:
                        with open(rank_allocation_path, "w", encoding="utf-8") as f:
                            json.dump(rank_allocation, f)
                        logger.info_rank0(f"Saved rank allocation to {rank_allocation_path}")
                    except Exception as e:
                        logger.warning_rank0(f"Failed to save rank allocation: {e}")

                logger.info_rank0("Optimized rank allocation:")
                total_ranks = 0
                for layer_name, rank in rank_allocation.items():
                    logger.info_rank0(f"  {layer_name}: rank={rank}")
                    total_ranks += rank
                logger.info_rank0(f"Total ranks used: {total_ranks}/{self.finetuning_args.rank_budget}")

                # Keep layer-wise allocation; application handled inside model by layer key

            # Broadcast to all ranks
            if broadcast_object_list is not None:
                obj_list = [rank_allocation]
                broadcast_object_list(obj_list)
                rank_allocation = obj_list[0]

            # Apply on every rank
            peft_model.apply_layer_wise_ranks(rank_allocation)
            logger.info_rank0("Layer-wise rank allocation applied successfully!")

    def _initialize_aira_moe_with_activations(self) -> None:
        """
        Initialize AIRA_MoE layers with activation-aware weights using training data.
        This method performs AwSVD initialization before training starts.
        """
        if not (hasattr(self.model, 'peft_config') and 
                self.finetuning_args.finetuning_type == "aira_moe"):
            return
        
        # Check if any layer requires AwSVD initialization
        requires_awsvd_init = False
        for module in self.model.modules():
            if hasattr(module, 'use_awsvd_init'):
                for adapter_name, use_awsvd in module.use_awsvd_init.items():
                    if use_awsvd and not module.activation_stats_collected.get(adapter_name, False):
                        requires_awsvd_init = True
                        break
            if requires_awsvd_init:
                break
        
        if not requires_awsvd_init:
            logger.info_rank0("No AIRA_MoE layers require AwSVD initialization. Skipping...")
            return
        
        logger.info_rank0("Initializing AIRA_MoE layers with activation-aware weights...")
        
        try:
            # Prepare cache paths
            train_dataloader = self.get_train_dataloader()
            cache_dir = os.path.join("./outputs", "aira_moe")
            activations_path = os.path.join(cache_dir, "activation_stats.pt")
            os.makedirs(cache_dir, exist_ok=True)

            # Get training dataloader
            train_dataloader = self.get_train_dataloader()
            
            # Move model to evaluation mode temporarily to avoid updating gradients
            original_training_state = self.model.training
            self.model.eval()
            
            # Get device
            device = next(self.model.parameters()).device
            
            # Decide how many samples to use based on PEFT config (fallback to finetuning args)
            max_samples = None
            try:
                peft_cfgs = getattr(self.model, 'peft_config', {})
                values: List[int] = []
                if isinstance(peft_cfgs, dict):
                    for _adapter, cfg in peft_cfgs.items():
                        if hasattr(cfg, 'awsvd_collect_steps') and cfg.awsvd_collect_steps is not None:
                            values.append(int(cfg.awsvd_collect_steps))
                else:
                    if hasattr(peft_cfgs, 'awsvd_collect_steps') and peft_cfgs.awsvd_collect_steps is not None:
                        values.append(int(peft_cfgs.awsvd_collect_steps))
                if len(values) > 0:
                    max_samples = max(values)
            except Exception:
                max_samples = None
            if max_samples is None:
                max_samples = int(getattr(self.finetuning_args, 'awsvd_collect_steps', 100))

            # Collect activations using forward hooks to aggregate stats and init once
            sample_count = 0
            hooks = []
            module_stats: Dict[Any, Dict[str, Any]] = {}
            module_name_map: Dict[Any, str] = {}

            def _make_forward_hook(mod):
                def _hook(module, inputs, output):
                    try:
                        # Skip if no adapter needs AwSVD on this module
                        if not (hasattr(module, 'use_awsvd_init') and hasattr(module, 'activation_stats_collected')):
                            return
                        needs = False
                        for adapter_name, use_awsvd in module.use_awsvd_init.items():
                            if use_awsvd and not module.activation_stats_collected.get(adapter_name, False):
                                needs = True
                                break
                        if not needs:
                            return

                        x = inputs[0]
                        if x is None:
                            return
                        # Flatten to [-1, in_features]
                        base_weight = module.get_base_layer().weight
                        in_features = base_weight.size(1)
                        X = x.detach()
                        X = X.to(torch.float32)
                        X = X.view(-1, in_features)
                        sum_x2 = (X * X).sum(dim=0).cpu()
                        count = X.shape[0]

                        state = module_stats.get(module)
                        if state is None:
                            state = {
                                'sum_x2': torch.zeros(in_features, dtype=torch.float32),
                                'count': 0,
                            }
                            module_stats[module] = state
                        state['sum_x2'] += sum_x2
                        state['count'] += count
                    except Exception as hook_e:
                        logger.warning_rank0(f"AwSVD forward hook error: {hook_e}")
                return _hook

            # Register hooks on all candidate AIRA_MoE layers
            for name, m in self.model.named_modules():
                if hasattr(m, 'use_awsvd_init') and hasattr(m, 'get_base_layer') and hasattr(m, '_awsvd_init'):
                    try:
                        hooks.append(m.register_forward_hook(_make_forward_hook(m)))
                        module_name_map[m] = name
                    except Exception:
                        continue

            # Try to load cached activation stats
            cached_stats_by_name: Optional[Dict[str, Dict[str, Any]]] = None
            if os.path.isfile(activations_path):
                try:
                    cached_stats_by_name = torch.load(activations_path, map_location="cpu")
                    logger.info_rank0(f"Loaded cached activation stats from {activations_path}")
                except Exception:
                    cached_stats_by_name = None

            logger.info_rank0(f"Collecting activations from {max_samples} samples for initialization (adapters disabled) ...")

            try:
                if cached_stats_by_name is None:
                    with self.model.disable_adapter():
                        with torch.no_grad():
                            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Collecting activations"):
                                if sample_count >= max_samples:
                                    break

                                # Move batch to device
                                if isinstance(batch, dict):
                                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                            for k, v in batch.items()}
                                else:
                                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                            for k, v in batch.data.items()}

                                # Forward pass to trigger hooks for activation collection
                                try:
                                    _ = self.model(**batch)
                                    batch_size = batch['input_ids'].size(0) if 'input_ids' in batch else 1
                                    sample_count += batch_size
                                except Exception as e:
                                    logger.warning_rank0(f"Error during forward pass for initialization: {e}")
                                    continue
            finally:
                # Remove all hooks regardless of success/failure
                for h in hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass

            # Perform one-shot AwSVD init using aggregated stats
            try:
                # Build a by-name stats dict (tensor kept on device cpu for save)
                if cached_stats_by_name is None:
                    stats_by_name: Dict[str, Dict[str, Any]] = {}
                    for module, state in module_stats.items():
                        name = module_name_map.get(module, None)
                        if name is None:
                            continue
                        stats_by_name[name] = {
                            'sum_x2': state['sum_x2'].detach().cpu(),
                            'count': int(state['count']),
                        }
                    # Save on main process
                    if self.accelerator.is_main_process:
                        try:
                            torch.save(stats_by_name, activations_path)
                            logger.info_rank0(f"Saved activation stats to {activations_path}")
                        except Exception as e:
                            logger.warning_rank0(f"Failed to save activation stats: {e}")
                else:
                    stats_by_name = cached_stats_by_name  # type: ignore

                # Apply initialization using stats_by_name
                for name, m in self.model.named_modules():
                    if name not in stats_by_name:
                        continue
                    state = stats_by_name[name]
                    sum_x2 = state['sum_x2']
                    count = int(state['count'])
                    if count == 0:
                        continue
                    mean_square = sum_x2 / max(1, count)
                    s_diag = torch.sqrt(mean_square + 1e-12).to(next(m.parameters()).device)
                    agg_input = s_diag.view(1, -1)
                    if hasattr(m, 'use_awsvd_init') and hasattr(m, 'activation_stats_collected'):
                        for adapter_name, use_awsvd in m.use_awsvd_init.items():
                            if use_awsvd and not m.activation_stats_collected.get(adapter_name, False):
                                m._awsvd_init(adapter_name, agg_input)
                logger.info_rank0("AwSVD one-shot initialization completed for all AIRA_MoE layers.")
            except Exception as e:
                logger.warning_rank0(f"Failed during AwSVD one-shot initialization: {e}")
            
            # Restore original training state
            if original_training_state:
                self.model.train()
            
            # Count initialized layers
            initialized_count = 0
            for module in self.model.modules():
                if hasattr(module, 'activation_stats_collected'):
                    for adapter_name, is_collected in module.activation_stats_collected.items():
                        if is_collected:
                            initialized_count += 1
            
            logger.info_rank0(f"Successfully initialized {initialized_count} AIRA_MoE layers with activation-aware weights!")
            
        except Exception as e:
            logger.warning_rank0(f"Failed to initialize AIRA_MoE layers with activations: {e}")
            logger.warning_rank0("Continuing with default initialization...")

    @override
    def save_model(self, output_dir: Optional[str] = None, *args, **kwargs) -> None:
        """
        Optionally merge adapters into the base model before saving when using AIRA_MoE.
        Controlled by `finetuning_args.merge_and_save`.
        """
        if (
            getattr(self, "finetuning_args", None) is not None
            and getattr(self.finetuning_args, "finetuning_type", None) == "aira_moe"
            and getattr(self.finetuning_args, "merge_and_save", False)
        ):
            model = self.model
            if hasattr(model, "merge_and_unload"):
                # Unwrap model if needed
                try:
                    unwrapped = self.accelerator.unwrap_model(model)
                except Exception:
                    unwrapped = model
                try:
                    logger.info_rank0("Merging AIRA_MoE adapters into base model before saving...")
                    merged = unwrapped.merge_and_unload()
                    # Replace in trainer for saving path
                    self.model = merged
                except Exception as e:
                    logger.warning_rank0(f"Failed to merge adapters before saving: {e}. Saving adapters only.")

        return super().save_model(output_dir=output_dir, *args, **kwargs)

    @override
    def train(self, **kwargs):
        """
        Override train method to add AIRA_MoE layer-wise rank allocation and initialization support.
        """
        # Setup AIRA_MoE layer-wise rank allocation before training
        self._setup_aira_moe_layer_wise_ranks()
        
        # Initialize AIRA_MoE layers with activation-aware weights before training
        self._initialize_aira_moe_with_activations()
        
        # Call the original train method
        train_output = super().train(**kwargs)

        # If AdaLoRA is used, ensure final budget allocation mask is applied at the end (safety)
        try:
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "update_and_allocate"):
                # Best-effort call; actual step-wise updates should be done in training loop via callbacks if desired.
                total_steps = getattr(self.finetuning_args, "adalora_total_step", None)
                if isinstance(total_steps, int) and total_steps > 0:
                    self.model.base_model.update_and_allocate(total_steps)
        except Exception:
            pass

        return train_output

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Fixes the loss value. See https://github.com/huggingface/transformers/pull/35438 for details.
        """
        # if True:
        #     outputs, router_logits = model(**inputs)
        #     print(f'routes_logits: {router_logits}')
        #     import sys
        #     sys.exit()
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        #     if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
        #         if return_outputs:
        #             loss = (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
        #         else:
        #             loss = loss / self.args.gradient_accumulation_steps
        # else:
        loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
        if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
            if return_outputs:
                loss = (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                loss = loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
