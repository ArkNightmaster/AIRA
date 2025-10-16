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
            
            try:
                # Get the PEFT model
                peft_model = self.model
                
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
                
                # Compute layer importance using the simple dataloader
                layer_importance = peft_model.compute_layer_importance(
                    simple_dataloader, 
                    device=device,
                    max_samples=max_samples
                )
                
                logger.info_rank0(f"Layer importance computed for {len(layer_importance)} layers")
                
                # Optimize rank allocation
                rank_allocation = peft_model.optimize_layer_ranks(
                    layer_importance, 
                    objective_function=self.finetuning_args.objective_function
                )
                
                logger.info_rank0("Optimized rank allocation:")
                total_ranks = 0
                for layer_name, rank in rank_allocation.items():
                    logger.info_rank0(f"  {layer_name}: rank={rank}")
                    total_ranks += rank
                logger.info_rank0(f"Total ranks used: {total_ranks}/{self.finetuning_args.rank_budget}")
                
                # Apply optimized ranks
                peft_model.apply_layer_wise_ranks(rank_allocation)
                logger.info_rank0("Layer-wise rank allocation applied successfully!")
                
            except Exception as e:
                logger.warning_rank0(f"Failed to apply layer-wise rank allocation: {e}")
                logger.warning_rank0("Continuing with uniform rank allocation...")

    @override
    def train(self, **kwargs):
        """
        Override train method to add AIRA_MoE layer-wise rank allocation support.
        """
        # Setup AIRA_MoE layer-wise rank allocation before training
        self._setup_aira_moe_layer_wise_ranks()
        
        # Call the original train method
        return super().train(**kwargs)

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
