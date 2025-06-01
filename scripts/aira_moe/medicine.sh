#!/bin/bash
# AIRA_MoE - Medicine Domain
# Full version with all AwLoRA technologies enabled
# Optimized for medical knowledge and reasoning

cd LLaMA-Factory

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /data/Llama-3.1-8B-Instruct \
    --dataset iCliniq,GenMedGPT-5k \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type aira_moe \
    --output_dir ./saves/Llama-3.1-8B-Instruct/AIRA_MoE_Medicine \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --val_size 0.1 \
    --plot_loss \
    --fp16 \
    --max_samples 1000 \
    --num_A 3 \
    --num_B 3 \
    --use_layer_wise_rank true \
    --rank_budget 128 \
    --theta_type lod \
    --objective_function log \
    --min_rank 16 \
    --max_rank 64 \
    --lod_threshold_M 4.0 \
    --use_awsvd_init true \
    --awsvd_collect_steps 120 \
    --use_activation_aware true \
    --activation_aware_mode outps \
    --activation_normalize true \
    --ddp_find_unused_parameters false

# Evaluation command (uncomment to run)
# CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli eval \
#     --model_name_or_path /data/Llama-3.1-8B-Instruct \
#     --template llama3 \
#     --task mmlu_test_Medicine \
#     --lang en \
#     --n_shot 0 \
#     --batch_size 8 \
#     --trust_remote_code \
#     --adapter_name_or_path ./saves/Llama-3.1-8B-Instruct/AIRA_MoE_Medicine \
#     --ddp_find_unused_parameters false 