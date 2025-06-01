#!/bin/bash
# AIRA_MoE Basic Version - Generality Domain
# Only enables CoLA collaborative adaptation (num_A, num_B)
# Disables advanced AwLoRA technologies for baseline comparison

cd LLaMA-Factory

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /data/Llama-3.1-8B-Instruct \
    --dataset databricks-dolly-15k \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type aira_moe \
    --output_dir ./saves/Llama-3.1-8B-Instruct/AIRA_MoE_Basic_Generality \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 20 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.1 \
    --plot_loss \
    --fp16 \
    --max_samples 1000 \
    --num_A 2 \
    --num_B 3 \
    --use_layer_wise_rank false \
    --use_awsvd_init false \
    --use_activation_aware false \
    --ddp_find_unused_parameters false

# Evaluation command (uncomment to run)
# CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli eval \
#     --model_name_or_path /data/Llama-3.1-8B-Instruct \
#     --template llama3 \
#     --task mmlu_test_None \
#     --lang en \
#     --n_shot 0 \
#     --batch_size 8 \
#     --trust_remote_code \
#     --adapter_name_or_path ./saves/Llama-3.1-8B-Instruct/AIRA_MoE_Basic_Generality \
#     --ddp_find_unused_parameters false 