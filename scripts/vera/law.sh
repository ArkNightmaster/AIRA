#!/bin/bash
# VeRA - Law Domain
# VeRA fine-tuning
# Optimized for legal reasoning and analysis
export DISABLE_VERSION_CHECK=true
export CUDA_VISIBLE_DEVICES=2,3
module add cuda/12.4

cd LLaMA-Factory

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /aifs4su/gov/models/Llama-3.1-8B-Instruct \
    --dataset us_terms,Lawyer-Instruct \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type vera \
    --vera_r 256 \
    --vera_dropout 0.05 \
    --target_modules q_proj v_proj \
    --output_dir ./saves/Llama-3.1-8B-Instruct/VERA_Law \
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
    --max_samples 1000

# Evaluation command
llamafactory-cli eval \
    --model_name_or_path /aifs4su/gov/models/Llama-3.1-8B-Instruct \
    --template llama3 \
    --task mmlu_test_Law \
    --lang en \
    --n_shot 0 \
    --batch_size 8 \
    --trust_remote_code \
    --adapter_name_or_path ./saves/Llama-3.1-8B-Instruct/VERA_Law \
    2>&1 | tee ./output/logs/vera_law.log
