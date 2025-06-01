#!/usr/bin/env python3
"""
AIRA_MoE Usage Example for LLaMA-Factory

This script demonstrates how to use AIRA_MoE (Activation-aware Improved Rank Allocation 
with Mixture of Experts) in LLaMA-Factory for efficient fine-tuning.

AIRA_MoE combines CoLA's collaborative low-rank adaptation with AwLoRA's three core technologies:
1. Layer-wise LoRA Rank allocation based on LOD outlier metrics
2. AwSVD-based LoRA matrix initialization
3. Activation-aware weighted forward propagation
"""

import os
import sys
import json
from typing import Dict, Any

# Add LLaMA-Factory to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def create_aira_moe_config() -> Dict[str, Any]:
    """Create a comprehensive AIRA_MoE configuration."""
    return {
        # Basic training configuration
        "model_name_or_path": "microsoft/DialoGPT-medium",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "aira_moe",
        
        # Dataset configuration
        "dataset": "alpaca_gpt4_en",
        "template": "default",
        "cutoff_len": 1024,
        "max_samples": 1000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        
        # Basic LoRA parameters
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "lora_target": "all",
        
        # CoLA collaborative parameters
        "num_A": 2,  # Number of A matrices
        "num_B": 2,  # Number of B matrices
        
        # AwLoRA Core Technology 1: Layer-wise Rank Allocation
        "use_layer_wise_rank": True,
        "lod_threshold_M": 2.0,
        "theta_type": "lod",  # or "act"
        "rank_budget": 64,
        "min_rank": 1,
        "max_rank": 16,
        "objective_function": "log",  # "log", "linear", "exp2", "cubic"
        
        # AwLoRA Core Technology 2: AwSVD Initialization
        "use_awsvd_init": True,
        "awsvd_collect_steps": 100,
        
        # AwLoRA Core Technology 3: Activation-aware Weighting
        "use_activation_aware": True,
        "activation_aware_mode": "inps",  # "inps" or "outps"
        "activation_normalize": True,
        
        # Training parameters
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "learning_rate": 5e-4,
        "num_train_epochs": 3.0,
        "max_steps": -1,
        "logging_steps": 10,
        "save_steps": 500,
        "warmup_steps": 0,
        "neftune_noise_alpha": 5,
        
        # Output configuration
        "output_dir": "./saves/aira_moe_example",
        "logging_dir": "./logs",
        "save_only_model": True,
        "plot_loss": True,
        
        # Hardware optimization
        "fp16": True,
        "ddp_timeout": 180000000,
        "include_num_input_tokens_seen": True,
        "group_by_length": True,
        "dataloader_pin_memory": False,
    }


def create_basic_aira_moe_config() -> Dict[str, Any]:
    """Create a basic AIRA_MoE configuration with minimal features."""
    return {
        # Basic training configuration
        "model_name_or_path": "microsoft/DialoGPT-small",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "aira_moe",
        
        # Dataset configuration
        "dataset": "alpaca_gpt4_en",
        "template": "default",
        "cutoff_len": 512,
        "max_samples": 100,
        
        # Basic LoRA parameters
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "lora_target": "all",
        
        # CoLA collaborative parameters
        "num_A": 1,
        "num_B": 1,
        
        # Disable advanced features for basic usage
        "use_layer_wise_rank": False,
        "use_awsvd_init": False,
        "use_activation_aware": False,
        
        # Training parameters
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-4,
        "num_train_epochs": 1.0,
        "logging_steps": 10,
        "save_steps": 100,
        
        # Output configuration
        "output_dir": "./saves/aira_moe_basic",
        "fp16": True,
    }


def create_advanced_aira_moe_config() -> Dict[str, Any]:
    """Create an advanced AIRA_MoE configuration with all features enabled."""
    return {
        # Basic training configuration
        "model_name_or_path": "meta-llama/Llama-2-7b-hf",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "aira_moe",
        
        # Dataset configuration
        "dataset": "alpaca_gpt4_en,identity",
        "template": "llama2",
        "cutoff_len": 2048,
        "max_samples": 10000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        
        # Basic LoRA parameters
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "all",
        
        # CoLA collaborative parameters
        "num_A": 3,  # More matrices for better collaboration
        "num_B": 3,
        
        # AwLoRA Core Technology 1: Layer-wise Rank Allocation
        "use_layer_wise_rank": True,
        "lod_threshold_M": 2.5,
        "theta_type": "lod",
        "rank_budget": 128,  # Larger budget for better performance
        "min_rank": 2,
        "max_rank": 32,
        "objective_function": "log",
        
        # AwLoRA Core Technology 2: AwSVD Initialization
        "use_awsvd_init": True,
        "awsvd_collect_steps": 200,
        
        # AwLoRA Core Technology 3: Activation-aware Weighting
        "use_activation_aware": True,
        "activation_aware_mode": "outps",  # Output-based weighting
        "activation_normalize": True,
        
        # Advanced training parameters
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "lr_scheduler_type": "cosine",
        "learning_rate": 3e-4,
        "num_train_epochs": 5.0,
        "max_steps": -1,
        "logging_steps": 5,
        "save_steps": 200,
        "warmup_steps": 100,
        "neftune_noise_alpha": 5,
        
        # Optimization
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        
        # Output configuration
        "output_dir": "./saves/aira_moe_advanced",
        "logging_dir": "./logs",
        "save_only_model": True,
        "save_safetensors": True,
        "plot_loss": True,
        
        # Hardware optimization
        "bf16": True,
        "ddp_timeout": 180000000,
        "include_num_input_tokens_seen": True,
        "group_by_length": True,
        "dataloader_pin_memory": False,
        
        # Evaluation
        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 100,
    }


def save_config(config: Dict[str, Any], filename: str) -> None:
    """Save configuration to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Configuration saved to: {filename}")


def print_usage_examples():
    """Print usage examples for different scenarios."""
    print("=" * 80)
    print("AIRA_MoE Usage Examples for LLaMA-Factory")
    print("=" * 80)
    
    print("\n1. Basic Usage (CoLA only):")
    print("   llamafactory-cli train examples/configs/aira_moe_basic.json")
    
    print("\n2. Standard Usage (CoLA + Activation-aware):")
    print("   llamafactory-cli train examples/configs/aira_moe_standard.json")
    
    print("\n3. Advanced Usage (All AwLoRA technologies):")
    print("   llamafactory-cli train examples/configs/aira_moe_advanced.json")
    
    print("\n4. Command Line Usage:")
    print("   llamafactory-cli train \\")
    print("     --model_name_or_path microsoft/DialoGPT-medium \\")
    print("     --stage sft \\")
    print("     --do_train \\")
    print("     --finetuning_type aira_moe \\")
    print("     --dataset alpaca_gpt4_en \\")
    print("     --template default \\")
    print("     --cutoff_len 1024 \\")
    print("     --lora_rank 8 \\")
    print("     --lora_alpha 16 \\")
    print("     --num_A 2 \\")
    print("     --num_B 2 \\")
    print("     --use_layer_wise_rank \\")
    print("     --use_awsvd_init \\")
    print("     --use_activation_aware \\")
    print("     --activation_aware_mode inps \\")
    print("     --per_device_train_batch_size 2 \\")
    print("     --gradient_accumulation_steps 4 \\")
    print("     --learning_rate 5e-4 \\")
    print("     --num_train_epochs 3.0 \\")
    print("     --output_dir ./saves/aira_moe_example \\")
    print("     --fp16")
    
    print("\n" + "=" * 80)
    print("AIRA_MoE Parameter Guide")
    print("=" * 80)
    
    print("\nCoLA Parameters:")
    print("  num_A: Number of A matrices (default: 1)")
    print("  num_B: Number of B matrices (default: 1)")
    print("  Note: Total combinations = num_A × num_B")
    
    print("\nAwLoRA Technology 1 - Layer-wise Rank Allocation:")
    print("  use_layer_wise_rank: Enable intelligent rank allocation")
    print("  rank_budget: Total rank budget for optimization")
    print("  theta_type: Importance metric ('lod' or 'act')")
    print("  objective_function: Optimization objective ('log', 'linear', 'exp2', 'cubic')")
    
    print("\nAwLoRA Technology 2 - AwSVD Initialization:")
    print("  use_awsvd_init: Enable activation-aware SVD initialization")
    print("  awsvd_collect_steps: Steps to collect activation statistics")
    
    print("\nAwLoRA Technology 3 - Activation-aware Weighting:")
    print("  use_activation_aware: Enable dynamic activation weighting")
    print("  activation_aware_mode: Weighting mode ('inps' or 'outps')")
    print("  activation_normalize: Normalize weights to [0,1]")


def main():
    """Main function to generate example configurations."""
    print("Generating AIRA_MoE example configurations...")
    
    # Create output directory
    config_dir = "examples/configs"
    os.makedirs(config_dir, exist_ok=True)
    
    # Generate basic configuration
    basic_config = create_basic_aira_moe_config()
    save_config(basic_config, f"{config_dir}/aira_moe_basic.json")
    
    # Generate standard configuration
    standard_config = create_aira_moe_config()
    save_config(standard_config, f"{config_dir}/aira_moe_standard.json")
    
    # Generate advanced configuration
    advanced_config = create_advanced_aira_moe_config()
    save_config(advanced_config, f"{config_dir}/aira_moe_advanced.json")
    
    # Print usage examples
    print_usage_examples()
    
    print(f"\n✓ Example configurations generated in {config_dir}/")
    print("✓ You can now use these configurations with llamafactory-cli train")


if __name__ == "__main__":
    main() 