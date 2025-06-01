#!/usr/bin/env python3
"""
AiraMoe Usage Example

This example demonstrates how to use AiraMoe (Activation-aware Improved Rank Allocation with Mixture of Experts)
which combines CoLA's collaborative low-rank adaptation with AwLoRA's three core technologies.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import get_peft_model, TaskType
from peft.tuners.aira_moe.config import AiraMoeConfig

# GPU Configuration - Only use GPU 0 and 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def setup_gpu_environment():
    """Setup GPU environment and return available devices."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return num_gpus
    else:
        print("CUDA not available, using CPU")
        return 0

def prepare_wikitext_data(tokenizer, max_length=512, batch_size=16):
    """Prepare WikiText-2 dataset for training."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Filter out empty texts and tokenize
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Create DataLoader
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

def basic_usage_example():
    """Basic AiraMoe usage example."""
    print("=== Basic AiraMoe Usage ===")
    
    # Setup GPU environment
    num_gpus = setup_gpu_environment()
    
    # Load model with auto device mapping
    model = AutoModelForCausalLM.from_pretrained(
        "/data/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure AiraMoe with basic settings
    config = AiraMoeConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        num_A=2,  # CoLA collaborative strategy
        num_B=2,
    )
    
    # Apply AiraMoe
    peft_model = get_peft_model(model, config)
    
    print(f"Original parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in peft_model.parameters()):,}")
    
    # Debug: Check all parameter names and their trainable status
    print("\nDebugging all parameters:")
    lora_params_found = 0
    for name, param in peft_model.named_parameters():
        if "lora" in name.lower() or "aira_moe" in name.lower():
            print(f"  {name}: dtype={param.dtype}, device={param.device}, requires_grad={param.requires_grad}")
            lora_params_found += 1
            if lora_params_found >= 5:  # Limit output
                break
    
    if lora_params_found == 0:
        print("  No LoRA parameters found! Checking first few parameters:")
        for i, (name, param) in enumerate(peft_model.named_parameters()):
            print(f"  {name}: requires_grad={param.requires_grad}")
            if i >= 10:  # Show first 10 parameters
                break
    
    # Check if any modules were actually replaced
    print("\nChecking module types:")
    for name, module in peft_model.named_modules():
        if "q_proj" in name:
            print(f"  {name}: {type(module)}")
            break
    
    # Prepare data
    dataloader = prepare_wikitext_data(tokenizer, max_length=512, batch_size=4)
    
    # Test forward pass with real data
    print("Testing forward pass with WikiText-2 data...")
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        print(f"Input dtype: {input_ids.dtype}, device: {input_ids.device}")
        
        with torch.no_grad():
            # Use autocast for mixed precision to handle dtype mismatch
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                output = peft_model(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(output, 'logits'):
            print(f"Output logits shape: {output.logits.shape}")
        else:
            print(f"Output shape: {output.shape}")
        break  # Only test one batch
    
    return peft_model, tokenizer, dataloader

def advanced_usage_example():
    """Advanced AiraMoe usage with all AwLoRA technologies enabled."""
    print("\n=== Advanced AiraMoe Usage (All AwLoRA Technologies) ===")
    
    # Setup GPU environment
    num_gpus = setup_gpu_environment()
    
    # Load model with auto device mapping
    model = AutoModelForCausalLM.from_pretrained(
        "/data/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure AiraMoe with all AwLoRA technologies
    config = AiraMoeConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        num_A=2,
        num_B=2,
        
        # AwLoRA Core Technology 1: Layer-wise Rank Allocation
        use_layer_wise_rank=True,
        lod_threshold_M=2.0,
        theta_type="lod",
        rank_budget=64,
        min_rank=1,
        max_rank=16,
        objective_function="log",
        
        # AwLoRA Core Technology 2: AwSVD Initialization
        use_awsvd_init=True,
        awsvd_collect_steps=100,
        
        # AwLoRA Core Technology 3: Activation-aware Weighting
        use_activation_aware=True,
        activation_aware_mode="inps",
        activation_normalize=True,
    )
    
    # Apply AiraMoe
    peft_model = get_peft_model(model, config)
    
    print(f"Model with all AwLoRA technologies enabled")
    print(f"Trainable parameters: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad):,}")
    
    # Prepare data
    dataloader = prepare_wikitext_data(tokenizer, max_length=512, batch_size=8)
    
    # Test forward pass with real data
    print("Testing forward pass with WikiText-2 data...")
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        with torch.no_grad():
            # Use autocast for mixed precision to handle dtype mismatch
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                output = peft_model(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(output, 'logits'):
            print(f"Output logits shape: {output.logits.shape}")
        else:
            print(f"Output shape: {output.shape}")
        break  # Only test one batch
    
    return peft_model, tokenizer, dataloader

def layer_wise_rank_allocation_example():
    """Demonstrate layer-wise rank allocation."""
    print("\n=== Layer-wise Rank Allocation Example ===")
    
    # Setup GPU environment
    num_gpus = setup_gpu_environment()
    
    # Load model with auto device mapping
    model = AutoModelForCausalLM.from_pretrained(
        "/data/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AiraMoeConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,  # Base rank
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        use_layer_wise_rank=True,
        rank_budget=32,  # Total rank budget
        min_rank=1,
        max_rank=12,
        objective_function="log",
    )
    
    peft_model = get_peft_model(model, config)
    
    # Prepare WikiText-2 data for layer importance computation
    dataloader = prepare_wikitext_data(tokenizer, max_length=512, batch_size=16)
    
    # Compute layer importance
    print("Computing layer importance...")
    layer_importance = peft_model.compute_layer_importance(dataloader, max_samples=160)
    
    print("Layer importance scores:")
    for layer_name, importance in layer_importance.items():
        print(f"  {layer_name}: LOD={importance['lod_mean']:.4f}, Act={importance['activation_mean']:.4f}")
    
    # Optimize rank allocation
    print("\nOptimizing rank allocation...")
    rank_allocation = peft_model.optimize_layer_ranks(layer_importance, objective_function="log")
    
    print("Optimized rank allocation:")
    total_ranks = 0
    for layer_name, rank in rank_allocation.items():
        print(f"  {layer_name}: rank={rank}")
        total_ranks += rank
    print(f"Total ranks used: {total_ranks}")
    
    # Apply optimized ranks
    print("\nApplying optimized rank allocation...")
    peft_model.apply_layer_wise_ranks(rank_allocation)
    
    print("Layer-wise rank allocation completed!")
    
    return peft_model, tokenizer, dataloader

def activation_aware_weighting_example():
    """Demonstrate activation-aware weighting."""
    print("\n=== Activation-aware Weighting Example ===")
    
    # Setup GPU environment
    num_gpus = setup_gpu_environment()
    
    # Model without activation-aware weighting
    print("Loading model for standard configuration...")
    base_model_standard = AutoModelForCausalLM.from_pretrained(
        "/data/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
        
    config_standard = AiraMoeConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        use_activation_aware=False,
    )
    
    # Model with activation-aware weighting
    print("Loading model for activation-aware configuration...")
    base_model_aware = AutoModelForCausalLM.from_pretrained(
        "/data/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
        
    config_aware = AiraMoeConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        use_activation_aware=True,
        activation_aware_mode="inps",
        activation_normalize=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    dataloader = prepare_wikitext_data(tokenizer, max_length=128, batch_size=16)
    
    # Test both models
    print("Creating PEFT models...")
    model_standard = get_peft_model(base_model_standard, config_standard)
    model_aware = get_peft_model(base_model_aware, config_aware)
    
    print("Running inference...")
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        print("Input activation statistics:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Input range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        
        with torch.no_grad():
            # Use autocast for mixed precision to handle dtype mismatch
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                output_standard = model_standard(input_ids=input_ids, attention_mask=attention_mask)
                output_aware = model_aware(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(output_standard, 'logits'):
            print(f"Standard model output logits variance: {output_standard.logits.var():.4f}")
            print(f"Activation-aware model output logits variance: {output_aware.logits.var():.4f}")
        else:
            print(f"Standard model output variance: {output_standard.var():.4f}")
            print(f"Activation-aware model output variance: {output_aware.var():.4f}")
        break  # Only test one batch
    
    return model_standard, model_aware, tokenizer, dataloader

def main():
    """Run all examples."""
    print("AiraMoe: Activation-aware Improved Rank Allocation with Mixture of Experts")
    print("=" * 80)
    
    try:
        # Basic usage
        basic_model, tokenizer1, dataloader1 = basic_usage_example()
        
        # Advanced usage with all technologies
        advanced_model, tokenizer2, dataloader2 = advanced_usage_example()
        
        # Layer-wise rank allocation
        rank_model, tokenizer3, dataloader3 = layer_wise_rank_allocation_example()
        
        # Activation-aware weighting
        standard_model, aware_model, tokenizer4, dataloader4 = activation_aware_weighting_example()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("\nAiraMoe combines:")
        print("1. CoLA's collaborative low-rank adaptation (num_A, num_B)")
        print("2. AwLoRA's layer-wise rank allocation (LOD-based optimization)")
        print("3. AwLoRA's AwSVD initialization (activation-aware SVD)")
        print("4. AwLoRA's activation-aware weighting (dynamic importance)")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 