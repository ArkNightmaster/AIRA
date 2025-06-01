#!/usr/bin/env python3
"""
Test script for AIRA_MoE integration in LLaMA-Factory.
"""

import os
import sys
import torch
import tempfile
from dataclasses import dataclass
from typing import Optional

# Add LLaMA-Factory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llamafactory.hparams import FinetuningArguments, ModelArguments
from llamafactory.model import load_model, load_tokenizer


@dataclass
class TestConfig:
    """Test configuration for AIRA_MoE integration."""
    model_name: str = "/data/Llama-3.1-8B-Instruct"  # Small model for testing
    finetuning_type: str = "aira_moe"
    lora_rank: int = 4
    lora_alpha: int = 8
    num_A: int = 2
    num_B: int = 2
    use_layer_wise_rank: bool = False  # Disable for simple test
    use_awsvd_init: bool = False       # Disable for simple test
    use_activation_aware: bool = True  # Enable activation-aware weighting


def test_aira_moe_basic():
    """Test basic AIRA_MoE functionality."""
    print("Testing AIRA_MoE basic functionality...")
    
    config = TestConfig()
    
    # Create model arguments
    model_args = ModelArguments(
        model_name_or_path=config.model_name,
        cache_dir=None,
    )
    
    # Create finetuning arguments
    finetuning_args = FinetuningArguments(
        finetuning_type=config.finetuning_type,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_target="all",
        num_A=config.num_A,
        num_B=config.num_B,
        use_layer_wise_rank=config.use_layer_wise_rank,
        use_awsvd_init=config.use_awsvd_init,
        use_activation_aware=config.use_activation_aware,
        activation_aware_mode="inps",
        activation_normalize=True,
    )
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(model_args)
        
        # Load model with AIRA_MoE
        print("Loading model with AIRA_MoE...")
        model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
        
        # Test forward pass
        print("Testing forward pass...")
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input: {test_input}")
        print(f"  Output shape: {outputs.logits.shape}")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Parameter statistics:")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params:.4f}")
        
        # Check AIRA_MoE specific attributes
        print(f"✓ AIRA_MoE configuration:")
        print(f"  Finetuning type: {finetuning_args.finetuning_type}")
        print(f"  LoRA rank: {finetuning_args.lora_rank}")
        print(f"  num_A: {finetuning_args.num_A}")
        print(f"  num_B: {finetuning_args.num_B}")
        print(f"  Use activation-aware: {finetuning_args.use_activation_aware}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aira_moe_advanced():
    """Test AIRA_MoE with advanced features."""
    print("\nTesting AIRA_MoE advanced functionality...")
    
    config = TestConfig()
    
    # Create model arguments
    model_args = ModelArguments(
        model_name_or_path=config.model_name,
        cache_dir=None,
    )
    
    # Create finetuning arguments with advanced features
    finetuning_args = FinetuningArguments(
        finetuning_type=config.finetuning_type,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_target="all",
        num_A=config.num_A,
        num_B=config.num_B,
        # Enable AwLoRA technologies
        use_layer_wise_rank=True,
        rank_budget=32,
        theta_type="lod",
        objective_function="log",
        use_awsvd_init=True,
        awsvd_collect_steps=10,  # Small number for testing
        use_activation_aware=True,
        activation_aware_mode="inps",
        activation_normalize=True,
    )
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(model_args)
        
        # Load model with AIRA_MoE advanced features
        print("Loading model with AIRA_MoE advanced features...")
        model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
        
        # Test forward pass
        print("Testing forward pass with advanced features...")
        test_input = "Hello, how are you today?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Advanced forward pass successful!")
        print(f"  Input: {test_input}")
        print(f"  Output shape: {outputs.logits.shape}")
        
        # Check advanced AIRA_MoE configuration
        print(f"✓ Advanced AIRA_MoE configuration:")
        print(f"  Layer-wise rank allocation: {finetuning_args.use_layer_wise_rank}")
        print(f"  Rank budget: {finetuning_args.rank_budget}")
        print(f"  AwSVD initialization: {finetuning_args.use_awsvd_init}")
        print(f"  Activation-aware weighting: {finetuning_args.use_activation_aware}")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all AIRA_MoE integration tests."""
    print("=" * 60)
    print("AIRA_MoE Integration Test for LLaMA-Factory")
    print("=" * 60)
    
    # Run basic test
    basic_success = test_aira_moe_basic()
    
    # Run advanced test
    advanced_success = test_aira_moe_advanced()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Basic functionality: {'✓ PASSED' if basic_success else '✗ FAILED'}")
    print(f"  Advanced functionality: {'✓ PASSED' if advanced_success else '✗ FAILED'}")
    
    if basic_success and advanced_success:
        print("\n🎉 All tests passed! AIRA_MoE integration is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit(main()) 