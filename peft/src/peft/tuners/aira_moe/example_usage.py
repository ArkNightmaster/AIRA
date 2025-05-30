#!/usr/bin/env python3
"""
AiraMoe Usage Example

This example demonstrates how to use AiraMoe (Activation-aware Improved Rank Allocation with Mixture of Experts)
which combines CoLA's collaborative low-rank adaptation with AwLoRA's three core technologies.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, TaskType
from .config import AiraMoeConfig

def basic_usage_example():
    """Basic AiraMoe usage example."""
    print("=== Basic AiraMoe Usage ===")
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 256)
            self.linear2 = nn.Linear(256, 128)
            self.linear3 = nn.Linear(128, 64)
            
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            return self.linear3(x)
    
    model = SimpleModel()
    
    # Configure AiraMoe with basic settings
    config = AiraMoeConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        target_modules=["linear1", "linear2", "linear3"],
        lora_dropout=0.1,
        num_A=2,  # CoLA collaborative strategy
        num_B=2,
    )
    
    # Apply AiraMoe
    peft_model = get_peft_model(model, config)
    
    print(f"Original parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in peft_model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 512)
    output = peft_model(x)
    print(f"Output shape: {output.shape}")
    
    return peft_model

def advanced_usage_example():
    """Advanced AiraMoe usage with all AwLoRA technologies enabled."""
    print("\n=== Advanced AiraMoe Usage (All AwLoRA Technologies) ===")
    
    # Create a simple model
    class AdvancedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(512, 512) for _ in range(4)
            ])
            self.output = nn.Linear(512, 10)
            
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.output(x)
    
    model = AdvancedModel()
    
    # Configure AiraMoe with all AwLoRA technologies
    config = AiraMoeConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        target_modules=["layers.0", "layers.1", "layers.2", "layers.3"],
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
    
    # Test forward pass
    x = torch.randn(8, 512)
    output = peft_model(x)
    print(f"Output shape: {output.shape}")
    
    return peft_model

def layer_wise_rank_allocation_example():
    """Demonstrate layer-wise rank allocation."""
    print("\n=== Layer-wise Rank Allocation Example ===")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    
    config = AiraMoeConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=4,  # Base rank
        target_modules=["0", "2", "4", "6"],  # Linear layers
        use_layer_wise_rank=True,
        rank_budget=32,  # Total rank budget
        min_rank=1,
        max_rank=12,
        objective_function="log",
    )
    
    peft_model = get_peft_model(model, config)
    
    # Create dummy training data
    class DummyDataLoader:
        def __init__(self, batch_size=16, num_batches=10):
            self.batch_size = batch_size
            self.num_batches = num_batches
            self.current = 0
            
        def __iter__(self):
            return self
            
        def __next__(self):
            if self.current >= self.num_batches:
                raise StopIteration
            self.current += 1
            return torch.randn(self.batch_size, 256)
    
    train_loader = DummyDataLoader()
    
    # Compute layer importance
    print("Computing layer importance...")
    layer_importance = peft_model.compute_layer_importance(train_loader, max_samples=160)
    
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
    
    return peft_model

def activation_aware_weighting_example():
    """Demonstrate activation-aware weighting."""
    print("\n=== Activation-aware Weighting Example ===")
    
    # Compare models with and without activation-aware weighting
    base_model = nn.Linear(128, 64)
    
    # Model without activation-aware weighting
    config_standard = AiraMoeConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        target_modules=[""],  # Will be applied to the single linear layer
        use_activation_aware=False,
    )
    
    # Model with activation-aware weighting
    config_aware = AiraMoeConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        target_modules=[""],
        use_activation_aware=True,
        activation_aware_mode="inps",
        activation_normalize=True,
    )
    
    # Create test input with varying activation patterns
    x = torch.randn(16, 128)
    x[:, :64] *= 2.0  # Make first half more active
    x[:, 64:] *= 0.5  # Make second half less active
    
    print("Input activation statistics:")
    print(f"  First half mean: {x[:, :64].abs().mean():.4f}")
    print(f"  Second half mean: {x[:, 64:].abs().mean():.4f}")
    
    # Test both models
    model_standard = get_peft_model(base_model.clone(), config_standard)
    model_aware = get_peft_model(base_model.clone(), config_aware)
    
    with torch.no_grad():
        output_standard = model_standard(x)
        output_aware = model_aware(x)
    
    print(f"Standard model output variance: {output_standard.var():.4f}")
    print(f"Activation-aware model output variance: {output_aware.var():.4f}")
    
    return model_standard, model_aware

def main():
    """Run all examples."""
    print("AiraMoe: Activation-aware Improved Rank Allocation with Mixture of Experts")
    print("=" * 80)
    
    try:
        # Basic usage
        basic_model = basic_usage_example()
        
        # Advanced usage with all technologies
        advanced_model = advanced_usage_example()
        
        # Layer-wise rank allocation
        rank_model = layer_wise_rank_allocation_example()
        
        # Activation-aware weighting
        standard_model, aware_model = activation_aware_weighting_example()
        
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