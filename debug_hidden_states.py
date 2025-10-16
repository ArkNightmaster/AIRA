#!/usr/bin/env python3
"""
Debug script for hidden_states dimension issue in AIRA_MoE
"""

import os
import sys
import torch
import torch.nn as nn

# Add LLaMA-Factory to path
sys.path.insert(0, '/data1/ldz/CoLA/LLaMA-Factory/src')

def debug_hidden_states():
    """Debug the hidden_states dimension issue"""
    print("=== Debugging Hidden States Dimension Issue ===")
    
    try:
        from llamafactory.hparams import ModelArguments, FinetuningArguments
        from llamafactory.model import load_model, load_tokenizer
        
        print("Loading model...")
        
        # Create model arguments
        model_args = ModelArguments(
            model_name_or_path="/data/Llama-3.1-8B-Instruct",
            cache_dir=None,
        )
        
        # Create AIRA_MoE finetuning arguments
        finetuning_args = FinetuningArguments(
            finetuning_type="aira_moe",
            lora_rank=4,
            lora_alpha=8,
            lora_target=["q_proj", "v_proj"],
            num_A=1,
            num_B=1,
            use_layer_wise_rank=True,
            rank_budget=32,
            theta_type="lod",
            objective_function="log",
            use_awsvd_init=False,
            use_activation_aware=True,
            activation_aware_mode="inps",
            activation_normalize=True,
        )
        
        # Load tokenizer and model
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
        
        print("✓ Model loaded successfully")
        
        # Create test data
        sample_text = "Hello, this is a test."
        tokenized = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=32)
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        
        # Hook to capture hidden_states in attention layers
        hidden_states_info = []
        
        def attention_hook(module, input, output):
            if hasattr(module, '__class__') and 'Attention' in module.__class__.__name__:
                hidden_states = input[0] if input else None
                if hidden_states is not None:
                    hidden_states_info.append({
                        'module_name': module.__class__.__name__,
                        'hidden_states_shape': hidden_states.shape,
                        'hidden_states_dim': hidden_states.dim()
                    })
                    print(f"  - {module.__class__.__name__}: hidden_states shape = {hidden_states.shape}")
        
        # Register hooks on all attention modules
        hooks = []
        for name, module in model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        print("Testing forward pass with hooks...")
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    print("✓ Forward pass successful")
                except Exception as e:
                    print(f"✗ Forward pass failed: {e}")
                    print(f"Hidden states info collected: {len(hidden_states_info)}")
                    for info in hidden_states_info:
                        print(f"  - {info}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
        
        return True
        
    except Exception as e:
        print(f"✗ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_hidden_states() 