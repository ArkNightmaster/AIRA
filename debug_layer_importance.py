#!/usr/bin/env python3
"""
Debug script for AIRA_MoE layer importance computation error
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset

# Add LLaMA-Factory to path
sys.path.insert(0, '/data1/ldz/CoLA/LLaMA-Factory/src')

def debug_layer_importance():
    """Debug the layer importance computation error"""
    print("=== Debugging AIRA_MoE Layer Importance Error ===")
    
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
        print(f"  - Model type: {type(model)}")
        
        # Create test data similar to what the trainer creates
        print("Creating test data...")
        
        sample_texts = [
            "Hello, this is a test for AIRA_MoE layer importance.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        # Tokenize texts
        max_length = 64
        tokenized = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        # Create data in the same format as the trainer
        simple_data = []
        for i in range(len(sample_texts)):
            sample_dict = {
                'input_ids': tokenized['input_ids'][i:i+1],  # [1, seq_len]
                'attention_mask': tokenized['attention_mask'][i:i+1]  # [1, seq_len]
            }
            simple_data.append(sample_dict)
        
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = SimpleDataset(simple_data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        print("✓ Test data created")
        print(f"  - Number of samples: {len(simple_data)}")
        print(f"  - Sample input_ids shape: {simple_data[0]['input_ids'].shape}")
        print(f"  - Sample attention_mask shape: {simple_data[0]['attention_mask'].shape}")
        
        # Test a single forward pass first
        print("Testing single forward pass...")
        model.eval()
        device = next(model.parameters()).device
        device_type = device.type if hasattr(device, 'type') else 'cuda' if 'cuda' in str(device) else 'cpu'
        
        with torch.no_grad():
            sample_batch = simple_data[0]
            input_ids = sample_batch['input_ids'].to(device)
            attention_mask = sample_batch['attention_mask'].to(device)
            
            print(f"  - Input IDs shape: {input_ids.shape}")
            print(f"  - Attention mask shape: {attention_mask.shape}")
            print(f"  - Device: {device}, Device type: {device_type}")
            
            try:
                # Test the forward pass
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                print("✓ Single forward pass successful")
                print(f"  - Output type: {type(outputs)}")
                if hasattr(outputs, 'logits'):
                    print(f"  - Logits shape: {outputs.logits.shape}")
            except Exception as e:
                print(f"✗ Single forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Now test layer importance computation
        print("Testing layer importance computation...")
        
        try:
            layer_importance = model.compute_layer_importance(
                dataloader,
                device=str(device),
                max_samples=2
            )
            print("✓ Layer importance computation successful")
            print(f"  - Computed importance for {len(layer_importance)} layers")
            
        except Exception as e:
            print(f"✗ Layer importance computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_layer_importance() 