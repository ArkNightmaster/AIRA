#!/usr/bin/env python3
"""
测试AIRA_MoE层重要性计算修复
"""

import os
import sys
import torch

# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_layer_importance_fix():
    """测试层重要性计算修复"""
    print("=== 测试AIRA_MoE层重要性计算修复 ===")
    
    try:
        sys.path.insert(0, '/data1/ldz/CoLA/LLaMA-Factory/src')
        from llamafactory.hparams import ModelArguments, FinetuningArguments
        from llamafactory.model import load_model, load_tokenizer
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        from torch.utils.data import DataLoader, TensorDataset
        
        print("正在加载模型...")
        
        # 创建模型参数
        model_args = ModelArguments(
            model_name_or_path="/data/Llama-3.1-8B-Instruct",
            cache_dir=None,
        )
        
        # 创建AIRA_MoE微调参数 - 启用层级rank分配
        finetuning_args = FinetuningArguments(
            finetuning_type="aira_moe",
            lora_rank=4,
            lora_alpha=8,
            lora_target=["q_proj", "v_proj"],
            num_A=1,
            num_B=1,
            use_layer_wise_rank=True,  # 启用层级rank分配
            rank_budget=32,
            theta_type="lod",
            objective_function="log",
            use_awsvd_init=False,
            use_activation_aware=True,
            activation_aware_mode="inps",
            activation_normalize=True,
        )
        
        # 加载tokenizer和模型
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        
        # 设置padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
        
        print("✓ 模型加载成功")
        print(f"  - 模型类型: {type(model)}")
        print(f"  - PEFT类型: {list(model.peft_config.values())[0].peft_type}")
        
        # 创建简单的训练数据 - 只使用两条测试数据
        print("创建测试数据...")
        sample_texts = [
            "Hello, this is a test for AIRA_MoE layer importance.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        # 创建数据集 - 确保所有样本长度一致
        max_length = 32
        
        # tokenize所有文本
        all_inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        # 创建DataLoader
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
            
            def __len__(self):
                return len(self.input_ids)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx]
                }
        
        dataset = SimpleDataset(all_inputs['input_ids'], all_inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)  # 使用batch_size=2
        
        print("✓ 测试数据创建成功")
        print(f"  - 数据集大小: {len(dataset)}")
        print(f"  - Batch大小: 2")
        
        # 测试层重要性计算
        print("测试层重要性计算...")
        
        # 直接调用compute_layer_importance方法
        device = str(next(model.parameters()).device)
        layer_importance = model.compute_layer_importance(
            dataloader,
            device=device,
            max_samples=2  # 只使用2个样本
        )
        
        print("✓ 层重要性计算成功")
        print(f"  - 计算了 {len(layer_importance)} 个层的重要性")
        
        # 显示部分结果
        for i, (layer_name, importance) in enumerate(layer_importance.items()):
            if i < 3:  # 只显示前3个
                print(f"  - {layer_name}: LOD={importance['lod_mean']:.4f}, Act={importance['activation_mean']:.4f}")
        
        # 测试rank优化
        print("测试rank优化...")
        rank_allocation = model.optimize_layer_ranks(
            layer_importance, 
            objective_function="log"
        )
        
        print("✓ Rank优化成功")
        print(f"  - 优化了 {len(rank_allocation)} 个层的rank")
        
        total_ranks = sum(rank_allocation.values())
        print(f"  - 总rank使用: {total_ranks}/{finetuning_args.rank_budget}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行测试"""
    print("AIRA_MoE 层重要性计算修复测试")
    print("=" * 50)
    
    success = test_layer_importance_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 测试通过！层重要性计算修复成功！")
        print("\n现在可以安全使用generality_full.sh进行完整的AIRA_MoE训练")
    else:
        print("❌ 测试失败，需要进一步检查")

if __name__ == "__main__":
    main() 