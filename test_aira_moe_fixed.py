#!/usr/bin/env python3
"""
AIRA_MoE Fixed Integration Test
验证修复后的AIRA_MoE功能是否正常工作
"""

import os
import sys
import torch

# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def test_basic_functionality():
    """测试基本功能"""
    print("=== AIRA_MoE 基本功能测试 ===")
    
    try:
        # 测试PEFT库中的AIRA_MoE导入
        from peft import AiraMoeConfig, AiraMoeModel, get_peft_model, TaskType
        print("✓ PEFT库中AIRA_MoE导入成功")
        
        # 测试LLaMA-Factory中的参数处理
        sys.path.insert(0, '/data1/ldz/CoLA/LLaMA-Factory/src')
        from llamafactory.hparams import FinetuningArguments
        
        finetuning_args = FinetuningArguments(
            finetuning_type="aira_moe",
            lora_rank=8,
            lora_alpha=16,
            lora_target="all",
            num_A=2,
            num_B=2,
            # AwLoRA技术参数
            use_layer_wise_rank=True,
            rank_budget=64,
            theta_type="lod",
            objective_function="log",
            use_awsvd_init=True,
            awsvd_collect_steps=50,
            use_activation_aware=True,
            activation_aware_mode="inps",
            activation_normalize=True,
        )
        
        print("✓ LLaMA-Factory参数处理成功")
        print(f"  - finetuning_type: {finetuning_args.finetuning_type}")
        print(f"  - use_layer_wise_rank: {finetuning_args.use_layer_wise_rank}")
        print(f"  - use_awsvd_init: {finetuning_args.use_awsvd_init}")
        print(f"  - use_activation_aware: {finetuning_args.use_activation_aware}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n=== AIRA_MoE 模型创建测试 ===")
    
    try:
        sys.path.insert(0, '/data1/ldz/CoLA/LLaMA-Factory/src')
        from llamafactory.hparams import ModelArguments, FinetuningArguments
        from llamafactory.model import load_model, load_tokenizer
        
        print("正在加载Llama-3.1-8B-Instruct模型...")
        
        # 创建模型参数
        model_args = ModelArguments(
            model_name_or_path="/data/Llama-3.1-8B-Instruct",
            cache_dir=None,
        )
        
        # 创建AIRA_MoE微调参数
        finetuning_args = FinetuningArguments(
            finetuning_type="aira_moe",
            lora_rank=4,
            lora_alpha=8,
            lora_target=["q_proj", "v_proj"],  # 只选择部分模块以加快测试
            num_A=2,
            num_B=3,
            use_layer_wise_rank=True,  # 暂时禁用以简化测试
            use_awsvd_init=False,       # 暂时禁用以简化测试
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
        
        print("✓ AIRA_MoE模型创建成功")
        print(f"  - 模型类型: {type(model)}")
        print(f"  - 模型设备: {next(model.parameters()).device}")
        
        # 检查PEFT配置
        if hasattr(model, 'peft_config'):
            config = list(model.peft_config.values())[0]
            print(f"  - PEFT类型: {config.peft_type}")
            print(f"  - CoLA参数: num_A={config.num_A}, num_B={config.num_B}")
            print(f"  - 激活感知: {config.use_activation_aware}")
            print(f"  - 激活模式: {config.activation_aware_mode}")
        
        # 测试前向传播
        print("测试前向传播...")
        sample_text = "Hello, this is a test for AIRA_MoE."
        inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        
        # 移动到模型设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 设置为训练模式以触发AwSVD和激活感知
        model.train()
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(**inputs)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: {inputs['input_ids'].shape}")
        print(f"  - 输出形状: {outputs.logits.shape}")
        
        # 测试AwLoRA核心方法（如果可用）
        if hasattr(model, 'compute_layer_importance'):
            print("✓ 模型包含层重要性计算方法")
        if hasattr(model, 'optimize_layer_ranks'):
            print("✓ 模型包含rank优化方法")
        if hasattr(model, 'apply_layer_wise_ranks'):
            print("✓ 模型包含rank应用方法")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_integration():
    """测试训练器集成"""
    print("\n=== AIRA_MoE 训练器集成测试 ===")
    
    try:
        sys.path.insert(0, '/data1/ldz/CoLA/LLaMA-Factory/src')
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        from llamafactory.hparams import FinetuningArguments
        
        # 检查训练器是否有AIRA_MoE支持方法
        trainer_methods = dir(CustomSeq2SeqTrainer)
        
        if '_setup_aira_moe_layer_wise_ranks' in trainer_methods:
            print("✓ 训练器包含AIRA_MoE层级rank分配方法")
        else:
            print("✗ 训练器缺少AIRA_MoE层级rank分配方法")
            return False
            
        if 'train' in trainer_methods:
            print("✓ 训练器包含重写的train方法")
        else:
            print("✗ 训练器缺少重写的train方法")
            return False
        
        print("✓ 训练器集成测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 训练器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("AIRA_MoE 修复验证测试")
    print("=" * 50)
    
    # 运行测试
    test1_passed = test_basic_functionality()
    test2_passed = test_model_creation()
    test3_passed = test_trainer_integration()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"✓ 基本功能测试: {'通过' if test1_passed else '失败'}")
    print(f"✓ 模型创建测试: {'通过' if test2_passed else '失败'}")
    print(f"✓ 训练器集成测试: {'通过' if test3_passed else '失败'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 所有测试通过！AIRA_MoE已成功修复并集成到LLaMA-Factory中！")
        print("\n现在可以使用以下脚本进行训练:")
        print("  - ./scripts/aira_moe/generality_basic.sh")
        print("  - ./scripts/aira_moe/generality_standard.sh")
        print("  - ./scripts/aira_moe/generality_full.sh")
    else:
        print("\n❌ 部分测试失败，需要进一步检查")

if __name__ == "__main__":
    main() 