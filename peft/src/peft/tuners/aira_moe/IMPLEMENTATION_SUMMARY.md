# AiraMoe Implementation Summary

## 概述

AiraMoe (Activation-aware Improved Rank Allocation with Mixture of Experts) 是一个创新的参数高效微调方法，成功结合了CoLA的协同低秩适应和AwLoRA的三个核心技术。

## 实现的文件结构

```
peft/src/peft/tuners/aira_moe/
├── __init__.py                 # 模块导出
├── config.py                   # AiraMoeConfig配置类
├── layer.py                    # AiraMoeLayer层实现
├── model.py                    # AiraMoeModel模型类
├── bnb.py                      # BitsAndBytes量化支持
├── gptq.py                     # GPTQ量化支持
├── test_aira_moe.py           # 测试脚本
├── example_usage.py           # 使用示例
├── README.md                  # 用户文档
└── IMPLEMENTATION_SUMMARY.md  # 实现总结
```

## 核心技术实现

### 1. CoLA协同低秩适应

**实现位置**: `layer.py` 中的 `update_layer()` 方法

**核心特性**:
- 支持多个A矩阵 (`num_A`) 和B矩阵 (`num_B`)
- 协同策略: `result += lora_B[j](lora_A[i](x)) * scaling`
- 所有A-B组合的输出求和

**代码示例**:
```python
# CoLA collaborative strategy
for i in range(self.num_A[active_adapter]):
    for j in range(self.num_B[active_adapter]):
        result += lora_B[j](lora_A[i](dropout(x))) * scaling
```

### 2. AwLoRA核心技术1: 基于LOD的层级Rank分配

**实现位置**: `model.py` 中的 `compute_layer_importance()` 和 `optimize_layer_ranks()` 方法

**核心算法**:
1. **LOD异常值计算**:
   ```python
   A_ij = torch.abs(weight_tensor) * input_norm.unsqueeze(0)
   mean_A_l = A_ij.mean()
   lod_value = (A_ij > M * mean_A_l).float().mean()
   ```

2. **约束优化求解**:
   - 目标函数: log, linear, exp2, cubic
   - 约束条件: rank预算、最小/最大rank
   - 使用scipy.optimize.minimize求解

3. **动态rank更新**:
   ```python
   def apply_layer_wise_ranks(self, rank_allocation):
       # 重新创建LoRA矩阵
       module.lora_A[adapter_name] = nn.ModuleList([
           nn.Linear(in_features, new_rank, bias=False) for _ in range(num_A)
       ])
   ```

### 3. AwLoRA核心技术2: AwSVD初始化

**实现位置**: `layer.py` 中的 `_awsvd_init()` 方法

**核心算法**:
1. **激活感知缩放**:
   ```python
   X = inputs.view(-1, weight.size(1))
   S_diag = torch.sqrt(torch.mean(X**2, dim=0))  # RMS
   S = torch.diag(S_diag)
   W_prime = weight @ S
   ```

2. **SVD分解与初始化**:
   ```python
   U, S_vals, Vh = torch.linalg.svd(W_prime, full_matrices=False)
   S_inv = torch.diag(1.0 / (S_diag + 1e-8))
   lora_A = (Vh @ S_inv) / self.num_A[adapter_name]
   lora_B = (U @ torch.diag(S_vals)) / self.num_B[adapter_name]
   ```

### 4. AwLoRA核心技术3: 激活感知加权

**实现位置**: `layer.py` 中的 `_compute_activation_weights()` 和前向传播

**核心算法**:
1. **激活权重计算**:
   ```python
   def _compute_activation_weights(self, x, mode, normalize=True):
       if mode == "inps":
           s_i = torch.mean(torch.abs(x), dim=0)  # 输入激活
       elif mode == "outps":
           base_output = F.linear(x, self.get_base_layer().weight, bias)
           s_i = torch.mean(torch.abs(base_output), dim=0)  # 输出激活
       
       if normalize:
           s_i = (s_i - s_i.min()) / (s_i.max() - s_i.min())
       return s_i
   ```

2. **加权前向传播**:
   ```python
   if self.activation_aware_mode[active_adapter] == "inps":
       x_weighted = dropout(x) * activation_weights.unsqueeze(0)
       result += lora_B[j](lora_A[i](x_weighted)) * scaling
   else:  # outps mode
       lora_output = lora_B[j](lora_A[i](dropout(x)))
       result += (lora_output * activation_weights.unsqueeze(0)) * scaling
   ```

## 配置参数详解

### 基础LoRA参数
- `r`: LoRA rank维度
- `lora_alpha`: LoRA缩放参数
- `lora_dropout`: Dropout率
- `target_modules`: 目标模块列表

### CoLA参数
- `num_A`: A矩阵数量 (默认: 1)
- `num_B`: B矩阵数量 (默认: 1)

### AwLoRA技术1参数 (层级Rank分配)
- `use_layer_wise_rank`: 启用层级rank分配
- `lod_threshold_M`: LOD阈值倍数 (默认: 2.0)
- `theta_type`: 重要性指标类型 ("act" | "lod")
- `rank_budget`: 总rank预算
- `min_rank`, `max_rank`: rank范围
- `objective_function`: 优化目标函数

### AwLoRA技术2参数 (AwSVD初始化)
- `use_awsvd_init`: 启用AwSVD初始化
- `awsvd_collect_steps`: 激活统计收集步数

### AwLoRA技术3参数 (激活感知加权)
- `use_activation_aware`: 启用激活感知加权
- `activation_aware_mode`: 加权模式 ("inps" | "outps")
- `activation_normalize`: 是否归一化权重

## 使用方法

### 基础使用
```python
from peft import AiraMoeConfig, get_peft_model

config = AiraMoeConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    num_A=2,  # CoLA协同策略
    num_B=2,
)

model = get_peft_model(base_model, config)
```

### 启用所有AwLoRA技术
```python
config = AiraMoeConfig(
    r=8,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    num_A=2, num_B=2,
    
    # 层级rank分配
    use_layer_wise_rank=True,
    rank_budget=64,
    objective_function="log",
    
    # AwSVD初始化
    use_awsvd_init=True,
    
    # 激活感知加权
    use_activation_aware=True,
    activation_aware_mode="inps",
)
```

### 动态rank分配
```python
# 计算层重要性
layer_importance = model.compute_layer_importance(train_loader)

# 优化rank分配
rank_allocation = model.optimize_layer_ranks(layer_importance)

# 应用优化结果
model.apply_layer_wise_ranks(rank_allocation)
```

## 技术优势

1. **参数效率**: 智能rank分配优化参数使用
2. **初始化质量**: AwSVD提供更好的初始化
3. **动态适应**: 激活感知加权自适应调整
4. **协同效应**: 多矩阵协同策略增强表达能力
5. **通用性**: 支持各种Transformer架构

## 量化支持

- **BitsAndBytes**: `bnb.py` 支持8bit和4bit量化
- **GPTQ**: `gptq.py` 支持GPTQ量化
- 自动检测量化配置并应用相应的层实现

## 测试与验证

- `test_aira_moe.py`: 完整的功能测试
- `example_usage.py`: 详细的使用示例
- 支持基础功能、AwLoRA技术、协同策略的独立测试

## 集成状态

AiraMoe已完全集成到PEFT库中:

1. **类型注册**: `PeftType.AIRA_MOE` 已添加到 `peft_types.py`
2. **映射配置**: 已添加到 `mapping.py` 的配置和模型映射
3. **模块导出**: 已添加到 `tuners/__init__.py`
4. **文档完整**: 提供完整的README和使用示例

## 依赖要求

- **核心依赖**: torch, transformers, peft
- **可选依赖**: scipy (用于层级rank分配优化)
- **量化依赖**: bitsandbytes, auto-gptq (可选)

## 性能特点

- **内存效率**: 相比全参数微调大幅减少内存使用
- **计算开销**: 激活感知加权增加少量计算开销
- **收敛速度**: AwSVD初始化加速收敛
- **适应性**: 动态rank分配提高参数利用效率

AiraMoe成功实现了CoLA和AwLoRA技术的完美融合，为参数高效微调提供了最先进的解决方案。 