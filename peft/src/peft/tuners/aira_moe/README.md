# AiraMoe: Activation-aware Improved Rank Allocation with Mixture of Experts

AiraMoe是一种创新的参数高效微调方法，结合了CoLA的协同低秩适应和AwLoRA的三个核心技术。

## 核心技术

### 1. 基于LOD的层级Rank分配
- 使用LOD (Layer-wise Outlier Detection) 异常值指标进行智能rank分配
- 支持多种目标函数：log, linear, exp2, cubic
- 约束优化求解最优rank分配

### 2. AwSVD初始化
- 激活感知的SVD初始化策略
- 根据输入激活对权重进行缩放后再进行SVD分解
- 提供更好的初始化质量，加速收敛

### 3. 激活感知加权前向传播
- 动态调整不同秩分量的贡献权重
- 支持基于输入激活(inps)和输出激活(outps)的加权
- 自适应归一化到[0,1]范围

## 配置参数

### 基础参数
- `r`: LoRA rank维度 (默认: 8)
- `lora_alpha`: LoRA缩放参数 (默认: 8)
- `lora_dropout`: Dropout率 (默认: 0.0)
- `num_A`: A矩阵数量，来自CoLA (默认: 1)
- `num_B`: B矩阵数量，来自CoLA (默认: 1)

### AwLoRA核心技术参数

#### 技术1: 层级Rank分配
- `use_layer_wise_rank`: 是否使用层级rank分配 (默认: False)
- `lod_threshold_M`: LOD异常值检测阈值倍数 (默认: 2.0)
- `theta_type`: 重要性指标类型 ("act" | "lod", 默认: "lod")
- `rank_budget`: 总rank预算 (默认: 64)
- `min_rank`: 最小rank (默认: 1)
- `max_rank`: 最大rank (默认: 16)
- `objective_function`: 优化目标函数 ("log" | "linear" | "exp2" | "cubic", 默认: "log")

#### 技术2: AwSVD初始化
- `use_awsvd_init`: 是否使用AwSVD初始化 (默认: False)
- `awsvd_collect_steps`: 收集激活统计的步数 (默认: 100)

#### 技术3: 激活感知加权
- `use_activation_aware`: 是否使用激活感知加权 (默认: False)
- `activation_aware_mode`: 加权模式 ("inps" | "outps", 默认: "inps")
- `activation_normalize`: 是否归一化激活权重 (默认: True)

## 使用示例

### 基础使用
```python
from peft import AiraMoeConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("your-model-name")

# 配置AiraMoe
config = AiraMoeConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    num_A=2,  # CoLA协同策略
    num_B=2,
)

# 应用AiraMoe
model = get_peft_model(model, config)
```

### 启用所有AwLoRA技术
```python
config = AiraMoeConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    num_A=2,
    num_B=2,
    
    # 启用层级rank分配
    use_layer_wise_rank=True,
    lod_threshold_M=2.0,
    theta_type="lod",
    rank_budget=64,
    min_rank=1,
    max_rank=16,
    objective_function="log",
    
    # 启用AwSVD初始化
    use_awsvd_init=True,
    awsvd_collect_steps=100,
    
    # 启用激活感知加权
    use_activation_aware=True,
    activation_aware_mode="inps",
    activation_normalize=True,
)

model = get_peft_model(model, config)
```

### 使用层级rank分配
```python
from torch.utils.data import DataLoader

# 创建训练数据加载器
train_loader = DataLoader(your_dataset, batch_size=32)

# 计算层重要性
layer_importance = model.compute_layer_importance(train_loader, max_samples=1000)

# 优化rank分配
rank_allocation = model.optimize_layer_ranks(layer_importance, objective_function="log")

# 应用优化的rank分配
model.apply_layer_wise_ranks(rank_allocation)

print("优化后的rank分配:", rank_allocation)
```

## 技术原理

### LOD异常值计算
```python
# 计算权重-激活交互矩阵
A_ij = torch.abs(weight_tensor) * input_norm.unsqueeze(0)
mean_A_l = A_ij.mean()

# LOD异常值比例
lod_value = (A_ij > M * mean_A_l).float().mean()
```

### AwSVD数学公式
```
原始: W = U @ S @ V^T
AwSVD: W' = W @ S_act, 其中 S_act = diag(sqrt(mean(X^2, dim=0)))
分解: W' = U' @ S' @ V'^T
初始化: A = V'^T @ S_act^(-1), B = U' @ diag(S')
```

### 激活感知权重计算
```python
# 通道重要性分数
s_i = torch.mean(torch.abs(activation), dim=0)

# 归一化到[0,1]
s_i = (s_i - s_i.min()) / (s_i.max() - s_i.min() + eps)
```

## 性能优势

1. **参数效率**: 通过智能rank分配，在相同参数预算下获得更好性能
2. **初始化质量**: AwSVD提供更好的初始化，加速收敛
3. **动态适应**: 激活感知加权根据数据特征动态调整
4. **协同效应**: 结合CoLA的多矩阵协同策略
5. **通用性**: 适用于各种Transformer架构

## 注意事项

1. **计算开销**: 激活感知加权会增加少量计算开销
2. **内存使用**: 需要额外存储激活统计信息
3. **超参数调优**: LOD阈值M和目标函数需要根据任务调整
4. **依赖项**: 层级rank分配需要scipy库进行优化

## 与其他方法的比较

| 方法 | 协同策略 | 智能Rank分配 | 激活感知初始化 | 动态加权 |
|------|----------|--------------|----------------|----------|
| LoRA | ❌ | ❌ | ❌ | ❌ |
| CoLA | ✅ | ❌ | ❌ | ❌ |
| AwLoRA | ❌ | ✅ | ✅ | ✅ |
| **AiraMoe** | ✅ | ✅ | ✅ | ✅ |

AiraMoe集成了所有先进技术，提供最全面的参数高效微调解决方案。 