#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoLA (Collaborative Low-rank Adaptation) 项目结构说明
=================================================

本文档详细介绍了CoLA项目的代码结构和ColaModel类的初始化流程。

项目结构概览
-----------
peft/src/peft/tuners/cola/
├── __init__.py          # 模块导出定义
├── config.py           # ColaConfig配置类
├── layer.py            # ColaLayer核心层实现
├── model.py            # ColaModel主模型类
└── readme.py           # 本说明文档

核心组件说明
-----------

1. ColaConfig (config.py)
   - 继承自PeftConfig基类
   - 定义CoLA适配器的所有配置参数
   - 主要参数包括：
     * r: LoRA的秩(rank)
     * lora_alpha: LoRA的缩放参数
     * target_modules: 目标模块列表
     * lora_dropout: dropout概率
     * num_A, num_B: 专家数量配置
     * fan_in_fan_out: 权重存储格式
     * init_lora_weights: 权重初始化方式

2. ColaLayer (layer.py)
   - 继承自BaseTunerLayer
   - 实现CoLA适配器的核心逻辑
   - 管理多个适配器的LoRA权重矩阵
   - 支持动态切换和合并适配器

3. ColaModel (model.py)
   - 继承自BaseTuner
   - 负责将CoLA适配器注入到预训练模型中
   - 管理适配器的生命周期

ColaModel初始化流程详解
====================

当调用 ColaModel(model, config, adapter_name) 时，会执行以下步骤：

第一阶段：基础初始化
-----------------
1. 调用父类BaseTuner的__init__方法
2. 设置基本属性：
   - self.model = model (原始模型)
   - self.peft_config = {adapter_name: config} (配置字典)
   - self.active_adapter = adapter_name (当前活跃适配器)
   - self.targeted_module_names = [] (目标模块名称列表)

第二阶段：适配器注入 (inject_adapter)
---------------------------------
这是最关键的步骤，包含以下子步骤：

2.1 配置验证和准备
   - 调用_check_new_adapter_config()验证配置
   - 获取模型配置信息
   - 调用_prepare_adapter_config()准备适配器配置
   - 调用_prepare_model()预处理模型结构

2.2 目标模块识别和替换
   - 遍历模型的所有命名模块
   - 对每个模块调用_check_target_module_exists()检查是否为目标模块
   - 如果是目标模块，调用_create_and_replace()进行替换

2.3 模块创建和替换过程 (_create_and_replace)
   - 解析配置参数（rank, alpha等）
   - 判断目标模块类型：
     * 如果已经是ColaLayer：调用update_layer()更新
     * 如果是普通层：调用_create_new_module()创建新的CoLA层
   - 调用_replace_module()替换原始模块

2.4 新模块创建 (_create_new_module)
   - 根据目标层类型选择对应的CoLA实现：
     * nn.Linear -> Linear (CoLA版本)
     * nn.Conv2d -> Conv2d (CoLA版本)  
     * nn.Embedding -> Embedding (CoLA版本)
   - 创建包装了原始层的CoLA层

第三阶段：后处理
--------------
3.1 设置活跃适配器
   - 调用set_adapter()设置当前活跃的适配器

3.2 标记可训练参数
   - 调用_mark_only_adapters_as_trainable()
   - 冻结原始模型参数，只保持适配器参数可训练

3.3 推理模式处理
   - 如果配置为推理模式，冻结所有适配器参数

CoLA层的核心特性
===============

1. 多专家架构
   - 每个适配器包含多个专家(experts)
   - num_A和num_B分别控制A矩阵和B矩阵的专家数量
   - 支持动态专家选择和组合

2. 权重管理
   - lora_A: 存储所有适配器的A矩阵专家
   - lora_B: 存储所有适配器的B矩阵专家
   - scaling: 存储缩放因子
   - 支持适配器的动态加载和卸载

3. 前向传播
   - 基础层输出 + LoRA增量输出
   - 支持多适配器并行推理
   - 动态专家选择机制

使用示例
=======

```python
from transformers import AutoModelForCausalLM
from peft import ColaConfig, ColaModel

# 1. 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. 创建CoLA配置
config = ColaConfig(
    r=8,                    # LoRA rank
    lora_alpha=32,          # LoRA alpha
    target_modules=["c_attn", "c_proj"],  # 目标模块
    lora_dropout=0.1,       # dropout
    num_A=4,               # A矩阵专家数量
    num_B=4,               # B矩阵专家数量
)

# 3. 创建CoLA模型 (此时执行上述初始化流程)
cola_model = ColaModel(model, config, "default")

# 4. 模型现在已经注入了CoLA适配器，可以进行训练或推理
```

关键设计理念
===========

1. 模块化设计：配置、层、模型分离，便于扩展和维护
2. 动态适配：支持运行时添加、删除、切换适配器
3. 多专家机制：提供更强的表达能力和灵活性
4. 兼容性：与现有PEFT框架无缝集成
5. 效率优化：最小化内存占用和计算开销

这种设计使得CoLA能够在保持模型性能的同时，提供灵活的参数高效微调能力。
"""

if __name__ == "__main__":
    print("CoLA项目结构说明文档")
    print("=" * 50)
    print("请查看本文件的文档字符串获取详细信息") 