# AIRA_MoE Training Scripts

This directory contains training scripts for AIRA_MoE (Activation-aware Improved Rank Allocation with Mixture of Experts) across different domains and configurations.

## Script Overview

scripts/aira_moe/
├── README.md                    # 详细的使用文档
├── generality_basic.sh          # 基础版本（仅CoLA）
├── generality_standard.sh       # 标准版本（CoLA + 激活感知）
├── generality_full.sh           # 完整版本（所有AwLoRA技术）
├── finance.sh                   # 金融领域
├── math.sh                      # 数学领域
├── medicine.sh                  # 医学领域
├── law.sh                       # 法律领域
└── multi-tasking.sh             # 多任务领域

### Basic Configuration Scripts
- `generality_basic.sh` - Basic AIRA_MoE with only CoLA collaborative adaptation
- `generality_standard.sh` - Standard AIRA_MoE with CoLA + activation-aware weighting
- `generality_full.sh` - Full AIRA_MoE with all AwLoRA technologies

### Domain-Specific Scripts
- `finance.sh` - Financial domain training
- `math.sh` - Mathematical reasoning training
- `medicine.sh` - Medical knowledge training
- `law.sh` - Legal reasoning training
- `multi-tasking.sh` - Multi-domain training

## AIRA_MoE Technology Levels

### 1. Basic Mode (generality_basic.sh)
- **Features**: CoLA collaborative adaptation only
- **Parameters**: `num_A=2, num_B=3`
- **Use Case**: Baseline comparison, minimal overhead
- **Technologies Disabled**: Layer-wise rank, AwSVD, activation-aware

### 2. Standard Mode (generality_standard.sh)
- **Features**: CoLA + Activation-aware weighting
- **Parameters**: `num_A=2, num_B=3, use_activation_aware=true`
- **Use Case**: Balanced performance and efficiency
- **Technologies**: CoLA + AwLoRA Tech 3

### 3. Full Mode (All other scripts)
- **Features**: All AwLoRA technologies enabled
- **Parameters**: Enhanced rank allocation, AwSVD initialization, activation-aware weighting
- **Use Case**: Maximum performance, research experiments
- **Technologies**: CoLA + AwLoRA Tech 1,2,3

## Key Parameters Explained

### CoLA Parameters
- `num_A`: Number of A matrices for collaborative adaptation (default: 2-4)
- `num_B`: Number of B matrices for collaborative adaptation (default: 3-4)

### AwLoRA Technology 1: Layer-wise Rank Allocation
- `use_layer_wise_rank`: Enable dynamic rank allocation (true/false)
- `rank_budget`: Total rank budget for allocation (128-256)
- `theta_type`: Allocation strategy ("lod" for LOD-based)
- `objective_function`: Optimization objective ("log" recommended)
- `min_rank`: Minimum rank per layer (2-4)
- `max_rank`: Maximum rank per layer (32-64)
- `lod_threshold_M`: LOD threshold multiplier (1.8-2.5)

### AwLoRA Technology 2: AwSVD Initialization
- `use_awsvd_init`: Enable AwSVD initialization (true/false)
- `awsvd_collect_steps`: Steps for activation collection (100-200)

### AwLoRA Technology 3: Activation-aware Weighting
- `use_activation_aware`: Enable activation-aware weighting (true/false)
- `activation_aware_mode`: Weighting mode ("inps"/"outps")
- `activation_normalize`: Normalize activations (true recommended)

## Domain-Specific Optimizations

### Finance Domain
- **Dataset**: fineval-en
- **Optimization**: Output-based activation weighting (`activation_aware_mode=outps`)
- **Focus**: Financial reasoning and analysis

### Math Domain
- **Dataset**: gsm8k
- **Optimization**: Higher LOD threshold (2.5), extended AwSVD collection (150 steps)
- **Focus**: Mathematical reasoning and problem-solving

### Medicine Domain
- **Dataset**: iCliniq, GenMedGPT-5k
- **Optimization**: Output-based weighting, moderate AwSVD collection (120 steps)
- **Focus**: Medical knowledge and diagnosis

### Law Domain
- **Dataset**: lawbench
- **Optimization**: Input-based activation weighting
- **Focus**: Legal reasoning and analysis

### Multi-tasking Domain
- **Dataset**: Multiple domains combined
- **Optimization**: Higher rank budget (256), extended parameters
- **Focus**: Cross-domain knowledge integration

## Usage Instructions

### 1. Make Scripts Executable
```bash
chmod +x scripts/aira_moe/*.sh
```

### 2. Run Training
```bash
# Basic mode
./scripts/aira_moe/generality_basic.sh

# Standard mode
./scripts/aira_moe/generality_standard.sh

# Full mode
./scripts/aira_moe/generality_full.sh

# Domain-specific
./scripts/aira_moe/finance.sh
./scripts/aira_moe/math.sh
./scripts/aira_moe/medicine.sh
./scripts/aira_moe/law.sh
./scripts/aira_moe/multi-tasking.sh
```

### 3. Run Evaluation
Uncomment the evaluation commands in each script and run them after training.

## Hardware Requirements

- **Basic/Standard Mode**: 2 GPUs (8GB+ VRAM each)
- **Full Mode**: 2-4 GPUs (8GB+ VRAM each)
- **Multi-tasking**: 4 GPUs recommended

## Performance Expectations

### Training Time (approximate)
- **Basic Mode**: ~2-3 hours per epoch
- **Standard Mode**: ~2.5-3.5 hours per epoch
- **Full Mode**: ~3-4 hours per epoch

### Memory Usage
- **Basic Mode**: ~6-7GB per GPU
- **Standard Mode**: ~7-8GB per GPU
- **Full Mode**: ~8-10GB per GPU

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`
2. **Slow Training**: Disable debug exports in medicine.sh for production runs
3. **Dataset Not Found**: Ensure datasets are properly configured in LLaMA-Factory

### Parameter Tuning Tips
1. **For Limited Resources**: Use basic mode or reduce `rank_budget`
2. **For Better Performance**: Increase `num_A`, `num_B`, and `rank_budget`
3. **For Stability**: Keep `lod_threshold_M` between 1.8-2.5

## Comparison with CoLA

AIRA_MoE extends CoLA with three additional technologies:
1. **Layer-wise Rank Allocation**: Dynamic rank distribution based on layer importance
2. **AwSVD Initialization**: Activation-aware SVD initialization for better convergence
3. **Activation-aware Weighting**: Dynamic weighting based on activation patterns

This results in improved performance while maintaining parameter efficiency. 