# 一键训练和测试脚本使用指南

## 概述

提供了三个一键训练和测试脚本，可以自动完成训练和测试流程：

1. **train_and_test.sh** - 完整配置版本（推荐）
2. **train_and_test_simple.sh** - 简化版本（快速使用）
3. **train_and_test.py** - Python版本（最灵活）

## 快速开始

### 方式1：简化版脚本（最简单）

```bash
# 1. 编辑脚本，修改基础配置
vim train_and_test_simple.sh

# 2. 执行
./train_and_test_simple.sh
```

### 方式2：完整配置脚本（推荐）

```bash
# 1. 编辑脚本，修改所有配置
vim train_and_test.sh

# 2. 执行
./train_and_test.sh
```

### 方式3：Python版本（最灵活）

```bash
# 使用默认配置
python train_and_test.py

# 自定义配置
python train_and_test.py \
    --epochs 100 \
    --batch_size 16 \
    --use_wavelet \
    --use_ugbd \
    --use_sdf \
    --save_every_epoch
```

## 脚本说明

### 1. train_and_test.sh（完整配置版）

**特点：**
- 所有参数都在脚本顶部配置区域
- 自动构建训练和测试命令
- 自动检测checkpoint目录
- 完整的错误处理

**使用步骤：**
1. 打开脚本，修改配置区域的所有参数
2. 保存并执行：`./train_and_test.sh`

**配置区域示例：**
```bash
# ==================== 配置区域 ====================
DINO_CKPT="./dinov3/dinov2_vitb14_pretrain.pth"
DATA_DIR="./segdata"
DATASET="usod10k"
EPOCHS=80
BATCH_SIZE=8
# ... 更多配置
# ==================== 配置结束 ====================
```

### 2. train_and_test_simple.sh（简化版）

**特点：**
- 使用默认配置
- 自动检测checkpoint目录
- 适合快速测试

**使用步骤：**
1. 修改脚本中的基础配置（DINO_CKPT, DATA_DIR, DATASET）
2. 执行：`./train_and_test_simple.sh`

### 3. train_and_test.py（Python版本）

**特点：**
- 命令行参数支持
- 更灵活的错误处理
- 可以跳过训练或测试阶段
- 自动查找checkpoint目录

**常用命令：**

```bash
# 完整流程（训练+测试）
python train_and_test.py \
    --dino_ckpt ./dinov3/dinov2_vitb14_pretrain.pth \
    --data_dir ./segdata \
    --dataset usod10k \
    --epochs 80 \
    --use_wavelet --use_ugbd --use_sdf \
    --save_every_epoch

# 只训练，不测试
python train_and_test.py --skip_test

# 只测试，不训练（需要指定checkpoint目录）
python train_and_test.py --skip_train --ckpt_dir ./runs/.../ckpts

# 只测试best checkpoint
python train_and_test.py --skip_train --test_only_best --ckpt_dir ./runs/.../ckpts
```

## 参数说明

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 80 |
| `--batch_size` | 批次大小 | 8 |
| `--lr` | 学习率 | 1e-4 |
| `--scheduler` | 学习率调度器 | plateau |
| `--patience` | Plateau调度器等待轮数 | 15 |
| `--early_stop` | 早停轮数 | 20 |
| `--save_every_epoch` | 保存每个epoch权重 | False |
| `--best_metric` | 最佳指标 | Fw_beta |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dino_size` | DINO模型大小 | s |
| `--use_wavelet` | 使用小波损失 | False |
| `--use_ugbd` | 使用UGBD细化 | False |
| `--use_sdf` | 使用SDF头 | False |

### 测试参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--test_split` | 测试集split名称 | test |
| `--test_only_best` | 只测试best checkpoint | False |
| `--skip_train` | 跳过训练 | False |
| `--skip_test` | 跳过测试 | False |
| `--ckpt_dir` | 手动指定checkpoint目录 | None |

## 输出文件

### 训练输出
- `ckpts/latest.pth` - 最新epoch
- `ckpts/best_*.pth` - 最佳指标checkpoint
- `ckpts/epoch_XXX.pth` - 每个epoch权重（如果启用）
- `metrics_history.csv` - 训练历史

### 测试输出
- `test_results_all_ckpts.csv` - 所有checkpoint测试结果
- `test_results_best_only.csv` - 只测试best时的结果

## 使用示例

### 示例1：完整训练和测试流程

```bash
# 使用完整配置脚本
./train_and_test.sh
```

### 示例2：快速测试（使用默认配置）

```bash
# 使用简化脚本
./train_and_test_simple.sh
```

### 示例3：自定义训练和测试

```bash
# 使用Python脚本
python train_and_test.py \
    --epochs 100 \
    --batch_size 16 \
    --scheduler plateau \
    --patience 20 \
    --early_stop 25 \
    --use_wavelet \
    --use_ugbd \
    --use_sdf \
    --save_every_epoch \
    --best_metric Fw_beta
```

### 示例4：只测试已有checkpoint

```bash
# 指定checkpoint目录
python train_and_test.py \
    --skip_train \
    --ckpt_dir ./runs_usod10k_wavelet/segdino_s_384x384_usod10k_mamba_wav_ugbd/ckpts \
    --test_only_best
```

### 示例5：继续训练并测试

```bash
# 先训练（可以修改epochs等参数）
python train_and_test.py --skip_test --epochs 100

# 然后测试
python train_and_test.py --skip_train
```

## 注意事项

1. **路径配置**：确保DINO checkpoint路径和数据目录路径正确
2. **磁盘空间**：启用`--save_every_epoch`会占用大量空间
3. **GPU内存**：根据GPU内存调整batch_size
4. **参数一致性**：测试时确保模型参数与训练时一致

## 故障排除

### 问题1：找不到checkpoint目录
- 检查`--save_root`路径是否正确
- 手动指定`--ckpt_dir`

### 问题2：训练失败
- 检查DINO checkpoint路径
- 检查数据目录结构
- 查看错误日志

### 问题3：测试失败
- 确保checkpoint目录存在
- 确保测试参数与训练时一致
- 检查测试集路径

### 问题4：内存不足
- 减小batch_size
- 减小input_h和input_w
- 关闭一些功能（如TTA）

## 推荐工作流程

1. **首次训练**：使用简化脚本快速测试
   ```bash
   ./train_and_test_simple.sh
   ```

2. **正式训练**：使用完整配置脚本
   ```bash
   # 编辑配置
   vim train_and_test.sh
   # 执行
   ./train_and_test.sh
   ```

3. **分析结果**：查看测试结果CSV
   ```bash
   cat ./runs_*/segdino_*/test_results_all_ckpts.csv
   ```

4. **选择最佳模型**：根据CSV结果选择最佳checkpoint

## 高级用法

### 批量实验

```bash
# 测试不同配置
for lr in 1e-4 5e-5 1e-5; do
    python train_and_test.py --lr $lr --skip_test
done

# 然后统一测试
python train_and_test.py --skip_train
```

### 只测试特定checkpoint

```bash
# 手动指定checkpoint目录
python train_and_test.py \
    --skip_train \
    --ckpt_dir /path/to/specific/ckpt/dir
```

## 相关文件

- `train_segdinov3.py` - 训练脚本
- `test_all_checkpoints.py` - 测试脚本
- `TRAINING_OPTIMIZATION_GUIDE.md` - 详细训练指南
- `OPTIMIZATION_SUMMARY.md` - 优化总结
