# 一键训练和测试脚本使用指南

## 📋 脚本列表

### 训练脚本
1. **train_one_click.sh** - 完整配置的训练脚本（推荐）
2. **train_one_click_simple.sh** - 简化版训练脚本（快速使用）

### 测试脚本
1. **test_one_click.sh** - 完整配置的测试脚本（推荐）
2. **test_one_click_simple.sh** - 简化版测试脚本（快速使用）

## 🚀 快速开始

### 方式1：简化版（最简单）

**训练：**
```bash
# 1. 编辑脚本，修改基础配置
vim train_one_click_simple.sh
# 修改：DINO_CKPT, DATA_DIR, DATASET

# 2. 执行训练
./train_one_click_simple.sh
```

**测试：**
```bash
# 自动查找checkpoint目录并测试
./test_one_click_simple.sh

# 或手动指定checkpoint目录
./test_one_click_simple.sh --ckpt_dir ./runs/.../ckpts
```

### 方式2：完整配置版（推荐）

**训练：**
```bash
# 1. 编辑脚本，修改所有配置
vim train_one_click.sh

# 2. 执行训练
./train_one_click.sh
```

**测试：**
```bash
# 1. 编辑脚本，修改配置（确保与训练时一致）
vim test_one_click.sh

# 2. 执行测试
./test_one_click.sh

# 或使用命令行参数
./test_one_click.sh --ckpt_dir ./runs/.../ckpts --test_only_best
```

## 📖 详细说明

### train_one_click.sh（完整配置版）

**特点：**
- 所有参数都在脚本顶部配置区域
- 完整的训练参数配置
- 自动确定checkpoint保存位置

**配置区域：**
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

**使用步骤：**
1. 打开脚本，修改配置区域的所有参数
2. 保存并执行：`./train_one_click.sh`

### train_one_click_simple.sh（简化版）

**特点：**
- 使用默认配置
- 只需修改基础配置即可
- 适合快速测试

**使用步骤：**
1. 修改脚本中的基础配置（DINO_CKPT, DATA_DIR, DATASET）
2. 执行：`./train_one_click_simple.sh`

### test_one_click.sh（完整配置版）

**特点：**
- 所有参数都在脚本顶部配置区域
- 支持命令行参数
- 自动查找checkpoint目录
- 确保测试参数与训练时一致

**配置区域：**
```bash
# ==================== 配置区域 ====================
DINO_CKPT="./dinov3/dinov2_vitb14_pretrain.pth"
DATA_DIR="./segdata"
DATASET="usod10k"
USE_UGBD=true
USE_SDF=true
# ... 更多配置（必须与训练时一致）
# ==================== 配置结束 ====================
```

**使用方法：**
```bash
# 使用脚本中的配置
./test_one_click.sh

# 使用命令行参数
./test_one_click.sh --ckpt_dir ./runs/.../ckpts
./test_one_click.sh --test_only_best
./test_one_click.sh --ckpt_dir ./runs/.../ckpts --test_only_best --output_csv ./my_results.csv
```

**命令行参数：**
- `--ckpt_dir DIR` - 指定checkpoint目录
- `--test_only_best` - 只测试best checkpoint
- `--output_csv FILE` - 指定输出CSV文件
- `--dataset NAME` - 指定数据集名称

### test_one_click_simple.sh（简化版）

**特点：**
- 使用默认配置
- 自动查找checkpoint目录
- 适合快速测试

**使用方法：**
```bash
# 自动查找checkpoint目录
./test_one_click_simple.sh

# 手动指定checkpoint目录（需要修改脚本）
vim test_one_click_simple.sh
# 修改 CKPT_DIR 变量
```

## ⚙️ 配置说明

### 训练配置要点

1. **基础配置**
   - `DINO_CKPT` - DINO checkpoint路径
   - `DATA_DIR` - 数据目录
   - `DATASET` - 数据集名称

2. **训练参数**
   - `EPOCHS` - 训练轮数
   - `BATCH_SIZE` - 批次大小
   - `LR` - 学习率
   - `SCHEDULER` - 学习率调度器（plateau/cosine/constant）

3. **功能开关**
   - `USE_WAVELET` - 使用小波损失
   - `USE_UGBD` - 使用UGBD细化
   - `USE_SDF` - 使用SDF头
   - `SAVE_EVERY_EPOCH` - 保存每个epoch权重

### 测试配置要点

**重要：测试配置必须与训练时一致！**

1. **模型配置**
   - `DINO_SIZE` - 必须与训练时一致
   - `USE_UGBD` - 必须与训练时一致
   - `USE_SDF` - 必须与训练时一致

2. **评估配置**
   - `DUAL_SAM_EVAL` - 双SAM评估
   - `EVAL_TTA` - 测试时增强

## 📊 使用示例

### 示例1：完整训练流程

```bash
# 1. 训练
./train_one_click.sh

# 2. 训练完成后，测试所有checkpoint
./test_one_click.sh

# 3. 或只测试best checkpoint
./test_one_click.sh --test_only_best
```

### 示例2：快速测试

```bash
# 1. 快速训练
./train_one_click_simple.sh

# 2. 快速测试
./test_one_click_simple.sh
```

### 示例3：指定checkpoint目录测试

```bash
# 测试特定目录的checkpoint
./test_one_click.sh --ckpt_dir ./runs_usod10k_wavelet/segdino_s_384x384_usod10k_mamba_wav_ugbd/ckpts
```

### 示例4：只测试best checkpoint

```bash
./test_one_click.sh --ckpt_dir ./runs/.../ckpts --test_only_best
```

## 📁 输出文件

### 训练输出
训练完成后，checkpoint保存在：
```
runs_usod10k_wavelet/
  segdino_s_384x384_usod10k_mamba_wav_ugbd/
    ckpts/
      latest.pth
      best_*.pth
      epoch_*.pth (如果启用save_every_epoch)
    metrics_history.csv
```

### 测试输出
测试完成后，结果保存在：
```
runs_usod10k_wavelet/
  segdino_s_384x384_usod10k_mamba_wav_ugbd/
    test_results_all_ckpts.csv
    test_results_best_only.csv (如果使用--test_only_best)
```

## 🔍 查看结果

```bash
# 查看训练历史
cat runs_*/segdino_*/metrics_history.csv

# 查看测试结果
cat runs_*/segdino_*/test_results_all_ckpts.csv
```

## ⚠️ 注意事项

1. **参数一致性**：测试时确保模型参数与训练时完全一致
2. **路径配置**：确保DINO checkpoint和数据目录路径正确
3. **磁盘空间**：启用save_every_epoch会占用大量空间
4. **GPU内存**：根据GPU内存调整batch_size

## 🐛 故障排除

### 问题1：找不到checkpoint目录
```bash
# 手动指定目录
./test_one_click.sh --ckpt_dir /path/to/ckpts
```

### 问题2：测试参数不匹配
- 检查测试脚本中的配置是否与训练时一致
- 确保USE_UGBD、USE_SDF等开关一致

### 问题3：训练失败
- 检查DINO checkpoint路径
- 检查数据目录结构
- 查看错误日志

## 📚 相关文档

- `TRAINING_OPTIMIZATION_GUIDE.md` - 训练优化指南
- `OPTIMIZATION_SUMMARY.md` - 优化总结
- `QUICK_START.md` - 快速开始指南
