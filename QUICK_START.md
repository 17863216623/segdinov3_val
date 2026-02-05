# 快速开始指南

## 🚀 一键训练和测试

### 训练脚本

**简化版（推荐新手）：**
```bash
# 1. 编辑脚本，修改基础配置
vim train_one_click_simple.sh
# 修改：DINO_CKPT, DATA_DIR, DATASET

# 2. 执行训练
./train_one_click_simple.sh
```

**完整配置版（推荐）：**
```bash
# 1. 编辑脚本，修改所有配置
vim train_one_click.sh

# 2. 执行训练
./train_one_click.sh
```

### 测试脚本

**简化版：**
```bash
# 自动查找checkpoint目录并测试
./test_one_click_simple.sh

# 或手动指定checkpoint目录
./test_one_click_simple.sh --ckpt_dir ./runs/.../ckpts
```

**完整配置版：**
```bash
# 1. 编辑脚本，确保配置与训练时一致
vim test_one_click.sh

# 2. 执行测试
./test_one_click.sh

# 或使用命令行参数
./test_one_click.sh --ckpt_dir ./runs/.../ckpts --test_only_best
```

## 📋 常用命令

### 训练
```bash
python train_segdinov3.py \
    --dino_ckpt ./dinov3/dinov2_vitb14_pretrain.pth \
    --data_dir ./segdata \
    --dataset usod10k \
    --scheduler plateau \
    --early_stop 20 \
    --save_every_epoch \
    --use_wavelet --use_ugbd --use_sdf
```

### 测试所有checkpoint
```bash
python test_all_checkpoints.py \
    --dino_ckpt ./dinov3/dinov2_vitb14_pretrain.pth \
    --ckpt_dir ./runs/.../ckpts \
    --data_dir ./segdata \
    --dataset usod10k \
    --output_csv ./test_results.csv \
    --use_ugbd --use_sdf
```

### 只测试best checkpoint
```bash
python test_all_checkpoints.py \
    --ckpt_dir ./runs/.../ckpts \
    --test_only_best \
    --output_csv ./test_results_best.csv \
    ...
```

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `train_one_click.sh` | 完整配置的训练脚本 |
| `train_one_click_simple.sh` | 简化版训练脚本 |
| `test_one_click.sh` | 完整配置的测试脚本 |
| `test_one_click_simple.sh` | 简化版测试脚本 |
| `train_segdinov3.py` | 训练脚本 |
| `test_all_checkpoints.py` | 测试脚本 |

## ⚙️ 关键参数

### 训练优化参数
- `--scheduler plateau` - 自适应学习率调度
- `--patience 15` - 等待15个epoch
- `--early_stop 20` - 早停阈值
- `--save_every_epoch` - 保存所有epoch

### 模型功能开关
- `--use_wavelet` - 小波损失
- `--use_ugbd` - UGBD细化
- `--use_sdf` - SDF形状先验

### 评估参数
- `--best_metric Fw_beta` - 主要优化指标
- `--dual_sam_eval` - 双SAM评估
- `--eval_tta` - 测试时增强

## 📊 输出文件

训练后会在以下位置生成文件：

```
runs_usod10k_wavelet/
  segdino_s_384x384_usod10k_mamba_wav_ugbd/
    ckpts/
      latest.pth
      best_*.pth
      epoch_*.pth (如果启用)
    metrics_history.csv
    test_results_all_ckpts.csv (测试后)
```

## 🔍 查看结果

```bash
# 查看训练历史
cat runs_*/segdino_*/metrics_history.csv

# 查看测试结果
cat runs_*/segdino_*/test_results_all_ckpts.csv
```

## 💡 提示

1. **首次使用**：先用简化脚本测试
2. **正式训练**：使用完整配置脚本
3. **参数调整**：根据GPU内存调整batch_size
4. **磁盘空间**：启用save_every_epoch会占用大量空间

## 📚 详细文档

- `ONE_CLICK_SCRIPTS.md` - 一键脚本详细说明
- `TRAINING_OPTIMIZATION_GUIDE.md` - 训练优化指南
- `OPTIMIZATION_SUMMARY.md` - 优化总结
