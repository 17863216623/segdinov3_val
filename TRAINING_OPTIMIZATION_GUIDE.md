# 训练优化指南

## 概述

本次优化主要改进了训练流程，将测试集改为验证集，添加了早停机制、基于验证指标的学习率调度，并支持保存每个epoch的权重。最后通过测试脚本对所有checkpoint进行评估。

## 主要改进

### 1. 验证集使用
- **之前**：使用`test` split作为验证集
- **现在**：优先使用`val` split，如果没有则使用`test` split
- **好处**：真正的验证集不会泄露到训练中，最终测试结果更可靠

### 2. 早停机制
- 添加了`--early_stop`参数，当验证指标连续N个epoch没有提升时自动停止训练
- 避免过拟合，节省训练时间

### 3. 学习率调度优化
- 新增`plateau`调度器：基于验证指标自动降低学习率
- 参数：
  - `--scheduler plateau`：使用ReduceLROnPlateau
  - `--patience 15`：等待15个epoch无提升后降低学习率
  - `--lr_factor 0.5`：学习率衰减因子
  - `--min_lr 1e-6`：最小学习率

### 4. 权重保存策略
- **之前**：只保存`latest.pth`和`best_*.pth`
- **现在**：
  - `latest.pth`：最新epoch的权重
  - `best_*.pth`：最佳指标的权重
  - `epoch_XXX.pth`：每个epoch的权重（需添加`--save_every_epoch`）

### 5. 测试脚本
- 新增`test_all_checkpoints.py`：可以批量测试所有保存的checkpoint
- 自动生成CSV报告，显示每个checkpoint在测试集上的表现

## 使用方法

### 训练命令（优化后）

```bash
python train_segdinov3.py \
    --dino_ckpt ./dinov3/dinov2_vitb14_pretrain.pth \
    --data_dir ./segdata \
    --dataset usod10k \
    --epochs 80 \
    --batch_size 8 \
    --lr 1e-4 \
    --scheduler plateau \
    --patience 15 \
    --lr_factor 0.5 \
    --early_stop 20 \
    --save_every_epoch \
    --use_wavelet \
    --use_ugbd \
    --use_sdf \
    --best_metric Fw_beta
```

### 关键参数说明

#### 学习率调度
- `--scheduler cosine`：余弦退火（默认）
- `--scheduler plateau`：基于验证指标的自适应调度（推荐）
- `--scheduler constant`：固定学习率

#### 早停
- `--early_stop 20`：连续20个epoch无提升则停止（0表示禁用）

#### 权重保存
- `--save_every_epoch`：保存每个epoch的权重（会占用更多磁盘空间）

#### 验证集配置
- 自动检测：优先使用`val` split，如果没有则使用`test` split
- 确保数据目录结构：
  ```
  segdata/
    usod10k/
      train/
        image/
        mask/
      val/    # 或 test/
        image/
        mask/
  ```

### 测试所有Checkpoint

训练完成后，使用测试脚本评估所有保存的checkpoint：

```bash
python test_all_checkpoints.py \
    --dino_ckpt ./dinov3/dinov2_vitb14_pretrain.pth \
    --ckpt_dir ./runs_usod10k_wavelet/segdino_s_384x384_usod10k_mamba_wav_ugbd/ckpts \
    --data_dir ./segdata \
    --dataset usod10k \
    --test_split test \
    --output_csv ./test_results_all_ckpts.csv \
    --use_ugbd \
    --use_sdf \
    --use_wavelet
```

### 只测试最佳Checkpoint

```bash
python test_all_checkpoints.py \
    --dino_ckpt ./dinov3/dinov2_vitb14_pretrain.pth \
    --ckpt_dir ./runs_usod10k_wavelet/segdino_s_384x384_usod10k_mamba_wav_ugbd/ckpts \
    --data_dir ./segdata \
    --dataset usod10k \
    --test_split test \
    --test_only_best \
    --output_csv ./test_results_best.csv \
    --use_ugbd \
    --use_sdf
```

## 训练流程优化建议

### 1. 初始训练阶段
```bash
# 使用plateau调度器，保存所有epoch
python train_segdinov3.py \
    --scheduler plateau \
    --patience 15 \
    --early_stop 25 \
    --save_every_epoch \
    ...
```

### 2. 微调阶段
如果发现模型在验证集上表现良好但仍有提升空间：
- 降低学习率（`--lr 5e-5`）
- 减小patience（`--patience 10`）
- 继续训练

### 3. 最终测试
- 使用`test_all_checkpoints.py`测试所有checkpoint
- 根据CSV报告选择最佳模型
- 在独立的测试集上验证

## 预期效果

使用验证集后，预期效果：

1. **更可靠的评估**：验证集不会泄露到训练中，评估更准确
2. **更好的泛化**：早停机制防止过拟合
3. **更优的指标**：基于验证指标的学习率调度能更好地优化模型
4. **更全面的分析**：保存所有epoch权重，可以分析训练过程

## 输出文件说明

### 训练输出
- `ckpts/latest.pth`：最新epoch
- `ckpts/best_*.pth`：最佳指标checkpoint
- `ckpts/epoch_XXX.pth`：每个epoch的权重（如果启用）
- `metrics_history.csv`：训练历史记录

### 测试输出
- `test_results_all_ckpts.csv`：所有checkpoint的测试结果
- 包含每个checkpoint的所有评估指标

## 注意事项

1. **磁盘空间**：启用`--save_every_epoch`会占用大量磁盘空间，建议定期清理旧checkpoint
2. **验证集大小**：确保验证集足够大，能代表数据分布
3. **测试集**：测试集应该完全独立，只在最终评估时使用
4. **指标选择**：根据任务选择合适的`--best_metric`（Fw_beta、mIoU等）

## 故障排除

### 问题1：找不到验证集
- 检查数据目录结构
- 确保`val`或`test`目录存在

### 问题2：早停过早触发
- 增加`--early_stop`值
- 检查验证集是否太小

### 问题3：学习率调度不工作
- 确保使用`--scheduler plateau`
- 检查`--best_metric`是否正确

### 问题4：测试脚本报错
- 确保测试脚本的参数与训练时一致（use_ugbd、use_sdf等）
- 检查checkpoint目录路径是否正确
