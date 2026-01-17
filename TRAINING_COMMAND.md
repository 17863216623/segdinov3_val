# 训练命令完整指南

## 🎯 目标指标
- **MAE**: 0.0168
- **Fwβ**: 0.9232

## 📊 当前最佳
- **MAE**: 0.0174 (epoch 47)
- **Fwβ**: 0.9200 (epoch 47)

## 🚀 推荐训练命令（激进优化）

### 一键运行
```bash
bash train_aggressive.sh
```

### 完整命令（手动执行）

```bash
python train_segdinov3.py \
  --data_dir ./segdata \
  --dataset usod10k \
  --mask_ext ".png" \
  --repo_dir ./dinov3 \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --dino_size s \
  --input_h 384 \
  --input_w 384 \
  --batch_size 8 \
  --epochs 120 \
  --num_workers 8 \
  --seed 42 \
  --mamba_num_queries 12 \
  --mamba_dim 192 \
  --mamba_layers 3 \
  --mamba_groups 4 \
  --mamba_kernel 9 \
  --mamba_mask_embed_dim 64 \
  --w_bce 1.0 \
  --w_iou 0.7 \
  --w_ssim 0.45 \
  --use_wavelet \
  --wav_levels 3 \
  --w_wav 0.35 \
  --w_wed 0.15 \
  --w_tv 0.10 \
  --w_bfce 0.18 \
  --bf_lam 1.5 \
  --use_ugbd \
  --ugbd_steps 4 \
  --ugbd_mid_ch 64 \
  --w_cldice 0.08 \
  --w_biou 0.25 \
  --w_uwb 0.20 \
  --uwb_lam 2.5 \
  --w_freq 0.10 \
  --freq_hf_boost 0.25 \
  --use_sdf \
  --sdf_mid_ch 64 \
  --sdf_lambda 0.35 \
  --w_sdf 0.15 \
  --w_eik 0.08 \
  --w_curv 0.04 \
  --sdf_scale 0.7 \
  --dual_sam_eval \
  --eval_tta \
  --tta_scales 1.0,0.75,1.25 \
  --lr 6e-5 \
  --weight_decay 1.5e-4 \
  --scheduler cosine \
  --warmup_epochs 12 \
  --use_ema \
  --ema_decay 0.999 \
  --use_aug \
  --grad_clip 0.8 \
  --best_metric Fw_beta
```

## 🔑 关键新增模块

### 1. Boundary IoU Loss (`--w_biou 0.25`)
- **作用**: 直接约束边界区域的 IoU
- **预期**: Fwβ +0.2~0.4%
- **为什么重要**: Fwβ 对边界质量非常敏感

### 2. Uncertainty-Weighted Boundary Loss (`--w_uwb 0.20`)
- **作用**: 在边界区域结合不确定性加权
- **预期**: Fwβ +0.15~0.3%, MAE -2~3%
- **参数**: `--uwb_lam 2.5`

### 3. Fourier Magnitude Loss (`--w_freq 0.10`)
- **作用**: 频域约束高频细节
- **预期**: Fwβ +0.1~0.2%, MAE -1~2%
- **参数**: `--freq_hf_boost 0.25`

## 📈 参数对比（vs 之前训练）

| 参数 | 之前 | 激进优化 | 变化 | 原因 |
|------|------|---------|------|------|
| **新增损失** | | | | |
| w_biou | 0 | **0.25** | 新增 | 边界IoU |
| w_uwb | 0 | **0.20** | 新增 | 不确定性加权边界 |
| w_freq | 0 | **0.10** | 新增 | Fourier幅度 |
| **基础损失** | | | | |
| w_iou | 0.6 | **0.7** | +17% | 增强IoU约束 |
| w_ssim | 0.4 | **0.45** | +12.5% | 增强结构相似性 |
| **Wavelet** | | | | |
| w_wav | 0.32 | **0.35** | +9% | 增强细节 |
| w_wed | 0.12 | **0.15** | +25% | 增强边缘 |
| w_tv | 0.08 | **0.10** | +25% | 增强平滑 |
| w_bfce | 0.15 | **0.18** | +20% | 增强边界焦点 |
| bf_lam | 1.2 | **1.5** | +25% | 边界放大 |
| **SDF** | | | | |
| sdf_lambda | 0.4 | **0.35** | -12.5% | 减少干扰 |
| w_sdf | 0.12 | **0.15** | +25% | 增强拟合 |
| w_eik | 0.06 | **0.08** | +33% | 增强Eikonal |
| w_curv | 0.03 | **0.04** | +33% | 增强曲率 |
| sdf_scale | 0.8 | **0.7** | -12.5% | 更温和 |
| **训练策略** | | | | |
| epochs | 100 | **120** | +20% | 更充分训练 |
| lr | 8e-5 | **6e-5** | -25% | 更稳定 |
| warmup_epochs | 10 | **12** | +20% | 更稳定初始化 |
| ema_decay | 0.998 | **0.999** | - | 更慢更新 |
| grad_clip | 1.0 | **0.8** | -20% | 更严格 |
| weight_decay | 1.2e-4 | **1.5e-4** | +25% | 更强正则化 |

## ⚡ 快速对比

### 之前训练（100 epochs）
```bash
bash train_optimized.sh
```
- 结果: MAE=0.0174, Fwβ=0.9200

### 激进优化（120 epochs）
```bash
bash train_aggressive.sh
```
- 预期: MAE=0.0168, Fwβ=0.9232 ✅

## 🎛️ 微调建议

### 如果 Fwβ 还不够高
```bash
# 增加边界相关损失
--w_biou 0.30
--w_uwb 0.25
--w_bfce 0.20
--bf_lam 2.0
```

### 如果 MAE 还不够低
```bash
# 增加平滑性损失
--w_tv 0.12
--w_curv 0.05
--w_ssim 0.50
```

### 如果训练不稳定
```bash
# 降低学习率，增加梯度裁剪
--lr 5e-5
--grad_clip 0.5
--warmup_epochs 15
```

## 📝 训练监控

### 关键指标
- **Fwβ**: 目标 0.9232，当前最佳 0.9200
- **MAE**: 目标 0.0168，当前最佳 0.0174

### 训练稳定性检查
- 训练 loss 应稳定下降
- 验证指标应在 epoch 50-80 达到最佳
- 如果后期下降，考虑提前停止

## 🔄 回退方案

如果激进优化效果不佳：

1. **部分回退**: 只添加 Boundary IoU (`--w_biou 0.20`)
2. **保守回退**: 使用 `train_optimized.sh` 的参数
3. **渐进式**: 一次只添加一个损失，观察效果

---

**最后更新**: 基于 epoch 47 最佳结果分析


