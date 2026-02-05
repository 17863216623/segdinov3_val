# MAE优化方案

## 目标
将测试结果的MAE从当前的0.0175降低到**< 0.016**

## 优化策略

### 1. 添加MAE直接优化损失

#### 新增损失函数
- **MAE损失** (`mae_loss`): 直接优化平均绝对误差
  - 权重: `w_mae = 0.8`
  - 作用: 直接最小化预测与真实值的绝对误差

- **Focal MAE损失** (`focal_mae_loss`): 对困难样本给予更多关注
  - 权重: `w_focal_mae = 0.3`
  - 参数: `alpha=0.25`, `gamma=2.0`
  - 作用: 对预测误差较大的区域给予更高权重

### 2. 优化损失权重配置

#### 增加的权重
- **SSIM损失**: `0.35 → 0.40` (提升细节质量)
- **小波损失**: `0.30 → 0.35` (提升细节保留)
- **小波边缘Dice**: `0.10 → 0.12` (提升边界精度)
- **TV损失**: `0.06 → 0.08` (提升预测平滑度)
- **边界Focal BCE**: `0.12 → 0.15` (提升边界精度)

#### SDF损失权重
- **SDF损失**: `0.10 → 0.15` (提升形状先验)
- **Eikonal损失**: `0.05 → 0.08` (提升SDF质量)
- **Curvature损失**: `0.02 → 0.03` (提升平滑度)

### 3. 训练策略优化

#### 训练轮数
- **EPOCHS**: `80 → 100` (增加训练轮数以更好优化MAE)

#### 学习率调度
- **PATIENCE**: `15 → 20` (给MAE更多优化时间)
- **EARLY_STOP**: `20 → 30` (允许更长时间优化)

#### 最佳指标
- **BEST_METRIC**: `Fw_beta → MAE` (直接以MAE为优化目标)

## 使用方法

### 使用优化后的配置训练

```bash
./train_one_click.sh
```

### 关键优化参数

训练脚本中已包含以下优化：

```bash
# MAE优化损失
W_MAE=0.8
W_FOCAL_MAE=0.3

# 训练配置
EPOCHS=100
PATIENCE=20
EARLY_STOP=30
BEST_METRIC="MAE"
```

## 预期效果

### 损失函数组合
1. **基础损失**: BCE + IoU/Dice + SSIM
2. **小波损失**: 细节损失 + 边缘Dice + TV + 边界Focal BCE
3. **MAE损失**: 直接MAE + Focal MAE (新增)
4. **SDF损失**: SDF L1 + Eikonal + Curvature
5. **边界损失**: Boundary IoU + UWB + Fourier (可选)

### 优化机制
- **直接优化**: MAE损失直接最小化平均绝对误差
- **困难样本关注**: Focal MAE对误差大的区域给予更高权重
- **细节保留**: 增加小波和SSIM权重
- **边界精度**: 增加边界相关损失权重
- **形状先验**: 增加SDF损失权重

## 验证方法

训练完成后，使用测试脚本验证：

```bash
./test_one_click.sh
```

查看测试结果中的MAE指标，应该达到 **< 0.016** 的要求。

## 进一步优化建议

如果MAE仍未达到要求，可以尝试：

1. **增加MAE损失权重**
   ```bash
   W_MAE=1.2
   W_FOCAL_MAE=0.5
   ```

2. **增加训练轮数**
   ```bash
   EPOCHS=120
   ```

3. **调整学习率**
   ```bash
   LR=8e-5  # 稍微降低学习率，更精细优化
   ```

4. **增加数据增强**
   - 在训练脚本中启用 `--use_aug`

5. **使用更精细的模型**
   - 考虑使用DINOv3-Base (`dino_size=b`)

## 注意事项

1. **训练时间**: 增加训练轮数会延长训练时间
2. **过拟合风险**: 注意监控验证集指标，避免过拟合
3. **损失平衡**: 如果其他指标下降，可以适当调整权重

## 技术细节

### MAE损失函数

```python
def mae_loss(prob, target):
    """直接优化平均绝对误差"""
    return torch.abs(prob - target).mean()
```

### Focal MAE损失函数

```python
def focal_mae_loss(prob, target, alpha=0.25, gamma=2.0):
    """对困难样本给予更多关注"""
    mae = torch.abs(prob - target)
    weight = torch.pow(mae, gamma)
    return (alpha * weight * mae).mean()
```

## 总结

通过添加MAE直接优化损失、调整损失权重、增加训练轮数和优化训练策略，预期可以将MAE从0.0175降低到< 0.016，满足要求。
