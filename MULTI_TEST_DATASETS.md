# 多测试集测试说明

## 概述

测试脚本现在支持同时测试多个测试集，并输出每个测试集的结果以及跨测试集的平均结果。

## 支持的测试集

默认配置测试以下三个测试集：
1. **usod10k_test** - USOD10K测试集
2. **usod_test** - USOD测试集
3. **cod10k_test** - COD10K测试集

## 使用方法

### 方式1：使用完整配置脚本

```bash
# 编辑脚本，修改配置
vim test_one_click.sh

# 修改测试集配置（如果需要）
TEST_DATASETS="usod10k_test usod_test cod10k_test"

# 执行测试
./test_one_click.sh
```

### 方式2：使用简化版脚本

```bash
# 编辑脚本，修改基础配置
vim test_one_click_simple.sh

# 执行测试
./test_one_click_simple.sh
```

### 方式3：直接使用Python脚本

```bash
python test_all_checkpoints.py \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --ckpt_dir ./runs/.../ckpts \
    --data_dir ./segdata \
    --dataset usod10k \
    --test_datasets usod10k_test usod_test cod10k_test \
    --output_csv ./test_results_all_datasets.csv \
    --use_ugbd --use_sdf
```

## 输出结果

### CSV文件格式

CSV文件包含以下列：

1. **基础信息**
   - `checkpoint`: checkpoint名称
   - `epoch`: epoch编号
   - `path`: checkpoint路径

2. **每个测试集的指标**（以测试集名称为前缀）
   - `{test_dataset}_val_loss`
   - `{test_dataset}_mIoU`
   - `{test_dataset}_S_alpha`
   - `{test_dataset}_Fw_beta`
   - `{test_dataset}_mE_phi`
   - `{test_dataset}_E_phi_adp`
   - `{test_dataset}_MAE`
   - `{test_dataset}_maxF`

3. **平均指标**（跨所有测试集）
   - `avg_val_loss`
   - `avg_mIoU`
   - `avg_S_alpha`
   - `avg_Fw_beta`
   - `avg_mE_phi`
   - `avg_E_phi_adp`
   - `avg_MAE`
   - `avg_maxF`

### 示例输出

```
checkpoint,epoch,path,usod10k_test_mIoU,usod10k_test_MAE,usod_test_mIoU,usod_test_MAE,cod10k_test_mIoU,cod10k_test_MAE,avg_mIoU,avg_MAE
epoch_045,45,./ckpts/epoch_045.pth,0.8693,0.0175,0.8521,0.0182,0.8645,0.0168,0.8619,0.0175
...
```

### 控制台输出

测试过程中会显示：
1. 每个checkpoint在每个测试集上的结果
2. 每个checkpoint的汇总结果
3. 最终的最佳结果汇总（按测试集和平均指标）

示例：
```
[Evaluating on] usod10k_test
[Result on usod10k_test] mIoU: 0.8693, Fwβ: 0.9211, maxF: 0.9286, MAE: 0.0175

[Evaluating on] usod_test
[Result on usod_test] mIoU: 0.8521, Fwβ: 0.9156, maxF: 0.9234, MAE: 0.0182

[Evaluating on] cod10k_test
[Result on cod10k_test] mIoU: 0.8645, Fwβ: 0.9189, maxF: 0.9256, MAE: 0.0168

[Summary for epoch_045]
  usod10k_test        : mIoU=0.8693, maxF=0.9286, MAE=0.0175
  usod_test           : mIoU=0.8521, maxF=0.9234, MAE=0.0182
  cod10k_test         : mIoU=0.8645, maxF=0.9256, MAE=0.0168
  Average             : mIoU=0.8619, maxF=0.9259, MAE=0.0175
```

## 配置说明

### test_one_click.sh

在脚本顶部配置区域：

```bash
# 多测试集配置
TEST_DATASETS="usod10k_test usod_test cod10k_test"
```

### test_one_click_simple.sh

在脚本顶部配置区域：

```bash
# 多测试集配置
TEST_DATASETS="usod10k_test usod_test cod10k_test"
```

### 自定义测试集

如果需要测试其他测试集，修改 `TEST_DATASETS` 变量：

```bash
# 只测试两个测试集
TEST_DATASETS="usod10k_test cod10k_test"

# 测试其他数据集
TEST_DATASETS="dataset1_test dataset2_test dataset3_test"
```

## 数据目录结构

确保测试集数据目录存在：

```
segdata/
  usod10k_test/
    image/
    mask/
  usod_test/
    image/
    mask/
  cod10k_test/
    image/
    mask/
```

或者：

```
segdata/
  usod10k/
    test/
      image/
      mask/
  usod/
    test/
      image/
      mask/
  cod10k/
    test/
      image/
      mask/
```

## 最佳结果汇总

测试完成后，会显示每个测试集和平均指标的最佳结果：

```
[Best Results Summary]
================================================================================

--- usod10k_test ---
Best mIoU      : epoch_045            @ epoch  45 = 0.8693
Best MAE       : epoch_045            @ epoch  45 = 0.0175
...

--- usod_test ---
Best mIoU      : epoch_050            @ epoch  50 = 0.8521
Best MAE       : epoch_048            @ epoch  48 = 0.0182
...

--- cod10k_test ---
Best mIoU      : epoch_045            @ epoch  45 = 0.8645
Best MAE       : epoch_045            @ epoch  45 = 0.0168
...

--- Average (across all test sets) ---
Best mIoU      : epoch_045            @ epoch  45 = 0.8619
Best MAE       : epoch_045            @ epoch  45 = 0.0175
...
```

## 注意事项

1. **数据路径**: 确保所有测试集的数据目录存在且可访问
2. **参数一致性**: 测试时确保模型参数与训练时一致
3. **内存使用**: 测试多个数据集会占用更多内存和时间
4. **CSV文件大小**: 多个测试集的结果会使CSV文件更大

## 故障排除

### 问题1：找不到测试集
- 检查数据目录结构
- 确认测试集名称正确
- 查看错误信息中的路径提示

### 问题2：某个测试集加载失败
- 脚本会继续测试其他测试集
- 检查失败测试集的数据目录
- 查看错误日志

### 问题3：结果不完整
- 检查CSV文件是否包含所有测试集的列
- 确认所有测试集都成功加载

## 相关文件

- `test_all_checkpoints.py` - 核心测试脚本
- `test_one_click.sh` - 完整配置测试脚本
- `test_one_click_simple.sh` - 简化版测试脚本
