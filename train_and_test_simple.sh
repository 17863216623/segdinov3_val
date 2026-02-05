#!/bin/bash
# 简化版一键训练和测试脚本
# 使用默认配置，适合快速测试

set -e

# 基础配置（根据实际情况修改）
DINO_CKPT="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_DIR="./segdata"
DATASET="usod10k"

echo "=========================================="
echo "一键训练和测试（简化版）"
echo "=========================================="

# 训练
echo "开始训练..."
python train_segdinov3.py \
    --dino_ckpt ${DINO_CKPT} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --epochs 80 \
    --batch_size 8 \
    --scheduler plateau \
    --patience 15 \
    --early_stop 20 \
    --save_every_epoch \
    --use_wavelet \
    --use_ugbd \
    --use_sdf \
    --best_metric Fw_beta \
    --dual_sam_eval \
    --eval_tta

# 自动检测checkpoint目录
SAVE_ROOT="./runs_usod10k_wavelet"
CKPT_DIR=$(find ${SAVE_ROOT} -type d -name "ckpts" | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "错误: 找不到checkpoint目录"
    exit 1
fi

echo "找到checkpoint目录: ${CKPT_DIR}"

# 测试
echo ""
echo "开始测试所有checkpoint..."
python test_all_checkpoints.py \
    --dino_ckpt ${DINO_CKPT} \
    --ckpt_dir ${CKPT_DIR} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --test_split test \
    --output_csv ${CKPT_DIR}/../test_results.csv \
    --use_ugbd \
    --use_sdf \
    --dual_sam_eval \
    --eval_tta

echo ""
echo "完成！结果保存在: ${CKPT_DIR}/../test_results.csv"
