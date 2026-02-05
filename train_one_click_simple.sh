#!/bin/bash
# 简化版一键训练脚本
# 使用默认配置，适合快速测试

set -e

# 基础配置（根据实际情况修改）
DINO_CKPT="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_DIR="./segdata"
DATASET="usod10k"

echo "=========================================="
echo "一键训练脚本（简化版）"
echo "=========================================="
echo "数据集: ${DATASET}"
echo ""

# 执行训练
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

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "训练完成后，可以使用以下命令测试："
echo "  ./test_one_click.sh"
echo "  或"
echo "  ./test_one_click.sh --ckpt_dir <checkpoint目录>"
echo ""
