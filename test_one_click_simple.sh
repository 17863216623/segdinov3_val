#!/bin/bash
# 简化版一键测试脚本
# 使用默认配置，自动查找checkpoint目录

set -e

# 基础配置（根据实际情况修改）
DINO_CKPT="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_DIR="./segdata"
DATASET="usod10k"

# 多测试集配置（三个测试集一起测试）
TEST_DATASETS="usod10k_test usod_test cod10k_test"

echo "=========================================="
echo "一键测试脚本（简化版）"
echo "=========================================="
echo "测试集: ${TEST_DATASETS}"
echo ""

# 自动检测checkpoint目录
SAVE_ROOT="./runs_usod10k_wavelet"
CKPT_DIR=$(find ${SAVE_ROOT} -type d -name "ckpts" 2>/dev/null | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "错误: 找不到checkpoint目录"
    echo "请使用: ./test_one_click.sh --ckpt_dir <checkpoint目录>"
    exit 1
fi

echo "找到checkpoint目录: ${CKPT_DIR}"
echo ""

# 执行测试（在多个测试集上）
python test_all_checkpoints.py \
    --dino_ckpt ${DINO_CKPT} \
    --ckpt_dir ${CKPT_DIR} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --test_split test \
    --test_datasets ${TEST_DATASETS} \
    --output_csv ${CKPT_DIR}/../test_results_all_datasets.csv \
    --use_ugbd \
    --use_sdf \
    --dual_sam_eval \
    --eval_tta

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "结果保存在: ${CKPT_DIR}/../test_results_all_datasets.csv"
echo ""
