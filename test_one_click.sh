#!/bin/bash
# 一键测试脚本
# 自动测试所有保存的checkpoint

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================
# 基础配置
DINO_CKPT="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_DIR="./segdata"
DATASET="usod10k"
REPO_DIR="./dinov3"

# 测试配置
TEST_SPLIT="test"
INPUT_H=384
INPUT_W=384
TEST_BATCH_SIZE=1
NUM_WORKERS=4

# 多测试集配置（可以测试多个测试集）
# 格式：用空格分隔的测试集名称列表
# 例如：TEST_DATASETS="usod10k_test usod_test cod10k_test"
# 如果不指定，则使用默认的 ${DATASET}_test
TEST_DATASETS="usod10k_test usod_test cod10k_test"  # 三个测试集一起测试

# 模型配置（必须与训练时一致）
DINO_SIZE="s"  # b or s
MAMBA_NUM_QUERIES=12
MAMBA_DIM=192
MAMBA_LAYERS=3
MAMBA_GROUPS=4
MAMBA_KERNEL=9
MAMBA_MASK_EMBED_DIM=64

# 功能开关（必须与训练时一致）
USE_UGBD=true
USE_SDF=true

# UGBD配置
UGBD_STEPS=3
UGBD_MID_CH=64

# SDF配置
SDF_MID_CH=64
SDF_LAMBDA=0.5

# 评估配置
DUAL_SAM_EVAL=true
EVAL_TTA=true
TTA_SCALES="1.0,0.75,1.25"
TTA_NO_FLIP=false

# Checkpoint目录（如果为空则自动查找）
CKPT_DIR=""

# 测试选项
TEST_ONLY_BEST=false  # true: 只测试best checkpoint, false: 测试所有checkpoint

# 输出配置
OUTPUT_CSV=""  # 如果为空则自动生成
# ==================== 配置结束 ====================

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt_dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --test_only_best)
            TEST_ONLY_BEST=true
            shift
            ;;
        --output_csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--ckpt_dir DIR] [--test_only_best] [--output_csv FILE] [--dataset NAME]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "一键测试脚本"
echo "=========================================="
echo "数据集: ${DATASET}"
echo "测试split: ${TEST_SPLIT}"
if [ -n "$TEST_DATASETS" ]; then
    echo "测试集: ${TEST_DATASETS}"
else
    echo "测试集: ${DATASET}_test (默认)"
fi
echo "=========================================="

# 自动查找checkpoint目录
if [ -z "$CKPT_DIR" ]; then
    SAVE_ROOT="./runs_usod10k_wavelet_0003"
    TAG="mamba"
    if [ "$USE_UGBD" = true ]; then
        TAG="${TAG}_ugbd"
    fi
    if [ "$USE_SDF" = true ]; then
        TAG="${TAG}_sdf"
    fi
    
    CKPT_DIR="${SAVE_ROOT}/segdino_${DINO_SIZE}_${INPUT_H}x${INPUT_W}_${DATASET}_${TAG}/ckpts"
    
    # 如果还是找不到，尝试查找任何包含ckpts的目录
    if [ ! -d "$CKPT_DIR" ]; then
        echo "警告: 默认checkpoint目录不存在: ${CKPT_DIR}"
        echo "尝试查找其他checkpoint目录..."
        POSSIBLE_DIRS=$(find ${SAVE_ROOT} -type d -name "ckpts" 2>/dev/null | head -1)
        if [ -n "$POSSIBLE_DIRS" ]; then
            CKPT_DIR="$POSSIBLE_DIRS"
            echo "找到checkpoint目录: ${CKPT_DIR}"
        else
            echo "错误: 找不到checkpoint目录，请使用 --ckpt_dir 参数指定"
            exit 1
        fi
    fi
fi

# 检查checkpoint目录是否存在
if [ ! -d "$CKPT_DIR" ]; then
    echo "错误: checkpoint目录不存在: ${CKPT_DIR}"
    exit 1
fi

echo "Checkpoint目录: ${CKPT_DIR}"

# 生成输出CSV路径
if [ -z "$OUTPUT_CSV" ]; then
    if [ "$TEST_ONLY_BEST" = true ]; then
        OUTPUT_CSV="${CKPT_DIR}/../test_results_best_only.csv"
    else
        OUTPUT_CSV="${CKPT_DIR}/../test_results_all_ckpts.csv"
    fi
fi

# 构建测试命令
TEST_CMD="python test_all_checkpoints.py \
    --dino_ckpt ${DINO_CKPT} \
    --ckpt_dir ${CKPT_DIR} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --repo_dir ${REPO_DIR} \
    --test_split ${TEST_SPLIT} \
    --input_h ${INPUT_H} \
    --input_w ${INPUT_W} \
    --batch_size ${TEST_BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --dino_size ${DINO_SIZE} \
    --mamba_num_queries ${MAMBA_NUM_QUERIES} \
    --mamba_dim ${MAMBA_DIM} \
    --mamba_layers ${MAMBA_LAYERS} \
    --mamba_groups ${MAMBA_GROUPS} \
    --mamba_kernel ${MAMBA_KERNEL} \
    --mamba_mask_embed_dim ${MAMBA_MASK_EMBED_DIM} \
    --ugbd_steps ${UGBD_STEPS} \
    --ugbd_mid_ch ${UGBD_MID_CH} \
    --sdf_mid_ch ${SDF_MID_CH} \
    --sdf_lambda ${SDF_LAMBDA}"

# 添加多测试集参数
if [ -n "$TEST_DATASETS" ]; then
    TEST_CMD="${TEST_CMD} --test_datasets ${TEST_DATASETS}"
fi

TEST_CMD="${TEST_CMD} --output_csv ${OUTPUT_CSV}"

# 添加测试可选参数
if [ "$USE_UGBD" = true ]; then
    TEST_CMD="${TEST_CMD} --use_ugbd"
fi

if [ "$USE_SDF" = true ]; then
    TEST_CMD="${TEST_CMD} --use_sdf"
fi

if [ "$DUAL_SAM_EVAL" = true ]; then
    TEST_CMD="${TEST_CMD} --dual_sam_eval"
fi

if [ "$EVAL_TTA" = true ]; then
    TEST_CMD="${TEST_CMD} --eval_tta"
    TEST_CMD="${TEST_CMD} --tta_scales ${TTA_SCALES}"
fi

if [ "$TTA_NO_FLIP" = true ]; then
    TEST_CMD="${TEST_CMD} --tta_no_flip"
fi

if [ "$TEST_ONLY_BEST" = true ]; then
    TEST_CMD="${TEST_CMD} --test_only_best"
fi

# 执行测试
echo ""
echo "=========================================="
echo "开始测试checkpoint..."
echo "=========================================="
echo "测试命令:"
echo "${TEST_CMD}"
echo ""

${TEST_CMD}

TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "测试失败，退出码: ${TEST_EXIT_CODE}"
    exit $TEST_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "测试结果保存在: ${OUTPUT_CSV}"
echo ""
echo "查看结果:"
echo "  cat ${OUTPUT_CSV}"
echo ""
