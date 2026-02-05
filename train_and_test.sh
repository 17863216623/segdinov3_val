#!/bin/bash
# 一键训练和测试脚本
# 自动执行训练，训练完成后自动测试所有checkpoint

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================
# 基础配置
DINO_CKPT="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_DIR="./segdata"
DATASET="usod10k"
REPO_DIR="./dinov3"

# 训练配置
EPOCHS=80
BATCH_SIZE=8
INPUT_H=384
INPUT_W=384
LR=1e-4
WEIGHT_DECAY=1e-4
SCHEDULER="plateau"  # cosine, plateau, constant
PATIENCE=15
LR_FACTOR=0.5
MIN_LR=1e-6
EARLY_STOP=20
WARMUP_EPOCHS=8
SEED=42
NUM_WORKERS=8

# 模型配置
DINO_SIZE="s"  # b or s
MAMBA_NUM_QUERIES=12
MAMBA_DIM=192
MAMBA_LAYERS=3
MAMBA_GROUPS=4
MAMBA_KERNEL=9
MAMBA_MASK_EMBED_DIM=64

# 功能开关
USE_WAVELET=true
USE_UGBD=true
USE_SDF=true
SAVE_EVERY_EPOCH=true

# 损失权重
W_BCE=1.0
W_IOU=0.5
USE_DICE=true
W_SSIM=0.35
WAV_LEVELS=3
W_WAV=0.30
W_WED=0.10
W_TV=0.06
W_BFCE=0.12
BF_LAM=1.0

# UGBD配置
UGBD_STEPS=3
UGBD_MID_CH=64

# SDF配置
SDF_MID_CH=64
SDF_LAMBDA=0.5
W_SDF=0.10
W_EIK=0.05
W_CURV=0.02
SDF_SCALE=1.0

# 其他训练参数
GRAD_CLIP=1.0
BEST_METRIC="Fw_beta"  # mIoU, S_alpha, Fw_beta, mE_phi, MAE, maxF
DUAL_SAM_EVAL=true
EVAL_TTA=true
TTA_SCALES="1.0,0.75,1.25"
TTA_NO_FLIP=false

# 测试配置
TEST_SPLIT="test"
TEST_BATCH_SIZE=1
TEST_ONLY_BEST=false  # true: 只测试best checkpoint, false: 测试所有checkpoint

# 输出配置
SAVE_ROOT="./runs_usod10k_wavelet"
# ==================== 配置结束 ====================

echo "=========================================="
echo "一键训练和测试脚本"
echo "=========================================="
echo "数据集: ${DATASET}"
echo "训练轮数: ${EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "学习率: ${LR}"
echo "调度器: ${SCHEDULER}"
echo "早停: ${EARLY_STOP}"
echo "=========================================="

# 构建训练命令
TRAIN_CMD="python train_segdinov3.py \
    --dino_ckpt ${DINO_CKPT} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --repo_dir ${REPO_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --input_h ${INPUT_H} \
    --input_w ${INPUT_W} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --scheduler ${SCHEDULER} \
    --patience ${PATIENCE} \
    --lr_factor ${LR_FACTOR} \
    --min_lr ${MIN_LR} \
    --early_stop ${EARLY_STOP} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKERS} \
    --dino_size ${DINO_SIZE} \
    --mamba_num_queries ${MAMBA_NUM_QUERIES} \
    --mamba_dim ${MAMBA_DIM} \
    --mamba_layers ${MAMBA_LAYERS} \
    --mamba_groups ${MAMBA_GROUPS} \
    --mamba_kernel ${MAMBA_KERNEL} \
    --mamba_mask_embed_dim ${MAMBA_MASK_EMBED_DIM} \
    --w_bce ${W_BCE} \
    --w_iou ${W_IOU} \
    --w_ssim ${W_SSIM} \
    --wav_levels ${WAV_LEVELS} \
    --w_wav ${W_WAV} \
    --w_wed ${W_WED} \
    --w_tv ${W_TV} \
    --w_bfce ${W_BFCE} \
    --bf_lam ${BF_LAM} \
    --ugbd_steps ${UGBD_STEPS} \
    --ugbd_mid_ch ${UGBD_MID_CH} \
    --sdf_mid_ch ${SDF_MID_CH} \
    --sdf_lambda ${SDF_LAMBDA} \
    --w_sdf ${W_SDF} \
    --w_eik ${W_EIK} \
    --w_curv ${W_CURV} \
    --sdf_scale ${SDF_SCALE} \
    --grad_clip ${GRAD_CLIP} \
    --best_metric ${BEST_METRIC} \
    --save_root ${SAVE_ROOT}"

# 添加可选参数
if [ "$USE_WAVELET" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_wavelet"
fi

if [ "$USE_DICE" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_dice"
fi

if [ "$USE_UGBD" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_ugbd"
fi

if [ "$USE_SDF" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_sdf"
fi

if [ "$SAVE_EVERY_EPOCH" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --save_every_epoch"
fi

if [ "$DUAL_SAM_EVAL" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --dual_sam_eval"
fi

if [ "$EVAL_TTA" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --eval_tta"
    TRAIN_CMD="${TRAIN_CMD} --tta_scales ${TTA_SCALES}"
fi

if [ "$TTA_NO_FLIP" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --tta_no_flip"
fi

# 执行训练
echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo "训练命令:"
echo "${TRAIN_CMD}"
echo ""

${TRAIN_CMD}

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "训练失败，退出码: ${TRAIN_EXIT_CODE}"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="

# 确定checkpoint目录
# 根据训练脚本的输出路径规则构建
TAG="mamba"
if [ "$USE_WAVELET" = true ]; then
    TAG="${TAG}_wav"
fi
if [ "$USE_UGBD" = true ]; then
    TAG="${TAG}_ugbd"
fi
if [ "$USE_SDF" = true ]; then
    TAG="${TAG}_sdf"
fi

CKPT_DIR="${SAVE_ROOT}/segdino_${DINO_SIZE}_${INPUT_H}x${INPUT_W}_${DATASET}_${TAG}/ckpts"

# 检查checkpoint目录是否存在
if [ ! -d "$CKPT_DIR" ]; then
    echo "警告: checkpoint目录不存在: ${CKPT_DIR}"
    echo "请手动指定checkpoint目录"
    exit 1
fi

echo "Checkpoint目录: ${CKPT_DIR}"

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

# 生成输出CSV路径
OUTPUT_CSV="${CKPT_DIR}/../test_results_all_ckpts.csv"
if [ "$TEST_ONLY_BEST" = true ]; then
    OUTPUT_CSV="${CKPT_DIR}/../test_results_best_only.csv"
fi
TEST_CMD="${TEST_CMD} --output_csv ${OUTPUT_CSV}"

# 执行测试
echo ""
echo "=========================================="
echo "开始测试所有checkpoint..."
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
echo "全部完成！"
echo "=========================================="
echo "训练checkpoint目录: ${CKPT_DIR}"
echo "测试结果CSV: ${OUTPUT_CSV}"
echo ""
echo "可以查看测试结果:"
echo "  cat ${OUTPUT_CSV}"
echo ""
