#!/bin/bash
# 一键训练脚本
# 自动执行完整的训练流程

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================
# 基础配置
DINO_CKPT="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_DIR="./segdata"
DATASET="usod10k"
REPO_DIR="./dinov3"

# 训练配置
EPOCHS=100  # 增加训练轮数以更好优化MAE
BATCH_SIZE=8
INPUT_H=384
INPUT_W=384
LR=1e-4
WEIGHT_DECAY=1e-4
SCHEDULER="plateau"  # cosine, plateau, constant
PATIENCE=20  # 增加patience，给MAE更多优化时间
LR_FACTOR=0.5
MIN_LR=1e-6
# 为了让训练完整跑完EPOCHS轮次，这里默认关闭早停（如需早停，请在命令行单独指定）
EARLY_STOP=0
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
USE_DICE=true

# 损失权重
W_BCE=1.0
W_IOU=0.5
W_SSIM=0.40  # 增加SSIM权重，有助于细节
WAV_LEVELS=3
W_WAV=0.35  # 增加小波损失权重
W_WED=0.12  # 增加小波边缘dice权重
W_TV=0.08  # 增加TV损失权重，平滑预测
W_BFCE=0.15  # 增加边界focal BCE权重
BF_LAM=1.0
# MAE优化损失（用于降低MAE，当前采用相对温和配置，避免破坏泛化）
W_MAE=0.5        # 从0.8下调，仍然给予MAE足够权重
W_FOCAL_MAE=0.15 # 从0.3下调，减弱对极难样本的过拟合风险
FOCAL_MAE_ALPHA=0.25
FOCAL_MAE_GAMMA=2.0

# UGBD配置
UGBD_STEPS=3
UGBD_MID_CH=64

# SDF配置
SDF_MID_CH=64
SDF_LAMBDA=0.5
W_SDF=0.15  # 增加SDF权重，有助于精确边界
W_EIK=0.08  # 增加Eikonal权重
W_CURV=0.03  # 增加curvature权重
SDF_SCALE=1.0

# 其他训练参数
GRAD_CLIP=1.0
BEST_METRIC="MAE"  # mIoU, S_alpha, Fw_beta, mE_phi, MAE, maxF (改为MAE以优化MAE指标)
DUAL_SAM_EVAL=true
EVAL_TTA=true
TTA_SCALES="1.0,0.75,1.25"
TTA_NO_FLIP=false

# 输出配置
SAVE_ROOT="./runs_usod10k_wavelet_0003"
# ==================== 配置结束 ====================

echo "=========================================="
echo "一键训练脚本"
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
    --w_mae ${W_MAE} \
    --w_focal_mae ${W_FOCAL_MAE} \
    --focal_mae_alpha ${FOCAL_MAE_ALPHA} \
    --focal_mae_gamma ${FOCAL_MAE_GAMMA} \
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

echo ""
echo "Checkpoint保存在: ${CKPT_DIR}"
echo ""
echo "训练完成后，可以使用以下命令测试："
echo "  ./test_one_click.sh --ckpt_dir ${CKPT_DIR}"
echo "  或"
echo "  python test_all_checkpoints.py --ckpt_dir ${CKPT_DIR} ..."
echo ""
