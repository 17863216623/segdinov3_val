#!/bin/bash

# ============================================
# SegDINOv3 + SDF 形状先验 优化训练脚本
# 目标：重现 MAE=0.0168, Fwβ=0.9232
# ============================================

# 设置默认参数（已优化）
DATA_DIR="./segdata"
DATASET="usod10k"
MASK_EXT=".png"
REPO_DIR="./dinov3"
DINO_CKPT="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DINO_SIZE="s"
INPUT_H=384
INPUT_W=384
BATCH_SIZE=8
EPOCHS=100  # 增加训练轮数以获得更好收敛
NUM_WORKERS=8
SEED=42

# Mamba解码器参数（保持不变）
MAMBA_NUM_QUERIES=12
MAMBA_DIM=192
MAMBA_LAYERS=3
MAMBA_GROUPS=4
MAMBA_KERNEL=9
MAMBA_MASK_EMBED_DIM=64

# Wavelet参数（微调）
USE_WAVELET="--use_wavelet"
WAV_LEVELS=3
W_WAV=0.32      # 从0.30提升到0.32，增强细节约束
W_WED=0.12      # 从0.10提升到0.12，增强边缘约束
W_TV=0.08       # 从0.06提升到0.08，增强平滑性
W_BFCE=0.15     # 从0.12提升到0.15，增强边界焦点
BF_LAM=1.2      # 从1.0提升到1.2，增强边界放大

# UGBD细化头参数（微调）
USE_UGBD="--use_ugbd"
UGBD_STEPS=4    # 从3增加到4，增强细化能力
UGBD_MID_CH=64
W_CLDICE=0.06   # 从0.05提升到0.06，增强拓扑约束

# SDF形状先验头参数（关键优化）
USE_SDF="--use_sdf"
SDF_MID_CH=64
SDF_LAMBDA=0.4  # 从0.5降低到0.4，减少SDF对logits的干扰
W_SDF=0.12      # 从0.10提升到0.12，增强SDF拟合
W_EIK=0.06      # 从0.05提升到0.06，增强Eikonal约束
W_CURV=0.03     # 从0.02提升到0.03，增强曲率平滑
SDF_SCALE=0.8   # 从1.0降低到0.8，SDF尺度更温和

# 基础损失权重（微调）
W_BCE=1.0       # 保持
W_IOU=0.6       # 从0.5提升到0.6，增强IoU约束
W_SSIM=0.4      # 从0.35提升到0.4，增强结构相似性

# 评估参数
DUAL_SAM_EVAL="--dual_sam_eval"
EVAL_TTA="--eval_tta"
TTA_SCALES="1.0,0.75,1.25"
TTA_NO_FLIP=""

# 训练器参数（优化）
LR=8e-5         # 从1e-4降低到8e-5，更稳定的学习
WEIGHT_DECAY=1.2e-4  # 从1e-4提升到1.2e-4，增强正则化
SCHEDULER="cosine"
WARMUP_EPOCHS=10     # 从8增加到10，更长的warmup
USE_EMA="--use_ema"
EMA_DECAY=0.998      # 从0.996提升到0.998，更慢的EMA更新

# 其他参数
BEST_METRIC="Fw_beta"
USE_AUG="--use_aug"  # 启用数据增强

# ============================================
# 构建训练命令
# ============================================

CMD="python train_segdinov3.py \
  --data_dir ${DATA_DIR} \
  --dataset ${DATASET} \
  --mask_ext ${MASK_EXT} \
  --repo_dir ${REPO_DIR} \
  --dino_ckpt ${DINO_CKPT} \
  --dino_size ${DINO_SIZE} \
  --input_h ${INPUT_H} \
  --input_w ${INPUT_W} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --num_workers ${NUM_WORKERS} \
  --seed ${SEED} \
  --mamba_num_queries ${MAMBA_NUM_QUERIES} \
  --mamba_dim ${MAMBA_DIM} \
  --mamba_layers ${MAMBA_LAYERS} \
  --mamba_groups ${MAMBA_GROUPS} \
  --mamba_kernel ${MAMBA_KERNEL} \
  --mamba_mask_embed_dim ${MAMBA_MASK_EMBED_DIM}"

# 添加基础损失权重
CMD="${CMD} --w_bce ${W_BCE} --w_iou ${W_IOU} --w_ssim ${W_SSIM}"

# 添加Wavelet参数
if [ -n "${USE_WAVELET}" ]; then
  CMD="${CMD} ${USE_WAVELET} --wav_levels ${WAV_LEVELS} --w_wav ${W_WAV} --w_wed ${W_WED} --w_tv ${W_TV} --w_bfce ${W_BFCE} --bf_lam ${BF_LAM}"
fi

# 添加UGBD参数
if [ -n "${USE_UGBD}" ]; then
  CMD="${CMD} ${USE_UGBD} --ugbd_steps ${UGBD_STEPS} --ugbd_mid_ch ${UGBD_MID_CH} --w_cldice ${W_CLDICE}"
fi

# 添加SDF参数
if [ -n "${USE_SDF}" ]; then
  CMD="${CMD} ${USE_SDF} --sdf_mid_ch ${SDF_MID_CH} --sdf_lambda ${SDF_LAMBDA} --w_sdf ${W_SDF} --w_eik ${W_EIK} --w_curv ${W_CURV} --sdf_scale ${SDF_SCALE}"
fi

# 添加评估参数
if [ -n "${DUAL_SAM_EVAL}" ]; then
  CMD="${CMD} ${DUAL_SAM_EVAL}"
fi
if [ -n "${EVAL_TTA}" ]; then
  CMD="${CMD} ${EVAL_TTA} --tta_scales ${TTA_SCALES}"
  if [ -n "${TTA_NO_FLIP}" ]; then
    CMD="${CMD} ${TTA_NO_FLIP}"
  fi
fi

# 添加训练器参数
CMD="${CMD} --lr ${LR} --weight_decay ${WEIGHT_DECAY} --scheduler ${SCHEDULER} --warmup_epochs ${WARMUP_EPOCHS}"
if [ -n "${USE_EMA}" ]; then
  CMD="${CMD} ${USE_EMA} --ema_decay ${EMA_DECAY}"
fi

# 添加数据增强
if [ -n "${USE_AUG}" ]; then
  CMD="${CMD} ${USE_AUG}"
fi

# 添加梯度裁剪
CMD="${CMD} --grad_clip 1.0"

# 添加其他参数
CMD="${CMD} --best_metric ${BEST_METRIC}"

# ============================================
# 打印配置并执行
# ============================================

echo "============================================"
echo "SegDINOv3 + SDF 优化训练配置"
echo "目标：MAE=0.0168, Fwβ=0.9232"
echo "============================================"
echo "数据集: ${DATASET}"
echo "输入尺寸: ${INPUT_H}x${INPUT_W}"
echo "批次大小: ${BATCH_SIZE}"
echo "训练轮数: ${EPOCHS}"
echo "学习率: ${LR}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Warmup Epochs: ${WARMUP_EPOCHS}"
echo "EMA Decay: ${EMA_DECAY}"
echo ""
echo "损失权重:"
echo "  BCE: ${W_BCE}, IoU: ${W_IOU}, SSIM: ${W_SSIM}"
echo "  Wavelet: WAV=${W_WAV}, WED=${W_WED}, TV=${W_TV}, BFCE=${W_BFCE}"
echo "  SDF: SDF=${W_SDF}, EIK=${W_EIK}, CURV=${W_CURV}"
echo "  clDice: ${W_CLDICE}"
echo ""
echo "模块:"
echo "  Wavelet: ${USE_WAVELET:-未启用}"
echo "  UGBD: ${USE_UGBD:-未启用} (steps=${UGBD_STEPS})"
echo "  SDF: ${USE_SDF:-未启用} (lambda=${SDF_LAMBDA}, scale=${SDF_SCALE})"
echo "  数据增强: ${USE_AUG:-未启用}"
echo ""
echo "最佳指标: ${BEST_METRIC}"
echo "============================================"
echo ""
echo "执行命令:"
echo "${CMD}"
echo ""
echo "开始训练..."
echo ""

# 执行训练
eval ${CMD}

echo ""
echo "训练完成！"
echo "检查最佳模型: ./runs_usod10k_wavelet/segdino_s_384x384_usod10k_mamba_wav_ugbd/ckpts/best_${BEST_METRIC}_*.pth"

