#!/bin/bash

# ============================================
# SegDINOv3 + SDF 激进优化训练脚本
# 目标：MAE=0.0168, Fwβ=0.9232
# 当前最佳：MAE=0.0174, Fwβ=0.9200
# ============================================

# 设置默认参数（激进优化）
DATA_DIR="./segdata"
DATASET="usod10k"
MASK_EXT=".png"
REPO_DIR="./dinov3"
DINO_CKPT="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DINO_SIZE="s"
INPUT_H=384
INPUT_W=384
BATCH_SIZE=8
EPOCHS=120  # 增加到120，给模型更多收敛时间
NUM_WORKERS=8
SEED=42

# Mamba解码器参数（保持不变）
MAMBA_NUM_QUERIES=12
MAMBA_DIM=192
MAMBA_LAYERS=3
MAMBA_GROUPS=4
MAMBA_KERNEL=9
MAMBA_MASK_EMBED_DIM=64

# 基础损失权重（激进调整）
W_BCE=1.0
W_IOU=0.7      # 从0.6进一步提升到0.7
W_SSIM=0.45    # 从0.4进一步提升到0.45

# Wavelet参数（激进调整）
USE_WAVELET="--use_wavelet"
WAV_LEVELS=3
W_WAV=0.35     # 从0.32提升到0.35
W_WED=0.15     # 从0.12提升到0.15
W_TV=0.10      # 从0.08提升到0.10
W_BFCE=0.18    # 从0.15提升到0.18
BF_LAM=1.5     # 从1.2提升到1.5

# UGBD细化头参数
USE_UGBD="--use_ugbd"
UGBD_STEPS=4
UGBD_MID_CH=64
W_CLDICE=0.08  # 从0.06提升到0.08

# 新增边界损失（关键！）
W_BIOU=0.25    # Boundary IoU损失，直接提升Fwβ
W_UWB=0.20     # Uncertainty-Weighted Boundary损失
UWB_LAM=2.5    # UWB lambda参数
W_FREQ=0.10    # Fourier magnitude损失，提升高频细节

# SDF形状先验头参数（精细调整）
USE_SDF="--use_sdf"
SDF_MID_CH=64
SDF_LAMBDA=0.35 # 从0.4进一步降低到0.35，减少干扰
W_SDF=0.15      # 从0.12提升到0.15
W_EIK=0.08      # 从0.06提升到0.08
W_CURV=0.04     # 从0.03提升到0.04
SDF_SCALE=0.7   # 从0.8降低到0.7，更温和

# 评估参数
DUAL_SAM_EVAL="--dual_sam_eval"
EVAL_TTA="--eval_tta"
TTA_SCALES="1.0,0.75,1.25"
TTA_NO_FLIP=""

# 训练器参数（精细调整）
LR=6e-5         # 从8e-5进一步降低到6e-5，更稳定
WEIGHT_DECAY=1.5e-4  # 从1.2e-4提升到1.5e-4，更强正则化
SCHEDULER="cosine"
WARMUP_EPOCHS=12     # 从10增加到12
USE_EMA="--use_ema"
EMA_DECAY=0.999      # 从0.998提升到0.999，更慢的EMA更新

# 其他参数
BEST_METRIC="Fw_beta"
USE_AUG="--use_aug"
GRAD_CLIP=0.8   # 从1.0降低到0.8，更严格的梯度裁剪

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

# 添加新增边界损失
CMD="${CMD} --w_biou ${W_BIOU} --w_uwb ${W_UWB} --uwb_lam ${UWB_LAM} --w_freq ${W_FREQ} --freq_hf_boost 0.25"

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

# 添加数据增强和梯度裁剪
if [ -n "${USE_AUG}" ]; then
  CMD="${CMD} ${USE_AUG}"
fi
CMD="${CMD} --grad_clip ${GRAD_CLIP}"

# 添加其他参数
CMD="${CMD} --best_metric ${BEST_METRIC}"

# ============================================
# 打印配置并执行
# ============================================

echo "============================================"
echo "SegDINOv3 + SDF 激进优化训练配置"
echo "目标：MAE=0.0168, Fwβ=0.9232"
echo "当前：MAE=0.0174, Fwβ=0.9200"
echo "============================================"
echo "数据集: ${DATASET}"
echo "输入尺寸: ${INPUT_H}x${INPUT_W}"
echo "批次大小: ${BATCH_SIZE}"
echo "训练轮数: ${EPOCHS}"
echo "学习率: ${LR}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Warmup Epochs: ${WARMUP_EPOCHS}"
echo "EMA Decay: ${EMA_DECAY}"
echo "Gradient Clip: ${GRAD_CLIP}"
echo ""
echo "基础损失权重:"
echo "  BCE: ${W_BCE}, IoU: ${W_IOU}, SSIM: ${W_SSIM}"
echo ""
echo "Wavelet损失:"
echo "  WAV=${W_WAV}, WED=${W_WED}, TV=${W_TV}, BFCE=${W_BFCE}, BF_LAM=${BF_LAM}"
echo ""
echo "新增边界损失（关键！）:"
echo "  Boundary IoU: ${W_BIOU}"
echo "  UWB: ${W_UWB} (lambda=${UWB_LAM})"
echo "  Fourier: ${W_FREQ}"
echo ""
echo "SDF损失:"
echo "  SDF=${W_SDF}, EIK=${W_EIK}, CURV=${W_CURV}"
echo "  Lambda=${SDF_LAMBDA}, Scale=${SDF_SCALE}"
echo ""
echo "其他:"
echo "  clDice: ${W_CLDICE}"
echo "  UGBD Steps: ${UGBD_STEPS}"
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


