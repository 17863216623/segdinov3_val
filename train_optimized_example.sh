#!/bin/bash
# 优化后的训练脚本示例
# 使用验证集、早停、学习率调度和保存所有epoch权重

python train_segdinov3.py \
    --dino_ckpt ./dinov3/dinov2_vitb14_pretrain.pth \
    --data_dir ./segdata \
    --dataset usod10k \
    --epochs 80 \
    --batch_size 8 \
    --input_h 384 \
    --input_w 384 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler plateau \
    --patience 15 \
    --lr_factor 0.5 \
    --min_lr 1e-6 \
    --early_stop 20 \
    --warmup_epochs 8 \
    --save_every_epoch \
    --use_wavelet \
    --wav_levels 3 \
    --w_wav 0.30 \
    --w_wed 0.10 \
    --w_tv 0.06 \
    --w_bfce 0.12 \
    --use_ugbd \
    --ugbd_steps 3 \
    --ugbd_mid_ch 64 \
    --use_sdf \
    --sdf_mid_ch 64 \
    --sdf_lambda 0.5 \
    --w_sdf 0.10 \
    --w_eik 0.05 \
    --w_curv 0.02 \
    --w_bce 1.0 \
    --w_iou 0.5 \
    --use_dice \
    --w_ssim 0.35 \
    --grad_clip 1.0 \
    --best_metric Fw_beta \
    --dual_sam_eval \
    --eval_tta \
    --num_workers 8 \
    --seed 42
