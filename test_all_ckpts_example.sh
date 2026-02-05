#!/bin/bash
# 测试所有checkpoint的示例脚本

# 设置checkpoint目录（根据实际训练输出调整）
CKPT_DIR="./runs_usod10k_wavelet/segdino_s_384x384_usod10k_mamba_wav_ugbd/ckpts"

# 测试所有checkpoint
python test_all_checkpoints.py \
    --dino_ckpt ./dinov3/dinov2_vitb14_pretrain.pth \
    --ckpt_dir ${CKPT_DIR} \
    --data_dir ./segdata \
    --dataset usod10k \
    --test_split test \
    --input_h 384 \
    --input_w 384 \
    --batch_size 1 \
    --num_workers 4 \
    --output_csv ./test_results_all_ckpts.csv \
    --use_ugbd \
    --ugbd_steps 3 \
    --ugbd_mid_ch 64 \
    --use_sdf \
    --sdf_mid_ch 64 \
    --sdf_lambda 0.5 \
    --mamba_num_queries 12 \
    --mamba_dim 192 \
    --mamba_layers 3 \
    --mamba_groups 4 \
    --mamba_kernel 9 \
    --mamba_mask_embed_dim 64 \
    --dual_sam_eval \
    --eval_tta

echo "测试完成！结果保存在 test_results_all_ckpts.csv"
