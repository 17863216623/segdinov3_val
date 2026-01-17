# WaveMamba_UGBR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

**WaveMamba_UGBR** is an advanced image segmentation framework that combines DINOv3 backbone with Mamba decoder, enhanced by Wavelet losses, UGBD (Uncertainty-Guided Boundary Diffusion) refiner, and SDF (Signed Distance Field) shape priors for high-quality boundary segmentation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Components](#model-components)
- [Loss Functions](#loss-functions)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

WaveMamba_UGBR is designed for high-precision image segmentation, particularly focusing on boundary quality. It integrates:

- **DINOv3 Backbone**: Frozen pre-trained vision transformer for robust feature extraction
- **DPT Head**: Multi-scale feature fusion
- **Mamba Decoder**: Lightweight state-space model decoder for efficient processing
- **UGBD Refiner**: Uncertainty-guided boundary refinement module
- **SDF Head**: Signed distance field shape prior for geometric constraints
- **Wavelet Losses**: Multi-scale wavelet domain losses for detail preservation

## ✨ Features

- **High Boundary Quality**: Multiple boundary-focused losses (Boundary IoU, Uncertainty-Weighted Boundary, Fourier Magnitude)
- **Efficient Architecture**: Lightweight decoder with minimal parameters
- **Multi-Scale Processing**: Wavelet domain losses for detail preservation
- **Shape Prior**: SDF constraints for geometric consistency
- **Uncertainty Guidance**: Adaptive refinement based on prediction uncertainty
- **Comprehensive Evaluation**: Support for multiple metrics (MAE, mIoU, S-measure, Fwβ, E-measure, maxF)

## 🏗️ Architecture

```
Input Image (B×3×H×W)
    ↓
┌─────────────────────────────────────┐
│  DINOv3 Backbone (Frozen)          │
│  - Multi-scale feature extraction  │
│  - Layers: [2, 5, 8, 11]           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  DPT Head                           │
│  - Feature projection & fusion      │
│  - Initial logits (B×1×H×W)         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Mamba Decoder                      │
│  - Image token encoding             │
│  - Query-based refinement           │
│  - Lightweight state-space model    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  UGBD Refiner (Optional)            │
│  - Uncertainty-guided refinement    │
│  - ConvGRU iterative refinement     │
│  - Boundary optimization            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  SDF Head (Optional)                │
│  - Signed distance field prediction │
│  - Geometric prior correction       │
│  - Residual fusion                  │
└─────────────────────────────────────┘
    ↓
Final Logits (B×1×H×W) → Sigmoid → Probability Map
```

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch 2.4+

### Step 1: Create Conda Environment

```bash
conda create -n wavemamba python=3.10.16
conda activate wavemamba
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Clone DINOv3 Repository

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

The `dinov3` directory should be in the project root.

### Step 4: Download DINOv3 Pretrained Weights

Download the pretrained DINOv3 weights and place them in `./web_pth/`:

- **DINOv3-Small**: `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- **DINOv3-Base**: `dinov3_vitb16_pretrain_lvd1689m-08c60483.pth`

You can download from [DINOv3 official repository](https://github.com/facebookresearch/dinov3).

## 📁 Dataset Preparation

### Supported Datasets

The framework supports various segmentation datasets. Organize your dataset in the following structure:

```
segdata/
└── {dataset_name}/
    ├── train/
    │   ├── image/      # Training images
    │   └── mask/        # Training masks (binary)
    └── test/
        ├── image/       # Test images
        └── mask/        # Test masks (binary)
```

### Example Datasets

- **USOD10K**: Underwater salient object detection dataset
- **TN3K**: Thyroid nodule segmentation dataset
- **Kvasir-SEG**: Polyp segmentation dataset
- **ISIC**: Skin lesion segmentation dataset

### Dataset Format

- **Images**: RGB images in `.jpg`, `.png`, or `.bmp` format
- **Masks**: Binary masks (0 for background, 255 for foreground) in `.png` or `.jpg` format
- **Naming**: Image and mask files should have the same base name (e.g., `image_001.jpg` and `image_001.png`)

## 🚀 Training

### Quick Start

For a quick start with default settings:

```bash
python train_segdinov3.py \
  --data_dir ./segdata \
  --dataset usod10k \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --dino_size s \
  --input_h 384 \
  --input_w 384 \
  --batch_size 8 \
  --epochs 80 \
  --lr 1e-4
```

### Full Training with All Modules

For best performance with all modules enabled:

```bash
bash train_aggressive.sh
```

Or manually:

```bash
python train_segdinov3.py \
  --data_dir ./segdata \
  --dataset usod10k \
  --mask_ext ".png" \
  --repo_dir ./dinov3 \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --dino_size s \
  --input_h 384 \
  --input_w 384 \
  --batch_size 8 \
  --epochs 120 \
  --num_workers 8 \
  --seed 42 \
  --mamba_num_queries 12 \
  --mamba_dim 192 \
  --mamba_layers 3 \
  --mamba_groups 4 \
  --mamba_kernel 9 \
  --mamba_mask_embed_dim 64 \
  --w_bce 1.0 \
  --w_iou 0.7 \
  --w_ssim 0.45 \
  --use_wavelet \
  --wav_levels 3 \
  --w_wav 0.35 \
  --w_wed 0.15 \
  --w_tv 0.10 \
  --w_bfce 0.18 \
  --bf_lam 1.5 \
  --use_ugbd \
  --ugbd_steps 4 \
  --ugbd_mid_ch 64 \
  --w_cldice 0.08 \
  --w_biou 0.25 \
  --w_uwb 0.20 \
  --uwb_lam 2.5 \
  --w_freq 0.10 \
  --freq_hf_boost 0.25 \
  --use_sdf \
  --sdf_mid_ch 64 \
  --sdf_lambda 0.35 \
  --w_sdf 0.15 \
  --w_eik 0.08 \
  --w_curv 0.04 \
  --sdf_scale 0.7 \
  --dual_sam_eval \
  --eval_tta \
  --tta_scales 1.0,0.75,1.25 \
  --lr 6e-5 \
  --weight_decay 1.5e-4 \
  --scheduler cosine \
  --warmup_epochs 12 \
  --use_ema \
  --ema_decay 0.999 \
  --use_aug \
  --grad_clip 0.8 \
  --best_metric Fw_beta
```

### Training Parameters

#### Basic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./segdata` | Root directory of datasets |
| `--dataset` | `usod10k` | Dataset name |
| `--input_h` | `384` | Input image height |
| `--input_w` | `384` | Input image width |
| `--batch_size` | `8` | Batch size |
| `--epochs` | `80` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--num_workers` | `8` | Number of data loading workers |

#### DINOv3 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dino_size` | `s` | DINOv3 size: `s` (small) or `b` (base) |
| `--dino_ckpt` | Required | Path to DINOv3 pretrained weights |
| `--repo_dir` | `./dinov3` | Path to DINOv3 repository |

#### Mamba Decoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mamba_num_queries` | `12` | Number of query vectors |
| `--mamba_dim` | `192` | Token dimension |
| `--mamba_layers` | `3` | Number of decoder layers |
| `--mamba_groups` | `4` | GroupSSM groups |
| `--mamba_kernel` | `9` | Convolution kernel size |

#### Loss Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--w_bce` | `1.0` | Binary cross-entropy weight |
| `--w_iou` | `0.7` | IoU loss weight |
| `--w_ssim` | `0.45` | SSIM loss weight |
| `--w_wav` | `0.35` | Wavelet detail loss weight |
| `--w_wed` | `0.15` | Wavelet edge dice weight |
| `--w_tv` | `0.10` | Total variation weight |
| `--w_bfce` | `0.18` | Boundary focal BCE weight |
| `--w_biou` | `0.25` | Boundary IoU weight |
| `--w_uwb` | `0.20` | Uncertainty-weighted boundary weight |
| `--w_freq` | `0.10` | Fourier magnitude weight |
| `--w_sdf` | `0.15` | SDF L1 loss weight |
| `--w_eik` | `0.08` | Eikonal loss weight |
| `--w_curv` | `0.04` | Curvature loss weight |

#### Training Strategy

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scheduler` | `cosine` | Learning rate scheduler |
| `--warmup_epochs` | `12` | Warmup epochs |
| `--weight_decay` | `1.5e-4` | Weight decay |
| `--grad_clip` | `0.8` | Gradient clipping threshold |
| `--use_ema` | False | Enable EMA |
| `--ema_decay` | `0.999` | EMA decay rate |
| `--use_aug` | False | Enable data augmentation |

### Training Process

1. **Initialization**:
   - Load DINOv3 pretrained weights
   - Initialize model components (backbone, decoder, refiner, SDF head)
   - Setup optimizer (AdamW) and scheduler (Cosine with Warmup)
   - Prepare data loaders

2. **Training Loop** (for each epoch):
   - **Forward Pass**:
     - Extract features with DINOv3 backbone
     - Generate initial logits with DPT head
     - Refine with Mamba decoder
     - Apply UGBD refiner (if enabled)
     - Apply SDF head (if enabled)
   
   - **Loss Calculation**:
     - Compute multiple losses (BCE, IoU, SSIM, Wavelet, Boundary, SDF)
     - Weighted sum of all losses
   
   - **Backward Pass**:
     - Compute gradients
     - Apply gradient clipping
     - Update parameters
     - Update EMA (if enabled)

3. **Validation**:
   - Evaluate on validation set
   - Compute metrics (MAE, mIoU, S-measure, Fwβ, E-measure, maxF)
   - Save best model checkpoint
   - Record metrics to CSV

4. **Checkpointing**:
   - Save latest checkpoint: `latest.pth`
   - Save best checkpoint: `best_{metric}_ep{epoch}_{value}.pth`
   - Save metrics history: `metrics_history.csv`

### Training Output

Training outputs are saved in `./runs_usod10k_wavelet/segdino_s_{input_h}x{input_w}_{dataset}_mamba_wav_ugbd/`:

```
runs_usod10k_wavelet/
└── segdino_s_384x384_usod10k_mamba_wav_ugbd/
    ├── ckpts/
    │   ├── latest.pth
    │   └── best_Fw_beta_ep068_0.9221.pth
    ├── metrics_history.csv
    └── model_profile.txt
```

## 📊 Evaluation

### Testing with Trained Model

```bash
python test_segdino.py \
  --data_dir ./segdata \
  --dataset usod10k \
  --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --ckpt ./runs_usod10k_wavelet/segdino_s_384x384_usod10k_mamba_wav_ugbd/ckpts/best_Fw_beta_ep068_0.9221.pth \
  --dino_size s \
  --input_h 384 \
  --input_w 384 \
  --batch_size 1
```

### Evaluation Metrics

The framework evaluates using multiple metrics:

| Metric | Description | Range | Higher is Better |
|--------|-------------|-------|------------------|
| **MAE** | Mean Absolute Error | [0, 1] | ❌ |
| **mIoU** | Mean Intersection over Union | [0, 1] | ✅ |
| **S_alpha** | S-measure (structure similarity) | [0, 1] | ✅ |
| **Fw_beta** | Weighted F-measure (β=0.3) | [0, 1] | ✅ |
| **mE_phi** | Mean E-measure | [0, 1] | ✅ |
| **E_phi_adp** | Adaptive E-measure | [0, 1] | ✅ |
| **maxF** | Maximum F-measure | [0, 1] | ✅ |

### Test-Time Augmentation (TTA)

Enable TTA for better evaluation results:

```bash
--eval_tta \
--tta_scales 1.0,0.75,1.25
```

TTA performs multi-scale testing with horizontal flipping and averages the predictions.

## 🧩 Model Components

### 1. DINOv3 Backbone

- **Architecture**: Vision Transformer (ViT)
- **Sizes**: Small (ViT-S/16) or Base (ViT-B/16)
- **Feature Extraction**: Layers [2, 5, 8, 11] for multi-scale features
- **Status**: Frozen during training (pretrained weights)

### 2. DPT Head

- **Function**: Multi-scale feature fusion
- **Input**: 4-layer features from DINOv3
- **Output**: Initial segmentation logits
- **Architecture**: Feature projection → Upsampling → Concatenation → Output convolution

### 3. Mamba Decoder

- **Architecture**: Lightweight state-space model
- **Components**:
  - ImageTokenEncoder: Encodes image + probability to tokens
  - GroupSSM1D: Grouped state-space module
  - Query mechanism: Learnable query vectors
- **Parameters**: ~2-5M trainable parameters

### 4. UGBD Refiner

- **Full Name**: Uncertainty-Guided Boundary Diffusion Refiner
- **Function**: Iterative boundary refinement
- **Components**:
  - Uncertainty computation: `uncert = 4 × prob × (1-prob)`
  - Edge extraction: Wavelet-like edge detection
  - ConvGRU: Iterative refinement (3-4 steps)
- **Output**: Residual refinement added to logits

### 5. SDF Head

- **Full Name**: Signed Distance Field Head
- **Function**: Geometric shape prior
- **Components**:
  - SDF prediction: Distance field from boundary
  - Eikonal constraint: Gradient magnitude = 1
  - Curvature constraint: Smoothness regularization
- **Fusion**: `refined_logits = logits - λ × sdf`

## 📉 Loss Functions

### Total Loss Formula

```
L_total = w_bce × L_BCE
        + w_iou × L_IoU
        + w_ssim × L_SSIM
        + w_wav × L_wavelet_detail
        + w_wed × L_wavelet_edge_dice
        + w_tv × L_TV
        + w_bfce × L_boundary_focal
        + w_biou × L_boundary_iou
        + w_uwb × L_uncertainty_weighted_boundary
        + w_freq × L_fourier
        + w_cldice × L_clDice
        + w_sdf × L_SDF_L1
        + w_eik × L_Eikonal
        + w_curv × L_Curvature
```

### Loss Categories

#### 1. Basic Losses

- **BCE**: Binary cross-entropy loss
- **IoU**: Intersection over Union loss
- **SSIM**: Structural similarity loss

#### 2. Wavelet Losses

- **Wavelet Detail Loss**: Preserves high-frequency details
- **Wavelet Edge Dice**: Edge region overlap
- **TV L1 Loss**: Total variation regularization
- **Boundary Focal BCE**: Boundary-focused classification

#### 3. Boundary Losses

- **Boundary IoU**: Direct boundary overlap optimization
- **Uncertainty-Weighted Boundary**: Uncertainty-guided boundary loss
- **Fourier Magnitude**: Frequency domain constraint

#### 4. Topology Losses

- **clDice**: Centerline Dice for topology preservation

#### 5. SDF Losses

- **SDF L1**: Signed distance field prediction loss
- **Eikonal Loss**: Gradient magnitude constraint
- **Curvature Loss**: Smoothness constraint

## 📈 Results

### Performance on USOD10K Dataset

**Best Model**: `best_Fw_beta_ep068_0.9221.pth` (Epoch 68)

| Metric | Value |
|--------|-------|
| **MAE** | 0.0168 |
| **mIoU** | 0.8730 |
| **S_alpha** | 0.9273 |
| **Fw_beta** | 0.9221 |
| **mE_phi** | 0.9704 |
| **E_phi_adp** | 0.9732 |
| **maxF** | 0.9263 |

*Results obtained with full training configuration including Wavelet losses, UGBD refiner, and SDF head. Training for 120 epochs with aggressive optimization settings.*

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{wavemamba2025,
  title={WaveMamba\_UGBR: Wavelet-Enhanced Mamba with Uncertainty-Guided Boundary Refinement for Image Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 🙏 Acknowledgments

- [DINOv3](https://github.com/facebookresearch/dinov3) - Vision transformer backbone
- [DPT](https://github.com/isl-org/DPT) - Dense prediction transformer
- [PySODMetrics](https://github.com/lartpang/PySODMetrics) - Evaluation metrics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Work

- SegDINO: Efficient medical image segmentation with DINO-V3
- Mamba: Efficient state-space models for long sequences
- UGBD: Uncertainty-guided boundary diffusion

---

**Last Updated**: 2025  
**Version**: WaveMamba_UGBR v1.0
