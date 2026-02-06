# train_segdinov3.py
# -*- coding: utf-8 -*-
import os, csv, cv2, glob, math, random, numpy as np
from typing import Dict, Optional, Tuple
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==== 你项目里的模块 ====
from mamba_decoder import GMSAMDecoderLite
from losses_wavelet import wavelet_detail_loss, wavelet_edge_dice, tv_l1_loss, boundary_focal_bce_loss
from losses_topology import cldice_loss
from losses import boundary_iou_loss_from_probs, uncertainty_weighted_boundary_loss, fourier_mag_loss
from ugbd_refiner import UGBDRefiner
from metrics_eval import Evaluator   # 独立评测
from sdf_head import SDFHead, batch_build_sdf, eikonal_loss, curvature_loss

# ----------------- 数据集 -----------------
IMG_EXTS = [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]

def _clip01(x): return np.clip(x,0.0,1.0)
def _rand(p):   return random.random()<p

def augment_underwater(img_rgb, mask, H, W):
    scale = random.uniform(0.75, 1.25)
    nh, nw = int(H*scale), int(W*scale)
    img = cv2.resize(img_rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
    msk = cv2.resize(mask, (nw,nh), interpolation=cv2.INTER_NEAREST)
    if nh >= H and nw >= W:
        y0 = random.randint(0, nh-H); x0 = random.randint(0, nw-W)
        img = img[y0:y0+H, x0:x0+W]; msk = msk[y0:y0+H, x0:x0+W]
    else:
        canvas = np.zeros((H,W,3), dtype=img.dtype); cm = np.zeros((H,W), dtype=msk.dtype)
        y0 = (H-nh)//2; x0 = (W-nw)//2
        canvas[y0:y0+nh, x0:x0+nw] = img; cm[y0:y0+nh, x0:x0+nw] = msk
        img, msk = canvas, cm
    if _rand(0.5): img = np.ascontiguousarray(img[:, ::-1, :]); msk = np.ascontiguousarray(msk[:, ::-1])
    if _rand(0.8):
        b,g,r = cv2.split(img)
        g = _clip01(g/255.0 * random.uniform(0.9,1.15))
        b = _clip01(b/255.0 * random.uniform(0.95,1.20))
        r = _clip01(r/255.0 * random.uniform(0.85,1.05))
        img = cv2.merge([(b*255).astype(np.uint8),(g*255).astype(np.uint8),(r*255).astype(np.uint8)])
    if _rand(0.8):
        imgf = _clip01(img.astype(np.float32)/255.0)
        imgf = _clip01(imgf*random.uniform(0.85,1.15) + random.uniform(-0.08,0.08))
        img = (imgf*255).astype(np.uint8)
    if _rand(0.3): img = cv2.GaussianBlur(img, random.choice([(3,3),(5,5)]), sigmaX=random.uniform(0.6,1.2))
    if _rand(0.25): img = np.clip(img.astype(np.float32)+np.random.normal(0,8,img.shape),0,255).astype(np.uint8)
    return img, msk

class SmartFolderDataset(Dataset):
    def __init__(self, root, split, img_dir_name="image", label_dir_name="mask",
                 img_ext=None, mask_ext=".png", size=(384,384), use_aug=False):
        super().__init__()
        self.size = size; self.use_aug = use_aug and (split=="train")
        root = os.path.abspath(root)
        sd = os.path.join(root, split)
        img_dir_split = os.path.join(sd, img_dir_name)
        msk_dir_split = os.path.join(sd, label_dir_name)
        img_dir_flat  = os.path.join(root, img_dir_name)
        msk_dir_flat  = os.path.join(root, label_dir_name)
        if os.path.isdir(img_dir_split) and os.path.isdir(msk_dir_split):
            self.img_dir, self.msk_dir = img_dir_split, msk_dir_split
        elif os.path.isdir(img_dir_flat) and os.path.isdir(msk_dir_flat):
            self.img_dir, self.msk_dir = img_dir_flat, msk_dir_flat
        else:
            raise FileNotFoundError(f"Not found:\n{img_dir_split}\n{msk_dir_split}\n{img_dir_flat}\n{msk_dir_flat}")
        self.img_ext, self.mask_ext = img_ext, mask_ext
        self.samples = self._pairs()
        if not self.samples: raise RuntimeError(f"No pairs under {self.img_dir}")
        print(f"[SmartFolderDataset] split='{split}' pairs={len(self.samples)} img_dir={self.img_dir}")

    def _pairs(self):
        img_paths=[]
        if self.img_ext:
            img_paths += glob.glob(os.path.join(self.img_dir, f"*{self.img_ext}"))
            img_paths += glob.glob(os.path.join(self.img_dir, f"*{self.img_ext.upper()}"))
        else:
            for e in IMG_EXTS:
                img_paths += glob.glob(os.path.join(self.img_dir, f"*{e}"))
                img_paths += glob.glob(os.path.join(self.img_dir, f"*{e.upper()}"))
        img_paths = sorted(list(dict.fromkeys(img_paths)))
        pairs=[]
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            if self.mask_ext:
                cand = os.path.join(self.msk_dir, stem+self.mask_ext)
                if os.path.isfile(cand): pairs.append((ip,cand))
            else:
                for e in [".png",".PNG",".jpg",".JPG",".jpeg",".JPEG",".bmp",".BMP"]:
                    cand = os.path.join(self.msk_dir, stem+e)
                    if os.path.isfile(cand):
                        pairs.append((ip,cand)); break
        return pairs

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        img = cv2.imread(ip, cv2.IMREAD_COLOR);  assert img is not None, f"read fail: {ip}"
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE); assert msk is not None, f"read fail: {mp}"
        H,W = self.size
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.use_aug: img_rgb, msk = augment_underwater(img_rgb, msk, H, W)
        else:
            img_rgb = cv2.resize(img_rgb, (W,H), interpolation=cv2.INTER_LINEAR)
            msk     = cv2.resize(msk, (W,H), interpolation=cv2.INTER_NEAREST)
        img_rgb = img_rgb.astype(np.float32)/255.0
        msk_bin = (msk>127).astype(np.float32)
        img_t = torch.from_numpy(np.transpose(img_rgb,(2,0,1)))  # [3,H,W]
        msk_t = torch.from_numpy(msk_bin[None,...])              # [1,H,W]
        return img_t, msk_t, ip

# ----------------- 基础训练件 -----------------
class BCEWithLogits(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__(); self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def forward(self, logits, target): return self.loss(logits, target)

def iou_loss(prob, target, eps=1e-6):
    inter = (prob*target).sum(dim=(1,2,3))
    union = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter + eps
    return (1 - (inter+eps)/union).mean()

def dice_loss(prob, target, eps=1e-6):
    inter = (prob*target).sum(dim=(1,2,3))
    den   = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (1 - (2*inter+eps)/den).mean()

def ssim_loss(prob, target):
    def _local(x,y,k=7):
        pad=k//2
        mx=nn.functional.avg_pool2d(x,k,1,pad); my=nn.functional.avg_pool2d(y,k,1,pad)
        vx=nn.functional.avg_pool2d(x*x,k,1,pad)-mx*mx
        vy=nn.functional.avg_pool2d(y*y,k,1,pad)-my*my
        cxy=nn.functional.avg_pool2d(x*y,k,1,pad)-mx*my
        c1=0.01**2; c2=0.03**2
        return 1 - (((2*mx*my+c1)*(2*cxy+c2))/(((mx*mx+my*my+c1)*(vx+vy+c2))+1e-12))
    return _local(prob,target).mean()

def mae_loss(prob, target):
    """MAE损失：直接优化平均绝对误差"""
    return torch.abs(prob - target).mean()

def focal_mae_loss(prob, target, alpha=0.25, gamma=2.0):
    """
    Focal MAE损失：对困难样本给予更多关注
    - 对误差较大的像素给更高权重，有助于拉低整体MAE
    """
    mae = torch.abs(prob - target)
    weight = torch.pow(mae, gamma)
    return (alpha * weight * mae).mean()

class EMA:
    def __init__(self, model, decay=0.996, name_prefix=""):
        self.decay = decay
        self.shadow = {}
        self.name_prefix = name_prefix
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[f"{self.name_prefix}{n}"] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                key = f"{self.name_prefix}{n}"
                if key not in self.shadow:
                    self.shadow[key] = p.detach().clone()
                else:
                    self.shadow[key].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                key = f"{self.name_prefix}{n}"
                if key in self.shadow:
                    backup[key] = p.detach().clone()
                    p.copy_(self.shadow[key])
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if p.requires_grad:
                key = f"{self.name_prefix}{n}"
                if key in backup:
                    p.copy_(backup[key])

def build_scheduler(optimizer, total_epochs, warmup_epochs=8, sched="cosine", patience=15, factor=0.5, min_lr=1e-6):
    if sched == "plateau":
        # 使用ReduceLROnPlateau，需要手动step
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience,
            verbose=True, min_lr=min_lr
        ), "plateau"
    else:
        def lr_lambda(e):
            if e < warmup_epochs:
                return float(e+1)/float(max(1, warmup_epochs))
            t = e - warmup_epochs
            T = max(1, total_epochs - warmup_epochs)
            return 0.5*(1.0+math.cos(math.pi*t/T)) if sched == "cosine" else 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), "lambda"

# ----------------- 复杂度/参数量 -----------------
class CombinedForProfile(nn.Module):
    def __init__(self, backbone, decoder, refine=None, sdf_head=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.refine = refine
        self.sdf_head = sdf_head

    def forward(self, x):
        logits = self.decoder(x, self.backbone(x))
        if self.refine is not None:
            logits = self.refine(x, logits)
        if self.sdf_head is not None:
            logits, _ = self.sdf_head(x, logits)
        return logits

def count_params(m): 
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train

def profile_complexity(model, input_size, device):
    dummy = torch.randn(*input_size, device=device)
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gmacs = macs / 1e9
        gflops = 2.0 * gmacs  # 近似 FLOPs ~= 2 * MACs
        return gmacs, gflops
    except Exception:
        return None, None

def dump_profile(exp_root, total_params, train_params, gmacs, gflops):
    path = os.path.join(exp_root, "model_profile.txt")
    with open(path, "w") as f:
        f.write(f"Total Params:     {total_params/1e6:.3f} M\n")
        f.write(f"Trainable Params: {train_params/1e6:.3f} M\n")
        if gmacs is not None:
            f.write(f"Complexity:       {gmacs:.3f} GMACs (thop)  ~ {gflops:.3f} GFLOPs (approx.)\n")
        else:
            f.write("Complexity:       N/A (pip install thop to enable)\n")
    print(f"[Model-Profile] {path}")

# ----------------- 训练/验证 -----------------
def train_one_epoch(backbone, decoder, loader, optimizer, device, epoch, args, refine=None, sdf_head=None):
    bce = BCEWithLogits()
    backbone.train(); decoder.train()
    if refine is not None: refine.train()
    total=0.0; pbar=tqdm(loader, desc=f"[Train e{epoch}]")
    for imgs, gts, _ in pbar:
        imgs, gts = imgs.to(device,non_blocking=True), gts.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = decoder(imgs, backbone(imgs))
        if refine is not None:
            logits = refine(imgs, logits)

        sdf_map = None
        if sdf_head is not None:
            logits, sdf_map = sdf_head(imgs, logits)

        prob = torch.sigmoid(logits)

        # ---------------- 动态损失权重（后期微调，更关注MAE，减弱正则） ----------------
        phase = float(epoch) / float(max(1, args.epochs))
        mae_boost = 1.0
        reg_scale = 1.0
        if phase >= getattr(args, "late_ft_start", 1.1):  # 默认>1表示不开启
            t = (phase - args.late_ft_start) / max(1e-6, 1.0 - args.late_ft_start)
            mae_boost = 1.0 + t * (args.late_ft_mae_boost - 1.0)
            reg_scale = 1.0 - t * (1.0 - args.late_ft_reg_decay)
        # MAE/正则分段 ramp（更平滑且可控）
        mae_ramp = 1.0
        if args.mae_ramp_end > args.mae_ramp_start:
            t = (phase - args.mae_ramp_start) / max(1e-6, args.mae_ramp_end - args.mae_ramp_start)
            mae_ramp = float(np.clip(t, 0.0, 1.0))
            mae_ramp = 1.0 + mae_ramp * (args.mae_ramp_max - 1.0)
        reg_ramp = 1.0
        if args.reg_ramp_end > args.reg_ramp_start:
            t = (phase - args.reg_ramp_start) / max(1e-6, args.reg_ramp_end - args.reg_ramp_start)
            reg_ramp = float(np.clip(t, 0.0, 1.0))
            reg_ramp = 1.0 - reg_ramp * (1.0 - args.reg_ramp_min)

        loss = 0.0
        # 主体分类/重叠相关损失
        loss += bce(logits,gts) * args.w_bce
        loss += (dice_loss(prob,gts) if args.use_dice else iou_loss(prob,gts)) * args.w_iou
        loss += ssim_loss(prob,gts) * args.w_ssim

        # 正则 & 细节/边界相关损失（在后期会适当衰减）
        if args.use_wavelet:
            loss += wavelet_detail_loss(prob,gts,levels=args.wav_levels) * args.w_wav * reg_scale * reg_ramp
            loss += wavelet_edge_dice(prob,gts,levels=args.wav_levels)   * args.w_wed * reg_scale * reg_ramp
            loss += tv_l1_loss(prob)                                     * args.w_tv * reg_scale * reg_ramp
            loss += boundary_focal_bce_loss(logits,gts,lam=args.bf_lam)  * args.w_bfce * reg_scale * reg_ramp
        if args.w_cldice>0:
            loss += args.w_cldice * cldice_loss(logits, gts) * reg_scale * reg_ramp
        
        # 边界相关损失（关键：提升Fwβ，同样在后期适当衰减）
        if args.w_biou > 0:
            loss += args.w_biou * boundary_iou_loss_from_probs(prob, gts) * reg_scale * reg_ramp
        if args.w_uwb > 0:
            loss += args.w_uwb * uncertainty_weighted_boundary_loss(prob, gts, lam=args.uwb_lam) * reg_scale * reg_ramp
        if args.w_freq > 0:
            loss += args.w_freq * fourier_mag_loss(prob, gts, log_mag=args.freq_log, highfreq_boost=args.freq_hf_boost) * reg_scale * reg_ramp
        
        # MAE相关损失（直接优化MAE指标，在后期被放大）
        if args.w_mae > 0:
            loss += args.w_mae * mae_loss(prob, gts) * mae_boost * mae_ramp
        if args.w_focal_mae > 0:
            loss += args.w_focal_mae * focal_mae_loss(
                prob, gts,
                alpha=getattr(args, "focal_mae_alpha", 0.25),
                gamma=getattr(args, "focal_mae_gamma", 2.0),
            ) * mae_boost * mae_ramp

        # 多尺度深监督（稳定收敛，提升细粒度一致性）
        if args.use_deep_supervision and args.ds_scales_list:
            for scale in args.ds_scales_list:
                if scale == 1.0:
                    continue
                logits_ds = nn.functional.interpolate(
                    logits, scale_factor=scale, mode="bilinear", align_corners=False
                )
                gts_ds = nn.functional.interpolate(gts, scale_factor=scale, mode="nearest")
                prob_ds = torch.sigmoid(logits_ds)
                ds_loss = bce(logits_ds, gts_ds) * args.ds_w_bce
                ds_loss += (
                    dice_loss(prob_ds, gts_ds) if args.use_dice else iou_loss(prob_ds, gts_ds)
                ) * args.ds_w_iou
                loss += args.ds_weight * ds_loss

        # SDF 形状先验损失（视为一种几何正则，在后期适当衰减）
        if (sdf_head is not None) and (sdf_map is not None):
            with torch.no_grad():
                sdf_gt = batch_build_sdf(gts, scale=args.sdf_scale)
            loss += args.w_sdf * nn.functional.l1_loss(sdf_map, sdf_gt) * reg_scale * reg_ramp
            loss += args.w_eik * eikonal_loss(sdf_map) * reg_scale * reg_ramp
            loss += args.w_curv * curvature_loss(sdf_map) * reg_scale * reg_ramp
        loss.backward()
        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(decoder.parameters()) +
                (list(refine.parameters()) if refine is not None else []) +
                (list(sdf_head.parameters()) if sdf_head is not None else []),
                max_norm=args.grad_clip
            )
        optimizer.step()
        total += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total/len(loader)

@torch.no_grad()
def evaluate_with_module(backbone, decoder, loader, device, args, refine=None, sdf_head=None):
    ev = Evaluator(
        dual_sam_eval=args.dual_sam_eval,
        eval_tta=args.eval_tta,
        tta_scales=args.tta_scales,
        tta_no_flip=args.tta_no_flip,
        vis_dir=args.val_vis_dir
    )
    backbone.eval(); decoder.eval()
    if refine is not None: refine.eval()
    if sdf_head is not None: sdf_head.eval()

    # 组合 refine：先 UGBD，再 SDFHead（仅 eval 时做几何校正）
    if (refine is None) and (sdf_head is None):
        refine_fn = None
    else:
        def refine_fn(imgs, logits):
            if refine is not None:
                logits_ = refine(imgs, logits)
            else:
                logits_ = logits
            if sdf_head is not None:
                logits_, _ = sdf_head(imgs, logits_)
            return logits_

    return ev.evaluate(backbone, decoder, loader, device, refine=refine_fn if (refine is not None or sdf_head is not None) else None)

# ----------------- 主入口 -----------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    # 基础
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="usod10k")
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--img_ext", type=str, default=None)
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--input_h", type=int, default=384)
    parser.add_argument("--input_w", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # DINOv3+DPT
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--dino_size", type=str, default="s", choices=["b","s"])

    # 解码器（Mamba）
    parser.add_argument("--mamba_num_queries", type=int, default=12)
    parser.add_argument("--mamba_dim", type=int, default=192)
    parser.add_argument("--mamba_layers", type=int, default=3)
    parser.add_argument("--mamba_groups", type=int, default=4)
    parser.add_argument("--mamba_kernel", type=int, default=9)
    parser.add_argument("--mamba_mask_embed_dim", type=int, default=64)

    # UGBD 细化头
    parser.add_argument("--use_ugbd", action="store_true")
    parser.add_argument("--ugbd_steps", type=int, default=3)
    parser.add_argument("--ugbd_mid_ch", type=int, default=64)

    # SDF 形状先验头
    parser.add_argument("--use_sdf", action="store_true")
    parser.add_argument("--sdf_mid_ch", type=int, default=64)
    parser.add_argument("--sdf_lambda", type=float, default=0.5, help="logits <- logits - lambda * sdf")
    parser.add_argument("--w_sdf", type=float, default=0.10, help="L1(SDF_pred, SDF_gt)")
    parser.add_argument("--w_eik", type=float, default=0.05, help="Eikonal: |∇SDF|-1")
    parser.add_argument("--w_curv", type=float, default=0.02, help="curvature smooth (Laplacian^2)")
    parser.add_argument("--sdf_scale", type=float, default=1.0, help="GT-SDF 归一后再乘的尺度")

    # 损失权重
    parser.add_argument("--w_bce", type=float, default=1.0)
    parser.add_argument("--w_iou", type=float, default=0.5)
    parser.add_argument("--use_dice", action="store_true")
    parser.add_argument("--w_ssim", type=float, default=0.35)
    # Wavelet/TV/BFCE
    parser.add_argument("--use_wavelet", action="store_true")
    parser.add_argument("--wav_levels", type=int, default=3)
    parser.add_argument("--w_wav",  type=float, default=0.30)
    parser.add_argument("--w_wed",  type=float, default=0.10)
    parser.add_argument("--w_tv",   type=float, default=0.06)
    parser.add_argument("--w_bfce", type=float, default=0.12)
    parser.add_argument("--bf_lam", type=float, default=1.0)
    # clDice
    parser.add_argument("--w_cldice", type=float, default=0.0)
    # 边界相关损失（新增）
    parser.add_argument("--w_biou", type=float, default=0.0, help="Boundary IoU loss weight")
    parser.add_argument("--w_uwb", type=float, default=0.0, help="Uncertainty-Weighted Boundary loss weight")
    parser.add_argument("--uwb_lam", type=float, default=2.0, help="UWB lambda parameter")
    parser.add_argument("--w_freq", type=float, default=0.0, help="Fourier magnitude loss weight")
    parser.add_argument("--freq_log", action="store_true", help="Use log magnitude in Fourier loss")
    parser.add_argument("--freq_hf_boost", type=float, default=0.2, help="High frequency boost in Fourier loss")
    # MAE相关损失（优化MAE指标）
    parser.add_argument("--w_mae", type=float, default=0.0, help="MAE loss weight (direct optimization)")
    parser.add_argument("--w_focal_mae", type=float, default=0.0, help="Focal MAE loss weight")
    parser.add_argument("--focal_mae_alpha", type=float, default=0.25, help="Focal MAE alpha")
    parser.add_argument("--focal_mae_gamma", type=float, default=2.0, help="Focal MAE gamma")
    # 分阶段微调（后期更关注MAE，减弱正则）
    # 注意：这里默认策略较“温和”，避免过度牺牲泛化，只做小幅MAE精修
    parser.add_argument("--late_ft_start", type=float, default=0.8,
                        help="进入后期微调阶段的起始比例，例如0.8表示80%% epoch之后")
    parser.add_argument("--late_ft_mae_boost", type=float, default=1.2,
                        help="后期阶段对MAE类损失的放大倍数")
    parser.add_argument("--late_ft_reg_decay", type=float, default=0.5,
                        help="后期阶段对正则/边界类损失的小数倍缩放系数")
    # 更平滑的 MAE/正则 ramp（用于长训练，避免后期震荡）
    parser.add_argument("--mae_ramp_start", type=float, default=1.1,
                        help="MAE ramp 开始比例（>1表示关闭）")
    parser.add_argument("--mae_ramp_end", type=float, default=1.1,
                        help="MAE ramp 结束比例（>1表示关闭）")
    parser.add_argument("--mae_ramp_max", type=float, default=1.0,
                        help="MAE ramp 最大放大倍数")
    parser.add_argument("--reg_ramp_start", type=float, default=1.1,
                        help="正则 ramp 开始比例（>1表示关闭）")
    parser.add_argument("--reg_ramp_end", type=float, default=1.1,
                        help="正则 ramp 结束比例（>1表示关闭）")
    parser.add_argument("--reg_ramp_min", type=float, default=1.0,
                        help="正则 ramp 最小缩放系数")

    # 数据增强
    parser.add_argument("--use_aug", action="store_true", help="启用训练时数据增强")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值，0表示禁用")
    # 多尺度深监督
    parser.add_argument("--use_deep_supervision", action="store_true",
                        help="Enable multi-scale deep supervision loss")
    parser.add_argument("--ds_scales", type=str, default="0.5,0.25",
                        help="Comma-separated scales for deep supervision")
    parser.add_argument("--ds_weight", type=float, default=0.3,
                        help="Overall deep supervision loss weight")
    parser.add_argument("--ds_w_bce", type=float, default=1.0,
                        help="BCE weight for deep supervision")
    parser.add_argument("--ds_w_iou", type=float, default=0.5,
                        help="IoU/Dice weight for deep supervision")

    # 评测
    parser.add_argument("--dual_sam_eval", action="store_true")
    parser.add_argument("--eval_tta", action="store_true")
    parser.add_argument("--tta_scales", type=str, default="1.0,0.75,1.25")
    parser.add_argument("--tta_no_flip", action="store_true")
    parser.add_argument("--val_vis_dir", type=str, default=None)

    # 训练器
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine","constant","plateau"])
    parser.add_argument("--warmup_epochs", type=int, default=8)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--use_ema_eval", action="store_true", help="Eval with EMA weights for stability")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0,
                        help="Freeze backbone for first N epochs to stabilize decoder training")
    parser.add_argument("--use_two_stage", action="store_true",
                        help="Enable two-stage training with backbone freeze warmup")
    parser.add_argument("--stage1_ratio", type=float, default=0.0,
                        help="Stage1 ratio for backbone freeze when --use_two_stage is set (0-1)")
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1,
                        help="Backbone learning rate multiplier relative to --lr")
    parser.add_argument("--mae_collapse_tol", type=float, default=0.35,
                        help="MAE collapse tolerance ratio to trigger recovery (e.g. 0.35 = 35%% worse)")
    parser.add_argument("--mae_collapse_lr_factor", type=float, default=0.5,
                        help="LR decay factor applied when MAE collapse is detected")
    parser.add_argument("--mae_collapse_restore", action="store_true",
                        help="Restore best checkpoint weights when MAE collapse is detected")
    # 早停和学习率调度
    parser.add_argument("--early_stop", type=int, default=0, help="早停轮数，0表示禁用")
    parser.add_argument("--patience", type=int, default=15, help="ReduceLROnPlateau的patience")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="学习率衰减因子")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="最小学习率")
    # 保存策略
    parser.add_argument("--save_every_epoch", action="store_true", help="保存每个epoch的权重")

    # 存储
    parser.add_argument("--save_root", type=str, default="./runs_usod10k_wavelet")
    parser.add_argument("--best_metric", type=str, default="Fw_beta",
                        choices=["mIoU","S_alpha","Fw_beta","mE_phi","MAE","maxF"])
    args = parser.parse_args()
    if args.use_deep_supervision:
        args.ds_scales_list = [
            float(s.strip()) for s in args.ds_scales.split(",") if s.strip()
        ]
    else:
        args.ds_scales_list = []

    # 随机性
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 路径
    tag = "mamba"
    if args.use_wavelet: tag += "_wav"
    if args.use_ugbd:    tag += "_ugbd"
    exp_root = os.path.join(args.save_root, f"segdino_{args.dino_size}_{args.input_h}x{args.input_w}_{args.dataset}_{tag}")
    os.makedirs(exp_root, exist_ok=True)
    ckpt_dir = os.path.join(exp_root, "ckpts"); os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(exp_root, "metrics_history.csv")

    # Backbone
    if args.dino_size=="b":
        vit = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        vit = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
    from dpt import DPT
    backbone = DPT(nclass=1, backbone=vit).to(device)

    # Decoder
    decoder = GMSAMDecoderLite(
        num_queries=args.mamba_num_queries, token_dim=args.mamba_dim,
        num_layers=args.mamba_layers, groups=args.mamba_groups, kernel_size=args.mamba_kernel,
        mask_embed_dim=args.mamba_mask_embed_dim
    ).to(device)

    # Refiner
    refine = UGBDRefiner(mid_ch=args.ugbd_mid_ch, steps=args.ugbd_steps).to(device) if args.use_ugbd else None

    # SDF 形状先验头
    sdf_head = SDFHead(mid_ch=args.sdf_mid_ch, fuse_lambda=args.sdf_lambda).to(device) if args.use_sdf else None

    # ---- 模型规模 & 复杂度（打印 + 落盘） ----
    prof_model = CombinedForProfile(backbone, decoder, refine, sdf_head).to(device).eval()
    total_p, train_p = count_params(prof_model)
    gmacs, gflops = profile_complexity(prof_model, (1,3,args.input_h,args.input_w), device)
    print(f"[Model] Total Params: {total_p/1e6:.3f}M | Trainable: {train_p/1e6:.3f}M")
    if gmacs is not None:
        print(f"[Model] Complexity: {gmacs:.3f} GMACs (thop)  ~ {gflops:.3f} GFLOPs (approx.)")
    else:
        print("[Model] Complexity: N/A  (pip install thop to enable)")
    dump_profile(exp_root, total_p, train_p, gmacs, gflops)

    # 优化器 & 调度 & EMA
    optimizer_groups = [
        {"params": backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
        {"params": decoder.parameters(), "lr": args.lr},
    ]
    if refine is not None:
        optimizer_groups.append({"params": refine.parameters(), "lr": args.lr})
    if sdf_head is not None:
        optimizer_groups.append({"params": sdf_head.parameters(), "lr": args.lr})
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in group["params"] if p.requires_grad], "lr": group["lr"]}
            for group in optimizer_groups
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler, sched_type = build_scheduler(
        optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs, 
        sched=args.scheduler, patience=args.patience, factor=args.lr_factor, min_lr=args.min_lr
    )
    ema_backbone = EMA(backbone, decay=args.ema_decay, name_prefix="backbone.") if args.use_ema else None
    ema_decoder = EMA(decoder, decay=args.ema_decay, name_prefix="decoder.") if args.use_ema else None
    ema_refine = EMA(refine, decay=args.ema_decay, name_prefix="refine.") if (args.use_ema and refine is not None) else None
    ema_sdf = EMA(sdf_head, decay=args.ema_decay, name_prefix="sdf_head.") if (args.use_ema and sdf_head is not None) else None

    # 数据
    H,W = args.input_h, args.input_w
    train_root = os.path.join(args.data_dir, args.dataset)
    
    # 验证集：优先使用独立的 val 目录（如 usod10k_val），如果没有则尝试 train_root 下的 val 子目录，
    # 再不行则回退到 *_test 或 train_root 本身（向后兼容旧数据结构）
    val_root = os.path.join(args.data_dir, args.dataset.rstrip("/") + "_val")
    val_split = "val"
    if not os.path.isdir(val_root):
        # 尝试 train_root 下的 val 子目录
        candidate = os.path.join(train_root, "val")
        if os.path.isdir(candidate):
            val_root = train_root
            val_split = "val"
        else:
            # 如果都没有，使用 *_test 或 train_root 作为验证集（兼容旧版）
            alt_val_root = os.path.join(args.data_dir, args.dataset.rstrip("/") + "_test")
            if os.path.isdir(alt_val_root):
                val_root = alt_val_root
                val_split = "test"
            else:
                val_root = train_root
                val_split = "test"
    
    train_set = SmartFolderDataset(train_root, "train", args.img_dir_name, args.label_dir_name, args.img_ext, args.mask_ext, (H,W), use_aug=args.use_aug)
    # 验证集仅用于每轮评估，不参与训练
    val_set   = SmartFolderDataset(val_root, val_split, args.img_dir_name, args.label_dir_name, args.img_ext, args.mask_ext, (H,W), use_aug=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=max(1,args.num_workers//2), drop_last=False, pin_memory=True)
    print(f"[Dataset] Train: {len(train_set)} samples, Val: {len(val_set)} samples (from '{val_root}', split='{val_split}')")

    # CSV 头
    if not os.path.exists(csv_path):
        with open(csv_path,"w",newline="") as f:
            csv.writer(f).writerow(["epoch","lr","train_loss",
                                    "val_loss","mIoU","S_alpha","Fw_beta","mE_phi","E_phi_adp","MAE","maxF","best_ckpt"])

    # best 记录
    best = {"mIoU":(-1,-1),"S_alpha":(-1,-1),"Fw_beta":(-1,-1),"mE_phi":(-1,-1),"MAE":(1e9,-1),"maxF":(-1,-1)}
    best_ckpt=""; 
    def better(new, old, key): return (new<old) if key=="MAE" else (new>old)
    best_state = None

    # 早停相关
    no_improve_count = 0
    best_metric_value = -1e9 if args.best_metric != "MAE" else 1e9

    # 训练循环
    freeze_epochs = args.freeze_backbone_epochs
    if args.use_two_stage and freeze_epochs <= 0 and args.stage1_ratio > 0:
        freeze_epochs = max(1, int(round(args.epochs * args.stage1_ratio)))
    for e in range(1, args.epochs+1):
        if args.freeze_backbone_epochs > 0:
            freeze_backbone = e <= args.freeze_backbone_epochs
        elif freeze_epochs > 0:
            freeze_backbone = e <= freeze_epochs
        else:
            freeze_backbone = False
        if freeze_epochs > 0 and e == 1:
            print(f"[Two-Stage] Freeze backbone for {freeze_epochs} epochs.")
        if freeze_epochs > 0 and e == freeze_epochs + 1:
            print("[Two-Stage] Unfreeze backbone for full finetune.")
        for p in backbone.parameters():
            p.requires_grad = not freeze_backbone
        tloss = train_one_epoch(backbone, decoder, train_loader, optimizer, device, e, args, refine=refine, sdf_head=sdf_head)
        if args.use_ema:
            ema_backbone.update(backbone)
            ema_decoder.update(decoder)
            if ema_refine is not None:
                ema_refine.update(refine)
            if ema_sdf is not None:
                ema_sdf.update(sdf_head)

        if args.use_ema_eval and args.use_ema:
            bk = ema_backbone.apply_shadow(backbone)
            dk = ema_decoder.apply_shadow(decoder)
            rk = ema_refine.apply_shadow(refine) if ema_refine is not None else None
            sk = ema_sdf.apply_shadow(sdf_head) if ema_sdf is not None else None
            val = evaluate_with_module(backbone, decoder, val_loader, device, args, refine=refine, sdf_head=sdf_head)
            ema_backbone.restore(backbone, bk)
            ema_decoder.restore(decoder, dk)
            if ema_refine is not None and rk is not None:
                ema_refine.restore(refine, rk)
            if ema_sdf is not None and sk is not None:
                ema_sdf.restore(sdf_head, sk)
        else:
            val = evaluate_with_module(backbone, decoder, val_loader, device, args, refine=refine, sdf_head=sdf_head)

        state = {
            "epoch": e,
            "backbone": backbone.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # 保存scheduler状态（如果支持）
        if hasattr(scheduler, 'state_dict'):
            try:
                state["scheduler"] = scheduler.state_dict()
            except:
                pass
        if refine is not None:
            state["refine"] = refine.state_dict()
        if sdf_head is not None:
            state["sdf_head"] = sdf_head.state_dict()
        
        # 保存latest
        torch.save(state, os.path.join(ckpt_dir,"latest.pth"))
        
        # 保存每个epoch的权重（如果启用）
        if args.save_every_epoch:
            epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{e:03d}.pth")
            torch.save(state, epoch_ckpt)

        key=args.best_metric; val_key = val[key]
        if better(val_key, best[key][0], key):
            best[key]=(val_key,e)
            best_ckpt=os.path.join(ckpt_dir, f"best_{key}_ep{e:03d}_{val_key:.4f}.pth")
            torch.save(state, best_ckpt)
            print(f"[Save] New best ({key}): {best_ckpt}")
            no_improve_count = 0  # 重置早停计数
            best_state = {
                "backbone": backbone.state_dict(),
                "decoder": decoder.state_dict(),
                "refine": refine.state_dict() if refine is not None else None,
                "sdf_head": sdf_head.state_dict() if sdf_head is not None else None,
                "optimizer": optimizer.state_dict(),
            }
        else:
            no_improve_count += 1

        for k in ["mIoU","S_alpha","Fw_beta","mE_phi","maxF"]:
            if val[k]>best[k][0]: best[k]=(val[k],e)
        if val["MAE"]<best["MAE"][0]: best["MAE"]=(val["MAE"],e)

        # 学习率调度
        if sched_type == "plateau":
            # ReduceLROnPlateau需要传入验证指标
            scheduler.step(val[args.best_metric])
        else:
            scheduler.step()

        with open(csv_path,"a",newline="") as f:
            csv.writer(f).writerow([e, f"{optimizer.param_groups[0]['lr']:.6e}", f"{tloss:.6f}",
                                    f"{val['val_loss']:.6f}", f"{val['mIoU']:.6f}", f"{val['S_alpha']:.6f}",
                                    f"{val['Fw_beta']:.6f}", f"{val['mE_phi']:.6f}", f"{val['E_phi_adp']:.6f}",
                                    f"{val['MAE']:.6f}", f"{val['maxF']:.6f}", best_ckpt])

        # MAE collapse recovery: restore best and reduce LR if MAE worsens sharply
        if args.mae_collapse_tol > 0 and best["MAE"][0] < 1e8:
            collapse_threshold = best["MAE"][0] * (1.0 + args.mae_collapse_tol)
            if val["MAE"] > collapse_threshold:
                print(f"[MAE Collapse] val MAE={val['MAE']:.4f} > {collapse_threshold:.4f}.")
                if args.mae_collapse_restore and best_state is not None:
                    backbone.load_state_dict(best_state["backbone"])
                    decoder.load_state_dict(best_state["decoder"])
                    if refine is not None and best_state["refine"] is not None:
                        refine.load_state_dict(best_state["refine"])
                    if sdf_head is not None and best_state["sdf_head"] is not None:
                        sdf_head.load_state_dict(best_state["sdf_head"])
                    optimizer.load_state_dict(best_state["optimizer"])
                    print("[MAE Collapse] Restored best checkpoint weights.")
                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"] * args.mae_collapse_lr_factor, args.min_lr)
                print(f"[MAE Collapse] Reduce LR by factor {args.mae_collapse_lr_factor}.")
        
        # 早停检查
        if args.early_stop > 0 and no_improve_count >= args.early_stop:
            print(f"[Early Stop] No improvement for {args.early_stop} epochs. Stopping training.")
            break

    print("="*66)
    print("[Summary] Best over all epochs (Dual-SAM metrics):")
    print(f"  mIoU     = {best['mIoU'][0]:.4f}  @ epoch {best['mIoU'][1]}")
    print(f"  Sα       = {best['S_alpha'][0]:.4f} @ epoch {best['S_alpha'][1]}")
    print(f"  Fwβ      = {best['Fw_beta'][0]:.4f} @ epoch {best['Fw_beta'][1]}")
    print(f"  mEϕ      = {best['mE_phi'][0]:.4f}  @ epoch {best['mE_phi'][1]}")
    print(f"  MAE      = {best['MAE'][0]:.4f}   @ epoch {best['MAE'][1]}")
    print(f"  maxF     = {best['maxF'][0]:.4f}  @ epoch {best['maxF'][1]}")
    print("="*66)
    print(f"[CSV] {csv_path}")
    if best_ckpt: print(f"[Best-CKPT] {best_ckpt}")

if __name__ == "__main__":
    main()
