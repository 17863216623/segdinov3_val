# metrics_eval.py
# -*- coding: utf-8 -*-
import os, math, numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# ───────────── SciPy（Fwβ 用） ─────────────
_HAS_SCIPY = True
try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
except Exception:
    _HAS_SCIPY = False

# ───────────── 可选可视化 ─────────────
def _tensor_to_rgb(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().cpu().float().clamp(0,1).numpy()
    x = (x*255).round().astype(np.uint8)
    x = np.transpose(x, (1,2,0))
    import cv2; return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def _mask_to_gray(t: torch.Tensor, thr: float=0.5) -> np.ndarray:
    m = t.detach().cpu().float()
    if m.ndim==3 and m.shape[0]==1: m = m[0]
    elif m.ndim==2: pass
    else: m = m.squeeze()
    if (m.max()>1) or (m.min()<0): m = torch.sigmoid(m)
    m = (m>thr).float()
    return (m*255).round().byte().cpu().numpy()

def _save_triplet(idx, img, logit, gt, out_dir, prefix="val"):
    import cv2, os
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{prefix}_{idx:05d}")
    cv2.imwrite(base+"_img.png",  _tensor_to_rgb(img))
    cv2.imwrite(base+"_pred.png", _mask_to_gray(logit, 0.5))
    cv2.imwrite(base+"_gt.png",   _mask_to_gray(gt,    0.5))

# ───────────── 基础指标 ─────────────
@torch.no_grad()
def mae(prob, gt):  # prob, gt: B×1×H×W
    return torch.abs(prob-gt).mean(dim=(1,2,3))

@torch.no_grad()
def mIoU_at_thr(prob, gt, thr=0.5, eps=1e-6):
    pred = (prob>thr).float(); g = (gt>0.5).float()
    inter = (pred*g).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + g.sum(dim=(1,2,3)) - inter + eps
    return (inter+eps)/union

def _ssim(x, y, c1=0.01**2, c2=0.03**2):
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(unbiased=False), y.var(unbiased=False)
    cxy = ((x-mx)*(y-my)).mean()
    num = (2*mx*my+c1)*(2*cxy+c2)
    den = (mx**2+my**2+c1)*(vx+vy+c2)
    return num/(den+1e-12)

@torch.no_grad()
def S_measure(prob, gt, alpha=0.5, lam=0.5):
    B = prob.size(0); out=[]
    for b in range(B):
        P = prob[b,0]; G = (gt[b,0]>0.5).float()
        if G.sum()==0: out.append(1.0 - P.mean()); continue
        if G.sum()==G.numel(): out.append(P.mean()); continue
        mu = G.mean()
        Fg, Bg = P[G==1], P[G==0]
        if Fg.numel()==0 or Bg.numel()==0:
            So = P.mean()
        else:
            mf, sf = Fg.mean(), Fg.std(unbiased=False)
            mb, sb = Bg.mean(), Bg.std(unbiased=False)
            OFG = (2*mf)/(mf**2 + 1 + 2*lam*(sf+1e-12))
            OBG = (2*(1-mb))/(((1-mb)**2) + 1 + 2*lam*(sb+1e-12))
            So  = mu*OFG + (1-mu)*OBG
        H,W = G.shape
        ys,xs = torch.where(G>0)
        cy = int(torch.round((ys.float().mean() if ys.numel() else torch.tensor(H/2))).item())
        cx = int(torch.round((xs.float().mean() if xs.numel() else torch.tensor(W/2))).item())
        regs=[(slice(0,cy),slice(0,cx)),(slice(0,cy),slice(cx,W)),(slice(cy,H),slice(0,cx)),(slice(cy,H),slice(cx,W))]
        wsum, ssum = 0.0, 0.0
        for ry,rx in regs:
            Gy,Py = G[ry,rx], P[ry,rx]
            area = float(Gy.numel()); 
            if area<1: continue
            ssum += _ssim(Py, Gy)*area; wsum += area
        Sr = ssum/(wsum+1e-12)
        out.append(alpha*So + (1-alpha)*Sr)
    return torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in out])

@torch.no_grad()
def _E_single(bin_pred, gt):
    gm, pm = gt.mean(), bin_pred.mean()
    align = (2*(bin_pred - pm)*(gt - gm))/(((bin_pred - pm)**2 + (gt - gm)**2)+1e-12)
    return ((align + 1)**2/4).mean()

@torch.no_grad()
def E_measures(prob, gt, num_thr=255):
    B = prob.size(0); mean_E, adp_E = [], []
    thresholds = torch.linspace(0,1,steps=num_thr,device=prob.device)
    for b in range(B):
        P = prob[b,0]; G = (gt[b,0]>0.5).float()
        Es = [ _E_single((P>t).float(), G) for t in thresholds ]
        mean_E.append(torch.stack(Es).mean())
        t_adp = (2*P.mean()).clamp(0,1)
        adp_E.append(_E_single((P>t_adp).float(), G))
    return torch.stack(mean_E), torch.stack(adp_E)

# ───────────── Fwβ（需要 SciPy） ─────────────
def _weighted_fbeta_numpy(candidate, gt, beta=0.3):
    if not _HAS_SCIPY: return float("nan")
    eps = np.finfo(np.float32).eps
    cand = candidate.astype(np.float32); gt_b = (gt>0.5).astype(np.float32)
    E = np.abs(cand - gt_b)
    not_gt = (gt_b < 0.5)
    dist, idx = distance_transform_edt(not_gt, return_indices=True)
    Et = E.copy(); Et[not_gt] = E[idx[0,not_gt], idx[1,not_gt]]
    sigma = 5.0
    EA = gaussian_filter(Et, sigma=sigma, truncate=3/sigma, mode="constant", cval=0.0)
    min_E_EA = np.where(gt_b>0.5, np.minimum(E,EA), E)
    B = np.ones_like(gt_b, dtype=np.float32)
    B[not_gt] = 2 - np.exp(np.log(1-0.5)/5 * dist[not_gt])
    Ew = min_E_EA * B
    TPw = gt_b.sum() - Ew[gt_b>0.5].sum()
    FPw = Ew[not_gt].sum()
    R = 1 - (Ew[gt_b>0.5].mean() if gt_b.sum()>0 else 0.0)
    P = TPw / (eps + TPw + FPw)
    beta2 = beta**2
    Q = (1 + beta2) * (R * P) / (eps + R + beta2 * P)
    return float(Q)

@torch.no_grad()
def Fw_beta(prob, gt, beta=0.3):
    vals=[]
    for b in range(prob.size(0)):
        P = prob[b,0].detach().cpu().clamp(0,1).numpy()
        G = (gt[b,0].detach().cpu().numpy()>0.5).astype(np.uint8)
        vals.append(_weighted_fbeta_numpy(P,G,beta=beta))
    return torch.tensor(vals, dtype=torch.float32, device=prob.device)

# ───────────── maxF（Fβ@best-threshold） ─────────────
def _fmax_single(prob, gt, beta2=0.3, n_th=255, eps=1e-8):
    P = prob.view(-1); G = (gt>0.5).float().view(-1)
    if G.sum()<1:  # 空目标
        return torch.tensor(1.0 if P.max()<=0.5 else 0.0, device=prob.device, dtype=prob.dtype)
    ths = torch.linspace(0,1,steps=n_th, device=prob.device)
    best = torch.tensor(0.0, device=prob.device)
    for t in ths:
        pred = (P>=t).float()
        tp = (pred*G).sum(); pp = pred.sum(); gg = G.sum()
        prec = tp/(pp+eps); rec = tp/(gg+eps)
        f = (1+beta2)*prec*rec/(beta2*prec + rec + eps)
        best = torch.maximum(best, f)
    return best

@torch.no_grad()
def maxF(prob, gt, beta2=0.3):
    out=[]
    for b in range(prob.size(0)):
        out.append(_fmax_single(prob[b,0], gt[b,0], beta2=beta2))
    return torch.stack(out)

# ───────────── TTA forward ─────────────
@torch.no_grad()
def forward_with_tta(backbone, decoder, x, scales=(1.0,0.75,1.25), do_flip=True):
    outs=[]; B,C,H,W = x.shape
    for s in scales:
        xi = x if s==1.0 else F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        for flip in ([False, True] if do_flip else [False]):
            xif = torch.flip(xi, dims=[3]) if flip else xi
            logits = decoder(xif, backbone(xif))
            if logits.shape[-2:] != (H,W):
                logits = F.interpolate(logits, size=(H,W), mode="bilinear", align_corners=False)
            if flip: logits = torch.flip(logits, dims=[3])
            outs.append(torch.sigmoid(logits))
    prob = torch.mean(torch.stack(outs, dim=0), dim=0)
    return torch.logit(prob.clamp(1e-6, 1-1e-6))

# ───────────── PySODMetrics 支持 ─────────────
_HAS_PYSOD = True
try:
    from py_sod_metrics import (
        MAE, Emeasure, Smeasure, Fmeasure, WeightedFmeasure,
        FmeasureV2, FmeasureHandler, IOUHandler
    )
except ImportError:
    _HAS_PYSOD = False
    print("[Warn] PySODMetrics 未安装，将使用自定义实现。安装: pip install pysodmetrics")

def _tensor_to_uint8_numpy(t: torch.Tensor) -> np.ndarray:
    """将 torch tensor 转换为 uint8 numpy array [0,255]，确保输出为 2D (HxW)"""
    arr = t.detach().cpu().float()
    
    # 处理不同维度，最终得到 2D (HxW)
    if arr.ndim == 4:
        # Bx1xHxW 或 BxCxHxW
        if arr.shape[1] == 1:
            arr = arr.squeeze(1)  # BxHxW
        arr = arr[0]  # 取第一个样本，得到 HxW
    elif arr.ndim == 3:
        # 1xHxW 或 BxHxW
        arr = arr[0]  # 取第一个，得到 HxW
    elif arr.ndim == 2:
        # 已经是 HxW
        pass
    else:
        # 其他情况，尝试 squeeze 然后取第一个
        arr = arr.squeeze()
        if arr.ndim > 2:
            arr = arr[0]
        elif arr.ndim < 2:
            raise ValueError(f"Cannot convert {t.shape} to 2D array")
    
    # 确保是 2D
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D with shape {arr.shape} after processing {t.shape}")
    
    # 归一化到 [0,1] 再转 [0,255]
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = torch.sigmoid(arr) if arr.max() > 1.0 else arr.clamp(0, 1)
    arr = (arr * 255.0).round().clamp(0, 255).byte().numpy()
    return arr.astype(np.uint8)

# ───────────── 评估器（使用 PySODMetrics） ─────────────
class Evaluator:
    def __init__(self, dual_sam_eval=False, eval_tta=False, tta_scales="1.0,0.75,1.25", tta_no_flip=False, vis_dir=None):
        self.dual = dual_sam_eval
        self.eval_tta = eval_tta
        self.scales = tuple(float(x) for x in tta_scales.split(",") if x)
        self.do_flip = not tta_no_flip
        self.vis_dir = vis_dir
        
        self.use_pysod = _HAS_PYSOD

    @torch.no_grad()
    def evaluate(self, backbone, decoder, loader, device, refine=None) -> Dict[str, float]:
        import numpy as np
        bce = torch.nn.BCEWithLogitsLoss()
        losses = []
        idx = 0
        
        if self.use_pysod:
            # 初始化 PySODMetrics 收集器（每次评估时重置）
            mae_metric = MAE()
            em_metric = Emeasure()
            sm_metric = Smeasure()
            fm_metric = Fmeasure()
            wfm_metric = WeightedFmeasure()
            fmv2 = FmeasureV2(
                metric_handlers={
                    "fm": FmeasureHandler(beta=0.3, with_adaptive=True, with_dynamic=True),
                    "iou": IOUHandler(with_adaptive=True, with_dynamic=True),
                }
            )
            # 使用 PySODMetrics 标准评估
            for imgs, gts, _ in loader:
                imgs, gts = imgs.to(device, non_blocking=True), gts.to(device, non_blocking=True)
                if self.eval_tta:
                    logits = forward_with_tta(backbone, decoder, imgs, scales=self.scales, do_flip=self.do_flip)
                else:
                    logits = decoder(imgs, backbone(imgs))
                if refine is not None: logits = refine(imgs, logits)
                prob = torch.sigmoid(logits)
                
                losses.append(bce(logits, gts).item())
                
                # 转换为 uint8 numpy 格式
                prob4eval = (prob > 0.5).float() if self.dual else prob
                B = prob4eval.size(0)
                for b in range(B):
                    # 直接取单个样本，避免维度问题
                    pred_single = prob4eval[b, 0] if prob4eval.ndim == 4 else prob4eval[b]
                    gt_single = gts[b, 0] if gts.ndim == 4 else gts[b]
                    
                    pred_np = _tensor_to_uint8_numpy(pred_single)
                    gt_np = _tensor_to_uint8_numpy(gt_single)
                    
                    # 确保是 2D 且尺寸一致
                    if pred_np.ndim != 2 or gt_np.ndim != 2:
                        raise ValueError(f"Expected 2D arrays, got pred: {pred_np.shape}, gt: {gt_np.shape}")
                    
                    if pred_np.shape != gt_np.shape:
                        import cv2
                        pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                    # 更新指标
                    mae_metric.step(pred_np, gt_np)
                    em_metric.step(pred_np, gt_np)
                    sm_metric.step(pred_np, gt_np)
                    fm_metric.step(pred_np, gt_np)
                    wfm_metric.step(pred_np, gt_np)
                    fmv2.step(pred_np, gt_np)
                
                if self.vis_dir: _save_triplet(idx, imgs[0], logits[0], gts[0], self.vis_dir, "val")
                idx += 1
            
            # 获取 PySODMetrics 结果
            mae_res = mae_metric.get_results()["mae"]
            sm_res = sm_metric.get_results()["sm"]
            wfm_res = wfm_metric.get_results()["wfm"]
            em_res = em_metric.get_results()["em"]
            fmv2_res = fmv2.get_results()
            
            res = {
                "val_loss": float(np.mean(losses)),
                "MAE": round(float(mae_res), 4),
                "S_alpha": round(float(sm_res), 4),
                "Fw_beta": round(float(wfm_res), 4),
                "mE_phi": round(float(em_res["curve"].mean()), 4),
                "E_phi_adp": round(float(em_res["adp"]), 4),
                "maxF": round(float(fmv2_res["fm"]["dynamic"].max()), 4),  # 使用 FmeasureV2 的 maxF
                "mIoU": round(float(fmv2_res["iou"]["dynamic"].mean()), 4),
            }
            # 额外指标
            res["maxE_phi"] = round(float(em_res["curve"].max()), 4)
            res["avgF"] = round(float(fmv2_res["fm"]["dynamic"].mean()), 4)
            res["adpF"] = round(float(fmv2_res["fm"]["adaptive"]), 4)
            res["maxIoU"] = round(float(fmv2_res["iou"]["dynamic"].max()), 4)
            res["adpIoU"] = round(float(fmv2_res["iou"]["adaptive"]), 4)
            
        else:
            # 回退到自定义实现
            mIoUs=[]; maes=[]; Ss=[]; Emes=[]; Eads=[]; Fws=[]; Fmx=[]
            for imgs, gts, _ in loader:
                imgs, gts = imgs.to(device,non_blocking=True), gts.to(device,non_blocking=True)
                if self.eval_tta:
                    logits = forward_with_tta(backbone, decoder, imgs, scales=self.scales, do_flip=self.do_flip)
                else:
                    logits = decoder(imgs, backbone(imgs))
                if refine is not None: logits = refine(imgs, logits)
                prob = torch.sigmoid(logits)

                losses.append(bce(logits, gts).item())
                prob4eval = (prob>0.5).float() if self.dual else prob
                mIoUs.append(mIoU_at_thr(prob4eval, gts).cpu().numpy())
                maes .append(mae(prob4eval, gts).cpu().numpy())
                Ss   .append(S_measure(prob4eval, gts).cpu().numpy())
                Em, Ea = E_measures(prob4eval, gts)
                Emes.append(Em.cpu().numpy()); Eads.append(Ea.cpu().numpy())
                Fws  .append(Fw_beta(prob4eval, gts).cpu().numpy())
                Fmx.append(maxF(prob4eval, gts).cpu().numpy())

                if self.vis_dir: _save_triplet(idx, imgs[0], logits[0], gts[0], self.vis_dir, "val")
                idx += 1

            def cat_mean(vs): return float(np.concatenate(vs,axis=0).mean()) if vs else 0.0
            res = {
                "val_loss": float(np.mean(losses)),
                "mIoU":     cat_mean(mIoUs),
                "S_alpha":  cat_mean(Ss),
                "mE_phi":   cat_mean(Emes),
                "E_phi_adp":cat_mean(Eads),
                "Fw_beta":  cat_mean(Fws),
                "MAE":      cat_mean(maes),
                "maxF":     cat_mean(Fmx),
            }
        
        print(f"[Eval] loss={res['val_loss']:.4f} | mIoU={res.get('mIoU', res.get('mIoU', 0)):.4f} | Sα={res['S_alpha']:.4f} | Fwβ={res['Fw_beta']:.4f} | mEϕ={res['mE_phi']:.4f} | MAE={res['MAE']:.4f} | maxF={res['maxF']:.4f}")
        if not _HAS_PYSOD:
            print("[Warn] 使用自定义实现，建议安装 PySODMetrics: pip install pysodmetrics")
        return res
