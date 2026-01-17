import torch


def fmax_torch(prob, gt, beta2=0.3, n_th=255, eps=1e-8):
    """
    prob: 1xHxW in [0,1] ; gt: 1xHxW in {0,1}
    返回单图像 F_beta 的最大值（跨阈值）
    """
    P = prob.view(-1)
    G = (gt > 0.5).float().view(-1)
    # 空目标的边界情形，定义 F=1 当 P 全 0，否则 0
    if G.sum() < 1:
        return torch.tensor(
            1.0 if P.max() <= 0.5 else 0.0,
            device=prob.device,
            dtype=prob.dtype,
        )

    ths = torch.linspace(0.0, 1.0, n_th, device=prob.device)
    best = torch.tensor(0.0, device=prob.device, dtype=prob.dtype)
    for t in ths:
        pred = (P >= t).float()
        tp = (pred * G).sum()
        pp = pred.sum()
        gg = G.sum()

        prec = tp / (pp + eps)
        rec  = tp / (gg + eps)
        f = (1 + beta2) * prec * rec / (beta2 * prec + rec + eps)
        best = torch.maximum(best, f)
    return best


@torch.no_grad()
def batch_maxF(logits, gts, beta2=0.3):
    """
    logits: Bx1xHxW ; gts: Bx1xHxW
    """
    probs = torch.sigmoid(logits)
    vals = []
    for b in range(probs.size(0)):
        vals.append(fmax_torch(probs[b, 0], gts[b, 0], beta2=beta2))
    return torch.stack(vals).mean().item()

# metrics_maxf.py
import torch

def fmax_torch(prob, gt, beta2=0.3, n_th=255, eps=1e-8):
    """
    prob: 1xHxW in [0,1] ; gt: 1xHxW in {0,1}
    返回单图像 F_beta 的最大值（跨阈值）
    """
    P = prob.view(-1)
    G = (gt > 0.5).float().view(-1)
    if G.sum() < 1:  # 空目标的边界情形，定义F=1当P全0，否则0
        return torch.tensor(1.0 if P.max() <= 0.5 else 0.0, device=prob.device, dtype=prob.dtype)
    ths = torch.linspace(0.0, 1.0, n_th, device=prob.device)
    best = torch.tensor(0.0, device=prob.device, dtype=prob.dtype)
    for t in ths:
        pred = (P >= t).float()
        tp = (pred * G).sum()
        pp = pred.sum()
        gg = G.sum()
        prec = tp / (pp + eps)
        rec  = tp / (gg + eps)
        f = (1+beta2) * prec * rec / (beta2 * prec + rec + eps)
        best = torch.maximum(best, f)
    return best

@torch.no_grad()
def batch_maxF(logits, gts, beta2=0.3):
    """
    logits: Bx1xHxW ; gts: Bx1xHxW
    """
    probs = torch.sigmoid(logits)
    vals = []
    for b in range(probs.size(0)):
        vals.append(fmax_torch(probs[b,0], gts[b,0], beta2=beta2))
    return torch.stack(vals).mean().item()
