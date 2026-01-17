# adapters.py
# -*- coding: utf-8 -*-
import math
from typing import Iterable, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 2D Conv Adapter (瓶颈) + 轻量“语义位移”门控
# -------------------------
class Adapter2d(nn.Module):
    """
    y = x + scale * Drop(Conv1x1(down) -> GELU -> Conv1x1(up))
    可选语义位移门控：基于全局均值/方差 + 1x1门控，近似地对 domain shift 敏感
    """
    def __init__(self, channels: int, reduction: int = 16, drop: float = 0.0,
                 use_semantic_gate: bool = True, init_scale: float = 1.0):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.act  = nn.GELU()
        self.up   = nn.Conv2d(hidden, channels, 1, bias=False)
        self.drop = nn.Dropout2d(p=drop) if drop > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.use_semantic_gate = use_semantic_gate
        if use_semantic_gate:
            # 简易门控：统计量 -> 1x1 两层MLP -> [0,1]
            gate_c = max(8, channels // 8)
            self.gate = nn.Sequential(
                nn.Conv2d(channels*2, gate_c, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(gate_c, channels, 1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.down.weight, nonlinearity="linear")
        nn.init.zeros_(self.up.weight)
        # 让初期接近恒等映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x
        y = self.up(self.act(self.down(x)))
        y = self.drop(y)
        if self.gate is not None:
            mu = x.mean(dim=(2,3), keepdim=True)
            var = x.var(dim=(2,3), keepdim=True, unbiased=False)
            stat = torch.cat([mu, torch.sqrt(var + 1e-6)], dim=1)
            g = self.gate(stat)  # [B,C,1,1]
            y = y * g
        return idt + self.scale * y

# -------------------------
# 仅训练 Adapter/Norm 等（PEFT-only）
# -------------------------
def set_peft_trainable(root: nn.Module,
                       trainable_keys: Tuple[str,...] = ("adapter", "norm", "bn", "ln", "iou_head", "to_mask_coeff"),
                       verbose: bool = True) -> Tuple[int,int]:
    total, trainable = 0, 0
    for n, p in root.named_parameters():
        total += p.numel()
        # 命名包含上述关键字的设为可训练，否则冻结
        if any(k in n.lower() for k in trainable_keys):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
    if verbose:
        print(f"[PEFT] Total params: {total/1e6:.2f}M, Trainable: {trainable/1e6:.2f}M ({100*trainable/max(1,total):.2f}%)")
    return total, trainable

# -------------------------
# 统计参数 & FLOPs（优先 thop，其次 fvcore；都无则仅参数）
# -------------------------
def count_params(root: nn.Module) -> Tuple[int,int]:
    tot = sum(p.numel() for p in root.parameters())
    trn = sum(p.numel() for p in root.parameters() if p.requires_grad)
    return tot, trn

def profile_flops(model: nn.Module, input_size: Tuple[int,int,int,int]) -> Optional[float]:
    inp = torch.randn(*input_size)
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(inp,), verbose=False)
        return float(flops) / 1e9  # GFLOPs
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, inp).total()
        return float(flops) / 1e9
    except Exception:
        return None

# -------------------------
# 简单 Combine 模型用于 FLOPs 估算
# -------------------------
class CombinedForProfile(nn.Module):
    def __init__(self, backbone: nn.Module, sam_dec: nn.Module=None, refine: nn.Module=None):
        super().__init__()
        self.backbone = backbone
        self.sam_dec  = sam_dec
        self.refine   = refine
    def forward(self, x):
        logits = self.backbone(x)
        if self.sam_dec is not None:
            logits = self.sam_dec(x, logits)
        if self.refine is not None:
            logits = self.refine(x, logits)
        return logits
