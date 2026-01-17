# sam_decoder.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from adapters import Adapter2d

# -------------------------
# 2D Sine-Cos Positional Encoding
# -------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, "dim must be multiple of 4"
        self.dim = dim
    def forward(self, H: int, W: int, device):
        pe = torch.zeros(1, self.dim, H, W, device=device)
        dim_half = self.dim // 2
        num_freq = dim_half // 2
        div_y = torch.exp(torch.arange(0, dim_half, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.0) / dim_half))
        div_x = torch.exp(torch.arange(0, dim_half, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.0) / dim_half))
        pos_y = torch.arange(0, H, device=device, dtype=torch.float32).unsqueeze(1)
        pos_x = torch.arange(0, W, device=device, dtype=torch.float32).unsqueeze(1)
        sin_y, cos_y = torch.sin(pos_y * div_y), torch.cos(pos_y * div_y)
        sin_x, cos_x = torch.sin(pos_x * div_x), torch.cos(pos_x * div_x)
        sin_y = sin_y.t().unsqueeze(0).unsqueeze(-1).expand(1, num_freq, H, W)
        cos_y = cos_y.t().unsqueeze(0).unsqueeze(-1).expand(1, num_freq, H, W)
        sin_x = sin_x.t().unsqueeze(0).unsqueeze(2).expand(1, num_freq, H, W)
        cos_x = cos_x.t().unsqueeze(0).unsqueeze(2).expand(1, num_freq, H, W)
        pe[:, 0:dim_half:2, :, :] = sin_y
        pe[:, 1:dim_half:2, :, :] = cos_y
        pe[:, dim_half::2, :, :]  = sin_x
        pe[:, dim_half+1::2, :, :] = cos_x
        return pe

# -------------------------
# Token 编码（stride=4），可选 Adapter
# -------------------------
class ImageTokenEncoder(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, out_ch=128,
                 use_adapter: bool=False, adapter_reduction: int=16, adapter_drop: float=0.0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch//2, 3, 2, 1, bias=False),  # H/2
            nn.BatchNorm2d(base_ch//2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, base_ch, 3, 2, 1, bias=False),# H/4
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.pos = PositionalEncoding2D(out_ch)
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapt = Adapter2d(out_ch, reduction=adapter_reduction, drop=adapter_drop,
                                   use_semantic_gate=True, init_scale=1.0)

    def forward(self, img, prob):
        x = torch.cat([img, prob], dim=1)  # Bx4xHxW
        feat = self.enc(x)                 # BxCxH/4xW/4
        if self.use_adapter:
            feat = self.adapt(feat)
        B, C, H4, W4 = feat.shape
        pos = self.pos(H4, W4, feat.device)
        feat = feat + pos
        tok = feat.flatten(2).permute(2, 0, 1)  # LxBxC
        return feat, tok

# -------------------------
# Transformer Decoder Block
# -------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.lin1 = nn.Linear(d_model, dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, q, kv):
        q2, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm1(q + self.dropout(q2))
        q2, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q = self.norm2(q + self.dropout(q2))
        ff = self.lin2(self.dropout(self.act(self.lin1(q))))
        q = self.norm3(q + self.dropout(ff))
        return q

# -------------------------
# SAM-like 轻量掩码解码器（可选 Adapter）
# -------------------------
class SAMLikeMaskDecoderLite(nn.Module):
    def __init__(self, num_queries=8, token_dim=128, nhead=4, num_layers=2, ff_dim=256, mask_embed_dim=64,
                 use_adapter: bool=False, adapter_reduction: int=16, adapter_drop: float=0.0):
        super().__init__()
        self.img_tok = ImageTokenEncoder(in_ch=4, base_ch=64, out_ch=token_dim,
                                         use_adapter=use_adapter, adapter_reduction=adapter_reduction,
                                         adapter_drop=adapter_drop)
        self.queries = nn.Parameter(torch.randn(num_queries, 1, token_dim) * 0.02)
        self.decoders = nn.ModuleList([
            DecoderBlock(d_model=token_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=0.0)
            for _ in range(num_layers)
        ])
        self.mask_embed = nn.Sequential(
            nn.Conv2d(token_dim, mask_embed_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(8, mask_embed_dim),
            nn.GELU(),
        )
        self.mask_adapter = Adapter2d(mask_embed_dim, reduction=adapter_reduction, drop=adapter_drop,
                                      use_semantic_gate=True, init_scale=1.0) if use_adapter else None
        self.to_mask_coeff = nn.Linear(token_dim, mask_embed_dim)
        self.iou_head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(),
            nn.Linear(token_dim, 1)
        )

    def forward(self, image, logits):
        prob = torch.sigmoid(logits)
        base_feat, tokens = self.img_tok(image, prob)       # BxCxH/4xW/4; LxBxC
        B, C, H4, W4 = base_feat.shape
        q = self.queries.expand(-1, B, -1)                  # NxBxD
        for blk in self.decoders:
            q = blk(q, tokens)
        q = q.permute(1,0,2)                                # BxNxD
        mask_feat = self.mask_embed(base_feat)              # BxE xH/4xW/4
        if self.mask_adapter is not None:
            mask_feat = self.mask_adapter(mask_feat)
        coeff = self.to_mask_coeff(q)                       # BxN xE
        masks_low = torch.einsum("bne,behw->bnhw", coeff, mask_feat)
        masks = F.interpolate(masks_low, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        weights = torch.sigmoid(self.iou_head(q)).squeeze(-1)  # BxN
        weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(1)  # Bx1xN
        residual = torch.bmm(weights, masks.view(B, masks.shape[1], -1)).view(B, 1, *logits.shape[-2:])
        return logits + residual
