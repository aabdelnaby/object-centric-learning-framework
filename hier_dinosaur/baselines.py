"""Flat (unstructured) patch controls for the HierRouter.

Both heads attend directly over the frozen DINOv3 patch tokens with the compound
"<object> <part>" query and feed the attended vector to the *same* query-conditioned
attribute MLP as the router. They emit ``log P(a)`` and are trained with ``nn.NLLLoss``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class PatchQueryDotProductColorHead(nn.Module):
    """Patch-QDot: a single explicit query·patch dot product, softmax-pooled.

    Args:
        d_query:              dimensionality of the projected query (= d_slot).
        d_vit:                patch-token dimensionality (384).
        d_slot:               projection target when ``project_patches`` is on.
        project_patches:      ``False`` = raw variant (dot product in the 384-d ViT space),
                              ``True`` = projected variant (one learned linear map to d_slot first).
        legacy_strip_tokens:  drop this many leading tokens before attending. The thesis runs used 4,
                              a register-token strip the feature cache had already applied, so they
                              attended over 192 of the 196 patches. Keep 0 for new runs;
                              ``normalize_config`` sets 4 when loading a thesis checkpoint.
    """

    def __init__(
        self,
        d_query: int,
        d_vit: int,
        d_slot: int,
        num_classes: int,
        project_patches: bool = False,
        legacy_strip_tokens: int = 0,
    ):
        super().__init__()
        self.project_patches = project_patches
        self.legacy_strip_tokens = int(legacy_strip_tokens)
        self.patch_proj = nn.Linear(d_vit, d_slot) if project_patches else nn.Identity()
        d_attn = d_slot if project_patches else d_vit
        self.d_attn = d_attn
        self.scale = 1.0 / math.sqrt(d_attn)
        self.q_proj = nn.Linear(d_query, d_attn)
        self.f_readout = nn.Linear(d_query, d_attn)
        self.color_head = nn.Sequential(
            nn.Linear(2 * d_attn, d_attn), nn.GELU(), nn.Linear(d_attn, num_classes),
        )

    def forward(self, patches: torch.Tensor, h_yx: torch.Tensor):
        """``patches`` (B, N, d_vit), ``h_yx`` (B, d_query) → ``(log P(a) (B, C), attn (B, N'))``."""
        if self.legacy_strip_tokens:
            patches = patches[:, self.legacy_strip_tokens:]
        patches = self.patch_proj(patches)
        q = self.q_proj(h_yx)
        scores = torch.einsum("bd,bnd->bn", q, patches) * self.scale
        attn = F.softmax(scores, dim=-1)
        z = torch.einsum("bn,bnd->bd", attn, patches)
        feat = torch.cat([z, self.f_readout(h_yx)], dim=-1)
        return F.log_softmax(self.color_head(feat), dim=-1), attn.detach()


class QueryCrossAttentionColorHead(nn.Module):
    """Patch-QCA: a learned multi-head cross-attention layer over the (projected) patch tokens."""

    def __init__(self, d_model: int, num_classes: int, num_heads: int = 8):
        super().__init__()
        self.q_attn = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.f_readout = nn.Linear(d_model, d_model)
        self.color_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, num_classes),
        )

    def forward(self, tokens: torch.Tensor, h_yx: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
        """``tokens`` (B, N, D), ``h_yx`` (B, D) → ``(log P(a) (B, C), attn (B, N))``."""
        q = self.q_attn(h_yx).unsqueeze(1)
        key_padding_mask = ~token_mask if token_mask is not None else None
        attn_out, attn_w = self.cross_attn(
            q, tokens, tokens, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        z = self.ln(attn_out.squeeze(1) + q.squeeze(1))
        feat = torch.cat([z, self.f_readout(h_yx)], dim=-1)
        return F.log_softmax(self.color_head(feat), dim=-1), attn_w.squeeze(1).detach()
