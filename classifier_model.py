"""SlotClassifier: frozen DINOSAUR slots + frozen RoBERTa → attribute classification.

Architecture
------------
1. image  → frozen DINOSAUR (ViT-S/16 + SlotAttention) → slots (B, 7, 256)
2. query  → frozen RoBERTa-Large                        → tokens (B, L, 1024)
3. tokens → TextProjector (trainable)                   → (B, L, 256)
4. gated cross-attention (trainable):
       slots = slots + tanh(α) * CA(slots, projected_tokens)
5. mean-pool slots → (B, 256)
6. linear ClassifierHead (trainable) → logits (B, num_classes)

Trainable parameters: TextProjector, GatedCrossAttention, ClassifierHead.
Everything else is frozen.

Loading DINOSAUR
----------------
We load ONLY the three sub-modules needed for slot extraction
(feature_extractor, conditioning, perceptual_grouping) directly from the
checkpoint state_dict.  This bypasses the full CombinedModel and avoids
strict-key errors from decoder/hierarchical layers we never run.

Checkpoint: checkpoints/epoch_67-step_500000_coco.ckpt
Config:     coco_feat_rec_dino_small16_auto  (ViT-S/16, 384→256, n_slots=7)

Run from the repo root:
    conda run -n oclf_env python classifier_model.py  # quick smoke test
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Trainable sub-modules
# ---------------------------------------------------------------------------

class TextProjector(nn.Module):
    """2-layer MLP projecting RoBERTa token features to slot dimension.

    Follows SteerViT convention: L2-normalise input, apply two linear layers
    with GELU activation, finish with LayerNorm.

    d_text (1024, RoBERTa-Large) → d_slot → d_slot
    """

    def __init__(self, d_text: int, d_slot: int):
        super().__init__()
        self.fc1 = nn.Linear(d_text, d_slot)
        self.fc2 = nn.Linear(d_slot, d_slot)
        self.act = nn.GELU()
        self.ln  = nn.LayerNorm(d_slot)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (B, L, d_text)
        Returns:
            (B, L, d_slot)
        """
        x = F.normalize(text_features, dim=-1)   # L2-norm over feature dim
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.ln(x)


class GatedCrossAttention(nn.Module):
    """Gated cross-attention: slots attend to projected text tokens.

    Implements the gate from SteerViT (arXiv 2604.02327), eq. (1)–(2):

        slots = slots + tanh(α) · CA(slots, text_tokens)

    ``α`` is a learnable scalar initialised to zero so the gate starts closed
    (identity at init) and opens gradually during training.

    Args:
        d_slot:    Dimension of slot (and projected text) vectors.
        num_heads: Number of attention heads (d_slot must be divisible by this).
    """

    def __init__(self, d_slot: int, num_heads: int):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_slot,
            num_heads=num_heads,
            batch_first=True,
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        slots: torch.Tensor,          # (B, N, d_slot) — queries
        text_tokens: torch.Tensor,    # (B, L, d_slot) — keys and values
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True = pad
    ) -> tuple:
        """Returns (updated_slots, attn_weights, ca_out).

        attn_weights: (B, N_slots, L) averaged over heads.
        ca_out:       (B, N_slots, d_slot) raw cross-attention output before gate.
        """
        ca_out, attn_weights = self.attn(
            query=slots,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        updated = slots + torch.tanh(self.alpha) * ca_out
        return updated, attn_weights, ca_out


class _FusionBlock(nn.Module):
    """Transformer encoder block that exposes attention weights.

    Equivalent to ``nn.TransformerEncoderLayer(...)`` but written by hand so we
    can request ``need_weights=True`` from the underlying ``MultiheadAttention``
    for visualisation. PyTorch's stock layer hard-codes ``need_weights=False``
    for the fast path.

    Args:
        activation:  Feed-forward non-linearity, ``"gelu"`` (default) or ``"relu"``.
        norm_first:  ``True`` (default) → pre-norm (the original behaviour used by
                     ``TransformerFusionPooler`` / ``VQATransformerPooler``).
                     ``False`` → post-norm, matching the standard
                     ``nn.TransformerEncoderLayer`` the paper's VQA model uses.
    """

    def __init__(
        self,
        d: int,
        num_heads: int,
        ff_hidden: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first
        act = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.mlp  = nn.Sequential(
            nn.Linear(d, ff_hidden), act(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, need_weights: bool = False):
        if self.norm_first:
            x_norm = self.ln1(x)
            attn_out, attn_w = self.attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights, average_attn_weights=True,
            )
            x = x + self.drop(attn_out)
            x = x + self.mlp(self.ln2(x))
        else:
            # Post-norm: sublayer → residual add → LayerNorm (standard transformer).
            attn_out, attn_w = self.attn(
                x, x, x,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights, average_attn_weights=True,
            )
            x = self.ln1(x + self.drop(attn_out))
            x = self.ln2(x + self.mlp(x))
        return x, attn_w


class TransformerFusionPooler(nn.Module):
    """Joint slot+text fusion via a small transformer encoder.

    Concatenates ``[CLS] + slots + projected_text_tokens``, adds learned type
    embeddings (CLS / slot / text), runs through ``num_layers`` self-attention
    blocks (each does slot↔slot, slot↔text, text↔text in one shot), and reads
    out the CLS token as the pooled feature for the classifier head.

    This is the multi-hop alternative to GatedCrossAttention: with ≥2 layers
    the model can localise a part *within* a vehicle slot (depth=2 questions
    on Super-CLEVR-3D require multi-step binding).

    Args:
        d_slot:     Slot dimension (also CLS / text-projection dim).
        num_heads:  Attention heads (d_slot must be divisible by this).
        num_layers: Stacked self-attention blocks.
        ff_hidden:  Feed-forward hidden dim. Defaults to 4*d_slot.
        dropout:    Block-internal dropout.
    """

    def __init__(
        self,
        d_slot: int,
        num_heads: int,
        num_layers: int = 2,
        ff_hidden: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        ff_hidden = ff_hidden if ff_hidden is not None else 4 * d_slot
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_slot))
        nn.init.normal_(self.cls_token, std=0.02)
        # 0 = CLS, 1 = slot, 2 = text
        self.type_embed = nn.Embedding(3, d_slot)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self.layers     = nn.ModuleList([
            _FusionBlock(d_slot, num_heads, ff_hidden, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_ln   = nn.LayerNorm(d_slot)

    def forward(
        self,
        slots: torch.Tensor,           # (B, N, d_slot)
        text_tokens: torch.Tensor,     # (B, L, d_slot)
        text_attn_mask: torch.Tensor,  # (B, L)  1 = real, 0 = pad
        return_attn: bool = False,
    ):
        """Returns ``(pooled, cls_to_slots, slot_to_text)``.

        - ``pooled``: (B, d_slot) — CLS readout for the classifier head.
        - ``cls_to_slots``: (B, N) or None — last-layer CLS attention to slot
          positions; analog of the prior pipeline's ``slot_ca_norms``.
        - ``slot_to_text``: (B, N, L) or None — last-layer slot-row × text-col
          attention sub-block; analog of ``cross_attn_weights``.
        """
        B, N, d = slots.shape
        L = text_tokens.shape[1]
        device = slots.device

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, slots, text_tokens], dim=1)  # (B, 1+N+L, d)

        type_ids = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.ones(N, dtype=torch.long, device=device),
            torch.full((L,), 2, dtype=torch.long, device=device),
        ], dim=0).unsqueeze(0).expand(B, -1)
        x = x + self.type_embed(type_ids)

        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=device)
        slot_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        text_pad  = text_attn_mask.eq(0)
        kpm = torch.cat([cls_mask, slot_mask, text_pad], dim=1)

        last_attn = None
        for k, layer in enumerate(self.layers):
            need_w = return_attn and (k == len(self.layers) - 1)
            x, w = layer(x, key_padding_mask=kpm, need_weights=need_w)
            if need_w:
                last_attn = w  # (B, T, T) where T = 1+N+L

        x = self.final_ln(x)
        pooled = x[:, 0]  # CLS

        if return_attn and last_attn is not None:
            cls_to_slots = last_attn[:, 0, 1:1 + N]            # (B, N)
            slot_to_text = last_attn[:, 1:1 + N, 1 + N:]       # (B, N, L)
            return pooled, cls_to_slots, slot_to_text
        return pooled, None, None


class VQATransformerPooler(nn.Module):
    """VQA-style joint transformer over ``[z', t', CLS]``.

    Follows the formulation from Ding et al. (2021a) — also used by Devlin
    et al. (2018) and Lu et al. (2019):

    1. Apply a separate linear layer to image tokens ``z`` and text tokens
       ``t`` to bring both to ``D_model - 2`` (so room remains for the
       modality one-hot).
    2. Add a sinusoidal positional encoding to the text tokens only — the
       image side (slots / register-augmented patches) is order-free here.
    3. Augment every vector with a 2-d one-hot indicating modality
       (``[1, 0]`` = image, ``[0, 1]`` = text), bringing the final dim to
       ``D_model``.
    4. A trainable ``CLS ∈ R^{D_model}`` is appended.
    5. Concatenate ``[z', t', CLS]`` and pass through an ``N_t``-layer
       transformer encoder.
    6. Return the transformed CLS — the surrounding model applies an MLP
       classification head to it.

    Args:
        d_model:      Final per-token dimension (= D_model). Must be ≥ 3 so
                      the 2-d modality one-hot fits.
        num_heads:    Attention heads (``d_model`` must be divisible by this).
        num_layers:   Number of stacked transformer encoder layers.
        ff_hidden:    Feed-forward hidden dim. Defaults to ``4 * d_model``.
        dropout:      Block-internal dropout.
        max_text_len: Maximum text length the sinusoidal-PE table is built for.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 2,
        ff_hidden: Optional[int] = None,
        dropout: float = 0.0,
        max_text_len: int = 512,
    ):
        super().__init__()
        if d_model < 3:
            raise ValueError("d_model must be > 2 to leave room for modality one-hot")
        ff_hidden = ff_hidden if ff_hidden is not None else 4 * d_model
        proj_dim = d_model - 2

        self.d_model  = d_model
        self.proj_dim = proj_dim
        self.img_proj = nn.Linear(d_model, proj_dim)
        self.txt_proj = nn.Linear(d_model, proj_dim)

        pe = self._build_sinusoidal_pe(max_text_len, proj_dim)
        self.register_buffer("text_pe", pe)  # (max_text_len, proj_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.layers   = nn.ModuleList([
            _FusionBlock(d_model, num_heads, ff_hidden, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        pe       = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        z: torch.Tensor,               # (B, N, d_model)
        t: torch.Tensor,               # (B, L, d_model)
        text_attn_mask: torch.Tensor,  # (B, L) 1 = real, 0 = pad
        return_attn: bool = False,
    ):
        """Returns ``(pooled, cls_to_img, img_to_text)``.

        - ``pooled``:      (B, d_model) — transformed CLS, consumed by the MLP head.
        - ``cls_to_img``:  (B, N) or None — last-layer CLS attention to image cols.
        - ``img_to_text``: (B, N, L) or None — last-layer image-row × text-col block,
          analog of the prior pipeline's slot×text cross-attention weights.
        """
        B, N, _ = z.shape
        L       = t.shape[1]
        device  = z.device

        if L > self.text_pe.shape[0]:
            raise ValueError(
                f"text length {L} exceeds VQATransformerPooler.max_text_len "
                f"({self.text_pe.shape[0]})"
            )

        z_proj = self.img_proj(z)                                            # (B, N, d_model-2)
        t_proj = self.txt_proj(t)                                            # (B, L, d_model-2)

        # Sinusoidal PE on text only (image side is order-free in this setup)
        t_proj = t_proj + self.text_pe[:L].to(dtype=t_proj.dtype).unsqueeze(0)

        # 2-d modality one-hot: image = [1, 0], text = [0, 1]
        img_mod = torch.zeros(B, N, 2, device=device, dtype=z_proj.dtype)
        img_mod[..., 0] = 1.0
        txt_mod = torch.zeros(B, L, 2, device=device, dtype=t_proj.dtype)
        txt_mod[..., 1] = 1.0
        z_full  = torch.cat([z_proj, img_mod], dim=-1)                       # (B, N, d_model)
        t_full  = torch.cat([t_proj, txt_mod], dim=-1)                       # (B, L, d_model)

        cls = self.cls_token.expand(B, -1, -1)                               # (B, 1, d_model)
        x   = torch.cat([z_full, t_full, cls], dim=1)                        # (B, N+L+1, d_model)

        img_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        text_pad = text_attn_mask.eq(0)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        kpm      = torch.cat([img_mask, text_pad, cls_mask], dim=1)

        last_attn = None
        for k, layer in enumerate(self.layers):
            need_w  = return_attn and (k == len(self.layers) - 1)
            x, w    = layer(x, key_padding_mask=kpm, need_weights=need_w)
            if need_w:
                last_attn = w                                                # (B, T, T)

        x      = self.final_ln(x)
        pooled = x[:, -1]                                                    # CLS at end

        if return_attn and last_attn is not None:
            cls_to_img  = last_attn[:, -1, :N]                               # (B, N)
            img_to_text = last_attn[:, :N, N:N + L]                          # (B, N, L)
            return pooled, cls_to_img, img_to_text
        return pooled, None, None


class VQAPaperPooler(nn.Module):
    """Paper-faithful VQA downstream model.

    Reproduces the downstream transformer from *"Exploring the Effectiveness of
    Object-Centric Representations in VQA"* (App. A.3), which itself follows
    Ding et al. (2021a). Differs from ``VQATransformerPooler`` in being exact:

    1. A **single** linear layer projects each modality — image tokens ``z``
       (slots / fixed regions, dim ``d_img``) and text tokens ``t`` (dim
       ``d_text``) — to ``d_model - 2``, each with dropout. No upstream
       TextProjector: raw encoder features go straight in.
    2. A sinusoidal positional encoding is added to the text tokens only.
    3. Every vector is augmented with a 2-d modality one-hot
       (``[1, 0]`` = image, ``[0, 1]`` = text), bringing the dim to ``d_model``.
    4. A trainable ``CLS ∈ R^{d_model}`` is appended.
    5. ``[z', t', CLS]`` is run through an ``N_t``-layer **standard** transformer
       encoder (post-norm, ReLU, ``dim_feedforward = d_model``).
    6. The transformed CLS is returned; the surrounding model applies the MLP head.

    Paper hyperparameters: ``d_model = 128``, ``dim_feedforward = 128``,
    ``dropout = 0.1``, ``N_t ∈ {2, 5, 15}`` (the T-2 / T-5 / T-15 variants).

    Args:
        d_img:        Image-token feature dim (``d_slot`` for slots, ``d_vit`` for
                      patch control).
        d_text:       Text-token feature dim (768 for T5-base, 1024 for RoBERTa-Large).
        d_model:      Transformer working dim (128 in the paper). Must be ≥ 3 so
                      the 2-d modality one-hot fits, and divisible by ``num_heads``.
        num_heads:    Attention heads.
        num_layers:   Transformer encoder layers (= T-n).
        dropout:      Projection + transformer dropout (0.1 in the paper).
        max_text_len: Length the sinusoidal-PE table is built for.
    """

    def __init__(
        self,
        d_img: int,
        d_text: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_text_len: int = 512,
    ):
        super().__init__()
        if d_model < 3:
            raise ValueError("d_model must be > 2 to leave room for modality one-hot")
        proj_dim = d_model - 2

        self.d_model  = d_model
        self.proj_dim = proj_dim
        self.img_proj = nn.Linear(d_img, proj_dim)
        self.txt_proj = nn.Linear(d_text, proj_dim)
        self.img_drop = nn.Dropout(dropout)
        self.txt_drop = nn.Dropout(dropout)

        pe = VQATransformerPooler._build_sinusoidal_pe(max_text_len, proj_dim)
        self.register_buffer("text_pe", pe)  # (max_text_len, proj_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Standard transformer encoder: post-norm, ReLU, ff = d_model.
        self.layers   = nn.ModuleList([
            _FusionBlock(
                d_model, num_heads, ff_hidden=d_model, dropout=dropout,
                activation="relu", norm_first=False,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        z: torch.Tensor,               # (B, N, d_img)
        t: torch.Tensor,               # (B, L, d_text)
        text_attn_mask: torch.Tensor,  # (B, L) 1 = real, 0 = pad
        return_attn: bool = False,
    ):
        """Returns ``(pooled, cls_to_img, img_to_text)``.

        - ``pooled``:      (B, d_model) — transformed CLS, consumed by the MLP head.
        - ``cls_to_img``:  (B, N) or None — last-layer CLS attention to image cols.
        - ``img_to_text``: (B, N, L) or None — last-layer image-row × text-col block.
        """
        B, N, _ = z.shape
        L       = t.shape[1]
        device  = z.device

        if L > self.text_pe.shape[0]:
            raise ValueError(
                f"text length {L} exceeds VQAPaperPooler.max_text_len "
                f"({self.text_pe.shape[0]})"
            )

        z_proj = self.img_drop(self.img_proj(z))                             # (B, N, d_model-2)
        t_proj = self.txt_drop(self.txt_proj(t))                             # (B, L, d_model-2)

        # Sinusoidal PE on text only (image side is order-free in this setup).
        t_proj = t_proj + self.text_pe[:L].to(dtype=t_proj.dtype).unsqueeze(0)

        # 2-d modality one-hot: image = [1, 0], text = [0, 1].
        img_mod = torch.zeros(B, N, 2, device=device, dtype=z_proj.dtype)
        img_mod[..., 0] = 1.0
        txt_mod = torch.zeros(B, L, 2, device=device, dtype=t_proj.dtype)
        txt_mod[..., 1] = 1.0
        z_full  = torch.cat([z_proj, img_mod], dim=-1)                       # (B, N, d_model)
        t_full  = torch.cat([t_proj, txt_mod], dim=-1)                       # (B, L, d_model)

        cls = self.cls_token.expand(B, -1, -1)                               # (B, 1, d_model)
        x   = torch.cat([z_full, t_full, cls], dim=1)                        # (B, N+L+1, d_model)

        img_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        text_pad = text_attn_mask.eq(0)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        kpm      = torch.cat([img_mask, text_pad, cls_mask], dim=1)

        last_attn = None
        for k, layer in enumerate(self.layers):
            need_w  = return_attn and (k == len(self.layers) - 1)
            x, w    = layer(x, key_padding_mask=kpm, need_weights=need_w)
            if need_w:
                last_attn = w                                                # (B, T, T)

        pooled = x[:, -1]                                                    # CLS at end

        if return_attn and last_attn is not None:
            cls_to_img  = last_attn[:, -1, :N]                               # (B, N)
            img_to_text = last_attn[:, :N, N:N + L]                          # (B, N, L)
            return pooled, cls_to_img, img_to_text
        return pooled, None, None


class HierRouter(nn.Module):
    """Structured path-marginalisation head for "What is the color of <x> of <y>?".

    Traverses the parent→child slot tree produced by
    ``SlotClassifier._refine_top_slots`` (P parent/object slots, each refined into
    K child/part slots) instead of pooling slots flat:

        P(j|y)    parent routing over object slots (empty parents masked out),
        P(k|j,x)  *local* child routing within each parent's K children,
        P(a|c_jk) a shared per-child colour head,
        P(a)      = Σ_{j,k} P(j|y) · P(k|j,x) · P(a|c_jk)   (marginal over paths).

    Returns ``log P(a)`` (train with ``nn.NLLLoss``). Child→parent membership is the
    fixed ``j = m // K`` grouping — the P-axis of the ``(B, P, K, Ds)`` child tensor —
    so it is never stored as an edge list and invalid cross-parent paths cannot be
    formed (the local child softmax is taken within row j; the path weight only ever
    multiplies a parent with its own children).

    The span vectors ``h_x`` (part) / ``h_y`` (object) are expected already projected
    into slot space (``SlotClassifier.text_projector``); this module specialises them
    into parent/child queries with a per-route linear.
    """

    def __init__(
        self,
        d_slot: int,
        num_classes: int,
        temperature: float = 1.0,
        child_scorer: str = "bilinear",
        entropy_weight: float = 0.0,
        use_children: bool = True,
        readout_query: bool = True,
        color_source: str = "slot",
        d_patch: Optional[int] = None,
        qdot_project_patches: bool = True,
        qdot_dropout: float = 0.0,
    ):
        super().__init__()
        if child_scorer not in ("bilinear", "mlp"):
            raise ValueError(f"unknown child_scorer={child_scorer!r}; expected bilinear|mlp")
        if color_source not in ("slot", "patch", "patch_qdot"):
            raise ValueError(f"unknown color_source={color_source!r}; expected slot|patch|patch_qdot")
        self.tau            = float(temperature)
        self.entropy_weight = float(entropy_weight)
        self.child_scorer   = child_scorer
        # Ablation switch: when False the model collapses to a *parent-only* router —
        # it routes only over object slots and reads the colour straight off the chosen
        # parent (P(a)=Σ_j P(j|y)·P(a|s_j)); the child level and the part span <x> are
        # dropped entirely. Lets us measure the utility of having children at all.
        self.use_children   = bool(use_children)
        # Query-conditioned colour readout: instead of P(colour | slot) the head reads
        # P(colour | slot, f_readout(q_readout)) where q_readout is the "<y> <x>" compound
        # noun (e.g. "car door"). The routed slot supplies the localised part; the text
        # tells the shared head *which* entity's colour to report. f_readout is learned.
        self.readout_query  = bool(readout_query)
        # Colour evidence source for P(a | ·). Default "slot": the routed child *slot
        # vector* c_jk (the abstracted DINOSAUR representation). "patch": pool the raw
        # DINO patches the child grounds to (Σ_n a_jkn · patch_n) and read colour from
        # *those* — routing (P(j|y), P(k|j,x)) still decides *which* child, only the
        # colour features change. Tests whether reading colour off the grounded patches
        # (which retain low-level colour the slot vector abstracts away) recovers patch-
        # level colour accuracy while keeping the routing's grounding.
        self.color_source = color_source
        if color_source in ("patch", "patch_qdot") and not bool(use_children):
            raise ValueError(f"color_source={color_source!r} requires use_children=True (the "
                             "patch evidence is read from the child slots' attention).")
        self.scale          = d_slot ** -0.5

        # Specialise the (shared) text-space span vectors into route-specific queries.
        self.q_parent = nn.Linear(d_slot, d_slot)
        if self.use_children:
            self.q_child = nn.Linear(d_slot, d_slot)
            if child_scorer == "mlp":
                # Score a (part-query, child slot, parent slot) triple so a part's meaning
                # can depend on its object (a "wheel" of a car vs a "leg" of a chair).
                self.child_mlp = nn.Sequential(
                    nn.Linear(3 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, 1),
                )
        # Shared colour head: reads a child slot (full model) or, in the parent-only
        # ablation, the parent/object slot directly. When readout_query is on it also
        # consumes the learned readout query (concatenated) → input dim 2*d_slot.
        if color_source == "patch_qdot":
            # Per-child query·patch dot-product colour readout confined to each child's
            # patches (owns its own query/patch projections + readout + colour MLP).
            self.qdot_readout = ChildMaskedQDotColorHead(
                d_query=d_slot, d_vit=int(d_patch or d_slot), d_slot=d_slot,
                num_classes=num_classes, project_patches=qdot_project_patches,
                dropout=qdot_dropout,
            )
        else:
            if self.readout_query:
                self.f_readout = nn.Linear(d_slot, d_slot)
            # Colour-evidence width: d_slot for a slot vector, d_patch (=d_vit) for a pooled
            # patch vector. The readout query (if on) always adds d_slot.
            d_color  = d_slot if color_source == "slot" else int(d_patch or d_slot)
            color_in = d_color + (d_slot if self.readout_query else 0)
            self.color_head = nn.Sequential(
                nn.Linear(color_in, d_slot), nn.GELU(), nn.Linear(d_slot, num_classes),
            )

    def _color_logp(self, slots: torch.Tensor, h_readout: Optional[torch.Tensor]):
        """log P(colour | slot[, readout query]).

        ``slots`` is ``(..., Ds)`` (``(B,P,Ds)`` parent-only or ``(B,P,K,Ds)`` full);
        ``h_readout`` is the slot-space "<y> <x>" query ``(B, Ds)``. When
        ``readout_query`` is on, ``f_readout(h_readout)`` is broadcast over the slot
        axes and concatenated before the colour MLP. Returns ``(..., C)`` log-probs.
        """
        if self.readout_query:
            g = self.f_readout(h_readout)                 # (B, Ds)
            while g.dim() < slots.dim():                  # add P (and K) singleton dims
                g = g.unsqueeze(1)
            g = g.expand(*slots.shape[:-1], g.shape[-1])   # (..., Ds)
            feat = torch.cat([slots, g], dim=-1)          # (..., 2*Ds)
        else:
            feat = slots
        return F.log_softmax(self.color_head(feat), dim=-1)

    def routing_modules(self):
        """The parent/child ROUTING submodules — everything except the colour head.

        Used to warm-start / freeze the proven routing while a new colour head is
        trained (``--router_init_ckpt --router_freeze_routing``). Excludes
        ``f_readout`` and ``color_head`` (the colour-readout subsystem).
        """
        mods = [self.q_parent]
        if self.use_children:
            mods.append(self.q_child)
            if self.child_scorer == "mlp":
                mods.append(self.child_mlp)
        return mods

    def _entropy_aux(self, logP_parent: torch.Tensor):
        """Optional Shannon-entropy regulariser on P(j|y) (encourages exploration)."""
        if not (self.training and self.entropy_weight > 0.0):
            return None
        pP = logP_parent.exp()                    # 0 at masked parents
        # torch.where evaluates both branches, so guard the -inf logp BEFORE the
        # multiply: 0 * -inf = NaN would poison the backward even on the zero branch.
        safe_logp = torch.where(pP > 0, logP_parent, torch.zeros_like(logP_parent))
        ent = -(pP * safe_logp).sum(dim=1)        # (B,) Shannon entropy of P(j|y)
        return -self.entropy_weight * ent.mean()  # add to loss → encourages exploration

    def forward(
        self,
        parent_slots: torch.Tensor,        # (B, P, Ds)
        child_slots: Optional[torch.Tensor],  # (B, P, K, Ds); None in parent-only mode
        h_x: Optional[torch.Tensor],       # (B, Ds)  child-routing query ("<y> <x>"), slot-space; None if parent-only
        h_y: torch.Tensor,                 # (B, Ds)  parent-routing query ("<y>"), slot-space
        nonempty: Optional[torch.Tensor] = None,  # (B, P) bool; empty parents masked
        h_readout: Optional[torch.Tensor] = None,  # (B, Ds)  "<y> <x>" readout query, slot-space
        child_color_feats: Optional[torch.Tensor] = None,  # (B,P,K,d_patch); colour evidence when color_source='patch'
        patches: Optional[torch.Tensor] = None,    # (B,N,d_vit) raw patches; color_source='patch_qdot'
        child_masks: Optional[torch.Tensor] = None,  # (B,P,K,N) per-child patch attn; color_source='patch_qdot'
    ):
        """Returns ``(log P(a) (B, C), aux_loss or None)``.

        Colour evidence per ``color_source``: 'slot' reads the child slot vector;
        'patch' reads ``child_color_feats`` (raw DINO patches pooled by each child's
        attention); 'patch_qdot' runs a per-child query·patch dot-product over
        ``patches`` confined by ``child_masks``. Routing is identical in all three.
        """
        # ── Parent routing  P(j|y) ─────────────────────────────────────────
        qy  = self.q_parent(h_y)                                      # (B, Ds)
        r_y = torch.einsum("bd,bpd->bp", qy, parent_slots) * self.scale   # (B, P)
        if nonempty is not None:
            # Guard rows with no non-empty parent (degenerate) → allow all of them.
            nonempty = nonempty | (~nonempty.any(dim=1, keepdim=True))
            r_y = r_y.masked_fill(~nonempty, float("-inf"))
        logP_parent = F.log_softmax(r_y / self.tau, dim=1)            # (B, P)

        # ── Parent-only ablation: read colour straight off the object slot ──
        if not self.use_children:
            #   P(a) = Σ_j P(j|y) · P(a | s_j[, readout])   (no child routing; <x> unused)
            logp_color = self._color_logp(parent_slots, h_readout)            # (B, P, C)
            logP_a = torch.logsumexp(logP_parent[..., None] + logp_color, dim=1)  # (B, C)
            self.last_logP_parent = logP_parent.detach()
            self.last_logP_child  = None
            return logP_a, self._entropy_aux(logP_parent)

        # ── Child routing  P(k|j,x)  (local softmax within each parent) ────
        B, P, K, Ds = child_slots.shape
        qx = self.q_child(h_x)                                        # (B, Ds)
        if self.child_scorer == "mlp":
            hx_e  = qx[:, None, None, :].expand(B, P, K, Ds)
            par_e = parent_slots[:, :, None, :].expand(B, P, K, Ds)
            r_x   = self.child_mlp(torch.cat([hx_e, child_slots, par_e], dim=-1)).squeeze(-1)
        else:
            r_x = torch.einsum("bd,bpkd->bpk", qx, child_slots) * self.scale  # (B, P, K)
        logP_child = F.log_softmax(r_x / self.tau, dim=2)            # (B, P, K)

        # ── Path log-weights  log w_jk = log P(j|y) + log P(k|j,x) ─────────
        logw = logP_parent[:, :, None] + logP_child                  # (B, P, K)

        # ── Per-child colour log-probs and log-space marginalisation ───────
        # slot: child slot vector; patch: raw DINO patches pooled by each child's
        # attention (child_color_feats); patch_qdot: a per-child query·patch dot-product
        # over the raw patches confined to that child's region.
        if self.color_source == "patch_qdot":
            if patches is None or child_masks is None:
                raise RuntimeError("color_source='patch_qdot' needs patches + child_masks")
            logp_color = self.qdot_readout(patches, child_masks, h_readout)  # (B, P, K, C)
        else:
            if self.color_source == "patch":
                if child_color_feats is None:
                    raise RuntimeError("color_source='patch' but child_color_feats is None")
                color_feats = child_color_feats
            else:
                color_feats = child_slots
            logp_color = self._color_logp(color_feats, h_readout)           # (B, P, K, C)
        logP_a = torch.logsumexp(logw[..., None] + logp_color, dim=(1, 2))   # (B, C)

        # Stash the routing distributions for interpretability / viz.
        self.last_logP_parent = logP_parent.detach()
        self.last_logP_child  = logP_child.detach()
        return logP_a, self._entropy_aux(logP_parent)


class QueryCrossAttentionColorHead(nn.Module):
    """Flat query-conditioned cross-attention colour head — the unstructured control
    for :class:`HierRouter`.

    The compound "<y> <x>" query (e.g. "car door", already projected into slot space)
    cross-attends *once* over a flat set of visual tokens — either the object/parent
    slots (ParentSlot-QCA) or the raw projected ViT patch tokens (Patch-QCA) — and the
    single attended vector ``z`` is read out with the *same* query-conditioned colour
    head as the router (``concat[z ; f_readout(h_yx)] → MLP``). There is **no** parent→
    child tree, no P(j|y)/P(k|j,x), and no path marginalisation: this isolates whether
    HierRouter's gains come from the structured routing or merely from the compound
    query + query-conditioned readout pulling the right visual evidence.

    Emits ``log P(a)`` (train with ``nn.NLLLoss``), matching HierRouter so the loss /
    trainer path is identical. ``num_heads`` does not exist in HierRouter (it has no
    multi-head attention), so it is a free hyper-parameter here and does not affect the
    controlled text/readout comparison.
    """

    def __init__(self, d_model: int, num_classes: int, num_heads: int = 8):
        super().__init__()
        self.q_attn     = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln         = nn.LayerNorm(d_model)
        # Mirror HierRouter's query-conditioned readout: f_readout(h_yx) concatenated
        # with the attended visual vector before the 2-layer colour MLP (input 2*d).
        self.f_readout  = nn.Linear(d_model, d_model)
        self.color_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,          # (B, N, D) slots or projected patches
        h_yx: torch.Tensor,                    # (B, D)    projected "<y> <x>" query
        token_mask: Optional[torch.Tensor] = None,  # (B, N) bool, True = valid token
    ):
        """Returns ``(log P(a) (B, C), attn_weights (B, N))``."""
        q = self.q_attn(h_yx).unsqueeze(1)                       # (B, 1, D)
        # nn.MultiheadAttention's key_padding_mask uses True = *pad* (ignored).
        key_padding_mask = ~token_mask if token_mask is not None else None
        attn_out, attn_w = self.cross_attn(
            q, visual_tokens, visual_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        z = self.ln(attn_out.squeeze(1) + q.squeeze(1))          # (B, D) residual + LN
        feat = torch.cat([z, self.f_readout(h_yx)], dim=-1)      # (B, 2D)
        self.last_attn = attn_w.squeeze(1).detach()              # (B, N) for viz parity
        return F.log_softmax(self.color_head(feat), dim=-1), self.last_attn


class PatchQueryDotProductColorHead(nn.Module):
    """Minimal patch baseline for :class:`HierRouter` — the *weakest* unstructured
    control (`pooler='patch_qdot'`).

    Where :class:`QueryCrossAttentionColorHead` uses a learned ``nn.MultiheadAttention``
    (full Q/K/V projections), this head uses a single **explicit query·patch dot-product**:
    the compound "<y> <x>" query (e.g. "car door", already projected into slot space) is
    mapped once into the patch space, dot-producted against the frozen ViT patch tokens,
    softmax-normalised into a patch distribution, and the attended vector ``z`` is read out
    with the *same* query-conditioned colour head as the router/QCA
    (``concat[z ; f_readout(h_yx)] → MLP``). There is **no** learned key/value projection,
    no multi-head attention, no parent→child tree and no path marginalisation: this isolates
    whether a single learned query over flat frozen patches already solves the task.

    Two variants (``project_patches``):
      * **raw** (default, Version B): patches are used as-is in ``d_vit`` space — the only
        learned visual-side parameter is ``q_proj`` (query → patch space). No learned visual
        projection at all.
      * **projected** (Version A): patches are first linearly projected ``d_vit → d_slot``,
        then the dot-product happens in ``d_slot``.

    Emits ``log P(a)`` (train with ``nn.NLLLoss``), matching HierRouter / QCA so the loss,
    trainer and checkpoint paths are identical. ``last_attn`` (the patch distribution) is
    stashed for visualisation (reshape (B, 196) → (B, 14, 14) on the ViT grid).
    """

    def __init__(
        self,
        d_query: int,
        d_vit: int,
        d_slot: int,
        num_classes: int,
        project_patches: bool = False,
        strip_registers: bool = True,
        temperature: float = 1.0,
        normalize: bool = False,
    ):
        super().__init__()
        self.project_patches = project_patches
        self.strip_registers = strip_registers
        self.temperature     = float(temperature)
        self.normalize       = normalize
        # Dot-product space: d_slot if patches are projected first, else raw d_vit.
        self.patch_proj = nn.Linear(d_vit, d_slot) if project_patches else nn.Identity()
        d_attn = d_slot if project_patches else d_vit
        self.d_attn = d_attn
        # Only the query side (and optional patch_proj) is learned on the visual path —
        # NO learned key/value projection (that is the whole point vs QCA).
        self.q_proj    = nn.Linear(d_query, d_attn)
        # Mirror HierRouter/QCA's query-conditioned readout: f_readout(h_yx) concatenated
        # with the attended patch vector before the 2-layer colour MLP (input 2*d_attn).
        self.f_readout = nn.Linear(d_query, d_attn)
        self.color_head = nn.Sequential(
            nn.Linear(2 * d_attn, d_attn), nn.GELU(), nn.Linear(d_attn, num_classes),
        )

    def forward(
        self,
        patches: torch.Tensor,   # (B, N, d_vit) frozen ViT patch+register tokens
        h_yx: torch.Tensor,       # (B, d_query)  projected "<y> <x>" query
    ):
        """Returns ``(log P(a) (B, C), attn (B, Np))``."""
        # DINOv3 cache token order is [reg×4, patch×196] (CLS already dropped) → strip the
        # 4 register tokens to attend over the 196 spatial patches (14×14 grid).
        if self.strip_registers:
            patches = patches[:, 4:]
        patches = self.patch_proj(patches)                       # (B, Np, d_attn)
        q = self.q_proj(h_yx)                                     # (B, d_attn)
        if self.normalize:
            # Cosine similarity: the sqrt(d) scale no longer applies (cosine ∈ [-1, 1]),
            # so only the temperature scales the logits.
            q       = F.normalize(q, dim=-1)
            patches = F.normalize(patches, dim=-1)
            scale   = 1.0 / self.temperature
        else:
            scale   = 1.0 / (math.sqrt(self.d_attn) * self.temperature)
        scores = torch.einsum("bd,bnd->bn", q, patches) * scale  # (B, Np)
        attn   = F.softmax(scores, dim=-1)                       # (B, Np)
        z      = torch.einsum("bn,bnd->bd", attn, patches)       # (B, d_attn)
        feat   = torch.cat([z, self.f_readout(h_yx)], dim=-1)    # (B, 2*d_attn)
        self.last_attn = attn.detach()                           # (B, Np) for viz
        return F.log_softmax(self.color_head(feat), dim=-1), self.last_attn


class ChildMaskedQDotColorHead(nn.Module):
    """Per-child query·patch dot-product colour readout CONFINED to a child's patches.

    The colour evidence for the :class:`HierRouter` patch_qdot mode. It is exactly the
    :class:`PatchQueryDotProductColorHead` mechanism — project the patches and the
    "<y> <x>" query into a common space, score each patch by an explicit dot-product,
    softmax, pool, then ``concat[z ; f_readout(q)] → colour MLP`` — EXCEPT the patch
    attention is restricted to the patches that belong to a given child node: the
    softmax weights are multiplied by that child's (soft) slot-attention mask and
    renormalised over the patches (the same confinement idiom as the recursive child
    slot attention). The query·patch scores are shared across children (the "<y> <x>"
    readout query is the same for the whole question); only the confining mask differs
    per child, so different children read colour from different grounded regions and the
    router's P(k|j,x) picks which one. Returns *per-child* log P(colour | path) (B,P,K,C)
    so the router can marginalise it over paths exactly like the slot/patch colour heads.

    vs HierRouter ``color_source='patch'`` (which pools patches by the slot mask alone):
    here a LEARNED query·patch dot-product reweights the patches *within* each child's
    region, so the colour head can focus on the colour-bearing patches of the part.
    """

    def __init__(
        self,
        d_query: int,
        d_vit: int,
        d_slot: int,
        num_classes: int,
        project_patches: bool = True,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.project_patches = project_patches
        self.temperature     = float(temperature)
        self.patch_proj = nn.Linear(d_vit, d_slot) if project_patches else nn.Identity()
        d_attn = d_slot if project_patches else d_vit
        self.d_attn = d_attn
        self.q_proj    = nn.Linear(d_query, d_attn)
        self.f_readout = nn.Linear(d_query, d_attn)
        # Dropout on the readout feature (regulariser for the finetune regime — the
        # colour head overfits fast). nn.Dropout has no params, so a head trained with
        # dropout=0 loads bit-for-bit into a dropout>0 model (LP-FT init stays valid).
        self.dropout    = nn.Dropout(dropout)
        self.color_head = nn.Sequential(
            nn.Linear(2 * d_attn, d_attn), nn.GELU(), nn.Linear(d_attn, num_classes),
        )

    def forward(
        self,
        patches: torch.Tensor,      # (B, N, d_vit) frozen ViT patches (already register-stripped)
        child_masks: torch.Tensor,  # (B, P, K, N) per-child soft patch assignment (confining mask)
        h_yx: torch.Tensor,         # (B, d_query)  projected "<y> <x>" query
    ):
        """Returns ``log P(colour | path) (B, P, K, C)`` and stashes per-child attention."""
        B, P, K, N = child_masks.shape
        p = self.patch_proj(patches)                              # (B, N, d_attn)
        q = self.q_proj(h_yx)                                     # (B, d_attn)
        scale  = 1.0 / (math.sqrt(self.d_attn) * self.temperature)
        scores = torch.einsum("bd,bnd->bn", q, p) * scale         # (B, N) shared across children
        attn   = F.softmax(scores, dim=-1)                        # (B, N) global qdot distribution
        # Confine to each child's patches: reweight by the (soft) child mask, renormalise.
        a = attn[:, None, None, :] * child_masks.clamp(min=0)     # (B, P, K, N)
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        z = torch.einsum("bpkn,bnd->bpkd", a.to(p.dtype), p)      # (B, P, K, d_attn)
        g = self.f_readout(h_yx)[:, None, None, :].expand(B, P, K, self.d_attn)
        feat = self.dropout(torch.cat([z, g], dim=-1))           # (B, P, K, 2*d_attn)
        self.last_attn = a.detach()                               # (B, P, K, N) for viz
        return F.log_softmax(self.color_head(feat), dim=-1)       # (B, P, K, C)


def _make_classifier_head(d_in: int, num_classes: int, pooler_type: str) -> nn.Module:
    """Builds the head consumed by the chosen pooler.

    ``vqa_transformer`` follows Ding et al. (2021a) and reads the transformed
    CLS through a 2-layer MLP. ``vqa_paper`` uses the paper's exact head
    (``Linear → LayerNorm → Dropout(0.1) → ReLU → Linear``). All other poolers
    use a single linear layer (preserves behaviour and weight shapes of pre-vqa
    checkpoints).
    """
    if pooler_type == "vqa_paper":
        return nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.LayerNorm(d_in),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(d_in, num_classes),
        )
    if pooler_type == "vqa_transformer":
        return nn.Sequential(
            nn.Linear(d_in, d_in), nn.GELU(),
            nn.Linear(d_in, num_classes),
        )
    return nn.Linear(d_in, num_classes)


def _apply_pooler(
    pooler_type: str,
    tokens: torch.Tensor,             # (B, N, d_slot)  — slots OR projected patches
    text_proj: torch.Tensor,          # (B, L, d_slot)
    attention_mask: torch.Tensor,     # (B, L)  1 = real token, 0 = pad
    gated_cross_attn: Optional["GatedCrossAttention"] = None,
    fusion_pooler:    Optional["TransformerFusionPooler"] = None,
    vqa_pooler:       Optional["VQATransformerPooler"] = None,
    vqa_paper_pooler: Optional["VQAPaperPooler"] = None,
    return_attn: bool = False,
):
    """Single source of truth for the pooler dispatch (shared by Slot/PatchClassifier).

    Returns ``(pooled, attn_for_viz, attn_to_text_for_viz)``. The diagnostic
    outputs are only populated when ``return_attn=True`` (used by viz).

    - gated_attn:             tokens ── GCA(text) ──► attn-weighted pool
    - transformer:            tokens ──────────────► transformer fusion ──► CLS
    - gated_then_transformer: tokens ── GCA(text) ──► transformer fusion ──► CLS
    - vqa_transformer:        tokens, text ──► VQA-style joint transformer ──► CLS
    - vqa_paper:              tokens, raw text ──► paper-faithful VQA transformer ──► CLS
    """
    if pooler_type == "vqa_paper":
        pooled, cls_to_img, img_to_text = vqa_paper_pooler(
            tokens, text_proj, attention_mask, return_attn=return_attn,
        )
        if return_attn:
            return pooled, cls_to_img, img_to_text
        return pooled, None, None

    if pooler_type == "vqa_transformer":
        pooled, cls_to_img, img_to_text = vqa_pooler(
            tokens, text_proj, attention_mask, return_attn=return_attn,
        )
        if return_attn:
            return pooled, cls_to_img, img_to_text
        return pooled, None, None

    if pooler_type in ("gated_attn", "gated_then_transformer"):
        key_padding_mask = attention_mask.eq(0)                            # (B, L)
        updated, gca_attn, gca_out = gated_cross_attn(
            tokens, text_proj, key_padding_mask,
        )
        if pooler_type == "gated_attn":
            tok_weights = gca_attn.sum(dim=-1).softmax(dim=-1)             # (B, N)
            pooled      = (updated * tok_weights.unsqueeze(-1)).sum(dim=1) # (B, d_slot)
            if return_attn:
                return pooled, gca_out.norm(dim=-1), gca_attn
            return pooled, None, None
        tokens = updated  # feed GCA output into the transformer

    pooled, cls_to_tokens, token_to_text = fusion_pooler(
        tokens, text_proj, attention_mask, return_attn=return_attn,
    )
    if return_attn:
        return pooled, cls_to_tokens, token_to_text
    return pooled, None, None


# ---------------------------------------------------------------------------
# DINOSAUR loading — only the 3 sub-modules we need
# ---------------------------------------------------------------------------

def _load_dinosaur_submodules(
    experiment_cfg_name: str,
    checkpoint_path: str,
    repo_root: str,
    n_slots: int = 7,
    finetune_ckpt_path: Optional[str] = None,
):
    """Instantiate and load weights for feature_extractor, conditioning,
    perceptual_grouping from a DINOSAUR checkpoint.

    feature_extractor and perceptual_grouping are always loaded strictly from
    the checkpoint.  conditioning weights are loaded only when n_slots matches
    the checkpoint value; otherwise the conditioning is randomly initialised
    (its weights are tiny and slot-attention converges regardless of init).

    Args:
        n_slots: Number of slots for the conditioning module.  Override this
                 to sweep over slot counts without re-training the backbone.

    Returns:
        (feature_extractor, conditioning, perceptual_grouping) — all frozen,
        in eval mode, on CPU.  Move to the desired device after returning.
    """
    import sys
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import ocl.cli._config  # registers OmegaConf resolvers
    import ocl.cli.train    # registers 'training_config' in ConfigStore

    import hydra
    import hydra_zen
    from hydra.core.global_hydra import GlobalHydra

    configs_dir = os.path.join(repo_root, "configs")
    map_location = None if torch.cuda.is_available() else torch.device("cpu")

    # ── Load Hydra config ──────────────────────────────────────────────────
    GlobalHydra.instance().clear()
    try:
        with hydra.initialize_config_dir(config_dir=configs_dir, version_base="1.1"):
            cfg = hydra.compose(
                config_name="training_config",
                overrides=[
                    f"+experiment={experiment_cfg_name}",
                    "models.feature_extractor.pretrained=false",
                    f"models.conditioning.n_slots={n_slots}",
                ],
            )
    finally:
        GlobalHydra.instance().clear()

    # ── Instantiate only the 3 modules we need ────────────────────────────
    feature_extractor   = hydra_zen.instantiate(cfg.models.feature_extractor,  _convert_="all")
    conditioning        = hydra_zen.instantiate(cfg.models.conditioning,        _convert_="all")
    perceptual_grouping = hydra_zen.instantiate(cfg.models.perceptual_grouping, _convert_="all")

    # ── Load matching weights from checkpoint ─────────────────────────────
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(repo_root, checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"DINOSAUR checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    sd   = ckpt["state_dict"]

    def _filtered_sd(prefix):
        """Strip 'models.<prefix>.' from keys that start with it."""
        full_prefix = f"models.{prefix}."
        return {k[len(full_prefix):]: v for k, v in sd.items()
                if k.startswith(full_prefix)}

    feature_extractor.load_state_dict(_filtered_sd("feature_extractor"))
    perceptual_grouping.load_state_dict(_filtered_sd("perceptual_grouping"))

    # Load conditioning only when n_slots matches the checkpoint value.
    # If it differs, keep the random init (nn.Parameter default).
    cond_sd = _filtered_sd("conditioning")
    try:
        conditioning.load_state_dict(cond_sd, strict=True)
        print(f"  Conditioning: loaded from checkpoint (n_slots={n_slots})")
    except RuntimeError:
        ckpt_n = next(iter(cond_sd.values())).shape[1]
        print(f"  Conditioning: random init (requested n_slots={n_slots}, "
              f"checkpoint n_slots={ckpt_n})")

    # ── Optionally overlay fine-tuned weights ────────────────────────────
    if finetune_ckpt_path is not None:
        ft = torch.load(finetune_ckpt_path, map_location=map_location)
        if "conditioning" in ft:
            try:
                conditioning.load_state_dict(ft["conditioning"], strict=True)
                print(f"  Conditioning: loaded from fine-tuned checkpoint")
            except RuntimeError as e:
                print(f"  Conditioning: fine-tuned load failed ({e}), keeping original")
        if "perceptual_grouping" in ft:
            try:
                perceptual_grouping.load_state_dict(ft["perceptual_grouping"], strict=True)
                print(f"  PerceptualGrouping: loaded from fine-tuned checkpoint")
            except RuntimeError as e:
                print(f"  PerceptualGrouping: fine-tuned load failed ({e}), keeping original")

    # ── Freeze all three ──────────────────────────────────────────────────
    for module in (feature_extractor, conditioning, perceptual_grouping):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    return feature_extractor, conditioning, perceptual_grouping


def _build_ftdinosaur(model_name: str):
    """Build a frozen, pretrained FT-DINOSAUR model from ``ftdinosaur_inference``.

    Returns the full ``DINOSAUR`` nn.Module (encoder + slot_init +
    slot_attention + decoder) in eval mode with every parameter frozen.  In the
    classifier we read slots from ``model(images, num_slots=N, decode=False)``
    and never run the decoder.

    Unlike the OCLF path (Lightning ``.ckpt`` loaded via Hydra + the ``routed/``
    wrappers), this is a self-contained module that consumes raw image tensors
    directly. The base14 variant uses slot_dim=256, matching ``d_slot``.
    """
    from ftdinosaur_inference import build_dinosaur

    model = build_dinosaur.build(model_name, pretrained=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SlotClassifier(nn.Module):
    """Full classification pipeline.

    Only ``text_projector``, ``gated_cross_attn``, and ``classifier_head``
    have ``requires_grad=True``.  DINOSAUR sub-modules and RoBERTa are frozen.

    Args:
        dinosaur_cfg_name:  Hydra experiment override string, e.g.
                            ``"projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto"``
        dinosaur_ckpt_path: Absolute or relative (from repo root) path to .ckpt.
        num_classes:        Number of output classes.
        d_slot:             Slot dimension (256 for coco_feat_rec_dino_small16_auto).
        d_text:             RoBERTa-Large hidden size (1024).
        num_heads:          Cross-attention heads (d_slot must be divisible by this).
        roberta_model:      HuggingFace model identifier for RoBERTa.
        repo_root:          Absolute path to the repo root.  Defaults to the
                            directory containing this file.
    """

    def __init__(
        self,
        dinosaur_cfg_name: str,
        dinosaur_ckpt_path: str,
        num_classes: int,
        n_slots: int = 7,
        d_slot: int = 256,
        d_text: int = 1024,
        num_heads: int = 8,
        roberta_model: str = "roberta-large",
        text_encoder_type: str = "roberta",
        t5_model: str = "t5-base",
        vqa_d_model: int = 128,
        load_text_encoder: bool = True,
        repo_root: Optional[str] = None,
        finetune_ckpt_path: Optional[str] = None,
        pooler: str = "gated_attn",
        pooler_layers: int = 2,
        pooler_dropout: float = 0.0,
        zero_image_feats: bool = False,
        text_onehot: bool = False,
        n_questions: Optional[int] = None,
        img_size: int = 224,
        slot_backend: str = "oclf",
        ftdinosaur_model: str = "dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k",
        recursive_infer: bool = False,
        recursive_children: int = 4,
        recursive_parents: int = 1,
        recursive_spread: float = 0.5,
        recursive_include_parents: bool = True,
        rank_method: str = "attention",
        router_temp: float = 1.0,
        router_entropy_weight: float = 0.0,
        child_scorer: str = "bilinear",
        router_use_children: bool = True,
        router_readout_query: bool = True,
        router_color_source: str = "slot",
        router_qdot_project_patches: bool = True,
        router_qdot_dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        valid_backends = ("oclf", "ftdinosaur")
        if slot_backend not in valid_backends:
            raise ValueError(
                f"unknown slot_backend={slot_backend!r}; expected one of {valid_backends}"
            )
        self.slot_backend = slot_backend
        self.n_slots = n_slots
        valid_poolers = ("gated_attn", "transformer", "gated_then_transformer",
                         "vqa_transformer", "vqa_paper", "hier_router", "qca")
        if pooler not in valid_poolers:
            raise ValueError(f"unknown pooler={pooler!r}; expected one of {valid_poolers}")
        self.pooler_type = pooler
        # Aux-loss slot read by the trainer after a hier_router forward (e.g. an
        # entropy regulariser); None for every other pooler / forward path.
        self._aux_loss = None

        # Slot ranking method for the recursive zoom-in pass:
        #  - "attention":   the transformer pooler's CLS->slot attention (cheap, no
        #                   grad, but only a transformer-family pooler exposes it).
        #  - "attribution": gradient saliency ||∂y_c/∂s_i|| of the predicted-answer
        #                   logit w.r.t. each slot (Simonyan et al. 2013; the AwGA
        #                   attribution of Evaluating OCL Beyond Object Discovery).
        #                   Pooler-agnostic — costs one extra backward through the head.
        valid_rank_methods = ("attention", "attribution")
        if rank_method not in valid_rank_methods:
            raise ValueError(
                f"unknown rank_method={rank_method!r}; expected one of {valid_rank_methods}"
            )
        self.rank_method = rank_method

        # Recursive zoom-in inference: after the pooler ranks the slots, refine the
        # top-ranked slot into `recursive_children` finer child slots and re-classify
        # on those alone.
        self.recursive_infer    = bool(recursive_infer)
        self.recursive_children = int(recursive_children)
        self.recursive_parents  = int(recursive_parents)
        # Strength of the spatial-spread prior on the child slot attention.
        # 0 = none (pure frozen SA — may collapse children onto one region),
        # 1 = strong (children forced to maximally-separated spatial cells).
        self.recursive_spread   = float(recursive_spread)
        # Whether the re-classification (second) pass sees the refined parent slot(s)
        # alongside their children (True), or only the children (False).
        self.recursive_include_parents = bool(recursive_include_parents)
        if self.recursive_infer:
            # The CLS->slot attention ranking is only exposed by transformer-family
            # poolers; the gradient-attribution ranking works for any pooler.
            # hier_router ranks pooler-free (by parent-mask mass), so it is exempt.
            if self.rank_method == "attention" and self.pooler_type != "hier_router":
                recursive_poolers = ("transformer", "gated_then_transformer",
                                     "vqa_transformer", "vqa_paper")
                if pooler not in recursive_poolers:
                    raise ValueError(
                        f"recursive_infer=True with rank_method='attention' needs a "
                        f"transformer-family pooler (one of {recursive_poolers}) for the "
                        f"CLS->slot ranking; got {pooler!r}. Use rank_method='attribution' "
                        f"to rank any pooler by gradient saliency instead."
                    )
            if self.recursive_children < 1:
                raise ValueError("recursive_children must be >= 1")
            if self.recursive_parents < 1:
                raise ValueError("recursive_parents must be >= 1")
            if self.recursive_spread < 0:
                raise ValueError("recursive_spread must be >= 0")
            if slot_backend != "oclf":
                raise NotImplementedError(
                    "recursive_infer is currently implemented for the oclf slot backend only."
                )
        if self.pooler_type == "hier_router" and not self.recursive_infer:
            raise ValueError(
                "pooler='hier_router' requires recursive_infer=True (it traverses the "
                "parent→child tree built by the recursive zoom-in)."
            )

        valid_text_encoders = ("roberta", "t5")
        if text_encoder_type not in valid_text_encoders:
            raise ValueError(
                f"unknown text_encoder_type={text_encoder_type!r}; "
                f"expected one of {valid_text_encoders}"
            )
        self.text_encoder_type = text_encoder_type
        self.zero_image_feats = zero_image_feats
        self.text_onehot      = text_onehot
        if text_onehot:
            if pooler == "vqa_paper":
                raise ValueError(
                    "pooler='vqa_paper' is incompatible with text_onehot "
                    "(the paper pooler consumes raw text features, not q-id embeddings)."
                )
            if n_questions is None or n_questions < 1:
                raise ValueError("text_onehot=True requires n_questions >= 1")
            # No language encoder needed: the "text" path is a learned lookup.
            load_text_encoder = False

        if repo_root is None:
            repo_root = os.path.dirname(os.path.abspath(__file__))

        # ── Frozen slot encoder ────────────────────────────────────────────
        if self.slot_backend == "ftdinosaur":
            # Self-contained FT-DINOSAUR module (encoder + slot attention).
            # Its slot_dim is 256, so it only plugs into a d_slot=256 head.
            if d_slot != 256:
                raise ValueError(
                    f"ftdinosaur backend has slot_dim=256 but d_slot={d_slot}; "
                    f"set d_slot=256 (the default)."
                )
            if finetune_ckpt_path is not None:
                raise ValueError(
                    "finetune_ckpt_path is only supported by the oclf backend."
                )
            self.ftdinosaur = _build_ftdinosaur(ftdinosaur_model)
            self.d_vit = d_slot   # hier_router (patch readout) is oclf-only; placeholder
            # No cached-feature path: there is no separate positional grid to
            # carry, the encoder consumes raw images directly.
        else:
            # ── Frozen DINOSAUR sub-modules (OCLF .ckpt) ───────────────────
            fe, cond, pg = _load_dinosaur_submodules(
                dinosaur_cfg_name, dinosaur_ckpt_path, repo_root,
                n_slots=n_slots, finetune_ckpt_path=finetune_ckpt_path,
            )
            self.dino_feature_extractor   = fe
            self.dino_conditioning        = cond
            self.dino_perceptual_grouping = pg

            # Cache the ViT positional grid (same for all images of the configured
            # img_size). Run one dummy forward so we don't need to carry positions
            # through the data pipeline in the cached-features training mode.
            _dummy = torch.zeros(1, 3, img_size, img_size)
            with torch.no_grad():
                _r = {"input": {"image": _dummy, "batch_size": 1}}
                _feat = self.dino_feature_extractor(inputs=_r)
                self.register_buffer("_dino_positions", _feat.positions.detach().cpu())
                # Raw ViT feature dim (384 for ViT-S/16). The hier_router patch-readout
                # colour head reads patches at this width.
                self.d_vit = int(_feat.features.shape[-1])

        # ── Frozen text encoder (optional — not needed when using feat cache) ──
        # RoBERTa-Large (1024-d) or T5-base encoder (768-d). Both expose
        # ``(...).last_hidden_state``, so the downstream code is encoder-agnostic.
        if load_text_encoder:
            if text_encoder_type == "t5":
                from transformers import T5EncoderModel
                self.text_encoder = T5EncoderModel.from_pretrained(t5_model)
            else:
                from transformers import RobertaModel
                self.text_encoder = RobertaModel.from_pretrained(roberta_model)
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.text_encoder = None

        # ── Trainable modules ──────────────────────────────────────────────
        # Normally `text_projector` is a TextProjector(d_text -> d_slot).
        # In text_onehot mode it is an nn.Embedding(n_q, d_slot) — equivalent to
        # "one-hot @ learnable W" but materialising only the looked-up row.
        # For the paper-faithful `vqa_paper` pooler the single text projection
        # lives *inside* the pooler, so this is an Identity and raw encoder
        # features flow straight through. Same attribute name keeps the
        # checkpoint / trainable_parameters bookkeeping uniform.
        if self.pooler_type == "vqa_paper":
            self.text_projector = nn.Identity()
        elif text_onehot:
            self.text_projector = nn.Embedding(n_questions, d_slot)
        else:
            self.text_projector = TextProjector(d_text=d_text, d_slot=d_slot)
        if self.pooler_type in ("gated_attn", "gated_then_transformer"):
            self.gated_cross_attn = GatedCrossAttention(d_slot=d_slot, num_heads=num_heads)
        if self.pooler_type in ("transformer", "gated_then_transformer"):
            self.fusion_pooler = TransformerFusionPooler(
                d_slot=d_slot, num_heads=num_heads,
                num_layers=pooler_layers, dropout=pooler_dropout,
            )
        if self.pooler_type == "vqa_transformer":
            self.vqa_pooler = VQATransformerPooler(
                d_model=d_slot, num_heads=num_heads,
                num_layers=pooler_layers, dropout=pooler_dropout,
            )
        if self.pooler_type == "vqa_paper":
            self.vqa_paper_pooler = VQAPaperPooler(
                d_img=d_slot, d_text=d_text, d_model=vqa_d_model,
                num_heads=num_heads, num_layers=pooler_layers,
            )
        if self.pooler_type == "hier_router":
            # Structured path-marginalisation head; it owns its own colour head and
            # consumes the parent/child tree from the recursive zoom-in, so the flat
            # `classifier_head` below is an unused Identity for this pooler.
            self.hier_router = HierRouter(
                d_slot=d_slot, num_classes=num_classes,
                temperature=router_temp, child_scorer=child_scorer,
                entropy_weight=router_entropy_weight,
                use_children=router_use_children,
                readout_query=router_readout_query,
                color_source=router_color_source,
                d_patch=self.d_vit,
                qdot_project_patches=router_qdot_project_patches,
                qdot_dropout=router_qdot_dropout,
            )
        if self.pooler_type == "qca":
            # Flat (unstructured) control: the "<y> <x>" query cross-attends over the
            # object/parent slots; same query-conditioned colour readout as the router,
            # no parent→child tree / path marginalisation. Emits answer log-probs.
            self.qca_head = QueryCrossAttentionColorHead(
                d_model=d_slot, num_classes=num_classes, num_heads=num_heads,
            )
        # The paper transformer works at vqa_d_model (128); every other pooler
        # reads out at d_slot. hier_router / qca emit answer log-probs directly.
        if self.pooler_type in ("hier_router", "qca"):
            self.classifier_head = nn.Identity()
        else:
            head_dim = vqa_d_model if self.pooler_type == "vqa_paper" else d_slot
            self.classifier_head = _make_classifier_head(head_dim, num_classes, self.pooler_type)

    # ------------------------------------------------------------------
    # Frozen inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_slots(self, images: torch.Tensor) -> torch.Tensor:
        """Run frozen DINOSAUR sub-modules and return slot vectors.

        Calls the three RoutableMixin-wrapped modules directly by building
        the routing dict they expect, without running any decoder.

        Args:
            images: (B, 3, 224, 224) normalised image tensor.
        Returns:
            slots: (B, N_slots, d_slot)
        """
        if self.slot_backend == "ftdinosaur":
            # decode=False → run encoder + slot attention only, skip the decoder.
            out = self.ftdinosaur(images, num_slots=self.n_slots, decode=False)
            return out["slots"]                                  # (B, n_slots, 256)

        B = images.shape[0]

        # The RoutableMixin-wrapped modules read their inputs from the dict
        # passed as ``inputs=``.  We build it incrementally.
        routing: dict = {"input": {"image": images, "batch_size": B}}

        # feature_extractor: video_path = "input.image"
        routing["feature_extractor"] = self.dino_feature_extractor(inputs=routing)

        # conditioning: batch_size_path = "input.batch_size"
        routing["conditioning"] = self.dino_conditioning(inputs=routing)

        # perceptual_grouping: feature_path = "feature_extractor",
        #                      conditioning_path = "conditioning"
        pg_output = self.dino_perceptual_grouping(inputs=routing)

        # PerceptualGroupingOutput.objects: (B, N_slots, d_slot)
        return pg_output.objects

    @torch.no_grad()
    def _extract_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen RoBERTa-Large and return per-token hidden states.

        Returns:
            (B, L, d_text)
        """
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state   # (B, L, 1024)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,          # (B, 3, H, W)
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)  1=real token, 0=pad
    ) -> torch.Tensor:
        """Returns logits of shape (B, num_classes)."""
        # 1. Frozen feature extraction
        slots      = self._extract_slots(images)                             # (B, N, d_slot)
        text_feats = self._extract_text_features(input_ids, attention_mask)  # (B, L, d_text)

        if self.zero_image_feats:
            slots = torch.zeros_like(slots)

        # 2. Project text tokens to slot dimension
        text_proj = self.text_projector(text_feats)                     # (B, L, d_slot)

        # 3. Run the configured pooler → pooled (B, d_slot)
        pooled, _, _ = self._run_pooler(slots, text_proj, attention_mask)
        return self.classifier_head(pooled)                             # (B, num_classes)

    def _run_pooler(self, slots, text_proj, attention_mask, return_attn: bool = False):
        return _apply_pooler(
            pooler_type      = self.pooler_type,
            tokens           = slots,
            text_proj        = text_proj,
            attention_mask   = attention_mask,
            gated_cross_attn = getattr(self, "gated_cross_attn", None),
            fusion_pooler    = getattr(self, "fusion_pooler", None),
            vqa_pooler       = getattr(self, "vqa_pooler", None),
            vqa_paper_pooler = getattr(self, "vqa_paper_pooler", None),
            return_attn      = return_attn,
        )

    # ------------------------------------------------------------------
    # Cached forward (skips frozen ViT and RoBERTa — use precomputed feats)
    # ------------------------------------------------------------------

    def forward_cached(
        self,
        dino_features: torch.Tensor,   # (B, 200, d_vit)  pre-computed ViT patches (DINOv3: 4 reg + 196 patch)
        text_hidden: torch.Tensor,     # (B, L, d_text) RoBERTa states, OR (B,) LongTensor q_ids in text_onehot mode.
        attention_mask: torch.Tensor,  # (B, L)
        spans: Optional[torch.Tensor] = None,  # (B, 4, d_text) span vecs; only the qca head reads ch3 ("<y> <x>")
    ) -> torch.Tensor:
        """Returns logits (B, num_classes).

        Bypasses the frozen ViT and RoBERTa encoders entirely.  Use this during
        the n_slots sweep after running ``precompute_features.py`` once.

        ftdinosaur: ``dino_features`` are cached *encoder* features
        (B, num_patches=256, 768). Slot attention is cheap, so we re-run it live
        (sampling a fresh random slot init each step, matching the non-cached
        path) and only skip the expensive ViT-B/14 forward.
        """
        if self.slot_backend == "ftdinosaur":
            if dino_features.dtype != torch.float32:
                dino_features = dino_features.float()
            with torch.no_grad():
                B = dino_features.shape[0]
                slot_init = self.ftdinosaur.slot_init(B, self.n_slots)
                slots, _  = self.ftdinosaur.slot_attention(dino_features, slot_init)
            if self.zero_image_feats:
                slots = torch.zeros_like(slots)
            text_proj   = self._project_text(text_hidden)
            pooled, _, _ = self._run_pooler(slots, text_proj, attention_mask)
            return self.classifier_head(pooled)

        from ocl.typing import FeatureExtractorOutput

        B = dino_features.shape[0]

        # Caches may be stored in fp16 to save disk; the frozen slot stack runs
        # in fp32. Cast once at the boundary so LayerNorm / MLPs see matching
        # dtype without paying the cast per layer.
        if dino_features.dtype != torch.float32:
            dino_features = dino_features.float()

        # Reconstruct FeatureExtractorOutput from cached patch features.
        feat_out = FeatureExtractorOutput(
            features=dino_features,                                    # (B, N, d_vit)
            positions=self._dino_positions.to(dino_features.device),  # (N, …)
        )

        # ParentSlot-QCA control: flat cross-attention over the object slots, no tree.
        if self.pooler_type == "qca":
            return self._qca_logits_slots(feat_out, B, spans)

        with torch.no_grad():
            routing: dict = {"input": {"batch_size": B}}
            routing["feature_extractor"] = feat_out
            routing["conditioning"]      = self.dino_conditioning(inputs=routing)
            pg_out = self.dino_perceptual_grouping(inputs=routing)

        slots = pg_out.objects  # (B, n_slots, d_slot) — no grad, frozen path

        if self.zero_image_feats:
            slots = torch.zeros_like(slots)

        text_proj = self._project_text(text_hidden)                     # (B, L, d_slot)
        pooled, _, _ = self._run_pooler(slots, text_proj, attention_mask)
        return self.classifier_head(pooled)

    def _qca_logits_slots(self, feat_out, B: int, spans):
        """ParentSlot-QCA: "<y> <x>" query cross-attends over the object slots → log P(a).

        The unstructured control for the hier_router: same frozen slots, same compound
        query (span ch3) and same query-conditioned colour readout, but a single flat
        cross-attention instead of the parent→child routing / path marginalisation.
        """
        if spans is None:
            raise RuntimeError(
                "pooler='qca' requires span features (x_vec/y_vec/…). Build the text "
                "cache with spans (precompute_text(..., with_spans=True) or "
                "--text_in_memory on a 'color of <x> of <y>' CSV)."
            )
        with torch.no_grad():
            slots, slot_masks, _ = self._slots_feats_from_featout(feat_out, B)
        if self.zero_image_feats:
            slots = torch.zeros_like(slots)
        # Mask empty object slots out of the attention, matching the router's P(j|y):
        # >2% of the per-image attention mass = "non-empty"; the OR-guard re-enables all
        # slots for a degenerate row where none clear the threshold (avoids all-pad).
        mass     = slot_masks.sum(dim=-1)                              # (B, n_slots)
        nonempty = mass > 0.02 * mass.sum(dim=1, keepdim=True)
        nonempty = nonempty | (~nonempty.any(dim=1, keepdim=True))
        h_yx = self.text_projector(spans[:, 3].to(dtype=slots.dtype))  # (B, Ds) "<y> <x>"
        logP_a, _ = self.qca_head(slots, h_yx, token_mask=nonempty)
        return logP_a

    def _project_text(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """Unified text→slot-dim projection covering both modes.

        - Normal: text_hidden is (B, L, d_text); TextProjector returns (B, L, d_slot).
        - text_onehot: text_hidden is (B,) LongTensor of question ids; the
          embedding returns (B, d_slot) which is unsqueezed to (B, 1, d_slot)
          so the rest of the pooler stack sees a length-1 "text" sequence.
        """
        if self.text_onehot:
            return self.text_projector(text_hidden).unsqueeze(1)
        return self.text_projector(text_hidden)

    # ------------------------------------------------------------------
    # Recursive (zoom-in) inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _slots_feats_from_featout(self, feat_out, B: int):
        """Run the frozen oclf slot stack, exposing the embedded features.

        Replicates ``SlotAttentionGrouping.forward`` (perceptual_grouping.py:197-208)
        but also returns the positionally-embedded features, which the zoom-in step
        needs in order to re-run the same frozen ``slot_attention`` on a masked subset.

        Returns ``(slots, slot_masks, embedded)`` where
        ``slot_masks`` is the per-slot patch attention (``feature_attributions``).
        """
        pg = self.dino_perceptual_grouping
        routing: dict = {"input": {"batch_size": B}, "feature_extractor": feat_out}
        conditioning = self.dino_conditioning(inputs=routing)
        if pg.positional_embedding is not None:
            embedded = pg.positional_embedding(feat_out.features, feat_out.positions)
        else:
            embedded = feat_out.features
        slots, attn = pg.slot_attention(embedded, conditioning)
        return slots, attn, embedded

    @torch.no_grad()
    def _spatial_child_prior(self, parent_mask):
        """Per-child spatial Gaussian prior over patches → (B, K, N) in [0, 1].

        Picks K anchor patches inside the parent region by weighted farthest-point
        sampling on the patch grid (``self._dino_positions``), then builds a Gaussian
        bump around each anchor whose width adapts to the parent's spatial extent.
        Used to spread the K children across distinct spatial parts of the parent so
        the re-run slot attention doesn't collapse them all onto one region (the
        object's DINO features are too uniform to break symmetry by appearance alone).

        Returns ``None`` if patch positions are unavailable (e.g. non-oclf backend).
        """
        pos = getattr(self, "_dino_positions", None)
        if pos is None:
            return None
        N    = parent_mask.shape[1]
        K    = self.recursive_children
        pos  = pos.to(parent_mask.device).float()           # (N, 2)
        if pos.dim() != 2 or pos.shape[0] != N:
            return None

        w      = parent_mask.clamp(min=0.0)                 # (B, N)
        valid  = w > (0.5 * w.mean(dim=1, keepdim=True))    # patches inside the parent

        # Weighted farthest-point sampling of K anchors (anchor 0 = peak of parent).
        a0      = w.argmax(dim=1)                           # (B,)
        anchors = [a0]
        d2      = ((pos[None] - pos[a0][:, None]) ** 2).sum(-1)  # (B, N)
        for _ in range(1, K):
            nxt = d2.masked_fill(~valid, -1.0).argmax(dim=1)
            anchors.append(nxt)
            d2 = torch.minimum(d2, ((pos[None] - pos[nxt][:, None]) ** 2).sum(-1))
        anchors    = torch.stack(anchors, dim=1)            # (B, K)
        anchor_pos = pos[anchors]                           # (B, K, 2)
        dist2      = ((pos[None, None] - anchor_pos[:, :, None]) ** 2).sum(-1)  # (B, K, N)

        # Adaptive width: scale to the parent's spatial spread. Kept moderately broad
        # (the Gaussians overlap a bit) so children are biased toward different regions
        # without being forced into hard, maximally-separated cells.
        wn       = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        centroid = (wn[:, :, None] * pos[None]).sum(dim=1)                      # (B, 2)
        extent   = (wn[:, :, None] * (pos[None] - centroid[:, None]) ** 2).sum(dim=1).sum(-1).sqrt()
        sigma    = (0.7 * extent).clamp(min=0.05)[:, None, None]                # (B, 1, 1)
        return torch.exp(-dist2 / (2.0 * sigma ** 2))                           # (B, K, N)

    @torch.no_grad()
    def _confined_slot_attention(self, embedded, parent_mask, seeds, spatial_prior=None):
        """Frozen slot attention confined to the parent region → child slots.

        Reuses the frozen ``slot_attention`` weights (to_q/k/v, GRU, norms, ff_mlp)
        but, after the per-patch softmax over child slots, re-weights every patch by
        ``parent_mask`` and renormalises each slot over patches. So each child only
        *aggregates* patches inside the parent region and the K children compete to
        tile it — i.e. a genuine decomposition of the parent into parts.

        This confinement must happen on the *attention* (not the input features):
        ``SlotAttention`` applies ``norm_input`` (a per-token LayerNorm) to the
        features first, which is scale-invariant and therefore cancels any
        multiplicative ``features * parent_mask`` gating — the children would then
        just re-segment the whole scene into objects again.

        ``spatial_prior`` (B, K, N), if given, biases each child toward a different
        region of the parent. Its strength is ``self.recursive_spread`` (the prior is
        raised to that power and applied every iteration): 0 disables it (pure frozen
        SA — children may collapse onto one region), ~0.5 is a gentle, liberal bias
        (children prefer different regions but aren't forced maximally apart), 1 is a
        strong push to maximally-separated spatial cells.

        Args:
            embedded:    (B, N, Df) positionally-embedded features (what the frozen
                         SlotAttentionGrouping feeds to slot_attention).
            parent_mask: (B, N) soft parent membership in [0, 1] (feature_attributions
                         of the chosen slot).
            seeds:       (B, K, Ds) initial child slots.

        Returns ``(child_slots (B, K, Ds), child_attn (B, K, N))`` where ``child_attn``
        is the per-child patch attention already confined to the parent region.
        """
        sa = self.dino_perceptual_grouping.slot_attention
        B, N, _ = embedded.shape
        K       = seeds.shape[1]
        H, dph  = sa.n_heads, sa.dims_per_head
        iters   = sa.iters

        feats = sa.norm_input(embedded)
        k = sa.to_k(feats).view(B, N, H, dph)
        v = sa.to_v(feats).view(B, N, H, dph)
        pm = parent_mask.clamp(min=0.0)[:, None, None, :]   # (B, 1, 1, N)
        gp = spatial_prior[:, :, None, :].clamp(min=1e-6) if spatial_prior is not None else None

        spread = float(getattr(self, "recursive_spread", 0.5))
        slots = seeds
        last_a = None
        for _ in range(iters):
            slots_prev = slots
            s = sa.norm_slots(slots)
            q = sa.to_q(s).view(B, K, H, dph)
            dots = torch.einsum("bihd,bjhd->bihj", q, k) * sa.scale      # (B, K, H, N)
            attn = dots.flatten(1, 2).softmax(dim=1).view(B, K, H, N)    # softmax over slots
            a = attn * pm                                               # confine to parent
            if gp is not None and spread > 0.0:
                a = a * (gp ** spread)                                  # gentle spatial bias
            last_a = a
            a = a + sa.eps
            a = a / a.sum(dim=-1, keepdim=True)                         # renormalise over patches
            updates = torch.einsum("bjhd,bihj->bihd", v, a)
            slots = sa.gru(updates.reshape(-1, sa.kvq_dim), slots_prev.reshape(-1, sa.dim))
            slots = slots.reshape(B, K, sa.dim)
            if sa.ff_mlp is not None:
                slots = sa.ff_mlp(slots)

        child_attn = last_a.mean(dim=2)   # (B, K, N) — confined + spatially-spread assignment
        return slots, child_attn

    @torch.no_grad()
    def _sample_child_seeds(self, B, embedded):
        """K initial child slots sampled from the frozen ``conditioning`` distribution.

        The frozen SA was trained to break symmetry at the conditioning's learned
        ``sigma`` scale (``mu + sigma*randn`` for RandomConditioning), so we seed the
        children from it rather than from a parent slot plus tiny noise (which collapses).
        """
        K = self.recursive_children
        cond = self.dino_conditioning(inputs={"input": {"batch_size": B}})  # (B, n_slots, Ds)
        if cond.shape[1] >= K:
            seeds = cond[:, :K]
        else:
            reps  = (K + cond.shape[1] - 1) // cond.shape[1]
            seeds = cond.repeat(1, reps, 1)[:, :K]
        return seeds.contiguous().to(device=embedded.device, dtype=embedded.dtype)

    @torch.no_grad()
    def _select_parents(self, slots, slot_masks, rank_scores):
        """Pick the top ``recursive_parents`` slots (non-empty preferred).

        Shared by the child-refinement path (``_refine_top_slots``) and the parent-only
        hier_router ablation, so both see *identical* parent slots / ranking. Returns
        ``(parent_slots (B,P,Ds), nonempty (B,P) bool, top_idx (B,P))``.
        """
        B, N_slots, Ds = slots.shape
        P = min(self.recursive_parents, N_slots)
        # Prefer non-empty slots: down-rank slots whose feature-attribution mask carries
        # almost no patches (>2% of the per-image attention mass = "non-empty"), so a
        # parent is never spent on a dead slot — common with many slots and/or an
        # untrained ranker. Empty slots only fill in if fewer than P are non-empty.
        mass     = slot_masks.sum(dim=-1)                                  # (B, N_slots)
        nonempty = mass > 0.02 * mass.sum(dim=1, keepdim=True)
        eff      = rank_scores.masked_fill(~nonempty, float("-inf"))
        top_idx  = eff.topk(P, dim=1).indices          # (B, P), non-empty first, highest score
        parent_slots = torch.gather(slots, 1, top_idx[:, :, None].expand(-1, -1, Ds))  # (B,P,Ds)
        return parent_slots, torch.gather(nonempty, 1, top_idx), top_idx

    @torch.no_grad()
    def _refine_top_slots(self, embedded, slots, slot_masks, rank_scores):
        """Refine the top ``recursive_parents`` slots, each into ``recursive_children``.

        Picks the P = ``recursive_parents`` highest-ranked slots and, for each one
        independently, re-runs the frozen DINOSAUR slot attention *confined to that
        slot's patches* (see ``_confined_slot_attention``) with K fresh slots spread
        across distinct spatial parts (``_spatial_child_prior``). The P*K children are
        concatenated and returned for re-classification. With P=1 this is the original
        single-parent zoom-in.

        Returns ``(child_slots (B, P*K, Ds), info)`` where ``info`` carries:
          - ``child_attn``   (B, P, K, N) per-child patch attention, grouped by parent
          - ``top_idx``      (B, P)        chosen parent slot indices (rank order)
          - ``parent_masks`` (B, P, N)     each chosen slot's patch mask (tree roots)
        """
        B, N_slots, _ = slots.shape
        P = min(self.recursive_parents, N_slots)
        ar = torch.arange(B, device=slots.device)
        parent_slots, nonempty_p, top_idx = self._select_parents(slots, slot_masks, rank_scores)

        child_slots_all, child_attn_all, parent_masks = [], [], []
        for p in range(P):
            idx_p = top_idx[:, p]                       # (B,)
            pmask = slot_masks[ar, idx_p]               # (B, N)
            seeds = self._sample_child_seeds(B, embedded)
            prior = self._spatial_child_prior(pmask)    # (B, K, N) or None
            cs, ca = self._confined_slot_attention(embedded, pmask, seeds, spatial_prior=prior)
            child_slots_all.append(cs)                  # (B, K, Ds)
            child_attn_all.append(ca)                   # (B, K, N)
            parent_masks.append(pmask)                  # (B, N)

        child_slots  = torch.cat(child_slots_all, dim=1)    # (B, P*K, Ds)
        info = {
            "child_attn":    torch.stack(child_attn_all, dim=1),  # (B, P, K, N)
            "top_idx":       top_idx,                             # (B, P)
            "parent_masks":  torch.stack(parent_masks, dim=1),    # (B, P, N)
            "parent_slots":  parent_slots,                        # (B, P, Ds)
            # Which of the chosen parents carry real patch mass (the rest are dead
            # slots only kept to fill P); the hier router masks these out of P(j|y).
            "nonempty":      nonempty_p,                          # (B, P) bool
        }
        return child_slots, info

    def _second_pass_tokens(self, parent_slots, child_slots):
        """Tokens fed to the re-classification pooler: parents+children, or children only."""
        if self.recursive_include_parents:
            return torch.cat([parent_slots, child_slots], dim=1)   # (B, P + P*K, Ds)
        return child_slots                                          # (B, P*K, Ds)

    def _slot_attributions(self, slots, text_proj, attention_mask):
        """Per-slot gradient saliency A_i = ||∂y_c/∂s_i||_2 (Simonyan et al. 2013).

        ``y_c`` is the predicted-answer logit. ``slots`` is treated as the input
        leaf, so the gradient flows only through the trainable pooler + head — the
        frozen DINOSAUR stack is upstream of the leaf and never sees a backward.
        This is the attribution score behind the AwGA metric in *Evaluating
        Object-Centric Models beyond Object Discovery*; here we use it to rank
        slots instead of (or alongside) the pooler's CLS->slot attention.

        Returns ``(B, N_slots)`` non-negative importances, higher = more responsible.
        """
        slots_leaf = slots.detach().requires_grad_(True)
        with torch.enable_grad():                       # override any outer no_grad
            pooled, _, _ = self._run_pooler(slots_leaf, text_proj, attention_mask)
            logits = self.classifier_head(pooled)       # (B, C)
            # Predicted-class logit per example. Summing over the batch is exact:
            # example b's logit depends only on slots[b], so ∂(Σ_b y_b)/∂s_{b,i} = ∂y_b/∂s_{b,i}.
            idx = logits.argmax(dim=1, keepdim=True)    # (B, 1)
            y   = logits.gather(1, idx).sum()
            grad = torch.autograd.grad(y, slots_leaf)[0]  # (B, N, Ds)
        return grad.norm(dim=-1)                        # (B, N)

    def _rank_slots(self, slots, text_proj, attention_mask):
        """Per-slot importance for the recursive ranking pass (``rank_method`` dispatch).

        - ``"attention"``:   transformer pooler's CLS->slot attention (cheap, no grad).
        - ``"attribution"``: gradient saliency (see ``_slot_attributions``), pooler-agnostic.

        Returns ``(B, N_slots)``.
        """
        if self.rank_method == "attribution":
            return self._slot_attributions(slots, text_proj, attention_mask)
        # "attention": only the argmax index is consumed downstream, so no grad is needed.
        with torch.no_grad():
            _, rank_scores, _ = self._run_pooler(
                slots, text_proj, attention_mask, return_attn=True,
            )
        if rank_scores is None:
            raise RuntimeError(
                "pooler returned no CLS->slot ranking; rank_method='attention' requires a "
                "transformer-family pooler (or use rank_method='attribution')."
            )
        return rank_scores

    @staticmethod
    def _pool_child_patches(child_attn, feats):
        """Pool raw patch features by each child's (un-normalised) patch attention.

        ``child_attn`` (B,P,K,N), ``feats`` (B,N,d) → (B,P,K,d). The child attention
        is renormalised over the N patches first, so the pooled vector is a convex
        combination (weighted mean) of the patch features the child grounds to.
        """
        w = child_attn / child_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.einsum("bpkn,bnd->bpkd", w.to(feats.dtype), feats)

    def _hier_router_logits(self, slots, slot_masks, embedded, spans, patch_feats=None):
        """Structured parent→child traversal → answer log-probs (B, num_classes).

        Refines *every* slot into K children (ranking by mask mass only orders them;
        with recursive_parents=n_slots all are kept), then routes language over the
        tree: P(j|y) over parents, P(k|j,x) within each parent, marginalise the
        per-child colour distribution over valid paths. ``spans`` is (B, 4, d_text):
        ch0 = part <x>, ch1 = object <y>, ch2 = "<x> of the <y>" combined phrase
        (parent-only ablation query), ch3 = "<y> <x>" compound noun (colour readout).
        """
        if spans is None:
            raise RuntimeError(
                "pooler='hier_router' requires span features (x_vec, y_vec). Build the "
                "text cache with spans (precompute_text(..., with_spans=True) or "
                "--text_in_memory on a 'color of <x> of <y>' CSV)."
            )
        # Rank by parent-mask mass: a cheap, pooler-free ordering. With
        # recursive_parents=n_slots this keeps all slots (order is irrelevant to the
        # permutation-equivariant router); the non-empty guard still applies.
        rank_scores = slot_masks.sum(dim=-1)

        # Parent-only ablation: no child refinement at all (skips the expensive confined
        # slot-attention), route only over object slots. The query is the FULL
        # "<x> of the <y>" phrase (span ch2, e.g. "knob of the door") — the model still
        # sees the part, it just lacks the hierarchical child level to localise it.
        if not self.hier_router.use_children:
            parent_slots, nonempty, _ = self._select_parents(slots, slot_masks, rank_scores)
            h_q = self.text_projector(spans[:, 2].to(dtype=parent_slots.dtype))  # (B, Ds)
            h_readout = self._readout_query(spans, parent_slots.dtype)
            logP_a, aux = self.hier_router(parent_slots, None, None, h_q, nonempty, h_readout)
            self._aux_loss = aux
            return logP_a

        child_slots, info = self._refine_top_slots(embedded, slots, slot_masks, rank_scores)
        B, P = info["parent_slots"].shape[:2]
        K, Ds = self.recursive_children, info["parent_slots"].shape[-1]
        child_slots = child_slots.view(B, P, K, Ds)               # parent(m)=m//K

        ref = info["parent_slots"]
        # Child routing queries with the FULL "<y> <x>" compound noun ("car door",
        # span ch3) — the part query carries object context — not the bare part <x>
        # (ch0, now unused). Parent routing still uses <y> alone (ch1).
        h_x = self.text_projector(spans[:, 3].to(dtype=ref.dtype))  # (B, Ds) "<y> <x>"
        h_y = self.text_projector(spans[:, 1].to(dtype=ref.dtype))  # (B, Ds) "<y>"
        h_readout = h_x if self.hier_router.readout_query else None  # readout reuses "<y> <x>"
        # Patch-readout mode: pool the raw DINO patches each child grounds to and let the
        # colour head read those instead of the child slot vector (routing is unchanged).
        color_kwargs = {}
        cs = self.hier_router.color_source
        if cs in ("patch", "patch_qdot"):
            feats = patch_feats if patch_feats is not None else embedded
            if cs == "patch":
                color_kwargs["child_color_feats"] = self._pool_child_patches(
                    info["child_attn"], feats)                       # (B,P,K,d_vit)
            else:
                color_kwargs["patches"]     = feats                  # (B,N,d_vit) raw
                color_kwargs["child_masks"] = info["child_attn"]     # (B,P,K,N)
        logP_a, aux = self.hier_router(
            info["parent_slots"], child_slots, h_x, h_y, info.get("nonempty"), h_readout,
            **color_kwargs,
        )
        self._aux_loss = aux
        return logP_a

    def _readout_query(self, spans, dtype):
        """Slot-space "<y> <x>" readout query (span ch3) for the colour head, or None."""
        if not self.hier_router.readout_query:
            return None
        return self.text_projector(spans[:, 3].to(dtype=dtype))  # (B, Ds)

    def _recursive_logits(self, slots, slot_masks, embedded, text_proj, attention_mask,
                          spans=None, patch_feats=None):
        """Rank slots → zoom into the top one → re-classify on its children."""
        if self.pooler_type == "hier_router":
            return self._hier_router_logits(slots, slot_masks, embedded, spans,
                                            patch_feats=patch_feats)
        rank_scores = self._rank_slots(slots, text_proj, attention_mask)
        child_slots, info = self._refine_top_slots(embedded, slots, slot_masks, rank_scores)
        tokens = self._second_pass_tokens(info["parent_slots"], child_slots)
        pooled, _, _ = self._run_pooler(tokens, text_proj, attention_mask)
        return self.classifier_head(pooled)

    def forward_recursive(
        self,
        images: torch.Tensor,          # (B, 3, H, W)
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        spans: Optional[torch.Tensor] = None,  # (B, 4, d_text) x / y / '<x> of the <y>' / '<y> <x>' span vecs (hier_router)
    ) -> torch.Tensor:
        """Recursive zoom-in variant of ``forward`` (non-cached). Logits (B, num_classes)."""
        B = images.shape[0]
        with torch.no_grad():
            routing: dict = {"input": {"image": images, "batch_size": B}}
            routing["feature_extractor"] = self.dino_feature_extractor(inputs=routing)
            slots, slot_masks, embedded = self._slots_feats_from_featout(
                routing["feature_extractor"], B,
            )
            patch_feats = routing["feature_extractor"].features   # (B, N, d_vit) raw DINO
            text_feats = self._extract_text_features(input_ids, attention_mask)
        text_proj = self.text_projector(text_feats)
        return self._recursive_logits(slots, slot_masks, embedded, text_proj, attention_mask,
                                      spans=spans, patch_feats=patch_feats)

    def forward_recursive_cached(
        self,
        dino_features: torch.Tensor,   # (B, N, d_vit)
        text_hidden: torch.Tensor,     # (B, L, d_text)  OR (B,) q_ids in text_onehot mode
        attention_mask: torch.Tensor,  # (B, L)
        spans: Optional[torch.Tensor] = None,  # (B, 4, d_text) x / y / '<x> of the <y>' / '<y> <x>' span vecs (hier_router)
    ) -> torch.Tensor:
        """Recursive zoom-in variant of ``forward_cached``. Logits (B, num_classes)."""
        from ocl.typing import FeatureExtractorOutput

        B = dino_features.shape[0]
        if dino_features.dtype != torch.float32:
            dino_features = dino_features.float()
        feat_out = FeatureExtractorOutput(
            features=dino_features,
            positions=self._dino_positions.to(dino_features.device),
        )
        with torch.no_grad():
            slots, slot_masks, embedded = self._slots_feats_from_featout(feat_out, B)
        text_proj = self._project_text(text_hidden)
        # Raw cached patches (B,N,d_vit) for the hier_router patch-readout colour head.
        return self._recursive_logits(slots, slot_masks, embedded, text_proj, attention_mask,
                                      spans=spans, patch_feats=dino_features)

    @torch.no_grad()
    def forward_recursive_viz(
        self,
        images: torch.Tensor,          # (B, 3, H, W)
        text_hidden: torch.Tensor,     # (B, L, d_text)  pre-encoded text features
        attention_mask: torch.Tensor,  # (B, L)
    ) -> dict:
        """Diagnostic forward for the recursive tree visualisation.

        Returns a dict with everything needed to draw the zoom-in forest (P =
        ``recursive_parents`` roots, each with K = ``recursive_children`` leaves):
          - ``logits``           (B, num_classes)
          - ``top_idx``          (B, P)          chosen parent slot indices (rank order)
          - ``parent_masks``     (B, P, N)       each parent's patch mask (tree roots)
          - ``child_masks``      (B, P, K, N)    per-child patch masks (tree leaves)
          - ``child_importance`` (B, P, K)       pass-2 CLS->child attention; higher = more
                                                 important, used to order children L->R
          - ``slot_rank``        (B, N_slots)    pass-1 CLS->slot ranking (attention or
                                                 attribution, per ``rank_method``)
          - ``slot_attr``        (B, N_slots)    pass-1 gradient-saliency attribution
                                                 ||∂y_c/∂s_i|| of each slot (always, regardless
                                                 of ``rank_method``)
          - ``child_attr``       (B, P, K)       pass-2 gradient-saliency attribution of each
                                                 child token w.r.t. the predicted answer
        """
        B = images.shape[0]
        routing: dict = {"input": {"image": images, "batch_size": B}}
        routing["feature_extractor"] = self.dino_feature_extractor(inputs=routing)
        slots, slot_masks, embedded = self._slots_feats_from_featout(
            routing["feature_extractor"], B,
        )
        text_proj = self.text_projector(text_hidden)

        rank_scores = self._rank_slots(slots, text_proj, attention_mask)
        child_slots, info = self._refine_top_slots(embedded, slots, slot_masks, rank_scores)
        tokens = self._second_pass_tokens(info["parent_slots"], child_slots)
        pooled, rank2, _ = self._run_pooler(
            tokens, text_proj, attention_mask, return_attn=True,
        )
        logits = self.classifier_head(pooled)
        P = info["top_idx"].shape[1]
        K = self.recursive_children
        # In include-parents mode the pooler's CLS->token attention covers the P parent
        # tokens first, then the P*K children — slice off the parents for child ordering.
        if rank2 is None:
            rank2 = torch.zeros(B, tokens.shape[1], device=logits.device)
        c0 = P if self.recursive_include_parents else 0
        child_importance = rank2[:, c0:c0 + P * K].reshape(B, P, K)

        # Gradient-saliency attribution for the tree annotation (always computed,
        # independent of how ranking/selection was done):
        #  - parents: ∂y_c/∂s_i of each first-pass slot.
        #  - children: ∂y_c/∂(token) of each second-pass child token; same predicted
        #    answer y_c as `logits`, so the child shares are w.r.t. the displayed prediction.
        slot_attr  = self._slot_attributions(slots, text_proj, attention_mask)   # (B, N_slots)
        tok_attr   = self._slot_attributions(tokens, text_proj, attention_mask)  # (B, P+P*K or P*K)
        child_attr = tok_attr[:, c0:c0 + P * K].reshape(B, P, K)                  # (B, P, K)
        return {
            "logits":           logits,
            "top_idx":          info["top_idx"],            # (B, P)
            "parent_masks":     info["parent_masks"],       # (B, P, N)
            "child_masks":      info["child_attn"],         # (B, P, K, N)
            "child_importance": child_importance,           # (B, P, K)  attention (legacy)
            "slot_rank":        rank_scores,                # (B, N_slots) ranking signal
            "slot_attr":        slot_attr,                  # (B, N_slots) attribution
            "child_attr":       child_attr,                 # (B, P, K)    attribution
        }

    @torch.no_grad()
    def forward_hier_router_viz(
        self,
        images: torch.Tensor,          # (B, 3, H, W)
        text_hidden: torch.Tensor,     # (B, L, d_text)  pre-encoded text features
        attention_mask: torch.Tensor,  # (B, L)  (unused by the router; uniform call site)
        spans: torch.Tensor,           # (B, 4, d_text)  ch0=<x>, ch1=<y>, ch2='<x> of the <y>', ch3='<y> <x>'
    ) -> dict:
        """Diagnostic forward exposing the hier_router's full routing trace.

        Mirrors ``_hier_router_logits`` but keeps the tree masks and the per-child
        colour distribution so the structured traversal can be drawn:
          - ``logits``       (B, C)       log P(a); argmax == predicted colour
          - ``P_parent``     (B, P)        P(j|y) over parent/object slots
          - ``P_child``      (B, P, K)     P(k|j,x) within each parent (local softmax)
          - ``w``            (B, P, K)     path weights P(j|y)·P(k|j,x) (a (P,K) simplex)
          - ``p_color``      (B, P, K, C)  per-child colour distribution P(a|c_jk)
          - ``parent_masks`` (B, P, N)     each parent's patch mask (tree roots)
          - ``child_attn``   (B, P, K, N)  per-child patch attention (tree leaves)
          - ``top_idx``      (B, P)        chosen parent slot indices (mask-mass order)
          - ``nonempty``     (B, P) bool   which parents carry real patch mass
        """
        if self.pooler_type != "hier_router":
            raise RuntimeError("forward_hier_router_viz requires pooler='hier_router'")
        B = images.shape[0]
        routing: dict = {"input": {"image": images, "batch_size": B}}
        routing["feature_extractor"] = self.dino_feature_extractor(inputs=routing)
        slots, slot_masks, embedded = self._slots_feats_from_featout(
            routing["feature_extractor"], B,
        )
        # Same ranking/refinement the router uses at train time (rank by mask mass).
        rank_scores = slot_masks.sum(dim=-1)

        # Parent-only ablation: no children — draw a degenerate K=1 tree where each
        # parent's single "child" is the parent itself, so the standard panel renders.
        if not self.hier_router.use_children:
            parent_slots, nonempty, top_idx = self._select_parents(slots, slot_masks, rank_scores)
            parent_masks = torch.gather(
                slot_masks, 1, top_idx[:, :, None].expand(-1, -1, slot_masks.shape[-1]),
            )                                                            # (B, P, N)
            h_q = self.text_projector(spans[:, 2].to(dtype=parent_slots.dtype))  # "<x> of the <y>"
            h_readout = self._readout_query(spans, parent_slots.dtype)
            logP_a, _ = self.hier_router(parent_slots, None, None, h_q, nonempty, h_readout)
            logP_parent = self.hier_router.last_logP_parent              # (B, P)
            logp_color  = self.hier_router._color_logp(parent_slots, h_readout)  # (B,P,C)
            return {
                "logits":       logP_a,                       # (B, C) = log P(a)
                "P_parent":     logP_parent.exp(),            # (B, P)
                "P_child":      torch.ones_like(logP_parent)[:, :, None],   # (B, P, 1)
                "w":            logP_parent.exp()[:, :, None],             # (B, P, 1) = P(j|y)
                "p_color":      logp_color.exp()[:, :, None, :],          # (B, P, 1, C)
                "parent_masks": parent_masks,                 # (B, P, N)
                "child_attn":   parent_masks[:, :, None, :],  # (B, P, 1, N) = parent mask
                "top_idx":      top_idx,                      # (B, P)
                "nonempty":     nonempty,                     # (B, P) bool
            }

        child_slots, info = self._refine_top_slots(embedded, slots, slot_masks, rank_scores)
        P, Ds = info["parent_slots"].shape[1], info["parent_slots"].shape[-1]
        K = self.recursive_children
        child_slots = child_slots.view(B, P, K, Ds)               # parent(m) = m // K

        ref = info["parent_slots"]
        h_x = self.text_projector(spans[:, 3].to(dtype=ref.dtype))  # child query = "<y> <x>"
        h_y = self.text_projector(spans[:, 1].to(dtype=ref.dtype))  # parent query = "<y>"
        h_readout = h_x if self.hier_router.readout_query else None
        cs = self.hier_router.color_source
        feats = routing["feature_extractor"].features            # (B, N, d_vit) raw patches
        color_kwargs, child_color = {}, None
        if cs == "patch":
            child_color = self._pool_child_patches(info["child_attn"], feats)  # (B,P,K,d_vit)
            color_kwargs["child_color_feats"] = child_color
        elif cs == "patch_qdot":
            color_kwargs["patches"]     = feats
            color_kwargs["child_masks"] = info["child_attn"]
        logP_a, _ = self.hier_router(
            info["parent_slots"], child_slots, h_x, h_y, info.get("nonempty"), h_readout,
            **color_kwargs,
        )
        # The router stashed log P(j|y) / log P(k|j,x); recombine for the path weights
        # and read the per-child colour distribution from the active colour head.
        logP_parent = self.hier_router.last_logP_parent          # (B, P)
        logP_child  = self.hier_router.last_logP_child           # (B, P, K)
        logw        = logP_parent[:, :, None] + logP_child       # (B, P, K)
        if cs == "patch_qdot":
            logp_color = self.hier_router.qdot_readout(feats, info["child_attn"], h_readout)
        else:
            color_feats = child_color if cs == "patch" else child_slots
            logp_color  = self.hier_router._color_logp(color_feats, h_readout)  # (B,P,K,C)
        return {
            "logits":       logP_a,                  # (B, C) = log P(a)
            "P_parent":     logP_parent.exp(),       # (B, P)
            "P_child":      logP_child.exp(),        # (B, P, K)
            "w":            logw.exp(),              # (B, P, K)
            "p_color":      logp_color.exp(),        # (B, P, K, C)
            "parent_masks": info["parent_masks"],    # (B, P, N)
            "child_attn":   info["child_attn"],      # (B, P, K, N)
            "top_idx":      info["top_idx"],         # (B, P)
            "nonempty":     info.get("nonempty"),    # (B, P) bool
        }

    # ------------------------------------------------------------------
    # Visualisation forward (slot masks + cross-attention weights)
    # ------------------------------------------------------------------

    def forward_with_viz(
        self,
        images: torch.Tensor,          # (B, 3, H, W)
        text_hidden: torch.Tensor,     # (B, L, d_text)  pre-encoded text features
        attention_mask: torch.Tensor,  # (B, L)
    ) -> tuple:
        """Forward pass that also returns slot spatial masks and cross-attention weights.

        Accepts pre-encoded text features so this works whether or not the
        RoBERTa encoder is loaded (i.e. in both cached and non-cached training
        modes).  Call within ``torch.no_grad()`` for efficiency.

        Args:
            images:         (B, 3, H, W) normalised image tensor.
            text_hidden:    (B, L, d_text) RoBERTa hidden states (pre-computed).
            attention_mask: (B, L)  1 = real token, 0 = pad.

        Returns:
            logits:              (B, num_classes)
            slot_masks:          (B, N_slots, N_patches) soft spatial assignments
                                 from slot attention (N_patches = 196 for ViT-S/16 224²).
            cross_attn_weights:  (B, N_slots, L) per-slot attention over text tokens,
                                 averaged across heads.
            slot_ca_norms:       (B, N_slots) L2 norm of the CA output per slot —
                                 how much each slot was moved by the text query.
            slot_attr:           (B, N_slots) gradient-saliency attribution
                                 ||∂y_c/∂s_i|| of the predicted-answer logit w.r.t.
                                 each slot (Simonyan 2013 / AwGA). Pooler-agnostic;
                                 the viz uses this when rank_method='attribution'.
        """
        B = images.shape[0]

        if self.slot_backend == "ftdinosaur":
            with torch.no_grad():
                out = self.ftdinosaur(images, num_slots=self.n_slots, decode=False)
            slots      = out["slots"]       # (B, N_slots, 256)
            slot_masks = out["slot_masks"]  # (B, N_slots, N_patches=256) soft assignment
        else:
            with torch.no_grad():
                routing: dict = {"input": {"image": images, "batch_size": B}}
                routing["feature_extractor"] = self.dino_feature_extractor(inputs=routing)
                routing["conditioning"]      = self.dino_conditioning(inputs=routing)
                pg_out = self.dino_perceptual_grouping(inputs=routing)

            slots      = pg_out.objects               # (B, N_slots, d_slot)
            slot_masks = pg_out.feature_attributions  # (B, N_slots, N_patches)

        text_proj = self.text_projector(text_hidden)                # (B, L, d_slot)

        # _run_pooler returns the diagnostic tensors most relevant to the active
        # pooler:
        #  - gated_attn → (||CA_out|| per slot, slot×text attn weights)
        #  - transformer / gated_then_transformer → (CLS→slot attn, slot→text attn)
        pooled, slot_ca_norms, cross_attn_weights = self._run_pooler(
            slots, text_proj, attention_mask, return_attn=True,
        )
        logits = self.classifier_head(pooled)                       # (B, num_classes)

        # Gradient-saliency attribution per slot (cheap extra backward through the
        # pooler+head; DINOSAUR stays frozen since `slots` is detached inside).
        slot_attr = self._slot_attributions(slots, text_proj, attention_mask)
        return logits, slot_masks, cross_attn_weights, slot_ca_norms, slot_attr

    def freeze_routing(self):
        """Freeze the hier_router ROUTING path + the text projector, leaving the
        colour head (``f_readout`` + ``color_head``) trainable.

        For training a new (e.g. patch-sourced) colour head on top of a previously
        trained routing: P(j|y) and P(k|j,x) — and the text projection that feeds
        them — are held fixed at their proven values, so only the colour readout is
        learned. Pair with ``--router_init_ckpt`` to load that proven routing first.
        """
        if not hasattr(self, "hier_router"):
            raise RuntimeError("freeze_routing requires pooler='hier_router'")
        for p in self.text_projector.parameters():
            p.requires_grad_(False)
        for m in self.hier_router.routing_modules():
            for p in m.parameters():
                p.requires_grad_(False)

    def trainable_parameters(self):
        """Return only the parameters that should be optimised."""
        params = (
            list(self.text_projector.parameters())
            + list(self.classifier_head.parameters())
        )
        if hasattr(self, "gated_cross_attn"):
            params += list(self.gated_cross_attn.parameters())
        if hasattr(self, "fusion_pooler"):
            params += list(self.fusion_pooler.parameters())
        if hasattr(self, "vqa_pooler"):
            params += list(self.vqa_pooler.parameters())
        if hasattr(self, "vqa_paper_pooler"):
            params += list(self.vqa_paper_pooler.parameters())
        if hasattr(self, "hier_router"):
            params += list(self.hier_router.parameters())
        if hasattr(self, "qca_head"):
            params += list(self.qca_head.parameters())
        return params


# ---------------------------------------------------------------------------
# Control model: raw ViT patch features instead of DINOSAUR slots
# ---------------------------------------------------------------------------

class PatchClassifier(nn.Module):
    """Control classifier: identical head to SlotClassifier but uses raw ViT
    patch tokens instead of slot-attention outputs.

    Architecture
    ------------
    1. dino_features (B, 200, d_vit) — frozen ViT-S/16 patch+register tokens
       (precomputed; same cache as SlotClassifier)
    2. patch_projector  Linear(d_vit, d_slot)               [trainable]
    3. text query → RoBERTa → TextProjector → (B, L, d_slot)
    4. Pooler — same choice as SlotClassifier (gated_attn / transformer /
       gated_then_transformer).                              [trainable]
    5. linear head → logits

    All trainable modules use the *same* pooler choice as the matching
    SlotClassifier run, so slot-vs-patch comparisons stay apples-to-apples.

    Args:
        d_vit:   ViT feature dimension (384 for ViT-S/16 / DINOv3-small).
        d_slot:  Projection target dimension — same as SlotClassifier for a
                 fair comparison (default 256).
        pooler:  'gated_attn' (default), 'transformer', or 'gated_then_transformer'.
    """

    def __init__(
        self,
        dinosaur_cfg_name: str,
        dinosaur_ckpt_path: str,
        num_classes: int,
        d_vit: int = 384,
        d_slot: int = 256,
        d_text: int = 1024,
        num_heads: int = 8,
        roberta_model: str = "roberta-large",
        text_encoder_type: str = "roberta",
        t5_model: str = "t5-base",
        vqa_d_model: int = 128,
        load_text_encoder: bool = True,
        repo_root: Optional[str] = None,
        pooler: str = "gated_attn",
        pooler_layers: int = 2,
        pooler_dropout: float = 0.0,
        zero_image_feats: bool = False,
        text_onehot: bool = False,
        n_questions: Optional[int] = None,
        patch_qdot_project_patches: bool = False,
        patch_qdot_strip_registers: bool = True,
        patch_qdot_temperature: float = 1.0,
        patch_qdot_normalize: bool = False,
    ):
        super().__init__()
        valid_poolers = ("gated_attn", "transformer", "gated_then_transformer",
                         "vqa_transformer", "vqa_paper", "qca", "patch_qdot")
        if pooler not in valid_poolers:
            raise ValueError(f"unknown pooler={pooler!r}; expected one of {valid_poolers}")
        self.pooler_type = pooler
        valid_text_encoders = ("roberta", "t5")
        if text_encoder_type not in valid_text_encoders:
            raise ValueError(
                f"unknown text_encoder_type={text_encoder_type!r}; "
                f"expected one of {valid_text_encoders}"
            )
        self.text_encoder_type = text_encoder_type
        self.zero_image_feats = zero_image_feats
        self.text_onehot      = text_onehot
        if text_onehot:
            if pooler == "vqa_paper":
                raise ValueError(
                    "pooler='vqa_paper' is incompatible with text_onehot "
                    "(the paper pooler consumes raw text features, not q-id embeddings)."
                )
            if n_questions is None or n_questions < 1:
                raise ValueError("text_onehot=True requires n_questions >= 1")
            load_text_encoder = False

        if repo_root is None:
            repo_root = os.path.dirname(os.path.abspath(__file__))

        # Load only the feature extractor; discard conditioning + pg.
        fe, _, _ = _load_dinosaur_submodules(
            dinosaur_cfg_name, dinosaur_ckpt_path, repo_root, n_slots=7,
        )
        self.dino_feature_extractor = fe

        # Frozen text encoder (RoBERTa-Large / T5-base encoder; not needed in cached mode)
        if load_text_encoder:
            if text_encoder_type == "t5":
                from transformers import T5EncoderModel
                self.text_encoder = T5EncoderModel.from_pretrained(t5_model)
            else:
                from transformers import RobertaModel
                self.text_encoder = RobertaModel.from_pretrained(roberta_model)
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.text_encoder = None

        # Trainable modules. For `vqa_paper` the paper applies its single linear
        # projection directly to raw features, so both projectors are Identity
        # and the pooler consumes raw patch (d_vit) and text (d_text) tokens.
        if self.pooler_type == "vqa_paper":
            self.patch_projector = nn.Identity()
            self.text_projector  = nn.Identity()
        elif self.pooler_type == "patch_qdot":
            # patch_qdot's head owns any patch projection (raw d_vit or projected d_slot),
            # so the classifier-level patch_projector stays Identity (no unused Linear in
            # the optimizer / checkpoint). Text path matches qca (TextProjector → d_slot).
            self.patch_projector = nn.Identity()
            self.text_projector  = TextProjector(d_text=d_text, d_slot=d_slot)
        else:
            self.patch_projector = nn.Linear(d_vit, d_slot)
            if text_onehot:
                self.text_projector = nn.Embedding(n_questions, d_slot)
            else:
                self.text_projector = TextProjector(d_text=d_text, d_slot=d_slot)
        if self.pooler_type in ("gated_attn", "gated_then_transformer"):
            self.gated_cross_attn = GatedCrossAttention(d_slot=d_slot, num_heads=num_heads)
        if self.pooler_type in ("transformer", "gated_then_transformer"):
            self.fusion_pooler = TransformerFusionPooler(
                d_slot=d_slot, num_heads=num_heads,
                num_layers=pooler_layers, dropout=pooler_dropout,
            )
        if self.pooler_type == "vqa_transformer":
            self.vqa_pooler = VQATransformerPooler(
                d_model=d_slot, num_heads=num_heads,
                num_layers=pooler_layers, dropout=pooler_dropout,
            )
        if self.pooler_type == "vqa_paper":
            self.vqa_paper_pooler = VQAPaperPooler(
                d_img=d_vit, d_text=d_text, d_model=vqa_d_model,
                num_heads=num_heads, num_layers=pooler_layers,
            )
        if self.pooler_type == "qca":
            # Patch-QCA control: the "<y> <x>" query cross-attends over the projected
            # raw ViT patch tokens (no slots). Same head as ParentSlot-QCA → log-probs.
            self.qca_head = QueryCrossAttentionColorHead(
                d_model=d_slot, num_classes=num_classes, num_heads=num_heads,
            )
        if self.pooler_type == "patch_qdot":
            # Patch-QDot control: the *weakest* flat baseline — the "<y> <x>" query
            # (projected to d_slot) dot-products over the frozen ViT patch tokens (no
            # learned key/value, no multi-head). Same query-conditioned colour readout
            # as the router/QCA → log-probs. raw (d_vit) vs projected (d_slot) per flag.
            self.patch_qdot_head = PatchQueryDotProductColorHead(
                d_query=d_slot, d_vit=d_vit, d_slot=d_slot, num_classes=num_classes,
                project_patches=patch_qdot_project_patches,
                strip_registers=patch_qdot_strip_registers,
                temperature=patch_qdot_temperature,
                normalize=patch_qdot_normalize,
            )
        if self.pooler_type in ("qca", "patch_qdot"):
            self.classifier_head = nn.Identity()
        else:
            head_dim = vqa_d_model if self.pooler_type == "vqa_paper" else d_slot
            self.classifier_head = _make_classifier_head(head_dim, num_classes, self.pooler_type)

    def forward_cached(
        self,
        dino_features: torch.Tensor,   # (B, 200, d_vit)
        text_hidden: torch.Tensor,     # (B, L, d_text)
        attention_mask: torch.Tensor,  # (B, L)
        spans: Optional[torch.Tensor] = None,  # (B, 4, d_text) span vecs; only qca reads ch3 ("<y> <x>")
    ) -> torch.Tensor:
        """Returns logits (B, num_classes). Same signature as SlotClassifier.forward_cached."""
        # Patch-QCA: project patches, then cross-attend the "<y> <x>" query over all
        # tokens (registers included, no mask) → query-conditioned colour log-probs.
        if self.pooler_type == "qca":
            if spans is None:
                raise RuntimeError(
                    "pooler='qca' requires span features. Build the text cache with spans "
                    "(precompute_text(..., with_spans=True) / --text_in_memory)."
                )
            patches = self.patch_projector(
                dino_features.float() if dino_features.dtype != torch.float32 else dino_features
            )                                                          # (B, N, d_slot)
            if self.zero_image_feats:
                patches = torch.zeros_like(patches)
            h_yx = self.text_projector(spans[:, 3].to(dtype=patches.dtype))  # (B, d_slot)
            logP_a, _ = self.qca_head(patches, h_yx)
            return logP_a

        # Patch-QDot: the "<y> <x>" query (projected to d_slot) dot-products over the raw
        # frozen patch tokens (the head strips the 4 registers + optionally projects them)
        # → query-conditioned colour log-probs. Weakest flat control (no learned K/V).
        if self.pooler_type == "patch_qdot":
            if spans is None:
                raise RuntimeError(
                    "pooler='patch_qdot' requires span features. Build the text cache with "
                    "spans (precompute_text(..., with_spans=True) / --text_in_memory)."
                )
            patches = (
                dino_features.float() if dino_features.dtype != torch.float32 else dino_features
            )                                                          # (B, N, d_vit) raw
            if self.zero_image_feats:
                patches = torch.zeros_like(patches)
            h_yx = self.text_projector(spans[:, 3].to(dtype=patches.dtype))  # (B, d_slot)
            logP_a, _ = self.patch_qdot_head(patches, h_yx)
            return logP_a

        # vqa_paper: patch_projector is Identity → pooler's img_proj consumes raw
        # d_vit patches (faithful single-linear projection).
        patches   = self.patch_projector(dino_features)               # (B, 200, d_slot or d_vit)
        if self.zero_image_feats:
            patches = torch.zeros_like(patches)
        if self.text_onehot:
            text_proj = self.text_projector(text_hidden).unsqueeze(1) # (B, 1, d_slot)
        else:
            text_proj = self.text_projector(text_hidden)              # (B, L, d_slot or d_text)
        pooled, _, _ = _apply_pooler(
            pooler_type      = self.pooler_type,
            tokens           = patches,
            text_proj        = text_proj,
            attention_mask   = attention_mask,
            gated_cross_attn = getattr(self, "gated_cross_attn", None),
            fusion_pooler    = getattr(self, "fusion_pooler", None),
            vqa_pooler       = getattr(self, "vqa_pooler", None),
            vqa_paper_pooler = getattr(self, "vqa_paper_pooler", None),
        )
        return self.classifier_head(pooled)

    def trainable_parameters(self):
        params = (
            list(self.patch_projector.parameters())
            + list(self.text_projector.parameters())
            + list(self.classifier_head.parameters())
        )
        if hasattr(self, "gated_cross_attn"):
            params += list(self.gated_cross_attn.parameters())
        if hasattr(self, "fusion_pooler"):
            params += list(self.fusion_pooler.parameters())
        if hasattr(self, "vqa_pooler"):
            params += list(self.vqa_pooler.parameters())
        if hasattr(self, "vqa_paper_pooler"):
            params += list(self.vqa_paper_pooler.parameters())
        if hasattr(self, "hier_router"):
            params += list(self.hier_router.parameters())
        if hasattr(self, "qca_head"):
            params += list(self.qca_head.parameters())
        if hasattr(self, "patch_qdot_head"):
            params += list(self.patch_qdot_head.parameters())
        return params


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DINO_CFG  = "projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto"
    DINO_CKPT = "checkpoints/epoch_67-step_500000_coco.ckpt"

    if not os.path.exists(DINO_CKPT):
        print(f"Checkpoint not found at {DINO_CKPT}. "
              f"Run from the repo root.", file=sys.stderr)
        sys.exit(1)

    print("Building SlotClassifier (loads RoBERTa-Large once)…")
    model = SlotClassifier(
        dinosaur_cfg_name=DINO_CFG,
        dinosaur_ckpt_path=DINO_CKPT,
        num_classes=10,
    )

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,}  /  Total: {n_total:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    B    = 2
    imgs = torch.randn(B, 3, 224, 224, device=device)
    ids  = torch.ones(B, 64, dtype=torch.long, device=device)
    mask = torch.ones(B, 64, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(imgs, ids, mask)
    print(f"Output logits shape: {logits.shape}")   # should be (2, 10)
    print("Smoke test passed.")
