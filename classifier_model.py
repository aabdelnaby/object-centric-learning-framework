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
        load_text_encoder: bool = True,
        repo_root: Optional[str] = None,
        finetune_ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        if repo_root is None:
            repo_root = os.path.dirname(os.path.abspath(__file__))

        # ── Frozen DINOSAUR sub-modules ────────────────────────────────────
        fe, cond, pg = _load_dinosaur_submodules(
            dinosaur_cfg_name, dinosaur_ckpt_path, repo_root,
            n_slots=n_slots, finetune_ckpt_path=finetune_ckpt_path,
        )
        self.dino_feature_extractor   = fe
        self.dino_conditioning        = cond
        self.dino_perceptual_grouping = pg

        # Cache the ViT positional grid (same for all 224×224 images).
        # Run one dummy forward so we don't need to carry positions through
        # the data pipeline in the cached-features training mode.
        _dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            _r = {"input": {"image": _dummy, "batch_size": 1}}
            _feat = self.dino_feature_extractor(inputs=_r)
            self.register_buffer("_dino_positions", _feat.positions.detach().cpu())

        # ── Frozen RoBERTa-Large (optional — not needed when using feat cache) ──
        if load_text_encoder:
            from transformers import RobertaModel
            self.text_encoder = RobertaModel.from_pretrained(roberta_model)
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.text_encoder = None

        # ── Trainable modules ──────────────────────────────────────────────
        self.text_projector    = TextProjector(d_text=d_text, d_slot=d_slot)
        self.gated_cross_attn  = GatedCrossAttention(d_slot=d_slot, num_heads=num_heads)
        self.classifier_head   = nn.Linear(d_slot, num_classes)

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

        # 2. Project text tokens to slot dimension
        text_proj = self.text_projector(text_feats)                     # (B, L, d_slot)

        # 3. Gated cross-attention
        #    key_padding_mask: True for padding positions (ignored by attn)
        key_padding_mask = attention_mask.eq(0)                         # (B, L)
        slots, attn_weights, _ = self.gated_cross_attn(slots, text_proj, key_padding_mask)

        # 4. Attention-weighted pool slots → classify
        slot_weights = attn_weights.sum(dim=-1).softmax(dim=-1)         # (B, N)
        pooled = (slots * slot_weights.unsqueeze(-1)).sum(dim=1)        # (B, d_slot)
        return self.classifier_head(pooled)                             # (B, num_classes)

    # ------------------------------------------------------------------
    # Cached forward (skips frozen ViT and RoBERTa — use precomputed feats)
    # ------------------------------------------------------------------

    def forward_cached(
        self,
        dino_features: torch.Tensor,   # (B, 200, d_vit)  pre-computed ViT patches (DINOv3: 4 reg + 196 patch)
        text_hidden: torch.Tensor,     # (B, L, d_text)   pre-computed RoBERTa states
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Returns logits (B, num_classes).

        Bypasses the frozen ViT and RoBERTa encoders entirely.  Use this during
        the n_slots sweep after running ``precompute_features.py`` once.
        """
        from ocl.typing import FeatureExtractorOutput

        B = dino_features.shape[0]

        # Reconstruct FeatureExtractorOutput from cached patch features.
        feat_out = FeatureExtractorOutput(
            features=dino_features,                                    # (B, 200, d_vit)
            positions=self._dino_positions.to(dino_features.device),  # (200, …)
        )

        with torch.no_grad():
            routing: dict = {"input": {"batch_size": B}}
            routing["feature_extractor"] = feat_out
            routing["conditioning"]      = self.dino_conditioning(inputs=routing)
            pg_out = self.dino_perceptual_grouping(inputs=routing)

        slots = pg_out.objects  # (B, n_slots, d_slot) — no grad, frozen path

        # Trainable path
        text_proj        = self.text_projector(text_hidden)             # (B, L, d_slot)
        key_padding_mask = attention_mask.eq(0)                         # (B, L)
        slots, attn_weights, _ = self.gated_cross_attn(slots, text_proj, key_padding_mask)
        slot_weights     = attn_weights.sum(dim=-1).softmax(dim=-1)     # (B, N)
        pooled           = (slots * slot_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_slot)
        return self.classifier_head(pooled)

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
        """
        B = images.shape[0]

        with torch.no_grad():
            routing: dict = {"input": {"image": images, "batch_size": B}}
            routing["feature_extractor"] = self.dino_feature_extractor(inputs=routing)
            routing["conditioning"]      = self.dino_conditioning(inputs=routing)
            pg_out = self.dino_perceptual_grouping(inputs=routing)

        slots      = pg_out.objects               # (B, N_slots, d_slot)
        slot_masks = pg_out.feature_attributions  # (B, N_slots, N_patches)

        text_proj        = self.text_projector(text_hidden)         # (B, L, d_slot)
        key_padding_mask = attention_mask.eq(0)                     # (B, L)

        updated_slots, cross_attn_weights, ca_out = self.gated_cross_attn(
            slots, text_proj, key_padding_mask,
        )  # cross_attn_weights: (B, N_slots, L), ca_out: (B, N_slots, d_slot)

        slot_weights = cross_attn_weights.sum(dim=-1).softmax(dim=-1)           # (B, N_slots)
        pooled       = (updated_slots * slot_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_slot)
        logits = self.classifier_head(pooled)                       # (B, num_classes)

        # Slot importance = L2 norm of the CA output per slot:
        # tells us how much each slot was moved by the text query.
        slot_ca_norms = ca_out.norm(dim=-1)                         # (B, N_slots)

        return logits, slot_masks, cross_attn_weights, slot_ca_norms

    def trainable_parameters(self):
        """Return only the parameters that should be optimised."""
        return (
            list(self.text_projector.parameters())
            + list(self.gated_cross_attn.parameters())
            + list(self.classifier_head.parameters())
        )


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
    2. patch_projector  Linear(d_vit, d_slot)       [trainable]
    3. text query → RoBERTa → TextProjector → (B, L, d_slot)
    4. GatedCrossAttention: patch tokens attend to text  [trainable]
    5. mean-pool over patch tokens → linear head → logits

    Only patch_projector, text_projector, gated_cross_attn, and classifier_head
    are trainable.  The ViT feature_extractor and RoBERTa are frozen.

    Args:
        d_vit:   ViT feature dimension (384 for ViT-S/16 / DINOv3-small).
        d_slot:  Projection target dimension — same as SlotClassifier for a
                 fair comparison (default 256).
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
        load_text_encoder: bool = True,
        repo_root: Optional[str] = None,
    ):
        super().__init__()

        if repo_root is None:
            repo_root = os.path.dirname(os.path.abspath(__file__))

        # Load only the feature extractor; discard conditioning + pg.
        fe, _, _ = _load_dinosaur_submodules(
            dinosaur_cfg_name, dinosaur_ckpt_path, repo_root, n_slots=7,
        )
        self.dino_feature_extractor = fe

        # Frozen RoBERTa (not needed in cached mode)
        if load_text_encoder:
            from transformers import RobertaModel
            self.text_encoder = RobertaModel.from_pretrained(roberta_model)
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.text_encoder = None

        # Trainable modules — identical to SlotClassifier except for patch_projector
        self.patch_projector  = nn.Linear(d_vit, d_slot)
        self.text_projector   = TextProjector(d_text=d_text, d_slot=d_slot)
        self.gated_cross_attn = GatedCrossAttention(d_slot=d_slot, num_heads=num_heads)
        self.classifier_head  = nn.Linear(d_slot, num_classes)

    def forward_cached(
        self,
        dino_features: torch.Tensor,   # (B, 200, d_vit)
        text_hidden: torch.Tensor,     # (B, L, d_text)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Returns logits (B, num_classes). Same signature as SlotClassifier.forward_cached."""
        patches          = self.patch_projector(dino_features)        # (B, 200, d_slot)
        text_proj        = self.text_projector(text_hidden)           # (B, L, d_slot)
        key_padding_mask = attention_mask.eq(0)                       # (B, L)
        patches          = self.gated_cross_attn(patches, text_proj, key_padding_mask)
        pooled           = patches.mean(dim=1)                        # (B, d_slot)
        return self.classifier_head(pooled)

    def trainable_parameters(self):
        return (
            list(self.patch_projector.parameters())
            + list(self.text_projector.parameters())
            + list(self.gated_cross_attn.parameters())
            + list(self.classifier_head.parameters())
        )


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
