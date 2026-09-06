"""Frozen DINOSAUR backbone: DINOv3 ViT-S/16 patch features + slot attention.

The slot module is trained once for feature reconstruction on COCO
(``experiments/01_train_dinosaur.sh`` with ``configs/experiment/dinosaur/dinov3_small16_coco.yaml``)
and is used frozen everywhere else. This module rebuilds the three sub-modules the downstream
code needs (feature extractor, conditioning prior, slot-attention grouping) from the same OCL
classes the Hydra config instantiates, and loads their weights from the Lightning checkpoint.
No decoder, no Hydra and no Lightning are needed at runtime.

The architecture constants below mirror ``_base_feature_recon.yaml`` + ``dinov3_small16_coco.yaml``;
``tests/test_dinosaur.py`` checks that the two constructions agree parameter-for-parameter.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import nn

from ocl.conditioning import RandomConditioning
from ocl.feature_extractors.timm import TimmFeatureExtractor
from ocl.feature_extractors.utils import transformer_compute_positions
from ocl.neural_networks import build_two_layer_mlp
from ocl.neural_networks.positional_embedding import DummyPositionEmbed
from ocl.neural_networks.wrappers import Sequential
from ocl.perceptual_grouping import SlotAttentionGrouping
from ocl.typing import FeatureExtractorOutput

BACKBONE = "vit_small_patch16_dinov3.lvd1689m"   # timm name of the frozen DINOv3 ViT-S/16
FEATURE_LEVEL = 12                                 # last block; CLS + 4 register tokens dropped
D_VIT = 384                                        # ViT-S embedding dim
D_SLOT = 256                                       # slot dim
IMG_SIZE = 224
PATCH_GRID = 14                                    # 224 / 16
N_PATCHES = PATCH_GRID * PATCH_GRID                # 196 patch tokens per image


def build_modules(n_slots: int) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Instantiate (feature_extractor, conditioning, grouping) exactly as the training config does."""
    feature_extractor = TimmFeatureExtractor(
        model_name=BACKBONE, pretrained=False, freeze=True, feature_level=FEATURE_LEVEL,
    )
    conditioning = RandomConditioning(object_dim=D_SLOT, n_slots=n_slots)
    grouping = SlotAttentionGrouping(
        feature_dim=D_SLOT,
        object_dim=D_SLOT,
        use_projection_bias=False,
        positional_embedding=Sequential(
            DummyPositionEmbed(),
            build_two_layer_mlp(D_VIT, D_SLOT, D_VIT, initial_layer_norm=True),
        ),
        ff_mlp=build_two_layer_mlp(D_SLOT, D_SLOT, 4 * D_SLOT, initial_layer_norm=True, residual=True),
    )
    return feature_extractor, conditioning, grouping


def patch_positions(img_size: int = IMG_SIZE) -> torch.Tensor:
    """(N, 2) normalised grid coordinates of the ViT patch tokens (same as the extractor emits)."""
    n_side = img_size // 16
    dummy = torch.zeros(1, n_side * n_side, 1)
    return transformer_compute_positions(dummy, image_size=(img_size, img_size))


class FrozenDinosaur(nn.Module):
    """Frozen DINOv3 encoder + DINOSAUR conditioning prior + slot attention.

    Args:
        n_slots:    number of object-level slots M sampled from the conditioning prior.
        checkpoint: path to the Lightning ``.ckpt`` of the pretrained DINOSAUR; ``None`` keeps
                    random weights (tests only).
    """

    def __init__(self, n_slots: int, checkpoint: Optional[str] = None):
        super().__init__()
        self.feature_extractor, self.conditioning, self.grouping = build_modules(n_slots)
        self.register_buffer("positions", patch_positions(IMG_SIZE))
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ── frozen for good: .train() never switches these modules to training mode ──
    def train(self, mode: bool = True):
        return super().train(False)

    @property
    def n_slots(self) -> int:
        return self.conditioning.n_slots

    @property
    def slot_attention(self):
        return self.grouping.slot_attention

    def load_checkpoint(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DINOSAUR checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")["state_dict"]

        def sub(prefix: str) -> dict:
            prefix = f"models.{prefix}."
            return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

        self.feature_extractor.load_state_dict(sub("feature_extractor"))
        self.grouping.load_state_dict(sub("perceptual_grouping"))
        # slots_mu / slots_logsigma are (1, 1, D): independent of the slot count.
        self.conditioning.load_state_dict(sub("conditioning"))

    # ── inference helpers ────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> FeatureExtractorOutput:
        """(B, 3, H, W) normalised images → patch features (B, N, D_VIT) + positions."""
        return self.feature_extractor(images)

    def from_cache(self, features: torch.Tensor) -> FeatureExtractorOutput:
        """Wrap cached patch features (B, N, D_VIT), any dtype, into the extractor's output type."""
        if features.dtype != torch.float32:
            features = features.float()
        return FeatureExtractorOutput(features=features, positions=self.positions.to(features.device))

    @torch.no_grad()
    def sample_slots(self, batch_size: int) -> torch.Tensor:
        """Draw (B, n_slots, D_SLOT) initial slots from the learned conditioning prior."""
        return self.conditioning(batch_size)

    @torch.no_grad()
    def group(self, feat_out: FeatureExtractorOutput):
        """Run slot attention on patch features.

        Returns ``(slots (B, M, D), masks (B, M, N), embedded (B, N, D))`` where ``masks`` is the
        per-slot patch attention (feature attributions) and ``embedded`` are the positionally
        embedded features consumed by slot attention (needed to re-run it inside a parent region).
        """
        batch_size = feat_out.features.shape[0]
        init = self.conditioning(batch_size)
        embedded = self.grouping.positional_embedding(feat_out.features, feat_out.positions)
        slots, masks = self.grouping.slot_attention(embedded, init)
        return slots, masks, embedded


def load_frozen_dinosaur(checkpoint: str, n_slots: int, device=None) -> FrozenDinosaur:
    model = FrozenDinosaur(n_slots=n_slots, checkpoint=checkpoint)
    if device is not None:
        model = model.to(device)
    return model
