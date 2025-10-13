"""DINOv3 segmentation feature extractor wrapper.

This wraps a DINOv3 segmentor (pretrained on ADE20K) and exposes its logits as
flat per-token features compatible with the rest of the framework.

Usage patterns supported:
- Load from a local clone via ``torch.hub.load`` with ``source='local'``.
- Or pass an already-instantiated ``nn.Module`` via the ``model`` argument.

Notes:
- The module expects inputs normalized by ImageNet mean/std upstream, which is
  already the default in dataset configs in this repo.
- The output features are segmentation logits of shape [B, N, C] where C is the
  number of classes (ADE20K: 150). Positions are computed on the logit grid.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ocl.feature_extractors.utils import (
    ImageFeatureExtractor,
    cnn_compute_positions_and_flatten,
)


class Dinov3SegmentorFeatureExtractor(ImageFeatureExtractor):
    """Feature extractor that exposes DINOv3 segmentor logits as features.

    Args:
        model: Optional pre-instantiated segmentor ``nn.Module``. If provided,
            ``repo_dir``/``hub_entry``/``weights`` are ignored.
        repo_dir: Path to a local clone of facebookresearch/dinov3.
        hub_entry: Name of the torch.hub entry for the segmentor
            (e.g. ``'dinov3_vitb14'`` or as documented in the repo).
        weights: Path to a checkpoint compatible with the hub entry.
        freeze: If True, do not compute gradients through the segmentor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        repo_dir: Optional[str] = "/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/dinov3",
        hub_entry: Optional[str] = None,
        weights: Optional[str] = "/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/ocl/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",
        backbone_weights: Optional[str] = "/home/kn/kn_kn/kn_pop550892/desktop/object-centric-learning-framework/ocl/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        freeze: bool = True,
        pool_grid_size: Optional[Union[int, Tuple[int, int]]] = 28,
        model_name: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        if model is None:
            if repo_dir is None or hub_entry is None or weights is None:
                raise ValueError(
                    "Provide either an instantiated model or (repo_dir, hub_entry, weights) for torch.hub.load(source='local')."
                )
            try:
                # Delay import to avoid hard dependency if unused
                import torch

                self.segmentor = torch.hub.load(
                    repo_dir, source="local", weights=weights, backbone_weights=backbone_weights
                )
            except Exception as e:  # pragma: no cover - depends on external repo
                raise RuntimeError(
                    "Failed to load DINOv3 segmentor via torch.hub.load. "
                    "Ensure the dinov3 repo is available locally and weights path is valid."
                ) from e
        else:
            self.segmentor = model

        self.freeze = freeze
        if freeze:
            for p in self.segmentor.parameters():
                p.requires_grad_(False)
        self.segmentor.eval() if freeze else self.segmentor.train(False)

        self._feature_dim: Optional[int] = None
        if isinstance(pool_grid_size, int) and pool_grid_size is not None:
            pool_grid_size = (pool_grid_size, pool_grid_size)
        self.pool_grid_size: Optional[Tuple[int, int]] = pool_grid_size  # e.g., (28,28) -> 784 tokens

    @property
    def feature_dim(self) -> int:
        # Best effort: fallback to 150 classes if not yet inferred
        return self._feature_dim or 150

    def _segmentor_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the underlying segmentor and return logits [B,C,H,W]."""
        with torch.set_grad_enabled(not self.freeze):
            out = self.segmentor(images)

        # Accept common output conventions
        if isinstance(out, torch.Tensor):
            logits = out
        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            logits = out[0]
        elif isinstance(out, dict):
            for key in ["logits", "pred", "out", "seg", "segmentation"]:
                if key in out and isinstance(out[key], torch.Tensor):
                    logits = out[key]
                    break
            else:
                raise RuntimeError("Unsupported dict output from segmentor; missing logits-like key.")
        else:
            raise RuntimeError("Unsupported output type from segmentor.")

        if logits.ndim != 4:
            raise RuntimeError(f"Expected 4D logits [B,C,H,W], got shape {list(logits.shape)}")
        return logits

    def forward_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # images: [B,3,H,W], normalized upstream
        logits = self._segmentor_forward(images)

        # Optionally downsample logits to a fixed token grid
        if self.pool_grid_size is not None:
            logits = F.adaptive_avg_pool2d(logits, self.pool_grid_size)

        # Track feature dim lazily
        self._feature_dim = int(logits.shape[1])

        # Flatten to [B, N, C] and compute positions on the logit grid
        flat, pos = cnn_compute_positions_and_flatten(logits)
        # Detach if the backbone is frozen
        if self.freeze:
            flat = flat.detach()
        return flat, pos
