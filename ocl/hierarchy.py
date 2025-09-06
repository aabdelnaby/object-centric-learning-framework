from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocl.perceptual_grouping import SlotAttention


class HierarchicalSlots(nn.Module):
    """Coarse-to-fine refinement producing child slots and gating masks.

    Inputs (via routed paths):
      - tokens: Float[B, N, D_feat]  Teacher tokens (e.g., ViT features)
      - parent_slots: Float[B, K1, D_slots]  Parent/object slots
      - parent_masks: Float[B, K1, N]       Parent masks on the patch grid

    Outputs (dict):
      - child_objects: Float[B, K1*K2, D_slots]  Refined child slots
      - child_gating_masks: Float[B, K1*K2, N]   Gating masks per child (copied parent masks)

    Notes:
      - Implementation is inference-only by default; parameters are created but training is not required.
      - Uses a tiny Slot Attention loop over K2 children per parent, gated by the parent mask.
    """

    def __init__(
        self,
        child_per_parent: int = 2,
        slot_dim: int = 128,
        sa_iters: int = 3,
        min_region_fraction: float = 0.0,
        n_heads: int = 1,
        kvq_dim: Optional[int] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        res_mlp: bool = True,
    ):
        super().__init__()
        self.K2 = int(child_per_parent)
        self.slot_dim = int(slot_dim)
        self.sa_iters = int(sa_iters)
        self.min_region_fraction = float(min_region_fraction)
        self.n_heads = int(n_heads)
        self.kvq_dim = kvq_dim
        self.use_projection_bias = bool(use_projection_bias)
        self.use_implicit_differentiation = bool(use_implicit_differentiation)

        # Slot-wise MLP as in SlotAttentionGrouping
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, 4 * self.slot_dim),
            nn.GELU(),
            nn.Linear(4 * self.slot_dim, self.slot_dim),
        )
        if res_mlp:
            self.mlp = nn.Sequential(self.mlp)

        # Lazily built SlotAttention instance with correct feature_dim
        self._sa: Optional[SlotAttention] = None

    def _build_sa_if_needed(self, feature_dim: int, device: torch.device, dtype: torch.dtype):
        if self._sa is None:
            self._sa = SlotAttention(
                dim=self.slot_dim,
                feature_dim=feature_dim,
                kvq_dim=self.kvq_dim,
                n_heads=self.n_heads,
                iters=self.sa_iters,
                ff_mlp=self.mlp,
                use_projection_bias=self.use_projection_bias,
                use_implicit_differentiation=self.use_implicit_differentiation,
            ).to(device=device, dtype=dtype)
        else:
            # Ensure correct device/dtype if module was created after parent .to()
            self._sa.to(device=device, dtype=dtype)

    def forward(
        self,
        tokens: torch.Tensor,  # [B, N, D_feat]
        parent_slots: torch.Tensor,  # [B, K1, Ds]
        parent_masks: torch.Tensor,  # [B, K1, N]
    ) -> Dict[str, torch.Tensor]:
        B, N, Df = tokens.shape
        _, K1, Ds = parent_slots.shape
        if Ds != self.slot_dim:
            raise AssertionError("slot_dim mismatch between parent slots and module configuration")

        # Ensure SlotAttention is initialized and placed on correct device/dtype
        self._build_sa_if_needed(Df, device=tokens.device, dtype=tokens.dtype)

        child_slots_out = []
        child_masks_out = []

        # Precompute threshold for tiny/empty parents
        min_tokens = int(self.min_region_fraction * N)

        for k in range(K1):
            w_parent = parent_masks[:, k, :]  # [B, N]

            # Determine active tokens (before normalization) for small-region guard
            token_counts = (w_parent > 0.0).sum(dim=-1)  # [B]
            # Seed children near the parent slot
            mu = parent_slots[:, k, :]  # [B, Ds]
            eps = 0.01 * torch.randn(B, self.K2, Ds, device=tokens.device, dtype=tokens.dtype)
            slots = mu[:, None, :].expand(-1, self.K2, -1) + eps  # [B, K2, Ds]

            # Skip refinement for very small/empty parents (keep seeds)
            if min_tokens > 0:
                # If any batch element is too small, we still run SA for other elements; this is a
                # simple heuristic: run SA only if all batches have reasonable support
                all_large = torch.all(token_counts >= min_tokens).item()
            else:
                all_large = True

            if all_large:
                # Gate tokens by parent mask: scale features token-wise so that K/V outside region vanish
                denom = w_parent.mean(dim=-1, keepdim=True)
                safe = (denom > 1e-6).float()
                w_scaled = safe * (w_parent / (denom + 1e-6)) + (1.0 - safe) * 1.0
                tokens_gated = tokens * w_scaled.unsqueeze(-1)
                slots, _ = self._sa(tokens_gated, slots)

            child_slots_out.append(slots)  # [B, K2, Ds]
            # Duplicate gating mask per child
            child_masks_out.append(w_parent[:, None, :].expand(-1, self.K2, -1))  # [B, K2, N]

        child_objects = torch.cat(child_slots_out, dim=1) if K1 > 0 else torch.empty(
            B, 0, Ds, device=tokens.device, dtype=tokens.dtype
        )
        child_gating = torch.cat(child_masks_out, dim=1) if K1 > 0 else torch.empty(
            B, 0, N, device=tokens.device, dtype=tokens.dtype
        )

        return {"child_objects": child_objects, "child_gating_masks": child_gating}
