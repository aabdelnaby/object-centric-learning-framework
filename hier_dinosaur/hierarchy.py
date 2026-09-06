"""Hierarchical-DINOSAUR: recursive inference over a frozen slot-attention module.

Given the object-level slots of an image, every slot is refined into K part-level sub-slots
by re-running the *same frozen* slot attention confined to the patches the parent slot
explains (Algorithm 1 of the thesis). Nothing here is trained.

Two details are essential (Section "Design considerations" in the thesis):
* confinement acts on the attention weights, not on the input features (slot attention
  applies a scale-invariant LayerNorm to the features, which would cancel a feature gate);
* child seeds are sampled from the learned conditioning prior (seeding from the parent slot
  plus noise makes the children collapse onto the same region).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .dinosaur import FrozenDinosaur
from ocl.typing import FeatureExtractorOutput


@dataclass
class SlotTree:
    """A depth-2 object→part decomposition of a batch of images (all tensors on one device).

    Attributes:
        slots:        (B, M, D)     all object-level slots
        slot_masks:   (B, M, N)     their patch attention (mass = how much of the image they explain)
        embedded:     (B, N, D)     positionally embedded patch features
        parent_slots: (B, P, D)     the P selected object slots (P = M in the thesis)
        parent_masks: (B, P, N)     their patch masks  (alpha_j)
        nonempty:     (B, P) bool   parents carrying > 2 % of the attention mass
        top_idx:      (B, P)        index of each parent in ``slots``
        child_slots:  (B, P, K, D)  part sub-slots  (c_jk);  None for the parent-only model
        child_attn:   (B, P, K, N)  confined part masks (alpha_hat_jk); None for parent-only
    """

    slots: torch.Tensor
    slot_masks: torch.Tensor
    embedded: torch.Tensor
    parent_slots: torch.Tensor
    parent_masks: torch.Tensor
    nonempty: torch.Tensor
    top_idx: torch.Tensor
    child_slots: Optional[torch.Tensor] = None
    child_attn: Optional[torch.Tensor] = None

    @property
    def n_children(self) -> int:
        return 0 if self.child_slots is None else self.child_slots.shape[2]


@torch.no_grad()
def select_parents(slots: torch.Tensor, slot_masks: torch.Tensor, n_parents: int):
    """Pick the ``n_parents`` slots with the largest mask mass, non-empty ones first.

    A slot is *non-empty* when its mask carries more than 2 % of the per-image attention
    mass. Returns ``(parent_slots (B,P,D), nonempty (B,P) bool, top_idx (B,P))``.
    """
    batch, n_slots, dim = slots.shape
    n_parents = min(n_parents, n_slots)
    mass = slot_masks.sum(dim=-1)                                   # (B, M)
    nonempty = mass > 0.02 * mass.sum(dim=1, keepdim=True)
    scores = mass.masked_fill(~nonempty, float("-inf"))
    top_idx = scores.topk(n_parents, dim=1).indices                 # (B, P)
    parent_slots = torch.gather(slots, 1, top_idx[:, :, None].expand(-1, -1, dim))
    return parent_slots, torch.gather(nonempty, 1, top_idx), top_idx


@torch.no_grad()
def spatial_child_prior(parent_mask: torch.Tensor, positions: torch.Tensor, n_children: int):
    """Per-child Gaussian bumps around K farthest-point anchors inside the parent → (B, K, N).

    Only used when ``spread > 0``; the thesis experiments use ``spread = 0`` (no prior).
    """
    n_patches = parent_mask.shape[1]
    pos = positions.to(parent_mask.device).float()                  # (N, 2)
    w = parent_mask.clamp(min=0.0)                                  # (B, N)
    valid = w > (0.5 * w.mean(dim=1, keepdim=True))

    anchors = [w.argmax(dim=1)]
    d2 = ((pos[None] - pos[anchors[0]][:, None]) ** 2).sum(-1)      # (B, N)
    for _ in range(1, n_children):
        nxt = d2.masked_fill(~valid, -1.0).argmax(dim=1)
        anchors.append(nxt)
        d2 = torch.minimum(d2, ((pos[None] - pos[nxt][:, None]) ** 2).sum(-1))
    anchor_pos = pos[torch.stack(anchors, dim=1)]                   # (B, K, 2)
    dist2 = ((pos[None, None] - anchor_pos[:, :, None]) ** 2).sum(-1)   # (B, K, N)

    wn = w / (w.sum(dim=1, keepdim=True) + 1e-6)
    centroid = (wn[:, :, None] * pos[None]).sum(dim=1)              # (B, 2)
    extent = (wn[:, :, None] * (pos[None] - centroid[:, None]) ** 2).sum(dim=1).sum(-1).sqrt()
    sigma = (0.7 * extent).clamp(min=0.05)[:, None, None]
    return torch.exp(-dist2 / (2.0 * sigma ** 2))


@torch.no_grad()
def sample_child_seeds(dinosaur: FrozenDinosaur, batch_size: int, n_children: int, like: torch.Tensor):
    """K initial child slots drawn from the frozen conditioning prior.

    The prior is sampled for the full slot count and sliced to K (tiled if K exceeds it), which
    keeps the symmetry-breaking scale the slot module was trained with.
    """
    seeds = dinosaur.sample_slots(batch_size)                       # (B, M, D)
    if seeds.shape[1] < n_children:
        reps = (n_children + seeds.shape[1] - 1) // seeds.shape[1]
        seeds = seeds.repeat(1, reps, 1)
    return seeds[:, :n_children].contiguous().to(device=like.device, dtype=like.dtype)


@torch.no_grad()
def confined_slot_attention(slot_attention, embedded, parent_mask, seeds, spatial_prior=None, spread=0.0):
    """Frozen slot attention confined to one parent region → K child slots.

    Reuses the frozen module's weights (norms, to_q/k/v, GRU, MLP). After the per-patch
    softmax over the K children, every patch is re-weighted by ``parent_mask`` (and optionally
    ``spatial_prior ** spread``) and the assignment is renormalised over patches before the
    slot update, so each child only aggregates evidence inside the parent region.

    Returns ``(child_slots (B, K, D), child_attn (B, K, N))``; ``child_attn`` is the
    head-averaged confined assignment of the last iteration, before the eps-renormalisation.
    """
    sa = slot_attention
    batch, n_patches, _ = embedded.shape
    n_children = seeds.shape[1]
    heads, dph = sa.n_heads, sa.dims_per_head

    feats = sa.norm_input(embedded)
    k = sa.to_k(feats).view(batch, n_patches, heads, dph)
    v = sa.to_v(feats).view(batch, n_patches, heads, dph)
    pm = parent_mask.clamp(min=0.0)[:, None, None, :]               # (B, 1, 1, N)
    prior = None
    if spatial_prior is not None and spread > 0.0:
        prior = spatial_prior[:, :, None, :].clamp(min=1e-6) ** spread

    slots = seeds
    last_assignment = None
    for _ in range(sa.iters):
        slots_prev = slots
        q = sa.to_q(sa.norm_slots(slots)).view(batch, n_children, heads, dph)
        dots = torch.einsum("bihd,bjhd->bihj", q, k) * sa.scale        # (B, K, H, N)
        attn = dots.flatten(1, 2).softmax(dim=1).view(batch, n_children, heads, n_patches)
        a = attn * pm                                                  # confine to the parent
        if prior is not None:
            a = a * prior
        last_assignment = a
        a = a + sa.eps
        a = a / a.sum(dim=-1, keepdim=True)                            # renormalise over patches
        updates = torch.einsum("bjhd,bihj->bihd", v, a)
        slots = sa.gru(updates.reshape(-1, sa.kvq_dim), slots_prev.reshape(-1, sa.dim))
        slots = slots.reshape(batch, n_children, sa.dim)
        if sa.ff_mlp is not None:
            slots = sa.ff_mlp(slots)

    return slots, last_assignment.mean(dim=2)


@torch.no_grad()
def build_tree(
    dinosaur: FrozenDinosaur,
    feat_out: FeatureExtractorOutput,
    n_parents: Optional[int] = None,
    n_children: int = 5,
    spread: float = 0.0,
    with_children: bool = True,
) -> SlotTree:
    """Object slots → (optionally) part sub-slots for a batch of patch features.

    Args:
        n_parents:     how many object slots to keep as parents (default: all of them).
        n_children:    K part sub-slots per parent.
        spread:        strength of the spatial prior (0 = off, as in the thesis).
        with_children: ``False`` builds the parent-only tree used by the ablation.
    """
    slots, slot_masks, embedded = dinosaur.group(feat_out)
    batch = slots.shape[0]
    n_parents = slots.shape[1] if n_parents is None else n_parents
    parent_slots, nonempty, top_idx = select_parents(slots, slot_masks, n_parents)
    parent_masks = torch.gather(slot_masks, 1, top_idx[:, :, None].expand(-1, -1, slot_masks.shape[-1]))
    tree = SlotTree(slots, slot_masks, embedded, parent_slots, parent_masks, nonempty, top_idx)
    if not with_children:
        return tree

    child_slots, child_attn = [], []
    for p in range(parent_slots.shape[1]):
        pmask = parent_masks[:, p]                                     # (B, N)
        seeds = sample_child_seeds(dinosaur, batch, n_children, embedded)
        prior = spatial_child_prior(pmask, dinosaur.positions, n_children) if spread > 0 else None
        cs, ca = confined_slot_attention(dinosaur.slot_attention, embedded, pmask, seeds, prior, spread)
        child_slots.append(cs)
        child_attn.append(ca)
    tree.child_slots = torch.stack(child_slots, dim=1)                 # (B, P, K, D)
    tree.child_attn = torch.stack(child_attn, dim=1)                   # (B, P, K, N)
    return tree
