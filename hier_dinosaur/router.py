"""The HierRouter: top-down object→part routing with log-space path marginalisation.

    P(j | y)      parent routing over object slots (empty slots masked out)
    P(k | j, x)   local child routing within each parent's K part sub-slots
    P(a | c_jk)   shared query-conditioned attribute head on the routed child slot
    P(a)          = sum_jk P(j|y) P(k|j,x) P(a|c_jk)      (marginal, used for training)

The module returns ``log P(a)`` and is trained with ``nn.NLLLoss``. Setting ``use_children=False``
gives the parent-only ablation ``P(a) = sum_j P(j|q) P(a|s_j)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class TextProjector(nn.Module):
    """Shared two-layer projector from T5 span vectors into slot space: L2-norm → Linear → GELU → Linear → LayerNorm."""

    def __init__(self, d_text: int, d_slot: int):
        super().__init__()
        self.fc1 = nn.Linear(d_text, d_slot)
        self.fc2 = nn.Linear(d_slot, d_slot)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_slot)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        x = F.normalize(text_features, dim=-1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.ln(x)


@dataclass
class RouterOutput:
    logp_answer: torch.Tensor           # (B, C)     log P(a)  (marginal over paths)
    logp_parent: torch.Tensor           # (B, P)     log P(j | y)
    logp_child: Optional[torch.Tensor]  # (B, P, K)  log P(k | j, x); None for parent-only
    logp_color: torch.Tensor            # (B, P, K, C) or (B, P, C): per-node attribute log-probs

    @property
    def logw(self) -> torch.Tensor:
        """Log path weights log w_jk = log P(j|y) + log P(k|j,x)   (B, P, K)."""
        if self.logp_child is None:
            return self.logp_parent[:, :, None]
        return self.logp_parent[:, :, None] + self.logp_child

    def map_path_logp(self):
        """MAP-path readout: pick (j*, k*) = argmax w_jk and return log P(a | c_{j*k*}) plus (j*, k*)."""
        logw = self.logw
        batch, n_parents, n_children = logw.shape
        best = logw.reshape(batch, -1).argmax(dim=1)
        j_star, k_star = best // n_children, best % n_children
        logp_color = self.logp_color
        if logp_color.dim() == 3:                     # parent-only: (B, P, C)
            logp_color = logp_color[:, :, None, :]
        idx = torch.arange(batch, device=logw.device)
        return logp_color[idx, j_star, k_star], j_star, k_star


class HierRouter(nn.Module):
    """Structured path-marginalisation head over a depth-2 slot tree.

    Args:
        d_slot:        slot dimensionality (queries are projected into slot space upstream).
        num_classes:   size of the attribute vocabulary C.
        temperature:   routing temperature tau applied to both softmaxes.
        child_scorer:  ``"mlp"`` scores [q_ch(h_yx); c_jk; s_j] with a 2-layer MLP (thesis),
                       ``"bilinear"`` uses a scaled dot product q_ch(h_yx)·c_jk.
        use_children:  ``False`` → parent-only ablation (no child level).
        readout_query: concatenate a learned readout vector f_ro(h_yx) to the routed slot before
                       the attribute head (thesis: on).
    """

    def __init__(
        self,
        d_slot: int,
        num_classes: int,
        temperature: float = 1.0,
        child_scorer: str = "bilinear",
        use_children: bool = True,
        readout_query: bool = True,
    ):
        super().__init__()
        if child_scorer not in ("bilinear", "mlp"):
            raise ValueError(f"unknown child_scorer={child_scorer!r}; expected bilinear|mlp")
        self.tau = float(temperature)
        self.child_scorer = child_scorer
        self.use_children = bool(use_children)
        self.readout_query = bool(readout_query)
        self.scale = d_slot ** -0.5

        self.q_parent = nn.Linear(d_slot, d_slot)
        if self.use_children:
            self.q_child = nn.Linear(d_slot, d_slot)
            if child_scorer == "mlp":
                self.child_mlp = nn.Sequential(
                    nn.Linear(3 * d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, 1),
                )
        if self.readout_query:
            self.f_readout = nn.Linear(d_slot, d_slot)
        color_in = d_slot + (d_slot if self.readout_query else 0)
        self.color_head = nn.Sequential(
            nn.Linear(color_in, d_slot), nn.GELU(), nn.Linear(d_slot, num_classes),
        )

    def color_logp(self, slots: torch.Tensor, h_readout: Optional[torch.Tensor]) -> torch.Tensor:
        """log P(a | slot[, f_ro(h_yx)]) for slots of shape (..., D) → (..., C)."""
        if self.readout_query:
            g = self.f_readout(h_readout)
            while g.dim() < slots.dim():
                g = g.unsqueeze(1)
            g = g.expand(*slots.shape[:-1], g.shape[-1])
            feat = torch.cat([slots, g], dim=-1)
        else:
            feat = slots
        return F.log_softmax(self.color_head(feat), dim=-1)

    def forward(
        self,
        parent_slots: torch.Tensor,                 # (B, P, D)
        child_slots: Optional[torch.Tensor],        # (B, P, K, D); None for parent-only
        h_x: Optional[torch.Tensor],                # (B, D) child-routing query (compound "<object> <part>")
        h_y: torch.Tensor,                          # (B, D) parent-routing query ("<object>", or the full phrase for parent-only)
        nonempty: Optional[torch.Tensor] = None,    # (B, P) bool; empty parents get P(j|y) = 0
        h_readout: Optional[torch.Tensor] = None,   # (B, D) readout query (same vector as h_x in the thesis)
    ) -> RouterOutput:
        # parent routing P(j | y)
        qy = self.q_parent(h_y)
        r_y = torch.einsum("bd,bpd->bp", qy, parent_slots) * self.scale
        if nonempty is not None:
            nonempty = nonempty | (~nonempty.any(dim=1, keepdim=True))   # degenerate rows keep all parents
            r_y = r_y.masked_fill(~nonempty, float("-inf"))
        logp_parent = F.log_softmax(r_y / self.tau, dim=1)

        if not self.use_children:
            logp_color = self.color_logp(parent_slots, h_readout)               # (B, P, C)
            logp_answer = torch.logsumexp(logp_parent[..., None] + logp_color, dim=1)
            return RouterOutput(logp_answer, logp_parent, None, logp_color)

        # child routing P(k | j, x), local softmax within each parent
        batch, n_parents, n_children, dim = child_slots.shape
        qx = self.q_child(h_x)
        if self.child_scorer == "mlp":
            hx_e = qx[:, None, None, :].expand(batch, n_parents, n_children, dim)
            par_e = parent_slots[:, :, None, :].expand(batch, n_parents, n_children, dim)
            r_x = self.child_mlp(torch.cat([hx_e, child_slots, par_e], dim=-1)).squeeze(-1)
        else:
            r_x = torch.einsum("bd,bpkd->bpk", qx, child_slots) * self.scale
        logp_child = F.log_softmax(r_x / self.tau, dim=2)

        # per-child attribute distribution and log-space marginalisation over paths
        logw = logp_parent[:, :, None] + logp_child
        logp_color = self.color_logp(child_slots, h_readout)                    # (B, P, K, C)
        logp_answer = torch.logsumexp(logw[..., None] + logp_color, dim=(1, 2))
        return RouterOutput(logp_answer, logp_parent, logp_child, logp_color)
