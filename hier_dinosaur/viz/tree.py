"""Hierarchical-DINOSAUR tree figure: input → object slots → part sub-slots (thesis Figure 3.1b).

Runs the frozen DINOv3 DINOSAUR on one image, refines every non-empty object slot into K part
sub-slots with the recursive inference of ``hierarchy.build_tree`` and draws the resulting tree.
Every node shows the original pixels gated by the node's attribution mask; no training and no
object or part supervision are involved.

    python -m hier_dinosaur.viz.tree --image data/coco/train2017/000000000139.jpg --out figures/hier_dinosaur_tree.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage

from ..dinosaur import FrozenDinosaur
from ..features import build_image_transform
from ..hierarchy import build_tree
from ..models import DEFAULT_DINOSAUR_CKPT, resolve_dinosaur_ckpt
from .routing import denorm, mask_to_alpha

BACKGROUND = np.array([1.0, 1.0, 1.0])


def gated(img_vis: np.ndarray, alpha: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    a = np.clip(alpha, 0, 1)[..., None] ** gamma
    return (img_vis * a + BACKGROUND * (1.0 - a)).clip(0, 1)


@torch.no_grad()
def compute_tree(image_path, dinosaur_ckpt, n_slots, n_children, seed, device):
    torch.manual_seed(seed)
    dinosaur = FrozenDinosaur(n_slots=n_slots, checkpoint=resolve_dinosaur_ckpt(dinosaur_ckpt)).to(device)
    transform = build_image_transform(224, "square")
    img_t = transform(PILImage.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    tree = build_tree(dinosaur, dinosaur.encode_images(img_t), n_parents=None, n_children=n_children)
    return denorm(img_t[0]), tree


def draw_tree(img_vis, tree, out_path, img_size=224, min_mass=0.02, node_in=1.6, title=None):
    parent_masks = tree.parent_masks[0].cpu()            # (P, N)
    child_attn = tree.child_attn[0].cpu()                # (P, K, N)
    nonempty = tree.nonempty[0].cpu().numpy()
    keep = [j for j in range(parent_masks.shape[0]) if nonempty[j]]
    n_children = child_attn.shape[1]
    n_leaves = len(keep) * n_children
    width = max(n_leaves, len(keep), 1) * node_in
    fig = plt.figure(figsize=(width + 0.5, 3 * node_in + 1.0))
    # node positions in figure fractions: level 0 (input), level 1 (objects), level 2 (parts)
    ys = [0.83, 0.50, 0.17]
    slot_w, slot_h = node_in / (width + 0.5) * 0.9, node_in / (3 * node_in + 1.0) * 0.85

    def add_node(xc, yc, image, label):
        ax = fig.add_axes([xc - slot_w / 2, yc - slot_h / 2, slot_w, slot_h])
        ax.imshow(image); ax.axis("off")
        ax.set_title(label, fontsize=8, pad=2)
        return ax

    leaf_x = [(i + 0.5) / max(n_leaves, 1) for i in range(n_leaves)]
    obj_x = [np.mean(leaf_x[j * n_children:(j + 1) * n_children]) for j in range(len(keep))]
    root_x = float(np.mean(obj_x)) if obj_x else 0.5
    add_node(root_x, ys[0], img_vis, "Input")
    line = fig.add_axes([0, 0, 1, 1]); line.axis("off"); line.set_xlim(0, 1); line.set_ylim(0, 1)
    for jj, j in enumerate(keep):
        pm = mask_to_alpha(parent_masks[j], img_size)
        add_node(obj_x[jj], ys[1], gated(img_vis, pm), f"Object {j}")
        line.plot([root_x, obj_x[jj]], [ys[0] - slot_h / 2, ys[1] + slot_h / 2 + 0.03], color="0.4", lw=0.8)
        for k in range(n_children):
            cm = mask_to_alpha(child_attn[j, k], img_size)
            x = leaf_x[jj * n_children + k]
            add_node(x, ys[2], gated(img_vis, cm), f"Part {j}.{k}")
            line.plot([obj_x[jj], x], [ys[1] - slot_h / 2, ys[2] + slot_h / 2 + 0.03], color="0.4", lw=0.8)
    if title:
        fig.suptitle(title, fontsize=10)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="figures/hier_dinosaur_tree.png")
    ap.add_argument("--n_slots", type=int, default=7, help="object slots M")
    ap.add_argument("--children", type=int, default=3, help="part sub-slots K per object slot")
    ap.add_argument("--dinosaur_ckpt", default=None, help=f"default: {DEFAULT_DINOSAUR_CKPT}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    img_vis, tree = compute_tree(args.image, args.dinosaur_ckpt, args.n_slots, args.children, args.seed,
                                 torch.device(args.device))
    draw_tree(img_vis, tree, args.out, title=args.title)
    print(f"→ {args.out}  (objects kept: {int(tree.nonempty.sum())}/{tree.parent_masks.shape[1]}, K={args.children})")


if __name__ == "__main__":
    main()
