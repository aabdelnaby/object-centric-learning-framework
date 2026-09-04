"""Compare DINOSAUR slot masks produced by multiple slot checkpoints on the
same images. Useful for inspecting how slot quality varies across training
checkpoints when training reconstruction loss isn't a reliable proxy.

Outputs a single PNG grid with rows = images and column groups = ckpts.
For each (image, ckpt) cell we show:
    - the input image (once per row)
    - the per-slot soft masks overlaid as a slot-coloured segmentation
    - the individual top-k slot masks

Usage:
    python visualize_slot_ckpts.py \\
        --dinosaur_cfg projects/bridging/dinosaur/superclevr3d_feat_rec_dino_large16_dinov3_448 \\
        --ckpt lightning_logs/sc3d_slots12_vitl_448_chain_4793103/checkpoints/epoch=6-step=17000.ckpt \\
        --ckpt lightning_logs/sc3d_slots12_vitl_448_chain_4793103/checkpoints/last.ckpt \\
        --csv FG-datset/superclevr3d/simple_color_vqa.csv \\
        --image_root FG-datset/superclevr3d/images \\
        --n_slots 12 --img_size 448 --n_images 8 \\
        --out_path slot_viz/vitl_448_ckpt_compare.png
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dinosaur_cfg", required=True,
                   help="Hydra experiment cfg name, e.g. projects/bridging/dinosaur/superclevr3d_feat_rec_dino_large16_dinov3_448")
    p.add_argument("--ckpt", action="append", required=True,
                   help="Repeat once per ckpt to compare (order = column order in the grid).")
    p.add_argument("--ckpt_label", action="append", default=None,
                   help="Optional human-readable label per ckpt (parallel to --ckpt).")
    p.add_argument("--csv", required=True,
                   help="CSV used only to pick a reproducible image sample (val split).")
    p.add_argument("--image_root", required=True)
    p.add_argument("--n_slots", type=int, default=12)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--top_k_slots", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_path", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
        transforms.CenterCrop(img_size),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def denorm(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (img_t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def slot_colors(n: int) -> np.ndarray:
    from matplotlib import cm
    cmap = cm.get_cmap("tab20" if n <= 20 else "turbo", n)
    return np.array([cmap(i)[:3] for i in range(n)], dtype=np.float32)


def load_dinosaur(cfg_name: str, ckpt_path: str, n_slots: int, device):
    from classifier_model import _load_dinosaur_submodules
    fe, cond, pg = _load_dinosaur_submodules(
        cfg_name, ckpt_path, REPO_ROOT, n_slots=n_slots,
    )
    return fe.to(device).eval(), cond.to(device).eval(), pg.to(device).eval()


@torch.no_grad()
def extract_masks(fe, cond, pg, images: torch.Tensor, img_size: int) -> np.ndarray:
    """Return (B, N_slots, H, W) numpy soft masks upsampled to img_size."""
    B = images.shape[0]
    routing = {"input": {"image": images, "batch_size": B}}
    routing["feature_extractor"] = fe(inputs=routing)
    routing["conditioning"]      = cond(inputs=routing)
    pg_out = pg(inputs=routing)
    attn = pg_out.feature_attributions      # (B, N_slots, N_patches)
    N_slots, N_patches = attn.shape[1], attn.shape[2]
    side = math.isqrt(N_patches)
    if side * side < N_patches:
        side += 1
    n_pad = side * side - N_patches
    if n_pad > 0:
        attn = torch.cat([attn, torch.zeros(*attn.shape[:2], n_pad,
                                            device=attn.device, dtype=attn.dtype)], dim=2)
    grids = attn.view(B, N_slots, side, side)
    upsampled = F.interpolate(grids, size=(img_size, img_size),
                              mode="bilinear", align_corners=False)
    return upsampled.cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpts  = list(args.ckpt)
    labels = list(args.ckpt_label) if args.ckpt_label else [Path(c).name for c in ckpts]
    if len(labels) != len(ckpts):
        raise ValueError(f"--ckpt_label count ({len(labels)}) must match --ckpt count ({len(ckpts)})")

    # ── Sample images (reproducible) ─────────────────────────────────────────
    df = pd.read_csv(args.csv)
    if "split" in df.columns:
        df = df[df["split"] == "val"]
    rng = np.random.RandomState(args.seed)
    unique_imgs = sorted(df["image_name"].unique().tolist())
    pick = rng.choice(unique_imgs, size=min(args.n_images, len(unique_imgs)), replace=False)
    print(f"Sampling {len(pick)} val images (seed={args.seed}).")

    tf = make_transform(args.img_size)
    images = torch.stack([
        tf(Image.open(os.path.join(args.image_root, name)).convert("RGB"))
        for name in pick
    ]).to(device)
    img_vis = [denorm(im) for im in images]

    colors = slot_colors(args.n_slots)
    top_k  = args.top_k_slots
    n_cols_per_ckpt = 1 + top_k         # overlay + top-k individual masks
    n_cols = 1 + len(ckpts) * n_cols_per_ckpt  # +1 for the input image column
    n_rows = len(pick)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.4, n_rows * 2.7),
        squeeze=False,
    )
    fig.suptitle("DINOSAUR slot masks per checkpoint", fontsize=14)

    # Column headers
    axes[0, 0].set_title("input", fontsize=10)
    for ci, lbl in enumerate(labels):
        base = 1 + ci * n_cols_per_ckpt
        axes[0, base].set_title(f"{lbl}\noverlay", fontsize=9)
        for k in range(top_k):
            axes[0, base + 1 + k].set_title(f"slot top-{k+1}", fontsize=9)

    # ── Per-ckpt forward → fill columns ──────────────────────────────────────
    for ci, (ckpt_path, label) in enumerate(zip(ckpts, labels)):
        print(f"[ckpt {ci+1}/{len(ckpts)}] loading {ckpt_path}")
        fe, cond, pg = load_dinosaur(args.dinosaur_cfg, ckpt_path, args.n_slots, device)
        masks = extract_masks(fe, cond, pg, images, args.img_size)   # (B, S, H, W)
        del fe, cond, pg
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # Rank slots per image by total mask mass — bigger = top
        slot_mass = masks.sum(axis=(-1, -2))                    # (B, S)
        rank_idx  = np.argsort(-slot_mass, axis=1)              # (B, S)

        base = 1 + ci * n_cols_per_ckpt
        for r in range(n_rows):
            # ── overlay: per-pixel argmax slot, slot-coloured ────────────────
            argmax = masks[r].argmax(axis=0)                    # (H, W)
            seg_rgb = colors[argmax]                            # (H, W, 3)
            # Blend with image for readability
            blended = 0.4 * img_vis[r] + 0.6 * seg_rgb
            ax = axes[r, base]
            ax.imshow(blended)
            ax.set_xticks([]); ax.set_yticks([])

            # ── individual top-k slot masks ──────────────────────────────────
            for k in range(top_k):
                slot_i = rank_idx[r, k]
                m = masks[r, slot_i]
                alpha = (m / max(m.max(), 1e-6)).clip(0, 1)[:, :, None]
                masked_img = img_vis[r] * alpha
                ax = axes[r, base + 1 + k]
                ax.imshow(masked_img)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(f"s{slot_i}", fontsize=7)

    # ── Input column ─────────────────────────────────────────────────────────
    for r in range(n_rows):
        axes[r, 0].imshow(img_vis[r])
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(pick[r].replace(".png", ""), fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=40)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
