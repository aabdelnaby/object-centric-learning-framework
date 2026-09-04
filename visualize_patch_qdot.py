#!/usr/bin/env python3
"""Visualize the inference of the Patch-QDot baseline (``pooler='patch_qdot'``).

Patch-QDot is the weakest flat control for ``hier_router``: the compound "<y> <x>"
query (e.g. "car door", span ch3) is mapped once into patch space and an EXPLICIT
query·patch dot-product (no learned key/value, no multi-head, no tree) scores the
frozen DINOv3 patch tokens; the attended vector is read out with the same
query-conditioned colour head as the router. This script draws, per sample, the
patch-attention distribution the query produces — reshaped back to the 14×14 ViT
grid and overlaid on the input — so you can eyeball whether the single learned query
attends to the actual part region.

Per-sample figure (1×3 panels):
  A  input image (+ question, pred vs. true colour)
  B  query→patch attention α_i overlaid on the image (14×14 grid → upsampled)
  C  predicted answer distribution P(a) (bars painted their literal colour)

Example:
    conda run --no-capture-output -n oclf_env python visualize_patch_qdot.py \
        --checkpoint runs/ade20k_patch_qdot_raw/<jid>/patches/best_model.pt \
        --n_samples 6 --out_dir viz_patch_qdot
"""

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
from PIL import Image as PILImage

from classifier_model import PatchClassifier
from precompute_features import precompute_text, parse_xy_phrases, build_image_transform


# ── Colour-name → RGB so the answer bars are painted their literal colour ────────
COLOR_RGB = {
    "black":  (0.05, 0.05, 0.05), "white":  (0.97, 0.97, 0.97),
    "gray":   (0.50, 0.50, 0.50), "grey":   (0.50, 0.50, 0.50),
    "red":    (0.86, 0.15, 0.15), "green":  (0.13, 0.62, 0.20),
    "blue":   (0.12, 0.36, 0.86), "yellow": (0.95, 0.85, 0.10),
    "orange": (0.95, 0.55, 0.05), "purple": (0.52, 0.20, 0.70),
    "pink":   (0.96, 0.45, 0.71), "brown":  (0.55, 0.27, 0.07),
    "unknown": (0.80, 0.80, 0.80),
}
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]
DIM = 0.30  # min brightness for non-region pixels in the attention overlay


def color_for(name: str):
    return COLOR_RGB.get(str(name).lower(), (0.70, 0.70, 0.70))


# ── Model loading ───────────────────────────────────────────────────────────────
def build_model(cfg: dict, num_classes: int, device: torch.device):
    """Reconstruct the PatchClassifier exactly as train.py did (cached/no-text-encoder)."""
    model = PatchClassifier(
        dinosaur_cfg_name  = cfg["dinosaur_cfg_name"],
        dinosaur_ckpt_path = cfg["dinosaur_ckpt"],
        num_classes        = num_classes,
        d_vit              = cfg["d_vit"],
        d_slot             = cfg["d_slot"],
        d_text             = cfg["d_text"],
        num_heads          = cfg["num_heads"],
        roberta_model      = cfg["roberta_model"],
        text_encoder_type  = cfg["text_encoder"],
        t5_model           = cfg["t5_model"],
        vqa_d_model        = cfg["vqa_d_model"],
        load_text_encoder  = False,            # text is precomputed separately
        pooler             = cfg["pooler"],
        pooler_layers      = cfg["pooler_layers"],
        pooler_dropout     = cfg["pooler_dropout"],
        zero_image_feats   = cfg["zero_image_feats"],
        patch_qdot_project_patches = cfg.get("patch_qdot_project_patches", False),
        patch_qdot_strip_registers = cfg.get("patch_qdot_strip_registers", True),
        patch_qdot_temperature     = cfg.get("patch_qdot_temperature", 1.0),
        patch_qdot_normalize       = cfg.get("patch_qdot_normalize", False),
    ).to(device)
    model.eval()
    return model


def load_trainable(model, trainable: dict) -> bool:
    """Restore the trained heads; return True iff the patch_qdot weights were present."""
    if "text_projector" in trainable:
        model.text_projector.load_state_dict(trainable["text_projector"])
    has_head = "patch_qdot_head" in trainable and len(trainable["patch_qdot_head"]) > 0
    if has_head:
        model.patch_qdot_head.load_state_dict(trainable["patch_qdot_head"])
    return has_head


# ── Image / attention helpers ─────────────────────────────────────────────────────
def denorm(t: torch.Tensor) -> np.ndarray:
    mean_t = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (t.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()


def attn_to_grid(attn_1d: torch.Tensor) -> torch.Tensor:
    """(N,) patch attention → (side, side). If N isn't a perfect square but N-4 is,
    drop the 4 leading register tokens (keep_registers run); else pad to the next square."""
    n = attn_1d.shape[0]
    if math.isqrt(n) ** 2 != n and math.isqrt(n - 4) ** 2 == (n - 4):
        attn_1d = attn_1d[4:]
        n = attn_1d.shape[0]
    side = math.isqrt(n)
    if side * side < n:
        side += 1
        attn_1d = torch.cat([attn_1d, attn_1d.new_zeros(side * side - n)])
    return attn_1d.float().view(side, side)


def grid_to_alpha(grid: torch.Tensor, img_size: int) -> np.ndarray:
    """(side, side) → (img_size, img_size) alpha in [0,1] (max-normalised)."""
    up = F.interpolate(grid[None, None], size=(img_size, img_size),
                       mode="bilinear", align_corners=False)[0, 0]
    return (up / (up.max() + 1e-6)).clamp(0, 1).cpu().numpy()


def overlay(img_vis: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (img_vis * (DIM + (1.0 - DIM) * alpha[:, :, None])).clip(0, 1)


def draw_color_bar(ax, probs, classes, pred_idx, true_idx, title, topk=8):
    order = np.argsort(probs)[::-1][:topk]
    labels = [classes[i] for i in order]
    vals   = [probs[i] for i in order]
    cols   = [color_for(classes[i]) for i in order]
    y = np.arange(len(order))
    ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    for i, idx in enumerate(order):
        mark = ""
        if idx == pred_idx: mark += "◀pred"
        if idx == true_idx: mark += " ✓true" if mark else "✓true"
        if mark:
            ax.text(vals[i] + 0.01, i, mark, va="center", fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=9)


# ── Sampling (mirror visualize_hier_routing.sample_rows) ──────────────────────────
def sample_rows(df, n_samples, seed):
    df = df[df["query"].apply(lambda q: parse_xy_phrases(q) is not None)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No rows match the 'color of <x> of <y>' template in this split.")
    rng = np.random.RandomState(seed)
    queries = sorted(df["query"].unique().tolist())
    rows = []
    per_q = max(1, n_samples // max(len(queries), 1))
    for q in queries:
        qr = df[df["query"] == q]
        rows.append(qr.sample(n=min(per_q, len(qr)), random_state=rng))
    out = pd.concat(rows).sample(frac=1.0, random_state=rng).head(n_samples)
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best_model.pt / last.pt of a patch_qdot run")
    ap.add_argument("--csv_path", default=None, help="override CSV (default: checkpoint config)")
    ap.add_argument("--image_root", default=None, help="override image root (default: checkpoint config)")
    ap.add_argument("--split", default="val", help="dataset split to sample from")
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out_dir", default="viz_patch_qdot")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    label_vocab = ck["label_vocab"]
    classes = [None] * len(label_vocab)
    for name, idx in label_vocab.items():
        classes[idx] = name
    if cfg["pooler"] != "patch_qdot":
        raise SystemExit(f"checkpoint pooler is {cfg['pooler']!r}, not 'patch_qdot'.")

    csv_path   = args.csv_path   or cfg["csv_path"]
    image_root = args.image_root or cfg["image_root"]
    variant = "projected" if cfg.get("patch_qdot_project_patches") else "raw"

    print(f"Building PatchClassifier (patch_qdot/{variant}, text={cfg['text_encoder']}) on {device} …")
    model = build_model(cfg, len(label_vocab), device)
    trained = load_trainable(model, ck["trainable_state"])
    if not trained:
        print("\n" + "=" * 78)
        print("  ⚠  This checkpoint has NO patch_qdot_head weights (q_proj/f_readout/color_head).")
        print("     The head is RANDOMLY INITIALISED — figures are a pipeline demo only,")
        print("     NOT the trained model.")
        print("=" * 78 + "\n")
    else:
        print(f"Loaded trained head (epoch {ck.get('epoch','?')}, val_acc {ck.get('val_acc','?')}).")

    df = pd.read_csv(csv_path)
    df = df[df["split"] == args.split]
    sample_df = sample_rows(df, args.n_samples, args.seed)
    queries = sample_df["query"].unique().tolist()

    txt = precompute_text(queries, device, text_encoder=cfg["text_encoder"], with_spans=True)
    img_size = int(cfg.get("img_size", 224))
    tf = build_image_transform(img_size, cfg.get("resize_mode", "crop"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "trained" if trained else "UNTRAINED"
    stamp = "" if trained else "   ⚠ UNTRAINED HEAD (random init) — PIPELINE DEMO ONLY"

    for i, row in sample_df.iterrows():
        q = row["query"]
        pil = PILImage.open(os.path.join(image_root, row["image_name"])).convert("RGB")
        img_t = tf(pil).unsqueeze(0).to(device)
        img_vis = denorm(img_t[0])
        h_yx_src = txt["readout_vec"][q].unsqueeze(0).to(device)   # span ch3 "<y> <x>"
        with torch.no_grad():
            routing = {"input": {"image": img_t, "batch_size": img_t.shape[0]}}
            feat_out = model.dino_feature_extractor(inputs=routing)
            patches = feat_out.features.float()                    # (1, N, d_vit)
            h_yx = model.text_projector(h_yx_src.to(dtype=patches.dtype))
            logP_a, attn = model.patch_qdot_head(patches, h_yx)    # attn (1, Np)
        probs = logP_a.exp()[0].cpu().numpy()
        pred_idx = int(logP_a[0].argmax())
        true_idx = label_vocab.get(str(row["label"]), -1)

        grid  = attn_to_grid(attn[0].cpu())
        alpha = grid_to_alpha(grid, img_size)
        x_phrase, y_phrase = parse_xy_phrases(q)

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        fig.suptitle(
            f"Patch-QDot/{variant}  |  Q: color of the {x_phrase} of the {y_phrase}  |  "
            f"pred={classes[pred_idx]}  true={row['label']}{stamp}",
            fontsize=11,
        )
        axes[0].imshow(img_vis); axes[0].set_title("input", fontsize=9); axes[0].axis("off")
        axes[1].imshow(overlay(img_vis, alpha))
        axes[1].set_title("query→patch attention α", fontsize=9); axes[1].axis("off")
        draw_color_bar(axes[2], probs, classes, pred_idx, true_idx, "P(answer colour)")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = out_dir / f"sample_{i:02d}_{tag}.png"
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i}] {q[:55]:<55s} pred={classes[pred_idx]:<7s} true={row['label']:<7s} → {save_path}")

    print(f"\nSaved {len(sample_df)} figure(s) to {out_dir}/")


if __name__ == "__main__":
    main()
