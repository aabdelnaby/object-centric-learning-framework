"""Routing-trace panels for a HierRouter checkpoint.

Per sample, a 2×5 figure of the traversal the router performs for
"What is the colour of the <part> of the <object>?":

    A input      B routed object slot j*     C routed part sub-slot k*   D P(j|y)     E P(k|j*,x)
    F w_jk       G responsibility w_jk·P(a*|c_jk)   H best-path P(a|c_j*k*)   I marginal P(a)   J summary

    python -m hier_dinosaur.viz.routing --checkpoint runs/paco/hier_router/best_model.pt --n_samples 6 --out_dir figures/routing
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
from PIL import Image as PILImage

from ..data import PRESETS, load_frame
from ..features import IMAGE_MEAN, IMAGE_STD, build_image_transform, resolve_image_path
from ..models import DEFAULT_DINOSAUR_CKPT, classes_from_vocab, is_router, load_checkpoint
from ..text import encode_spans, parse_xy

COLOR_RGB = {
    "black": (0.05, 0.05, 0.05), "white": (0.97, 0.97, 0.97), "gray": (0.50, 0.50, 0.50),
    "grey": (0.50, 0.50, 0.50), "red": (0.86, 0.15, 0.15), "green": (0.13, 0.62, 0.20),
    "blue": (0.12, 0.36, 0.86), "yellow": (0.95, 0.85, 0.10), "orange": (0.95, 0.55, 0.05),
    "purple": (0.52, 0.20, 0.70), "pink": (0.96, 0.45, 0.71), "brown": (0.55, 0.27, 0.07),
    "buff": (0.94, 0.86, 0.60), "iridescent": (0.35, 0.70, 0.75), "olive": (0.50, 0.50, 0.15),
    "rufous": (0.66, 0.27, 0.12), "unknown": (0.80, 0.80, 0.80),
}
DIM = 0.30


def color_for(name: str):
    return COLOR_RGB.get(str(name).lower(), (0.70, 0.70, 0.70))


def denorm(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def mask_to_alpha(mask_1d: torch.Tensor, img_size: int) -> np.ndarray:
    """Patch mask (N,) → (img_size, img_size) alpha in [0, 1], max-normalised."""
    n = mask_1d.shape[0]
    side = math.isqrt(n)
    if side * side < n:
        side += 1
    m = mask_1d.float()
    if side * side > n:
        m = torch.cat([m, m.new_zeros(side * side - n)])
    up = F.interpolate(m.view(1, 1, side, side), size=(img_size, img_size), mode="bilinear", align_corners=False)[0, 0]
    return (up / (up.max() + 1e-6)).clamp(0, 1).cpu().numpy()


def overlay(img_vis: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (img_vis * (DIM + (1.0 - DIM) * alpha[:, :, None])).clip(0, 1)


def draw_dist_bar(ax, probs, title, highlight=None, nonempty=None):
    x = np.arange(len(probs))
    cols = []
    for i in x:
        if nonempty is not None and not bool(nonempty[i]):
            cols.append("#cccccc")
        elif highlight is not None and i == highlight:
            cols.append("#dd8452")
        else:
            cols.append("#4c72b0")
    ax.bar(x, probs, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.tick_params(labelsize=6)


def draw_color_bar(ax, probs, classes, pred_idx, title, topk=8):
    order = list(np.argsort(probs)[::-1][:topk])
    y = np.arange(len(order))
    vals = [float(probs[i]) for i in order]
    bars = ax.barh(y, vals, color=[color_for(classes[i]) for i in order], edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([classes[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_title(title, fontsize=9)
    for i, idx in enumerate(order):
        if idx == pred_idx:
            bars[i].set_edgecolor("crimson")
            bars[i].set_linewidth(2.2)
        ax.text(min(vals[i] + 0.02, 0.86), y[i], f"{vals[i]:.2f}", va="center", fontsize=7)


def draw_heat(ax, mat, title, star=None):
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("child k", fontsize=8)
    ax.set_ylabel("parent j", fontsize=8)
    ax.tick_params(labelsize=6)
    if star is not None:
        ax.plot(star[1], star[0], marker="*", color="red", markersize=13, markeredgecolor="white", markeredgewidth=0.6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_sample(out, img_vis, img_size, classes, true_label, x_phrase, y_phrase, save_path):
    P_parent = out["P_parent"][0].cpu().numpy()
    P_child = out["P_child"][0].cpu().numpy()
    w = out["w"][0].cpu().numpy()
    p_color = out["p_color"][0].cpu().numpy()
    marginal = out["logits"][0].exp().cpu().numpy()
    parent_masks, child_attn = out["parent_masks"][0], out["child_attn"][0]
    top_idx = out["top_idx"][0].cpu().numpy()
    nonempty = out["nonempty"][0].cpu().numpy() if out["nonempty"] is not None else None

    jstar = int(P_parent.argmax())
    kstar = int(P_child[jstar].argmax())
    pred_idx = int(marginal.argmax())
    pred_name, true_name = classes[pred_idx], str(true_label)
    resp = w * p_color[:, :, pred_idx]

    fig, axd = plt.subplot_mosaic([["A", "B", "C", "D", "E"], ["F", "G", "H", "I", "J"]], figsize=(21, 8.4))
    ok = "✓" if pred_name == true_name else "✗"
    fig.suptitle(f'Q: "color of the [{x_phrase}] of the [{y_phrase}]?"     '
                 f'pred: {pred_name} ({marginal[pred_idx]:.2f})   true: {true_name} {ok}', fontsize=12)
    axd["A"].imshow(img_vis); axd["A"].axis("off"); axd["A"].set_title("input", fontsize=9)
    axd["B"].imshow(overlay(img_vis, mask_to_alpha(parent_masks[jstar], img_size))); axd["B"].axis("off")
    axd["B"].set_title(f"y=[{y_phrase}]\nparent j*={jstar} (slot {top_idx[jstar]})  P(j*|y)={P_parent[jstar]:.2f}", fontsize=9)
    axd["C"].imshow(overlay(img_vis, mask_to_alpha(child_attn[jstar, kstar], img_size))); axd["C"].axis("off")
    axd["C"].set_title(f"x=[{x_phrase}]\nchild k*={kstar}  P(k*|j*,x)={P_child[jstar, kstar]:.2f}", fontsize=9)
    draw_dist_bar(axd["D"], P_parent, "P(j|y)  over parent slots", highlight=jstar, nonempty=nonempty)
    draw_dist_bar(axd["E"], P_child[jstar], f"P(k|j*,x)  within parent {jstar}", highlight=kstar)
    draw_heat(axd["F"], w, "path weight  w_jk = P(j|y)·P(k|j,x)", star=(jstar, kstar))
    draw_heat(axd["G"], resp, f"responsibility  w_jk·P({pred_name}|c_jk)", star=(jstar, kstar))
    draw_color_bar(axd["H"], p_color[jstar, kstar], classes, pred_idx, "best-path  P(a | c_{j*k*})")
    draw_color_bar(axd["I"], marginal, classes, pred_idx, "marginal  P(a)")
    axd["J"].axis("off")
    n_par = int(nonempty.sum()) if nonempty is not None else len(P_parent)
    txt = (f"argmax path: j*={jstar} (slot {top_idx[jstar]}) → k*={kstar}\n"
           f"  P(j*|y)      = {P_parent[jstar]:.3f}\n  P(k*|j*,x)   = {P_child[jstar, kstar]:.3f}\n"
           f"  w(j*,k*)     = {w[jstar, kstar]:.3f}\n"
           f"  best-path {pred_name:<7s}= {p_color[jstar, kstar, pred_idx]:.3f}\n"
           f"  marginal  {pred_name:<7s}= {marginal[pred_idx]:.3f}\n\n"
           f"top-path share of mass: {w.max():.3f}\n"
           f"parent entropy H[P(j|y)] = {-(P_parent * np.log(P_parent + 1e-9)).sum():.2f} nats\n"
           f"(uniform over {n_par} parents = {math.log(max(n_par, 1)):.2f})")
    axd["J"].text(0.0, 0.98, txt, va="top", ha="left", fontsize=9, family="monospace")
    fig.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def sample_rows(df: pd.DataFrame, n_samples: int, seed: int, dataset: str) -> pd.DataFrame:
    """A few rows stratified over distinct questions (template-matching rows only)."""
    df = df[df["query"].apply(lambda q: parse_xy(q, dataset) is not None)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("no rows match the question template in this split")
    rng = np.random.RandomState(seed)
    queries = sorted(df["query"].unique().tolist())
    per_q = max(1, n_samples // max(len(queries), 1))
    parts = [df[df["query"] == q].sample(n=min(per_q, (df["query"] == q).sum()), random_state=rng) for q in queries]
    return pd.concat(parts).sample(frac=1.0, random_state=rng).head(n_samples).reset_index(drop=True)


@torch.no_grad()
def render_training_samples(model, val_rows, spans, label_vocab, dataset, image_root, transform, img_size,
                            device, epoch, out_dir, n_samples):
    """Per-checkpoint routing figures of a FIXED validation sample (seed 0) so routing can be watched sharpening."""
    classes = classes_from_vocab(label_vocab)
    sample = sample_rows(val_rows, n_samples, 0, dataset)
    was_training = model.training
    model.eval()
    for i, row in sample.iterrows():
        pil = PILImage.open(resolve_image_path(image_root, row["image_name"])).convert("RGB")
        img_t = transform(pil).unsqueeze(0).to(device)
        out = model.trace_images(img_t, spans[row["query"]].unsqueeze(0).to(device))
        x_phrase, y_phrase = parse_xy(row["query"], dataset)
        draw_sample(out, denorm(img_t[0]), img_size, classes, row["label"], x_phrase, y_phrase,
                    Path(out_dir) / f"hier_routing_s{i:02d}_e{epoch:03d}.png")
    if was_training:
        model.train()
    print(f"  [viz] {len(sample)} routing figures → {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dinosaur_ckpt", default=None, help=f"default: {DEFAULT_DINOSAUR_CKPT}")
    ap.add_argument("--out_dir", default="figures/routing")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, info = load_checkpoint(args.checkpoint, device=device, dinosaur_ckpt=args.dinosaur_ckpt)
    cfg = info["config"]
    if not is_router(cfg["model"]):
        raise SystemExit(f"{args.checkpoint} is a {cfg['model']} checkpoint, not a router")
    spec = PRESETS[cfg["dataset"]]
    csv = args.csv or (cfg.get("csv") if cfg.get("csv") and os.path.exists(cfg["csv"]) else spec.csv)
    image_root = args.image_root or spec.image_root
    split = args.split or spec.val_split
    df = load_frame(csv, cfg.get("category_filter"))
    sample = sample_rows(df[df["split"] == split], args.n_samples, args.seed, cfg["dataset"])
    spans = encode_spans(sample["query"].unique().tolist(), cfg["dataset"], device)
    transform = build_image_transform(cfg["img_size"], cfg["resize_mode"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in sample.iterrows():
        pil = PILImage.open(resolve_image_path(image_root, row["image_name"])).convert("RGB")
        img_t = transform(pil).unsqueeze(0).to(device)
        out = model.trace_images(img_t, spans[row["query"]].unsqueeze(0).to(device))
        x_phrase, y_phrase = parse_xy(row["query"], cfg["dataset"])
        path = out_dir / f"sample_{i:02d}.png"
        draw_sample(out, denorm(img_t[0]), cfg["img_size"], info["classes"], row["label"], x_phrase, y_phrase, path)
        pred = info["classes"][int(out["logits"][0].argmax())]
        print(f"  [{i}] {row['query'][:60]:<60s} pred={pred:<8s} true={row['label']:<8s} → {path}")


if __name__ == "__main__":
    main()
