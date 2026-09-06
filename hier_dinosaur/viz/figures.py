"""Thesis localisation figures: [input | routed object slot | routed part sub-slot | P(answer)] per row.

    # a pool of candidate rows (per-sample strips + contact sheets)
    python -m hier_dinosaur.viz.figures --checkpoint runs/paco/hier_router/best_model.pt \\
        --n_samples 200 --seed 0 --page_rows 20 --colorbar --out_dir figures/paco_candidates --prefix paco

    # the final grid from chosen (seed, index) pairs of such pools
    python -m hier_dinosaur.viz.figures --checkpoint ... --n_samples 200 --select "0:146,10,114" --colorbar \\
        --out_dir figures --prefix paco_localization_extra

Row titles report the question, the prediction and the ground truth; the object slot is tinted
cyan and the part sub-slot orange, each outlined with a contour of its mask.
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
from PIL import Image as PILImage

from ..data import PRESETS, load_frame
from ..features import build_image_transform, resolve_image_path
from ..models import DEFAULT_DINOSAUR_CKPT, is_router, load_checkpoint
from ..text import encode_spans, parse_xy
from .routing import denorm, draw_color_bar, mask_to_alpha

OBJ_TINT = np.array([0.20, 0.85, 1.00])    # cyan   – routed object
PART_TINT = np.array([1.00, 0.35, 0.15])   # orange – routed part


def clean_mask(ax, img_vis, alpha, tint, title=None, contour="white"):
    a = alpha[..., None]
    comp = img_vis * (1.0 - 0.40 * a) + tint[None, None, :] * (0.40 * a)
    ax.imshow(comp.clip(0, 1))
    try:
        ax.contour(alpha, levels=[0.45], colors=[contour], linewidths=1.4)
    except Exception:
        pass
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def sample_rows(df, dataset, n, seed):
    df = df[df["query"].apply(lambda q: parse_xy(q, dataset) is not None)]
    if "label_rank" in df.columns:
        df = df[df["label_rank"] == 1]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("no rows match the question template for this split")
    rng = np.random.RandomState(seed)
    queries = sorted(df["query"].unique().tolist())
    per_q = max(1, n // max(len(queries), 1))
    rows = [df[df["query"] == q].sample(n=min(per_q, (df["query"] == q).sum()), random_state=rng) for q in queries]
    return pd.concat(rows).sample(frac=1.0, random_state=rng).head(n).reset_index(drop=True)


def parse_select(spec: str):
    """Parse a row selection into (seed, index) pairs.

    Accepts a seed followed by its rows, "0:12,30,7", several such groups separated by ";",
    and the redundant form "0:12,0:30" where every row repeats its seed.
    """
    pairs, seed = [], None
    for token in spec.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            head, token = token.split(":", 1)
            seed = int(head)
        if seed is None:
            raise ValueError(f"--select must start with a seed, e.g. '0:12,30'; got {spec!r}")
        pairs.append((seed, int(token)))
    if not pairs:
        raise ValueError(f"--select selected no rows: {spec!r}")
    return pairs


@torch.no_grad()
def run_sample(model, row, dataset, spans, transform, img_size, image_root, device, classes):
    q = row["query"]
    pil = PILImage.open(resolve_image_path(image_root, str(row["image_name"]))).convert("RGB")
    img_t = transform(pil).unsqueeze(0).to(device)
    out = model.trace_images(img_t, spans[q].unsqueeze(0).to(device))
    P_parent, P_child = out["P_parent"][0].cpu().numpy(), out["P_child"][0].cpu().numpy()
    jstar = int(P_parent.argmax())
    kstar = int(P_child[jstar].argmax())
    marginal = out["logits"][0].exp().cpu().numpy()
    x_phrase, y_phrase = parse_xy(q, dataset)
    return dict(
        img_vis=denorm(img_t[0]),
        obj_alpha=mask_to_alpha(out["parent_masks"][0][jstar], img_size),
        part_alpha=mask_to_alpha(out["child_attn"][0][jstar, kstar], img_size),
        x=x_phrase or "whole", y=y_phrase, marginal=marginal,
        pred=classes[int(marginal.argmax())], true=str(row["label"]),
        Pj=float(P_parent[jstar]), Pk=float(P_child[jstar, kstar]),
    )


def render_strip(s, classes, save_path, with_colorbar=True):
    ncol = 4 if with_colorbar else 3
    fig, ax = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.2))
    ok = "✓" if s["pred"] == s["true"] else "✗"
    ax[0].imshow(s["img_vis"]); ax[0].axis("off")
    ax[0].set_title(f'Q: colour of [{s["x"]}] of [{s["y"]}]\npred {s["pred"]} / true {s["true"]} {ok}', fontsize=9)
    clean_mask(ax[1], s["img_vis"], s["obj_alpha"], OBJ_TINT, f'object [{s["y"]}]  P(j*|y)={s["Pj"]:.2f}')
    clean_mask(ax[2], s["img_vis"], s["part_alpha"], PART_TINT, f'part [{s["x"]}]  P(k*|j*,x)={s["Pk"]:.2f}')
    if with_colorbar:
        draw_color_bar(ax[3], s["marginal"], classes, int(s["marginal"].argmax()), "P(answer)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def render_grid(samples, classes, save_path, with_colorbar=False, title=None, row_ids=None):
    n = len(samples)
    ncol = 4 if with_colorbar else 3
    fig, axes = plt.subplots(n, ncol, figsize=(3.0 * ncol, 3.0 * n), squeeze=False)
    for r, s in enumerate(samples):
        ok = "✓" if s["pred"] == s["true"] else "✗"
        rid = f"[{row_ids[r]}] " if row_ids is not None else ""
        axes[r][0].imshow(s["img_vis"]); axes[r][0].axis("off")
        axes[r][0].set_title(f'{rid}colour of [{s["x"]}] of [{s["y"]}]  pred {s["pred"]}/true {s["true"]} {ok}', fontsize=9)
        clean_mask(axes[r][1], s["img_vis"], s["obj_alpha"], OBJ_TINT, f'object [{s["y"]}]' if r == 0 else None)
        clean_mask(axes[r][2], s["img_vis"], s["part_alpha"], PART_TINT, f'part [{s["x"]}]' if r == 0 else None)
        if with_colorbar:
            draw_color_bar(axes[r][3], s["marginal"], classes, int(s["marginal"].argmax()), "P(answer)" if r == 0 else "")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99 if title else 1.0))
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--split", default=None, help="default: PACO val / CUB test")
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dinosaur_ckpt", default=None, help=f"default: {DEFAULT_DINOSAUR_CKPT}")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--prefix", default="fig")
    ap.add_argument("--rows", default=None, help="comma-separated indices → tight final grid")
    ap.add_argument("--page_rows", type=int, default=0, help="split the contact sheet into pages of N rows")
    ap.add_argument("--select", default=None,
                    help='rows for one grid, e.g. "0:12,30,7" or "1:4,6;3:1,9" to mix seeds')
    ap.add_argument("--colorbar", action="store_true", help="add the P(answer) column")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, info = load_checkpoint(args.checkpoint, device=device, dinosaur_ckpt=args.dinosaur_ckpt)
    cfg, classes = info["config"], info["classes"]
    if not is_router(cfg["model"]):
        raise SystemExit(f"{args.checkpoint} is a {cfg['model']} checkpoint, not a router")
    dataset = cfg["dataset"]
    spec = PRESETS[dataset]
    csv = args.csv or (cfg.get("csv") if cfg.get("csv") and os.path.exists(cfg["csv"]) else spec.csv)
    image_root = args.image_root or spec.image_root
    split = args.split or spec.val_split
    print(f"Loaded {dataset} router (val_acc {info['val_acc']:.4f}, epoch {info['epoch']}); split={split}")
    transform = build_image_transform(cfg["img_size"], cfg["resize_mode"])
    img_size = cfg["img_size"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_full = load_frame(csv, cfg.get("category_filter"))
    df_full = df_full[df_full["split"] == split]

    if args.select:
        pairs = parse_select(args.select)
        pools, chosen = {}, []
        for sd, ix in pairs:
            if sd not in pools:
                pools[sd] = sample_rows(df_full, dataset, args.n_samples, sd)
            chosen.append(pools[sd].iloc[ix])
        spans = encode_spans(list({r["query"] for r in chosen}), dataset, device)
        samples = [run_sample(model, r, dataset, spans, transform, img_size, image_root, device, classes) for r in chosen]
        for (sd, ix), s in zip(pairs, samples):
            ok = "✓" if s["pred"] == s["true"] else "✗"
            print(f"  s{sd}[{ix}] colour of [{s['x']}] of [{s['y']}]  pred={s['pred']:<8s} true={s['true']:<8s} {ok}")
        render_grid(samples, classes, out_dir / f"{args.prefix}.png", with_colorbar=args.colorbar)
        print(f"  → {out_dir}/{args.prefix}.png  ({len(samples)} rows)")
        return

    sample_df = sample_rows(df_full, dataset, args.n_samples, args.seed)
    spans = encode_spans(sample_df["query"].unique().tolist(), dataset, device)
    samples = []
    for i, row in sample_df.iterrows():
        s = run_sample(model, row, dataset, spans, transform, img_size, image_root, device, classes)
        samples.append(s)
        slug = f"{s['x']}-of-{s['y']}".replace(" ", "_").replace("/", "-")
        ctag = "OK" if s["pred"] == s["true"] else "x"
        render_strip(s, classes, out_dir / f"{args.prefix}_{i:03d}_{slug}_{ctag}.png", with_colorbar=args.colorbar)
        ok = "✓" if s["pred"] == s["true"] else "✗"
        print(f"  [{i:3d}] colour of [{s['x']}] of [{s['y']}]  pred={s['pred']:<8s} true={s['true']:<8s} {ok}")

    if args.rows:
        pick = [int(x) for x in args.rows.split(",")]
        render_grid([samples[i] for i in pick], classes, out_dir / f"{args.prefix}_final.png", with_colorbar=args.colorbar)
        print(f"  → {out_dir}/{args.prefix}_final.png  (rows {pick})")
    elif args.page_rows > 0:
        for p in range(math.ceil(len(samples) / args.page_rows)):
            lo, hi = p * args.page_rows, min((p + 1) * args.page_rows, len(samples))
            render_grid(samples[lo:hi], classes, out_dir / f"{args.prefix}_contact_p{p:02d}.png",
                        with_colorbar=args.colorbar, row_ids=list(range(lo, hi)), title=f"{args.prefix}  rows {lo}-{hi - 1}")
            print(f"  → {out_dir}/{args.prefix}_contact_p{p:02d}.png  (rows {lo}-{hi - 1})")
    else:
        render_grid(samples, classes, out_dir / f"{args.prefix}_contact.png", with_colorbar=args.colorbar,
                    row_ids=list(range(len(samples))))
        print(f"  → {out_dir}/{args.prefix}_contact.png  (all {len(samples)})")


if __name__ == "__main__":
    main()
