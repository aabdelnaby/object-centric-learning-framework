#!/usr/bin/env python3
"""Render clean, thesis-ready HierRouter localisation figures.

Reuses the model-loading + mask-extraction machinery of ``visualize_hier_routing``
but draws compact, publication-style panels: each sample is a row of
``[input | routed object mask | routed part mask]`` (+ optional colour bar), with
the queried object/part tinted and outlined with a contour so the localisation is
legible at print scale. Dataset-aware (PACO/ADE ``ade20k`` and ``cub``).

Outputs, into ``--out_dir``:
  * ``<prefix>_contact.png``  one grid with all sampled rows (for triage)
  * ``<prefix>_NN.png``       per-sample strip (drop straight into LaTeX)
The printed summary lists each row's index / query / pred / true / ✓✗ so the good
ones can be picked with ``--rows i,j,k`` for a tight final grid ``<prefix>_final.png``.

Example:
    conda run --no-capture-output -n oclf_env python make_thesis_figs.py \
        --checkpoint runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt \
        --split val --n_samples 12 --out_dir thesis_figs/paco --prefix paco
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
from PIL import Image as PILImage

from visualize_hier_routing import (
    build_model, load_trainable, make_transform, denorm, mask_to_alpha,
    draw_color_bar,
)
from precompute_features import precompute_text, parse_xy_for

OBJ_TINT = np.array([0.20, 0.85, 1.00])   # cyan  – routed object <y>
PART_TINT = np.array([1.00, 0.35, 0.15])  # orange – routed part <x>


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


def sample_rows_ds(df, dataset, n, seed, category_filter=None):
    df = df[df["query"].apply(lambda q: parse_xy_for(q, dataset) is not None)]
    if "label_rank" in df.columns:
        df = df[df["label_rank"] == 1]
    if category_filter:
        df = df[df["query"].str.contains(category_filter, case=False, na=False)]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No rows match the template/filter for this split.")
    rng = np.random.RandomState(seed)
    queries = sorted(df["query"].unique().tolist())
    per_q = max(1, n // max(len(queries), 1))
    rows = [df[df["query"] == q].sample(n=min(per_q, (df["query"] == q).sum()),
                                        random_state=rng) for q in queries]
    out = pd.concat(rows).sample(frac=1.0, random_state=rng).head(n)
    return out.reset_index(drop=True)


def run_sample(model, row, dataset, txt, tf, img_size, image_root, device, classes):
    q = row["query"]
    name = str(row["image_name"])
    pil = PILImage.open(os.path.join(image_root, name)).convert("RGB")
    img_t = tf(pil).unsqueeze(0).to(device)
    img_vis = denorm(img_t[0])
    text_hidden = txt["hidden"][q].unsqueeze(0).to(device)
    attn_mask = txt["masks"][q].unsqueeze(0).to(device)
    spans = torch.stack(
        [txt["x_vec"][q], txt["y_vec"][q], txt["xy_vec"][q], txt["readout_vec"][q]], dim=0
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.forward_hier_router_viz(img_t, text_hidden, attn_mask, spans)
    P_parent = out["P_parent"][0].cpu().numpy()
    P_child = out["P_child"][0].cpu().numpy()
    jstar = int(P_parent.argmax())
    kstar = int(P_child[jstar].argmax())
    marginal = out["logits"][0].exp().cpu().numpy()
    obj_alpha = mask_to_alpha(out["parent_masks"][0][jstar], img_size)
    part_alpha = mask_to_alpha(out["child_attn"][0][jstar, kstar], img_size)
    x_phrase, y_phrase = parse_xy_for(q, dataset)
    return dict(
        img_vis=img_vis, obj_alpha=obj_alpha, part_alpha=part_alpha,
        x=x_phrase or "whole", y=y_phrase, marginal=marginal,
        pred=classes[int(marginal.argmax())], true=str(row["label"]),
        Pj=float(P_parent[jstar]), Pk=float(P_child[jstar, kstar]),
    )


def render_strip(s, classes, save_path, with_colorbar=True):
    ncol = 4 if with_colorbar else 3
    fig, ax = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.2))
    ok = "✓" if s["pred"] == s["true"] else "✗"
    ax[0].imshow(s["img_vis"]); ax[0].axis("off")
    ax[0].set_title(f'Q: colour of [{s["x"]}] of [{s["y"]}]\n'
                    f'pred {s["pred"]} / true {s["true"]} {ok}', fontsize=9)
    clean_mask(ax[1], s["img_vis"], s["obj_alpha"], OBJ_TINT,
               f'object [{s["y"]}]  P(j*|y)={s["Pj"]:.2f}')
    clean_mask(ax[2], s["img_vis"], s["part_alpha"], PART_TINT,
               f'part [{s["x"]}]  P(k*|j*,x)={s["Pk"]:.2f}')
    if with_colorbar:
        draw_color_bar(ax[3], s["marginal"], classes, int(s["marginal"].argmax()),
                       "P(answer)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def render_grid(samples, classes, save_path, with_colorbar=False, title=None,
                row_ids=None):
    n = len(samples)
    ncol = 4 if with_colorbar else 3
    fig, axes = plt.subplots(n, ncol, figsize=(3.0 * ncol, 3.0 * n),
                             squeeze=False)
    for r, s in enumerate(samples):
        ok = "✓" if s["pred"] == s["true"] else "✗"
        rid = f"[{row_ids[r]}] " if row_ids is not None else ""
        axes[r][0].imshow(s["img_vis"]); axes[r][0].axis("off")
        axes[r][0].set_title(f'{rid}colour of [{s["x"]}] of [{s["y"]}]  '
                             f'pred {s["pred"]}/true {s["true"]} {ok}', fontsize=9)
        clean_mask(axes[r][1], s["img_vis"], s["obj_alpha"], OBJ_TINT,
                   f'object [{s["y"]}]' if r == 0 else None)
        clean_mask(axes[r][2], s["img_vis"], s["part_alpha"], PART_TINT,
                   f'part [{s["x"]}]' if r == 0 else None)
        if with_colorbar:
            draw_color_bar(axes[r][3], s["marginal"], classes,
                           int(s["marginal"].argmax()),
                           "P(answer)" if r == 0 else "")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99 if title else 1.0))
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default=None, help="default: val (ade20k) / test (cub)")
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="thesis_figs")
    ap.add_argument("--prefix", default="fig")
    ap.add_argument("--rows", default=None, help="comma indices → tight final grid")
    ap.add_argument("--page_rows", type=int, default=0,
                    help="if >0, split the contact sheet into pages of this many "
                         "rows (keeps global row indices for cross-reference)")
    ap.add_argument("--select", default=None,
                    help='"seed:idx,idx;seed:idx" → one combined grid across seeds '
                         '(uses --n_samples for the deterministic per-seed pool)')
    ap.add_argument("--colorbar", action="store_true", help="add P(answer) column")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    dataset = cfg.get("dataset", "superclevr3d")
    split = args.split or ("test" if dataset == "cub" else "val")
    label_vocab = ck["label_vocab"]
    classes = [None] * len(label_vocab)
    for nm, idx in label_vocab.items():
        classes[idx] = nm

    model = build_model(cfg, len(label_vocab), device)
    if not load_trainable(model, ck["trainable_state"]):
        raise SystemExit("checkpoint has no trained router weights.")
    print(f"Loaded {dataset} router (val_acc {ck.get('val_acc'):.4f}, "
          f"epoch {ck.get('epoch')}); split={split}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tf, img_size = make_transform(cfg)

    if args.select:
        df_full = pd.read_csv(cfg["csv_path"])
        df_full = df_full[df_full["split"] == split]
        pairs = []  # ordered (seed, idx)
        for grp in args.select.split(";"):
            grp = grp.strip()
            if not grp:
                continue
            sd, idxs = grp.split(":")
            for ix in idxs.split(","):
                pairs.append((int(sd), int(ix)))
        seed_pool = {}
        chosen = []
        for sd, ix in pairs:
            if sd not in seed_pool:
                seed_pool[sd] = sample_rows_ds(df_full, dataset, args.n_samples, sd,
                                               cfg.get("category_filter"))
            chosen.append(seed_pool[sd].iloc[ix])
        queries = list({r["query"] for r in chosen})
        txt = precompute_text(queries, device, text_encoder=cfg["text_encoder"],
                              with_spans=True, span_dataset=dataset)
        samples = [run_sample(model, r, dataset, txt, tf, img_size,
                              cfg["image_root"], device, classes) for r in chosen]
        for (sd, ix), s in zip(pairs, samples):
            ok = "✓" if s["pred"] == s["true"] else "✗"
            print(f"  s{sd}[{ix}] colour of [{s['x']}] of [{s['y']}]  "
                  f"pred={s['pred']:<8s} true={s['true']:<8s} {ok}")
        render_grid(samples, classes, out_dir / f"{args.prefix}.png",
                    with_colorbar=args.colorbar)
        print(f"  → {out_dir}/{args.prefix}.png  ({len(samples)} rows)")
        return

    df = pd.read_csv(cfg["csv_path"])
    df = df[df["split"] == split]
    sample_df = sample_rows_ds(df, dataset, args.n_samples, args.seed,
                               cfg.get("category_filter"))
    queries = sample_df["query"].unique().tolist()
    txt = precompute_text(queries, device, text_encoder=cfg["text_encoder"],
                          with_spans=True, span_dataset=dataset)
    tf, img_size = make_transform(cfg)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, row in sample_df.iterrows():
        s = run_sample(model, row, dataset, txt, tf, img_size,
                       cfg["image_root"], device, classes)
        samples.append(s)
        slug = f"{s['x']}-of-{s['y']}".replace(" ", "_").replace("/", "-")
        ctag = "OK" if s["pred"] == s["true"] else "x"
        render_strip(s, classes,
                     out_dir / f"{args.prefix}_{i:03d}_{slug}_{ctag}.png",
                     with_colorbar=args.colorbar)
        ok = "✓" if s["pred"] == s["true"] else "✗"
        print(f"  [{i:2d}] colour of [{s['x']}] of [{s['y']}]  "
              f"pred={s['pred']:<8s} true={s['true']:<8s} {ok}")

    if args.rows:
        pick = [int(x) for x in args.rows.split(",")]
        render_grid([samples[i] for i in pick], classes,
                    out_dir / f"{args.prefix}_final.png",
                    with_colorbar=args.colorbar)
        print(f"  → {out_dir}/{args.prefix}_final.png  (rows {pick})")
    elif args.page_rows and args.page_rows > 0:
        n = len(samples)
        npage = math.ceil(n / args.page_rows)
        for p in range(npage):
            lo, hi = p * args.page_rows, min((p + 1) * args.page_rows, n)
            render_grid(samples[lo:hi], classes,
                        out_dir / f"{args.prefix}_contact_p{p:02d}.png",
                        with_colorbar=args.colorbar,
                        row_ids=list(range(lo, hi)),
                        title=f"{args.prefix}  rows {lo}-{hi - 1}")
            print(f"  → {out_dir}/{args.prefix}_contact_p{p:02d}.png  "
                  f"(rows {lo}-{hi - 1})")
    else:
        render_grid(samples, classes, out_dir / f"{args.prefix}_contact.png",
                    with_colorbar=args.colorbar,
                    row_ids=list(range(len(samples))))
        print(f"  → {out_dir}/{args.prefix}_contact.png  (all {len(samples)})")


if __name__ == "__main__":
    main()
