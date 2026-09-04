#!/usr/bin/env python3
"""Visualize the inference of the hierarchical routing pooler (``pooler='hier_router'``).

For each sampled "What is the color of the <x> of the <y>?" example it draws the
structured traversal the router actually performs:

    P(j|y)  parent/object routing  →  P(k|j,x) child/part routing  →  P(a|c_jk)
    →  P(a) = Σ_{j,k} P(j|y)·P(k|j,x)·P(a|c_jk)   (marginal over parent→child paths)

Per-sample figure (2×5 panels):
  A  input image (+ question, pred vs. true colour)
  B  parent slot j* overlaid on the image            (the routed object <y>)
  C  child slot k* of j* overlaid on the image       (the routed part  <x>)
  D  P(j|y) over all parent slots (dead parents greyed, j* highlighted)
  E  P(k|j*,x) over j*'s K children (k* highlighted)
  F  path-weight heatmap w_jk = P(j|y)·P(k|j,x)       (★ at the argmax path)
  G  answer-responsibility heatmap w_jk·P(a*|c_jk)    (which path made the prediction)
  H  best-path colour distribution P(a|c_{j*k*})      (bars painted their literal colour)
  I  marginal answer distribution P(a)                (compare against H)
  J  numeric summary of the argmax path

The bars in H/I are painted with each colour class's own RGB, so the colour
read-out is legible at a glance.

Example:
    conda run --no-capture-output -n oclf_env python visualize_hier_routing.py \
        --checkpoint runs/.../slots_11/best_model.pt \
        --n_samples 6 --out_dir viz_hier_routing
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms

from classifier_model import SlotClassifier
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
DIM = 0.30  # min brightness for non-region pixels in mask overlays


def color_for(name: str):
    return COLOR_RGB.get(str(name).lower(), (0.70, 0.70, 0.70))


# ── Model loading ───────────────────────────────────────────────────────────────
def build_model(cfg: dict, num_classes: int, device: torch.device):
    """Reconstruct the SlotClassifier exactly as train.py did (cached/no-text-encoder)."""
    model = SlotClassifier(
        dinosaur_cfg_name  = cfg["dinosaur_cfg_name"],
        dinosaur_ckpt_path = cfg["dinosaur_ckpt"],
        num_classes        = num_classes,
        n_slots            = cfg["n_slots"],
        d_slot             = cfg["d_slot"],
        d_text             = cfg["d_text"],
        num_heads          = cfg["num_heads"],
        roberta_model      = cfg["roberta_model"],
        text_encoder_type  = cfg["text_encoder"],
        t5_model           = cfg["t5_model"],
        vqa_d_model        = cfg["vqa_d_model"],
        load_text_encoder  = False,            # text is precomputed separately
        finetune_ckpt_path = cfg["finetune_ckpt_path"],
        pooler             = cfg["pooler"],
        pooler_layers      = cfg["pooler_layers"],
        pooler_dropout     = cfg["pooler_dropout"],
        zero_image_feats   = cfg["zero_image_feats"],
        img_size           = cfg["img_size"],
        slot_backend       = cfg["slot_backend"],
        ftdinosaur_model   = cfg["ftdinosaur_model"],
        recursive_infer    = cfg["recursive_infer"],
        recursive_children = cfg["recursive_children"],
        recursive_parents  = cfg["recursive_parents"],
        recursive_spread   = cfg["recursive_spread"],
        recursive_include_parents = cfg["recursive_include_parents"],
        rank_method        = cfg["rank_method"],
        router_temp        = cfg["router_temp"],
        router_entropy_weight = cfg["router_entropy_weight"],
        child_scorer       = cfg["child_scorer"],
        # Router colour-readout config (newer checkpoints): patch / patch_qdot heads,
        # parent-only / no-readout ablations. Default to the original slot-readout head
        # so pre-existing checkpoints rebuild unchanged.
        router_use_children = not cfg.get("router_parent_only", False),
        router_readout_query = cfg.get("router_readout_query", True),
        router_color_source = cfg.get("router_color_source", "slot"),
        router_qdot_project_patches = cfg.get("router_qdot_project_patches", True),
        router_qdot_dropout = cfg.get("router_qdot_dropout", 0.0),
    ).to(device)
    model.eval()
    return model


def load_trainable(model, trainable: dict) -> bool:
    """Restore the trained heads; return True iff the router weights were present."""
    if "text_projector" in trainable:
        model.text_projector.load_state_dict(trainable["text_projector"])
    has_router = "hier_router" in trainable and len(trainable["hier_router"]) > 0
    if has_router:
        model.hier_router.load_state_dict(trainable["hier_router"])
    return has_router


# ── Image / mask helpers (mirror make_recursive_tree_viz in train.py) ────────────
def make_transform(cfg: dict):
    img_size = int(cfg.get("img_size", 224))
    if cfg.get("slot_backend") == "ftdinosaur":
        from ftdinosaur_inference import build_dinosaur as _bd
        return _bd.build_preprocessing(cfg["ftdinosaur_model"]), img_size
    tf = build_image_transform(img_size, cfg.get("resize_mode", "crop"))
    return tf, img_size


def denorm(t: torch.Tensor) -> np.ndarray:
    mean_t = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (t.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()


def mask_to_alpha(mask_1d: torch.Tensor, img_size: int) -> np.ndarray:
    """Patch mask (N,) → (img_size, img_size) alpha in [0,1] (max-normalised)."""
    n = mask_1d.shape[0]
    side = math.isqrt(n)
    if side * side < n:
        side += 1
    pad = side * side - n
    m = mask_1d.float()
    if pad > 0:
        m = torch.cat([m, m.new_zeros(pad)])
    grid = m.view(1, 1, side, side)
    up = F.interpolate(grid, size=(img_size, img_size), mode="bilinear", align_corners=False)
    a = up[0, 0]
    return (a / (a.max() + 1e-6)).clamp(0, 1).cpu().numpy()


def overlay(img_vis: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (img_vis * (DIM + (1.0 - DIM) * alpha[:, :, None])).clip(0, 1)


# ── Panel drawers ────────────────────────────────────────────────────────────────
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
    cols = [color_for(classes[i]) for i in order]
    bars = ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([classes[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_title(title, fontsize=9)
    for i, idx in enumerate(order):
        if idx == pred_idx:
            bars[i].set_edgecolor("crimson")
            bars[i].set_linewidth(2.2)
        ax.text(min(vals[i] + 0.02, 0.86), y[i], f"{vals[i]:.2f}",
                va="center", fontsize=7)


def draw_heat(ax, mat, title, star=None):
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("child k", fontsize=8)
    ax.set_ylabel("parent j", fontsize=8)
    ax.tick_params(labelsize=6)
    if star is not None:
        ax.plot(star[1], star[0], marker="*", color="red", markersize=13,
                markeredgecolor="white", markeredgewidth=0.6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_sample(out, img_vis, img_size, classes, row, x_phrase, y_phrase,
                trained: bool, save_path: Path):
    P_parent = out["P_parent"][0].cpu().numpy()          # (P,)
    P_child  = out["P_child"][0].cpu().numpy()           # (P, K)
    w        = out["w"][0].cpu().numpy()                 # (P, K)
    p_color  = out["p_color"][0].cpu().numpy()           # (P, K, C)
    marginal = out["logits"][0].exp().cpu().numpy()      # (C,)
    parent_masks = out["parent_masks"][0]                # (P, N)
    child_attn   = out["child_attn"][0]                  # (P, K, N)
    top_idx  = out["top_idx"][0].cpu().numpy()           # (P,) original slot ids
    nonempty = out["nonempty"][0].cpu().numpy() if out["nonempty"] is not None else None

    jstar = int(P_parent.argmax())
    kstar = int(P_child[jstar].argmax())
    pred_idx = int(marginal.argmax())
    pred_name = classes[pred_idx]
    true_name = str(row["label"])
    resp = w * p_color[:, :, pred_idx]                   # (P, K) answer responsibility

    fig, axd = plt.subplot_mosaic(
        [["A", "B", "C", "D", "E"],
         ["F", "G", "H", "I", "J"]],
        figsize=(21, 8.4),
    )
    ok = "✓" if pred_name == true_name else "✗"
    stamp = "" if trained else "   ⚠ UNTRAINED ROUTER (random init) — PIPELINE DEMO ONLY"
    fig.suptitle(
        f'Q: "color of the [{x_phrase}] of the [{y_phrase}]?"     '
        f'pred: {pred_name} ({marginal[pred_idx]:.2f})   true: {true_name} {ok}{stamp}',
        fontsize=12, color=("black" if trained else "crimson"),
    )

    # A: clean input
    axd["A"].imshow(img_vis); axd["A"].axis("off")
    axd["A"].set_title("input", fontsize=9)
    # B: routed parent (object <y>)
    axd["B"].imshow(overlay(img_vis, mask_to_alpha(parent_masks[jstar], img_size)))
    axd["B"].axis("off")
    axd["B"].set_title(f"y=[{y_phrase}]\nparent j*={jstar} (slot {top_idx[jstar]})  "
                       f"P(j*|y)={P_parent[jstar]:.2f}", fontsize=9)
    # C: routed child (part <x>)
    axd["C"].imshow(overlay(img_vis, mask_to_alpha(child_attn[jstar, kstar], img_size)))
    axd["C"].axis("off")
    axd["C"].set_title(f"x=[{x_phrase}]\nchild k*={kstar}  "
                       f"P(k*|j*,x)={P_child[jstar, kstar]:.2f}", fontsize=9)
    # D: parent routing
    draw_dist_bar(axd["D"], P_parent, "P(j|y)  over parent slots",
                  highlight=jstar, nonempty=nonempty)
    # E: child routing within j*
    draw_dist_bar(axd["E"], P_child[jstar], f"P(k|j*,x)  within parent {jstar}",
                  highlight=kstar)
    # F/G: path weights + responsibility heatmaps
    draw_heat(axd["F"], w, "path weight  w_jk = P(j|y)·P(k|j,x)", star=(jstar, kstar))
    draw_heat(axd["G"], resp, f"responsibility  w_jk·P({pred_name}|c_jk)", star=(jstar, kstar))
    # H/I: colour read-outs
    draw_color_bar(axd["H"], p_color[jstar, kstar], classes, pred_idx,
                   "best-path  P(a | c_{j*k*})")
    draw_color_bar(axd["I"], marginal, classes, pred_idx, "marginal  P(a)")
    # J: numeric summary
    axd["J"].axis("off")
    txt = (
        f"argmax path: j*={jstar} (slot {top_idx[jstar]}) → k*={kstar}\n"
        f"  P(j*|y)      = {P_parent[jstar]:.3f}\n"
        f"  P(k*|j*,x)   = {P_child[jstar, kstar]:.3f}\n"
        f"  w(j*,k*)     = {w[jstar, kstar]:.3f}\n"
        f"  best-path {pred_name:<7s}= {p_color[jstar, kstar, pred_idx]:.3f}\n"
        f"  marginal  {pred_name:<7s}= {marginal[pred_idx]:.3f}\n\n"
        f"top-path share of mass: {w.max():.3f}\n"
        f"parent entropy H[P(j|y)] = {-(P_parent*np.log(P_parent+1e-9)).sum():.2f} nats\n"
        f"(uniform over {(nonempty.sum() if nonempty is not None else len(P_parent))} "
        f"parents = {math.log(max(int(nonempty.sum()) if nonempty is not None else len(P_parent),1)):.2f})"
    )
    axd["J"].text(0.0, 0.98, txt, va="top", ha="left", fontsize=9, family="monospace")

    fig.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ── Sampling + driver ─────────────────────────────────────────────────────────────
def sample_rows(df, n_samples, seed):
    """Stratify a few rows across distinct queries (template-matching only)."""
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
    ap.add_argument("--checkpoint", required=True, help="best_model.pt / last.pt of a hier_router run")
    ap.add_argument("--csv_path", default=None, help="override CSV (default: checkpoint config)")
    ap.add_argument("--image_root", default=None, help="override image root (default: checkpoint config)")
    ap.add_argument("--split", default="val", help="dataset split to sample from")
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--out_dir", default="viz_hier_routing")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    label_vocab = ck["label_vocab"]
    classes = [None] * len(label_vocab)
    for name, idx in label_vocab.items():
        classes[idx] = name
    if cfg["pooler"] != "hier_router":
        raise SystemExit(f"checkpoint pooler is {cfg['pooler']!r}, not 'hier_router'.")

    csv_path   = args.csv_path   or cfg["csv_path"]
    image_root = args.image_root or cfg["image_root"]

    print(f"Building SlotClassifier (n_slots={cfg['n_slots']}, K={cfg['recursive_children']}, "
          f"text={cfg['text_encoder']}) on {device} …")
    model = build_model(cfg, len(label_vocab), device)
    trained = load_trainable(model, ck["trainable_state"])
    if not trained:
        print("\n" + "=" * 78)
        print("  ⚠  This checkpoint has NO hier_router weights (q_parent/q_child/color_head).")
        print("     The router is RANDOMLY INITIALISED — figures are a pipeline demo only,")
        print("     NOT the trained model. Re-train with the save-bug fix to get real results.")
        print("=" * 78 + "\n")
    else:
        print(f"Loaded trained router (epoch {ck.get('epoch','?')}, "
              f"val_acc {ck.get('val_acc','?')}).")

    df = pd.read_csv(csv_path)
    df = df[df["split"] == args.split]
    sample_df = sample_rows(df, args.n_samples, args.seed)
    queries = sample_df["query"].unique().tolist()

    txt = precompute_text(queries, device, text_encoder=cfg["text_encoder"], with_spans=True)
    tf, img_size = make_transform(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "trained" if trained else "UNTRAINED"
    for i, row in sample_df.iterrows():
        q = row["query"]
        pil = PILImage.open(os.path.join(image_root, row["image_name"])).convert("RGB")
        img_t = tf(pil).unsqueeze(0).to(device)
        img_vis = denorm(img_t[0])
        text_hidden = txt["hidden"][q].unsqueeze(0).to(device)
        attn_mask   = txt["masks"][q].unsqueeze(0).to(device)
        # 4-channel spans: ch0=<x>, ch1=<y>, ch2="<x> of the <y>", ch3="<y> <x>" readout
        # (the readout / patch / patch_qdot colour heads consume ch3).
        spans = torch.stack(
            [txt["x_vec"][q], txt["y_vec"][q], txt["xy_vec"][q], txt["readout_vec"][q]],
            dim=0,
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.forward_hier_router_viz(img_t, text_hidden, attn_mask, spans)
        x_phrase, y_phrase = parse_xy_phrases(q)
        save_path = out_dir / f"sample_{i:02d}_{tag}.png"
        draw_sample(out, img_vis, img_size, classes, row, x_phrase, y_phrase,
                    trained, save_path)
        pred = classes[int(out["logits"][0].argmax())]
        print(f"  [{i}] {q[:60]:<60s} pred={pred:<7s} true={row['label']:<7s} → {save_path}")

    print(f"\nSaved {len(sample_df)} figure(s) to {out_dir}/")


if __name__ == "__main__":
    main()
