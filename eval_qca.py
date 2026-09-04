#!/usr/bin/env python3
"""Eval a QCA baseline checkpoint on PACO val: top-1/2/3 + attention-map figures.

Auto-detects ParentSlot-QCA (SlotClassifier) vs Patch-QCA (PatchClassifier) from the
saved config (``patch_control``).

What the attention map shows
----------------------------
The QCA head computes a single softmax over its visual tokens — that map is what the
"<y> <x>" query (e.g. "car door") actually reads to produce the colour log-probs.

  * Patch-QCA   — attn over 200 ViT tokens (4 register + 196 patch); we drop the 4
                   register tokens (no spatial position), reshape the remaining 196
                   into a 14×14 grid and upsample to the image to overlay.
  * ParentSlot  — attn over 9 object slots. Each slot has its own DINOSAUR spatial
                   mask; the "area the model used" is the WEIGHTED UNION of those
                   slot masks:  spatial(b, n) = Σ_j  α[b, j] · slot_mask[b, j, n].
                   We also print the per-slot weights so you can see WHICH slot the
                   head leaned on.

Per-sample figure (1×3):
  A  input image
  B  attention overlay (the area the model used to produce its answer)
  C  top-3 predicted colours with probabilities (bars painted their literal colour)

Outputs into <ckpt_dir>/eval/:
  summary.json        top-1/2/3 acc + meta
  predictions.csv     per-row top-3 names + probs + (parentslot only) per-slot weights
  viz/sample_##.png   ~50 attention-map figures, stratified across queries
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from precompute_features import precompute_text, parse_xy_phrases
from superclevr3d_dataset import SuperCLEVR3DCachedFeatDataset
import visualize_hier_routing as vhr   # reuse: denorm, color_for, COLOR_RGB, draw_color_bar, mask_to_alpha, overlay, make_transform


# ── Model build / load ──────────────────────────────────────────────────────────
def build_model(cfg, num_classes, device):
    from classifier_model import SlotClassifier, PatchClassifier
    common = dict(
        dinosaur_cfg_name  = cfg["dinosaur_cfg_name"],
        dinosaur_ckpt_path = cfg["dinosaur_ckpt"],
        num_classes        = num_classes,
        d_slot             = cfg["d_slot"],
        d_text             = cfg["d_text"],
        num_heads          = cfg["num_heads"],
        roberta_model      = cfg["roberta_model"],
        text_encoder_type  = cfg["text_encoder"],
        t5_model           = cfg["t5_model"],
        vqa_d_model        = cfg["vqa_d_model"],
        load_text_encoder  = False,
        pooler             = cfg["pooler"],
        pooler_layers      = cfg["pooler_layers"],
        pooler_dropout     = cfg["pooler_dropout"],
        zero_image_feats   = cfg["zero_image_feats"],
    )
    if cfg.get("patch_control", False):
        model = PatchClassifier(d_vit=cfg["d_vit"], **common).to(device)
    else:
        model = SlotClassifier(
            n_slots            = cfg["n_slots"],
            img_size           = cfg["img_size"],
            slot_backend       = cfg["slot_backend"],
            ftdinosaur_model   = cfg["ftdinosaur_model"],
            finetune_ckpt_path = cfg.get("finetune_ckpt_path"),
            recursive_infer    = cfg["recursive_infer"],
            recursive_children = cfg["recursive_children"],
            recursive_parents  = cfg["recursive_parents"],
            recursive_spread   = cfg["recursive_spread"],
            recursive_include_parents = cfg["recursive_include_parents"],
            rank_method        = cfg["rank_method"],
            router_temp        = cfg["router_temp"],
            router_entropy_weight = cfg["router_entropy_weight"],
            child_scorer       = cfg["child_scorer"],
            **common,
        ).to(device)
    model.eval()
    return model


def load_trainable(model, trainable):
    model.text_projector.load_state_dict(trainable["text_projector"])
    model.classifier_head.load_state_dict(trainable["classifier_head"])
    if "qca_head" in trainable and hasattr(model, "qca_head"):
        model.qca_head.load_state_dict(trainable["qca_head"])
    if "patch_projector" in trainable and hasattr(model, "patch_projector"):
        model.patch_projector.load_state_dict(trainable["patch_projector"])


# ── Forward helpers that also return the attention weights ──────────────────────
@torch.no_grad()
def forward_parentslot(model, dino_features, spans):
    """ParentSlot-QCA: log P(a), per-slot attention, slot masks, nonempty mask."""
    from ocl.typing import FeatureExtractorOutput
    B = dino_features.shape[0]
    if dino_features.dtype != torch.float32:
        dino_features = dino_features.float()
    feat_out = FeatureExtractorOutput(
        features  = dino_features,
        positions = model._dino_positions.to(dino_features.device),
    )
    slots, slot_masks, _ = model._slots_feats_from_featout(feat_out, B)   # slots (B,9,256), slot_masks (B,9,N)
    mass     = slot_masks.sum(dim=-1)
    nonempty = mass > 0.02 * mass.sum(dim=1, keepdim=True)
    nonempty = nonempty | (~nonempty.any(dim=1, keepdim=True))
    h_yx = model.text_projector(spans[:, 3].to(dtype=slots.dtype))         # (B, 256)
    logP, attn = model.qca_head(slots, h_yx, token_mask=nonempty)          # logP (B,C), attn (B,9)
    return logP, attn, slot_masks, nonempty


@torch.no_grad()
def forward_patch(model, dino_features, spans):
    """Patch-QCA: log P(a), attention over 200 ViT tokens."""
    df = dino_features.float() if dino_features.dtype != torch.float32 else dino_features
    patches = model.patch_projector(df)                                    # (B, 200, 256)
    h_yx    = model.text_projector(spans[:, 3].to(dtype=patches.dtype))    # (B, 256)
    logP, attn = model.qca_head(patches, h_yx)                             # logP (B,C), attn (B,200)
    return logP, attn


def topk_correct(logits, labels, ks=(1, 2, 3)):
    maxk = max(ks)
    _, topk = logits.topk(maxk, dim=1)
    match = topk.eq(labels.unsqueeze(1))
    return {k: match[:, :k].any(dim=1).float().sum().item() for k in ks}


# ── Attention-map figure ───────────────────────────────────────────────────────
def render_sample(
    save_path: Path,
    img_vis: np.ndarray,
    img_size: int,
    classes: list,
    query: str,
    label: str,
    probs: np.ndarray,           # (C,)
    attn_overlay: np.ndarray,    # (img_size, img_size) in [0,1]
    suptitle_extra: str = "",
):
    pred_idx  = int(probs.argmax())
    pred_name = classes[pred_idx]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ok = "✓" if pred_name == label else "✗"

    axs[0].imshow(img_vis); axs[0].axis("off"); axs[0].set_title("input", fontsize=9)

    axs[1].imshow(vhr.overlay(img_vis, attn_overlay)); axs[1].axis("off")
    axs[1].set_title("query → attention overlay", fontsize=9)

    vhr.draw_color_bar(axs[2], probs, classes, pred_idx, "top-3 P(colour)", topk=3)

    title = f'Q: "{query}"   pred: {pred_name} ({probs[pred_idx]:.2f})   true: {label} {ok}'
    if suptitle_extra:
        title += "\n" + suptitle_extra
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def patch_attn_to_overlay(attn: torch.Tensor, img_size: int) -> tuple:
    """Convert flat ViT-token attention (N,) to an (img_size, img_size) heatmap.

    Auto-detects the layout: the largest square ≤ N is the patch grid (14×14, 16×16,
    …); any leftover prefix is treated as register / CLS tokens (no spatial position)
    and dropped from the map. For the combined_square cache N=196 → 0 registers,
    14×14 grid. For a fresh DINOv3 cache that keeps the 4 register tokens N=200 →
    4 registers, 14×14 grid. Returns (overlay, register_weight).
    """
    N = attn.shape[0]
    side = int(math.isqrt(N))
    n_reg = N - side * side
    reg = float(attn[:n_reg].sum().item()) if n_reg > 0 else 0.0
    patch = attn[n_reg:].view(side, side)
    patch = patch / (patch.max() + 1e-9)
    grid  = patch.view(1, 1, side, side)
    up    = F.interpolate(grid, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return up[0, 0].clamp(0, 1).cpu().numpy(), reg


def slot_attn_to_overlay(attn_slots: torch.Tensor, slot_masks: torch.Tensor, img_size: int) -> np.ndarray:
    """ParentSlot-QCA: weighted union of DINOSAUR slot masks.

      spatial(n) = Σ_j  α[j] · slot_mask[j, n]

    attn_slots: (n_slots,), slot_masks: (n_slots, N_patches). N_patches is square (14*14 etc.).
    """
    m = (attn_slots[:, None] * slot_masks).sum(dim=0)            # (N_patches,)
    return vhr.mask_to_alpha(m, img_size)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv_path",   default=None)
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--dino_cache", default=None)
    ap.add_argument("--split",      default="val")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_viz",      type=int, default=50)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir",    default=None, help="default: <ckpt_dir>/eval")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    ck  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    label_vocab = ck["label_vocab"]
    num_classes = len(label_vocab)
    classes = [None] * num_classes
    for name, idx in label_vocab.items():
        classes[idx] = name
    is_patch = bool(cfg.get("patch_control", False))
    print(f"  epoch={ck['epoch']}  val_acc(reported)={ck['val_acc']:.4f}  "
          f"variant={'Patch-QCA' if is_patch else 'ParentSlot-QCA'}  classes={num_classes}")

    csv_path        = args.csv_path   or cfg["csv_path"]
    image_root      = args.image_root or cfg["image_root"]
    dino_cache_path = args.dino_cache or cfg["dino_cache"]
    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.checkpoint).parent / "eval")
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Build + load
    print(f"Building model (pooler={cfg['pooler']}, num_heads={cfg['num_heads']}) on {device} …")
    model = build_model(cfg, num_classes, device)
    load_trainable(model, ck["trainable_state"])
    print(f"  trainable params: {sum(p.numel() for p in model.trainable_parameters()):,}")

    # Text in RAM (T5 with spans)
    df_full = pd.read_csv(csv_path)
    df_full["label"] = df_full["label"].astype(str)
    uq = sorted(df_full[df_full["split"] == args.split]["query"].unique().tolist())
    print(f"Computing in-memory T5 text features (with spans) for {len(uq)} unique queries …")
    in_mem_text = precompute_text(uq, device=torch.device("cpu"),
                                  text_encoder=cfg["text_encoder"], with_spans=True)

    # DINO cache
    print(f"Loading DINO cache: {dino_cache_path}")
    in_mem_dino = torch.load(dino_cache_path, map_location="cpu")

    # Dataset
    ds = SuperCLEVR3DCachedFeatDataset(
        csv_path        = csv_path,
        label_vocab     = label_vocab,
        dino_cache_path = dino_cache_path,
        text_cache_path = cfg.get("text_cache", ""),
        split           = args.split,
        return_spans    = True,
        text_cache      = in_mem_text,
        dino_cache      = in_mem_dino,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset rows: {len(ds):,}")

    # ── Full pass: top-k + collect everything needed for the figures ──────────
    n_total = 0
    correct = {1: 0, 2: 0, 3: 0}
    pred_rows = []
    all_attn      = []   # per-sample (n_slots,) or (200,)
    all_slot_masks = []  # ParentSlot only: (n_slots, N_patches) per sample
    df_iter = ds.df.reset_index(drop=True)
    cursor = 0

    print("Running val …")
    for batch in tqdm(loader, unit="batch"):
        dino_feat, text_hidden, attn_mask, labels, spans = [t.to(device) for t in batch]
        if is_patch:
            logP, attn = forward_patch(model, dino_feat, spans)
            attn_cpu = attn.cpu()
            for i in range(attn_cpu.shape[0]):
                all_attn.append(attn_cpu[i])
        else:
            logP, attn, slot_masks, nonempty = forward_parentslot(model, dino_feat, spans)
            attn_cpu = attn.cpu()
            slot_masks_cpu = slot_masks.cpu()
            for i in range(attn_cpu.shape[0]):
                all_attn.append(attn_cpu[i])
                all_slot_masks.append(slot_masks_cpu[i])

        bs = labels.size(0)
        for k, v in topk_correct(logP, labels).items():
            correct[k] += v
        n_total += bs

        probs = logP.exp().cpu()
        top3 = probs.topk(3, dim=1)
        for i in range(bs):
            row = df_iter.iloc[cursor + i]
            entry = {
                "image_name": row["image_name"],
                "query":      row["query"],
                "label":      classes[int(labels[i])],
                "top1":       classes[int(top3.indices[i, 0])], "p1": float(top3.values[i, 0]),
                "top2":       classes[int(top3.indices[i, 1])], "p2": float(top3.values[i, 1]),
                "top3":       classes[int(top3.indices[i, 2])], "p3": float(top3.values[i, 2]),
            }
            if not is_patch:
                # Per-slot α for the diagnostic CSV
                a = attn_cpu[i].tolist()
                entry["slot_attn"] = ";".join(f"{x:.3f}" for x in a)
                entry["slot_argmax"] = int(np.argmax(a))
            else:
                n_attn = attn_cpu.shape[1]
                side = int(math.isqrt(n_attn))
                n_reg = n_attn - side * side
                entry["register_weight"] = float(attn_cpu[i, :n_reg].sum().item()) if n_reg > 0 else 0.0
                entry["patch_weight"]    = float(attn_cpu[i, n_reg:].sum().item())
            pred_rows.append(entry)
        cursor += bs

    acc = {k: correct[k] / n_total for k in (1, 2, 3)}
    print(f"\n=== {('Patch-QCA' if is_patch else 'ParentSlot-QCA')} val (n={n_total:,}) ===")
    print(f"  top1={acc[1]:.4f}  top2={acc[2]:.4f}  top3={acc[3]:.4f}    "
          f"(checkpoint reported: {ck['val_acc']:.4f})")

    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
    print(f"Wrote per-sample predictions → {out_dir/'predictions.csv'}")

    # ── Viz: ~50 samples, stratified by query, mix of right/wrong ─────────────
    pred_df = pd.DataFrame(pred_rows)
    pred_df["_idx"] = pred_df.index
    pred_df["_xy"] = pred_df["query"].apply(lambda q: parse_xy_phrases(q) is not None)
    matched = pred_df[pred_df["_xy"]].copy()
    rng = np.random.RandomState(args.seed)
    n_viz = min(args.n_viz, len(matched))
    queries = sorted(matched["query"].unique())
    per_q = max(1, n_viz // max(len(queries), 1))
    parts = [matched[matched["query"] == q].sample(n=min(per_q, (matched["query"] == q).sum()),
                                                   random_state=rng)
             for q in queries]
    sample = (pd.concat(parts).sample(frac=1.0, random_state=rng).head(n_viz).reset_index(drop=True))
    print(f"Rendering {len(sample)} viz figures → {viz_dir}/")

    tf, img_size = vhr.make_transform(cfg)
    from PIL import Image as PILImage
    for i, row in tqdm(list(sample.iterrows()), unit="img"):
        q = row["query"]
        img_path = os.path.join(image_root, row["image_name"])
        try:
            pil = PILImage.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"  [skip] missing image {img_path}"); continue
        img_t   = tf(pil)
        img_vis = vhr.denorm(img_t)

        src_idx = int(row["_idx"])
        attn_i  = all_attn[src_idx]

        # Re-construct top-k probabilities from the predictions row (avoids a 2nd forward).
        probs = np.zeros(num_classes, dtype=np.float32)
        for k, p in [(row["top1"], row["p1"]), (row["top2"], row["p2"]), (row["top3"], row["p3"])]:
            probs[label_vocab[k]] = p
        # Distribute the remaining mass uniformly (cosmetic — full distribution lives in
        # predictions.csv if you need it).
        remain = max(0.0, 1.0 - probs.sum())
        if remain > 0:
            unseen = [c for c in range(num_classes) if probs[c] == 0]
            if unseen:
                probs[unseen] = remain / len(unseen)

        suptitle_extra = ""
        if is_patch:
            overlay_, reg = patch_attn_to_overlay(attn_i, img_size)
            if reg > 0:
                suptitle_extra = f"Σ attention on register tokens = {reg:.2f}  (drops out of map)"
            else:
                suptitle_extra = "no register tokens in cache — full attention is spatial"
        else:
            sm = all_slot_masks[src_idx]                    # (n_slots, N_patches)
            overlay_ = slot_attn_to_overlay(attn_i, sm, img_size)
            j_star = int(attn_i.argmax())
            top_alphas = ", ".join(f"s{j}={attn_i[j]:.2f}"
                                   for j in attn_i.argsort(descending=True)[:3].tolist())
            suptitle_extra = f"slot α: {top_alphas}    (argmax slot j*={j_star})"

        save_path = viz_dir / f"sample_{i:02d}_pred-{row['top1']}_true-{row['label']}.png"
        render_sample(save_path, img_vis, img_size, classes,
                      query=q, label=row["label"], probs=probs,
                      attn_overlay=overlay_, suptitle_extra=suptitle_extra)

    summary = {
        "checkpoint": str(args.checkpoint),
        "variant":    "Patch-QCA" if is_patch else "ParentSlot-QCA",
        "epoch":      int(ck["epoch"]),
        "checkpoint_reported_val_acc": float(ck["val_acc"]),
        "split":      args.split,
        "n_samples":  int(n_total),
        "top1_acc":   acc[1],
        "top2_acc":   acc[2],
        "top3_acc":   acc[3],
        "n_viz":      int(len(sample)),
        "out_dir":    str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary → {out_dir/'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
