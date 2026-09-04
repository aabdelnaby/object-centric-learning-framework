"""Diagnostic evaluation of a trained Super-CLEVR-3D classifier.

For each test-split question, runs the model and aggregates accuracy by
depth, n_objects, attribute_type, and the depth × n_objects cross.

Usage (from repo root):
    conda run -n oclf_env python eval_superclevr3d.py \\
        --ckpt runs/superclevr3d_classifier_checkpoints/slots_12/best_model.pt \\
        --dinosaur_ckpt outputs/superclevr3d_dinov3/slots_12/<jobid>/checkpoints/last.ckpt \\
        --n_slots 12 \\
        --out_dir eval/superclevr3d/slots_12

For a slot-vs-patch comparison, run twice (once per checkpoint) and a third
time with --diff_a/--diff_b pointing at the two output dirs to write
slot_minus_patch.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

DEFAULT_CSV   = "FG-datset/superclevr3d/parts_vqa.csv"
DEFAULT_IMG   = "FG-datset/superclevr3d/images"
DEFAULT_DCFG  = "projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3"
DEFAULT_VOCAB = "label_vocab_superclevr3d_parts.json"

NOBJ_BUCKETS = [(3, 4), (5, 6), (7, 8), (9, 10)]


def nobj_bucket(n: int) -> str:
    for lo, hi in NOBJ_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "other"


def depth_bucket(d: int) -> str:
    return f"{d}+" if d >= 4 else str(d)


def predict_all(model, loader, device, use_cache):
    """Return predicted-label tensor (N,) on CPU."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x1, x2, attention_mask, _ = [t.to(device) for t in batch]
            if use_cache:
                logits = model.forward_cached(x1, x2, attention_mask)
            else:
                logits = model(x1, x2, attention_mask)
            preds.append(logits.argmax(1).cpu())
    return torch.cat(preds, dim=0)


def write_table(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate(df: pd.DataFrame, by: list[str]) -> list[dict]:
    g = df.groupby(by, dropna=False)
    out = []
    for key, sub in g:
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n"]  = int(len(sub))
        row["acc"] = float((sub["correct"] == 1).mean())
        out.append(row)
    return sorted(out, key=lambda r: tuple(r.get(k, "") for k in by))


def heatmap_png(df: pd.DataFrame, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index="depth", columns="nobj_bucket",
                           values="correct", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("n_objects bucket")
    ax.set_ylabel("depth")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="accuracy")
    ax.set_title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Trainable-state checkpoint from train.py (.pt)")
    ap.add_argument("--dinosaur_ckpt", default=None,
                    help="Path to DINOSAUR slot-attention .ckpt (Hydra-managed)")
    ap.add_argument("--dinosaur_cfg",  default=DEFAULT_DCFG)
    ap.add_argument("--n_slots", type=int, default=12)
    ap.add_argument("--patch_control", action="store_true", default=False)
    ap.add_argument("--csv",        default=DEFAULT_CSV)
    ap.add_argument("--image_root", default=DEFAULT_IMG)
    ap.add_argument("--vocab",      default=DEFAULT_VOCAB)
    ap.add_argument("--feat_cache", action="store_true", default=False,
                    help="Use precomputed DINO/text caches; --dino_cache/--text_cache required")
    ap.add_argument("--dino_cache", default="FG-datset/superclevr3d/dino_feat_cache.pt")
    ap.add_argument("--text_cache", default="FG-datset/superclevr3d/text_feat_cache.pt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--diff_a", default=None,
                    help="Optional: dir of run A (slot). Pairs with --diff_b to emit slot_minus_b.csv")
    ap.add_argument("--diff_b", default=None,
                    help="Optional: dir of run B (e.g. patch).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Diff-only mode: combine two existing depth_x_nobj.csv files.
    if args.diff_a and args.diff_b:
        a = pd.read_csv(Path(args.diff_a) / "depth_x_nobj.csv")
        b = pd.read_csv(Path(args.diff_b) / "depth_x_nobj.csv")
        merged = a.merge(b, on=["depth", "nobj_bucket"], suffixes=("_a", "_b"))
        merged["acc_diff_a_minus_b"] = merged["acc_a"] - merged["acc_b"]
        merged.to_csv(out_dir / "diff_a_minus_b.csv", index=False)
        print(f"Wrote {out_dir/'diff_a_minus_b.csv'}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.vocab) as f:
        label_vocab = json.load(f)
    inv_vocab = {v: k for k, v in label_vocab.items()}
    num_classes = len(label_vocab)
    print(f"Vocab size: {num_classes}")

    # ── Build model ──────────────────────────────────────────────────────
    from classifier_model import SlotClassifier, PatchClassifier

    # Peek at the checkpoint to figure out which pooler it was trained with.
    state     = torch.load(args.ckpt, map_location=device)
    trainable = state["trainable_state"]
    saved_cfg = state.get("config", {}) or {}
    pooler        = saved_cfg.get("pooler", "gated_attn")
    pooler_layers = saved_cfg.get("pooler_layers", 2)
    pooler_dropout= saved_cfg.get("pooler_dropout", 0.0)
    # Fallback when config is missing: trust whichever sub-module key is present.
    has_gca = "gated_cross_attn" in trainable
    has_xfm = "fusion_pooler"   in trainable
    has_vqa = "vqa_pooler"      in trainable
    if has_vqa:
        pooler = "vqa_transformer"
    elif has_gca and has_xfm:
        pooler = "gated_then_transformer"
    elif has_xfm:
        pooler = "transformer"
    elif has_gca:
        pooler = "gated_attn"

    model_kwargs = dict(
        dinosaur_cfg_name  = args.dinosaur_cfg,
        dinosaur_ckpt_path = args.dinosaur_ckpt,
        num_classes        = num_classes,
        d_slot             = 256,
        d_text             = 1024,
        num_heads          = 8,
        roberta_model      = "roberta-large",
        load_text_encoder  = not args.feat_cache,
    )
    if args.patch_control:
        model = PatchClassifier(
            **model_kwargs, d_vit=384,
            pooler=pooler,
            pooler_layers=pooler_layers,
            pooler_dropout=pooler_dropout,
        ).to(device)
    else:
        model = SlotClassifier(
            **model_kwargs,
            n_slots=args.n_slots,
            pooler=pooler,
            pooler_layers=pooler_layers,
            pooler_dropout=pooler_dropout,
        ).to(device)

    model.text_projector.load_state_dict(trainable["text_projector"])
    model.classifier_head.load_state_dict(trainable["classifier_head"])
    if "gated_cross_attn" in trainable and hasattr(model, "gated_cross_attn"):
        model.gated_cross_attn.load_state_dict(trainable["gated_cross_attn"])
    if "fusion_pooler" in trainable and hasattr(model, "fusion_pooler"):
        model.fusion_pooler.load_state_dict(trainable["fusion_pooler"])
    if "vqa_pooler" in trainable and hasattr(model, "vqa_pooler"):
        model.vqa_pooler.load_state_dict(trainable["vqa_pooler"])
    if "patch_projector" in trainable and hasattr(model, "patch_projector"):
        model.patch_projector.load_state_dict(trainable["patch_projector"])
    print(f"Loaded {args.ckpt} (epoch={state.get('epoch')}, val_acc={state.get('val_acc'):.4f}, pooler={pooler})")

    # ── Test loader ──────────────────────────────────────────────────────
    from superclevr3d_dataset import (
        SuperCLEVR3DAttributeDataset,
        SuperCLEVR3DCachedFeatDataset,
    )
    if args.feat_cache:
        ds = SuperCLEVR3DCachedFeatDataset(
            csv_path=args.csv,
            dino_cache_path=args.dino_cache,
            text_cache_path=args.text_cache,
            split="test",
            label_vocab=label_vocab,
        )
    else:
        from transformers import RobertaTokenizer
        tok = RobertaTokenizer.from_pretrained("roberta-large")
        ds = SuperCLEVR3DAttributeDataset(
            csv_path=args.csv,
            image_root=args.image_root,
            split="test",
            label_vocab=label_vocab,
            tokenizer=tok,
        )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    print(f"Test rows: {len(ds):,}")

    # ── Run inference ────────────────────────────────────────────────────
    pred = predict_all(model, loader, device, args.feat_cache)

    # ── Build per-row dataframe with metadata ────────────────────────────
    df = ds.df.copy()
    df["pred_idx"] = pred.numpy()
    df["pred"]     = df["pred_idx"].map(inv_vocab)
    df["correct"]  = (df["pred"] == df["label"]).astype(int)
    df["nobj_bucket"] = df["n_objects"].apply(nobj_bucket)
    df["depth_bucket"] = df["depth"].apply(depth_bucket)

    # ── Tables ───────────────────────────────────────────────────────────
    print(f"Overall test accuracy: {df['correct'].mean():.4f}")
    write_table(aggregate(df, ["depth"]),                   out_dir / "per_depth.csv")
    write_table(aggregate(df, ["nobj_bucket"]),             out_dir / "per_nobj.csv")
    write_table(aggregate(df, ["depth", "nobj_bucket"]),    out_dir / "depth_x_nobj.csv")
    write_table(aggregate(df, ["attribute_type"]),          out_dir / "per_attribute.csv")

    summary = {
        "n_test":         int(len(df)),
        "overall_acc":    float(df["correct"].mean()),
        "n_classes":      num_classes,
        "n_slots":        args.n_slots,
        "patch_control":  bool(args.patch_control),
        "ckpt":           args.ckpt,
        "dinosaur_ckpt":  args.dinosaur_ckpt,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    heatmap_png(df, out_dir / "depth_nobj_heatmap.png",
                title=("Patch control" if args.patch_control
                       else f"Slots={args.n_slots}") + "  |  test accuracy")

    print(f"Wrote tables to {out_dir}")


if __name__ == "__main__":
    main()
