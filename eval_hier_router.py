#!/usr/bin/env python3
"""Eval the trained HierRouter checkpoint on PACO val.

Reports top-1 / top-2 / top-3 accuracy for two readouts of the router's structured
distribution:

  * MARGINAL  P(a) = Σ_{j,k} P(j|y)·P(k|j,x) · P(a | c_jk)        (what's trained)
  * BEST-PATH P(a | c_{j*k*})  where (j*, k*) = argmax_{j,k} P(j|y)·P(k|j,x)

The marginal is the model's standard output (what its val_acc tracks); the best-path
distribution skips the marginalisation and asks "what does the single argmax parent→child
path predict on its own?" — useful when you want to attribute a prediction to one slot
pair rather than the mixture.

Also dumps a per-sample CSV (top-1/2/3 names + probs for both readouts) and renders
~50 routing PNGs via visualize_hier_routing.draw_sample. Defaults assume:
    runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt

Usage:
    conda run --no-capture-output -n oclf_env python eval_hier_router.py \\
        [--checkpoint PATH] [--split val] [--n_viz 50] [--device cuda]
"""
import argparse
import json
import os
import sys
from pathlib import Path

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
import visualize_hier_routing as vhr


def build_model(cfg, num_classes, device):
    from classifier_model import SlotClassifier
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
        load_text_encoder  = False,
        finetune_ckpt_path = cfg.get("finetune_ckpt_path"),
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
    ).to(device)
    model.eval()
    return model


def load_trainable(model, trainable):
    model.text_projector.load_state_dict(trainable["text_projector"])
    model.classifier_head.load_state_dict(trainable["classifier_head"])
    if "hier_router" in trainable and hasattr(model, "hier_router"):
        model.hier_router.load_state_dict(trainable["hier_router"])


@torch.no_grad()
def viz_cached(model, dino_features, spans):
    """Run the hier_router on cached DINO features and expose its full trace.

    Equivalent to ``forward_hier_router_viz`` but takes already-extracted DINO features
    (so we can iterate the val set using the same cache the training loop did). Only
    the ``use_children=True`` path is implemented (matches the trained checkpoint).
    """
    from ocl.typing import FeatureExtractorOutput
    B = dino_features.shape[0]
    if dino_features.dtype != torch.float32:
        dino_features = dino_features.float()
    feat_out = FeatureExtractorOutput(
        features  = dino_features,
        positions = model._dino_positions.to(dino_features.device),
    )
    slots, slot_masks, embedded = model._slots_feats_from_featout(feat_out, B)
    rank_scores = slot_masks.sum(dim=-1)

    child_slots, info = model._refine_top_slots(embedded, slots, slot_masks, rank_scores)
    P, Ds = info["parent_slots"].shape[1], info["parent_slots"].shape[-1]
    K = model.recursive_children
    child_slots = child_slots.view(B, P, K, Ds)

    ref = info["parent_slots"]
    h_x = model.text_projector(spans[:, 3].to(dtype=ref.dtype))   # "<y> <x>" (child query + readout)
    h_y = model.text_projector(spans[:, 1].to(dtype=ref.dtype))   # "<y>" (parent query)
    h_readout = h_x if model.hier_router.readout_query else None
    logP_a, _ = model.hier_router(
        info["parent_slots"], child_slots, h_x, h_y, info.get("nonempty"), h_readout,
    )
    logP_parent = model.hier_router.last_logP_parent      # (B, P)
    logP_child  = model.hier_router.last_logP_child       # (B, P, K)
    logw        = logP_parent[:, :, None] + logP_child    # (B, P, K)  log w_jk
    logp_color  = model.hier_router._color_logp(child_slots, h_readout)  # (B,P,K,C)

    return {
        "logits":       logP_a,                  # (B, C)  marginal log P(a)
        "logP_parent":  logP_parent,             # (B, P)
        "logP_child":   logP_child,              # (B, P, K)
        "logw":         logw,                    # (B, P, K)
        "logp_color":   logp_color,              # (B, P, K, C)
        "parent_masks": info["parent_masks"],
        "child_attn":   info["child_attn"],
        "top_idx":      info["top_idx"],
        "nonempty":     info.get("nonempty"),
    }


def best_path_logits(out):
    """Pick (j*, k*) per sample by argmax of w_jk, return P(a | c_{j*k*}) log-probs (B, C)."""
    logw = out["logw"]                                   # (B, P, K)
    B, P, K = logw.shape
    flat = logw.view(B, P * K)
    best = flat.argmax(dim=1)                            # (B,)
    j_star, k_star = best // K, best % K
    logp_color = out["logp_color"]                       # (B, P, K, C)
    idx = torch.arange(B, device=logp_color.device)
    return logp_color[idx, j_star, k_star], j_star, k_star  # (B, C), (B,), (B,)


def topk_correct(log_or_probs: torch.Tensor, labels: torch.Tensor, ks=(1, 2, 3)) -> dict:
    maxk = max(ks)
    _, topk = log_or_probs.topk(maxk, dim=1)
    match = topk.eq(labels.unsqueeze(1))
    return {k: match[:, :k].any(dim=1).float().sum().item() for k in ks}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt")
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
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    label_vocab = ck["label_vocab"]
    num_classes = len(label_vocab)
    classes = [None] * num_classes
    for name, idx in label_vocab.items():
        classes[idx] = name
    print(f"  epoch={ck['epoch']}  val_acc(reported)={ck['val_acc']:.4f}  classes={num_classes}")

    csv_path        = args.csv_path   or cfg["csv_path"]
    image_root      = args.image_root or cfg["image_root"]
    dino_cache_path = args.dino_cache or cfg["dino_cache"]

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.checkpoint).parent / "eval")
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Build model + load weights ─────────────────────────────────────────────
    print(f"Building SlotClassifier (n_slots={cfg['n_slots']}, K={cfg['recursive_children']}, "
          f"num_heads={cfg['num_heads']}, text={cfg['text_encoder']}) on {device} …")
    model = build_model(cfg, num_classes, device)
    load_trainable(model, ck["trainable_state"])
    print(f"  trainable params: {sum(p.numel() for p in model.trainable_parameters()):,}")

    # ── In-memory T5 spans (mirror --text_in_memory at train time) ─────────────
    df_full = pd.read_csv(csv_path)
    df_full["label"] = df_full["label"].astype(str)
    uq = sorted(df_full[df_full["split"] == args.split]["query"].unique().tolist())
    print(f"Computing in-memory T5 text features (with spans) for {len(uq)} unique queries …")
    in_mem_text = precompute_text(uq, device=torch.device("cpu"),
                                  text_encoder=cfg["text_encoder"], with_spans=True)

    # ── DINO cache from disk ───────────────────────────────────────────────────
    print(f"Loading DINO cache: {dino_cache_path}")
    in_mem_dino = torch.load(dino_cache_path, map_location="cpu")

    # ── Cached dataset (returns spans) ─────────────────────────────────────────
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

    # ── Full-val pass: top-1/2/3 for MARGINAL and BEST-PATH ────────────────────
    n_total = 0
    corr_marg = {1: 0, 2: 0, 3: 0}
    corr_best = {1: 0, 2: 0, 3: 0}
    pred_rows = []
    df_iter = ds.df.reset_index(drop=True)
    cursor = 0
    print("Running val …")
    with torch.no_grad():
        for batch in tqdm(loader, unit="batch"):
            dino_feat, text_hidden, attn_mask, labels, spans = [t.to(device) for t in batch]
            out = viz_cached(model, dino_feat, spans)
            marg_log   = out["logits"]                                          # (B, C)
            best_log, j_star, k_star = best_path_logits(out)                    # (B, C)

            bs = labels.size(0)
            for k, v in topk_correct(marg_log,  labels).items(): corr_marg[k] += v
            for k, v in topk_correct(best_log, labels).items(): corr_best[k] += v
            n_total += bs

            mp = marg_log.exp().cpu(); bp = best_log.exp().cpu()
            mtop3 = mp.topk(3, dim=1); btop3 = bp.topk(3, dim=1)
            for i in range(bs):
                row = df_iter.iloc[cursor + i]
                pred_rows.append({
                    "image_name": row["image_name"],
                    "query":      row["query"],
                    "label":      classes[int(labels[i])],
                    "j_star":     int(j_star[i]),
                    "k_star":     int(k_star[i]),
                    "marg_top1":  classes[int(mtop3.indices[i, 0])], "marg_p1": float(mtop3.values[i, 0]),
                    "marg_top2":  classes[int(mtop3.indices[i, 1])], "marg_p2": float(mtop3.values[i, 1]),
                    "marg_top3":  classes[int(mtop3.indices[i, 2])], "marg_p3": float(mtop3.values[i, 2]),
                    "best_top1":  classes[int(btop3.indices[i, 0])], "best_p1": float(btop3.values[i, 0]),
                    "best_top2":  classes[int(btop3.indices[i, 1])], "best_p2": float(btop3.values[i, 1]),
                    "best_top3":  classes[int(btop3.indices[i, 2])], "best_p3": float(btop3.values[i, 2]),
                })
            cursor += bs

    marg = {k: corr_marg[k] / n_total for k in (1, 2, 3)}
    best = {k: corr_best[k] / n_total for k in (1, 2, 3)}
    print(f"\n=== HierRouter eval on {args.split}  (n={n_total:,}) ===")
    print(f"  MARGINAL  P(a)               : top1={marg[1]:.4f}  top2={marg[2]:.4f}  top3={marg[3]:.4f}")
    print(f"  BEST-PATH P(a | c_{{j*k*}})    : top1={best[1]:.4f}  top2={best[2]:.4f}  top3={best[3]:.4f}")
    print(f"  (checkpoint reported val_acc : {ck['val_acc']:.4f})")

    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
    print(f"Wrote per-sample predictions → {out_dir/'predictions.csv'}")

    # ── Viz: ~50 stratified samples ────────────────────────────────────────────
    pred_df = pd.DataFrame(pred_rows)
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
            print(f"  [skip] missing image {img_path}")
            continue
        img_t = tf(pil).unsqueeze(0).to(device)
        img_vis = vhr.denorm(img_t[0])
        text_hidden = in_mem_text["hidden"][q].unsqueeze(0).to(device)
        attn_mask   = in_mem_text["masks"][q].unsqueeze(0).to(device)
        spans = torch.stack([
            in_mem_text["x_vec"][q],
            in_mem_text["y_vec"][q],
            in_mem_text["xy_vec"][q],
            in_mem_text["readout_vec"][q],
        ], dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            # Use the model's existing viz forward (re-encodes one image — fine for 50).
            out = model.forward_hier_router_viz(img_t, text_hidden, attn_mask, spans)
        x_phrase, y_phrase = parse_xy_phrases(q)
        row_for_draw = pd.Series({"label": row["label"], "image_name": row["image_name"]})

        pred_idx  = int(out["logits"][0].argmax())
        pred_name = classes[pred_idx]
        ok = "ok" if pred_name == row["label"] else "wrong"
        save_path = viz_dir / f"sample_{i:02d}_{ok}_pred-{pred_name}_true-{row['label']}.png"
        vhr.draw_sample(out, img_vis, img_size, classes, row_for_draw,
                        x_phrase, y_phrase, True, save_path)

    # ── Summary JSON ───────────────────────────────────────────────────────────
    summary = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(ck["epoch"]),
        "checkpoint_reported_val_acc": float(ck["val_acc"]),
        "split": args.split,
        "n_samples": int(n_total),
        "marginal":  {"top1": marg[1], "top2": marg[2], "top3": marg[3]},
        "best_path": {"top1": best[1], "top2": best[2], "top3": best[3]},
        "n_viz": int(len(sample)),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary → {out_dir/'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
