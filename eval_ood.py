#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a held-out OOD scene CSV (top-1 / top-3).

Works for BOTH the hier_router (SlotClassifier) and the flat patch baseline
(PatchClassifier): the model type, cache mode and recursive flag are read from
the config saved inside the checkpoint, and the forward pass reuses train.py's own
``_forward_logits`` so the number matches the training-time val accuracy exactly.

Typical use (after training each model on the no-bedroom CSV):
    python eval_ood.py \
        --checkpoint runs/ood_bedroom_hier_router/<job>/slots_7/best_model.pt \
        --csv_path   FG-datset/paco_questions_bedroom_only.csv \
        --split      val \
        --dino_cache FG-datset/dino_feat_cache_combined_square.pt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from train import _forward_logits, _load_trainable_state          # noqa: E402
from superclevr3d_dataset import SuperCLEVR3DCachedFeatDataset     # noqa: E402
from precompute_features import precompute_text                    # noqa: E402

SPAN_POOLERS = ("hier_router", "qca", "patch_qdot")


def build_model(cfg, num_classes, device):
    """Rebuild the trained model from its saved config (Slot or Patch)."""
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
        load_text_encoder  = False,           # text comes from the in-memory cache
        pooler             = cfg["pooler"],
        pooler_layers      = cfg["pooler_layers"],
        pooler_dropout     = cfg["pooler_dropout"],
        zero_image_feats   = cfg["zero_image_feats"],
    )
    if cfg.get("patch_control"):
        model = PatchClassifier(
            d_vit = cfg["d_vit"],
            patch_qdot_project_patches = cfg["patch_qdot_project_patches"],
            patch_qdot_strip_registers = cfg["patch_qdot_strip_registers"],
            patch_qdot_temperature     = cfg["patch_qdot_temperature"],
            patch_qdot_normalize       = cfg["patch_qdot_normalize"],
            **common,
        )
    else:
        model = SlotClassifier(
            n_slots            = cfg["n_slots"],
            finetune_ckpt_path = cfg.get("finetune_ckpt_path"),
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
            router_use_children = not cfg.get("router_parent_only", False),
            router_readout_query = cfg.get("router_readout_query", True),
            router_color_source = cfg.get("router_color_source", "slot"),
            router_qdot_project_patches = cfg.get("router_qdot_project_patches", True),
            router_qdot_dropout = cfg.get("router_qdot_dropout", 0.0),
            **common,
        )
    return model.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv_path", required=True, help="OOD CSV (e.g. *_bedroom_only.csv)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--dino_cache", default=None,
                    help="override cfg['dino_cache'] (default: as trained)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default=None,
                    help="default: <ckpt_dir>/ood_<csv_stem>.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    label_vocab = ck["label_vocab"]
    num_classes = len(label_vocab)
    print(f"  epoch={ck['epoch']}  in-domain val_acc(reported)={ck['val_acc']:.4f}  "
          f"classes={num_classes}  pooler={cfg['pooler']}  "
          f"patch_control={cfg.get('patch_control', False)}")

    use_cache    = cfg["feat_cache"]
    recursive    = cfg.get("recursive_infer", False)
    return_spans = cfg["pooler"] in SPAN_POOLERS
    dino_cache_path = args.dino_cache or cfg["dino_cache"]

    model = build_model(cfg, num_classes, device)
    _load_trainable_state(model, ck["trainable_state"])

    # in-memory T5 text (mirrors --text_in_memory used at train time)
    df = pd.read_csv(args.csv_path)
    df["label"] = df["label"].astype(str)
    uq = sorted(df[df["split"] == args.split]["query"].unique().tolist())
    print(f"Computing in-memory {cfg['text_encoder']} text for {len(uq)} unique queries "
          f"(with_spans={return_spans}) …")
    in_mem_text = precompute_text(uq, device=torch.device("cpu"),
                                  text_encoder=cfg["text_encoder"],
                                  with_spans=return_spans, span_dataset=cfg["dataset"])

    print(f"Loading DINO cache: {dino_cache_path}")
    in_mem_dino = torch.load(dino_cache_path, map_location="cpu")

    ds = SuperCLEVR3DCachedFeatDataset(
        csv_path        = args.csv_path,
        dino_cache_path = dino_cache_path,
        text_cache_path = cfg.get("text_cache", "") or "",
        split           = args.split,
        label_vocab     = label_vocab,
        return_spans    = return_spans,
        text_cache      = in_mem_text,
        dino_cache      = in_mem_dino,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"OOD eval rows: {len(ds):,}  (csv={os.path.basename(args.csv_path)}, split={args.split})")

    top1 = top3 = total = 0
    for batch in tqdm(loader, unit="batch"):
        batch = [t.to(device) for t in batch]
        x1, x2, attn_mask, labels = batch[:4]
        spans = batch[4] if len(batch) > 4 else None
        logits = _forward_logits(model, x1, x2, attn_mask, use_cache, recursive, spans=spans)
        top1 += (logits.argmax(1) == labels).sum().item()
        k = min(3, logits.size(1))
        t3 = logits.topk(k, dim=1).indices
        top3 += (t3 == labels.unsqueeze(1)).any(1).sum().item()
        total += labels.size(0)

    acc1, acc3 = top1 / total, top3 / total
    print(f"\n=== OOD result ===")
    print(f"  rows           : {total}")
    print(f"  top-1 accuracy : {acc1:.4f}")
    print(f"  top-3 accuracy : {acc3:.4f}")
    print(f"  (in-domain val_acc was {ck['val_acc']:.4f}; OOD gap = {ck['val_acc']-acc1:+.4f})")

    out_json = args.out_json or str(
        Path(args.checkpoint).parent /
        f"ood_{os.path.splitext(os.path.basename(args.csv_path))[0]}.json")
    with open(out_json, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "csv_path": args.csv_path,
            "split": args.split,
            "pooler": cfg["pooler"],
            "patch_control": bool(cfg.get("patch_control", False)),
            "rows": total,
            "ood_top1": acc1,
            "ood_top3": acc3,
            "in_domain_val_acc": float(ck["val_acc"]),
        }, f, indent=2)
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
