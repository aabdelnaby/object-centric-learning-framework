"""Train one thesis model on cached DINOv3 features.

    python -m hier_dinosaur.train --model hier_router --dataset paco --out runs/paco/hier_router
    python -m hier_dinosaur.train --model patch_qdot_projected --dataset paco --out runs/paco/patch_qdot_projected
    python -m hier_dinosaur.train --model hier_router --dataset cub  --out runs/cub/hier_router

Only the head (TextProjector + HierRouter, or TextProjector + patch head) is trained: AdamW,
linear warm-up then cosine decay, NLL on log P(a), gradient clipping at 1.0, batch 128.
The run directory receives ``best_model.pt`` (best validation accuracy), ``last.pt`` (resumable),
``metrics.csv`` (one row per epoch), ``per_query_stats.csv`` (per-question accuracy every
``--checkpoint_every`` epochs when the split has at most 200 distinct questions, i.e. CUB) and,
for routers, routing-trace figures of a fixed validation sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PRESETS, CachedFeatureDataset, build_label_vocab, check_dataset, load_frame, split_frame
from .features import build_image_transform, load_dino_cache
from .models import (CONFIG_FORMAT, DEFAULT_DINOSAUR_CKPT, MODELS, build_model, collect_trainable_state,
                     is_router, load_trainable_state, save_checkpoint)
from .text import encode_spans

# Dataset-specific defaults used in the thesis runs.
DATASET_DEFAULTS = {
    "paco": dict(n_slots=9, warmup_steps=10000, checkpoint_every=25, viz_n=4),
    "cub": dict(n_slots=7, warmup_steps=2000, checkpoint_every=30, viz_n=20),
}
PER_QUERY_MAX_QUERIES = 200


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--dataset", required=True, choices=list(PRESETS))
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--csv", default=None, help="question CSV (default: dataset preset)")
    ap.add_argument("--dino_cache", default=None, help="cached patch features (default: dataset preset)")
    ap.add_argument("--image_root", default=None, help="image root for the routing figures (default: preset)")
    ap.add_argument("--dinosaur_ckpt", default=DEFAULT_DINOSAUR_CKPT)
    ap.add_argument("--category_filter", default=None, help="keep questions containing this word (CUB: color)")
    # architecture
    ap.add_argument("--n_slots", type=int, default=None, help="object slots M (default: 9 PACO / 7 CUB)")
    ap.add_argument("--children", type=int, default=5, help="part sub-slots K per object slot")
    ap.add_argument("--parents", type=int, default=None, help="object slots kept as parents (default: all)")
    ap.add_argument("--spread", type=float, default=0.0, help="spatial-prior strength (thesis: 0)")
    ap.add_argument("--router_temp", type=float, default=0.95)
    ap.add_argument("--child_scorer", default="mlp", choices=["mlp", "bilinear"])
    ap.add_argument("--no_readout_query", action="store_true", help="plain P(a | slot) attribute head")
    ap.add_argument("--num_heads", type=int, default=8, help="Patch-QCA attention heads")
    # optimisation
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--warmup_steps", type=int, default=None, help="default: 10000 PACO / 2000 CUB")
    ap.add_argument("--max_steps", type=int, default=500_000, help="step budget (converted to epochs)")
    ap.add_argument("--max_epochs", type=int, default=None, help="overrides --max_steps")
    ap.add_argument("--patience", type=int, default=1_000_000, help="early-stopping patience in epochs")
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    # bookkeeping
    ap.add_argument("--checkpoint_every", type=int, default=None, help="per-query eval + figures every N epochs")
    ap.add_argument("--viz_n", type=int, default=None, help="routing figures per checkpoint (routers only)")
    ap.add_argument("--skip_per_query_eval", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--limit_train", type=int, default=None, help="smoke tests: use only N training rows")
    ap.add_argument("--limit_val", type=int, default=None, help="smoke tests: use only N validation rows")
    return ap.parse_args()


def make_config(args, spec) -> Dict:
    d = DATASET_DEFAULTS[args.dataset]
    pick = lambda v, k: d[k] if v is None else v
    return {
        "format": CONFIG_FORMAT, "model": args.model, "dataset": args.dataset,
        "csv": args.csv or spec.csv, "dino_cache": args.dino_cache or spec.dino_cache,
        "image_root": args.image_root or spec.image_root, "dinosaur_ckpt": args.dinosaur_ckpt,
        "category_filter": args.category_filter if args.category_filter is not None else spec.category_filter,
        "train_split": spec.train_split, "val_split": spec.val_split,
        "n_slots": pick(args.n_slots, "n_slots"), "children": args.children, "parents": args.parents,
        "spread": args.spread, "router_temp": args.router_temp, "child_scorer": args.child_scorer,
        "readout_query": not args.no_readout_query, "num_heads": args.num_heads, "legacy_strip_tokens": 0,
        "d_text": 768, "img_size": 224, "resize_mode": "square",
        "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "warmup_steps": pick(args.warmup_steps, "warmup_steps"), "max_steps": args.max_steps,
        "max_epochs": args.max_epochs, "patience": args.patience, "min_delta": args.min_delta,
        "seed": args.seed, "checkpoint_every": pick(args.checkpoint_every, "checkpoint_every"),
        "viz_n": pick(args.viz_n, "viz_n"), "skip_per_query_eval": args.skip_per_query_eval,
        "limit_train": args.limit_train, "limit_val": args.limit_val,
    }


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, desc=""):
    """One pass over ``loader``. Returns ``(mean loss, accuracy, per-row correctness array)``."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total, hits = 0.0, 0, 0, []
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True, mininterval=30.0)
    with torch.set_grad_enabled(training):
        for feats, spans, labels in pbar:
            feats, spans, labels = feats.to(device), spans.to(device), labels.to(device)
            logits = model(feats, spans)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            hit = (logits.argmax(1) == labels)
            hits.append(hit.cpu().numpy())
            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct += int(hit.sum().item())
            total += bs
            pbar.set_postfix(loss=f"{total_loss / total:.3f}", acc=f"{correct / total:.3f}")
    return total_loss / max(total, 1), correct / max(total, 1), np.concatenate(hits) if hits else np.zeros(0)


def per_query_accuracy(rows, hits: np.ndarray) -> Dict[str, tuple]:
    out = {}
    for query, idx in rows.groupby("query").indices.items():
        out[query] = (float(hits[idx].mean()), int(len(idx)))
    return out


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    spec = check_dataset(args.dataset)
    cfg = make_config(args, spec)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={cfg['model']}  dataset={cfg['dataset']}  device={device}  out={out_dir}")

    # ── data ─────────────────────────────────────────────────────────────────
    df = load_frame(cfg["csv"], cfg["category_filter"])
    label_vocab = build_label_vocab(df, cfg["train_split"])
    train_rows = split_frame(df, cfg["train_split"], label_vocab)
    val_rows = split_frame(df, cfg["val_split"], label_vocab)
    if cfg["limit_train"]:
        train_rows = train_rows.head(cfg["limit_train"]).reset_index(drop=True)
    if cfg["limit_val"]:
        val_rows = val_rows.head(cfg["limit_val"]).reset_index(drop=True)
    queries = sorted(set(train_rows["query"]) | set(val_rows["query"]))
    (out_dir / "label_vocab.json").write_text(json.dumps(label_vocab, indent=2))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"train rows: {len(train_rows):,}  val rows ({cfg['val_split']}): {len(val_rows):,}  "
          f"classes: {len(label_vocab)}  questions: {len(queries)}")

    spans = encode_spans(queries, cfg["dataset"], device)
    print(f"Loading feature cache {cfg['dino_cache']} …", flush=True)
    dino_features = load_dino_cache(cfg["dino_cache"])["features"]
    train_ds = CachedFeatureDataset(train_rows, dino_features, spans, label_vocab)
    val_ds = CachedFeatureDataset(val_rows, dino_features, spans, label_vocab)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # ── model / optimiser ────────────────────────────────────────────────────
    model = build_model(cfg, len(label_vocab), device=device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters() if p.requires_grad)
    print(f"trainable parameters: {n_trainable:,}")

    steps_per_epoch = max(1, len(train_loader))
    max_epochs = cfg["max_epochs"] or math.ceil(cfg["max_steps"] / steps_per_epoch)
    total_steps = max_epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = cosine_with_warmup(optimizer, cfg["warmup_steps"], total_steps)
    criterion = nn.NLLLoss()
    print(f"AdamW lr={cfg['lr']} wd={cfg['weight_decay']} | warmup={cfg['warmup_steps']} "
          f"cosine over {total_steps:,} steps | {steps_per_epoch} steps/epoch → max_epochs={max_epochs}")

    # ── resume ───────────────────────────────────────────────────────────────
    start_epoch, best_val_acc, epochs_no_improve = 0, 0.0, 0
    last_path = out_dir / "last.pt"
    resumed = False
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        load_trainable_state(model, state["trainable_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch, best_val_acc = int(state["epoch"]), float(state["best_val_acc"])
        epochs_no_improve, resumed = int(state["epochs_no_improve"]), True
        print(f"Resumed from {last_path}: epoch {start_epoch}, best val acc {best_val_acc:.4f}")

    def open_csv(name, header):
        path = out_dir / name
        append = resumed and path.exists()
        fh = open(path, "a" if append else "w", newline="")
        writer = csv.writer(fh)
        if not append:
            writer.writerow(header)
        return fh, writer

    metrics_fh, metrics_w = open_csv("metrics.csv", ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best"])
    pq_fh, pq_w = open_csv("per_query_stats.csv", ["epoch", "query", "val_acc", "n_samples"])
    do_per_query = len(queries) <= PER_QUERY_MAX_QUERIES and not cfg["skip_per_query_eval"]
    do_viz = is_router(cfg["model"]) and cfg["viz_n"] > 0
    transform = build_image_transform(cfg["img_size"], cfg["resize_mode"])

    # ── epochs ───────────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch + 1, max_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, _ = run_epoch(model, train_loader, device, criterion, optimizer, scheduler,
                                           desc=f"epoch {epoch}/{max_epochs} train")
            va_loss, va_acc, hits = run_epoch(model, val_loader, device, criterion,
                                              desc=f"epoch {epoch}/{max_epochs} val")
            is_best = va_acc > best_val_acc + cfg["min_delta"]
            if is_best:
                best_val_acc, epochs_no_improve = va_acc, 0
                save_checkpoint(out_dir / "best_model.pt", model, cfg, label_vocab, epoch, va_acc)
            else:
                epochs_no_improve += 1
            save_checkpoint(last_path, model, cfg, label_vocab, epoch, va_acc, extra={
                "best_val_acc": best_val_acc, "epochs_no_improve": epochs_no_improve,
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            })
            marker = " ↑ best" if is_best else f" (no improvement {epochs_no_improve}/{cfg['patience']})"
            print(f"Epoch {epoch:4d}/{max_epochs} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val loss={va_loss:.4f} acc={va_acc:.4f}{marker} | {time.time() - t0:.0f}s", flush=True)
            metrics_w.writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}", int(is_best)])
            metrics_fh.flush()

            if cfg["checkpoint_every"] > 0 and epoch % cfg["checkpoint_every"] == 0:
                if do_per_query:
                    print(f"  [epoch {epoch}] per-question validation accuracy")
                    for q, (acc, n) in sorted(per_query_accuracy(val_rows, hits).items()):
                        print(f"    {q:<55s} acc={acc:.4f}  n={n}")
                        pq_w.writerow([epoch, q, f"{acc:.6f}", n])
                    pq_fh.flush()
                if do_viz:
                    try:
                        from .viz.routing import render_training_samples
                        render_training_samples(model, val_rows, spans, label_vocab, cfg["dataset"],
                                                cfg["image_root"], transform, cfg["img_size"], device,
                                                epoch, out_dir, cfg["viz_n"])
                    except Exception as err:  # a figure must never kill a long run
                        print(f"  [viz] skipped: {type(err).__name__}: {err}")

            if epochs_no_improve >= cfg["patience"]:
                print(f"Early stopping after {cfg['patience']} epochs without improvement.")
                break
    finally:
        metrics_fh.close()
        pq_fh.close()

    print(f"Done. Best validation accuracy {best_val_acc:.4f} → {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
