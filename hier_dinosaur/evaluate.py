"""Evaluate trained checkpoints (Tables 1 and 3, CUB per-part accuracy).

    # one checkpoint: top-1/2/3 accuracy; routers also get the MAP-path readout and per-question
    # accuracy when the split has at most 200 distinct questions (CUB)
    python -m hier_dinosaur.evaluate --checkpoint runs/paco/hier_router/best_model.pt --out results/paco_hier_router.json

    # the thesis CUB per-part table from the statistics a training run recorded
    python -m hier_dinosaur.evaluate --from_stats runs/cub/hier_router/per_query_stats.csv --epoch 30

    # print Tables 1 / 3 / CUB from a directory of result JSONs
    python -m hier_dinosaur.evaluate --summarize results/

Routers sample the part sub-slot seeds from the conditioning prior, so their numbers move by
about ±0.005 between passes; ``--seed`` (default 0) makes a pass reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PRESETS, CachedFeatureDataset, load_frame, split_frame
from .features import load_dino_cache
from .models import DEFAULT_DINOSAUR_CKPT, is_router, load_checkpoint, write_json
from .text import encode_spans

PER_QUERY_MAX_QUERIES = 200
TOPK = (1, 2, 3)


def topk_hits(logp: torch.Tensor, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
    top = logp.topk(max(TOPK), dim=1).indices
    match = top.eq(labels[:, None])
    return {k: match[:, :k].any(dim=1) for k in TOPK}


def resolve(override: Optional[str], from_cfg: Optional[str], preset: str) -> str:
    """CLI override > path stored in the checkpoint (if it exists here) > dataset preset."""
    if override:
        return override
    if from_cfg and os.path.exists(from_cfg):
        return from_cfg
    return preset


@torch.no_grad()
def evaluate_checkpoint(checkpoint: str, split: Optional[str] = None, csv: Optional[str] = None,
                        dino_cache: Optional[str] = None, batch_size: int = 128, seed: int = 0,
                        device=None, dinosaur_ckpt: Optional[str] = None,
                        per_query: Optional[bool] = None, predictions_csv: Optional[str] = None,
                        dino_features: Optional[Dict] = None) -> Dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, info = load_checkpoint(checkpoint, device=device, dinosaur_ckpt=dinosaur_ckpt)
    cfg, label_vocab, classes = info["config"], info["label_vocab"], info["classes"]
    spec = PRESETS[cfg["dataset"]]
    split = split or spec.val_split
    csv = resolve(csv, cfg.get("csv"), spec.csv)
    dino_cache = resolve(dino_cache, cfg.get("dino_cache"), spec.dino_cache)

    df = load_frame(csv, cfg.get("category_filter"))
    rows = split_frame(df, split, label_vocab)
    queries = sorted(rows["query"].unique())
    spans = encode_spans(queries, cfg["dataset"], device)
    if dino_features is None:
        print(f"Loading feature cache {dino_cache} …", flush=True)
        dino_features = load_dino_cache(dino_cache)["features"]
    ds = CachedFeatureDataset(rows, dino_features, spans, label_vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"{cfg['model']} | {checkpoint} | split={split} n={len(ds):,} | stored val_acc={info['val_acc']:.4f} "
          f"(epoch {info['epoch']})")

    router = is_router(cfg["model"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    hits = {k: [] for k in TOPK}
    map_hits = {k: [] for k in TOPK}
    preds, map_preds, j_stars, k_stars = [], [], [], []
    for feats, sp, labels in tqdm(loader, unit="batch", leave=False):
        feats, sp, labels = feats.to(device), sp.to(device), labels.to(device)
        if router:
            out = model.route(model.tree(model.dinosaur.from_cache(feats)), sp)
            logp = out.logp_answer
            map_logp, j_star, k_star = out.map_path_logp()
            for k, h in topk_hits(map_logp, labels).items():
                map_hits[k].append(h.cpu())
            map_preds.append(map_logp.argmax(1).cpu()); j_stars.append(j_star.cpu()); k_stars.append(k_star.cpu())
        else:
            logp = model(feats, sp)
        for k, h in topk_hits(logp, labels).items():
            hits[k].append(h.cpu())
        preds.append(logp.argmax(1).cpu())

    hits = {k: torch.cat(v).numpy() for k, v in hits.items()}
    result = {
        "model": cfg["model"], "dataset": cfg["dataset"], "checkpoint": str(checkpoint), "split": split,
        "n": int(len(ds)), "seed": seed, "epoch": info["epoch"], "stored_val_acc": info["val_acc"],
        "top1": float(hits[1].mean()), "top2": float(hits[2].mean()), "top3": float(hits[3].mean()),
    }
    if router:
        mh = {k: torch.cat(v).numpy() for k, v in map_hits.items()}
        result["map_path"] = {f"top{k}": float(mh[k].mean()) for k in TOPK}
    print(f"  marginal  top1={result['top1']:.4f}  top2={result['top2']:.4f}  top3={result['top3']:.4f}")
    if router:
        m = result["map_path"]
        print(f"  MAP path  top1={m['top1']:.4f}  top2={m['top2']:.4f}  top3={m['top3']:.4f}")

    if per_query or (per_query is None and len(queries) <= PER_QUERY_MAX_QUERIES):
        pq = {}
        for q, idx in rows.groupby("query").indices.items():
            pq[q] = {"acc": float(hits[1][idx].mean()), "n": int(len(idx))}
        result["per_query"] = pq
        result["per_query_macro"] = float(np.mean([v["acc"] for v in pq.values()]))
        print("  per-question accuracy:")
        for q, v in sorted(pq.items()):
            print(f"    {q:<55s} acc={v['acc']:.4f}  n={v['n']}")
        print(f"  macro average over {len(pq)} questions: {result['per_query_macro']:.4f}")

    if predictions_csv:
        pred_df = rows[["image_name", "query", "label"]].copy()
        pred_df["pred"] = [classes[i] for i in torch.cat(preds).tolist()]
        if router:
            pred_df["map_pred"] = [classes[i] for i in torch.cat(map_preds).tolist()]
            pred_df["j_star"] = torch.cat(j_stars).tolist()
            pred_df["k_star"] = torch.cat(k_stars).tolist()
        Path(predictions_csv).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(predictions_csv, index=False)
        print(f"  predictions → {predictions_csv}")
    return result


def per_query_from_stats(stats_csv: str, epoch: int) -> Dict:
    """Per-question accuracies recorded by the trainer at a given epoch (the thesis CUB table used epoch 30)."""
    df = pd.read_csv(stats_csv)
    df = df[df["epoch"] == epoch]
    if df.empty:
        raise SystemExit(f"no rows for epoch {epoch} in {stats_csv}; available: {sorted(pd.read_csv(stats_csv)['epoch'].unique())}")
    pq = {r["query"]: {"acc": float(r["val_acc"]), "n": int(r["n_samples"])} for _, r in df.iterrows()}
    macro = float(np.mean([v["acc"] for v in pq.values()]))
    print(f"per-question accuracy at epoch {epoch} ({stats_csv}):")
    for q, v in sorted(pq.items()):
        print(f"  {q:<55s} {100 * v['acc']:5.1f}  (n={v['n']})")
    print(f"  macro average: {100 * macro:.1f}")
    return {"stats_csv": stats_csv, "epoch": epoch, "per_query": pq, "per_query_macro": macro}


ROW_NAMES = {
    "hier_router": "HierRouter (full, object→part, slot-readout)",
    "patch_qdot_projected": "Patch-QDot (projected)",
    "patch_qdot_raw": "Patch-QDot (raw)",
    "patch_qca": "Patch-QCA",
    "hier_router_parent_only": "HierRouter (parent-only, no part level)",
}


def summarize(results_dir: str) -> None:
    """Print Table 1 (PACO top-k), Table 3 (marginal vs MAP) and the CUB per-part table from result JSONs."""
    files = sorted(Path(results_dir).glob("*.json"))
    results = [json.loads(f.read_text()) for f in files]
    paco = {r["model"]: r for r in results if r.get("dataset") == "paco" and "top1" in r}
    if paco:
        print("\nTable 1 — colour accuracy, PACO val")
        print(f"| {'Model':<46} | top-1 | top-2 | top-3 | n |")
        print("|---|:-:|:-:|:-:|:-:|")
        for m in ROW_NAMES:
            if m in paco:
                r = paco[m]
                print(f"| {ROW_NAMES[m]:<46} | {r['top1']:.3f} | {r['top2']:.3f} | {r['top3']:.3f} | {r['n']} |")
    if "hier_router" in paco and "map_path" in paco["hier_router"]:
        r = paco["hier_router"]
        print("\nTable 3 — marginal vs MAP-path readout, full HierRouter, PACO val")
        print("| Readout | top-1 | top-2 | top-3 |\n|---|:-:|:-:|:-:|")
        print(f"| Marginal P(a) | {r['top1']:.3f} | {r['top2']:.3f} | {r['top3']:.3f} |")
        m = r["map_path"]
        print(f"| MAP path P(a | c_j*k*) | {m['top1']:.3f} | {m['top2']:.3f} | {m['top3']:.3f} |")
    cub = [r for r in results if r.get("dataset") == "cub" and "per_query" in r]
    grounding = [r for r in results if "grounding" in r]
    for r in cub:
        print(f"\nCUB-200 — colour accuracy ({r.get('checkpoint') or r.get('stats_csv')})")
        if "top1" in r:
            print(f"overall top-1: {100 * r['top1']:.1f}")
        print("| Part | acc (%) |\n|---|:-:|")
        for q, v in sorted(r["per_query"].items()):
            part = q.replace("What is the ", "").replace(" color of the bird?", "")
            print(f"| {part} | {100 * v['acc']:.1f} |")
        print(f"| Average | {100 * r['per_query_macro']:.1f} |")
    for r in grounding:
        print(f"\nTable 2 — grounding faithfulness (n={r['n']}, chance mass={r['chance']:.3f})")
        print("| Model | mass-in-mask | pointing | IoU@mean | col_acc |\n|---|:-:|:-:|:-:|:-:|")
        for name, m in r["grounding"].items():
            print(f"| {name} | {m['mass']:.3f} | {m['pointing']:.3f} | {m['iou']:.3f} | {m['col_acc']:.3f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default=None, help="default: the dataset's validation split (PACO val / CUB test)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dino_cache", default=None)
    ap.add_argument("--dinosaur_ckpt", default=None, help=f"default: {DEFAULT_DINOSAUR_CKPT}")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--per_query", action="store_true", help="force per-question accuracies")
    ap.add_argument("--predictions", default=None, help="write per-row predictions CSV")
    ap.add_argument("--out", default=None, help="result JSON (default: <checkpoint dir>/eval_<split>.json)")
    ap.add_argument("--from_stats", default=None, help="per_query_stats.csv of a training run")
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--summarize", default=None, help="directory of result JSONs")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if args.from_stats:
        result = per_query_from_stats(args.from_stats, args.epoch)
        result["dataset"] = "cub"
        if args.out:
            write_json(args.out, result)
        return
    if not args.checkpoint:
        raise SystemExit("pass --checkpoint, --from_stats or --summarize")
    result = evaluate_checkpoint(args.checkpoint, args.split, args.csv, args.dino_cache, args.batch_size,
                                 args.seed, args.device, args.dinosaur_ckpt,
                                 per_query=True if args.per_query else None, predictions_csv=args.predictions)
    out = args.out or str(Path(args.checkpoint).parent / f"eval_{result['split']}.json")
    write_json(out, result)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
