"""Grounding faithfulness (Table 2): does the model's attention land on the queried part?

For every PACO validation question with a ground-truth part mask, the model's spatial
attention over the queried part is compared with the mask:

    HierRouter   path-weighted child attention  sum_jk w_jk alpha_hat_jk   (and the argmax-path mask)
    Patch-QDot   the query·patch attention

Metrics (attention upsampled to the 384×384 evaluation frame, mask resized the same way):
    mass-in-mask   attention mass inside the part
    pointing       the attention peak lies inside the part
    IoU@mean       IoU of {attention > its mean} with the part
    chance         mean part-area fraction (what a uniform map would score)

    python -m hier_dinosaur.grounding --router_ckpt runs/paco/hier_router/best_model.pt \\
        --patch_ckpt runs/paco/patch_qdot_projected/best_model.pt --patch_ckpt runs/paco/patch_qdot_raw/best_model.pt \\
        --n 800 --out results/paco_grounding.json
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from .dinosaur import N_PATCHES, PATCH_GRID
from .features import build_image_transform, resolve_image_path
from .models import DEFAULT_DINOSAUR_CKPT, is_router, load_checkpoint, normalize_config, write_json
from .text import encode_spans

DEFAULT_MASKS_CSV = "data/paco/paco_val_masks.csv"
EVAL_SIZE = 384


def recon_query(part_name: str, object_name: str) -> str:
    return f"What is the color of the {part_name} of the {object_name}?"


def att_to_map(att: torch.Tensor, size: int, legacy_grid: bool = False) -> torch.Tensor:
    """(B, N) attention over patches → (B, size, size) probability maps (each sums to 1).

    N = 196 is the 14×14 ViT grid. A model trained with the legacy 4-token strip attends over
    192 tokens (patches 4..195); those are put back on their true grid positions.

    ``legacy_grid`` reproduces the thesis evaluation, which laid the first 169 of those 192 values
    on a 13×13 grid, carrying the training-time token offset into the map. See the note in
    experiments/07_eval_tables.sh for when to use it.
    """
    batch, n = att.shape
    if legacy_grid and n != N_PATCHES:
        side = math.isqrt(n)
        g = att[:, :side * side].reshape(batch, 1, side, side).clamp(min=0).float()
    else:
        if n == N_PATCHES:
            grid = att
        elif n < N_PATCHES:                      # legacy strip: token i is really patch i + offset
            grid = torch.cat([att.new_zeros(batch, N_PATCHES - n), att], dim=1)
        else:
            raise ValueError(f"unexpected attention width {n} (the ViT grid has {N_PATCHES} patches)")
        g = grid.reshape(batch, 1, PATCH_GRID, PATCH_GRID).clamp(min=0).float()
    up = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    return up / up.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-9)


def score(maps: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
    """maps (B,H,W) probabilities, masks (B,H,W) in {0,1} → per-sample metrics."""
    batch = maps.shape[0]
    mf, mm = maps.reshape(batch, -1), masks.reshape(batch, -1)
    mass = (mf * mm).sum(1)
    point = mm[torch.arange(batch), mf.argmax(1)]
    pred = (mf > mf.mean(1, keepdim=True)).float()
    inter = (pred * mm).sum(1)
    union = ((pred + mm) > 0).float().sum(1).clamp(min=1)
    return {"mass": mass, "point": point, "iou": inter / union, "area": mm.mean(1)}


def load_samples(masks_csv: str, image_root: str, n: int, seed: int, transform):
    df = pd.read_csv(masks_csv)
    df["query"] = [recon_query(p, o) for p, o in zip(df["part_name"], df["object_name"])]
    df["label"] = df["geom_label"].astype(str)
    if 0 < n < len(df):
        df = df.sample(n=n, random_state=seed)
    df = df.reset_index(drop=True)
    csv_dir = os.path.dirname(os.path.abspath(masks_csv))
    imgs, masks, keep = [], [], []
    for _, r in df.iterrows():
        img_path = resolve_image_path(image_root, r["image_path"])
        mask_path = r["part_mask_path"] if os.path.isabs(r["part_mask_path"]) else os.path.join(csv_dir, r["part_mask_path"])
        try:
            pil = PILImage.open(img_path).convert("RGB")
            m = PILImage.open(mask_path).resize((EVAL_SIZE, EVAL_SIZE), PILImage.NEAREST)
        except Exception:
            keep.append(False)
            continue
        mk = np.array(m) > 0
        if mk.ndim == 3:
            mk = mk.any(-1)
        if mk.sum() == 0:
            keep.append(False)
            continue
        imgs.append(transform(pil))
        masks.append(torch.from_numpy(mk.astype(np.float32)))
        keep.append(True)
    df = df[np.array(keep)].reset_index(drop=True)
    return df, torch.stack(imgs), torch.stack(masks)


@torch.no_grad()
def run_model(name: str, ckpt: str, df, imgs, masks, spans, device, batch: int, dinosaur_ckpt=None,
              legacy_grid: bool = False):
    model, info = load_checkpoint(ckpt, device=device, dinosaur_ckpt=dinosaur_ckpt)
    label_vocab = info["label_vocab"]
    labels = torch.tensor([label_vocab.get(l, -1) for l in df["label"]])
    router = is_router(info["config"]["model"])
    acc = {"mass": [], "point": [], "iou": [], "area": []}
    acc_arg = {k: [] for k in acc}
    correct = 0
    for s in range(0, len(df), batch):
        e = min(s + batch, len(df))
        img, mk = imgs[s:e].to(device), masks[s:e].to(device)
        sp = torch.stack([spans[q] for q in df["query"].iloc[s:e]]).to(device)
        if router:
            out = model.trace_images(img, sp)
            w, ca = out["w"], out["child_attn"]                                  # (B,P,K), (B,P,K,N)
            marg = torch.einsum("bpk,bpkn->bn", w, ca)
            flat = w.reshape(w.shape[0], -1).argmax(1)
            arg = ca[torch.arange(w.shape[0]), flat // ca.shape[2], flat % ca.shape[2]]
            logits = out["logits"]
            for tgt, store in ((marg, acc), (arg, acc_arg)):
                sc = score(att_to_map(tgt, EVAL_SIZE, legacy_grid), mk)
                for k in store:
                    store[k].append(sc[k].cpu())
        else:
            logits, attn = model.attend_images(img, sp)
            sc = score(att_to_map(attn, EVAL_SIZE, legacy_grid), mk)
            for k in acc:
                acc[k].append(sc[k].cpu())
        correct += int((logits.argmax(1).cpu() == labels[s:e]).sum())
    col_acc = correct / max(len(df), 1)
    results = {name: {k: float(torch.cat(v).mean()) for k, v in acc.items()}}
    results[name]["col_acc"] = col_acc
    if router:
        results[name + " (argmax path)"] = {k: float(torch.cat(v).mean()) for k, v in acc_arg.items()}
        results[name + " (argmax path)"]["col_acc"] = col_acc
    per_sample = {name: {k: torch.cat(v).numpy() for k, v in acc.items()}}
    if router:
        per_sample[name + " (argmax path)"] = {k: torch.cat(v).numpy() for k, v in acc_arg.items()}
    return results, per_sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router_ckpt", default=None, help="HierRouter best_model.pt")
    ap.add_argument("--patch_ckpt", action="append", default=[], help="Patch-QDot best_model.pt (repeatable)")
    ap.add_argument("--masks_csv", default=DEFAULT_MASKS_CSV)
    ap.add_argument("--image_root", default="data/coco")
    ap.add_argument("--dinosaur_ckpt", default=None, help=f"default: {DEFAULT_DINOSAUR_CKPT}")
    ap.add_argument("--n", type=int, default=800, help="questions to score (-1 = all)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/paco_grounding.json")
    ap.add_argument("--per_sample_csv", default=None)
    ap.add_argument("--legacy_grid", action="store_true",
                    help="score Patch-QDot attention as the thesis evaluation did (see experiments/07_eval_tables.sh)")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    transform = build_image_transform(224, "square")
    df, imgs, masks = load_samples(args.masks_csv, args.image_root, args.n, args.seed, transform)
    chance = float(masks.mean())
    print(f"Scoring {len(df)} PACO val questions with non-empty part masks (chance mass = {chance:.4f}, device={device})")
    spans = encode_spans(sorted(df["query"].unique()), "paco", device)

    results, per_sample = {}, {}
    if args.router_ckpt:
        torch.manual_seed(args.seed)
        r, ps = run_model("HierRouter (path-weighted)", args.router_ckpt, df, imgs, masks, spans, device, args.batch, args.dinosaur_ckpt, args.legacy_grid)
        results.update(r); per_sample.update(ps)
    for ckpt in args.patch_ckpt:
        model_name = normalize_config(torch.load(ckpt, map_location="cpu")["config"])["model"]
        name = {"patch_qdot_raw": "Patch-QDot (raw)", "patch_qdot_projected": "Patch-QDot (projected)"}.get(model_name, model_name)
        r, ps = run_model(name, ckpt, df, imgs, masks, spans, device, args.batch, args.dinosaur_ckpt, args.legacy_grid)
        results.update(r); per_sample.update(ps)

    print(f"\n{'model':<34}{'mass_in_mask':>13}{'lift':>8}{'pointing':>10}{'iou@mean':>10}{'col_acc':>9}")
    for name, m in results.items():
        print(f"{name:<34}{m['mass']:>13.4f}{m['mass'] - m['area']:>8.4f}{m['point']:>10.4f}{m['iou']:>10.4f}{m['col_acc']:>9.4f}")
    print(f"{'(uniform / chance)':<34}{chance:>13.4f}{0.0:>8.4f}{chance:>10.4f}")

    payload = {
        "dataset": "paco", "n": int(len(df)), "seed": args.seed, "chance": chance,
        "masks_csv": args.masks_csv, "legacy_grid": bool(args.legacy_grid),
        "grounding": {name: {"mass": m["mass"], "pointing": m["point"], "iou": m["iou"], "col_acc": m["col_acc"]}
                      for name, m in results.items()},
    }
    write_json(args.out, payload)
    print(f"→ {args.out}")
    if args.per_sample_csv:
        rows = []
        for name, m in per_sample.items():
            for i in range(len(df)):
                rows.append(dict(model=name, idx=i, query=df["query"].iloc[i], mass=float(m["mass"][i]),
                                 point=float(m["point"][i]), iou=float(m["iou"][i]), area=float(m["area"][i])))
        Path(args.per_sample_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.per_sample_csv, index=False)
        print(f"per-sample metrics → {args.per_sample_csv}")


if __name__ == "__main__":
    main()
