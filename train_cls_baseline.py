"""Minimal CLS-baseline trainer for Super-CLEVR-3D simple_color VQA.

Image side  : frozen DINOv2-S/14 CLS token (auto-precomputed on first run).
Text  side  : learned ``nn.Embedding(n_unique_queries, d_hidden)`` keyed by
              question identity — equivalent to "one-hot @ learnable W".
Head        : Linear(2*d_hidden, d_hidden) → GELU → Linear(d_hidden, n_classes)
              on ``concat(img_proj, text_emb)``.

Sanity baseline used to diagnose persistent overfitting on the slot pipeline.
If even this minimal model doesn't generalise, the issue is in the data /
labels / splits / leakage, not the architecture.

Usage:
    python train_cls_baseline.py \\
        --csv FG-datset/superclevr3d/simple_color_vqa.csv \\
        --image_root FG-datset/superclevr3d/images \\
        --cls_cache FG-datset/superclevr3d/dinov2_cls_cache.pt \\
        --attribute_filter color --depth_filter 2 --max_n_objects 3 \\
        --checkpoint_dir runs/cls_baseline_simple_onehot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Filters (mirror train.py's superclevr3d filter semantics) ────────────────

def filter_df(
    df: pd.DataFrame,
    attribute_filter: Optional[str],
    depth_filter: Optional[int],
    max_n_objects: Optional[int],
) -> pd.DataFrame:
    if attribute_filter is not None:
        df = df[df["attribute_type"] == attribute_filter]
    if depth_filter is not None:
        df = df[df["depth"] == depth_filter]
    if max_n_objects is not None:
        df = df[df["n_objects"] <= max_n_objects]
    return df.reset_index(drop=True)


# ── CLS cache: load existing, or precompute via frozen DINOv2 ────────────────

class _UniqueImageDataset(Dataset):
    def __init__(self, names, image_root, transform):
        self.names      = names
        self.image_root = image_root
        self.transform  = transform

    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        img  = Image.open(os.path.join(self.image_root, name)).convert("RGB")
        return name, self.transform(img)


def load_or_precompute_cls(
    image_names,
    image_root: str,
    cache_path: Path,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Return {image_name: tensor(384,)} of DINOv2-S/14 CLS features.

    Loads from disk if ``cache_path`` exists. Otherwise extracts CLS for the
    provided names via frozen DINOv2 (one online download on first run) and
    saves the cache atomically.
    """
    if cache_path.exists():
        print(f"Loading CLS cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"CLS cache not found — running DINOv2-S/14 over "
          f"{len(image_names)} unique images …")
    # Avoid torch.hub.load — the facebookresearch/dinov2 main branch uses
    # PEP-604 union syntax (`float | None`) which requires Python ≥ 3.10.
    # timm ships the same weights as `vit_small_patch14_dinov2.lvd142m` and
    # the same backbone, so we use it instead. Pretrained default input is
    # 518×518; we pass img_size=224 so timm interpolates the pos embed for
    # a 16×16 patch grid that matches our other 224×224 pipelines.
    import timm
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained  = True,
        num_classes = 0,            # drop classifier head
        global_pool = "token",      # return CLS token (B, 384)
        img_size    = 224,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    loader = DataLoader(
        _UniqueImageDataset(image_names, image_root, transform),
        batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    cache: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for names, imgs in tqdm(loader, desc="DINOv2 CLS"):
            imgs = imgs.to(device, non_blocking=True)
            cls  = model(imgs)                              # (B, 384)
            cls  = cls.detach().cpu()
            for name, vec in zip(names, cls):
                cache[name] = vec.clone()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, tmp)
    os.replace(tmp, cache_path)
    print(f"Saved CLS cache → {cache_path}  ({len(cache)} images)")
    return cache


# ── Dataset ──────────────────────────────────────────────────────────────────

class CLSOnehotDataset(Dataset):
    def __init__(self, df, cls_cache, question_vocab, label_vocab):
        self.df             = df.reset_index(drop=True)
        self.cls_cache      = cls_cache
        self.question_vocab = question_vocab
        self.label_vocab    = label_vocab

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row  = self.df.iloc[i]
        cls  = self.cls_cache[row["image_name"]]
        q_id = torch.tensor(self.question_vocab[row["query"]], dtype=torch.long)
        y    = torch.tensor(self.label_vocab[row["label"]],   dtype=torch.long)
        return cls, q_id, y


# ── Model ────────────────────────────────────────────────────────────────────

class CLSOnehotClassifier(nn.Module):
    """Two-tower min-baseline: concat(img_proj(CLS), text_emb(q_id)) → MLP."""

    def __init__(self, num_classes: int, n_questions: int, d_img: int = 384, d_hidden: int = 256):
        super().__init__()
        self.img_proj = nn.Linear(d_img, d_hidden)
        self.text_emb = nn.Embedding(n_questions, d_hidden)
        self.head     = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, cls_feat: torch.Tensor, q_id: torch.Tensor) -> torch.Tensor:
        img = self.img_proj(cls_feat)
        txt = self.text_emb(q_id)
        return self.head(torch.cat([img, txt], dim=-1))


# ── Train / eval epochs ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, opt, sched, criterion, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for cls, q, y in loader:
        cls, q, y = cls.to(device), q.to(device), y.to(device)
        logits = model(cls, q)
        loss   = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total_loss    += loss.item() * y.size(0)
        total_correct += (logits.argmax(-1) == y).sum().item()
        total         += y.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for cls, q, y in loader:
        cls, q, y = cls.to(device), q.to(device), y.to(device)
        logits = model(cls, q)
        loss   = criterion(logits, y)
        total_loss    += loss.item() * y.size(0)
        total_correct += (logits.argmax(-1) == y).sum().item()
        total         += y.size(0)
    return total_loss / total, total_correct / total


# ── Resume helpers (atomic last.pt + best.pt) ────────────────────────────────

def save_last(out_dir, model, opt, sched, epoch, best_val, no_improve,
              label_vocab, question_vocab, cfg):
    tmp = out_dir / "last.pt.tmp"
    torch.save({
        "epoch":              epoch,
        "best_val_acc":       best_val,
        "epochs_no_improve":  no_improve,
        "model_state":        model.state_dict(),
        "optimizer_state":    opt.state_dict(),
        "scheduler_state":    sched.state_dict(),
        "label_vocab":        label_vocab,
        "question_vocab":     question_vocab,
        "config":             cfg,
    }, tmp)
    os.replace(tmp, out_dir / "last.pt")


def load_last(path, model, opt, sched, device):
    s = torch.load(path, map_location=device)
    model.load_state_dict(s["model_state"])
    opt.load_state_dict(s["optimizer_state"])
    sched.load_state_dict(s["scheduler_state"])
    return int(s["epoch"]), float(s["best_val_acc"]), int(s["epochs_no_improve"])


def save_best(out_dir, model, epoch, val_acc, label_vocab, question_vocab, cfg):
    torch.save({
        "epoch":          epoch,
        "val_acc":        val_acc,
        "model_state":    model.state_dict(),
        "label_vocab":    label_vocab,
        "question_vocab": question_vocab,
        "config":         cfg,
    }, out_dir / "best.pt")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",              default="FG-datset/superclevr3d/simple_color_vqa.csv")
    p.add_argument("--image_root",       default="FG-datset/superclevr3d/images")
    p.add_argument("--cls_cache",        default="FG-datset/superclevr3d/dinov2_cls_cache.pt")
    p.add_argument("--attribute_filter", default=None)
    p.add_argument("--depth_filter",     type=int, default=None)
    p.add_argument("--max_n_objects",    type=int, default=None)
    p.add_argument("--checkpoint_dir",   required=True)
    p.add_argument("--d_hidden",         type=int,   default=256)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-2)
    p.add_argument("--label_smoothing",  type=float, default=0.0)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--warmup_steps",     type=int,   default=200)
    p.add_argument("--max_epochs",       type=int,   default=100)
    p.add_argument("--patience",         type=int,   default=100)
    p.add_argument("--precompute_batch_size", type=int, default=64)
    p.add_argument("--resume",           action="store_true",
                   help="Auto-resume from <checkpoint_dir>/last.pt if it exists.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}   checkpoint_dir: {out_dir}")

    # ── Load + filter CSV ───────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df["label"] = df["label"].astype(str)
    df = filter_df(df, args.attribute_filter, args.depth_filter, args.max_n_objects)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    print(f"Filtered rows  — train: {len(train_df):,}  val: {len(val_df):,}")

    # ── Vocabs (built from train) ───────────────────────────────────────────
    label_vocab    = {l: i for i, l in enumerate(sorted(train_df["label"].unique()))}
    question_vocab = {q: i for i, q in enumerate(sorted(train_df["query"].unique()))}
    print(f"label_vocab: {len(label_vocab)} classes   "
          f"question_vocab: {len(question_vocab)} unique queries")

    val_df = val_df[
        val_df["query"].isin(question_vocab) & val_df["label"].isin(label_vocab)
    ].reset_index(drop=True)
    print(f"In-vocab val rows: {len(val_df):,}")

    with open(out_dir / "label_vocab.json",    "w") as f: json.dump(label_vocab,    f, indent=2)
    with open(out_dir / "question_vocab.json", "w") as f: json.dump(question_vocab, f, indent=2)

    # ── CLS cache (precomputes on first run) ────────────────────────────────
    needed    = sorted(set(train_df["image_name"]) | set(val_df["image_name"]))
    cls_cache = load_or_precompute_cls(
        needed, args.image_root, Path(args.cls_cache),
        args.precompute_batch_size, device,
    )

    # ── Datasets / loaders ──────────────────────────────────────────────────
    train_ds = CLSOnehotDataset(train_df, cls_cache, question_vocab, label_vocab)
    val_ds   = CLSOnehotDataset(val_df,   cls_cache, question_vocab, label_vocab)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    # ── Model / opt / sched ─────────────────────────────────────────────────
    model = CLSOnehotClassifier(
        num_classes = len(label_vocab),
        n_questions = len(question_vocab),
        d_hidden    = args.d_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_epochs * max(1, len(train_loader))
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    cfg = vars(args)

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch, best_val, no_improve = 0, 0.0, 0
    resume_path = out_dir / "last.pt"
    if args.resume and resume_path.exists():
        start_epoch, best_val, no_improve = load_last(resume_path, model, opt, sched, device)
        print(f"Resumed from {resume_path}: epoch={start_epoch+1}, "
              f"best_val_acc={best_val:.4f}, no_improve={no_improve}/{args.patience}")

    metrics_path  = out_dir / "metrics.csv"
    metrics_mode  = "a" if (args.resume and metrics_path.exists()) else "w"
    metrics_fh    = open(metrics_path, metrics_mode, newline="")
    metrics_writer = csv.writer(metrics_fh)
    if metrics_mode == "w":
        metrics_writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","best"])

    # ── Train loop ──────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch + 1, args.max_epochs + 1):
            tl, ta = train_one_epoch(model, train_loader, opt, sched, criterion, device)
            vl, va = eval_epoch     (model, val_loader,          criterion, device)
            is_best = va > best_val
            if is_best:
                best_val   = va
                no_improve = 0
                save_best(out_dir, model, epoch, va, label_vocab, question_vocab, cfg)
            else:
                no_improve += 1

            save_last(out_dir, model, opt, sched, epoch, best_val, no_improve,
                      label_vocab, question_vocab, cfg)

            marker = " ↑ best" if is_best else f" (no improve {no_improve}/{args.patience})"
            print(f"Epoch {epoch:3d}/{args.max_epochs} | "
                  f"train loss={tl:.4f} acc={ta:.4f} | "
                  f"val   loss={vl:.4f} acc={va:.4f}{marker}")
            metrics_writer.writerow([epoch, f"{tl:.6f}", f"{ta:.6f}",
                                     f"{vl:.6f}", f"{va:.6f}", int(is_best)])
            metrics_fh.flush()

            if no_improve >= args.patience:
                print(f"Early stop at epoch {epoch}.")
                break
    finally:
        metrics_fh.close()

    print(f"Done. best_val_acc={best_val:.4f}")


if __name__ == "__main__":
    main()
