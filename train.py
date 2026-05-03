"""Train SlotClassifier on CUB-200-2011 attribute classification.

Usage (from repo root):
    # Full dataset, n_slots=7, with early stopping
    conda run -n oclf_env python train.py --feat_cache

    # Sweep a specific slot count
    conda run -n oclf_env python train.py --n_slots 15 --feat_cache

    # Without feature cache (slower, raw images + RoBERTa each step)
    conda run -n oclf_env python train.py --n_slots 7

    # Single attribute
    conda run -n oclf_env python train.py --n_slots 7 \\
        --query_filter "What is the wing color of the bird?"

The DINOSAUR checkpoint and text encoder are frozen.  Only TextProjector,
GatedCrossAttention, and ClassifierHead are optimised with AdamW + cosine LR.

Early stopping: training halts when val accuracy has not improved by more than
``min_delta`` for ``patience`` consecutive epochs.

Checkpoints (trainable weights only) and per-epoch metrics are saved to
``cub_classifier_checkpoints/slots_{n_slots}/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

# ── Hyperparameters ──────────────────────────────────────────────────────────

CONFIG = {
    # --- Data ---
    "csv_path":         "FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv",
    "image_root":       "FG-datset/CUB_200_2011/images",
    "dino_cache":       "FG-datset/CUB_200_2011/dino_feat_cache.pt",
    "text_cache":       "FG-datset/CUB_200_2011/text_feat_cache.pt",
    "label_vocab_path": "label_vocab.json",
    "query_filter":     None,
    "category_filter":  None,   # substring match, e.g. "color" → all color queries

    # --- Model ---
    "dinosaur_cfg_name": "projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto_dinov3",
    "dinosaur_ckpt":     None, 
    "finetune_ckpt_path": None,   # path to cub_slot_finetune/best_checkpoint.pt
    "roberta_model":     "roberta-large",
    "patch_control":     False,  # use PatchClassifier instead of SlotClassifier
    "n_slots":           7,
    "d_vit":             384,    # ViT-S/16 feature dim (PatchClassifier only)
    "d_slot":            256,
    "d_text":            1024,
    "num_heads":         8,
    "max_text_len":      64,

    # --- Optimisation ---
    "lr":            1e-4,
    "weight_decay":  1e-2,
    "batch_size":    64,
    "max_epochs":    50,
    "warmup_steps":  200,

    # --- Early stopping ---
    "patience":   10,
    "min_delta":  1e-4,

    # --- Checkpoint visualisation ---
    "checkpoint_every": 50,   # produce per-query stats + viz every N epochs (0 = off)
    "viz_n_samples":    4,    # number of val images in each viz grid
    "viz_top_k_slots":  3,    # how many top cross-attending slots to highlight

    # --- Infra ---
    "num_workers":    4,
    "checkpoint_dir": "cub_classifier_checkpoints_100_slots",
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "feat_cache":     False,   # set True via --feat_cache flag
}


# ── LR schedule ─────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train / eval loops ───────────────────────────────────────────────────────

def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    use_cache: bool,
) -> tuple[float, float]:
    model.train()
    for _attr in ("dino_feature_extractor", "dino_conditioning", "dino_perceptual_grouping"):
        _m = getattr(model, _attr, None)
        if _m is not None:
            _m.eval()
    if model.text_encoder is not None:
        model.text_encoder.eval()

    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        x1, x2, attention_mask, labels = [t.to(device) for t in batch]

        optimizer.zero_grad()
        if use_cache:
            logits = model.forward_cached(x1, x2, attention_mask)
        else:
            logits = model(x1, x2, attention_mask)

        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_cache: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        x1, x2, attention_mask, labels = [t.to(device) for t in batch]

        if use_cache:
            logits = model.forward_cached(x1, x2, attention_mask)
        else:
            logits = model(x1, x2, attention_mask)

        loss = criterion(logits, labels)
        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

    return total_loss / total, correct / total


# ── Checkpoint / metrics helpers ─────────────────────────────────────────────

def _query_slug(query: str | None) -> str:
    if query is None:
        return "all_queries"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", query).strip("_").lower()
    return slug[:60]


def save_best_checkpoint(model, epoch, val_acc, label_vocab, cfg, out_dir, query):
    fname = "best_model.pt"
    path  = out_dir / fname
    trainable_state = {
        "text_projector":   model.text_projector.state_dict(),
        "gated_cross_attn": model.gated_cross_attn.state_dict(),
        "classifier_head":  model.classifier_head.state_dict(),
    }
    if hasattr(model, "patch_projector"):
        trainable_state["patch_projector"] = model.patch_projector.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "label_vocab": label_vocab,
            "config": cfg,
            "trainable_state": trainable_state,
        },
        path,
    )
    return path


def open_metrics_writer(out_dir: Path):
    """Open a CSV for per-epoch metrics; return (file_handle, csv.writer)."""
    path = out_dir / "metrics.csv"
    fh   = open(path, "w", newline="")
    w    = csv.writer(fh)
    w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best"])
    return fh, w


# ── Per-query evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def eval_per_query(
    model,
    queries: list[str],
    cfg: dict,
    label_vocab: dict,
    device: torch.device,
    use_cache: bool,
    tokenizer=None,
) -> dict[str, tuple[float, int]]:
    """Evaluate accuracy separately for each sub-query.

    Args:
        queries:    List of unique query strings to evaluate.
        tokenizer:  Required when ``use_cache=False``.

    Returns:
        Dict mapping query string → (accuracy, n_samples).
    """
    from cub_dataset import CUBAttributeDataset, CUBCachedFeatDataset

    criterion = nn.CrossEntropyLoss()
    results   = {}

    for query in sorted(queries):
        ds_kwargs = dict(
            csv_path     = cfg["csv_path"],
            label_vocab  = label_vocab,
            query_filter = query,
        )
        if use_cache:
            ds = CUBCachedFeatDataset(
                **ds_kwargs,
                dino_cache_path = cfg["dino_cache"],
                text_cache_path = cfg["text_cache"],
                split           = "test",
            )
        else:
            ds = CUBAttributeDataset(
                **ds_kwargs,
                image_root   = cfg["image_root"],
                split        = "test",
                tokenizer    = tokenizer,
                max_text_len = cfg["max_text_len"],
            )

        if len(ds) == 0:
            continue

        loader  = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        _, acc  = eval_epoch(model, loader, criterion, device, use_cache)
        results[query] = (acc, len(ds))

    return results


# ── Checkpoint visualisation ─────────────────────────────────────────────────

def make_checkpoint_viz(
    model,
    val_df: "pd.DataFrame",
    cfg: dict,
    label_vocab: dict,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    get_text_feats,          # callable(query) → (text_hidden (1,L,d), attn_mask (1,L))
    n_samples: int = 4,
    top_k_slots: int = 3,
) -> Path:
    """Produce a visualisation grid for random val samples.

    Each row shows one sample:
      col 0 — original image
      col 1 — all slot masks overlaid (each slot gets a distinct colour)
      cols 2..2+top_k — individual masks of the top cross-attending slots
                        with their importance percentage in the title.

    Args:
        get_text_feats: callable that accepts a query string and returns
                        ``(text_hidden, attn_mask)`` tensors on ``device``.
                        For cached training supply a lambda that reads from
                        the text cache; for non-cached supply a tokenize+encode
                        lambda.

    Returns:
        Path to the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    from torchvision import transforms

    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD  = [0.229, 0.224, 0.225]

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])
    mean_t = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGE_STD).view(3, 1, 1)

    def denorm(t: torch.Tensor) -> np.ndarray:
        return (t.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

    # ── Sample rows ──────────────────────────────────────────────────────────
    val_df = val_df[val_df["split"] == "test"]
    unique_queries = sorted(val_df["query"].unique().tolist())
    n_queries      = len(unique_queries)

    rng = np.random.RandomState(epoch)
    sample_rows = []
    per_q = max(1, n_samples // n_queries) if n_queries else 1
    for q in unique_queries:
        q_rows = val_df[val_df["query"] == q]
        n      = min(per_q, len(q_rows))
        sample_rows.append(q_rows.sample(n=n, random_state=rng))
    sample_df = pd.concat(sample_rows).head(n_samples).reset_index(drop=True)

    n_actual = len(sample_df)
    n_cols   = 2 + top_k_slots
    fig, axes = plt.subplots(n_actual, n_cols, figsize=(n_cols * 3, n_actual * 3.5))
    if n_actual == 1:
        axes = axes[None, :]

    from matplotlib import cm as _cm

    def _make_slot_colors(n_slots: int) -> np.ndarray:
        """Build (n_slots, 3) float RGB array matching the repo convention:
        tab20 for ≤20 slots, turbo for >20 (same as ocl/visualizations.py)."""
        if n_slots <= 20:
            mpl = _cm.get_cmap("tab20", n_slots)(range(n_slots))
        else:
            mpl = _cm.get_cmap("turbo", n_slots)(range(n_slots))
        return np.array([c[:3] for c in mpl], dtype=np.float32)

    model.eval()
    inv_vocab = {v: k for k, v in label_vocab.items()}

    for i, row in sample_df.iterrows():
        # ── Load raw image ────────────────────────────────────────────────
        img_path   = os.path.join(cfg["image_root"], row["image_name"])
        pil_img    = PILImage.open(img_path).convert("RGB")
        img_tensor = tf(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        img_vis    = denorm(img_tensor[0])                 # (224, 224, 3) numpy

        # ── Text features ────────────────────────────────────────────────
        query = row["query"]
        text_hidden, attn_mask = get_text_feats(query)

        # ── Forward with viz ─────────────────────────────────────────────
        with torch.no_grad():
            logits, slot_masks_raw, cross_attn_w, slot_ca_norms = model.forward_with_viz(
                img_tensor, text_hidden, attn_mask
            )

        pred_label = inv_vocab.get(logits.argmax(1).item(), "?")
        true_label = row["label"]

        # slot_masks_raw: (1, N_slots, N_patches)
        # N_patches may be 192 instead of 196 because the feature extractor hook
        # unconditionally drops 4 "register" tokens even on plain DINO ViT-S (which
        # has none).  Use ceiling-sqrt and zero-pad to the next perfect square so
        # the reshape to a 2-D grid always succeeds.
        N_slots   = slot_masks_raw.shape[1]
        N_patches = slot_masks_raw.shape[2]
        grid_side = math.isqrt(N_patches)
        if grid_side * grid_side < N_patches:
            grid_side += 1
        n_pad = grid_side * grid_side - N_patches
        if n_pad > 0:
            pad = torch.zeros(
                *slot_masks_raw.shape[:2], n_pad,
                dtype=slot_masks_raw.dtype, device=slot_masks_raw.device,
            )
            slot_masks_raw = torch.cat([slot_masks_raw, pad], dim=2)
        grid_h = grid_w = grid_side
        masks = slot_masks_raw[0].view(N_slots, grid_h, grid_w)
        masks = F.interpolate(
            masks.unsqueeze(0).float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0].cpu().numpy()  # (N_slots, 224, 224)

        # Per-slot importance: L2 norm of the cross-attention output per slot,
        # i.e. how much each slot was moved by the text query.
        # (Summing attention weights would give ≈1 for every slot since they are
        # already normalised over text tokens, making all slots look equal.)
        slot_importance = slot_ca_norms[0].cpu().numpy()  # (N_slots,)
        slot_importance = slot_importance / (slot_importance.sum() + 1e-8)
        top_k_idx       = slot_importance.argsort()[::-1][:top_k_slots]

        # One distinct colour per slot — tab20/turbo matching the repo convention.
        slot_colors = _make_slot_colors(N_slots)  # (N_slots, 3) float

        ax_row = axes[i]

        # col 0 — original image
        ax_row[0].imshow(img_vis)
        ax_row[0].set_title(
            f"true: {true_label}\npred: {pred_label}",
            fontsize=7, color=("green" if true_label == pred_label else "red"),
        )
        ax_row[0].axis("off")

        # col 1 — argmax segmentation overlaid on the image.
        # Each pixel gets the colour of its dominant slot, blended with the
        # original image so spatial content remains visible.
        argmax_slots = masks.argmax(axis=0)       # (224, 224) int
        seg_rgb      = slot_colors[argmax_slots]  # (224, 224, 3)
        seg_overlay  = 0.55 * seg_rgb + 0.45 * img_vis
        ax_row[1].imshow(seg_overlay.clip(0, 1))
        ax_row[1].set_title(f"{query[:35]}\n(segmentation overlay)", fontsize=6)
        ax_row[1].axis("off")

        # cols 2.. — slot reconstruction: original image masked by slot attention.
        #
        # feature_attributions is softmax-over-slots per patch, so each value is
        # in [0, 1] and patches sum to 1 across slots.  Multiplying by N_slots
        # re-centres so a "fair share" slot (1/N_slots at every patch) shows the
        # image at full brightness, slots that attend more to a region are brighter
        # there, and slots that attend less are darker.  This avoids the all-black
        # viridis problem caused by per-slot min-max normalisation when one slot
        # dominates and all others have near-uniform (≈0) attention.
        for k, si in enumerate(top_k_idx):
            scaled_alpha = (masks[si] * N_slots).clip(0, 1)  # (224, 224)
            masked_img   = img_vis * scaled_alpha[:, :, None]  # (224, 224, 3)
            ax_row[2 + k].imshow(masked_img.clip(0, 1))
            ax_row[2 + k].set_title(
                f"slot {si}  ({slot_importance[si]:.1%})", fontsize=8
            )
            ax_row[2 + k].axis("off")

    plt.suptitle(f"Epoch {epoch} — checkpoint visualisation", fontsize=9)
    plt.tight_layout()

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    viz_path = viz_dir / f"epoch_{epoch:04d}.png"
    plt.savefig(viz_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return viz_path


# ── Argparse ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CUB slot classifier")
    p.add_argument("--query_filter",    type=str,   default=None,
                   help="Exact-match filter on the query column")
    p.add_argument("--category_filter", type=str,   default=None,
                   help="Substring filter on the query column (e.g. 'color')")
    p.add_argument("--n_slots",         type=int,   default=None,
                   help="Number of DINOSAUR slots (default: 7)")
    p.add_argument("--max_epochs",      type=int,   default=None)
    p.add_argument("--patience",        type=int,   default=None,
                   help="Early-stopping patience in epochs (default: 10)")
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--warmup_steps",    type=int,   default=None)
    p.add_argument("--dinosaur_ckpt",   type=str,   default=None)
    p.add_argument("--finetune_ckpt",   type=str,   default=None,
                   help="Path to cub_slot_finetune/best_checkpoint.pt from finetune_slots.py")
    p.add_argument("--checkpoint_dir",  type=str,   default=None)
    p.add_argument("--checkpoint_every",type=int,   default=None,
                   help="Epoch interval for per-query stats + viz (0 = off, default: 50)")
    p.add_argument("--viz_n_samples",   type=int,   default=None,
                   help="Val images per checkpoint viz grid (default: 4)")
    p.add_argument("--feat_cache",      action="store_true", default=False,
                   help="Use pre-computed ViT+RoBERTa features (run precompute_features.py first)")
    p.add_argument("--patch_control",  action="store_true", default=False,
                   help="Use PatchClassifier (raw ViT patches) instead of SlotClassifier")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = dict(CONFIG)

    if args.query_filter      is not None: cfg["query_filter"]      = args.query_filter
    if args.category_filter   is not None: cfg["category_filter"]   = args.category_filter
    if args.n_slots           is not None: cfg["n_slots"]           = args.n_slots
    if args.max_epochs        is not None: cfg["max_epochs"]        = args.max_epochs
    if args.patience          is not None: cfg["patience"]          = args.patience
    if args.lr                is not None: cfg["lr"]                = args.lr
    if args.batch_size        is not None: cfg["batch_size"]        = args.batch_size
    if args.warmup_steps      is not None: cfg["warmup_steps"]      = args.warmup_steps
    if args.dinosaur_ckpt     is not None: cfg["dinosaur_ckpt"]       = args.dinosaur_ckpt
    if args.finetune_ckpt     is not None: cfg["finetune_ckpt_path"]  = args.finetune_ckpt
    if args.checkpoint_dir    is not None: cfg["checkpoint_dir"]      = args.checkpoint_dir
    if args.checkpoint_every  is not None: cfg["checkpoint_every"]  = args.checkpoint_every
    if args.viz_n_samples     is not None: cfg["viz_n_samples"]     = args.viz_n_samples
    if args.feat_cache:                    cfg["feat_cache"]         = True
    if args.patch_control:                 cfg["patch_control"]      = True

    device       = torch.device(cfg["device"])
    use_cache    = cfg["feat_cache"]
    n_slots      = cfg["n_slots"]
    patch_control = cfg["patch_control"]

    print(f"Device: {device}  |  n_slots: {n_slots}  |  feat_cache: {use_cache}  |  patch_control: {patch_control}")

    # ── Build label vocab ────────────────────────────────────────────────────
    from cub_dataset import CUBAttributeDataset, CUBCachedFeatDataset, build_label_vocab

    df_full = pd.read_csv(cfg["csv_path"])
    if cfg["query_filter"] is not None:
        df_filt = df_full[df_full["query"] == cfg["query_filter"]]
    elif cfg["category_filter"] is not None:
        df_filt = df_full[
            df_full["query"].str.contains(cfg["category_filter"], case=False, na=False)
        ]
    else:
        df_filt = df_full
    label_vocab = build_label_vocab(df_filt)
    num_classes = len(label_vocab)

    active_queries = sorted(df_filt["query"].unique().tolist())
    filter_desc = (
        cfg["query_filter"]
        or (f"category='{cfg['category_filter']}' ({len(active_queries)} queries)")
        if cfg["category_filter"] is not None
        else "(all queries)"
    )
    print(f"Query filter: {filter_desc}")
    print(f"Label vocab : {num_classes} classes")
    if cfg["category_filter"] is not None:
        for q in active_queries:
            print(f"  · {q}")

    # Persist vocab
    ckpt_dir  = Path(cfg["checkpoint_dir"]) / ("patches" if patch_control else f"slots_{n_slots}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = ckpt_dir / "label_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(label_vocab, f, indent=2)
    print(f"Label vocab → {vocab_path}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    ds_kwargs = dict(
        csv_path        = cfg["csv_path"],
        label_vocab     = label_vocab,
        query_filter    = cfg["query_filter"],
        category_filter = cfg["category_filter"],
    )

    tokenizer = None
    if use_cache:
        train_ds = CUBCachedFeatDataset(
            **ds_kwargs,
            dino_cache_path = cfg["dino_cache"],
            text_cache_path = cfg["text_cache"],
            split = "train",
        )
        val_ds = CUBCachedFeatDataset(
            **ds_kwargs,
            dino_cache_path = cfg["dino_cache"],
            text_cache_path = cfg["text_cache"],
            split = "test",
        )
    else:
        tokenizer = RobertaTokenizer.from_pretrained(cfg["roberta_model"])
        train_ds = CUBAttributeDataset(
            **ds_kwargs,
            image_root   = cfg["image_root"],
            split        = "train",
            tokenizer    = tokenizer,
            max_text_len = cfg["max_text_len"],
        )
        val_ds = CUBAttributeDataset(
            **ds_kwargs,
            image_root   = cfg["image_root"],
            split        = "test",
            tokenizer    = tokenizer,
            max_text_len = cfg["max_text_len"],
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Batch: {cfg['batch_size']}")

    # ── Build model ──────────────────────────────────────────────────────────
    from classifier_model import SlotClassifier, PatchClassifier

    if patch_control:
        print("Loading PatchClassifier (raw ViT patches, control) …")
        model = PatchClassifier(
            dinosaur_cfg_name  = cfg["dinosaur_cfg_name"],
            dinosaur_ckpt_path = cfg["dinosaur_ckpt"],
            num_classes        = num_classes,
            d_vit              = cfg["d_vit"],
            d_slot             = cfg["d_slot"],
            d_text             = cfg["d_text"],
            num_heads          = cfg["num_heads"],
            roberta_model      = cfg["roberta_model"],
            load_text_encoder  = not use_cache,
        ).to(device)
    else:
        print(f"Loading SlotClassifier (n_slots={n_slots}) …")
        model = SlotClassifier(
            dinosaur_cfg_name  = cfg["dinosaur_cfg_name"],
            dinosaur_ckpt_path = cfg["dinosaur_ckpt"],
            num_classes        = num_classes,
            n_slots            = n_slots,
            d_slot             = cfg["d_slot"],
            d_text             = cfg["d_text"],
            num_heads          = cfg["num_heads"],
            roberta_model      = cfg["roberta_model"],
            load_text_encoder  = not use_cache,
            finetune_ckpt_path = cfg["finetune_ckpt_path"],
        ).to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"Trainable params: {n_trainable:,}")

    # ── Text-feature accessor for checkpoint visualisation ───────────────────
    # Returns (text_hidden (1,L,d_text), attn_mask (1,L)) on `device`.
    if use_cache:
        _text_hidden_cache = val_ds.text_hidden   # dict {query: (L, d_text)}
        _attn_mask_cache   = val_ds.attn_masks    # dict {query: (L,)}
        def get_text_feats(query: str):
            hidden = _text_hidden_cache[query].unsqueeze(0).to(device)
            mask   = _attn_mask_cache[query].unsqueeze(0).to(device)
            return hidden, mask
    else:
        def get_text_feats(query: str):
            enc = tokenizer(
                query,
                max_length     = cfg["max_text_len"],
                padding        = "max_length",
                truncation     = True,
                return_tensors = "pt",
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                hidden = model._extract_text_features(input_ids, attn_mask)
            return hidden, attn_mask

    # ── Optimiser & LR schedule ──────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    total_steps = cfg["max_epochs"] * len(train_loader)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, cfg["warmup_steps"], total_steps
    )
    criterion   = nn.CrossEntropyLoss()

    # ── Training loop with early stopping ───────────────────────────────────
    best_val_acc      = 0.0
    epochs_no_improve = 0
    metrics_fh, metrics_writer = open_metrics_writer(ckpt_dir)

    ckpt_every = cfg["checkpoint_every"]

    # Per-query stats CSV (appended every checkpoint_every epochs)
    pq_csv_path = ckpt_dir / "per_query_stats.csv"
    pq_csv_fh   = open(pq_csv_path, "w", newline="")
    pq_writer   = csv.writer(pq_csv_fh)
    pq_writer.writerow(["epoch", "query", "val_acc", "n_samples"])

    # Val DataFrame for visualisation (test split only)
    val_df_for_viz = df_filt[df_filt["split"] == "test"].copy()

    try:
        for epoch in range(1, cfg["max_epochs"] + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, device, use_cache
            )
            val_loss, val_acc = eval_epoch(
                model, val_loader, criterion, device, use_cache
            )

            is_best = val_acc > best_val_acc + cfg["min_delta"]
            if is_best:
                best_val_acc      = val_acc
                epochs_no_improve = 0
                save_best_checkpoint(
                    model, epoch, val_acc, label_vocab, cfg, ckpt_dir,
                    cfg["query_filter"]
                )
            else:
                epochs_no_improve += 1

            marker = " ↑ best" if is_best else f" (no improve {epochs_no_improve}/{cfg['patience']})"
            print(
                f"Epoch {epoch:3d}/{cfg['max_epochs']} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val   loss={val_loss:.4f} acc={val_acc:.4f}{marker}"
            )
            metrics_writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", int(is_best),
            ])
            metrics_fh.flush()

            # ── Checkpoint: per-query stats + visualisation ──────────────
            if ckpt_every > 0 and epoch % ckpt_every == 0:
                print(f"\n  [checkpoint epoch {epoch}] per-query evaluation …")
                pq_results = eval_per_query(
                    model, active_queries, cfg, label_vocab, device, use_cache,
                    tokenizer=tokenizer,
                )
                for q, (acc, n) in sorted(pq_results.items()):
                    print(f"    {q:<55s}  acc={acc:.4f}  n={n}")
                    pq_writer.writerow([epoch, q, f"{acc:.6f}", n])
                pq_csv_fh.flush()

                if not patch_control:
                    print(f"  [checkpoint epoch {epoch}] generating visualisation …")
                    viz_path = make_checkpoint_viz(
                        model       = model,
                        val_df      = val_df_for_viz,
                        cfg         = cfg,
                        label_vocab = label_vocab,
                        device      = device,
                        epoch       = epoch,
                        out_dir     = ckpt_dir,
                        get_text_feats = get_text_feats,
                        n_samples   = cfg["viz_n_samples"],
                        top_k_slots = cfg["viz_top_k_slots"],
                    )
                    print(f"  → {viz_path}\n")

            if epochs_no_improve >= cfg["patience"]:
                print(f"\nEarly stopping: no improvement for {cfg['patience']} epochs.")
                break
    finally:
        metrics_fh.close()
        pq_csv_fh.close()

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}  (saved to {ckpt_dir})")


if __name__ == "__main__":
    main()
