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
``runs/cub_classifier_checkpoints/slots_{n_slots}/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer

# ── Hyperparameters ──────────────────────────────────────────────────────────

CONFIG = {
    # --- Data ---
    "dataset":          "cub",   # "cub", "superclevr3d", or "ade20k" (switch via --dataset)
    "csv_path":         "FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv",
    "image_root":       "FG-datset/CUB_200_2011/images",
    "dino_cache":       "FG-datset/CUB_200_2011/dino_feat_cache.pt",
    "text_cache":       "FG-datset/CUB_200_2011/text_feat_cache.pt",
    "label_vocab_path": "label_vocab.json",
    "query_filter":     None,
    "category_filter":  None,   # substring match, e.g. "color" → all color queries
    "attribute_filter": None,   # superclevr3d only: exact match on attribute_type column
    "depth_filter":     None,   # superclevr3d only: exact match on depth column
    "max_n_objects":    None,   # superclevr3d only: keep rows with n_objects <= this
    "object_category":  None,   # superclevr3d only: whole-word match on query, e.g. "car"
    "text_onehot":      False,  # diagnostic: replace RoBERTa+TextProjector with nn.Embedding(n_q, d_slot)
    "text_in_memory":   False,  # compute text features in RAM at run start (no on-disk text cache; avoids quota/IO)
    "dino_in_memory":   False,  # compute ViT/DINO image features in RAM at run start (no on-disk dino cache; avoids quota/IO)

    # --- Model ---
    "slot_backend":      "oclf",   # "oclf" (Lightning .ckpt via Hydra) or "ftdinosaur"
    "ftdinosaur_model":  "dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k",
    "dinosaur_cfg_name": "projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto_dinov3",
    "dinosaur_ckpt":     None,
    "finetune_ckpt_path": None,   # path to runs/cub_slot_finetune/best_checkpoint.pt
    "roberta_model":     "roberta-large",
    "text_encoder":      "roberta",   # 'roberta' (RoBERTa-Large, 1024-d) or 't5' (T5-base encoder, 768-d, paper-faithful)
    "t5_model":          "t5-base",
    "vqa_d_model":       128,         # transformer working dim for the 'vqa_paper' pooler (paper uses 128)
    "patch_control":     False,  # use PatchClassifier instead of SlotClassifier
    "n_slots":           7,
    "d_vit":             384,    # ViT-S/16 feature dim (PatchClassifier only)
    "d_slot":            256,
    "d_text":            1024,   # text-encoder hidden size; auto-set to 768 when text_encoder='t5'
    "num_heads":         8,
    "max_text_len":      64,
    "pooler":            "gated_attn",  # 'gated_attn' (default, single-hop) or 'transformer' (multi-hop fusion)
    "pooler_layers":     2,             # only used when pooler == 'transformer'
    "pooler_dropout":    0.0,           # only used when pooler == 'transformer'
    "recursive_infer":   False,         # zoom-in: rank slots via the transformer pooler, refine the top slot(s) into child slots, re-classify on them (needs a transformer-family pooler)
    "recursive_children": 4,            # number of child slots each refined parent slot is split into (recursive_infer only)
    "recursive_parents": 1,             # number of top-ranked parent slots to refine independently; their children are pooled for the answer (recursive_infer only)
    "recursive_spread":  0.5,           # strength of the child spatial-spread prior: 0=pure SA (may collapse), ~0.5=gentle/liberal, 1=maximally-separated cells (recursive_infer only)
    "recursive_include_parents": True,  # second pass sees the refined parent slot(s) alongside their children (recursive_infer only)
    "router_temp":          1.0,        # hier_router: softmax temperature on P(j|y) and P(k|j,x)
    "router_entropy_weight": 0.0,       # hier_router: entropy regulariser weight on P(j|y) (0 = off)
    "child_scorer":         "bilinear", # hier_router: child-routing scorer ("bilinear" | "mlp")
    "router_parent_only":   False,      # hier_router ABLATION: drop the child level (P(a)=Σ_j P(j|y)·P(a|s_j), <x> unused) to test the utility of children
    "router_readout_query": True,       # hier_router: condition the colour readout on f_readout("<y> <x>") e.g. "car door" (False = P(colour|slot) only)
    "router_color_source":  "slot",     # hier_router: colour evidence — "slot" (routed child slot vector) or "patch" (raw DINO patches the child grounds to, pooled by its attention)
    "router_init_ckpt":     None,        # hier_router: warm-start routing (text_projector + q_parent/q_child/child_mlp) from this best_model.pt; colour head stays fresh
    "router_freeze_routing": False,      # hier_router: freeze routing path + text projector so only the colour head trains (use with router_init_ckpt)
    "router_init_color_head": False,     # hier_router LP-FT: also warm-start the colour head from router_init_ckpt (not just routing), then finetune
    "router_qdot_project_patches": True, # hier_router patch_qdot: project patches d_vit->d_slot before the per-child dot-product (False = raw d_vit)
    "router_qdot_dropout":  0.0,         # hier_router patch_qdot: dropout on the qdot colour-head readout feature (finetune regulariser)
    "rank_method":       "attention",   # recursive slot ranking: 'attention' (CLS->slot attn, transformer poolers only) or 'attribution' (gradient saliency ||dy/ds_i||, any pooler)
    "patch_qdot_project_patches": False, # patch_qdot: project patches d_vit->d_slot before the dot-product (Version A); default raw d_vit (Version B)
    "patch_qdot_strip_registers": True,  # patch_qdot: drop the 4 DINOv3 register tokens → attend over the 196 spatial patches (14x14)
    "patch_qdot_temperature":     1.0,   # patch_qdot: softmax temperature on the query·patch scores
    "patch_qdot_normalize":       False, # patch_qdot: L2-normalise query+patches (cosine) before the dot-product
    "zero_image_feats":  False,         # diagnostic: zero slots/patches before pooler (text-only floor)
    "label_smoothing":   0.0,           # cross-entropy label smoothing (0 = off)
    "augment":           False,         # train-time image augmentation (RandomResizedCrop); only effective when feat_cache is off
    "img_size":          224,           # input resolution; must match the resolution the slot ckpt was trained at
    "resize_mode":       "crop",        # image preprocessing: crop (Resize+CenterCrop, aspect-preserving) | square (Resize((S,S)), no crop, distorts aspect) | pad (pad-to-square then resize)
    "overfit_n":         0,             # diagnostic: train AND eval on the first N train samples (0 = off). Both loops see the exact same data.
    "train_frac":        1.0,           # quick check: train on a random fraction of the train split (1.0 = full). Val/test stay full.

    # --- Optimisation ---
    "optimizer":     "adamw",   # 'adamw' (repo default) or 'adam' (paper-faithful; no weight decay)
    "lr_schedule":   "cosine",  # 'cosine' (warmup→cosine decay) or 'constant' (paper-faithful; optional warmup then flat)
    "max_steps":     None,      # optional step budget; overrides max_epochs (max_epochs = ceil(max_steps/steps_per_epoch))
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
    "checkpoint_dir": "runs/cub_classifier_checkpoints_100_slots",
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "feat_cache":     False,   # set True via --feat_cache flag
}


# Per-dataset overrides applied on top of CONFIG when --dataset is passed.
DATASET_CONFIGS = {
    "cub": {},
    "superclevr3d": {
        "csv_path":         "FG-datset/superclevr3d/parts_vqa.csv",
        "image_root":       "FG-datset/superclevr3d/images",
        "dino_cache":       "FG-datset/superclevr3d/dino_feat_cache.pt",
        "text_cache":       "FG-datset/superclevr3d/text_feat_cache.pt",
        "label_vocab_path": "label_vocab_superclevr3d_parts.json",
        "dinosaur_cfg_name": "projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3",
        "n_slots":          12,
        "checkpoint_dir":   "runs/superclevr3d_classifier_checkpoints",
        # Per-query eval at each checkpoint is auto-skipped when there are
        # >200 unique queries (sc3d has ~70k); viz still runs at every
        # checkpoint_every epochs.
    },
    # ADE20K part-color VQA. Same CSV schema and {train, val} splits as
    # superclevr3d, so it reuses the superclevr3d_dataset module (see the
    # dataset-module dispatch in main()). image_name is relative to the
    # ADE20K_2021_17_01/images root (e.g. "ADE/training/.../ADE_train_*.jpg").
    # Natural images → default to the COCO-trained plain-DINO backbone; override
    # with --dinosaur_cfg / --dinosaur_ckpt / --n_slots as needed.
    "ade20k": {
        "csv_path":         "FG-datset/ade20k/parts_color_vqa.csv",
        "image_root":       "FG-datset/ade20k/ADE20K_2021_17_01/images",
        "dino_cache":       "FG-datset/ade20k/dino_feat_cache.pt",
        "text_cache":       "FG-datset/ade20k/text_feat_cache.pt",
        "label_vocab_path": "label_vocab_ade20k_parts_color.json",
        "dinosaur_cfg_name": "projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto",
        "n_slots":          11,
        "checkpoint_dir":   "runs/ade20k_classifier_checkpoints",
        # ~1.1k unique queries → per-query eval auto-skipped (>200); viz still runs.
    },
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


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Constant LR after an optional linear warmup (paper-faithful downstream schedule).

    With ``warmup_steps=0`` this is a flat constant learning rate for the whole
    run — matching the paper's downstream VQA training (lr 1e-4, no decay).
    """
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train / eval loops ───────────────────────────────────────────────────────

def _forward_logits(model, x1, x2, attention_mask, use_cache: bool, recursive: bool, spans=None):
    """Dispatch to the right forward method given cache/recursive flags.

    ``spans`` (B, 4, d_text), when present, carries the part <x> / object <y> / combined phrase / readout noun
    vectors consumed by the hier_router head; ignored by every other forward.
    """
    if recursive:
        if use_cache:
            return model.forward_recursive_cached(x1, x2, attention_mask, spans=spans)
        return model.forward_recursive(x1, x2, attention_mask, spans=spans)
    if use_cache:
        # spans are ignored by every cached head except qca (reads ch3 "<y> <x>").
        return model.forward_cached(x1, x2, attention_mask, spans=spans)
    return model(x1, x2, attention_mask)


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    use_cache: bool,
    desc: str = "train",
    recursive: bool = False,
) -> tuple[float, float]:
    model.train()
    for _attr in ("dino_feature_extractor", "dino_conditioning", "dino_perceptual_grouping", "ftdinosaur"):
        _m = getattr(model, _attr, None)
        if _m is not None:
            _m.eval()
    if model.text_encoder is not None:
        model.text_encoder.eval()

    total_loss, correct, total = 0.0, 0, 0

    # mininterval=30s keeps SLURM logs small (one progress line per ~30s) while
    # still proving the epoch is making progress.
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False,
                dynamic_ncols=True, mininterval=30.0)
    for batch in pbar:
        batch = [t.to(device) for t in batch]
        x1, x2, attention_mask, labels = batch[:4]
        spans = batch[4] if len(batch) > 4 else None   # hier_router span vectors

        optimizer.zero_grad()
        logits = _forward_logits(model, x1, x2, attention_mask, use_cache, recursive, spans=spans)

        loss = criterion(logits, labels)
        # hier_router (and any future head) may stash an auxiliary loss term
        # (e.g. an entropy regulariser) for the trainer to add.
        aux = getattr(model, "_aux_loss", None)
        if aux is not None:
            loss = loss + aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs
        pbar.set_postfix(loss=f"{total_loss/total:.3f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_cache: bool,
    desc: str = "val",
    recursive: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, unit="batch", leave=False,
                dynamic_ncols=True, mininterval=30.0)
    for batch in pbar:
        batch = [t.to(device) for t in batch]
        x1, x2, attention_mask, labels = batch[:4]
        spans = batch[4] if len(batch) > 4 else None   # hier_router span vectors

        logits = _forward_logits(model, x1, x2, attention_mask, use_cache, recursive, spans=spans)

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


def _collect_trainable_state(model):
    """Dump every trainable sub-module's state_dict in one go.

    Shared by the best- and last-checkpoint savers and the resume loader so
    they stay in lockstep — add a new trainable module here and both paths
    cover it automatically.
    """
    trainable_state = {
        "text_projector":   model.text_projector.state_dict(),
        "classifier_head":  model.classifier_head.state_dict(),
    }
    if hasattr(model, "gated_cross_attn"):
        trainable_state["gated_cross_attn"] = model.gated_cross_attn.state_dict()
    if hasattr(model, "fusion_pooler"):
        trainable_state["fusion_pooler"] = model.fusion_pooler.state_dict()
    if hasattr(model, "vqa_pooler"):
        trainable_state["vqa_pooler"] = model.vqa_pooler.state_dict()
    if hasattr(model, "vqa_paper_pooler"):
        trainable_state["vqa_paper_pooler"] = model.vqa_paper_pooler.state_dict()
    if hasattr(model, "patch_projector"):
        trainable_state["patch_projector"] = model.patch_projector.state_dict()
    if hasattr(model, "hier_router"):
        trainable_state["hier_router"] = model.hier_router.state_dict()
    if hasattr(model, "qca_head"):
        trainable_state["qca_head"] = model.qca_head.state_dict()
    if hasattr(model, "patch_qdot_head"):
        trainable_state["patch_qdot_head"] = model.patch_qdot_head.state_dict()
    return trainable_state


def _load_trainable_state(model, trainable):
    """Inverse of ``_collect_trainable_state``: restore weights in place."""
    model.text_projector.load_state_dict(trainable["text_projector"])
    model.classifier_head.load_state_dict(trainable["classifier_head"])
    if "gated_cross_attn" in trainable and hasattr(model, "gated_cross_attn"):
        model.gated_cross_attn.load_state_dict(trainable["gated_cross_attn"])
    if "fusion_pooler" in trainable and hasattr(model, "fusion_pooler"):
        model.fusion_pooler.load_state_dict(trainable["fusion_pooler"])
    if "vqa_pooler" in trainable and hasattr(model, "vqa_pooler"):
        model.vqa_pooler.load_state_dict(trainable["vqa_pooler"])
    if "vqa_paper_pooler" in trainable and hasattr(model, "vqa_paper_pooler"):
        model.vqa_paper_pooler.load_state_dict(trainable["vqa_paper_pooler"])
    if "patch_projector" in trainable and hasattr(model, "patch_projector"):
        model.patch_projector.load_state_dict(trainable["patch_projector"])
    if "hier_router" in trainable and hasattr(model, "hier_router"):
        model.hier_router.load_state_dict(trainable["hier_router"])
    if "qca_head" in trainable and hasattr(model, "qca_head"):
        model.qca_head.load_state_dict(trainable["qca_head"])
    if "patch_qdot_head" in trainable and hasattr(model, "patch_qdot_head"):
        model.patch_qdot_head.load_state_dict(trainable["patch_qdot_head"])


def load_pretrained_router(model, ckpt_path, load_color_head=False, device="cpu"):
    """Warm-start the hier_router from a previously trained checkpoint.

    Always loads the text projector and the routing params (q_parent / q_child /
    child_mlp). By default the colour-readout subsystem (``f_readout`` / ``color_head``
    / ``qdot_readout``) is SKIPPED so a fresh colour head can be trained on top.

    With ``load_color_head=True`` (the LP-FT recipe) it ALSO loads every colour-head
    tensor whose shape matches the current model — e.g. init the trained qdot head from
    the frozen linear-probe run, then finetune everything. Shape-mismatched keys are
    silently skipped, so pointing at a checkpoint with a different colour head (or a
    different ``color_source``) just loads whatever lines up. Returns
    ``(routing_keys, head_keys)`` actually loaded.
    """
    if not hasattr(model, "hier_router"):
        raise RuntimeError("--router_init_ckpt requires pooler='hier_router'")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    src = ck["trainable_state"]
    if "text_projector" in src:
        model.text_projector.load_state_dict(src["text_projector"])
    src_hr = src.get("hier_router", {})
    dst_hr = model.hier_router.state_dict()
    routing_pref = ("q_parent.", "q_child.", "child_mlp.")
    routing_keys, head_keys = [], []
    filt = {}
    for k, v in src_hr.items():
        is_routing = k.startswith(routing_pref)
        if not (is_routing or load_color_head):
            continue
        if k in dst_hr and dst_hr[k].shape == v.shape:  # shape-filter the head keys
            filt[k] = v
            (routing_keys if is_routing else head_keys).append(k)
    # strict=False: load only the collected keys, leave the rest at current init.
    model.hier_router.load_state_dict(filt, strict=False)
    return sorted(routing_keys), sorted(head_keys)


def save_best_checkpoint(model, epoch, val_acc, label_vocab, cfg, out_dir, query):
    fname = "best_model.pt"
    path  = out_dir / fname
    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "label_vocab": label_vocab,
            "config": cfg,
            "trainable_state": _collect_trainable_state(model),
        },
        path,
    )
    return path


def save_last_checkpoint(
    model, epoch, optimizer, scheduler, best_val_acc, epochs_no_improve,
    label_vocab, cfg, out_dir,
):
    """Resumable checkpoint: trainable weights + optimizer + scheduler + epoch.

    Saved every epoch so chained chunks (e.g. SLURM array %1) can pick up the
    schedule mid-stream after a wall-time SIGTERM. Atomic via tmp-then-rename
    so a kill mid-write can't corrupt the file.
    """
    path     = out_dir / "last.pt"
    tmp_path = out_dir / "last.pt.tmp"
    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "epochs_no_improve": epochs_no_improve,
            "label_vocab": label_vocab,
            "config": cfg,
            "trainable_state":  _collect_trainable_state(model),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)
    return path


def load_last_checkpoint(path, model, optimizer, scheduler, device):
    """Restore (model, optimizer, scheduler) and return resume state.

    Returns ``(start_epoch, best_val_acc, epochs_no_improve)``: caller starts
    the loop from ``start_epoch + 1``.
    """
    state = torch.load(path, map_location=device)
    _load_trainable_state(model, state["trainable_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    return (
        int(state["epoch"]),
        float(state["best_val_acc"]),
        int(state["epochs_no_improve"]),
    )


def open_metrics_writer(out_dir: Path, append: bool = False):
    """Open a CSV for per-epoch metrics; return (file_handle, csv.writer).

    With ``append=True`` (resume mode), open in 'a' and skip the header so we
    don't double-write it.
    """
    path = out_dir / "metrics.csv"
    mode = "a" if append and path.exists() else "w"
    fh   = open(path, mode, newline="")
    w    = csv.writer(fh)
    if mode == "w":
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
    text_cache: dict | None = None,
    dino_cache: dict | None = None,
) -> dict[str, tuple[float, int]]:
    """Evaluate accuracy separately for each sub-query.

    Args:
        queries:    List of unique query strings to evaluate.
        tokenizer:  Required when ``use_cache=False``.
        text_cache: Pre-built in-memory text cache (``--text_in_memory``); when
                    given, the cached datasets reuse it instead of loading the
                    on-disk text cache (which may not exist).
        dino_cache: Pre-built in-memory DINO cache (``--dino_in_memory``); same
                    semantics for the image-feature cache.

    Returns:
        Dict mapping query string → (accuracy, n_samples).
    """
    if cfg["dataset"] == "cub":
        from cub_dataset import (
            CUBAttributeDataset  as AttributeDataset,
            CUBCachedFeatDataset as CachedFeatDataset,
        )
    else:
        from superclevr3d_dataset import (
            SuperCLEVR3DAttributeDataset  as AttributeDataset,
            SuperCLEVR3DCachedFeatDataset as CachedFeatDataset,
        )

    criterion = nn.CrossEntropyLoss()
    results   = {}

    for query in sorted(queries):
        ds_kwargs = dict(
            csv_path     = cfg["csv_path"],
            label_vocab  = label_vocab,
            query_filter = query,
        )
        if use_cache:
            _mem_kwargs = {}
            if text_cache is not None:
                _mem_kwargs["text_cache"] = text_cache
            if dino_cache is not None:
                _mem_kwargs["dino_cache"] = dino_cache
            ds = CachedFeatDataset(
                **ds_kwargs,
                dino_cache_path = cfg["dino_cache"],
                text_cache_path = cfg["text_cache"],
                split           = "test",
                return_spans    = (cfg.get("pooler") in ("hier_router", "qca", "patch_qdot")),
                **_mem_kwargs,
            )
        else:
            ds = AttributeDataset(
                **ds_kwargs,
                image_root   = cfg["image_root"],
                split        = "test",
                tokenizer    = tokenizer,
                max_text_len = cfg["max_text_len"],
            )

        if len(ds) == 0:
            continue

        loader  = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        _, acc  = eval_epoch(model, loader, criterion, device, use_cache,
                             recursive=cfg.get("recursive_infer", False))
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
    img_size   = int(cfg.get("img_size", 224))

    if cfg.get("slot_backend") == "ftdinosaur":
        # Match the encoder's training preprocessing (square resize, ImageNet
        # norm). denorm below uses the same ImageNet mean/std, so the displayed
        # image still reconstructs correctly.
        from ftdinosaur_inference import build_dinosaur as _bd
        tf = _bd.build_preprocessing(cfg["ftdinosaur_model"])
    else:
        from precompute_features import build_image_transform
        tf = build_image_transform(img_size, cfg.get("resize_mode", "crop"))
    mean_t = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGE_STD).view(3, 1, 1)

    def denorm(t: torch.Tensor) -> np.ndarray:
        return (t.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

    # ── Sample rows ──────────────────────────────────────────────────────────
    # ``val_df`` is already filtered to the appropriate split (test for cub,
    # val for superclevr3d) by the caller; skip the redundant filter that
    # silently empties the dataframe on superclevr3d.
    rng = np.random.RandomState(epoch)
    unique_queries = val_df["query"].unique().tolist()
    n_queries      = len(unique_queries)

    if n_queries > 5 * n_samples:
        # Too many distinct queries to stratify — just take a flat random sample.
        sample_df = val_df.sample(n=min(n_samples, len(val_df)),
                                  random_state=rng).reset_index(drop=True)
    else:
        sample_rows = []
        per_q = max(1, n_samples // n_queries) if n_queries else 1
        for q in sorted(unique_queries):
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
    rank_method = cfg.get("rank_method", "attention")  # slot-importance source (also used in suptitle)

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
            logits, slot_masks_raw, cross_attn_w, slot_ca_norms, slot_attr = \
                model.forward_with_viz(img_tensor, text_hidden, attn_mask)

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
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )[0].cpu().numpy()  # (N_slots, img_size, img_size)

        # Per-slot importance drives both the slot ordering and which slot is
        # flagged as "top". Two sources, selected by cfg["rank_method"]:
        #  - "attribution": gradient saliency ||∂y_c/∂s_i|| (Simonyan 2013 / AwGA) —
        #    how much the predicted-answer logit depends on each slot.
        #  - "attention":   L2 norm of the cross-attention output per slot, i.e. how
        #    much each slot was moved by the text query. (Summing attention weights
        #    would give ≈1 for every slot since they are normalised over text tokens.)
        importance_src = slot_attr if rank_method == "attribution" else slot_ca_norms
        slot_importance = importance_src[0].cpu().numpy()  # (N_slots,)
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
        # Wrap the question instead of hard-truncating — sc3d queries can run
        # past 80 chars at depth=3/4 ("What is the colour of the door of the …").
        wrapped_q = textwrap.fill(query, width=42)
        ax_row[1].set_title(wrapped_q, fontsize=7)
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
        # Panels are sorted descending by importance, so k == 0 is the
        # highest-attribution slot — flag it with a gold frame + ★ in the title.
        for k, si in enumerate(top_k_idx):
            scaled_alpha = (masks[si] * N_slots).clip(0, 1)  # (224, 224)
            masked_img   = img_vis * scaled_alpha[:, :, None]  # (224, 224, 3)
            ax = ax_row[2 + k]
            ax.imshow(masked_img.clip(0, 1))
            is_top = (k == 0)
            ax.set_title(
                f"{'★ ' if is_top else ''}slot {si}  ({slot_importance[si]:.1%})",
                fontsize=8,
                color=("darkgoldenrod" if is_top else "black"),
                fontweight=("bold" if is_top else "normal"),
            )
            if is_top:
                # Keep the frame (gold) but drop ticks — axis("off") hides spines.
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("gold")
                    spine.set_linewidth(4)
            else:
                ax.axis("off")

    _imp_name = "∂y/∂s attribution" if rank_method == "attribution" else "cross-attn norm"
    plt.suptitle(
        f"Epoch {epoch} — checkpoint visualisation  "
        f"(slot importance: {_imp_name}; ★ gold = highest)",
        fontsize=9,
    )
    plt.tight_layout()

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    viz_path = viz_dir / f"epoch_{epoch:04d}.png"
    plt.savefig(viz_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return viz_path


def make_hier_router_viz(
    model,
    val_df: "pd.DataFrame",
    cfg: dict,
    label_vocab: dict,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    get_text_feats,          # callable(query) → (text_hidden (1,L,d), attn_mask (1,L))
    get_spans=None,          # callable(query) → spans (1, 4, d_text)
    n_samples: int = 4,
    **_ignored,              # accept (and ignore) top_k_slots etc. for a uniform call site
) -> Path:
    """Per-checkpoint visualisation of the hier_router's parent→child traversal.

    Draws the structured routing — P(j|y), P(k|j,x), path weights w_jk, the per-child
    colour P(a|c_jk) and the marginal P(a) — for a FIXED set of val samples each
    checkpoint (seed 0, not epoch-seeded), so the routing can be watched sharpening
    over training. Panel layout: see visualize_hier_routing.py::draw_sample.
    """
    import visualize_hier_routing as vhr
    from precompute_features import parse_xy_for
    from PIL import Image as PILImage

    _ds = cfg["dataset"]   # dataset-aware span parsing (CUB vs ADE/SC3D templates)

    if get_spans is None:
        print("  [hier_router viz] no span accessor (cached span path required); skipping.")
        return out_dir

    classes = [None] * len(label_vocab)
    for name, idx in label_vocab.items():
        classes[idx] = name
    tf, img_size = vhr.make_transform(cfg)

    # Fixed sample set across checkpoints → watch the same images' routing evolve.
    df = val_df[val_df["query"].apply(lambda q: parse_xy_for(q, _ds) is not None)]
    if len(df) == 0:
        print("  [hier_router viz] no template-matching val queries; skipping.")
        return out_dir
    rng     = np.random.RandomState(0)
    queries = sorted(df["query"].unique().tolist())
    rows, per_q = [], max(1, n_samples // max(len(queries), 1))
    for q in queries:
        qr = df[df["query"] == q]
        rows.append(qr.sample(n=min(per_q, len(qr)), random_state=rng))
    sample_df = (pd.concat(rows).sample(frac=1.0, random_state=rng)
                 .head(n_samples).reset_index(drop=True))

    was_training = model.training
    model.eval()
    last_path = out_dir
    for i, row in sample_df.iterrows():
        q   = row["query"]
        pil = PILImage.open(os.path.join(cfg["image_root"], row["image_name"])).convert("RGB")
        img_t   = tf(pil).unsqueeze(0).to(device)
        img_vis = vhr.denorm(img_t[0])
        text_hidden, attn_mask = get_text_feats(q)
        spans = get_spans(q)
        with torch.no_grad():
            out = model.forward_hier_router_viz(img_t, text_hidden, attn_mask, spans)
        x_phrase, y_phrase = parse_xy_for(q, _ds)
        save_path = Path(out_dir) / f"hier_routing_s{i:02d}_e{epoch:03d}.png"
        vhr.draw_sample(out, img_vis, img_size, classes, row,
                        x_phrase, y_phrase, True, save_path)
        last_path = save_path
    if was_training:
        model.train()
    return last_path


def make_recursive_tree_viz(
    model,
    val_df: "pd.DataFrame",
    cfg: dict,
    label_vocab: dict,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    get_text_feats,          # callable(query) → (text_hidden (1,L,d), attn_mask (1,L))
    n_samples: int = 4,
    **_ignored,              # accept (and ignore) top_k_slots etc. for a uniform call site
) -> Path:
    """Tree visualisation for ``--recursive_infer``.

    One row per sample: the original image, then a two-level tree — the top
    slot chosen by the first pass (root), and the child slots it was refined
    into (leaves), drawn left→right in descending order of their second-pass
    CLS→child importance. Edges connect the root to each child.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    from PIL import Image as PILImage
    from torchvision import transforms

    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD  = [0.229, 0.224, 0.225]
    img_size   = int(cfg.get("img_size", 224))

    if cfg.get("slot_backend") == "ftdinosaur":
        from ftdinosaur_inference import build_dinosaur as _bd
        tf = _bd.build_preprocessing(cfg["ftdinosaur_model"])
    else:
        from precompute_features import build_image_transform
        tf = build_image_transform(img_size, cfg.get("resize_mode", "crop"))
    mean_t = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGE_STD).view(3, 1, 1)

    def denorm(t: torch.Tensor) -> np.ndarray:
        return (t.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

    def mask_to_alpha(mask_1d: torch.Tensor) -> np.ndarray:
        """Patch mask (N,) → (img_size, img_size) alpha in [0, 1].

        Max-normalised so the mask's strongest patch maps to 1 — this keeps the
        highlighted region visible regardless of the mask's absolute scale (soft
        slot/child masks can be small after confining to the parent region)."""
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

    DIM = 0.30  # min brightness for non-region pixels, so context stays visible

    def overlay(alpha_np: np.ndarray) -> np.ndarray:
        return (img_vis * (DIM + (1.0 - DIM) * alpha_np[:, :, None])).clip(0, 1)

    inv_vocab = {v: k for k, v in label_vocab.items()}
    K = int(cfg["recursive_children"])
    P = max(1, int(cfg.get("recursive_parents", 1)))

    # ── Sample rows (same stratified sampling as make_checkpoint_viz) ─────────
    rng = np.random.RandomState(epoch)
    unique_queries = val_df["query"].unique().tolist()
    n_queries      = len(unique_queries)
    if n_queries > 5 * n_samples:
        sample_df = val_df.sample(n=min(n_samples, len(val_df)),
                                  random_state=rng).reset_index(drop=True)
    else:
        rows = []
        per_q = max(1, n_samples // n_queries) if n_queries else 1
        for q in sorted(unique_queries):
            q_rows = val_df[val_df["query"] == q]
            rows.append(q_rows.sample(n=min(per_q, len(q_rows)), random_state=rng))
        sample_df = pd.concat(rows).head(n_samples).reset_index(drop=True)

    n_actual = len(sample_df)
    fig_w = max(8.0, 1.1 * P * K + 2.0)
    fig = plt.figure(figsize=(fig_w, 3.6 * n_actual))

    for i, row in sample_df.iterrows():
        pil_img    = PILImage.open(os.path.join(cfg["image_root"], row["image_name"])).convert("RGB")
        img_tensor = tf(pil_img).unsqueeze(0).to(device)
        img_vis    = denorm(img_tensor[0])
        query      = row["query"]
        text_hidden, attn_mask = get_text_feats(query)

        with torch.no_grad():
            out = model.forward_recursive_viz(img_tensor, text_hidden, attn_mask)

        pred_label = inv_vocab.get(out["logits"].argmax(1).item(), "?")
        true_label = row["label"]

        top_idx      = out["top_idx"][0].tolist()            # (P,)
        parent_masks = out["parent_masks"][0]                # (P, N)
        child_masks  = out["child_masks"][0]                 # (P, K, N)
        # Annotate with normalized gradient-saliency attribution (∂y/∂s), not the
        # pooler's attention: parent share over all slots, child share within a parent.
        child_imp    = out["child_attr"][0].float()          # (P, K)  attribution
        slot_attr    = out["slot_attr"][0].float()           # (N_slots,) attribution
        slot_share   = slot_attr / (slot_attr.sum() + 1e-8)
        P_actual     = len(top_idx)

        # ── Vertical band for this sample (reserve top 5% for the suptitle) ───
        usable_top = 0.95
        band_h   = usable_top / n_actual
        band_bot = usable_top - (i + 1) * band_h
        pad_v    = 0.10 * band_h
        node_h   = 0.30 * band_h

        # original image (far left, vertically centred)
        ax_img = fig.add_axes([0.005, band_bot + 0.22 * band_h, 0.12, 0.55 * band_h])
        ax_img.imshow(img_vis)
        ax_img.set_title(f"true: {true_label}\npred: {pred_label}", fontsize=7,
                         color=("green" if true_label == pred_label else "red"))
        ax_img.set_xlabel(textwrap.fill(query, width=24), fontsize=6)
        ax_img.set_xticks([]); ax_img.set_yticks([])

        # forest: the top-P parents split the remaining width into P blocks,
        # each a root (parent) over its K children (left→right by importance).
        fx0, fx1 = 0.17, 0.995
        block_w  = (fx1 - fx0) / P_actual
        for p in range(P_actual):
            bx0 = fx0 + p * block_w
            bcx = bx0 + 0.5 * block_w

            root_w  = min(0.16, block_w * 0.6)
            ax_root = fig.add_axes([bcx - root_w / 2, band_bot + band_h - pad_v - node_h,
                                    root_w, node_h])
            ax_root.imshow(overlay(mask_to_alpha(parent_masks[p])))
            ax_root.set_title(f"P{p}: slot {top_idx[p]} ({slot_share[top_idx[p]].item():.0%})",
                              fontsize=7)
            ax_root.axis("off")

            order   = torch.argsort(child_imp[p], descending=True).tolist()
            imp_sh  = child_imp[p] / (child_imp[p].sum() + 1e-8)
            inner_x0, inner_w = bx0 + 0.02 * block_w, block_w * 0.96
            child_w = min(0.13, (inner_w / K) * 0.85)
            for rank, ci in enumerate(order):
                cx   = inner_x0 + inner_w * ((rank + 0.5) / K)
                ax_c = fig.add_axes([cx - child_w / 2, band_bot + pad_v, child_w, node_h])
                # child masks are already confined to their parent by _confined_slot_attention.
                ax_c.imshow(overlay(mask_to_alpha(child_masks[p, ci])))
                ax_c.set_title(f"c{ci} ({imp_sh[ci].item():.0%})", fontsize=6)
                ax_c.axis("off")
                con = ConnectionPatch(
                    xyA=(0.5, 0.0), coordsA=ax_root.transAxes,
                    xyB=(0.5, 1.0), coordsB=ax_c.transAxes,
                    color="0.4", lw=1.0, alpha=0.6,
                )
                fig.add_artist(con)

    plt.suptitle(f"Epoch {epoch} — recursive zoom-in forest "
                 f"(top-{P} parent(s) → children L→R by attribution; "
                 f"% = normalized ∂y/∂s attribution)", fontsize=9)

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_path = viz_dir / f"epoch_{epoch:04d}_tree.png"
    plt.savefig(viz_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return viz_path


# ── Argparse ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train slot classifier (CUB or Super-CLEVR-3D)")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="cub",
                   help="Which CSV-driven dataset to train on")
    p.add_argument("--query_filter",    type=str,   default=None,
                   help="Exact-match filter on the query column")
    p.add_argument("--category_filter", type=str,   default=None,
                   help="Substring filter on the query column (e.g. 'color')")
    p.add_argument("--attribute_filter", type=str, default=None,
                   help="(superclevr3d only) Exact match on the attribute_type column")
    p.add_argument("--depth_filter",    type=int,   default=None,
                   help="(superclevr3d only) Exact match on the depth column")
    p.add_argument("--object_category", type=str,   default=None,
                   help="superclevr3d only: whole-word match on the query string "
                        "(e.g. 'car', 'motorbike'). Restricts to questions about "
                        "the given object category.")
    p.add_argument("--max_n_objects",   type=int,   default=None,
                   help="(superclevr3d only) Keep only rows whose n_objects "
                        "column is <= this value. Useful for restricting to "
                        "simpler scenes (n_objects ranges 3-10 in simple_color_vqa).")
    p.add_argument("--n_slots",         type=int,   default=None,
                   help="Number of DINOSAUR slots (default: 7)")
    p.add_argument("--max_epochs",      type=int,   default=None)
    p.add_argument("--patience",        type=int,   default=None,
                   help="Early-stopping patience in epochs (default: 10)")
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--warmup_steps",    type=int,   default=None)
    p.add_argument("--optimizer",       type=str,   default=None,
                   choices=["adamw", "adam"],
                   help="Optimizer. 'adamw' (repo default, decoupled weight decay) or "
                        "'adam' (paper-faithful; no weight decay).")
    p.add_argument("--lr_schedule",     type=str,   default=None,
                   choices=["cosine", "constant"],
                   help="LR schedule. 'cosine' (repo default: warmup→cosine decay to 0) or "
                        "'constant' (paper-faithful: optional warmup then flat lr).")
    p.add_argument("--max_steps",       type=int,   default=None,
                   help="Optimizer-step budget (e.g. 600000 to match the paper). Overrides "
                        "--max_epochs: max_epochs = ceil(max_steps / steps_per_epoch).")
    p.add_argument("--slot_backend",    type=str,   default=None,
                   choices=["oclf", "ftdinosaur"],
                   help="Slot encoder. 'oclf' (default) loads DINOSAUR sub-modules "
                        "from a Lightning .ckpt via Hydra. 'ftdinosaur' uses the "
                        "self-contained ftdinosaur_inference model (ViT-B/14 DINOv2, "
                        "slot_dim=256). ftdinosaur requires --feat_cache OFF and "
                        "is currently wired for --dataset cub.")
    p.add_argument("--ftdinosaur_model", type=str,  default=None,
                   help="ftdinosaur checkpoint name when --slot_backend ftdinosaur "
                        "(default: dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k).")
    p.add_argument("--dinosaur_ckpt",   type=str,   default=None)
    p.add_argument("--dinosaur_cfg",    type=str,   default=None,
                   help="Override Hydra config path (e.g. when reusing a COCO-trained checkpoint on sc3d)")
    p.add_argument("--finetune_ckpt",   type=str,   default=None,
                   help="Path to runs/cub_slot_finetune/best_checkpoint.pt from finetune_slots.py")
    p.add_argument("--checkpoint_dir",  type=str,   default=None)
    p.add_argument("--checkpoint_every",type=int,   default=None,
                   help="Epoch interval for per-query stats + viz (0 = off, default: 50)")
    p.add_argument("--viz_n_samples",   type=int,   default=None,
                   help="Val images per checkpoint viz grid (default: 4)")
    p.add_argument("--feat_cache",      action="store_true", default=False,
                   help="Use pre-computed ViT+RoBERTa features (run precompute_features.py first)")
    p.add_argument("--text_in_memory",  action="store_true", default=False,
                   help="Compute text features in RAM at run start instead of loading "
                        "an on-disk text cache (requires --feat_cache). Encodes the "
                        "filtered CSV's unique queries once with the chosen --text_encoder, "
                        "keeps the dict in memory (shared by train+val), and writes nothing "
                        "to disk. Use this for Super-CLEVR (many unique questions → huge "
                        "text cache) or when /pfs quota is tight.")
    p.add_argument("--dino_in_memory",  action="store_true", default=False,
                   help="Compute DINO/ViT image features in RAM at run start instead of "
                        "loading an on-disk feature cache (requires --feat_cache). Loads "
                        "the frozen feature extractor once, encodes the filtered CSV's "
                        "unique images, keeps the dict in memory (shared by train+val), "
                        "and writes nothing to disk. Use this to skip a separate "
                        "precompute_features.py run or when /pfs quota is tight. Honours "
                        "--slot_backend (oclf vs ftdinosaur), --dinosaur_ckpt/--dinosaur_cfg, "
                        "and --img_size.")
    p.add_argument("--patch_control",  action="store_true", default=False,
                   help="Use PatchClassifier (raw ViT patches) instead of SlotClassifier")
    p.add_argument("--pooler",          type=str,   default=None,
                   choices=["gated_attn", "transformer", "gated_then_transformer",
                            "vqa_transformer", "vqa_paper", "hier_router", "qca", "patch_qdot"],
                   help="Slot/text fusion pooler. 'gated_attn' (default) is the original "
                        "single-hop gated cross-attention. 'transformer' concatenates "
                        "[CLS]+slots+text and passes them through a small transformer encoder. "
                        "'gated_then_transformer' runs gated cross-attention first, then "
                        "feeds its updated slots into the transformer for the CLS readout. "
                        "'vqa_transformer' follows Ding et al. (2021a): separate img/text linear "
                        "projections, sinusoidal PE on text only, 2-d modality one-hot, "
                        "[z', t', CLS] concatenation, transformer encoder, MLP head on CLS. "
                        "'vqa_paper' is the exact VQA downstream model from the OC-VQA paper: "
                        "single linear projection per modality on raw features, d_model=128, "
                        "ff=128, standard post-norm/ReLU transformer (T-n via --pooler_layers), "
                        "2-layer MLP head (LN+Dropout+ReLU). Pair with --text_encoder t5. "
                        "'hier_router' is the structured path-marginalisation head for "
                        "'color of <x> of <y>': it traverses the recursive parent->child slot "
                        "tree (P(j|y)*P(k|j,x)*P(a|c_jk)) and emits answer log-probs (NLL loss). "
                        "Requires --recursive_infer (auto-on) and a spans-enabled text cache. "
                        "'qca' is the flat (unstructured) control for hier_router: the '<y> <x>' "
                        "compound query cross-attends once over the object slots (SlotClassifier) "
                        "or the raw patch tokens (--patch_control), with the same query-conditioned "
                        "colour readout, no tree/marginalisation. Cached path + spans, NLL loss. "
                        "'patch_qdot' is the WEAKEST flat control (requires --patch_control): the "
                        "'<y> <x>' query dot-products (no learned key/value, no multi-head) over the "
                        "frozen ViT patch tokens, same query-conditioned colour readout. raw "
                        "(d_vit patches) vs projected (--patch_qdot_project_patches → d_slot). "
                        "Cached path + spans, NLL loss.")
    # ── patch_qdot (Patch-QDot) baseline knobs ───────────────────────────────
    p.add_argument("--patch_qdot_project_patches", action="store_true", default=False,
                   help="pooler='patch_qdot': project patches d_vit->d_slot before the query "
                        "dot-product (Version A). Default off = raw d_vit patches (Version B).")
    p.add_argument("--patch_qdot_keep_registers", action="store_true", default=False,
                   help="pooler='patch_qdot': attend over all 200 tokens (keep the 4 DINOv3 "
                        "register tokens). Default off = strip registers → 196 spatial patches.")
    p.add_argument("--patch_qdot_temperature", type=float, default=1.0,
                   help="pooler='patch_qdot': softmax temperature on the query·patch scores "
                        "(divides the logits; default 1.0).")
    p.add_argument("--patch_qdot_normalize", action="store_true", default=False,
                   help="pooler='patch_qdot': L2-normalise query+patches (cosine similarity) "
                        "before the dot-product (then only temperature scales the logits).")
    p.add_argument("--text_encoder",    type=str,   default=None,
                   choices=["roberta", "t5"],
                   help="Frozen text encoder. 'roberta' (default, RoBERTa-Large, 1024-d) or "
                        "'t5' (T5-base encoder, 768-d) to match the OC-VQA paper. Selecting 't5' "
                        "sets d_text=768 and defaults the text cache to a *_t5.pt file.")
    p.add_argument("--vqa_d_model",     type=int,   default=None,
                   help="Transformer working dim for --pooler vqa_paper (default 128, the paper value).")
    p.add_argument("--pooler_layers",   type=int,   default=None,
                   help="Number of transformer layers when --pooler transformer/vqa_*. "
                        "For vqa_paper this selects T-n (paper uses 2, 5, 15). Default 2.")
    p.add_argument("--pooler_dropout",  type=float, default=None,
                   help="Dropout inside the transformer pooler (default 0.0)")
    p.add_argument("--recursive_infer", action="store_true", default=False,
                   help="Recursive zoom-in inference: rank the slots via the transformer "
                        "pooler's CLS->slot attention, refine the top-ranked slot into "
                        "--recursive_children finer child slots (re-running the frozen "
                        "DINOSAUR slot attention on that slot's masked DINO patches), then "
                        "re-classify on those children alone. Applies to train AND eval. "
                        "Requires a transformer-family --pooler and the oclf slot backend.")
    p.add_argument("--recursive_children", type=int, default=None,
                   help="Number of child slots each refined parent slot is split into "
                        "under --recursive_infer (default 4).")
    p.add_argument("--recursive_parents", type=int, default=None,
                   help="Number of top-ranked parent slots to refine independently under "
                        "--recursive_infer; their children are pooled for the answer "
                        "(default 1 = single top slot).")
    p.add_argument("--recursive_spread", type=float, default=None,
                   help="Strength of the child spatial-spread prior in the second slot-"
                        "attention pass (default 0.5). 0 = pure frozen SA init (most "
                        "liberal, children may collapse together); ~0.5 = gentle bias to "
                        "different regions; 1 = forced maximally-separated spatial cells.")
    p.add_argument("--recursive_children_only", action="store_true", default=False,
                   help="Under --recursive_infer, feed ONLY the child slots to the second "
                        "(re-classification) pass. Default is to include the refined parent "
                        "slot(s) alongside their children.")
    p.add_argument("--rank_method", type=str, default=None,
                   choices=["attention", "attribution"],
                   help="How --recursive_infer ranks slots to pick the parent(s) to zoom "
                        "into. 'attention' (default): the transformer pooler's CLS->slot "
                        "attention (transformer-family poolers only). 'attribution': "
                        "gradient saliency ||dy_c/ds_i|| of the predicted-answer logit "
                        "w.r.t. each slot (Simonyan 2013 / AwGA), works for any pooler.")
    p.add_argument("--router_temp", type=float, default=None,
                   help="(--pooler hier_router) softmax temperature on the parent P(j|y) and "
                        "child P(k|j,x) routing distributions (default 1.0).")
    p.add_argument("--router_entropy_weight", type=float, default=None,
                   help="(--pooler hier_router) weight of the entropy regulariser on P(j|y) "
                        "added to the loss to encourage exploration (default 0.0 = off).")
    p.add_argument("--child_scorer", type=str, default=None, choices=["bilinear", "mlp"],
                   help="(--pooler hier_router) child-routing scorer: 'bilinear' dot-product "
                        "(default) or 'mlp' over (part-query, child slot, parent slot).")
    p.add_argument("--router_parent_only", action="store_true", default=False,
                   help="(--pooler hier_router) ABLATION: drop the child level entirely — route "
                        "only over object slots and read colour off the parent (P(a)=Σ_j P(j|y)·"
                        "P(a|s_j); the part word <x> is unused). Tests the utility of children.")
    p.add_argument("--router_no_readout", action="store_true", default=False,
                   help="(--pooler hier_router) ABLATION: do NOT condition the colour readout on the "
                        "query — use P(colour|slot) instead of P(colour|slot, f_readout('<y> <x>')).")
    p.add_argument("--router_color_source", type=str, default=None,
                   choices=["slot", "patch", "patch_qdot"],
                   help="(--pooler hier_router) colour evidence for P(a|.): 'slot' (default) reads the "
                        "routed child SLOT vector; 'patch' pools the raw DINO patches the child grounds "
                        "to (Σ_n a_jkn·patch_n); 'patch_qdot' runs a learned query·patch dot-product "
                        "(project patches + '<y> <x>' query, dot, softmax) CONFINED to each child's "
                        "patches, then reads out — routing is unchanged in all three.")
    p.add_argument("--router_qdot_raw", action="store_true", default=False,
                   help="(--router_color_source patch_qdot) do NOT project the patches before the "
                        "dot-product (use raw d_vit); default projects d_vit→d_slot first.")
    p.add_argument("--router_init_ckpt", type=str, default=None,
                   help="(--pooler hier_router) warm-start the ROUTING (text projector + q_parent/q_child/"
                        "child_mlp) from this best_model.pt; the colour head stays freshly initialised. Use "
                        "with --router_color_source patch + --router_freeze_routing to train ONLY a patch "
                        "colour head on top of a previously-trained routing.")
    p.add_argument("--router_freeze_routing", action="store_true", default=False,
                   help="(--pooler hier_router) freeze the routing path + text projector (only the colour "
                        "head trains). Requires --router_init_ckpt (otherwise the routing would be frozen "
                        "at random init).")
    p.add_argument("--router_init_color_head", action="store_true", default=False,
                   help="(--pooler hier_router) ALSO warm-start the colour head (color_head/f_readout or "
                        "qdot_readout) from --router_init_ckpt, not just the routing. The LP-FT recipe: "
                        "init from a trained linear-probe (frozen-head) run, then finetune everything.")
    p.add_argument("--router_qdot_dropout", type=float, default=None,
                   help="(--router_color_source patch_qdot) dropout on the qdot colour-head readout "
                        "feature (regulariser for the finetune regime; default 0.0).")
    p.add_argument("--zero_image_feats", action="store_true", default=False,
                   help="Diagnostic: zero slots/patches before the pooler so the head sees text only.")
    p.add_argument("--label_smoothing", type=float, default=None,
                   help="Cross-entropy label smoothing eps (default 0.0).")
    p.add_argument("--weight_decay",    type=float, default=None,
                   help="AdamW weight_decay (default 1e-2).")
    p.add_argument("--num_heads",       type=int,   default=None,
                   help="Attention heads in pooler / GCA (default 8).")
    p.add_argument("--img_size",        type=int,   default=None,
                   help="Input image resolution (default 224). Must match the resolution "
                        "the slot ckpt was trained at — e.g. use 448 with a 448-trained "
                        "DINOSAUR ckpt. Only effective when --feat_cache is OFF.")
    p.add_argument("--resize_mode",     choices=["crop", "square", "pad"], default=None,
                   help="Image preprocessing before the backbone: 'crop' (Resize shorter "
                        "side + CenterCrop; aspect-preserving but crops long-axis edges, "
                        "default), 'square' (Resize to S×S; keeps all content, distorts "
                        "aspect), or 'pad' (pad to square then resize; keeps content + "
                        "aspect with bars). Changes cached features → re-precompute/retrain.")
    p.add_argument("--augment",         action="store_true", default=False,
                   help="Enable train-time image augmentation (RandomResizedCrop). "
                        "Only effective when --feat_cache is OFF.")
    p.add_argument("--overfit_n",       type=int,   default=None,
                   help="Diagnostic: train AND validate on the first N train samples "
                        "(0 = off). The same N rows go through both the train and eval "
                        "loops, so train acc == val acc iff the eval loop is correct.")
    p.add_argument("--train_frac",      type=float, default=None,
                   help="Quick check: train on a random fraction of the train split "
                        "(e.g. 0.2 for 20%%). Validation/test sets stay full. "
                        "Seeded for reproducibility. Default 1.0 (full train set).")
    p.add_argument("--seed",            type=int,   default=None,
                   help="Global RNG seed (torch/numpy/random) for reproducible / "
                        "multi-seed runs. Default None = unseeded (system entropy).")
    p.add_argument("--csv_path",        type=str,   default=None,
                   help="Override the dataset CSV path (default: per --dataset).")
    p.add_argument("--text_cache",      type=str,   default=None,
                   help="Override the RoBERTa text-feature cache path "
                        "(default: per --dataset).")
    p.add_argument("--dino_cache",      type=str,   default=None,
                   help="Override the DINOSAUR ViT-feature cache path "
                        "(default: per --dataset).")
    p.add_argument("--skip_per_query_eval", action="store_true", default=False,
                   help="Skip the slow per-query evaluation at each checkpoint. "
                        "Useful when there are many queries with large caches "
                        "(sc3d: 109 queries × full ~300MB cache reload each = "
                        "several minutes per checkpoint). The viz step still runs.")
    p.add_argument("--text_onehot",    action="store_true", default=False,
                   help="Diagnostic: drop the language encoder. The text path "
                        "becomes nn.Embedding(n_unique_queries, d_slot) keyed "
                        "by question identity (equivalent to one-hot @ "
                        "learnable W). Requires --feat_cache. Disables the "
                        "checkpoint viz (which uses RoBERTa).")
    p.add_argument("--resume",         action="store_true", default=False,
                   help="Auto-resume from <checkpoint_dir>/last.pt if it exists. "
                        "Restores model trainable weights, optimizer state, LR "
                        "scheduler, epoch counter, best_val_acc, and the "
                        "no-improve patience counter. Designed for chained "
                        "wall-time-limited chunks (e.g. SLURM array %%1).")
    p.add_argument("--resume_from",    type=str,   default=None,
                   help="Explicit path to a last.pt checkpoint to resume from. "
                        "Overrides --resume's auto-detect.")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[seed] global RNG seeded with {args.seed}", flush=True)
    cfg  = dict(CONFIG)
    cfg["dataset"] = args.dataset
    cfg.update(DATASET_CONFIGS[args.dataset])
    cfg["seed"] = args.seed

    if args.query_filter      is not None: cfg["query_filter"]      = args.query_filter
    if args.category_filter   is not None: cfg["category_filter"]   = args.category_filter
    if args.attribute_filter  is not None: cfg["attribute_filter"]  = args.attribute_filter
    if args.depth_filter      is not None: cfg["depth_filter"]      = args.depth_filter
    if args.max_n_objects     is not None: cfg["max_n_objects"]     = args.max_n_objects
    if args.object_category   is not None: cfg["object_category"]   = args.object_category
    if args.text_onehot:                   cfg["text_onehot"]        = True
    if args.n_slots           is not None: cfg["n_slots"]           = args.n_slots
    if args.max_epochs        is not None: cfg["max_epochs"]        = args.max_epochs
    if args.patience          is not None: cfg["patience"]          = args.patience
    if args.lr                is not None: cfg["lr"]                = args.lr
    if args.batch_size        is not None: cfg["batch_size"]        = args.batch_size
    if args.warmup_steps      is not None: cfg["warmup_steps"]      = args.warmup_steps
    if args.optimizer         is not None: cfg["optimizer"]         = args.optimizer
    if args.lr_schedule       is not None: cfg["lr_schedule"]       = args.lr_schedule
    if args.max_steps         is not None: cfg["max_steps"]         = args.max_steps
    if args.slot_backend      is not None: cfg["slot_backend"]        = args.slot_backend
    if args.ftdinosaur_model  is not None: cfg["ftdinosaur_model"]    = args.ftdinosaur_model
    if args.dinosaur_ckpt     is not None: cfg["dinosaur_ckpt"]       = args.dinosaur_ckpt
    if args.dinosaur_cfg      is not None: cfg["dinosaur_cfg_name"]   = args.dinosaur_cfg
    if args.finetune_ckpt     is not None: cfg["finetune_ckpt_path"]  = args.finetune_ckpt
    if args.checkpoint_dir    is not None: cfg["checkpoint_dir"]      = args.checkpoint_dir
    if args.checkpoint_every  is not None: cfg["checkpoint_every"]  = args.checkpoint_every
    if args.viz_n_samples     is not None: cfg["viz_n_samples"]     = args.viz_n_samples
    if args.img_size          is not None: cfg["img_size"]          = args.img_size
    if args.resize_mode       is not None: cfg["resize_mode"]       = args.resize_mode
    if args.feat_cache:                    cfg["feat_cache"]         = True
    if args.text_in_memory:                cfg["text_in_memory"]     = True
    if args.dino_in_memory:                cfg["dino_in_memory"]     = True
    if args.patch_control:                 cfg["patch_control"]      = True
    if args.pooler            is not None: cfg["pooler"]            = args.pooler
    if args.text_encoder      is not None: cfg["text_encoder"]      = args.text_encoder
    if args.vqa_d_model       is not None: cfg["vqa_d_model"]       = args.vqa_d_model
    if args.pooler_layers     is not None: cfg["pooler_layers"]     = args.pooler_layers
    if args.pooler_dropout    is not None: cfg["pooler_dropout"]    = args.pooler_dropout
    if args.recursive_infer:               cfg["recursive_infer"]   = True
    if args.recursive_children is not None: cfg["recursive_children"] = args.recursive_children
    if args.recursive_parents is not None:  cfg["recursive_parents"]  = args.recursive_parents
    if args.router_temp is not None:           cfg["router_temp"]           = args.router_temp
    if args.router_entropy_weight is not None: cfg["router_entropy_weight"] = args.router_entropy_weight
    if args.child_scorer is not None:          cfg["child_scorer"]          = args.child_scorer
    if args.router_parent_only:                cfg["router_parent_only"]    = True
    if args.router_no_readout:                 cfg["router_readout_query"]  = False
    if args.router_color_source is not None:   cfg["router_color_source"]   = args.router_color_source
    if args.router_init_ckpt is not None:      cfg["router_init_ckpt"]      = args.router_init_ckpt
    if args.router_freeze_routing:             cfg["router_freeze_routing"] = True
    if args.router_init_color_head:            cfg["router_init_color_head"] = True
    if args.router_qdot_raw:                   cfg["router_qdot_project_patches"] = False
    if args.router_qdot_dropout is not None:   cfg["router_qdot_dropout"]   = args.router_qdot_dropout
    if args.patch_qdot_project_patches:        cfg["patch_qdot_project_patches"] = True
    if args.patch_qdot_keep_registers:         cfg["patch_qdot_strip_registers"] = False
    if args.patch_qdot_temperature is not None: cfg["patch_qdot_temperature"]    = args.patch_qdot_temperature
    if args.patch_qdot_normalize:              cfg["patch_qdot_normalize"]       = True
    if args.recursive_spread is not None:   cfg["recursive_spread"]   = args.recursive_spread
    if args.recursive_children_only:        cfg["recursive_include_parents"] = False
    if args.rank_method       is not None: cfg["rank_method"]       = args.rank_method
    if args.zero_image_feats:              cfg["zero_image_feats"]  = True
    if args.label_smoothing   is not None: cfg["label_smoothing"]   = args.label_smoothing
    if args.weight_decay      is not None: cfg["weight_decay"]      = args.weight_decay
    if args.num_heads         is not None: cfg["num_heads"]         = args.num_heads
    if args.augment:                       cfg["augment"]           = True
    if args.overfit_n         is not None: cfg["overfit_n"]         = args.overfit_n
    if args.train_frac        is not None: cfg["train_frac"]        = args.train_frac
    if args.csv_path          is not None: cfg["csv_path"]          = args.csv_path
    if args.text_cache        is not None: cfg["text_cache"]        = args.text_cache
    if args.dino_cache        is not None: cfg["dino_cache"]        = args.dino_cache
    if args.skip_per_query_eval:           cfg["skip_per_query_eval"] = True

    # ── T5 text-encoder derived settings ─────────────────────────────────────
    # T5-base encoder emits 768-d tokens (vs RoBERTa-Large 1024). When --text_encoder
    # t5 is selected and the user didn't override --text_cache, point at a *_t5.pt
    # cache so it never collides with the RoBERTa cache (different hidden size).
    if cfg["text_encoder"] == "t5":
        cfg["d_text"] = 768
        if args.text_cache is None:
            base = cfg["text_cache"]
            cfg["text_cache"] = (
                base[:-3] + "_t5.pt" if base.endswith(".pt") else base + "_t5"
            )

    device       = torch.device(cfg["device"])
    use_cache    = cfg["feat_cache"]
    n_slots      = cfg["n_slots"]

    # hier_router traverses the recursive parent→child tree, so it implies
    # --recursive_infer and, unless overridden, refines EVERY slot (P = n_slots) so
    # the parent routing P(j|y) is a real distribution over all candidate objects.
    if cfg["pooler"] == "hier_router":
        if not use_cache:
            raise SystemExit(
                "pooler='hier_router' requires the cached path (--feat_cache, e.g. "
                "--feat_cache --dino_in_memory --text_in_memory): the non-cached "
                "AttributeDataset does not emit the <x>/<y> span vectors the router needs."
            )
        if not cfg["recursive_infer"]:
            print("pooler='hier_router' → enabling --recursive_infer.")
        cfg["recursive_infer"] = True
        if args.recursive_parents is None:
            cfg["recursive_parents"] = n_slots
            print(f"pooler='hier_router' → recursive_parents defaulted to n_slots={n_slots} "
                  f"(refine all slots).")
        if cfg.get("router_color_source") in ("patch", "patch_qdot") and cfg.get("router_parent_only", False):
            raise SystemExit(
                f"--router_color_source {cfg.get('router_color_source')} is incompatible with "
                "--router_parent_only: the patch evidence is read from the child slots' "
                "attention, which the parent-only ablation removes."
            )
        if cfg.get("router_freeze_routing") and not cfg.get("router_init_ckpt"):
            raise SystemExit(
                "--router_freeze_routing requires --router_init_ckpt (otherwise the "
                "routing would be frozen at random init). Pass a trained best_model.pt."
            )
        if cfg.get("router_init_color_head") and not cfg.get("router_init_ckpt"):
            raise SystemExit(
                "--router_init_color_head requires --router_init_ckpt (the colour head "
                "is loaded from that checkpoint)."
            )

    # qca is the flat control for hier_router: it reads the same '<y> <x>' span query
    # from the cached text features, so it likewise requires the cached path. Unlike
    # hier_router it does NOT enable --recursive_infer (no tree), which is also what
    # keeps it compatible with --patch_control (Patch-QCA).
    if cfg["pooler"] == "qca":
        if not use_cache:
            raise SystemExit(
                "pooler='qca' requires the cached path (--feat_cache, e.g. "
                "--feat_cache --dino_in_memory --text_in_memory): the non-cached "
                "AttributeDataset does not emit the '<y> <x>' span vector qca needs."
            )
    # patch_qdot is the weakest flat patch control: it dot-products the '<y> <x>' span
    # query over the raw ViT patch tokens, so it lives ONLY in PatchClassifier and needs
    # the cached span path — require both --patch_control and the cached path.
    if cfg["pooler"] == "patch_qdot":
        if not cfg["patch_control"]:
            raise SystemExit(
                "pooler='patch_qdot' is a patch baseline and requires --patch_control "
                "(it dot-products over ViT patch tokens; there is no SlotClassifier path)."
            )
        if not use_cache:
            raise SystemExit(
                "pooler='patch_qdot' requires the cached path (--feat_cache, e.g. "
                "--feat_cache --dino_in_memory --text_in_memory): the non-cached "
                "AttributeDataset does not emit the '<y> <x>' span vector it needs."
            )
    patch_control = cfg["patch_control"]
    slot_backend = cfg["slot_backend"]

    # ── Recursive zoom-in guards ──────────────────────────────────────────────
    if cfg["recursive_infer"]:
        if patch_control:
            raise SystemExit(
                "--recursive_infer is not supported with --patch_control "
                "(PatchClassifier has no slot stack to refine)."
            )
        if slot_backend != "oclf":
            raise SystemExit(
                "--recursive_infer currently supports the oclf slot backend only."
            )

    # --text_in_memory only applies to the cached-feature path (the non-cached
    # path already encodes text live each step). Warn rather than fail silently.
    if cfg["text_in_memory"] and not use_cache:
        print("WARN: --text_in_memory has no effect without --feat_cache "
              "(non-cached training already encodes text live); ignoring.")
        cfg["text_in_memory"] = False

    # --dino_in_memory only applies to the cached-feature path (the non-cached
    # path already runs the ViT live each step). Warn rather than fail silently.
    if cfg["dino_in_memory"] and not use_cache:
        print("WARN: --dino_in_memory has no effect without --feat_cache "
              "(non-cached training already runs the ViT live); ignoring.")
        cfg["dino_in_memory"] = False
    if cfg["dino_in_memory"] and slot_backend == "oclf" and not cfg["dinosaur_ckpt"]:
        raise SystemExit(
            "--dino_in_memory with --slot_backend oclf requires --dinosaur_ckpt "
            "(the checkpoint to load the frozen feature extractor from)."
        )

    # ── ftdinosaur backend guards ────────────────────────────────────────────
    # ftdinosaur has no PatchClassifier variant and is wired for CUB (the only
    # dataset whose AttributeDataset takes an injectable image_transform). With
    # --feat_cache it consumes a cache of *encoder* features (256x768) and re-runs
    # slot attention live; build it via:
    #   precompute_features.py --slot_backend ftdinosaur
    if slot_backend == "ftdinosaur":
        if patch_control:
            raise SystemExit("--slot_backend ftdinosaur is incompatible with --patch_control.")
        if cfg["dataset"] != "cub":
            raise SystemExit("--slot_backend ftdinosaur is currently wired only for --dataset cub.")
        # Default to the ftdinosaur encoder-feature cache unless overridden.
        if use_cache and args.dino_cache is None:
            cfg["dino_cache"] = "FG-datset/CUB_200_2011/ftdino_feat_cache.pt"

    if cfg["pooler"] == "transformer":
        pooler_desc = f"transformer×{cfg['pooler_layers']}"
    elif cfg["pooler"] == "gated_then_transformer":
        pooler_desc = f"gated_attn → transformer×{cfg['pooler_layers']}"
    elif cfg["pooler"] in ("vqa_transformer", "vqa_paper"):
        pooler_desc = f"{cfg['pooler']}×{cfg['pooler_layers']} (T-{cfg['pooler_layers']})"
    else:
        pooler_desc = cfg["pooler"]
    txt_desc = (
        f"t5({cfg['t5_model']},d={cfg['d_text']})" if cfg["text_encoder"] == "t5"
        else f"roberta({cfg['roberta_model']},d={cfg['d_text']})"
    )
    print(
        f"Device: {device}  |  backend: {slot_backend}  |  n_slots: {n_slots}  "
        f"|  feat_cache: {use_cache}  |  patch_control: {patch_control}  |  pooler: {pooler_desc}"
        f"  |  text_encoder: {txt_desc}"
        + ("  |  ZERO_IMAGE_FEATS" if cfg["zero_image_feats"] else "")
    )
    if slot_backend == "ftdinosaur":
        print(f"ftdinosaur model: {cfg['ftdinosaur_model']}")
    print(
        f"Reg: lr={cfg['lr']}  wd={cfg['weight_decay']}  "
        f"label_smoothing={cfg['label_smoothing']}  "
        f"pooler_dropout={cfg['pooler_dropout']}  "
        f"num_heads={cfg['num_heads']}  "
        f"augment={cfg['augment']}"
    )
    if cfg["augment"] and use_cache:
        print("WARN: --augment has no effect when --feat_cache is on (features are pre-baked).")

    # ── Build label vocab ────────────────────────────────────────────────────
    if cfg["dataset"] == "cub":
        from cub_dataset import (
            CUBAttributeDataset      as AttributeDataset,
            CUBCachedFeatDataset     as CachedFeatDataset,
            build_label_vocab        as build_label_vocab,
        )
        _extra_filter_kwargs = {}
    elif cfg["dataset"] in ("superclevr3d", "ade20k"):
        # ade20k shares superclevr3d's CSV schema ({train,val} splits, query/
        # label/attribute_type/depth/n_objects columns), so it reuses the same
        # dataset classes and label-vocab builder.
        from superclevr3d_dataset import (
            SuperCLEVR3DAttributeDataset      as AttributeDataset,
            SuperCLEVR3DCachedFeatDataset     as CachedFeatDataset,
            build_label_vocab_superclevr3d    as build_label_vocab,
        )
        _extra_filter_kwargs = {
            "attribute_filter": cfg["attribute_filter"],
            "depth_filter":     cfg["depth_filter"],
            "max_n_objects":    cfg["max_n_objects"],
            "object_category":  cfg["object_category"],
        }
    else:
        raise ValueError(f"Unknown dataset {cfg['dataset']!r}")

    df_full = pd.read_csv(cfg["csv_path"])
    df_full["label"] = df_full["label"].astype(str)
    if cfg["query_filter"] is not None:
        df_filt = df_full[df_full["query"] == cfg["query_filter"]]
    elif cfg["category_filter"] is not None:
        df_filt = df_full[
            df_full["query"].str.contains(cfg["category_filter"], case=False, na=False)
        ]
    else:
        df_filt = df_full
    if cfg["attribute_filter"] is not None and "attribute_type" in df_filt.columns:
        df_filt = df_filt[df_filt["attribute_type"] == cfg["attribute_filter"]]
    if cfg["depth_filter"] is not None and "depth" in df_filt.columns:
        df_filt = df_filt[df_filt["depth"] == cfg["depth_filter"]]
    if cfg["max_n_objects"] is not None and "n_objects" in df_filt.columns:
        df_filt = df_filt[df_filt["n_objects"] <= cfg["max_n_objects"]]
    if cfg["object_category"] is not None:
        import re as _re
        pat = rf"\b{_re.escape(cfg['object_category'])}\b"
        df_filt = df_filt[df_filt["query"].str.contains(pat, case=False, na=False, regex=True)]
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

    # ── Question vocab for text_onehot diagnostic ───────────────────────────
    # Built from the train rows of df_filt (already passed every active
    # filter), then persisted so the eval/viz scripts can reload it. All
    # val/test queries in simple_color_vqa are subsets of train, so OOV is
    # impossible there; rows that *do* fall outside the vocab are dropped by
    # the dataset with a warning.
    question_vocab = None
    if cfg["text_onehot"]:
        if cfg["dataset"] != "superclevr3d":
            raise ValueError("--text_onehot is only wired for --dataset superclevr3d")
        if not use_cache:
            raise ValueError("--text_onehot requires --feat_cache (cached DINOSAUR features)")
        train_queries  = sorted(df_filt[df_filt["split"] == "train"]["query"].unique())
        question_vocab = {q: i for i, q in enumerate(train_queries)}
        q_vocab_path   = ckpt_dir / "question_vocab.json"
        with open(q_vocab_path, "w") as f:
            json.dump(question_vocab, f, indent=2)
        print(f"Question vocab: {len(question_vocab)} unique queries → {q_vocab_path}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    ds_kwargs = dict(
        csv_path        = cfg["csv_path"],
        label_vocab     = label_vocab,
        query_filter    = cfg["query_filter"],
        category_filter = cfg["category_filter"],
        **_extra_filter_kwargs,
    )

    # superclevr3d/ade20k use a {train, val(, test)} split; CUB uses {train, test}.
    val_split_name = "test" if cfg["dataset"] == "cub" else "val"

    tokenizer = None
    in_mem_text = None   # set below when --text_in_memory; reused by per-query eval
    in_mem_dino = None   # set below when --dino_in_memory; reused by per-query eval
    if use_cache:
        cached_extra = {}
        if cfg["text_onehot"] and cfg["dataset"] == "superclevr3d":
            cached_extra = dict(text_onehot=True, question_vocab=question_vocab)
        if cfg["pooler"] in ("hier_router", "qca", "patch_qdot"):
            # The router / qca / patch_qdot heads need per-query span vectors from the text
            # cache (built with_spans below / by precompute_features --with_spans): hier_router
            # reads ch1/ch3, qca + patch_qdot read ch3 ("<y> <x>", e.g. "car door").
            cached_extra["return_spans"] = True

        # The in-memory precompute only needs the images/queries in the splits that
        # are actually consumed: the train split + the val split, plus test only when
        # the per-query eval will run. Restricting to these avoids encoding every
        # other split's data (e.g. computing test features when --skip_per_query_eval).
        if "split" in df_filt.columns:
            needed_splits = {"train", val_split_name}
            if (cfg.get("checkpoint_every", 0) > 0
                    and not cfg.get("skip_per_query_eval")
                    and len(active_queries) <= 200):
                needed_splits.add("test")
            df_used = df_filt[df_filt["split"].isin(needed_splits)]
            print(f"In-memory precompute restricted to splits {sorted(needed_splits)} "
                  f"({len(df_used):,}/{len(df_filt):,} rows).")
        else:
            df_used = df_filt

        # --text_in_memory: encode the filtered CSV's unique queries once in RAM
        # (shared by train+val) instead of loading an on-disk text cache. Avoids
        # the huge Super-CLEVR text cache / quota / corrupt-partial-write issues.
        if cfg["text_in_memory"]:
            if cfg["text_onehot"]:
                raise SystemExit("--text_in_memory is incompatible with --text_onehot.")
            from precompute_features import precompute_text
            uq = sorted(df_used["query"].unique().tolist())
            print(f"Computing in-memory {cfg['text_encoder']} text features for "
                  f"{len(uq)} unique queries (no disk cache) …")
            in_mem_text = precompute_text(uq, device, text_encoder=cfg["text_encoder"],
                                          with_spans=(cfg["pooler"] in ("hier_router", "qca", "patch_qdot")),
                                          span_dataset=cfg["dataset"])
            cached_extra["text_cache"] = in_mem_text  # passed to both datasets (one shared copy)

        # --dino_in_memory: run the frozen ViT over the filtered CSV's unique
        # images once in RAM (shared by train+val) instead of loading an on-disk
        # feature cache. Mirrors precompute_features.py's image path. The
        # temporary extractor is freed after; the model loads its own frozen copy.
        if cfg["dino_in_memory"]:
            uimgs = sorted(df_used["image_name"].unique().tolist())
            print(f"Computing in-memory DINO features for {len(uimgs)} unique "
                  f"images via {slot_backend} (no disk cache) …")
            if slot_backend == "ftdinosaur":
                from precompute_features import precompute_ftdinosaur_encoder
                from classifier_model import _build_ftdinosaur
                from ftdinosaur_inference import build_dinosaur as _bd
                _ft_model   = _build_ftdinosaur(cfg["ftdinosaur_model"])
                _ft_preproc = _bd.build_preprocessing(cfg["ftdinosaur_model"])
                in_mem_dino = precompute_ftdinosaur_encoder(
                    _ft_model, _ft_preproc, uimgs, cfg["image_root"], device,
                    batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                )
                del _ft_model
            else:
                from precompute_features import precompute_dino
                from classifier_model import _load_dinosaur_submodules
                _repo_root = os.path.dirname(os.path.abspath(__file__))
                _fe, _, _  = _load_dinosaur_submodules(
                    cfg["dinosaur_cfg_name"], cfg["dinosaur_ckpt"], _repo_root,
                    n_slots=n_slots,
                )
                _fe.eval()
                in_mem_dino = precompute_dino(
                    _fe, uimgs, cfg["image_root"], device,
                    img_size=cfg["img_size"], batch_size=cfg["batch_size"],
                    num_workers=cfg["num_workers"], resize_mode=cfg["resize_mode"],
                )
                del _fe
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cached_extra["dino_cache"] = in_mem_dino  # passed to both datasets (one shared copy)

        train_ds = CachedFeatDataset(
            **ds_kwargs,
            dino_cache_path = cfg["dino_cache"],
            text_cache_path = cfg["text_cache"],
            split = "train",
            **cached_extra,
        )
        val_ds = CachedFeatDataset(
            **ds_kwargs,
            dino_cache_path = cfg["dino_cache"],
            text_cache_path = cfg["text_cache"],
            split = val_split_name,
            **cached_extra,
        )
    else:
        if cfg["text_encoder"] == "t5":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg["t5_model"])
        else:
            tokenizer = RobertaTokenizer.from_pretrained(cfg["roberta_model"])
        # Only sc3d's AttributeDataset accepts `augment` and `img_size`; CUB's does not.
        _aug_kwargs = (
            {"augment": cfg["augment"], "img_size": cfg["img_size"]}
            if cfg["dataset"] in ("superclevr3d", "ade20k") else {}
        )
        # ftdinosaur (CUB only): override the image transform with the model's
        # own preprocessing so slots are computed at the resolution/normalisation
        # the encoder was trained with (square resize at 224, no center crop).
        _ft_kwargs = {}
        if slot_backend == "ftdinosaur":
            from ftdinosaur_inference import build_dinosaur as _bd
            _ft_kwargs = {"image_transform": _bd.build_preprocessing(cfg["ftdinosaur_model"])}
        train_ds = AttributeDataset(
            **ds_kwargs,
            image_root   = cfg["image_root"],
            split        = "train",
            tokenizer    = tokenizer,
            max_text_len = cfg["max_text_len"],
            **_aug_kwargs,
            **_ft_kwargs,
        )
        val_ds = AttributeDataset(
            **ds_kwargs,
            image_root   = cfg["image_root"],
            split        = val_split_name,
            tokenizer    = tokenizer,
            max_text_len = cfg["max_text_len"],
            **_aug_kwargs,
            **_ft_kwargs,
        )

    # Overfit-N diagnostic: the same N train rows feed both loops.
    # Train mode runs gradient updates; val mode runs the same eval path used
    # on the real val set. Train acc and val acc on the same examples must
    # converge to the same number — if they don't, the eval loop is buggy.
    if cfg["overfit_n"] and cfg["overfit_n"] > 0:
        from torch.utils.data import Subset
        n = min(cfg["overfit_n"], len(train_ds))
        indices = list(range(n))
        overfit_subset_train = Subset(train_ds, indices)
        overfit_subset_val   = Subset(train_ds, indices)
        train_loader = DataLoader(
            overfit_subset_train, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            overfit_subset_val, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        print(f"OVERFIT-{n} MODE: train and val share the first {n} train rows. "
              f"checkpoint_every forced to 0 (viz disabled).")
        cfg["checkpoint_every"] = 0
        print(f"Train: {n}  |  Val: {n}  (same rows)  |  Batch: {cfg['batch_size']}")
    else:
        # Quick check: randomly subsample the train split to `train_frac` (val stays full).
        frac = cfg.get("train_frac", 1.0)
        if frac < 1.0:
            from torch.utils.data import Subset
            full_n = len(train_ds)
            n      = max(1, int(round(frac * full_n)))
            g      = torch.Generator().manual_seed(0)
            indices = torch.randperm(full_n, generator=g)[:n].tolist()
            train_ds = Subset(train_ds, indices)
            print(f"TRAIN SUBSET: {n:,}/{full_n:,} rows ({frac:.0%}) [seed=0]")
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

    onehot_kwargs = {}
    if cfg["text_onehot"]:
        onehot_kwargs = dict(text_onehot=True, n_questions=len(question_vocab))

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
            text_encoder_type  = cfg["text_encoder"],
            t5_model           = cfg["t5_model"],
            vqa_d_model        = cfg["vqa_d_model"],
            load_text_encoder  = not use_cache,
            pooler             = cfg["pooler"],
            pooler_layers      = cfg["pooler_layers"],
            pooler_dropout     = cfg["pooler_dropout"],
            zero_image_feats   = cfg["zero_image_feats"],
            patch_qdot_project_patches = cfg["patch_qdot_project_patches"],
            patch_qdot_strip_registers = cfg["patch_qdot_strip_registers"],
            patch_qdot_temperature     = cfg["patch_qdot_temperature"],
            patch_qdot_normalize       = cfg["patch_qdot_normalize"],
            **onehot_kwargs,
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
            text_encoder_type  = cfg["text_encoder"],
            t5_model           = cfg["t5_model"],
            vqa_d_model        = cfg["vqa_d_model"],
            load_text_encoder  = not use_cache,
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
            router_use_children = not cfg.get("router_parent_only", False),
            router_readout_query = cfg.get("router_readout_query", True),
            router_color_source = cfg.get("router_color_source", "slot"),
            router_qdot_project_patches = cfg.get("router_qdot_project_patches", True),
            router_qdot_dropout = cfg.get("router_qdot_dropout", 0.0),
            **onehot_kwargs,
        ).to(device)

    # ── Optional: warm-start the routing from a trained run, then (optionally) freeze
    # it so only the colour head trains (e.g. a patch colour head on proven routing). ──
    if cfg.get("router_init_ckpt"):
        load_head = cfg.get("router_init_color_head", False)
        rk, hk = load_pretrained_router(model, cfg["router_init_ckpt"],
                                        load_color_head=load_head, device=device)
        msg = (f"Warm-started hier_router routing from {cfg['router_init_ckpt']} "
               f"({len(rk)} routing keys)")
        msg += (f" + colour head ({len(hk)} keys → LP-FT init)" if hk
                else "; colour head left fresh.")
        print(msg)
        if load_head and not hk:
            print("  WARNING: --router_init_color_head set but no colour-head keys "
                  "matched (different color_source / shape?). Colour head is fresh.")
    if cfg.get("router_freeze_routing"):
        model.freeze_routing()
        print("Froze routing path + text projector → only the colour head trains.")

    n_trainable = sum(p.numel() for p in model.trainable_parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.trainable_parameters() if not p.requires_grad)
    print(f"Trainable params: {n_trainable:,}" +
          (f"  (frozen: {n_frozen:,})" if n_frozen else ""))

    # hier_router has its own checkpoint viz (make_hier_router_viz draws the routing
    # traversal), so we no longer disable viz for it — the old flat-pooler tree viz
    # was meaningless for the router.

    # ── Text-feature accessor for checkpoint visualisation ───────────────────
    # Returns (text_hidden (1,L,d_text), attn_mask (1,L)) on `device`.
    # get_spans(query) → (1, 4, d_text) x / y / '<x> of the <y>' / '<y> <x>' span vectors (hier_router viz only).
    get_spans = None
    if cfg["text_onehot"]:
        # Viz uses forward_with_viz, which expects RoBERTa hidden states. In
        # onehot mode we drop the RoBERTa encoder entirely, so viz can't run.
        # Force it off so the loop doesn't try and crash.
        if cfg["checkpoint_every"] != 0:
            print("text_onehot=True → disabling checkpoint viz "
                  "(viz path requires RoBERTa hidden states).")
            cfg["checkpoint_every"] = 0
        def get_text_feats(query: str):
            raise RuntimeError("get_text_feats called in text_onehot mode")
    elif use_cache:
        _text_hidden_cache = val_ds.text_hidden   # dict {query: (L, d_text)}
        _attn_mask_cache   = val_ds.attn_masks    # dict {query: (L,)}
        def get_text_feats(query: str):
            hidden = _text_hidden_cache[query].unsqueeze(0).to(device)
            mask   = _attn_mask_cache[query].unsqueeze(0).to(device)
            return hidden, mask
        if cfg["pooler"] == "hier_router":
            _x_vec, _y_vec = val_ds.x_vec, val_ds.y_vec
            _xy_vec, _readout_vec = val_ds.xy_vec, val_ds.readout_vec
            def get_spans(query: str):
                return torch.stack(
                    [_x_vec[query], _y_vec[query], _xy_vec[query], _readout_vec[query]],
                    dim=0,
                ).unsqueeze(0).to(device)
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

    # ── Step budget → epoch budget ───────────────────────────────────────────
    # When --max_steps is given (e.g. 600k to match the paper), translate it to
    # an epoch count from the actual steps/epoch so the budget is robust to the
    # filtered train-set size.
    if cfg.get("max_steps"):
        steps_per_epoch  = max(1, len(train_loader))
        cfg["max_epochs"] = math.ceil(cfg["max_steps"] / steps_per_epoch)
        print(
            f"max_steps={cfg['max_steps']:,} → max_epochs={cfg['max_epochs']:,} "
            f"({steps_per_epoch} steps/epoch)"
        )

    # ── Optimiser & LR schedule ──────────────────────────────────────────────
    # Only optimise params that require grad (routing may be frozen via --router_freeze_routing).
    opt_params = [p for p in model.trainable_parameters() if p.requires_grad]
    if cfg["optimizer"] == "adam":
        # Paper-faithful: plain Adam (no weight decay).
        optimizer = torch.optim.Adam(opt_params, lr=cfg["lr"])
    else:
        optimizer = torch.optim.AdamW(
            opt_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
    total_steps = cfg["max_epochs"] * len(train_loader)
    if cfg["lr_schedule"] == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, cfg["warmup_steps"])
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, cfg["warmup_steps"], total_steps
        )
    print(
        f"Optim: {cfg['optimizer']}  |  lr_schedule: {cfg['lr_schedule']}  "
        f"|  lr={cfg['lr']}  |  warmup={cfg['warmup_steps']}  "
        f"|  max_epochs={cfg['max_epochs']:,}"
    )
    if cfg["pooler"] in ("hier_router", "qca", "patch_qdot"):
        # All emit answer log-probabilities (the router marginalises a per-child colour
        # mixture; qca / patch_qdot log-softmax their colour head) — train with NLL, not CE
        # (CE would apply a second softmax). label_smoothing is not applied in these modes.
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # ── Optional resume from last.pt ─────────────────────────────────────────
    resume_path = None
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from given but not found: {resume_path}")
    elif args.resume:
        candidate = ckpt_dir / "last.pt"
        if candidate.exists():
            resume_path = candidate

    start_epoch       = 0
    best_val_acc      = 0.0
    epochs_no_improve = 0
    if resume_path is not None:
        start_epoch, best_val_acc, epochs_no_improve = load_last_checkpoint(
            resume_path, model, optimizer, scheduler, device,
        )
        print(
            f"Resumed from {resume_path}: start_epoch={start_epoch + 1}, "
            f"best_val_acc={best_val_acc:.4f}, "
            f"epochs_no_improve={epochs_no_improve}/{cfg['patience']}"
        )
        if start_epoch >= cfg["max_epochs"]:
            print(
                f"Already at max_epochs={cfg['max_epochs']}; nothing to do. "
                f"Increase --max_epochs to keep training."
            )
            metrics_fh, _ = open_metrics_writer(ckpt_dir, append=True)
            metrics_fh.close()
            return

    metrics_fh, metrics_writer = open_metrics_writer(ckpt_dir, append=resume_path is not None)

    ckpt_every = cfg["checkpoint_every"]

    # Per-query stats CSV (appended every checkpoint_every epochs).
    # On resume, keep the existing rows rather than truncating.
    pq_csv_path = ckpt_dir / "per_query_stats.csv"
    pq_mode     = "a" if (resume_path is not None and pq_csv_path.exists()) else "w"
    pq_csv_fh   = open(pq_csv_path, pq_mode, newline="")
    pq_writer   = csv.writer(pq_csv_fh)
    if pq_mode == "w":
        pq_writer.writerow(["epoch", "query", "val_acc", "n_samples"])

    # Val DataFrame for visualisation (validation split for sc3d, test split for cub).
    val_df_for_viz = df_filt[df_filt["split"] == val_split_name].copy()

    try:
        for epoch in range(start_epoch + 1, cfg["max_epochs"] + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, device, use_cache,
                desc=f"epoch {epoch}/{cfg['max_epochs']} train",
                recursive=cfg["recursive_infer"],
            )
            val_loss, val_acc = eval_epoch(
                model, val_loader, criterion, device, use_cache,
                desc=f"epoch {epoch}/{cfg['max_epochs']} val",
                recursive=cfg["recursive_infer"],
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

            # Resumable checkpoint — written every epoch so the next chained
            # chunk can pick up mid-schedule. Atomic via tmp-then-rename.
            save_last_checkpoint(
                model, epoch, optimizer, scheduler,
                best_val_acc, epochs_no_improve,
                label_vocab, cfg, ckpt_dir,
            )

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
                # Per-query eval only makes sense when there is a small number
                # of distinct query templates (CUB has 28). With Super-CLEVR-3D
                # we have ~70k unique question strings, so we skip it; the
                # diagnostic eval lives in eval_superclevr3d.py instead.
                if len(active_queries) <= 200 and not cfg.get("skip_per_query_eval"):
                    print(f"\n  [checkpoint epoch {epoch}] per-query evaluation …")
                    pq_results = eval_per_query(
                        model, active_queries, cfg, label_vocab, device, use_cache,
                        tokenizer=tokenizer,
                        text_cache=in_mem_text, dino_cache=in_mem_dino,
                    )
                    for q, (acc, n) in sorted(pq_results.items()):
                        print(f"    {q:<55s}  acc={acc:.4f}  n={n}")
                        pq_writer.writerow([epoch, q, f"{acc:.6f}", n])
                    pq_csv_fh.flush()

                # qca is a flat cross-attention control with no slot/routing trace to
                # draw (and no forward_with_viz path), so skip the per-checkpoint viz.
                if not patch_control and cfg["pooler"] != "qca":
                    print(f"  [checkpoint epoch {epoch}] generating visualisation …")
                    # A diagnostic viz must never kill an (often long, unattended)
                    # training run — log and continue on any failure.
                    try:
                        if cfg["pooler"] == "hier_router":
                            viz_path = make_hier_router_viz(
                                model       = model,
                                val_df      = val_df_for_viz,
                                cfg         = cfg,
                                label_vocab = label_vocab,
                                device      = device,
                                epoch       = epoch,
                                out_dir     = ckpt_dir,
                                get_text_feats = get_text_feats,
                                get_spans   = get_spans,
                                n_samples   = cfg["viz_n_samples"],
                            )
                        else:
                            viz_fn = (make_recursive_tree_viz if cfg["recursive_infer"]
                                      else make_checkpoint_viz)
                            viz_path = viz_fn(
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
                    except Exception as _viz_err:
                        import traceback
                        print(f"  [viz] skipped (error: {_viz_err})")
                        traceback.print_exc()

            if epochs_no_improve >= cfg["patience"]:
                print(f"\nEarly stopping: no improvement for {cfg['patience']} epochs.")
                break
    finally:
        metrics_fh.close()
        pq_csv_fh.close()

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}  (saved to {ckpt_dir})")


if __name__ == "__main__":
    main()
