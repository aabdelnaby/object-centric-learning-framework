"""Precompute frozen ViT (DINOSAUR) and RoBERTa-Large features for CUB-200-2011.

Running training with pre-computed features skips the two large frozen encoders
every step, making each epoch ~20× faster.  Run this script once before the
n_slots sweep.

Usage (from repo root):
    conda run -n oclf_env python precompute_features.py

Outputs
-------
FG-datset/CUB_200_2011/dino_feat_cache.pt
    {
      "features":  {image_name (str): tensor (200, 384)},
      "positions": tensor (200, …)     # same for all images
    }
    Note: DINOv3 (vit_small_patch16_dinov3) strips CLS but keeps 4 register
    tokens, giving 4 + 196 = 200 tokens per image (not 196 as in plain DINO).

FG-datset/CUB_200_2011/text_feat_cache.pt
    {
      "hidden": {query (str): tensor (64, 1024)},
      "masks":  {query (str): tensor (64,)}
    }
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / config (mirrors train.py CONFIG)
# ---------------------------------------------------------------------------

REPO_ROOT      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH       = "FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv"
IMAGE_ROOT     = "FG-datset/CUB_200_2011/images"
DINO_CACHE_OUT = "FG-datset/CUB_200_2011/dino_feat_cache.pt"
TEXT_CACHE_OUT = "FG-datset/CUB_200_2011/text_feat_cache.pt"

DINO_CFG       = "projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto_dinov3"
DINO_CKPT      = "checkpoints/dinov3/epoch_20-step_155206.ckpt"
ROBERTA_MODEL  = "roberta-large"
MAX_TEXT_LEN   = 64
BATCH_SIZE     = 64

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Image helper dataset
# ---------------------------------------------------------------------------

class _UniqueImageDataset(Dataset):
    """Yields (image_name, image_tensor) for every unique image in the CSV."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

    def __init__(self, image_names: list[str], image_root: str):
        self.names = image_names
        self.root  = image_root

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img  = Image.open(os.path.join(self.root, name)).convert("RGB")
        return name, self.transform(img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def precompute_dino(feature_extractor, image_names: list[str], device: torch.device) -> dict:
    """Run frozen ViT on all unique images; return {image_name: (196, d_vit)}."""
    ds     = _UniqueImageDataset(image_names, IMAGE_ROOT)
    loader = DataLoader(
        ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
        collate_fn  = lambda b: (
            [x[0] for x in b],
            torch.stack([x[1] for x in b]),
        ),
    )

    feat_cache = {}
    positions  = None

    feature_extractor = feature_extractor.to(device)

    print(f"  Computing ViT features for {len(image_names)} images …")
    with torch.no_grad():
        for names, imgs in tqdm(loader, unit="batch"):
            imgs = imgs.to(device)
            routing = {"input": {"image": imgs, "batch_size": imgs.shape[0]}}
            feat_out = feature_extractor(inputs=routing)

            feats = feat_out.features.cpu()           # (B, 200, d_vit) with DINOv3
            if positions is None:
                positions = feat_out.positions.cpu()  # (200, …) with DINOv3

            for name, f in zip(names, feats):
                feat_cache[name] = f                  # (200, d_vit) with DINOv3

    return {"features": feat_cache, "positions": positions}


def precompute_text(queries: list[str], device: torch.device) -> dict:
    """Run frozen RoBERTa-Large on all unique queries; return hidden states."""
    from transformers import RobertaModel, RobertaTokenizer

    print(f"  Computing RoBERTa features for {len(queries)} unique queries …")
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    roberta   = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device).eval()
    for p in roberta.parameters():
        p.requires_grad_(False)

    hidden_cache = {}
    mask_cache   = {}

    with torch.no_grad():
        for q in tqdm(queries, unit="query"):
            enc = tokenizer(
                q,
                max_length  = MAX_TEXT_LEN,
                padding     = "max_length",
                truncation  = True,
                return_tensors = "pt",
            )
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out  = roberta(input_ids=ids, attention_mask=mask)
            hidden_cache[q] = out.last_hidden_state.squeeze(0).cpu()  # (L, 1024)
            mask_cache[q]   = mask.squeeze(0).cpu()                   # (L,)

    return {"hidden": hidden_cache, "masks": mask_cache}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}. Run from repo root.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    unique_images  = df["image_name"].unique().tolist()
    unique_queries = df["query"].unique().tolist()
    print(f"Unique images: {len(unique_images)} | Unique queries: {len(unique_queries)}")

    # ── ViT features ─────────────────────────────────────────────────────
    if os.path.exists(DINO_CACHE_OUT):
        print(f"DINO cache already exists at {DINO_CACHE_OUT} — skipping.")
    else:
        from classifier_model import _load_dinosaur_submodules
        print("Loading DINOSAUR feature extractor …")
        fe, _, _ = _load_dinosaur_submodules(DINO_CFG, DINO_CKPT, REPO_ROOT, n_slots=7)
        fe.eval()

        dino_cache = precompute_dino(fe, unique_images, device)
        torch.save(dino_cache, DINO_CACHE_OUT)
        print(f"  Saved → {DINO_CACHE_OUT}  "
              f"({len(dino_cache['features'])} entries, "
              f"feat shape: {next(iter(dino_cache['features'].values())).shape})")

    # ── RoBERTa features ──────────────────────────────────────────────────
    if os.path.exists(TEXT_CACHE_OUT):
        print(f"Text cache already exists at {TEXT_CACHE_OUT} — skipping.")
    else:
        text_cache = precompute_text(unique_queries, device)
        torch.save(text_cache, TEXT_CACHE_OUT)
        print(f"  Saved → {TEXT_CACHE_OUT}  ({len(text_cache['hidden'])} queries)")

    print("Done.")


if __name__ == "__main__":
    main()
