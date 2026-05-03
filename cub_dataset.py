"""CUB-200-2011 dataset for attribute classification with DINOSAUR + text queries.

Run from the repo root:
    conda run -n oclf_env python cub_dataset.py  # quick sanity check

Expected directory layout (actual paths — note FG-datset not data/):
    FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv
    FG-datset/CUB_200_2011/images/<class_folder>/<image>.jpg
"""

import json
import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import RobertaTokenizer


# ---------------------------------------------------------------------------
# Label vocabulary helpers
# ---------------------------------------------------------------------------

def build_label_vocab(df: pd.DataFrame) -> dict:
    """Build a deterministic label→index mapping.

    Only considers rows with label_rank == 1 and split == "train" within the
    provided DataFrame.  Pass a query-filtered DataFrame when training one
    classifier per attribute.

    Returns:
        dict mapping label string → integer index (sorted for reproducibility).
    """
    train_labels = df[(df["split"] == "train") & (df["label_rank"] == 1)]["label"].unique()
    return {label: idx for idx, label in enumerate(sorted(train_labels))}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CUBAttributeDataset(Dataset):
    """PyTorch Dataset for CUB-200-2011 attribute classification.

    Each item is ``(image_tensor, input_ids, attention_mask, label_idx)``.

    Image transforms match DINOSAUR's COCO ccrop eval pipeline:
        ToTensor → Resize(224, BICUBIC) → clamp(0,1) → CenterCrop(224)
        → Normalize(ImageNet mean/std)

    Args:
        csv_path:       Path to cub200_ranked_classification_dataset.csv.
        image_root:     Directory that contains the per-class sub-directories
                        referenced in the ``image_name`` column.
        split:          ``"train"`` or ``"test"``.
        label_vocab:    dict from :func:`build_label_vocab`.
        tokenizer:      Initialised ``RobertaTokenizer``.
        max_text_len:   Max token sequence length (padded/truncated to this).
        query_filter:   Optional string; if provided only rows whose ``query``
                        column matches exactly are retained.
    """

    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        split: str,
        label_vocab: dict,
        tokenizer: RobertaTokenizer,
        max_text_len: int = 64,
        query_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ):
        """
        Args:
            query_filter:    Exact match on the ``query`` column.
            category_filter: Case-insensitive substring match on the ``query``
                             column (e.g. ``"color"`` selects all color queries).
                             Ignored when ``query_filter`` is set.
        """
        super().__init__()
        self.image_root  = image_root
        self.label_vocab = label_vocab
        self.tokenizer   = tokenizer
        self.max_text_len = max_text_len

        df = pd.read_csv(csv_path)
        df = df[(df["split"] == split) & (df["label_rank"] == 1)]
        if query_filter is not None:
            df = df[df["query"] == query_filter]
        elif category_filter is not None:
            df = df[df["query"].str.contains(category_filter, case=False, na=False)]
        # Drop rows whose label never appeared in the training split's vocab
        # (can happen for test-only label values).
        df = df[df["label"].isin(label_vocab)]
        self.df = df.reset_index(drop=True)

        # Matches the ccrop preprocessing config used for coco_feat_rec_dino_base16
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),  # bicubic can overshoot
            transforms.CenterCrop(224),
            transforms.Normalize(mean=self.IMAGE_MEAN, std=self.IMAGE_STD),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # --- image ---
        img_path = os.path.join(self.image_root, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)          # (3, 224, 224)

        # --- text ---
        enc = self.tokenizer(
            row["query"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)       # (max_text_len,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (max_text_len,)

        label_idx = torch.tensor(self.label_vocab[row["label"]], dtype=torch.long)

        return image_tensor, input_ids, attention_mask, label_idx


# ---------------------------------------------------------------------------
# Cached-features dataset (fast n_slots sweep after precompute_features.py)
# ---------------------------------------------------------------------------

class CUBCachedFeatDataset(Dataset):
    """CUB dataset backed by pre-computed ViT + RoBERTa features.

    Each item is ``(dino_feat, text_hidden, attention_mask, label_idx)``
    where dino_feat is (200, vit_dim) for DINOv3 (4 register + 196 patch tokens)
    and text_hidden is (L, d_text).

    Args:
        csv_path:       Path to the CSV file.
        dino_cache_path: Path to ``dino_feat_cache.pt`` produced by
                         ``precompute_features.py``.
        text_cache_path: Path to ``text_feat_cache.pt`` produced by
                         ``precompute_features.py``.
        split:          ``"train"`` or ``"test"``.
        label_vocab:    dict from :func:`build_label_vocab`.
        query_filter:   Optional query string filter (same semantics as
                        :class:`CUBAttributeDataset`).
    """

    def __init__(
        self,
        csv_path: str,
        dino_cache_path: str,
        text_cache_path: str,
        split: str,
        label_vocab: dict,
        query_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ):
        """
        Args:
            query_filter:    Exact match on the ``query`` column.
            category_filter: Case-insensitive substring match on the ``query``
                             column (e.g. ``"color"`` selects all color queries).
                             Ignored when ``query_filter`` is set.
        """
        super().__init__()
        self.label_vocab = label_vocab

        df = pd.read_csv(csv_path)
        df = df[(df["split"] == split) & (df["label_rank"] == 1)]
        if query_filter is not None:
            df = df[df["query"] == query_filter]
        elif category_filter is not None:
            df = df[df["query"].str.contains(category_filter, case=False, na=False)]
        df = df[df["label"].isin(label_vocab)]
        self.df = df.reset_index(drop=True)

        dino_cache = torch.load(dino_cache_path, map_location="cpu")
        self.dino_feats    = dino_cache["features"]   # dict {image_name: (200, d_vit)} with DINOv3
        self.dino_positions = dino_cache["positions"] # (200, …) with DINOv3

        text_cache = torch.load(text_cache_path, map_location="cpu")
        self.text_hidden   = text_cache["hidden"]     # dict {query: (L, d_text)}
        self.attn_masks    = text_cache["masks"]      # dict {query: (L,)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dino_feat    = self.dino_feats[row["image_name"]]  # (200, d_vit) with DINOv3
        text_hidden  = self.text_hidden[row["query"]]      # (L, d_text)
        attn_mask    = self.attn_masks[row["query"]]       # (L,)
        label_idx    = torch.tensor(self.label_vocab[row["label"]], dtype=torch.long)
        return dino_feat, text_hidden, attn_mask, label_idx


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    CSV_PATH   = "FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv"
    IMAGE_ROOT = "FG-datset/CUB_200_2011/images"
    VOCAB_OUT  = "label_vocab.json"

    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}.  "
              f"Run this script from the repo root.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    vocab = build_label_vocab(df)
    print(f"Label vocab size: {len(vocab)} classes")

    with open(VOCAB_OUT, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved label vocab → {VOCAB_OUT}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    ds = CUBAttributeDataset(CSV_PATH, IMAGE_ROOT, "train", vocab, tokenizer)
    print(f"Train dataset: {len(ds)} samples")

    img, ids, mask, lbl = ds[0]
    print(f"  image:         {img.shape}   dtype={img.dtype}")
    print(f"  input_ids:     {ids.shape}   dtype={ids.dtype}")
    print(f"  attention_mask:{mask.shape}  dtype={mask.dtype}")
    print(f"  label_idx:     {lbl.item()}")
