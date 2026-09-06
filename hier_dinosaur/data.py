"""Question CSVs, label vocabularies and the cached-feature dataset.

Both datasets share one CSV schema (columns used: ``image_name, split, query, label`` and,
when present, ``label_rank``):

    paco   PACO-LVIS part-colour questions, splits train / val, 12 colour classes
    cub    CUB-200 part-attribute questions, splits train / test, filtered to the 16 colour
           queries (15 colour classes) with ``category_filter="color"``

Each dataset item is ``(dino_features (196, 384), spans (4, d_text), label_idx)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .text import DATASETS


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv: str
    dino_cache: str
    image_root: str
    train_split: str
    val_split: str
    category_filter: Optional[str]


PRESETS: Dict[str, DatasetSpec] = {
    "paco": DatasetSpec(
        name="paco",
        csv="data/paco/paco_questions.csv",
        dino_cache="data/caches/dino_paco_square224.pt",
        image_root="data/coco",
        train_split="train",
        val_split="val",
        category_filter=None,
    ),
    "cub": DatasetSpec(
        name="cub",
        csv="data/cub/cub200_questions.csv",
        dino_cache="data/caches/dino_cub_square224.pt",
        image_root="data/cub/CUB_200_2011/images",
        train_split="train",
        val_split="test",
        category_filter="color",
    ),
}


def load_frame(csv_path: str, category_filter: Optional[str] = None) -> pd.DataFrame:
    """Read a question CSV; labels become strings; optional case-insensitive query filter."""
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(str)
    if category_filter:
        df = df[df["query"].str.contains(category_filter, case=False, na=False)]
    return df.reset_index(drop=True)


def build_label_vocab(df: pd.DataFrame, train_split: str = "train") -> Dict[str, int]:
    """Sorted answer → index map over the training rows (rank-1 labels when the column exists)."""
    rows = df[df["split"] == train_split]
    if "label_rank" in rows.columns:
        rows = rows[rows["label_rank"] == 1]
    return {lbl: i for i, lbl in enumerate(sorted(rows["label"].astype(str).unique()))}


def split_frame(df: pd.DataFrame, split: str, label_vocab: Dict[str, int]) -> pd.DataFrame:
    """Rows of one split with a known label (and rank-1 labels when the column exists)."""
    rows = df[df["split"] == split]
    if "label_rank" in rows.columns:
        rows = rows[rows["label_rank"] == 1]
    rows = rows[rows["label"].isin(label_vocab)]
    return rows.reset_index(drop=True)


class CachedFeatureDataset(Dataset):
    """Rows of a split backed by cached patch features and a span table."""

    def __init__(self, rows: pd.DataFrame, dino_features: Dict[str, torch.Tensor],
                 spans: Dict[str, torch.Tensor], label_vocab: Dict[str, int]):
        self.df = rows.reset_index(drop=True)
        self.dino_features = dino_features
        self.spans = spans
        self.label_vocab = label_vocab
        missing = set(self.df["image_name"]) - set(dino_features)
        if missing:
            raise KeyError(f"{len(missing)} images of this split are missing from the feature cache, "
                           f"e.g. {sorted(missing)[:3]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = self.dino_features[row["image_name"]]
        spans = self.spans[row["query"]]
        label = torch.tensor(self.label_vocab[row["label"]], dtype=torch.long)
        return feat, spans, label


def check_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    return PRESETS[name]
