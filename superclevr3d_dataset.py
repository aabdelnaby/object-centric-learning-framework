"""Super-CLEVR-3D part-VQA dataset, sibling to cub_dataset.CUBAttributeDataset.

Returns the same tuple shape so train.py and classifier_model.SlotClassifier
work unchanged.

Item format:
    (image_tensor (3, 224, 224), input_ids (L,), attention_mask (L,), label_idx)

Expected CSV schema (produced by build_superclevr3d_vqa_csv.py):
    image_name, split, query, label, label_rank=1,
    depth, n_objects, attribute_type, image_index, ...
"""
from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import RobertaTokenizer


def build_label_vocab_superclevr3d(df: pd.DataFrame) -> dict:
    """Sorted train-answer -> int vocab. Mirrors cub_dataset.build_label_vocab."""
    train_labels = df[df["split"] == "train"]["label"].unique()
    return {lbl: i for i, lbl in enumerate(sorted(map(str, train_labels)))}


def _filter_df(
    df: pd.DataFrame,
    split: str,
    label_vocab: dict,
    query_filter: Optional[str],
    category_filter: Optional[str],
    attribute_filter: Optional[str],
    depth_filter: Optional[int],
    max_n_objects: Optional[int] = None,
    object_category: Optional[str] = None,
) -> pd.DataFrame:
    df = df[df["split"] == split]
    if query_filter is not None:
        df = df[df["query"] == query_filter]
    elif category_filter is not None:
        df = df[df["query"].str.contains(category_filter, case=False, na=False)]
    if attribute_filter is not None:
        df = df[df["attribute_type"] == attribute_filter]
    if depth_filter is not None:
        df = df[df["depth"] == depth_filter]
    if max_n_objects is not None and "n_objects" in df.columns:
        df = df[df["n_objects"] <= max_n_objects]
    if object_category is not None:
        # Whole-word match so "car" doesn't pick up "carrier" / "cargo".
        pat = rf"\b{re.escape(object_category)}\b"
        df = df[df["query"].str.contains(pat, case=False, na=False, regex=True)]
    df = df[df["label"].astype(str).isin(label_vocab)]
    return df.reset_index(drop=True)


class SuperCLEVR3DAttributeDataset(Dataset):
    """Drop-in sibling of cub_dataset.CUBAttributeDataset for Super-CLEVR-3D."""

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
        attribute_filter: Optional[str] = None,
        depth_filter: Optional[int] = None,
        max_n_objects: Optional[int] = None,
        augment: bool = False,
        img_size: int = 224,
        object_category: Optional[str] = None,
    ):
        super().__init__()
        self.image_root = image_root
        self.label_vocab = label_vocab
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.img_size = img_size

        df = pd.read_csv(csv_path)
        df["label"] = df["label"].astype(str)
        self.df = _filter_df(
            df, split, label_vocab,
            query_filter, category_filter, attribute_filter, depth_filter,
            max_n_objects=max_n_objects,
            object_category=object_category,
        )

        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.85, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.Normalize(mean=self.IMAGE_MEAN, std=self.IMAGE_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
                transforms.CenterCrop(img_size),
                transforms.Normalize(mean=self.IMAGE_MEAN, std=self.IMAGE_STD),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        enc = self.tokenizer(
            row["query"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label_idx      = torch.tensor(self.label_vocab[row["label"]], dtype=torch.long)
        return image_tensor, input_ids, attention_mask, label_idx


class SuperCLEVR3DCachedFeatDataset(Dataset):
    """Cached-features sibling of cub_dataset.CUBCachedFeatDataset."""

    def __init__(
        self,
        csv_path: str,
        dino_cache_path: str,
        text_cache_path: str,
        split: str,
        label_vocab: dict,
        query_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        attribute_filter: Optional[str] = None,
        depth_filter: Optional[int] = None,
        max_n_objects: Optional[int] = None,
        text_onehot: bool = False,
        question_vocab: Optional[dict] = None,
        object_category: Optional[str] = None,
        text_cache: Optional[dict] = None,
        dino_cache: Optional[dict] = None,
        return_spans: bool = False,
    ):
        """If ``text_onehot=True``, the dataset returns ``(dino_feat, q_id,
        attn_mask_of_ones(1), label)`` where ``q_id`` is the integer index of
        ``row["query"]`` in ``question_vocab``. The RoBERTa text cache is not
        loaded in this mode — the model is expected to use an ``nn.Embedding``
        keyed by ``q_id`` instead of a language encoder.
        """
        super().__init__()
        self.label_vocab    = label_vocab
        self.text_onehot    = text_onehot
        self.question_vocab = question_vocab
        self.return_spans   = return_spans

        df = pd.read_csv(csv_path)
        df["label"] = df["label"].astype(str)
        self.df = _filter_df(
            df, split, label_vocab,
            query_filter, category_filter, attribute_filter, depth_filter,
            max_n_objects=max_n_objects,
            object_category=object_category,
        )

        if text_onehot:
            if question_vocab is None:
                raise ValueError("text_onehot=True requires a question_vocab")
            # Drop rows whose query isn't in the vocab (OOV for this split).
            in_vocab = self.df["query"].isin(question_vocab)
            n_drop = int((~in_vocab).sum())
            if n_drop:
                print(
                    f"  [text_onehot] dropping {n_drop} {split} rows with OOV queries "
                    f"(vocab size={len(question_vocab)})"
                )
            self.df = self.df[in_vocab].reset_index(drop=True)

        # dino_cache may be supplied pre-built (computed in RAM at run start, no
        # disk round-trip — see train.py --dino_in_memory); otherwise load it.
        if dino_cache is None:
            dino_cache = torch.load(dino_cache_path, map_location="cpu")
        self.dino_feats     = dino_cache["features"]
        self.dino_positions = dino_cache.get("positions")

        if text_onehot:
            self.text_hidden = None
            self.attn_masks  = None
        else:
            # text_cache may be supplied pre-built (computed in RAM at run start,
            # no disk round-trip — see train.py --text_in_memory); else load it.
            if text_cache is None:
                text_cache = torch.load(text_cache_path, map_location="cpu")
            self.text_hidden = text_cache["hidden"]
            self.attn_masks  = text_cache["masks"]
            if return_spans:
                if "x_vec" not in text_cache or "y_vec" not in text_cache:
                    raise ValueError(
                        "return_spans=True but the text cache has no x_vec/y_vec. Rebuild "
                        "it with precompute_text(..., with_spans=True) / "
                        "precompute_features.py --with_spans (needed by pooler='hier_router')."
                    )
                self.x_vec = text_cache["x_vec"]
                self.y_vec = text_cache["y_vec"]
                # Combined "<x> of the <y>" phrase vec (parent-only router query); old
                # caches built before this existed fall back to the object vec (channel
                # 2 is ignored by the full parent→child model anyway).
                self.xy_vec = text_cache.get("xy_vec", self.y_vec)
                # "<y> <x>" compound-noun vec (e.g. "car door") for the query-conditioned
                # colour readout. Old caches fall back to the part vec; channel 3 is only
                # consumed when the readout head is enabled (needs a fresh cache anyway).
                self.readout_vec = text_cache.get("readout_vec", self.x_vec)

        if return_spans and text_onehot:
            raise ValueError("return_spans is incompatible with text_onehot.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dino_feat   = self.dino_feats[row["image_name"]]
        label_idx   = torch.tensor(self.label_vocab[row["label"]], dtype=torch.long)
        if self.text_onehot:
            # Length-1 "text sequence": just the question identity.
            q_id      = torch.tensor(self.question_vocab[row["query"]], dtype=torch.long)
            attn_mask = torch.ones(1, dtype=torch.long)
            return dino_feat, q_id, attn_mask, label_idx
        text_hidden = self.text_hidden[row["query"]]
        attn_mask   = self.attn_masks[row["query"]]
        if self.return_spans:
            # (4, d_text): ch0 = part <x>, ch1 = object <y>, ch2 = "<x> of the <y>"
            # (parent-only router query), ch3 = "<y> <x>" compound noun (colour readout).
            q = row["query"]
            spans = torch.stack(
                [self.x_vec[q], self.y_vec[q], self.xy_vec[q], self.readout_vec[q]], dim=0,
            )
            return dino_feat, text_hidden, attn_mask, label_idx, spans
        return dino_feat, text_hidden, attn_mask, label_idx


if __name__ == "__main__":
    import json, sys
    CSV = "FG-datset/superclevr3d/parts_vqa.csv"
    IMG = "FG-datset/superclevr3d/images"
    if not os.path.exists(CSV):
        print(f"CSV missing at {CSV}. Run build_superclevr3d_vqa_csv.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(CSV, nrows=10000)
    df["label"] = df["label"].astype(str)
    vocab = build_label_vocab_superclevr3d(df)
    print(f"Smoke-test vocab size on first 10k rows: {len(vocab)}")
    tok = RobertaTokenizer.from_pretrained("roberta-large")
    ds = SuperCLEVR3DAttributeDataset(CSV, IMG, "train", vocab, tok)
    print(f"Train rows visible to dataset: {len(ds)}")
    if len(ds):
        img, ids, mask, lbl = ds[0]
        print(f"  image: {tuple(img.shape)}  ids: {tuple(ids.shape)}  label: {lbl.item()}")
