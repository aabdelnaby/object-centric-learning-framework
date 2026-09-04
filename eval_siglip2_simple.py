"""Zero-shot contrastive eval on simple_color_vqa using SigLIP-2.

SigLIP-2 isn't generative — it scores (image, text) pairs. We turn the VQA
into multiple-choice retrieval:

   For each (image, "What is the color of the {part} of the {category}?"),
   build 8 candidate captions, one per color from the vocab:
       "the {part} of the {category} is {color}"
   Encode all 8 with SigLIP-2's text tower, dot against the image embedding,
   take argmax → predicted color. Compare to the CSV gold label.

This is the standard SigLIP-style zero-shot classification, just with the
"label" being a question-specific color caption rather than a generic class
name. Much faster than BLIP-2 (no generation; ~10 sec/1000 pairs on A100).

Why this is a useful sanity check: SigLIP-2 is a strong vision-language
alignment model. If it can't pick the right color from 8 even when the
candidate caption is literally the answer plugged into the question, the
task is harder than it looks for VLMs — likely because of referential
ambiguity ("the {category}" with multiple matching objects in scene) and
fine-grained part localisation.

Usage:
    python eval_siglip2_simple.py \\
        --csv FG-datset/superclevr3d/simple_color_vqa.csv \\
        --image_root FG-datset/superclevr3d/images \\
        --max_n_objects 5 --n_eval 3000 \\
        --out_dir runs/vlm_eval_simple_siglip2_base
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


COLOR_VOCAB = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]

QUESTION_RE_D2 = re.compile(r"What is the color of the (.+?) of the ([^?]+?)\??$", re.IGNORECASE)
QUESTION_RE_D1 = re.compile(r"What is the color of the ([^?]+?)\??$",                 re.IGNORECASE)


def build_candidate_caption(question: str, color: str) -> str:
    """Turn a depth-1 or depth-2 color VQA question into a declarative caption.

    - depth-2 ("What is the color of the {part} of the {category}?"):
        → "the {part} of the {category} is {color}"
    - depth-1 ("What is the color of the {subject}?"):
        → "the {subject} is {color}"
    """
    m = QUESTION_RE_D2.match(question.strip())
    if m:
        part, cat = m.group(1).strip(), m.group(2).strip()
        return f"the {part} of the {cat} is {color}"
    m = QUESTION_RE_D1.match(question.strip())
    if m:
        subj = m.group(1).strip()
        return f"the {subj} is {color}"
    return f"a photo where the answer to '{question.rstrip('?')}' is {color}"


def filter_df(df, attribute_filter, depth_filter, max_n_objects, split):
    df = df[df["split"] == split]
    if attribute_filter is not None:
        df = df[df["attribute_type"] == attribute_filter]
    if depth_filter is not None:
        df = df[df["depth"] == depth_filter]
    if max_n_objects is not None:
        df = df[df["n_objects"] <= max_n_objects]
    return df.reset_index(drop=True)


# ── Image dataset ────────────────────────────────────────────────────────────

class _ImgDS(Dataset):
    def __init__(self, names, image_root, transform):
        self.names = names
        self.image_root = image_root
        self.transform = transform
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(os.path.join(self.image_root, name)).convert("RGB")
        return name, self.transform(images=img, return_tensors="pt")["pixel_values"][0]


def encode_images(image_names, image_root, img_proc, model, device, batch_size, dtype):
    """Return {image_name: tensor(d_img,)} L2-normalised image features."""
    ds = _ImgDS(image_names, image_root, img_proc)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=(device.type == "cuda"))
    feats: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for names, pix in tqdm(loader, desc="SigLIP-2 image encode"):
            pix = pix.to(device, dtype, non_blocking=True)
            vec = model.get_image_features(pixel_values=pix)
            vec = torch.nn.functional.normalize(vec, dim=-1).detach().cpu().float()
            for n, v in zip(names, vec):
                feats[n] = v
    return feats


def encode_texts(captions: List[str], tokenizer, model, device, batch_size, dtype, max_len=64):
    """Return tensor (n_captions, d_text), L2-normalised."""
    out = []
    with torch.no_grad():
        for start in tqdm(range(0, len(captions), batch_size), desc="SigLIP-2 text encode"):
            chunk = captions[start:start + batch_size]
            enc = tokenizer(chunk, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            vec = model.get_text_features(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
            )
            vec = torch.nn.functional.normalize(vec, dim=-1).detach().cpu().float()
            out.append(vec)
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",              default="FG-datset/superclevr3d/simple_color_vqa.csv")
    ap.add_argument("--image_root",       default="FG-datset/superclevr3d/images")
    ap.add_argument("--model",            default="google/siglip2-base-patch16-224")
    ap.add_argument("--attribute_filter", default="color")
    ap.add_argument("--depth_filter",     type=int, default=2)
    ap.add_argument("--max_n_objects",    type=int, default=5)
    ap.add_argument("--split",            default="val", choices=["train","val","test"])
    ap.add_argument("--n_eval",           type=int, default=3000,
                    help="Number of (image, question) pairs to evaluate; 0 = full split.")
    ap.add_argument("--img_batch_size",   type=int, default=64)
    ap.add_argument("--txt_batch_size",   type=int, default=128)
    ap.add_argument("--seed",             type=int, default=0)
    ap.add_argument("--dtype",            default="fp16", choices=["fp32","fp16","bf16"])
    ap.add_argument("--out_dir",          required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    print(f"Device: {device}   dtype: {dtype}   model: {args.model}")

    # ── Filter + sample ─────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df["label"] = df["label"].astype(str)
    df = filter_df(df, args.attribute_filter, args.depth_filter,
                   args.max_n_objects, split=args.split)
    print(f"After filters — rows: {len(df):,}  unique queries: {df['query'].nunique()}  "
          f"unique images: {df['image_name'].nunique()}")
    if args.n_eval and args.n_eval < len(df):
        df = df.sample(n=args.n_eval, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(df):,} pairs (seed={args.seed})")
    print("Gold label distribution:", dict(Counter(df["label"]).most_common()))

    # ── Load SigLIP-2 (image_processor + fast tokenizer separately) ─────────
    print("Loading SigLIP-2 …")
    model     = AutoModel.from_pretrained(args.model, torch_dtype=dtype).eval().to(device)
    img_proc  = AutoImageProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    print(f"img_proc: {type(img_proc).__name__}   tokenizer: {type(tokenizer).__name__}")

    # ── Precompute text candidates per unique question ──────────────────────
    unique_queries = sorted(df["query"].unique().tolist())
    query_to_idx   = {q: i for i, q in enumerate(unique_queries)}
    captions = [build_candidate_caption(q, c) for q in unique_queries for c in COLOR_VOCAB]
    print(f"Unique queries: {len(unique_queries)}  captions to encode: {len(captions)}")
    txt_feats = encode_texts(captions, tokenizer, model, device,
                              args.txt_batch_size, dtype)
    # reshape to (n_queries, n_colors, d)
    txt_feats = txt_feats.view(len(unique_queries), len(COLOR_VOCAB), -1)
    print(f"Text features: {tuple(txt_feats.shape)}")

    # ── Precompute image features (unique images only) ──────────────────────
    unique_images = sorted(df["image_name"].unique().tolist())
    img_feats     = encode_images(unique_images, args.image_root, img_proc, model,
                                  device, args.img_batch_size, dtype)
    print(f"Encoded {len(img_feats)} unique images")

    # ── Score each row ──────────────────────────────────────────────────────
    rows = []
    n_correct = 0
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="scoring"):
        img_vec = img_feats[row["image_name"]]                 # (d,)
        cand    = txt_feats[query_to_idx[row["query"]]]        # (n_colors, d)
        sims    = (cand @ img_vec).tolist()                    # (n_colors,)
        pred_i  = int(torch.tensor(sims).argmax().item())
        pred    = COLOR_VOCAB[pred_i]
        gold    = row["label"].lower()
        ok      = (pred == gold)
        n_correct += int(ok)
        rows.append({
            "image_name": row["image_name"],
            "query":      row["query"],
            "gold":       gold,
            "pred":       pred,
            "correct":    int(ok),
            "n_objects":  int(row["n_objects"]),
            **{f"sim_{c}": s for c, s in zip(COLOR_VOCAB, sims)},
        })

    elapsed = time.time() - t0
    print(f"\nScored {len(rows)} pairs in {elapsed:.1f} s "
          f"({len(rows)/max(elapsed, 1e-9):.1f} pairs/s)")

    # ── Aggregate + save ────────────────────────────────────────────────────
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / "predictions.csv", index=False)

    overall   = res_df["correct"].mean()
    per_gold  = res_df.groupby("gold").agg(
        n=("correct", "size"), acc=("correct", "mean"),
    ).round(4)
    pred_dist = res_df["pred"].value_counts().to_dict()
    gold_dist = res_df["gold"].value_counts().to_dict()
    confusion = (
        res_df.pivot_table(index="gold", columns="pred",
                           values="image_name", aggfunc="count", fill_value=0)
    )

    summary = {
        "model":        args.model,
        "n_pairs":      len(res_df),
        "overall_acc":  float(overall),
        "elapsed_sec":  float(elapsed),
        "gold_dist":    gold_dist,
        "pred_dist":    pred_dist,
        "per_gold":     per_gold.to_dict(orient="index"),
        "config":       vars(args),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOverall acc: {overall:.4f}  (chance = {1/len(COLOR_VOCAB):.4f})")
    print(f"\nGold dist : {gold_dist}")
    print(f"Pred dist : {pred_dist}")
    print(f"\nPer-gold accuracy:\n{per_gold.to_string()}")
    print(f"\nConfusion (rows=gold, cols=pred):\n{confusion.to_string()}")
    print(f"\nSummary  → {out_dir/'summary.json'}")
    print(f"Predictions → {out_dir/'predictions.csv'}")


if __name__ == "__main__":
    main()
