"""Zero-shot VLM evaluation on simple_color_vqa (Super-CLEVR-3D).

Default model: ``Salesforce/blip2-flan-t5-xl`` — BLIP-2 with a FlanT5-XL
decoder. Ungated on Hugging Face, standard transformers API, decent at
short factoid VQA (color words), and small enough to run a few thousand
val pairs in <25 min on an A100.

This is a diagnostic baseline: how well does a strong off-the-shelf VLM
do on the *simple* slice (color, depth=2, ≤5 objects) that our slot model
has been overfitting on? If the VLM is near-ceiling, the data + labels
are sound and our slot model's val-acc collapse is an architecture or
training problem. If the VLM also struggles, the gold labels / question
generation deserve a closer look.

Usage:
    python eval_vlm_simple.py \\
        --csv FG-datset/superclevr3d/simple_color_vqa.csv \\
        --image_root FG-datset/superclevr3d/images \\
        --max_n_objects 5 \\
        --n_eval 3000 \\
        --out_dir runs/vlm_eval_simple_blip2_flant5xl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor


# The full color vocabulary in simple_color_vqa.csv. Used for parsing the
# generated answer back to a label and for per-class accuracy reporting.
COLOR_VOCAB = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]


def filter_df(df, attribute_filter, depth_filter, max_n_objects, split="val"):
    df = df[df["split"] == split]
    if attribute_filter is not None:
        df = df[df["attribute_type"] == attribute_filter]
    if depth_filter is not None:
        df = df[df["depth"] == depth_filter]
    if max_n_objects is not None:
        df = df[df["n_objects"] <= max_n_objects]
    return df.reset_index(drop=True)


def parse_pred(gen_text: str) -> str:
    """Pick the first color-vocab word that appears in the generated string.

    BLIP-2 / FlanT5 typically answers with a single word (e.g. ``red``), but
    sometimes drops a stray ``the`` or punctuation. We lowercase, strip
    non-alphabetic chars, and return the first known color we see — or
    ``"<unparsed>"`` if none.
    """
    s = re.sub(r"[^a-z ]", " ", gen_text.lower()).split()
    for tok in s:
        if tok in COLOR_VOCAB:
            return tok
    return "<unparsed>"


def format_prompt(question: str) -> str:
    # BLIP-2 FlanT5 convention: terse "Question: ... Answer:" prompt.
    return f"Question: {question} Answer:"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",              default="FG-datset/superclevr3d/simple_color_vqa.csv")
    p.add_argument("--image_root",       default="FG-datset/superclevr3d/images")
    p.add_argument("--model",            default="Salesforce/blip2-flan-t5-xl")
    p.add_argument("--attribute_filter", default="color")
    p.add_argument("--depth_filter",     type=int, default=2)
    p.add_argument("--max_n_objects",    type=int, default=5)
    p.add_argument("--split",            default="val", choices=["train", "val", "test"])
    p.add_argument("--n_eval",           type=int, default=3000,
                   help="Number of (image, question) pairs to evaluate; "
                        "0 = full split.")
    p.add_argument("--batch_size",       type=int, default=8)
    p.add_argument("--max_new_tokens",   type=int, default=8)
    p.add_argument("--seed",             type=int, default=0)
    p.add_argument("--dtype",            default="fp16",
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--out_dir",          required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    print("Gold label distribution:",
          dict(Counter(df["label"]).most_common()))

    # ── Load VLM ────────────────────────────────────────────────────────────
    print("Loading processor + model …")
    processor = AutoProcessor.from_pretrained(args.model)
    model     = AutoModelForVision2Seq.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Total params: {n_params/1e9:.2f} B")

    # ── Inference loop ──────────────────────────────────────────────────────
    rows = []
    n_correct = 0
    n_parsed  = 0
    t0 = time.time()
    pbar = tqdm(range(0, len(df), args.batch_size), desc="VLM eval")
    for start in pbar:
        chunk     = df.iloc[start : start + args.batch_size]
        images    = [
            Image.open(os.path.join(args.image_root, name)).convert("RGB")
            for name in chunk["image_name"]
        ]
        prompts   = [format_prompt(q) for q in chunk["query"]]

        inputs = processor(images=images, text=prompts,
                           return_tensors="pt", padding=True).to(device, dtype)
        # text input ids must stay long
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        gens = processor.batch_decode(out_ids, skip_special_tokens=True)

        for (_, row), gen in zip(chunk.iterrows(), gens):
            gold = row["label"].lower()
            pred = parse_pred(gen)
            ok   = (pred == gold)
            n_correct += int(ok)
            n_parsed  += int(pred != "<unparsed>")
            rows.append({
                "image_name": row["image_name"],
                "query":      row["query"],
                "gold":       gold,
                "pred":       pred,
                "gen_raw":    gen.strip(),
                "correct":    int(ok),
                "n_objects":  int(row["n_objects"]),
            })

        # Live readout
        done = len(rows)
        pbar.set_postfix(acc=f"{n_correct/done:.4f}", parsed=f"{n_parsed/done:.3f}")

    elapsed = time.time() - t0
    print(f"\nDone — {len(rows)} pairs in {elapsed/60:.1f} min "
          f"({len(rows)/elapsed:.2f} pairs/s)")

    # ── Aggregate + save ────────────────────────────────────────────────────
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / "predictions.csv", index=False)

    overall = res_df["correct"].mean()
    parsed  = (res_df["pred"] != "<unparsed>").mean()
    per_gold = res_df.groupby("gold").agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
    ).round(4)
    confusion = (
        res_df.pivot_table(index="gold", columns="pred",
                           values="image_name", aggfunc="count", fill_value=0)
    )

    summary = {
        "model":          args.model,
        "n_pairs":        len(res_df),
        "overall_acc":    float(overall),
        "parsed_rate":    float(parsed),
        "elapsed_min":    float(elapsed / 60),
        "per_gold":       per_gold.to_dict(orient="index"),
        "config":         vars(args),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOverall acc : {overall:.4f}")
    print(f"Parse rate  : {parsed:.4f}  (fraction of generations containing a known color)")
    print(f"\nPer-gold accuracy:")
    print(per_gold.to_string())
    print(f"\nConfusion (rows = gold, cols = pred):")
    print(confusion.to_string())
    print(f"\nSummary  → {out_dir / 'summary.json'}")
    print(f"Predictions → {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
