"""Annotate part crops with InternVL3-14B, forced to one of 11 basic colours (the VLM fallback label).

Reads ``paco_<split>_masks.csv`` and the crops from ``extract_part_crops.py``, shows each part-only
crop to InternVL3-14B and asks for the single dominant colour from

    black white gray red orange yellow green blue purple pink brown

The response is synonym-mapped onto that set (``unknown`` if unresolvable). Sharded for a SLURM
array: shard K handles the rows with ``question_index % num_shards == K`` and writes
``<out-dir>/predictions_shard_K.csv``; ``--skip-done`` resumes. Runs in the ``internvl_env``
environment (environment-internvl.yml) with the weights pre-downloaded (HF_HUB_OFFLINE=1 on nodes
without internet).

    python data_prep/paco/annotate_colors_internvl.py --split train --only-unknown --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

MODEL_PATH = "OpenGVLab/InternVL3-14B"

BASIC_COLORS = ["black", "white", "gray", "red", "orange", "yellow",
                "green", "blue", "purple", "pink", "brown"]

# Map any non-basic color word the VLM might emit down to a basic term.
SYNONYMS = {
    "grey": "gray", "silver": "gray", "charcoal": "gray", "slate": "gray",
    "beige": "brown", "tan": "brown", "cream": "brown", "ivory": "brown",
    "khaki": "brown", "gold": "brown", "golden": "brown", "bronze": "brown",
    "wood": "brown", "wooden": "brown", "chocolate": "brown", "coffee": "brown",
    "maroon": "red", "crimson": "red", "scarlet": "red", "burgundy": "red",
    "navy": "blue", "azure": "blue", "cyan": "blue", "turquoise": "blue",
    "teal": "green", "olive": "green", "lime": "green", "emerald": "green",
    "violet": "purple", "lavender": "purple", "indigo": "purple", "mauve": "purple",
    "magenta": "pink", "rose": "pink", "salmon": "pink", "coral": "pink",
    "amber": "orange", "peach": "orange",
    "off-white": "white", "offwhite": "white", "snow": "white",
}

PROMPT = (
    "<image>\n"
    "What is the single most dominant color of the object shown in this image? "
    "Answer with exactly one word, chosen ONLY from this list: "
    "black, white, gray, red, orange, yellow, green, blue, purple, pink, brown. "
    "Do not explain. Answer with the color word only."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_WORD_RE = re.compile(r"[a-z\-]+")


def parse_color(text: str) -> str:
    """Resolve a free-text VLM response to a basic color, or 'unknown'."""
    text = (text or "").strip().lower()
    words = _WORD_RE.findall(text)
    # 1) first token that is already a basic color
    for w in words:
        if w in BASIC_COLORS:
            return w
    # 2) first token that maps via synonyms
    for w in words:
        if w in SYNONYMS:
            return SYNONYMS[w]
    # 3) substring fallback (handles "brownish", "light-blue" etc.)
    for c in BASIC_COLORS:
        if c in text:
            return c
    for k, v in SYNONYMS.items():
        if k in text:
            return v
    return "unknown"


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_pixel_values(image_file, input_size=448, dtype=torch.bfloat16):
    """Single 448-tile preprocessing — our crops are already square & part-only."""
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0).to(dtype)


def load_model():
    from transformers import AutoModel, AutoTokenizer
    print(f"Loading {MODEL_PATH} …", flush=True)
    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                  trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(MODEL_PATH, use_flash_attn=True, **kwargs)
    except Exception as e:
        print(f"flash-attn unavailable ({e}); falling back to eager attention.", flush=True)
        model = AutoModel.from_pretrained(MODEL_PATH, use_flash_attn=False, **kwargs)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--data-dir", default="data/paco")
    ap.add_argument("--only-unknown", action="store_true", help="only parts without a PACO colour (thesis setting)")
    ap.add_argument("--crops-dir", default=None, help="default: <data-dir>/part_crops/<split>")
    ap.add_argument("--out-dir", default=None, help="default: <data-dir>/preds/<split>")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=50)
    args = ap.parse_args()
    args.crops_dir = args.crops_dir or os.path.join(args.data_dir, "part_crops", args.split)
    args.out_dir = args.out_dir or os.path.join(args.data_dir, "preds", args.split)

    df = pd.read_csv(os.path.join(args.data_dir, f"paco_{args.split}_masks.csv"))
    if args.only_unknown:
        df = df[df["geom_label"].fillna("unknown") == "unknown"]
    df = df[df["question_index"] % args.num_shards == args.shard].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"predictions_shard_{args.shard}.csv")

    done = set()
    if args.skip_done and os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = set(prev["question_index"].tolist())
        print(f"Resuming: {len(done)} rows already done in {out_path}", flush=True)

    todo = [r for r in df.itertuples(index=False) if r.question_index not in done]
    print(f"Shard {args.shard}/{args.num_shards}: {len(todo)} crops to annotate "
          f"({len(df)} total in shard).", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    model, tokenizer = load_model()
    gen_cfg = dict(max_new_tokens=8, do_sample=False)

    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["question_index", "vlm_color", "raw_response"])

    n_unknown = 0
    for i, r in enumerate(todo, 1):
        qid = int(r.question_index)
        crop_path = os.path.join(args.crops_dir, f"{qid}.jpg")
        if not os.path.exists(crop_path):
            writer.writerow([qid, "unknown", "MISSING_CROP"])
            n_unknown += 1
            continue
        try:
            pv = load_pixel_values(crop_path).cuda()
            with torch.no_grad():
                resp = model.chat(tokenizer, pv, PROMPT, gen_cfg)
        except Exception as e:
            writer.writerow([qid, "unknown", f"ERROR:{type(e).__name__}:{e}"])
            n_unknown += 1
            continue
        color = parse_color(resp)
        if color == "unknown":
            n_unknown += 1
        writer.writerow([qid, color, resp.strip().replace("\n", " ")[:80]])
        if i % args.flush_every == 0:
            f.flush()
            print(f"  [{args.shard}] {i}/{len(todo)}  unknown={n_unknown}", flush=True)

    f.flush()
    f.close()
    print(f"Shard {args.shard} done. {len(todo)} annotated, {n_unknown} unknown. "
          f"→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
