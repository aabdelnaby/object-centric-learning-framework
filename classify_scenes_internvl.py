"""Classify the scene/setting of each PACO/COCO image with InternVL3-14B.

For every unique image in ``paco_questions.csv`` we show InternVL the full image
together with its human-written COCO captions ("description") and force it to pick
exactly one scene label from a fixed taxonomy (kitchen, living room, ...). The
captions resolve cases an object-centric crop is ambiguous about ("a small bed in
a blue room" -> bedroom), and the image overrides captions that omit the setting.

This is the higher-quality counterpart to ``classify_scenes_clip.py``; the two
agree-rate is a useful reliability check before holding a scene out for the OOD
experiment (hier_router vs baseline on an unseen scene).

Sharded for a SLURM job array: shard K processes every unique image whose index
satisfies ``idx % num_shards == K`` and writes ``preds_scene/scene_shard_K.csv``.
Resumable via ``--skip-done`` (skips image_names already in the shard CSV).

Run (single GPU):
    conda run -n internvl_env python classify_scenes_internvl.py --shard 0 --num-shards 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import OrderedDict

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

MODEL_PATH = "OpenGVLab/InternVL3-14B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Canonical scene labels (written to CSV) -> the human-readable phrase shown to
# the VLM in the category list. Keep in sync with classify_scenes_clip.py so the
# two methods are directly comparable.
SCENE_PHRASES = OrderedDict([
    ("kitchen",      "kitchen"),
    ("dining_room",  "dining room"),
    ("living_room",  "living room"),
    ("bedroom",      "bedroom"),
    ("bathroom",     "bathroom"),
    ("office",       "office"),
    ("store",        "store or shop"),
    ("restaurant",   "restaurant or cafe"),
    ("street",       "street"),
    ("park_nature",  "park or nature"),
    ("sports",       "sports venue"),
    ("beach_water",  "beach or water"),
    ("vehicle",      "vehicle interior"),
    ("workshop",     "workshop or garage"),
    ("other",        "other"),
])

# Phrase/keyword -> canonical label, for parsing the free-text answer. Longer,
# more specific phrases first so "dining room" wins over "room", etc.
CANON = [
    ("dining room", "dining_room"), ("dining", "dining_room"),
    ("living room", "living_room"), ("living", "living_room"), ("lounge", "living_room"),
    ("bedroom", "bedroom"), ("bed room", "bedroom"),
    ("bathroom", "bathroom"), ("restroom", "bathroom"), ("toilet", "bathroom"),
    ("kitchen", "kitchen"),
    ("office", "office"), ("workspace", "office"), ("study", "office"),
    ("restaurant", "restaurant"), ("cafe", "restaurant"), ("bar", "restaurant"), ("diner", "restaurant"),
    ("store", "store"), ("shop", "store"), ("market", "store"), ("supermarket", "store"),
    ("street", "street"), ("sidewalk", "street"), ("road", "street"), ("urban", "street"),
    ("park", "park_nature"), ("garden", "park_nature"), ("nature", "park_nature"),
    ("forest", "park_nature"), ("field", "park_nature"), ("outdoor", "park_nature"),
    ("sport", "sports"), ("stadium", "sports"), ("gym", "sports"), ("court", "sports"),
    ("beach", "beach_water"), ("water", "beach_water"), ("ocean", "beach_water"),
    ("lake", "beach_water"), ("pool", "beach_water"),
    ("vehicle", "vehicle"), ("car", "vehicle"), ("airplane", "vehicle"),
    ("cabin", "vehicle"), ("bus", "vehicle"), ("train", "vehicle"),
    ("workshop", "workshop"), ("garage", "workshop"), ("shed", "workshop"),
    ("other", "other"),
]


def parse_scene(text: str) -> str:
    """Resolve a free-text VLM answer to a canonical scene label, or 'unknown'."""
    t = (text or "").strip().lower()
    for phrase, lab in CANON:
        if phrase in t:
            return lab
    return "unknown"


def build_prompt(captions):
    cats = ", ".join(SCENE_PHRASES.values())
    desc = " ".join(captions).strip()
    desc_block = f'Human description of the image: "{desc}"\n' if desc else ""
    return (
        "<image>\n"
        + desc_block +
        "Look at the image and the description and decide WHERE this photo was taken "
        "(its scene or setting). Choose exactly ONE category from this list:\n"
        f"{cats}.\n"
        "Answer with only the category name, nothing else."
    )


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_pixel_values(image_file, input_size=448, dtype=torch.bfloat16):
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


def coco_id(path):
    m = re.search(r"(\d+)\.jpg$", path)
    return int(m.group(1)) if m else None


def load_unique_images(csv_path):
    imgs = OrderedDict()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            imgs.setdefault(r["image_name"], set()).add(r.get("split", ""))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="FG-datset/paco_questions.csv")
    ap.add_argument("--captions",
                    default="scripts/datasets/data/coco/annotations/captions_train2017.json")
    ap.add_argument("--out-dir", default="FG-datset/preds_scene")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-captions", action="store_true",
                    help="ablation: classify from the image only")
    ap.add_argument("--flush-every", type=int, default=50)
    args = ap.parse_args()

    imgs = load_unique_images(args.csv)
    names = list(imgs.keys())
    names = [n for i, n in enumerate(names) if i % args.num_shards == args.shard]
    if args.limit:
        names = names[: args.limit]

    cap_by_img = {}
    if not args.no_captions and os.path.exists(args.captions):
        caps = json.load(open(args.captions))
        for a in caps["annotations"]:
            cap_by_img.setdefault(a["image_id"], []).append(a["caption"].strip())
        print(f"[caps] loaded captions for {len(cap_by_img)} images", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"scene_shard_{args.shard}.csv")

    done = set()
    if args.skip_done and os.path.exists(out_path):
        with open(out_path, newline="") as f:
            done = {r["image_name"] for r in csv.DictReader(f)}
        print(f"Resuming: {len(done)} images already done in {out_path}", flush=True)

    todo = [n for n in names if n not in done]
    print(f"Shard {args.shard}/{args.num_shards}: {len(todo)} images to classify "
          f"({len(names)} in shard).", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    model, tokenizer = load_model()
    gen_cfg = dict(max_new_tokens=12, do_sample=False)

    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["image_name", "coco_id", "splits", "scene", "raw_response"])

    n_unknown = 0
    t0 = time.time()
    for i, n in enumerate(todo, 1):
        cid = coco_id(n)
        caps = cap_by_img.get(cid, []) if cap_by_img else []
        try:
            pv = load_pixel_values(n).cuda()
            prompt = build_prompt(caps)
            with torch.no_grad():
                resp = model.chat(tokenizer, pv, prompt, gen_cfg)
        except Exception as e:
            writer.writerow([n, cid, "|".join(sorted(s for s in imgs[n] if s)),
                             "unknown", f"ERROR:{type(e).__name__}:{e}"])
            n_unknown += 1
            continue
        scene = parse_scene(resp)
        if scene == "unknown":
            n_unknown += 1
        writer.writerow([n, cid, "|".join(sorted(s for s in imgs[n] if s)),
                         scene, resp.strip().replace("\n", " ")[:80]])
        if i % args.flush_every == 0:
            f.flush()
            rate = i / (time.time() - t0)
            print(f"  [{args.shard}] {i}/{len(todo)}  unknown={n_unknown}  "
                  f"{rate:.2f} img/s ({1/rate:.2f}s/img)", flush=True)

    f.flush()
    f.close()
    dt = time.time() - t0
    print(f"Shard {args.shard} done. {len(todo)} classified, {n_unknown} unknown "
          f"in {dt:.0f}s ({len(todo)/dt:.2f} img/s, {dt/len(todo):.2f}s/img). "
          f"→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
