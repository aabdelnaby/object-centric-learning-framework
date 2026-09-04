#!/usr/bin/env python
"""Zero-shot scene classification of the PACO/COCO images with CLIP (open_clip).

Goal: attach a scene label (kitchen, living room, bedroom, ...) to every unique
image referenced in FG-datset/paco_questions.csv so we can hold out one scene
type for an OOD experiment (hier_router vs baseline on an unseen scene).

Method
------
* CLIP zero-shot: build a text prototype per scene by averaging CLIP text
  embeddings over several prompt templates (prompt ensembling), then score each
  image by cosine similarity and softmax over scenes.
* Cross-check (offline, free): COCO captions_train2017.json. If any of the ~5
  human captions contains a room keyword we record that as `caption_scene`; the
  `agree` column flags whether CLIP and the captions point at the same scene.
  Captions are high-precision where a room is named, so this is a cheap audit
  signal -- not used to override CLIP.

Output: one row per unique image with the CLIP label, confidence, the top-3
scenes, the caption-derived scene, and an agreement flag.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, OrderedDict

import torch
from PIL import Image

# --------------------------------------------------------------------------- #
# Scene taxonomy. Keys are the canonical labels written to the CSV. Values are
# the noun phrases fed into the prompt templates (synonyms broaden recall and
# are merged into one prototype per canonical label). Edit freely -- a coarser
# taxonomy gives cleaner zero-shot separation; a finer one gives more held-out
# choices. "outdoor_*" classes catch the non-household COCO images.
# --------------------------------------------------------------------------- #
SCENES = OrderedDict([
    ("kitchen",        ["kitchen"]),
    ("dining_room",    ["dining room", "dining table setting"]),
    ("living_room",    ["living room", "lounge"]),
    ("bedroom",        ["bedroom"]),
    ("bathroom",       ["bathroom", "restroom"]),
    ("office",         ["office", "desk workspace", "study room"]),
    ("store",          ["store", "shop", "market", "supermarket"]),
    ("restaurant",     ["restaurant", "cafe", "bar interior"]),
    ("street",         ["street", "city sidewalk", "road"]),
    ("park_nature",    ["park", "garden", "forest", "field outdoors"]),
    ("sports",         ["sports field", "stadium", "gym", "tennis court"]),
    ("beach_water",    ["beach", "lake shore", "ocean waterfront"]),
    ("vehicle",        ["inside a vehicle", "car interior", "airplane cabin"]),
    ("workshop",       ["workshop", "garage", "tool shed"]),
])

PROMPT_TEMPLATES = [
    "a photo taken in a {}.",
    "a photo of a {}.",
    "this picture was taken in a {}.",
    "an indoor photo of a {}.",
    "a scene of a {}.",
]

# Keyword -> canonical scene, for the caption cross-check. First hit wins, so
# order more specific phrases before generic ones.
CAPTION_KEYWORDS = [
    ("kitchen", "kitchen"),
    ("dining room", "dining_room"),
    ("dining table", "dining_room"),
    ("living room", "living_room"),
    ("lounge", "living_room"),
    ("bedroom", "bedroom"),
    ("bathroom", "bathroom"),
    ("restroom", "bathroom"),
    ("toilet", "bathroom"),
    ("office", "office"),
    ("desk", "office"),
    ("restaurant", "restaurant"),
    ("cafe", "restaurant"),
    ("kitchenette", "kitchen"),
    ("store", "store"),
    ("shop", "store"),
    ("market", "store"),
    ("street", "street"),
    ("sidewalk", "street"),
    ("road", "street"),
    ("park", "park_nature"),
    ("garden", "park_nature"),
    ("field", "park_nature"),
    ("forest", "park_nature"),
    ("beach", "beach_water"),
    ("ocean", "beach_water"),
    ("lake", "beach_water"),
    ("stadium", "sports"),
    ("court", "sports"),
    ("gym", "sports"),
    ("garage", "workshop"),
    ("workshop", "workshop"),
]


def caption_scene(captions):
    """Return the first scene keyword found across an image's captions, or ''."""
    blob = " ".join(captions).lower()
    for kw, scene in CAPTION_KEYWORDS:
        if kw in blob:
            return scene
    return ""


def coco_id(path):
    m = re.search(r"(\d+)\.jpg$", path)
    return int(m.group(1)) if m else None


def load_unique_images(csv_path):
    """image_name -> set(splits), preserving first-seen order."""
    imgs = OrderedDict()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            name = r["image_name"]
            imgs.setdefault(name, set()).add(r.get("split", ""))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="FG-datset/paco_questions.csv")
    ap.add_argument("--out", default="FG-datset/paco_image_scenes.csv")
    ap.add_argument("--captions",
                    default="scripts/datasets/data/coco/annotations/captions_train2017.json")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="classify only the first N images (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import open_clip

    imgs = load_unique_images(args.csv)
    names = list(imgs.keys())
    if args.limit:
        names = names[: args.limit]
    print(f"[data] {len(names)} unique images "
          f"(of {sum(len(v) for v in imgs.values())} split-tags)", flush=True)

    # captions cross-check ---------------------------------------------------
    cap_by_img = {}
    if args.captions and os.path.exists(args.captions):
        caps = json.load(open(args.captions))
        for a in caps["annotations"]:
            cap_by_img.setdefault(a["image_id"], []).append(a["caption"].strip())
        print(f"[caps] loaded captions for {len(cap_by_img)} images", flush=True)
    else:
        print(f"[caps] WARNING: captions not found at {args.captions}", flush=True)

    # model ------------------------------------------------------------------
    print(f"[clip] loading {args.model} / {args.pretrained} on {args.device}", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device).eval()

    # text prototypes: one normalized vector per canonical scene -------------
    labels = list(SCENES.keys())
    with torch.no_grad():
        protos = []
        for lab in labels:
            prompts = [t.format(syn) for syn in SCENES[lab] for t in PROMPT_TEMPLATES]
            tok = tokenizer(prompts).to(args.device)
            tfeat = model.encode_text(tok)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            protos.append(tfeat.mean(dim=0))
        text_protos = torch.stack(protos)
        text_protos = text_protos / text_protos.norm(dim=-1, keepdim=True)
    print(f"[clip] built {len(labels)} scene prototypes", flush=True)

    # classify ---------------------------------------------------------------
    fields = ["image_name", "coco_id", "splits", "scene", "confidence",
              "top3", "caption_scene", "agree"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    counts, agree_n, agree_d = Counter(), 0, 0

    with open(args.out, "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fields)
        w.writeheader()
        for start in range(0, len(names), args.batch_size):
            batch_names = names[start: start + args.batch_size]
            tensors, ok = [], []
            for n in batch_names:
                try:
                    im = Image.open(n).convert("RGB")
                    tensors.append(preprocess(im))
                    ok.append(n)
                except Exception as e:  # missing/corrupt image
                    print(f"[skip] {n}: {e}", flush=True)
            if not tensors:
                continue
            x = torch.stack(tensors).to(args.device)
            with torch.no_grad():
                feat = model.encode_image(x)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                logits = (100.0 * feat @ text_protos.T)
                probs = logits.softmax(dim=-1).cpu()

            for i, n in enumerate(ok):
                p = probs[i]
                order = p.argsort(descending=True)
                top = order[0].item()
                scene = labels[top]
                conf = p[top].item()
                top3 = ";".join(f"{labels[j]}:{p[j]:.2f}" for j in order[:3].tolist())
                cid = coco_id(n)
                cap_sc = caption_scene(cap_by_img.get(cid, [])) if cap_by_img else ""
                if cap_sc:
                    agree_d += 1
                    agree_n += int(cap_sc == scene)
                counts[scene] += 1
                w.writerow({
                    "image_name": n,
                    "coco_id": cid,
                    "splits": "|".join(sorted(s for s in imgs[n] if s)),
                    "scene": scene,
                    "confidence": f"{conf:.4f}",
                    "top3": top3,
                    "caption_scene": cap_sc,
                    "agree": "" if not cap_sc else int(cap_sc == scene),
                })
            done = start + len(batch_names)
            print(f"[clip] {done}/{len(names)}", flush=True)

    # summary ----------------------------------------------------------------
    print("\n=== scene distribution (CLIP argmax) ===", flush=True)
    for sc, c in counts.most_common():
        print(f"  {sc:14s} {c:6d}  ({100*c/max(1,sum(counts.values())):.1f}%)", flush=True)
    if agree_d:
        print(f"\n[audit] caption named a room for {agree_d} images; "
              f"CLIP agreed on {agree_n} ({100*agree_n/agree_d:.1f}%)", flush=True)
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
