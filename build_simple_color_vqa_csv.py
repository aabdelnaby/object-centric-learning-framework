"""Build a simplified part-color VQA CSV for Super-CLEVR-3D.

Goal (per the overfit-diagnosis plan): replace the noisy
`parts_vqa.csv` with a clean dataset of questions of the form
    "What is the color of <part> of <object>?"
where <object> is a super-category (car, bus, motorbike, aeroplane,
bicycle) and the scene contains *exactly one* object of that super-
category. Nothing else (texture, material, sub-type, spatial
relations beyond what's baked into the part name) is referenced.

The CSV schema matches `parts_vqa.csv` so the existing
`SuperCLEVR3DCachedFeatDataset`, `_filter_df`, and trainer work
unchanged.

Run from repo root:
    conda run -n oclf_env python build_simple_color_vqa_csv.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# Fine-stem -> super category, from Super-CLEVR's metadata_part.json:
#   Super-CLEVR/question_generation/metadata_part.json["types"]["Shape"]
SHAPE_TO_SUPER = {
    # car
    "suv": "car", "wagon": "car", "minivan": "car",
    "sedan": "car", "truck": "car", "addi": "car",
    # bus
    "articulated": "bus", "regular": "bus", "double": "bus", "school": "bus",
    # motorbike
    "chopper": "motorbike", "dirtbike": "motorbike",
    "scooter": "motorbike", "cruiser": "motorbike",
    # aeroplane
    "jet": "aeroplane", "fighter": "aeroplane",
    "biplane": "aeroplane", "airliner": "aeroplane",
    # bicycle
    "road": "bicycle", "utility": "bicycle",
    "mountain": "bicycle", "tandem": "bicycle",
}


def split_for(image_index: int) -> str:
    """Same image-index ranges as build_superclevr3d_vqa_csv.py:61."""
    if image_index < 20000:
        return "train"
    if image_index < 25000:
        return "val"
    return "test"


# Tokens that act as positional modifiers in part names. We always move
# these to the front so phrasing reads naturally: e.g. `wheel_front` →
# "front wheel", `crank_arm_left` → "left crank arm".
POSITION_WORDS = {
    "front", "back", "left", "right",
    "mid", "middle", "upper", "lower", "top", "bottom",
}


def clean(name: str) -> str:
    """Normalise a snake_case part name to natural English phrasing.

    Rules:
      1. Strip trailing '_s' tokens (opaque positional codes in the
         source data, e.g. `door_right_s` → `door_right`).
      2. Split into tokens. Move positional tokens (front/back/left/
         right/mid/upper/...) ahead of the rest, preserving each
         group's internal order.

    Examples:
      front_left_wheel → "front left wheel"
      wheel_front      → "front wheel"
      crank_arm_left   → "left crank arm"
      door_back_right  → "back right door"
      bumper_back      → "back bumper"
      back_bumper      → "back bumper"   (collapses to the same string)
      fin              → "fin"
    """
    tokens = name.split("_")
    while tokens and tokens[-1] == "s":
        tokens = tokens[:-1]
    positions = [t for t in tokens if t in POSITION_WORDS]
    others    = [t for t in tokens if t not in POSITION_WORDS]
    return " ".join(positions + others)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes",     default="FG-datset/superclevr3d/scenes.json")
    ap.add_argument("--out_csv",    default="FG-datset/superclevr3d/simple_color_vqa.csv")
    ap.add_argument("--vocab_out",  default="label_vocab_superclevr3d_simple.json")
    ap.add_argument("--image_name_fmt", default="superCLEVR_new_{idx:06d}.png",
                    help="image_filename template; must match keys in dino_feat_cache.pt")
    ap.add_argument("--dino_cache", default="FG-datset/superclevr3d/dino_feat_cache.pt",
                    help="Drop rows whose image_name is not a key in this cache. "
                         "Pass '' or 'none' to skip this filter.")
    args = ap.parse_args()

    cache_imgs = None
    if args.dino_cache and args.dino_cache.lower() != "none":
        import torch
        print(f"Loading DINO cache keys from {args.dino_cache} (filter only — features not used) …")
        cache = torch.load(args.dino_cache, map_location="cpu", weights_only=False)
        cache_imgs = set(cache["features"].keys())
        print(f"  {len(cache_imgs):,} image keys in cache")

    print(f"Loading scenes from {args.scenes} …")
    with open(args.scenes) as f:
        scenes_data = json.load(f)
    scenes = scenes_data.get("scenes", scenes_data)
    print(f"  {len(scenes)} scenes")

    rows = []
    unknown_shapes = Counter()
    skipped_multi_instance = 0
    skipped_missing_cache  = 0
    kept_objects = 0

    for s in scenes:
        idx       = s["image_index"]
        img_name  = s.get("image_filename") or args.image_name_fmt.format(idx=idx)
        if cache_imgs is not None and img_name not in cache_imgs:
            skipped_missing_cache += 1
            continue
        objects   = s.get("objects", [])

        # Count instances at the SUPER-category level. The user's spec:
        # single instance of the referent (e.g. one car), regardless of
        # sub-type (sedan/truck/etc.).
        supers_in_scene = []
        for o in objects:
            sup = SHAPE_TO_SUPER.get(o["shape"])
            if sup is None:
                unknown_shapes[o["shape"]] += 1
                supers_in_scene.append(None)
            else:
                supers_in_scene.append(sup)
        super_counts = Counter(s for s in supers_in_scene if s is not None)

        for obj, sup in zip(objects, supers_in_scene):
            if sup is None:
                continue
            if super_counts[sup] != 1:
                skipped_multi_instance += 1
                continue
            kept_objects += 1

            for part_name, part_data in obj.get("parts", {}).items():
                color = part_data.get("color")
                if color is None:
                    continue
                query = f"What is the color of the {clean(part_name)} of the {sup}?"
                rows.append({
                    "image_index":        idx,
                    "image_name":         img_name,
                    "split":              split_for(idx),
                    "query":              query,
                    "label":              str(color),
                    "label_rank":         1,
                    "attribute_type":     "color",
                    "depth":              2,
                    "n_objects":          len(objects),
                    "n_program_nodes":    4,
                    "question_index":     len(rows),
                    "template_filename":  "simple_color_part",
                })

    df = pd.DataFrame(rows)
    print(f"\nGenerated {len(df):,} questions.")
    print(f"  Singleton-object kept: {kept_objects:,}")
    print(f"  Objects skipped (multi-instance super-cat): {skipped_multi_instance:,}")
    if cache_imgs is not None:
        print(f"  Scenes skipped (image not in DINO cache): {skipped_missing_cache}")
    if unknown_shapes:
        print(f"  WARNING: unknown shapes encountered: {dict(unknown_shapes)}")

    print("\nPer-split row counts:")
    print(df["split"].value_counts())

    print("\nPer-class balance (train):")
    train = df[df["split"] == "train"]
    for c, n in train["label"].value_counts().sort_index().items():
        print(f"  {c:<8} {n:6,}  ({100*n/len(train):.1f} %)")

    print(f"\nDistinct query strings: {df['query'].nunique():,}")
    print(f"  (compared to ~8,639 in old color/depth=2 parts_vqa.csv slice)")

    # Sanity: super-category balance in train
    print("\nPer-super-category row counts (train):")
    sup_counts = train["query"].str.extract(r"of the (\w+)\?$")[0].value_counts()
    for s, n in sup_counts.items():
        print(f"  {s:<10} {n:,}")

    # Vocab built on train labels only — matches build_label_vocab_superclevr3d.
    train_labels = sorted(train["label"].unique().tolist())
    vocab = {lbl: i for i, lbl in enumerate(train_labels)}
    print(f"\nTrain answer vocab ({len(vocab)} classes): {list(vocab.keys())}")

    val_oov  = (~df[df["split"] == "val" ]["label"].isin(vocab)).sum()
    test_oov = (~df[df["split"] == "test"]["label"].isin(vocab)).sum()
    print(f"OOV answers in val:  {val_oov}")
    print(f"OOV answers in test: {test_oov}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote CSV → {out_csv}")

    with open(args.vocab_out, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Wrote vocab → {args.vocab_out}")


if __name__ == "__main__":
    main()
