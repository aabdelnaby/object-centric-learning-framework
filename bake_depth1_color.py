"""Bake a depth-1 color VQA CSV from Super-CLEVR-3D scenes.json.

For every scene with ≤ MAX_N_OBJECTS objects, iterate its objects; for each
object whose top-level shape maps to a category that is *unique* in that
scene (i.e. the scene contains exactly one object of that category), emit:

    query  = "What is the color of the {category}?"
    label  = object's top-level color
    depth  = 1

Skipping ambiguous categories keeps the gold answer unambiguous from the
(image, question) pair alone — exactly the same uniqueness property the
depth-2 simple_color_vqa generator enforces.

Output schema matches simple_color_vqa.csv so the existing eval/train code
paths (filters, dataset loader, eval_siglip2_simple.py) work unchanged.
"""

import argparse
import json
from collections import Counter

import pandas as pd

DEFAULT_SCENES   = "FG-datset/superclevr3d/scenes.json"
DEFAULT_META     = "Super-CLEVR/question_generation/metadata_part.json"
DEFAULT_OUT      = "FG-datset/superclevr3d/depth1_color_synthesized.csv"
DEFAULT_SPLITSRC = "FG-datset/superclevr3d/simple_color_vqa.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes",       default=DEFAULT_SCENES)
    ap.add_argument("--metadata",     default=DEFAULT_META)
    ap.add_argument("--split_source", default=DEFAULT_SPLITSRC,
                    help="CSV to copy train/val/test splits from (joined by "
                         "image_name). scenes.json itself tags every scene "
                         "as split='new', so we inherit the post-split CSV "
                         "assignment to keep comparisons matched.")
    ap.add_argument("--max_n_objects", type=int, default=5,
                    help="Skip scenes with more objects than this.")
    ap.add_argument("--out",          default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.metadata) as f:
        md = json.load(f)
    TAX          = {k: set(v) for k, v in md["types"]["Shape"].items()}
    SHAPE_TO_CAT = {s: c for c, shapes in TAX.items() for s in shapes}

    with open(args.scenes) as f:
        scenes = json.load(f)["scenes"]
    print(f"Loaded {len(scenes):,} scenes")

    # image_name → split, inherited from the post-split source CSV.
    src = pd.read_csv(args.split_source, usecols=["image_name", "split"])
    img2split = dict(src.drop_duplicates("image_name").set_index("image_name")["split"])
    print(f"Loaded split mapping for {len(img2split):,} images "
          f"({dict(src['split'].value_counts())})")

    rows = []
    skipped_ambiguous_category = 0
    skipped_too_many_objects   = 0
    skipped_unknown_shape      = 0
    skipped_no_split           = 0
    for sc in scenes:
        if len(sc["objects"]) > args.max_n_objects:
            skipped_too_many_objects += 1
            continue
        split = img2split.get(sc["image_filename"])
        if split is None:
            skipped_no_split += 1
            continue
        cat_counts = Counter(
            SHAPE_TO_CAT.get(o["shape"]) for o in sc["objects"]
        )
        for o in sc["objects"]:
            cat = SHAPE_TO_CAT.get(o["shape"])
            if cat is None:
                skipped_unknown_shape += 1
                continue
            if cat_counts[cat] != 1:
                skipped_ambiguous_category += 1
                continue
            rows.append({
                "image_index":       sc["image_index"],
                "image_name":        sc["image_filename"],
                "split":             split,
                "query":             f"What is the color of the {cat}?",
                "label":             o["color"],
                "label_rank":        1,
                "attribute_type":    "color",
                "depth":             1,
                "n_objects":         len(sc["objects"]),
                "n_program_nodes":   2,
                "question_index":    0,
                "template_filename": "depth1_color_synthesized",
            })

    out_df = pd.DataFrame(rows)
    print(f"\nGenerated {len(out_df):,} depth-1 color rows.")
    print(f"  splits          : {out_df['split'].value_counts().to_dict()}")
    print(f"  unique queries  : {out_df['query'].nunique()}")
    print(f"  labels          : {out_df['label'].value_counts().to_dict()}")
    print(f"  skipped: ambiguous-category={skipped_ambiguous_category:,}  "
          f"too-many-objects={skipped_too_many_objects:,}  "
          f"unknown-shape={skipped_unknown_shape:,}  "
          f"no-split-source={skipped_no_split:,}")

    out_df.to_csv(args.out, index=False)
    print(f"\nWrote → {args.out}")


if __name__ == "__main__":
    main()
