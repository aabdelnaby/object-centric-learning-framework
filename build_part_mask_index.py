"""Build a mask-index sidecar for ``parts_color_vqa.csv``.

The VQA CSV doesn't store which annotation entity each question refers to, so it
can't be used on its own to locate a part's segmentation mask. This script re-runs
the *exact* same selection as ``build_ade20k_color_vqa_csv.py`` (reusing
``iter_question_parts``) and the *exact* same image iteration / global
``question_index`` counter, emitting one row per question with the path to the
part's ``instance_mask`` PNG.

Because the iteration order is identical, ``question_index`` here aligns 1:1 with
``parts_color_vqa.csv``. The script asserts that the row count and the
(image_index, query) of each row match the VQA CSV.

Output columns:
    question_index, image_index, image_name, split,
    object_raw_name, part_raw_name, geom_label, json_path, part_mask_path

Run from repo root:
    conda run -n oclf_env python build_part_mask_index.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import pandas as pd

import build_ade20k_color_vqa_csv as B  # reuse iter_question_parts, helpers, _load_json

ca = B.ca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="FG-datset/ade20k/ADE20K_2021_17_01/index_ade20k.pkl")
    ap.add_argument("--base-dir", default=None,
                    help="Dir the index 'folder' paths are relative to. "
                         "Default: parent of the index file's directory.")
    ap.add_argument("--vqa-csv", default="FG-datset/ade20k/parts_color_vqa.csv",
                    help="The VQA CSV to align against (for verification).")
    ap.add_argument("--out-csv", default="FG-datset/ade20k/parts_color_vqa_masks.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.index, "rb") as f:
        index = pickle.load(f)
    filenames, folders = index["filename"], index["folder"]
    base_dir = args.base_dir or os.path.dirname(os.path.dirname(os.path.abspath(args.index)))
    print(f"base_dir = {base_dir}")
    print(f"{len(filenames)} images in index")

    pairs = list(enumerate(zip(filenames, folders)))
    if args.limit:
        pairs = pairs[:args.limit]

    rows = []
    for image_index, (fn, folder) in pairs:
        json_path = os.path.join(base_dir, folder, fn).replace(".jpg", ".json")
        if not os.path.exists(json_path):
            continue
        try:
            objects = ca._load_json(json_path)["annotation"]["object"]
        except Exception:
            continue
        img_name = B.relative_image_name(folder, fn)
        split = B.split_for(folder)
        json_dir = os.path.dirname(json_path)

        for obj, part, color in B.iter_question_parts(objects):
            mask_rel = part.get("instance_mask")
            mask_path = os.path.join(json_dir, mask_rel) if mask_rel else ""
            rows.append({
                "question_index":  len(rows),
                "image_index":     image_index,
                "image_name":      img_name,
                "split":           split,
                "object_raw_name": (obj.get("raw_name") or "").strip(),
                "part_raw_name":   (part.get("raw_name") or "").strip(),
                "geom_label":      color,
                "json_path":       json_path,
                "part_mask_path":  mask_path,
            })

    df = pd.DataFrame(rows)
    print(f"\nGenerated {len(df):,} mask-index rows.")

    # ---- Verify alignment with the VQA CSV ----------------------------------
    if os.path.exists(args.vqa_csv) and not args.limit:
        vqa = pd.read_csv(args.vqa_csv)
        assert len(df) == len(vqa), f"row count mismatch: masks {len(df)} vs vqa {len(vqa)}"
        # question_index columns must be identical and contiguous
        assert (df["question_index"].values == vqa["question_index"].values).all(), \
            "question_index misaligned"
        assert (df["image_index"].values == vqa["image_index"].values).all(), \
            "image_index misaligned"
        # the part/object names must reconstruct the VQA query verbatim
        rebuilt = ("What is the color of the " + df["part_raw_name"]
                   + " of the " + df["object_raw_name"] + "?")
        mism = (rebuilt.values != vqa["query"].values)
        assert mism.sum() == 0, f"{mism.sum()} queries do not reconstruct"
        assert (df["geom_label"].values == vqa["label"].values).all(), "label mismatch"
        print("Alignment with VQA CSV verified: row count, question_index, "
              "image_index, query, and label all match.")

    # ---- Mask file existence check ------------------------------------------
    n_missing_path = (df["part_mask_path"] == "").sum()
    sample = df.head(2000)
    n_missing_file = sum(not os.path.exists(p) for p in sample["part_mask_path"] if p)
    print(f"Rows with no instance_mask field: {n_missing_path}")
    print(f"Missing mask files (first 2000 sampled): {n_missing_file}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote → {args.out_csv}")


if __name__ == "__main__":
    main()
