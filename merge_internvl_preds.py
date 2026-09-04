"""Merge InternVL3 shard predictions into the VQA CSV and report agreement.

Concatenates ``preds/predictions_shard_*.csv``, joins the ``vlm_color`` onto a copy
of ``parts_color_vqa.csv`` by ``question_index``, and prints:
  * coverage (rows annotated / unknown),
  * the VLM color histogram,
  * agreement between the VLM color and the geometric ``label`` after both are
    mapped to the same 11 basic colors (a sanity check on both annotators).

Run from repo root:
    conda run -n oclf_env python merge_internvl_preds.py
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

# Geometric 21-color palette -> 11 basic colors (same target set as the VLM prompt).
FINE_TO_BASIC = {
    "black": "black", "white": "white", "cream": "brown", "beige": "brown",
    "tan": "brown", "light brown": "brown", "brown": "brown", "dark brown": "brown",
    "gray": "gray", "light gray": "gray", "dark gray": "gray", "silver": "gray",
    "red": "red", "dark red": "red", "orange": "orange", "yellow": "yellow",
    "gold": "brown", "green": "green", "dark green": "green", "olive": "green",
    "blue": "blue", "light blue": "blue", "dark blue": "blue", "navy": "blue",
    "purple": "purple", "violet": "purple", "pink": "pink", "magenta": "pink",
}


def to_basic(label: str) -> str:
    label = str(label).strip().lower()
    if label in FINE_TO_BASIC:
        return FINE_TO_BASIC[label]
    # token fallback: e.g. "light brown" not in map -> match trailing color word
    for token in label.split()[::-1]:
        if token in FINE_TO_BASIC:
            return FINE_TO_BASIC[token]
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa-csv", default="FG-datset/ade20k/parts_color_vqa.csv")
    ap.add_argument("--preds-glob", default="FG-datset/ade20k/preds/predictions_shard_*.csv")
    ap.add_argument("--out-csv", default="FG-datset/ade20k/parts_color_vqa_internvl.csv")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.preds_glob))
    if not shards:
        raise SystemExit(f"No shard predictions matched {args.preds_glob}")
    print(f"Merging {len(shards)} shard files")
    preds = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    preds = preds.drop_duplicates("question_index", keep="last")
    print(f"  {len(preds):,} predictions")

    vqa = pd.read_csv(args.vqa_csv)
    merged = vqa.merge(preds[["question_index", "vlm_color", "raw_response"]],
                       on="question_index", how="left")

    n = len(merged)
    n_missing = merged["vlm_color"].isna().sum()
    n_unknown = (merged["vlm_color"] == "unknown").sum()
    print(f"\nRows: {n:,}  |  no prediction: {n_missing:,}  |  unknown: {n_unknown:,}")

    print("\nVLM color histogram:")
    for c, k in merged["vlm_color"].value_counts(dropna=False).items():
        print(f"  {str(c):<10} {k:6,}")

    ok = merged[merged["vlm_color"].notna() & (merged["vlm_color"] != "unknown")].copy()
    ok["geom_basic"] = ok["label"].map(to_basic)
    agree = (ok["geom_basic"] == ok["vlm_color"]).mean()
    print(f"\nGeometric-vs-VLM agreement (basic colors, n={len(ok):,}): {agree:.1%}")

    print("\nTop geometric→VLM disagreements:")
    dis = ok[ok["geom_basic"] != ok["vlm_color"]]
    pair = (dis["geom_basic"] + " → " + dis["vlm_color"]).value_counts().head(12)
    for p, k in pair.items():
        print(f"  {p:<22} {k:5,}")

    merged.to_csv(args.out_csv, index=False)
    print(f"\nWrote → {args.out_csv}")


if __name__ == "__main__":
    main()
