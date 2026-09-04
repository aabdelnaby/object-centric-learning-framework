"""Fold InternVL shard predictions into the PACO VQA CSV's ``label`` column.

The adapter (``build_paco_color_vqa_csv.py``) leaves ``label``/``raw_response`` blank
and puts PACO's human dominant-color (mapped to the 11 basic colors, ``unknown`` where
PACO annotated no color) in ``geom_label``; ``annotate_colors_internvl.py`` produces a
per-crop ``vlm_color``.

**PACO's human color is the PRIMARY answer.** ``label = geom_label`` (PACO) wherever
PACO has a color for the part, and falls back to ``vlm_color`` (the VLM) only where it
does not (``geom_label == 'unknown'``). The VLM raw text is kept in ``raw_response`` for
provenance, and the parsed ``vlm_color`` survives in ``preds/`` for analysis. Also
reports the label-source breakdown + PACO-vs-VLM agreement.

Run from repo root:
    conda run -n oclf_env python finalize_paco_csv.py --split val
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd

COLS = ["image_index", "image_name", "split", "query", "geom_label", "label_rank",
        "attribute_type", "depth", "n_objects", "n_program_nodes", "question_index",
        "template_filename", "label", "raw_response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--vqa-csv", default=None,
                    help="Default: FG-datset/coco_paco/parts_color_vqa_paco_<split>.csv")
    ap.add_argument("--preds-glob", default="FG-datset/coco_paco/preds/predictions_shard_*.csv")
    ap.add_argument("--out-csv", default=None, help="Default: overwrite --vqa-csv.")
    args = ap.parse_args()

    vqa_csv = args.vqa_csv or f"FG-datset/coco_paco/parts_color_vqa_paco_{args.split}.csv"
    out = args.out_csv or vqa_csv

    shards = sorted(glob.glob(args.preds_glob))
    if not shards:
        raise SystemExit(f"No shard predictions matched {args.preds_glob}")
    preds = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    preds = preds.drop_duplicates("question_index", keep="last")
    print(f"{len(shards)} shard files, {len(preds):,} predictions")

    vqa = pd.read_csv(vqa_csv).drop(columns=["label", "raw_response"], errors="ignore")
    merged = vqa.merge(preds[["question_index", "vlm_color", "raw_response"]],
                       on="question_index", how="left")
    n_missing = merged["vlm_color"].isna().sum()

    # PACO human color is the primary answer; VLM is the fallback only where PACO
    # has no color for the part (geom_label == 'unknown').
    paco = merged["geom_label"].fillna("unknown")
    vlm = merged["vlm_color"].fillna("unknown")
    merged["label"] = paco.where(paco != "unknown", vlm)
    n_from_paco = int((paco != "unknown").sum())
    n_from_vlm = int(((paco == "unknown") & (vlm != "unknown")).sum())
    merged = merged.drop(columns=["vlm_color"])[COLS]

    n = len(merged)
    n_unknown = (merged["label"] == "unknown").sum()
    print(f"rows={n:,}  no VLM prediction={n_missing:,}  label==unknown={n_unknown:,}")
    print(f"label source: PACO={n_from_paco:,} ({100*n_from_paco/n:.1f}%)  "
          f"VLM-fallback={n_from_vlm:,} ({100*n_from_vlm/n:.1f}%)")
    print("\nlabel (answer = PACO-primary, VLM-fallback) histogram:")
    for c, k in merged["label"].value_counts().items():
        print(f"  {c:<10} {k:6,}")

    both = (paco != "unknown") & (vlm != "unknown")
    if both.any():
        agree = (paco[both] == vlm[both]).mean()
        print(f"\nPACO-vs-VLM agreement where both known (n={int(both.sum()):,}): {agree:.1%}  "
              f"(secondary check; PACO is used as the answer regardless)")
        dis = both & (paco != vlm)
        pair = (paco[dis] + " (PACO) vs " + vlm[dis] + " (VLM)").value_counts().head(8)
        for p, k in pair.items():
            print(f"  {p:<30} {k:5,}")

    merged.to_csv(out, index=False)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
