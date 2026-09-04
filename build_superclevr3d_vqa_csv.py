"""Convert Super-CLEVR-3D part-question JSONs into a flat VQA CSV.

Run from repo root:
    conda run -n oclf_env python build_superclevr3d_vqa_csv.py

Inputs (from FG-datset/superclevr3d/):
    questions/superclevr_questions_parts.json
    scenes.json

Outputs:
    FG-datset/superclevr3d/parts_vqa.csv
        Columns: image_index, image_name, split, query, label, label_rank,
                 attribute_type, depth, n_objects, n_program_nodes,
                 question_index, template_filename
    label_vocab_superclevr3d_parts.json
        Sorted answer-string -> int mapping built on the train split.

Splits are derived from image_index using the official
Super-CLEVR-3D split (0-19999 train / 20000-24999 val / 25000-29997 test).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_DATA_ROOT = "FG-datset/superclevr3d"
QUESTIONS_PATH    = "questions/superclevr_questions_parts.json"
SCENES_PATH       = "scenes.json"

# Operations that bump reasoning depth (cross object<->part or visibility boundary).
DEPTH_OPS = {
    "object2part",
    "object2part_all",
    "part2object",
    "filter_occludee",
    "partfilter_occludee",
}

# Map terminal program op -> attribute_type tag.
QUERY_OP_TO_ATTRIBUTE = {
    "query_color":          "color",
    "query_material":       "material",
    "query_size":           "size",
    "query_shape":          "shape",
    "partquery_color":      "color",
    "partquery_material":   "material",
    "partquery_size":       "size",
    "partquery_partname":   "partname",
    "partquery_occlusion":  "occlusion",
    "query_pose":           "pose",
    "same_pose":            "pose_relation",
    "vertical_pose":        "pose_relation",
}


def split_for(image_index: int) -> str:
    if image_index < 20000:
        return "train"
    if image_index < 25000:
        return "val"
    return "test"


def compute_depth(program: list[dict]) -> int:
    """Depth = 1 + count of object<->part / visibility traversals."""
    depth = 1
    for node in program:
        if node.get("type") in DEPTH_OPS:
            depth += 1
    return depth


def attribute_type(program: list[dict]) -> str:
    """Tag from the last node whose type begins with `query_` or `partquery_`."""
    for node in reversed(program):
        t = node.get("type", "")
        if t.startswith("query_") or t.startswith("partquery_"):
            return QUERY_OP_TO_ATTRIBUTE.get(t, t)
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--questions_path", default=None,
                    help=f"Defaults to <data_root>/{QUESTIONS_PATH}")
    ap.add_argument("--scenes_path", default=None,
                    help=f"Defaults to <data_root>/{SCENES_PATH}")
    ap.add_argument("--out_csv", default=None,
                    help="Defaults to <data_root>/parts_vqa.csv")
    ap.add_argument("--vocab_out", default="label_vocab_superclevr3d_parts.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows for a quick smoke run")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    qpath     = Path(args.questions_path) if args.questions_path else data_root / QUESTIONS_PATH
    spath     = Path(args.scenes_path)    if args.scenes_path    else data_root / SCENES_PATH
    out_csv   = Path(args.out_csv)        if args.out_csv        else data_root / "parts_vqa.csv"

    print(f"Loading scenes from {spath} …")
    with open(spath) as f:
        scenes_data = json.load(f)
    scenes = scenes_data.get("scenes", scenes_data)
    n_objects_by_index = {s["image_index"]: len(s.get("objects", [])) for s in scenes}
    print(f"  {len(n_objects_by_index)} scenes, n_objects range "
          f"{min(n_objects_by_index.values())}-{max(n_objects_by_index.values())}")

    print(f"Loading questions from {qpath} …")
    with open(qpath) as f:
        q_data = json.load(f)
    questions = q_data.get("questions", q_data)
    if args.limit:
        questions = questions[: args.limit]
    print(f"  {len(questions)} questions")

    rows = []
    missing_scene = 0
    for q in questions:
        idx = q["image_index"]
        nobj = n_objects_by_index.get(idx)
        if nobj is None:
            missing_scene += 1
            continue
        program = q.get("program", [])
        rows.append({
            "image_index":      idx,
            "image_name":       q["image_filename"],
            "split":            split_for(idx),
            "query":            q["question"],
            "label":            str(q["answer"]),
            "label_rank":       1,
            "attribute_type":   attribute_type(program),
            "depth":            compute_depth(program),
            "n_objects":        nobj,
            "n_program_nodes":  len(program),
            "question_index":   q.get("question_index", -1),
            "template_filename": q.get("template_filename", ""),
        })

    if missing_scene:
        print(f"  WARNING: {missing_scene} questions referenced missing scenes; dropped.")

    df = pd.DataFrame(rows)
    print(f"\nRow count: {len(df):,}")
    print("Per-split counts:")
    print(df["split"].value_counts())
    print("\nDepth distribution:")
    print(df["depth"].value_counts().sort_index())
    print("\nAttribute_type distribution:")
    print(df["attribute_type"].value_counts())
    print("\nn_objects distribution:")
    print(df["n_objects"].value_counts().sort_index())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote CSV → {out_csv}")

    # Build label vocab from the train split.
    train_labels = sorted(df[df["split"] == "train"]["label"].unique().tolist())
    vocab = {lbl: i for i, lbl in enumerate(train_labels)}
    print(f"Train answer vocab: {len(vocab)} unique answers")
    with open(args.vocab_out, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Wrote vocab → {args.vocab_out}")

    # How many test/val rows have answers unseen in train?
    val_unseen  = (~df[df["split"] == "val" ]["label"].isin(vocab)).sum()
    test_unseen = (~df[df["split"] == "test"]["label"].isin(vocab)).sum()
    print(f"OOV answers in val:  {val_unseen}")
    print(f"OOV answers in test: {test_unseen}")


if __name__ == "__main__":
    main()
