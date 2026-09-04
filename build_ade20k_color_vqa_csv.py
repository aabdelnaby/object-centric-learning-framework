"""Build a part-color VQA CSV for ADE20K.

Questions have the form:
    "What is the color of the <part> of the <object>?"
generated for every (object, part) pair that is *unambiguous* in an image:

  * the OBJECT is a top-level entity (``part_level == 0``) whose ``raw_name``
    appears **exactly once** among the image's objects, and
  * the PART is one of that object's direct parts (``part_level == 1``, linked via
    ``parts.hasparts``) whose ``raw_name`` appears **exactly once** within that object.

The answer is the part's color, read from the ``color:<name>`` tag written into its
``attributes`` by ``add_color_attributes.py`` (entities tagged ``color:none`` or with
no color tag are skipped). Because both referents are unique, "the <part> of the
<object>" resolves to a single entity, so each question has one well-defined answer.

The CSV schema matches ``FG-datset/superclevr3d/parts_vqa.csv`` so the same dataset /
trainer plumbing can read it.

Run from repo root:
    conda run -n oclf_env python build_ade20k_color_vqa_csv.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter

import pandas as pd

# Reuse the tolerant JSON loader (a few ADE20K JSONs are Latin-1, not UTF-8).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "FG-datset", "ade20k", "ADE20K", "utils"))
import color_attributes as ca  # noqa: E402


def split_for(folder: str) -> str:
    """ADE20K ships a training/validation split encoded in the folder path."""
    if "/training/" in "/" + folder + "/":
        return "train"
    if "/validation/" in "/" + folder + "/":
        return "val"
    return "unknown"


def relative_image_name(folder: str, filename: str) -> str:
    """Path under .../images, e.g. 'ADE/training/work_place/lobby/ADE_train_00011521.jpg'."""
    full = os.path.join(folder, filename)
    marker = "images" + os.sep
    return full.split(marker, 1)[1] if marker in full else full


def part_color(obj) -> str | None:
    """Color name from an entity's ``color:<name>`` attribute, or None / 'none'."""
    for a in (obj.get("attributes") or []):
        if isinstance(a, str) and a.startswith("color:"):
            name = a[len("color:"):].strip()
            return None if name in ("", "none") else name
    return None


def _as_id_list(v):
    """ADE20K stores 'hasparts'/'ispartof' as a list of ids OR a single int id."""
    if isinstance(v, list):
        return v
    if isinstance(v, int):
        return [v]
    return []


def iter_question_parts(objects):
    """Yield (object, part, color) for each unambiguous object/part pair.

    Same selection as ``iter_questions`` but also returns the full ``object``
    and ``part`` annotation dicts (so callers can reach ``id`` / ``instance_mask``).
    """
    id2obj = {o.get("id"): o for o in objects}
    level0 = [o for o in objects if int(o["parts"]["part_level"]) == 0]
    obj_counts = Counter(
        (o.get("raw_name") or "").strip() for o in level0 if (o.get("raw_name") or "").strip())

    for obj in level0:
        oname = (obj.get("raw_name") or "").strip()
        if not oname or obj_counts[oname] != 1:
            continue  # object must be uniquely named in the image
        parts = [id2obj[pid] for pid in _as_id_list(obj["parts"].get("hasparts"))
                 if pid in id2obj and int(id2obj[pid]["parts"]["part_level"]) == 1]
        if not parts:
            continue
        part_counts = Counter(
            (p.get("raw_name") or "").strip() for p in parts if (p.get("raw_name") or "").strip())
        for part in parts:
            pname = (part.get("raw_name") or "").strip()
            if not pname or part_counts[pname] != 1:
                continue  # part must be unique within this object
            color = part_color(part)
            if color is not None:
                yield obj, part, color


def iter_questions(objects):
    """Yield (object_name, part_name, color) for each unambiguous object/part pair."""
    for obj, part, color in iter_question_parts(objects):
        yield (obj.get("raw_name") or "").strip(), (part.get("raw_name") or "").strip(), color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="FG-datset/ade20k/ADE20K_2021_17_01/index_ade20k.pkl")
    ap.add_argument("--base-dir", default=None,
                    help="Dir the index 'folder' paths are relative to. "
                         "Default: parent of the index file's directory.")
    ap.add_argument("--out-csv", default="FG-datset/ade20k/parts_color_vqa.csv")
    ap.add_argument("--vocab-out", default="label_vocab_ade20k_color.json")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N images (debug).")
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
    n_images_with_q = 0
    for image_index, (fn, folder) in pairs:
        json_path = os.path.join(base_dir, folder, fn).replace(".jpg", ".json")
        if not os.path.exists(json_path):
            continue
        try:
            objects = ca._load_json(json_path)["annotation"]["object"]
        except Exception:
            continue
        n_objects = sum(1 for o in objects if int(o["parts"]["part_level"]) == 0)
        img_name = relative_image_name(folder, fn)
        split = split_for(folder)

        before = len(rows)
        for oname, pname, color in iter_questions(objects):
            rows.append({
                "image_index":       image_index,
                "image_name":        img_name,
                "split":             split,
                "query":             f"What is the color of the {pname} of the {oname}?",
                "label":             color,
                "label_rank":        1,
                "attribute_type":    "color",
                "depth":             2,
                "n_objects":         n_objects,
                "n_program_nodes":   4,
                "question_index":    len(rows),
                "template_filename": "ade20k_color_part",
            })
        if len(rows) > before:
            n_images_with_q += 1

    df = pd.DataFrame(rows)
    print(f"\nGenerated {len(df):,} questions from {n_images_with_q:,} images.")
    if len(df):
        print("\nPer-split row counts:")
        print(df["split"].value_counts())
        print(f"\nDistinct query strings: {df['query'].nunique():,}")
        print("\nTop-15 answer labels:")
        for c, n in df["label"].value_counts().head(15).items():
            print(f"  {c:<12} {n:6,}")

        train = df[df["split"] == "train"]
        train_labels = sorted(train["label"].unique().tolist())
        vocab = {lbl: i for i, lbl in enumerate(train_labels)}
        val_oov = (~df[df["split"] == "val"]["label"].isin(vocab)).sum()
        print(f"\nTrain answer vocab ({len(vocab)} classes): {list(vocab.keys())}")
        print(f"OOV answers in val: {val_oov}")
        with open(args.vocab_out, "w") as f:
            json.dump(vocab, f, indent=2)
        print(f"Wrote vocab → {args.vocab_out}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV → {args.out_csv}")


if __name__ == "__main__":
    main()
