#!/usr/bin/env python
"""Merge InternVL scene shard CSVs into one per-image table and audit coverage.

Combines FG-datset/preds_scene/scene_shard_*.csv into
FG-datset/paco_image_scenes_vlm.csv, checks every unique image in
paco_questions.csv is covered, and (if the CLIP labels exist) reports the
VLM-vs-CLIP agreement rate per scene as a reliability cross-check.
"""
import argparse
import csv
import glob
import os
from collections import Counter, OrderedDict


def load_unique_images(csv_path):
    imgs = OrderedDict()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            imgs.setdefault(r["image_name"], None)
    return list(imgs.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default="FG-datset/preds_scene")
    ap.add_argument("--questions", default="FG-datset/paco_questions.csv")
    ap.add_argument("--clip", default="FG-datset/paco_image_scenes.csv")
    ap.add_argument("--out", default="FG-datset/paco_image_scenes_vlm.csv")
    args = ap.parse_args()

    rows = {}
    for path in sorted(glob.glob(os.path.join(args.shards_dir, "scene_shard_*.csv"))):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["image_name"]] = r  # last write wins (resumed shards)
    print(f"[merge] {len(rows)} image rows from "
          f"{len(glob.glob(os.path.join(args.shards_dir, 'scene_shard_*.csv')))} shards")

    all_imgs = load_unique_images(args.questions)
    missing = [n for n in all_imgs if n not in rows]
    print(f"[cover] {len(all_imgs)-len(missing)}/{len(all_imgs)} images covered; "
          f"{len(missing)} missing")
    if missing[:5]:
        print("        e.g.", missing[:5])

    fields = ["image_name", "coco_id", "splits", "scene", "raw_response"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n in all_imgs:
            if n in rows:
                w.writerow({k: rows[n].get(k, "") for k in fields})
    print(f"[done] wrote {args.out}")

    counts = Counter(r["scene"] for r in rows.values())
    print("\n=== VLM scene distribution ===")
    tot = sum(counts.values())
    for sc, c in counts.most_common():
        print(f"  {sc:14s} {c:6d}  ({100*c/max(1,tot):.1f}%)")

    # cross-check vs CLIP
    if os.path.exists(args.clip):
        clip = {}
        with open(args.clip, newline="") as f:
            for r in csv.DictReader(f):
                clip[r["image_name"]] = r["scene"]
        both = [(rows[n]["scene"], clip[n]) for n in rows if n in clip]
        agree = sum(v == c for v, c in both)
        print(f"\n[audit] VLM vs CLIP agree on {agree}/{len(both)} "
              f"({100*agree/max(1,len(both)):.1f}%)")


if __name__ == "__main__":
    main()
