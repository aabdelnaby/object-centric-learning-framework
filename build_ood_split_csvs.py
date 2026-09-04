#!/usr/bin/env python
"""Build the train-vs-OOD-eval CSV pair for the scene-holdout experiment.

Given the per-question scene labels (from classify_scenes_internvl.py, merged into
paco_questions_with_scene.csv), split the questions into:

  * <stem>_no_<scene>.csv   -- every question whose image is NOT the held-out
    scene. Keeps the original train/val split, so `train.py --csv_path` trains on
    split=='train' and validates on split=='val' exactly as usual, and picks
    best_model.pt on this in-domain (non-held-out) val set.

  * <stem>_<scene>_only.csv -- every question whose image IS the held-out scene,
    with split forced to 'val' so `eval_hier_router.py --split val` evaluates ALL
    of them. This is the OOD test set.

The column schema is identical to the input CSV (the extra `scene`/`scene_conf`
columns are harmless to the loaders, which index by name).
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="FG-datset/paco_questions_with_scene.csv")
    ap.add_argument("--holdout", default="bedroom")
    ap.add_argument("--out_dir", default="FG-datset")
    ap.add_argument("--eval_split", default="val",
                    help="split value written for the held-out (OOD eval) rows")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    assert "scene" in df.columns, f"{args.in_csv} has no 'scene' column"
    assert (df["scene"] == args.holdout).any(), \
        f"no rows with scene=='{args.holdout}'"

    in_dist = df[df["scene"] != args.holdout].copy()
    ood = df[df["scene"] == args.holdout].copy()
    ood["split"] = args.eval_split  # eval all held-out questions under one split

    stem = os.path.splitext(os.path.basename(args.in_csv))[0]
    # drop trailing _with_scene from the stem for tidier names
    stem = stem.replace("_with_scene", "")
    train_path = os.path.join(args.out_dir, f"{stem}_no_{args.holdout}.csv")
    eval_path = os.path.join(args.out_dir, f"{stem}_{args.holdout}_only.csv")
    in_dist.to_csv(train_path, index=False)
    ood.to_csv(eval_path, index=False)

    def report(name, d):
        sp = d["split"].value_counts().to_dict()
        print(f"  {name}: {len(d):6d} questions, {d['image_name'].nunique():5d} images, "
              f"split={sp}")

    print(f"holdout scene = '{args.holdout}'")
    report(os.path.basename(train_path), in_dist)
    report(os.path.basename(eval_path), ood)

    # OOV safety: every color label in the OOD eval must exist in the training split
    train_labels = set(in_dist[in_dist["split"] == "train"]["label"].astype(str)
                       .str.strip().str.lower())
    ood_labels = set(ood["label"].astype(str).str.strip().str.lower())
    oov = ood_labels - train_labels
    print(f"  color-label OOV in OOD set vs train: {len(oov)} "
          f"{'' if not oov else sorted(oov)}")
    print(f"\nwrote:\n  {train_path}\n  {eval_path}")


if __name__ == "__main__":
    main()
