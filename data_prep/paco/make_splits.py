"""Concatenate the finalised PACO train and val question CSVs into the training file.

    python data_prep/paco/make_splits.py            # data/paco/paco_train.csv + paco_val.csv → data/paco/paco_questions.csv

The thesis training file has 50,679 train and 2,656 val questions (12 answer classes, the
54 questions whose colour could not be determined by PACO or the VLM keep the label "unknown").
"""

from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/paco/paco_train.csv")
    ap.add_argument("--val-csv", default="data/paco/paco_val.csv")
    ap.add_argument("--out", default="data/paco/paco_questions.csv")
    args = ap.parse_args()

    train, val = pd.read_csv(args.train_csv), pd.read_csv(args.val_csv)
    for name, df in (("train", train), ("val", val)):
        if df["label"].isna().any() or (df["label"].astype(str) == "").any():
            raise SystemExit(f"{name} CSV has empty labels; run finalize_csv.py first")
    out = pd.concat([train, val], ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"train {len(train):,} + val {len(val):,} = {len(out):,} questions, "
          f"{out['label'].nunique()} answer classes → {args.out}")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
