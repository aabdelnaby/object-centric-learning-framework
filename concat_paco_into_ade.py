"""Concatenate PACO VQA rows onto the ADE20K CSV → the combined hier_router dataset.

ADE rows keep their relative ``image_name`` (resolved under the ade ``image_root``);
PACO rows carry an absolute ``image_name`` (``os.path.join(image_root, abs)`` returns
the abs path, so they resolve with no train.py change). ``question_index`` is
reassigned globally contiguous; ``image_index`` is left as-is (ADE <=27,572 and PACO
1,000,000+coco_id are already disjoint). train.py reads neither index as a key — splits
are driven solely by the ``split`` column.

Run from repo root:
    conda run -n oclf_env python concat_paco_into_ade.py \
        --paco-csvs FG-datset/coco_paco/parts_color_vqa_paco_val.csv
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade-csv", default="FG-datset/ade20k/parts_color_vqa_internvl_val30.csv")
    ap.add_argument("--paco-csvs", nargs="+",
                    default=["FG-datset/coco_paco/parts_color_vqa_paco_val.csv"])
    ap.add_argument("--out-csv", default="FG-datset/parts_color_vqa_combined.csv")
    args = ap.parse_args()

    ade = pd.read_csv(args.ade_csv)
    frames = [ade]
    sources = ["ade"] * len(ade)
    for p in args.paco_csvs:
        df = pd.read_csv(p)
        assert list(df.columns) == list(ade.columns), \
            f"column mismatch in {p}:\n  {list(df.columns)}\n  vs\n  {list(ade.columns)}"
        frames.append(df)
        sources += ["paco"] * len(df)

    comb = pd.concat(frames, ignore_index=True)
    comb["question_index"] = range(len(comb))      # globally contiguous (metadata only)

    print(f"ADE rows: {len(ade):,}  PACO rows: {len(comb) - len(ade):,}  total: {len(comb):,}")
    tmp = comb.copy()
    tmp["source"] = sources
    print("\nrows by source x split:")
    print(tmp.groupby(["source", "split"]).size())

    tr_vocab = set(comb[comb["split"] == "train"]["label"].astype(str))
    val_lbls = set(comb[comb["split"] == "val"]["label"].astype(str))
    print(f"\ntrain label classes ({len(tr_vocab)}): {sorted(tr_vocab)}")
    print(f"val labels OOV vs train: {sorted(val_lbls - tr_vocab)}")
    # image_index disjointness sanity
    assert comb["image_index"].notna().all()
    print(f"\nimage_index range: {comb['image_index'].min()}..{comb['image_index'].max()}")

    comb.to_csv(args.out_csv, index=False)
    print(f"Wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
