"""Build the CUB-200-2011 part-attribute question CSV.

    python data_prep/cub/build_cub_csv.py --cub-root data/cub/CUB_200_2011 --out data/cub/cub200_questions.csv

Columns: image_name, query, label, label_rank, class_label, split. For each image and each
attribute category (e.g. ``has_bill_shape``) the attribute values marked present are kept and
dense-ranked by annotator certainty (rank 1 = most certain); ``--max-rank`` keeps the top tiers.
Training and evaluation use rank-1 rows only. The thesis uses the 16 colour questions
("What is the <part> color of the bird?", selected with ``category_filter=color``), 15 colour
classes, official train/test split.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def load_attributes(path):
    """attribute_id → (category, value), e.g. 1 → ('bill_shape', 'curved_(up_or_down)')."""
    attrs = {}
    with open(path) as f:
        for line in f:
            aid, name = line.split()
            cat, val = name.split("::")
            attrs[int(aid)] = (cat[len("has_"):], val)
    return attrs


def category_to_query(cat: str) -> str:
    return "What is the {} of the bird?".format(cat.replace("_", " "))


def read_pairs(path):
    with open(path) as f:
        return [line.split() for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-root", default="data/cub/CUB_200_2011", help="extracted CUB_200_2011 directory")
    ap.add_argument("--min-certainty", type=int, default=1, help="keep labels with certainty_id >= this")
    ap.add_argument("--max-rank", type=int, default=2, help="keep the top N certainty tiers per (image, category)")
    ap.add_argument("--out", default="data/cub/cub200_questions.csv")
    args = ap.parse_args()
    root = args.cub_root

    attr_path = os.path.join(root, "attributes", "attributes.txt")
    if not os.path.exists(attr_path):                      # the official release ships it one level up
        attr_path = os.path.join(os.path.dirname(root.rstrip("/")), "attributes.txt")
    attrs = load_attributes(attr_path)
    images = {int(i): n for i, n in read_pairs(os.path.join(root, "images.txt"))}
    split = {int(i): ("train" if int(t) == 1 else "test") for i, t in read_pairs(os.path.join(root, "train_test_split.txt"))}
    classes = {int(c): n for c, n in read_pairs(os.path.join(root, "classes.txt"))}
    image_class = {int(i): classes[int(c)] for i, c in read_pairs(os.path.join(root, "image_class_labels.txt"))}

    present = defaultdict(list)
    with open(os.path.join(root, "attributes", "image_attribute_labels.txt")) as f:
        for line in f:
            p = line.split()
            iid, aid, is_present, cert = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            if is_present == 1 and cert >= args.min_certainty:
                present[(iid, attrs[aid][0])].append((aid, cert))

    cat_order = list(dict.fromkeys(attrs[aid][0] for aid in sorted(attrs)))
    rows = []
    for iid in sorted(images):
        for cat in cat_order:
            items = present.get((iid, cat))
            if not items:
                continue
            items = sorted(items, key=lambda x: (-x[1], x[0]))
            rank, prev_cert = 0, None
            for aid, cert in items:
                if cert != prev_cert:
                    rank += 1
                    prev_cert = cert
                if rank > args.max_rank:
                    break
                rows.append([images[iid], category_to_query(cat), attrs[aid][1], rank, image_class[iid], split[iid]])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "query", "label", "label_rank", "class_label", "split"])
        w.writerows(rows)
    print(f"wrote {len(rows):,} rows to {args.out}")


if __name__ == "__main__":
    main()
