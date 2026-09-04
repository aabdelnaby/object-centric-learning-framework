#!/usr/bin/env python
"""Score each scene as a held-out OOD candidate for the part-color VQA experiment.

A good held-out scene for testing *scene* generalization should:
  * have enough test questions for a stable metric;
  * ask about (object, part) pairs and colors that ALSO occur in the remaining
    (training) scenes -- so a wrong answer reflects the novel scene, not a novel
    object/part the model never saw (which would confound the result).

For each scene S we treat "train" = all questions NOT in S and report:
  q            #questions in S (test size)
  obj/pair     #unique objects / #unique (object,part) pairs in S
  pairCov      % of S's questions whose (object,part) pair also appears in train
  objCov       % of S's questions whose object also appears in train
  colCov       % of S's questions whose color label also appears in train
  newPairs     #(object,part) pairs unique to S (never in train)
"""
import argparse
import csv
import re
from collections import Counter, defaultdict

PART_OBJ = re.compile(r"color of the (.+?) of the (.+?)\s*\?", re.I)
OBJ_ONLY = re.compile(r"color of the (.+?)\s*\?", re.I)


def parse_obj_part(q):
    m = PART_OBJ.search(q)
    if m:
        return m.group(2).strip().lower(), m.group(1).strip().lower()
    m = OBJ_ONLY.search(q)
    if m:
        return m.group(1).strip().lower(), ""   # whole-object question
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="FG-datset/paco_questions_with_scene.csv")
    ap.add_argument("--min_q", type=int, default=600,
                    help="ignore scenes with fewer questions than this")
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            obj, part = parse_obj_part(r["query"])
            rows.append((r["scene"], obj, part, (r.get("label") or "").strip().lower()))

    scenes = sorted({s for s, *_ in rows})
    by_scene = defaultdict(list)
    for s, o, p, c in rows:
        by_scene[s].append((o, p, c))

    print(f"{'scene':13s} {'q':>5s} {'obj':>4s} {'pair':>5s} "
          f"{'pairCov':>7s} {'objCov':>6s} {'colCov':>6s} {'newPairs':>8s}")
    results = []
    for s in scenes:
        if s in ("unknown",):
            continue
        test = by_scene[s]
        if len(test) < args.min_q:
            continue
        train_pairs = {(o, p) for ss in scenes if ss != s for (o, p, c) in by_scene[ss]}
        train_objs = {o for ss in scenes if ss != s for (o, p, c) in by_scene[ss]}
        train_cols = {c for ss in scenes if ss != s for (o, p, c) in by_scene[ss]}
        npair = sum((o, p) in train_pairs for (o, p, c) in test)
        nobj = sum(o in train_objs for (o, p, c) in test)
        ncol = sum(c in train_cols for (o, p, c) in test)
        test_pairs = {(o, p) for (o, p, c) in test}
        new_pairs = test_pairs - train_pairs
        n = len(test)
        pc, oc, cc = 100*npair/n, 100*nobj/n, 100*ncol/n
        print(f"{s:13s} {n:5d} {len({o for o,p,c in test}):4d} {len(test_pairs):5d} "
              f"{pc:6.1f}% {oc:5.1f}% {cc:5.1f}% {len(new_pairs):8d}")
        results.append((s, n, pc, len(new_pairs)))

    # rank: distinct scene + high pair coverage + enough questions
    print("\n=== ranked by (pairCov desc, q desc) among scenes with >= min_q ===")
    for s, n, pc, npn in sorted(results, key=lambda x: (-x[2], -x[1])):
        print(f"  {s:13s} q={n:5d}  pairCov={pc:5.1f}%  newPairs={npn}")


if __name__ == "__main__":
    main()
