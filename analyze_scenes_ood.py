#!/usr/bin/env python
"""Summarize the CLIP scene labels and merge them into the questions CSV.

Reads FG-datset/paco_image_scenes.csv (per-image scene labels) and:
  1. prints the scene distribution over images and over QUESTIONS (weighting by
     how many questions each image carries -- what the OOD test actually sees),
     broken down by the original PACO train/val split;
  2. reports CLIP confidence and the caption agreement rate per scene, so you can
     judge label reliability before holding a scene out;
  3. writes FG-datset/paco_questions_with_scene.csv -- the original questions CSV
     with `scene` and `scene_conf` columns appended (joined on image_name);
  4. optionally (--holdout SCENE) writes an OOD split column `ood_split`
     (test = held-out scene, train = everything else).
"""
import argparse
import csv
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="FG-datset/paco_image_scenes.csv")
    ap.add_argument("--questions", default="FG-datset/paco_questions.csv")
    ap.add_argument("--out", default="FG-datset/paco_questions_with_scene.csv")
    ap.add_argument("--holdout", default="", help="scene label to mark as OOD test")
    ap.add_argument("--conf_min", type=float, default=0.30,
                    help="confidence below this is flagged low-confidence")
    args = ap.parse_args()

    # per-image scene table (confidence/caption columns are optional: the VLM
    # CSV has neither, the CLIP CSV has both)
    scene_of, conf_of, cap_of, agree_of = {}, {}, {}, {}
    with open(args.scenes, newline="") as f:
        for r in csv.DictReader(f):
            n = r["image_name"]
            scene_of[n] = r["scene"]
            conf_of[n] = float(r["confidence"]) if r.get("confidence") else 1.0
            cap_of[n] = r.get("caption_scene", "")
            agree_of[n] = r.get("agree", "")

    # join onto questions, count by split
    img_count = Counter()                       # images per scene
    q_count = Counter()                         # questions per scene
    img_by_split = defaultdict(Counter)         # split -> scene -> #images
    q_by_split = defaultdict(Counter)           # split -> scene -> #questions
    seen_img = set()
    rows = []
    with open(args.questions, newline="") as f:
        rd = csv.DictReader(f)
        qfields = rd.fieldnames
        for r in rd:
            n = r["image_name"]
            sc = scene_of.get(n, "UNKNOWN")
            r["scene"] = sc
            r["scene_conf"] = f"{conf_of.get(n, 0):.4f}"
            rows.append(r)
            sp = r.get("split", "")
            q_count[sc] += 1
            q_by_split[sp][sc] += 1
            if n not in seen_img:
                seen_img.add(n)
                img_count[sc] += 1
                img_by_split[sp][sc] += 1

    n_img = sum(img_count.values())
    n_q = sum(q_count.values())

    print(f"\n=== scene distribution: {n_img} images / {n_q} questions ===")
    print(f"{'scene':14s} {'imgs':>6s} {'img%':>6s} {'ques':>7s} {'q%':>6s} "
          f"{'meanConf':>9s} {'capAgree':>9s}")
    for sc, _ in img_count.most_common():
        imgs = [n for n in scene_of if scene_of[n] == sc]
        confs = [conf_of[n] for n in imgs]
        agr = [int(agree_of[n]) for n in imgs if agree_of[n] in ("0", "1")]
        mc = sum(confs) / len(confs) if confs else 0
        ca = (100 * sum(agr) / len(agr)) if agr else float("nan")
        print(f"{sc:14s} {img_count[sc]:6d} {100*img_count[sc]/n_img:5.1f}% "
              f"{q_count[sc]:7d} {100*q_count[sc]/n_q:5.1f}% {mc:9.3f} "
              f"{ca:8.1f}%")

    print("\n=== by original PACO split (images) ===")
    splits = sorted(img_by_split)
    allsc = [sc for sc, _ in img_count.most_common()]
    print(f"{'scene':14s} " + " ".join(f"{s:>8s}" for s in splits))
    for sc in allsc:
        print(f"{sc:14s} " + " ".join(f"{img_by_split[s][sc]:8d}" for s in splits))

    low = sum(1 for n in conf_of if conf_of[n] < args.conf_min)
    print(f"\n[conf] {low}/{n_img} images below conf {args.conf_min} "
          f"({100*low/n_img:.1f}%) -- candidates for manual review/relabel")

    # write merged questions CSV (+ optional OOD split)
    out_fields = qfields + ["scene", "scene_conf"]
    if args.holdout:
        out_fields += ["ood_split"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            if args.holdout:
                r["ood_split"] = "test" if r["scene"] == args.holdout else "train"
            w.writerow(r)
    print(f"\n[done] wrote {args.out}")
    if args.holdout:
        n_test = sum(1 for r in rows if r["scene"] == args.holdout)
        print(f"[ood] holdout='{args.holdout}': {n_test} test questions, "
              f"{n_q - n_test} train questions")


if __name__ == "__main__":
    main()
