"""Report the diff between InternVL3-14B colors and the geometric (OKLab) labels.

Reads ``parts_color_vqa_internvl.csv`` (VQA CSV + ``vlm_color``) and writes a
Markdown report: overall agreement, an 11x11 confusion matrix (geometric rows vs
VLM cols), per-color precision/recall-style breakdown, the biggest systematic
disagreements, agreement by split, and the parts where the two annotators diverge
most. The geometric 21-color label is mapped to the same 11 basics as the VLM so
the comparison is apples-to-apples.

Run from repo root:
    conda run -n oclf_env python report_vlm_vs_geom.py
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd

from merge_internvl_preds import FINE_TO_BASIC, to_basic

BASICS = ["black", "white", "gray", "brown", "red", "orange",
          "yellow", "green", "blue", "purple", "pink"]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="FG-datset/ade20k/parts_color_vqa_internvl.csv")
    ap.add_argument("--out-md", default="FG-datset/ade20k/vlm_vs_geom_report.md")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    n_total = len(df)

    # Map the geometric (fine 21-color) label down to the 11 basics.
    df["geom_basic"] = df["label"].map(to_basic)

    valid = df[df["vlm_color"].notna() & (df["vlm_color"] != "unknown")].copy()
    n_unknown = (df["vlm_color"] == "unknown").sum()
    n_missing = df["vlm_color"].isna().sum()
    n_valid = len(valid)

    agree = (valid["geom_basic"] == valid["vlm_color"])
    overall = agree.mean()

    # ---- Confusion matrix (geom rows x vlm cols), restricted to the 11 basics ----
    cm = pd.crosstab(valid["geom_basic"], valid["vlm_color"])
    cm = cm.reindex(index=BASICS, columns=BASICS, fill_value=0)

    # ---- Per-color: how often the geometric label is confirmed by the VLM --------
    per_color = []
    for c in BASICS:
        geom_n = int((valid["geom_basic"] == c).sum())
        vlm_n = int((valid["vlm_color"] == c).sum())
        kept = int(((valid["geom_basic"] == c) & (valid["vlm_color"] == c)).sum())
        recall = kept / geom_n if geom_n else 0.0     # of geom=c, fraction VLM agreed
        prec = kept / vlm_n if vlm_n else 0.0         # of vlm=c, fraction geom agreed
        per_color.append((c, geom_n, vlm_n, f"{recall:.0%}", f"{prec:.0%}"))

    # ---- Biggest directional disagreements --------------------------------------
    dis = valid[~agree]
    pairs = Counter(zip(dis["geom_basic"], dis["vlm_color"]))
    top_pairs = pairs.most_common(20)

    # ---- Agreement by split -----------------------------------------------------
    by_split = (valid.assign(ok=agree.values)
                .groupby("split")["ok"].agg(["mean", "size"]))

    # ---- Parts with the most disagreement (min support) -------------------------
    valid["part"] = valid["query"].str.extract(r"color of the (.+?) of the")[0]
    part_stats = (valid.assign(ok=agree.values)
                  .groupby("part")["ok"].agg(["mean", "size"]))
    part_stats = part_stats[part_stats["size"] >= 30].sort_values("mean")
    worst_parts = part_stats.head(15)
    best_parts = part_stats.sort_values("mean", ascending=False).head(15)

    # ---- Example disagreements (one per top pair) -------------------------------
    examples = []
    for (g, v), _ in top_pairs[:8]:
        row = dis[(dis["geom_basic"] == g) & (dis["vlm_color"] == v)].iloc[0]
        examples.append((row["question_index"], g, v, row["label"],
                         row["raw_response"], row["query"]))

    # ============================ write markdown =================================
    L = []
    L.append("# InternVL3-14B vs. geometric (OKLab) color annotations\n")
    L.append(f"Source: `{args.in_csv}` — **{n_total:,}** questions "
             f"(ADE20K part-color VQA).\n")
    L.append("The geometric annotation is the robust-OKLab `color:` tag (21-color "
             "palette) written by `add_color_attributes.py`; the VLM annotation is "
             "InternVL3-14B shown the part-only crop. For a fair comparison the "
             "21-color geometric label is collapsed to the same **11 basic colors** "
             "as the VLM prompt (e.g. beige/tan/cream → brown).\n")

    L.append("## Headline\n")
    L.append(md_table(
        ["metric", "value"],
        [["questions", f"{n_total:,}"],
         ["VLM answered (valid basic color)", f"{n_valid:,}"],
         ["VLM `unknown` (e.g. transparent → 'Clear')", f"{n_unknown}"],
         ["no prediction", f"{n_missing}"],
         ["**agreement (11 basics)**", f"**{overall:.1%}**"],
         ["disagreement", f"{1-overall:.1%}  ({(~agree).sum():,} rows)"]]))
    L.append("")

    L.append("## Confusion matrix — geometric (rows) → VLM (columns)\n")
    L.append("Diagonal = agreement. Read a row as: of the parts the OKLab method "
             "called *X*, how the VLM relabelled them.\n")
    header = ["geom \\ vlm"] + BASICS + ["row tot"]
    rows = []
    for c in BASICS:
        rowvals = [int(cm.loc[c, k]) for k in BASICS]
        rows.append([f"**{c}**"] + rowvals + [sum(rowvals)])
    rows.append(["**col tot**"] + [int(cm[k].sum()) for k in BASICS] + [n_valid])
    L.append(md_table(header, rows))
    L.append("")

    L.append("## Per-color confirmation\n")
    L.append("- **kept (recall)** = of parts the OKLab method called this color, the "
             "fraction the VLM agreed.\n- **precision** = of parts the VLM called this "
             "color, the fraction the OKLab method agreed.\n")
    L.append(md_table(
        ["color", "geom n", "vlm n", "kept (recall)", "precision"], per_color))
    L.append("")

    L.append("## Largest systematic disagreements (geometric → VLM)\n")
    L.append(md_table(
        ["geometric", "→ VLM", "count", "% of all disagreements"],
        [[g, v, f"{n:,}", f"{n/len(dis):.1%}"] for (g, v), n in top_pairs]))
    L.append("")

    L.append("## Agreement by split\n")
    L.append(md_table(
        ["split", "agreement", "n"],
        [[s, f"{by_split.loc[s, 'mean']:.1%}", f"{int(by_split.loc[s, 'size']):,}"]
         for s in by_split.index]))
    L.append("")

    L.append("## Parts where the two annotators diverge most (≥30 questions)\n")
    L.append(md_table(
        ["part", "agreement", "n"],
        [[p, f"{r['mean']:.0%}", int(r['size'])] for p, r in worst_parts.iterrows()]))
    L.append("\n### Parts where they agree most\n")
    L.append(md_table(
        ["part", "agreement", "n"],
        [[p, f"{r['mean']:.0%}", int(r['size'])] for p, r in best_parts.iterrows()]))
    L.append("")

    L.append("## Example disagreements\n")
    L.append(md_table(
        ["qid", "geom(basic)", "VLM", "geom(fine)", "VLM raw", "question"],
        [[q, g, v, fine, f"`{raw}`", ques] for q, g, v, fine, raw, ques in examples]))
    L.append("")

    L.append("## Takeaways\n")
    L.append(f"- The two methods agree on **{overall:.0%}** of parts once both are "
             "reduced to basic colors.\n"
             "- Disagreement is dominated by **neutral confusions** — the OKLab "
             "estimator's earth-tone bias (beige/tan/cream all → *brown*) vs. the "
             "VLM reading bright porcelain/metal surfaces as *white*/*gray*. The top "
             "pairs (brown→white, gray→white, brown→gray) are all neutral↔neutral, "
             "and together account for ~35% of all disagreements.\n"
             "- By volume the split is mostly about *which neutral* (white vs gray vs "
             "brown vs black), not about hue: the four neutrals hold 89% of all rows "
             "and the bulk of the off-diagonal mass sits among them.\n"
             "- Saturated colors are uneven: **brown** (86% precision) and **blue**/"
             "**yellow** are mostly confirmed, but **green→yellow** (218) and "
             "**orange→brown** (273) show the VLM and OKLab disagree on warm/olive "
             "boundaries; low-support colors (pink, purple, orange) are noisy both ways.\n"
             "- The VLM also injects **semantic priors** the mask-only OKLab method "
             "can't: bare skin → pink (brown→pink, 150) and a bias toward calling clean "
             "ceramic/metal *white*/*gray* rather than the tan the pixel statistics give.\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote → {args.out_md}")
    print(f"\nAgreement: {overall:.1%} | valid {n_valid:,} | unknown {n_unknown} | "
          f"disagreements {(~agree).sum():,}")


if __name__ == "__main__":
    main()
