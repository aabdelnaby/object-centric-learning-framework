"""Evaluate PACO part-color VQA on InternVL3-14B with the FULL image + question.

Unlike ``annotate_colors_internvl.py`` (which shows InternVL an isolated part *crop*
and asks for the dominant colour), this script shows the **whole COCO scene** and the
row's natural-language **question** (e.g. "What is the color of the strap of the belt?"),
forcing a single answer from the 11 basic colours. Full images are fed with InternVL's
standard dynamic tiling so small parts keep enough resolution.

Reuses ``load_model`` / ``parse_color`` / ``BASIC_COLORS`` from ``annotate_colors_internvl``.

Inference is sharded for a SLURM array exactly like the annotator: shard K processes
every row with ``question_index % num_shards == K`` and writes
``preds_vqa/predictions_shard_K.csv`` (resumable via ``--skip-done``).

Run (single GPU smoke test):
    conda run -n internvl_env python eval_paco_vqa_internvl.py \
        --in-csv FG-datset/coco_paco/parts_color_vqa_paco_val.csv \
        --out-dir /tmp/vqa_smoke --num-shards 1 --limit 25

Score (CPU, after the array finishes):
    conda run -n internvl_env python eval_paco_vqa_internvl.py \
        --in-csv FG-datset/coco_paco/parts_color_vqa_paco_val.csv \
        --out-dir FG-datset/coco_paco/preds_vqa --score
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from annotate_colors_internvl import (
    BASIC_COLORS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_model,
    parse_color,
)

# ---------------------------------------------------------------------------
# Full-image dynamic tiling (canonical InternVL model-card preprocessing).
# A single 448 resize would lose small parts in a full scene; tiling keeps them.
# ---------------------------------------------------------------------------

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    cols = target_width // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12, dtype=torch.bfloat16):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values.to(dtype)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_COLOR_LIST = ", ".join(BASIC_COLORS)


def build_prompt(query: str) -> str:
    return (
        "<image>\n"
        f"{query} Choose exactly one color from this list and answer with that one "
        f"word only: {_COLOR_LIST}. Do not explain."
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(args):
    df = pd.read_csv(args.in_csv)
    df = df[df["question_index"] % args.num_shards == args.shard].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"predictions_shard_{args.shard}.csv")

    done = set()
    if args.skip_done and os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = set(prev["question_index"].tolist())
        print(f"Resuming: {len(done)} rows already done in {out_path}", flush=True)

    todo = [r for r in df.itertuples(index=False) if r.question_index not in done]
    print(f"Shard {args.shard}/{args.num_shards}: {len(todo)} questions to run "
          f"({len(df)} total in shard).", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    model, tokenizer = load_model()
    gen_cfg = dict(max_new_tokens=10, do_sample=False)

    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["question_index", "pred_color", "raw_response"])

    n_unknown = 0
    for i, r in enumerate(todo, 1):
        qid = int(r.question_index)
        img_path = r.image_name
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            writer.writerow([qid, "unknown", "MISSING_IMAGE"])
            n_unknown += 1
            continue
        try:
            pv = load_image(img_path, max_num=args.max_num).cuda()
            with torch.no_grad():
                resp = model.chat(tokenizer, pv, build_prompt(r.query), gen_cfg)
        except Exception as e:  # noqa: BLE001
            writer.writerow([qid, "unknown", f"ERROR:{type(e).__name__}:{e}"])
            n_unknown += 1
            continue
        color = parse_color(resp)
        if color == "unknown":
            n_unknown += 1
        writer.writerow([qid, color, resp.strip().replace("\n", " ")[:80]])
        if i % args.flush_every == 0:
            f.flush()
            print(f"  [{args.shard}] {i}/{len(todo)}  unknown={n_unknown}", flush=True)

    f.flush()
    f.close()
    print(f"Shard {args.shard} done. {len(todo)} run, {n_unknown} unknown. → {out_path}",
          flush=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _accuracy_block(name, gt, pred):
    """Return a printable report comparing pred vs a ground-truth column.

    Rows where gt is unknown/NaN are excluded. Reports accuracy both over all
    usable-gt rows (unknown predictions count as wrong) and over the subset the
    model actually answered (pred != unknown).
    """
    usable = gt.notna() & (gt != "unknown")
    gt, pred = gt[usable], pred[usable]
    n = len(gt)
    answered = pred != "unknown"
    n_answered = int(answered.sum())
    correct = (pred == gt)
    acc_all = correct.sum() / n if n else 0.0
    acc_ans = correct[answered].sum() / n_answered if n_answered else 0.0
    lines = [
        f"=== Accuracy vs {name} ===",
        f"  usable-gt rows           : {n}",
        f"  answered (pred != unknown): {n_answered}  ({n - n_answered} unparseable)",
        f"  accuracy (all usable)    : {acc_all:.4f}   ({int(correct.sum())}/{n})",
        f"  accuracy (answered only) : {acc_ans:.4f}   ({int(correct[answered].sum())}/{n_answered})",
    ]
    # per-color accuracy (over usable-gt rows, unknown pred = wrong)
    lines.append("  per-color accuracy (gt class -> acc, support):")
    for c in BASIC_COLORS:
        m = gt == c
        sup = int(m.sum())
        if sup:
            lines.append(f"    {c:<7}: {correct[m].sum() / sup:.3f}  (n={sup})")
    return "\n".join(lines)


def _confusion_matrix(gt, pred):
    """11x11 text confusion matrix over rows where both gt & pred are basic colors."""
    cols = BASIC_COLORS
    idx = {c: i for i, c in enumerate(cols)}
    m = [[0] * len(cols) for _ in cols]
    mask = gt.isin(cols) & pred.isin(cols)
    for g, p in zip(gt[mask], pred[mask]):
        m[idx[g]][idx[p]] += 1
    short = [c[:3] for c in cols]
    header = "gt\\pred  " + " ".join(f"{s:>4}" for s in short)
    rows = [header]
    for i, c in enumerate(cols):
        rows.append(f"{c:<7} " + " ".join(f"{m[i][j]:>4}" for j in range(len(cols))))
    return "=== Confusion matrix (gt rows x pred cols, basic colors only) ===\n" + "\n".join(rows)


def score(args):
    shard_files = sorted(glob.glob(os.path.join(args.out_dir, "predictions_shard_*.csv")))
    if not shard_files:
        raise SystemExit(f"No prediction shards found in {args.out_dir}")
    preds = pd.concat([pd.read_csv(p) for p in shard_files], ignore_index=True)
    preds = preds.drop_duplicates("question_index", keep="last")
    print(f"Loaded {len(preds)} predictions from {len(shard_files)} shard file(s).")

    df = pd.read_csv(args.in_csv)
    keep = ["question_index", "query", "label", "geom_label"]
    merged = df[keep].merge(preds[["question_index", "pred_color", "raw_response"]],
                            on="question_index", how="inner")
    print(f"Merged {len(merged)} rows (of {len(df)} in {os.path.basename(args.in_csv)}).")

    report = [
        f"PACO color-VQA on InternVL3-14B (full image + question)",
        f"input csv : {args.in_csv}",
        f"preds dir : {args.out_dir}",
        f"scored    : {len(merged)} rows",
        "",
        _accuracy_block("label (authoritative GT: PACO color, else crop fallback)",
                        merged["label"], merged["pred_color"]),
        "",
        _accuracy_block("geom_label (PACO-annotated subset only)",
                        merged["geom_label"], merged["pred_color"]),
        "",
        _confusion_matrix(merged["label"], merged["pred_color"]),
    ]
    report_txt = "\n".join(report)
    print("\n" + report_txt)

    metrics_path = os.path.join(args.out_dir, "metrics_val.txt")
    with open(metrics_path, "w") as fh:
        fh.write(report_txt + "\n")

    merged["correct_vs_label"] = (
        (merged["label"].notna()) & (merged["label"] != "unknown")
        & (merged["pred_color"] == merged["label"])
    )
    pred_path = os.path.join(args.out_dir, "predictions_val.csv")
    merged.to_csv(pred_path, index=False)
    print(f"\nWrote {metrics_path}\nWrote {pred_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="FG-datset/coco_paco/parts_color_vqa_paco_val.csv")
    ap.add_argument("--out-dir", default="FG-datset/coco_paco/preds_vqa")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-num", type=int, default=12, help="max InternVL tiles per image")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--score", action="store_true",
                    help="aggregate shard predictions and report metrics (no GPU)")
    args = ap.parse_args()

    if args.score:
        score(args)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
