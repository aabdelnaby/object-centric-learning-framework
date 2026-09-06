"""Extract part-only crops for the VLM colour fallback.

For every row of a ``paco_<split>_masks.csv`` (from ``build_vqa_csv.py``) whose PACO colour is
unknown (``--only-unknown``, the thesis setting) the image and the binary part mask are cropped
to the mask's bounding box (+ margin), every non-part pixel is blacked out, the crop is padded
to a square and resized to ``--size`` (448, the InternVL tile), and saved as
``<out-dir>/<question_index>.jpg``.

    python data_prep/paco/extract_part_crops.py --split train --only-unknown --workers 16
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image


def extract_one(mask_path: str, image_path: str, size: int, margin: float,
                bg: int) -> Image.Image | None:
    """Return a square, part-only crop, or None if the mask is empty/unreadable."""
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
    except Exception:
        return None
    ys, xs = np.where(mask > 127)
    if ys.size == 0:
        return None

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    # Masks can be a slightly different resolution than the image; rescale mask to image.
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
        ys, xs = np.where(mask > 127)
        if ys.size == 0:
            return None

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    mh, mw = y1 - y0, x1 - x0
    py, px = int(round(mh * margin)), int(round(mw * margin))
    y0, y1 = max(0, y0 - py), min(H, y1 + py)
    x0, x1 = max(0, x0 - px), min(W, x1 + px)

    crop = np.asarray(img)[y0:y1, x0:x1].copy()
    m = mask[y0:y1, x0:x1] > 127
    crop[~m] = bg  # black out (or gray out) everything that isn't the part

    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    canvas = np.full((side, side, 3), bg, dtype=np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop

    out = Image.fromarray(canvas)
    if side != size:
        out = out.resize((size, size), Image.LANCZOS)
    return out


def _worker(row, out_dir, size, margin, bg, skip_done):
    qid, mask_path, image_path = row
    out_path = os.path.join(out_dir, f"{qid}.jpg")
    if skip_done and os.path.exists(out_path):
        return ("skip", qid)
    crop = extract_one(mask_path, image_path, size, margin, bg)
    if crop is None:
        return ("empty", qid)
    crop.save(out_path, quality=95)
    return ("ok", qid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--data-dir", default="data/paco", help="directory holding paco_<split>_masks.csv and part_masks/")
    ap.add_argument("--coco-root", default="data/coco", help="prefix for relative image paths")
    ap.add_argument("--out-dir", default=None, help="default: <data-dir>/part_crops/<split>")
    ap.add_argument("--only-unknown", action="store_true",
                    help="only parts without a PACO colour (geom_label == unknown); the thesis setting")
    ap.add_argument("--size", type=int, default=448, help="Output square side (px).")
    ap.add_argument("--margin", type=float, default=0.08, help="Bbox padding as a fraction.")
    ap.add_argument("--bg", type=int, default=0, help="Background fill (0=black, 128=gray).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()

    mask_index = os.path.join(args.data_dir, f"paco_{args.split}_masks.csv")
    out_dir = args.out_dir or os.path.join(args.data_dir, "part_crops", args.split)
    df = pd.read_csv(mask_index)
    if args.only_unknown:
        df = df[df["geom_label"].fillna("unknown") == "unknown"]
    if args.limit:
        df = df.head(args.limit)
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir

    def resolve(path, root):
        return path if os.path.isabs(path) else os.path.join(root, path)

    img_paths = [resolve(p, args.coco_root) for p in df["image_path"]]
    mask_paths = [resolve(p, args.data_dir) for p in df["part_mask_path"]]
    rows = list(zip(df["question_index"], mask_paths, img_paths))
    print(f"Extracting {len(rows):,} crops → {args.out_dir} "
          f"(size={args.size}, workers={args.workers})")

    fn = partial(_worker, out_dir=args.out_dir, size=args.size, margin=args.margin,
                 bg=args.bg, skip_done=args.skip_done)

    counts = {"ok": 0, "skip": 0, "empty": 0}
    empties = []
    with Pool(args.workers) as pool:
        for i, (status, qid) in enumerate(pool.imap_unordered(fn, rows, chunksize=32), 1):
            counts[status] += 1
            if status == "empty":
                empties.append(int(qid))
            if i % 2000 == 0:
                print(f"  {i:,}/{len(rows):,}  ok={counts['ok']:,} "
                      f"skip={counts['skip']:,} empty={counts['empty']:,}")

    print(f"\nDone. ok={counts['ok']:,} skip={counts['skip']:,} empty(no mask px)={counts['empty']:,}")
    if empties:
        ep = os.path.join(args.out_dir, "_empty_question_indices.txt")
        with open(ep, "w") as f:
            f.write("\n".join(map(str, sorted(empties))))
        print(f"Wrote {len(empties)} empty-mask question indices → {ep}")


if __name__ == "__main__":
    main()
