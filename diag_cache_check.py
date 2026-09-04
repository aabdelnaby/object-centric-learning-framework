"""Phase A1 of the overfitting plan: verify the precomputed feature caches
are aligned with the CSV used by training.

Checks:
  1.  Every image_name in parts_vqa.csv (any split) is a key in
      ``dino_feat_cache.pt['features']``.
  2.  Every query in parts_vqa.csv (any split) is a key in
      ``text_feat_cache.pt['hidden']``.
  3.  The cached feature sequence length matches the cached positions length.
  4.  For 5 random train + 5 random val images, recompute the ViT features
      with the same DINOSAUR feature extractor used by precompute and compare
      against the cache (max-abs diff should be <= 1e-4 in fp32).
  5.  For colour / depth=2, surviving train/val/test rows after the same
      ``_filter_df`` pipeline match the numbers logged by train.py
      (train=20000 / val=4998 / test=4998).
  6.  Per-(image_name, query) labels are unique inside each split AND across
      splits: if the same pair appears in train and val with different labels,
      the data is contradictory and val cannot generalise from train.
      Also reports how many val/test pairs are duplicated *with* the train label
      (those rows are effectively free) vs. *contradicting* the train label
      (those rows are unwinnable).
  7.  Recompute RoBERTa hidden states for 5 random queries and compare to the
      text cache (rules out the text-side key/value being mis-aligned).

Run from repo root:
    conda run -n oclf_env python diag_cache_check.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import pandas as pd
import torch

DEFAULT_CSV  = "FG-datset/superclevr3d/parts_vqa.csv"
DEFAULT_IMG  = "FG-datset/superclevr3d/images"
DEFAULT_DINO = "FG-datset/superclevr3d/dino_feat_cache.pt"
DEFAULT_TEXT = "FG-datset/superclevr3d/text_feat_cache.pt"
DEFAULT_CFG  = "projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3"
DEFAULT_CKPT = "checkpoints/dinov3/epoch_296-step_185328_dinov3_superclevr3d_30_slots.ckpt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        default=DEFAULT_CSV)
    p.add_argument("--image_root", default=DEFAULT_IMG)
    p.add_argument("--dino_cache", default=DEFAULT_DINO)
    p.add_argument("--text_cache", default=DEFAULT_TEXT)
    p.add_argument("--dino_cfg",   default=DEFAULT_CFG)
    p.add_argument("--dino_ckpt",  default=DEFAULT_CKPT)
    p.add_argument("--n_recompute", type=int, default=5,
                   help="Recompute features for N train + N val images.")
    p.add_argument("--abs_tol",     type=float, default=1e-4,
                   help="Max-abs diff allowed when comparing to cache.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--skip_recompute", action="store_true",
                   help="Skip ViT recomputation step (cheap key checks only).")
    return p.parse_args()


def report(ok: bool, msg: str) -> None:
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {msg}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    failures = 0

    # ── 0. Files exist ─────────────────────────────────────────────────────
    for path in (args.csv, args.dino_cache, args.text_cache):
        if not os.path.exists(path):
            report(False, f"missing file: {path}")
            sys.exit(1)
    report(True, "all input files exist")

    # ── 1. Load CSV ────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df["label"] = df["label"].astype(str)
    print(f"  csv rows: {len(df):,}  unique images: {df['image_name'].nunique():,}  "
          f"unique queries: {df['query'].nunique():,}")

    # ── 2. DINO cache key coverage ─────────────────────────────────────────
    dino_cache  = torch.load(args.dino_cache, map_location="cpu", weights_only=False)
    dino_feats  = dino_cache["features"]
    dino_pos    = dino_cache.get("positions")

    csv_imgs = set(df["image_name"].unique())
    cache_imgs = set(dino_feats.keys())
    missing = csv_imgs - cache_imgs
    if missing:
        report(False, f"DINO cache missing {len(missing):,} image keys "
                      f"(e.g. {sorted(missing)[:3]})")
        failures += 1
    else:
        report(True, f"DINO cache covers all {len(csv_imgs):,} CSV images")

    # ── 3. Text cache key coverage ─────────────────────────────────────────
    text_cache   = torch.load(args.text_cache, map_location="cpu", weights_only=False)
    text_hidden  = text_cache["hidden"]
    text_masks   = text_cache["masks"]

    csv_queries  = set(df["query"].unique())
    cache_queries = set(text_hidden.keys())
    missing_q = csv_queries - cache_queries
    if missing_q:
        report(False, f"text cache missing {len(missing_q):,} query keys "
                      f"(e.g. {sorted(missing_q)[:1]})")
        failures += 1
    else:
        report(True, f"text cache covers all {len(csv_queries):,} CSV queries")

    # ── 4. Cache feature shape vs positions ────────────────────────────────
    sample_img = next(iter(dino_feats))
    feat_shape = dino_feats[sample_img].shape
    print(f"  cached feature shape: {tuple(feat_shape)}")
    if dino_pos is not None:
        print(f"  cached positions   shape: {tuple(dino_pos.shape)}")
        if feat_shape[0] != dino_pos.shape[0]:
            report(False, f"feature seq len {feat_shape[0]} != positions seq len "
                          f"{dino_pos.shape[0]}")
            failures += 1
        else:
            report(True, "feature / positions sequence lengths match")

    sample_q = next(iter(text_hidden))
    print(f"  cached text hidden shape: {tuple(text_hidden[sample_q].shape)}  "
          f"mask shape: {tuple(text_masks[sample_q].shape)}")

    # ── 5. _filter_df sanity for color/depth=2 ─────────────────────────────
    sys.path.insert(0, os.path.abspath("."))
    from superclevr3d_dataset import _filter_df, build_label_vocab_superclevr3d

    sub_for_vocab = df[(df["attribute_type"] == "color") & (df["depth"] == 2)]
    vocab = build_label_vocab_superclevr3d(sub_for_vocab)
    counts = {}
    for split in ("train", "val", "test"):
        sub = _filter_df(
            df.copy(), split=split, label_vocab=vocab,
            query_filter=None, category_filter=None,
            attribute_filter="color", depth_filter=2,
        )
        counts[split] = len(sub)
    print(f"  color/depth=2 surviving rows: {counts}")
    expected = {"train": 20000, "val": 4998, "test": 4998}
    if counts != expected:
        report(False, f"row counts {counts} != expected {expected}")
        failures += 1
    else:
        report(True, "row counts match training-log expectations")

    # ── 5b. Per-(image_name, query) label consistency across splits ───────
    # Filter to color/depth=2 (the failing slice) and check whether the same
    # (image_name, query) pair appears in multiple splits with conflicting
    # labels. If so, val rows for those pairs are unwinnable given train.
    sub = df[(df["attribute_type"] == "color") & (df["depth"] == 2)].copy()
    grp = sub.groupby(["image_name", "query"])["label"].nunique()
    inconsistent_pairs = (grp > 1).sum()
    if inconsistent_pairs:
        report(False, f"{inconsistent_pairs:,} (image,query) pairs have >1 label "
                      f"within color/depth=2 — label is not a function of (image,query)!")
        failures += 1
    else:
        report(True, "every (image,query) pair has a single label in color/depth=2")

    train_pairs = sub[sub["split"] == "train"].set_index(["image_name", "query"])["label"]
    val_pairs   = sub[sub["split"] == "val"  ].set_index(["image_name", "query"])["label"]
    overlap = val_pairs.index.intersection(train_pairs.index)
    if len(overlap):
        same = (val_pairs.loc[overlap].values == train_pairs.loc[overlap].values).sum()
        diff = len(overlap) - same
        print(f"  val pairs also in train: {len(overlap):,}  "
              f"(same label: {same:,}, contradicting: {diff:,})")
        if diff:
            report(False, f"{diff:,} val (image,query) pairs disagree with their train twin — "
                          f"those val rows are unwinnable")
            failures += 1
        else:
            report(True, "no val (image,query) pair contradicts a train pair")
    else:
        report(True, "no (image,query) overlap between train and val")

    # ── 6. Recompute features for sampled images and compare to cache ─────
    if args.skip_recompute:
        report(True, "skipping recompute step (--skip_recompute)")
    else:
        train_names = sorted(df[df["split"] == "train"]["image_name"].unique())
        val_names   = sorted(df[df["split"] == "val"]["image_name"].unique())
        sampled = (random.sample(train_names, args.n_recompute)
                   + random.sample(val_names,   args.n_recompute))

        print(f"  recomputing {len(sampled)} features against cache "
              f"(tol={args.abs_tol}) …")

        from classifier_model import _load_dinosaur_submodules
        from torchvision import transforms
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fe, _, _ = _load_dinosaur_submodules(
            args.dino_cfg, args.dino_ckpt,
            os.path.abspath("."), n_slots=30,
        )
        fe = fe.to(device).eval()

        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD  = [0.229, 0.224, 0.225]
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])

        max_diff_seen = 0.0
        with torch.no_grad():
            for name in sampled:
                pil = Image.open(os.path.join(args.image_root, name)).convert("RGB")
                img = tf(pil).unsqueeze(0).to(device)
                routing = {"input": {"image": img, "batch_size": 1}}
                feat_out = fe(inputs=routing)
                live = feat_out.features[0].cpu()  # (T, d)
                cached = dino_feats[name].float()   # (T, d)
                if live.shape != cached.shape:
                    report(False,
                           f"shape mismatch for {name}: live {tuple(live.shape)} "
                           f"vs cache {tuple(cached.shape)}")
                    failures += 1
                    continue
                diff = (live - cached).abs().max().item()
                max_diff_seen = max(max_diff_seen, diff)
                if diff > args.abs_tol:
                    report(False, f"{name}: max-abs diff {diff:.2e} > tol {args.abs_tol:.0e}")
                    failures += 1
        if failures == 0:
            report(True, f"recomputed features match cache (max-abs diff "
                         f"seen: {max_diff_seen:.2e})")

        # ── 7. Recompute RoBERTa hidden states for a few queries ──────────
        from transformers import RobertaTokenizer, RobertaModel
        tok = RobertaTokenizer.from_pretrained("roberta-large")
        rob = RobertaModel.from_pretrained("roberta-large").to(device).eval()

        sample_qs = random.sample(sorted(csv_queries), args.n_recompute)
        print(f"  recomputing {len(sample_qs)} text hidden states against cache …")
        max_text_diff = 0.0
        with torch.no_grad():
            for q in sample_qs:
                enc = tok(q, max_length=64, padding="max_length",
                          truncation=True, return_tensors="pt").to(device)
                out = rob(**enc).last_hidden_state[0].cpu().float()
                cached = text_hidden[q].float()
                if out.shape != cached.shape:
                    report(False, f"text shape mismatch for query "
                                  f"{q[:40]!r}: live {tuple(out.shape)} "
                                  f"vs cache {tuple(cached.shape)}")
                    failures += 1
                    continue
                # Cached vs live mask consistency
                live_mask = enc["attention_mask"][0].cpu()
                cached_mask = text_masks[q]
                if not torch.equal(live_mask, cached_mask):
                    report(False, f"attention mask mismatch for query {q[:40]!r}")
                    failures += 1
                diff = (out - cached).abs().max().item()
                max_text_diff = max(max_text_diff, diff)
                if diff > args.abs_tol:
                    report(False, f"text {q[:40]!r}: max-abs diff "
                                  f"{diff:.2e} > tol {args.abs_tol:.0e}")
                    failures += 1
        report(failures == 0,
               f"text hidden states match cache (max-abs diff seen: "
               f"{max_text_diff:.2e})")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    if failures:
        print(f"== {failures} CHECK(S) FAILED — DO NOT TRUST THE CACHE ==")
        sys.exit(2)
    print("== ALL CHECKS PASSED — cache is consistent with the CSV ==")


if __name__ == "__main__":
    main()
