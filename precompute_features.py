"""Precompute frozen ViT (DINOSAUR) and RoBERTa-Large features for any
CSV-driven attribute-VQA dataset (CUB-200-2011 or Super-CLEVR-3D).

Running training with pre-computed features skips the two large frozen encoders
every step, making each epoch ~20× faster.  Run this script once before the
n_slots sweep.

Usage (from repo root):
    # CUB defaults (back-compat):
    conda run -n oclf_env python precompute_features.py

    # Super-CLEVR-3D, custom slot checkpoint:
    conda run -n oclf_env python precompute_features.py \\
        --csv FG-datset/superclevr3d/parts_vqa.csv \\
        --image_root FG-datset/superclevr3d/images \\
        --dino_cache_out FG-datset/superclevr3d/dino_feat_cache.pt \\
        --text_cache_out FG-datset/superclevr3d/text_feat_cache.pt \\
        --dino_cfg projects/bridging/dinosaur/superclevr3d_feat_rec_dino_small16_dinov3 \\
        --dino_ckpt outputs/superclevr3d_dinov3/slots_12/<jobid>/checkpoints/last.ckpt \\
        --skip_text   # text cache for ~315k unique questions is huge — usually skip

Outputs
-------
<dino_cache_out>
    {
      "features":  {image_name (str): tensor (200, 384)},
      "positions": tensor (200, …)
    }
    Note: DINOv3 (vit_small_patch16_dinov3) strips CLS but keeps 4 register
    tokens, giving 4 + 196 = 200 tokens per image.

<text_cache_out>
    {
      "hidden": {query (str): tensor (64, 1024)},
      "masks":  {query (str): tensor (64,)}
    }
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults (CUB; overridable via CLI for Super-CLEVR-3D etc.)
# ---------------------------------------------------------------------------

REPO_ROOT      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH       = "FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv"
IMAGE_ROOT     = "FG-datset/CUB_200_2011/images"
DINO_CACHE_OUT = "FG-datset/CUB_200_2011/dino_feat_cache.pt"
FTDINO_CACHE_OUT = "FG-datset/CUB_200_2011/ftdino_feat_cache.pt"
TEXT_CACHE_OUT = "FG-datset/CUB_200_2011/text_feat_cache.pt"

DINO_CFG       = "projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto_dinov3"
DINO_CKPT      = "checkpoints/dinov3/epoch_20-step_155206.ckpt"
ROBERTA_MODEL  = "roberta-large"
T5_MODEL       = "t5-base"
MAX_TEXT_LEN   = 64
BATCH_SIZE     = 64

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]


def _pad_to_square(t: "torch.Tensor") -> "torch.Tensor":
    """Pad a (C,H,W) image tensor to a centred square, filling with the ImageNet
    per-channel mean (≈ neutral / ~0 after Normalize)."""
    c, h, w = t.shape
    m = max(h, w)
    canvas = torch.tensor(IMAGE_MEAN, dtype=t.dtype).view(c, 1, 1).repeat(1, m, m).clone()
    top, left = (m - h) // 2, (m - w) // 2
    canvas[:, top:top + h, left:left + w] = t
    return canvas


def build_image_transform(img_size: int = 224, resize_mode: str = "crop"):
    """Preprocessing for the (oclf / plain-DINO) backbone. Shared by the in-memory
    DINO precompute and every viz path so train and viz stay byte-for-byte consistent.

    resize_mode:
      "crop"   — Resize(shorter→S) + CenterCrop(S): aspect-preserving, crops the
                 long-axis edges (ImageNet-style; the historical default).
      "square" — Resize((S,S)): keeps ALL content, distorts aspect ratio.
      "pad"    — pad to a centred square (ImageNet-mean fill) + Resize((S,S)):
                 keeps all content AND aspect, at the cost of padding bars.
    """
    bic   = transforms.InterpolationMode.BICUBIC
    clamp = transforms.Lambda(lambda x: x.clamp(0.0, 1.0))
    if resize_mode == "crop":
        geo = [transforms.Resize(img_size, interpolation=bic), clamp,
               transforms.CenterCrop(img_size)]
    elif resize_mode == "square":
        geo = [transforms.Resize((img_size, img_size), interpolation=bic), clamp]
    elif resize_mode == "pad":
        geo = [transforms.Lambda(_pad_to_square),
               transforms.Resize((img_size, img_size), interpolation=bic), clamp]
    else:
        raise ValueError(f"unknown resize_mode={resize_mode!r}; expected crop|square|pad")
    return transforms.Compose(
        [transforms.ToTensor()] + geo + [transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)]
    )


# ---------------------------------------------------------------------------
# Image helper dataset
# ---------------------------------------------------------------------------

class _UniqueImageDataset(Dataset):
    """Yields (image_name, image_tensor) for every unique image in the CSV."""

    def __init__(self, image_names: list[str], image_root: str, img_size: int = 224,
                 transform=None, resize_mode: str = "crop"):
        self.names = image_names
        self.root  = image_root
        if transform is not None:
            # e.g. ftdinosaur's own build_preprocessing — match the encoder's
            # training preprocessing.
            self.transform = transform
        else:
            self.transform = build_image_transform(img_size, resize_mode)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img  = Image.open(os.path.join(self.root, name)).convert("RGB")
        return name, self.transform(img)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            default=CSV_PATH)
    p.add_argument("--image_root",     default=IMAGE_ROOT)
    p.add_argument("--dino_cache_out", default=DINO_CACHE_OUT)
    p.add_argument("--text_cache_out", default=TEXT_CACHE_OUT)
    p.add_argument("--slot_backend",   default="oclf", choices=["oclf", "ftdinosaur"],
                   help="'oclf' caches OCLF ViT patch features (re-run through "
                        "conditioning+pg at train time). 'ftdinosaur' caches the "
                        "ftdinosaur ViT-B/14 encoder features (256x768); slot "
                        "attention is re-run live at train time.")
    p.add_argument("--ftdinosaur_model",
                   default="dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k",
                   help="ftdinosaur checkpoint name (used when --slot_backend ftdinosaur).")
    p.add_argument("--dino_cfg",       default=DINO_CFG)
    p.add_argument("--dino_ckpt",      default=DINO_CKPT)
    p.add_argument("--n_slots",        type=int, default=7,
                   help="Slot count to instantiate while loading checkpoint.")
    p.add_argument("--text_encoder", default="roberta", choices=["roberta", "t5"],
                   help="Text encoder for the text cache: 'roberta' (RoBERTa-Large, 1024-d) "
                        "or 't5' (T5-base encoder, 768-d, paper-faithful). When 't5' and "
                        "--text_cache_out is left at the default, the output path is "
                        "suffixed with _t5.")
    p.add_argument("--with_spans", action="store_true", default=False,
                   help="Also cache per-query part <x> / object <y> span vectors "
                        "(x_vec/y_vec) for the hier_router head.")
    p.add_argument("--span_dataset", default="superclevr3d",
                   choices=["superclevr3d", "ade20k", "cub"],
                   help="Question template used to parse spans when --with_spans: "
                        "superclevr3d/ade20k ('color of <x> of <y>') or cub "
                        "('What is the <part> <attribute> of the bird?').")
    p.add_argument("--skip_text", action="store_true", default=False,
                   help="Skip the text cache (useful when there are 100k+ unique queries).")
    p.add_argument("--img_size",       type=int, default=224,
                   help="Input image resolution; must match the slot ckpt training resolution.")
    p.add_argument("--resize_mode",    choices=["crop", "square", "pad"], default="crop",
                   help="Preprocessing: crop (Resize+CenterCrop, default), square (Resize((S,S)), "
                        "no crop, distorts aspect), or pad (pad-to-square then resize). Changes the "
                        "ViT features → use a distinct --dino_cache_out per mode.")
    p.add_argument("--batch_size",     type=int, default=BATCH_SIZE,
                   help=f"DataLoader batch size for ViT forward (default {BATCH_SIZE}).")
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--fp16",  action="store_true", default=False,
                   help="Store ViT features as float16 (halves disk + RAM, ~no accuracy hit on frozen feats).")
    p.add_argument("--attribute_filter", type=str, default=None,
                   help="Optional: only cache images whose rows match this attribute_type.")
    p.add_argument("--depth_filter",     type=int, default=None,
                   help="Optional: only cache images whose rows match this depth.")
    p.add_argument("--max_n_objects",    type=int, default=None,
                   help="Optional: only cache images with n_objects <= this.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def precompute_dino(
    feature_extractor,
    image_names: list[str],
    image_root: str,
    device: torch.device,
    img_size: int = 224,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    fp16: bool = False,
    resize_mode: str = "crop",
) -> dict:
    """Run frozen ViT on all unique images; return {image_name: (N, d_vit)}."""
    ds     = _UniqueImageDataset(image_names, image_root, img_size=img_size,
                                 resize_mode=resize_mode)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device.type == "cuda"),
        collate_fn  = lambda b: (
            [x[0] for x in b],
            torch.stack([x[1] for x in b]),
        ),
    )

    feat_cache = {}
    positions  = None
    store_dtype = torch.float16 if fp16 else torch.float32

    feature_extractor = feature_extractor.to(device)

    print(f"  Computing ViT features for {len(image_names)} images at {img_size}x{img_size} "
          f"(resize_mode={resize_mode}, batch={batch_size}, dtype={store_dtype}) …")
    with torch.no_grad():
        for names, imgs in tqdm(loader, unit="batch"):
            imgs = imgs.to(device)
            routing = {"input": {"image": imgs, "batch_size": imgs.shape[0]}}
            feat_out = feature_extractor(inputs=routing)

            feats = feat_out.features.to(store_dtype).cpu()   # (B, N, d_vit)
            if positions is None:
                positions = feat_out.positions.cpu()          # (N, …) keep fp32

            for name, f in zip(names, feats):
                feat_cache[name] = f

    return {"features": feat_cache, "positions": positions}


def precompute_ftdinosaur_encoder(
    model,
    preproc,
    image_names: list[str],
    image_root: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    fp16: bool = False,
) -> dict:
    """Run the frozen ftdinosaur ViT-B/14 encoder on all unique images.

    Caches the *encoder* features (B, num_patches=256, 768), NOT slots: slot
    attention is cheap and is re-run live at train time so the random slot init
    stays fresh each epoch. Returns {image_name: (256, 768)}.
    """
    ds     = _UniqueImageDataset(image_names, image_root, transform=preproc)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device.type == "cuda"),
        collate_fn  = lambda b: (
            [x[0] for x in b],
            torch.stack([x[1] for x in b]),
        ),
    )

    feat_cache  = {}
    store_dtype = torch.float16 if fp16 else torch.float32
    model       = model.to(device).eval()

    print(f"  Computing ftdinosaur encoder features for {len(image_names)} images "
          f"(batch={batch_size}, dtype={store_dtype}) …")
    with torch.no_grad():
        for names, imgs in tqdm(loader, unit="batch"):
            imgs  = imgs.to(device)
            feats = model.encoder(imgs)                 # (B, 256, 768)
            feats = feats.to(store_dtype).cpu()
            for name, f in zip(names, feats):
                feat_cache[name] = f

    # positions=None: ftdinosaur bakes positional info into the encoder output,
    # so the cached path needs no separate positional grid (unlike the OCLF path).
    return {"features": feat_cache, "positions": None}


# Templated part-question parser, shared with the datasets / hier_router pipeline.
# Matches "...color of the <x> of the <y>?" → (part <x>, object <y>).
_XY_PHRASE_RE = re.compile(r"color of the (.+?) of the (.+?)\s*\?*\s*$", re.IGNORECASE)


def parse_xy_phrases(query: str):
    """Parse 'What is the color of the <x> of the <y>?' → (x_phrase, y_phrase).

    Returns ``None`` if the query does not match the part-of-object template
    (e.g. a depth-1 question), so callers can fall back to zero span vectors.
    """
    m = _XY_PHRASE_RE.search(str(query).strip())
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


# CUB-200 question parser. CUB queries are "What is the <part> <attribute> of the
# bird?" (e.g. "What is the back color of the bird?"). Unlike the ADE/SC3D
# "color of <x> of <y>" template, the object <y> is always "bird" and the
# attribute (color/pattern/shape/...) is part of the phrase — and it is what
# distinguishes "back color" from "back pattern" (both route to the same part).
_CUB_ATTRS   = {"color", "pattern", "length", "shape", "size"}
_CUB_RE      = re.compile(r"what is the (.+?) of the bird\s*\?*\s*$", re.IGNORECASE)


def parse_cub_xy_phrases(query: str):
    """Parse a CUB query 'What is the <part> <attribute> of the bird?' →
    ``(part, "bird", attribute)``.

    The object ``<y>`` is always ``"bird"``; the part ``<x>`` is the body part
    naming the region to route to (e.g. "back", "under tail"); ``attribute`` names
    the property asked (color/pattern/length/shape/size) and is what disambiguates
    e.g. "back color" from "back pattern" (both route to the same part).

    Whole-bird questions with no body part ("What is the shape/size of the bird?")
    return ``part=""`` (routing degenerates to the whole bird). Returns ``None`` if
    the query doesn't match the CUB template, so callers fall back to zero spans.
    """
    m = _CUB_RE.search(str(query).strip())
    if not m:
        return None
    middle = m.group(1).strip()
    toks   = middle.split()
    if len(toks) > 1 and toks[-1].lower() in _CUB_ATTRS:
        return " ".join(toks[:-1]), "bird", toks[-1].lower()
    # Whole-bird attribute (e.g. "shape", "size") — no body part.
    return "", "bird", middle


def _span_phrases(query: str, span_dataset: str):
    """Per-query phrase strings for the 4 hier_router text channels, or ``None``
    when the query doesn't match the dataset template (→ all-zero spans).

    Returns ``(x_phrase, y_phrase, xy_phrase, readout_phrase)`` where any element
    may itself be ``None`` to request a zero vector for that single channel
    (e.g. CUB whole-bird questions have no part → x/xy are ``None``):
      - ch0 ``x``       = part ``<x>``
      - ch1 ``y``       = object ``<y>``                 (parent routing)
      - ch2 ``xy``      = parent-only routing query
      - ch3 ``readout`` = child-routing + answer-readout query (carries the attribute on CUB)
    """
    if span_dataset == "cub":
        pr = parse_cub_xy_phrases(query)
        if pr is None:
            return None
        part, obj, attr = pr
        if part:
            return (part, obj, f"{part} of the {obj}", f"{obj} {part} {attr}")
        # Whole-bird attribute: no part → zero x/xy; readout = "bird <attribute>".
        return (None, obj, None, f"{obj} {attr}")
    # Default: ADE20K / Super-CLEVR-3D "color of <x> of <y>".
    pr = parse_xy_phrases(query)
    if pr is None:
        return None
    x, y = pr
    return (x, y, f"{x} of the {y}", f"{y} {x}")


def parse_xy_for(query: str, dataset: str = "superclevr3d"):
    """Dataset-aware ``(x_phrase, y_phrase)`` for hier_router viz labels / filtering.

    Returns the part ``<x>`` and object ``<y>`` display phrases, or ``None`` when the
    query doesn't match the dataset template. (CUB whole-bird questions have no part →
    ``x_phrase=""``.) Thin wrapper over :func:`_span_phrases`.
    """
    ph = _span_phrases(query, dataset)
    if ph is None:
        return None
    return (ph[0] or ""), ph[1]


def precompute_text(
    queries: list[str],
    device: torch.device,
    batch_size: int = 64,
    text_encoder: str = "roberta",
    with_spans: bool = False,
    span_dataset: str = "superclevr3d",
) -> dict:
    """Run a frozen text encoder on all unique queries; return hidden states.

    ``text_encoder='roberta'`` → RoBERTa-Large (1024-d); ``'t5'`` → T5-base
    encoder (768-d). Both produce a ``(query → (L, d_text))`` hidden dict plus a
    ``(query → (L,))`` attention-mask dict, consumed by the *CachedFeatDataset
    classes regardless of which encoder produced them.

    ``with_spans=True`` additionally parses each query into its 4 hier_router text
    channels (``_span_phrases``, dispatched by ``span_dataset`` ∈ {superclevr3d,
    ade20k, cub}), encodes each phrase once and mean-pools over real tokens, and
    returns ``x_vec`` / ``y_vec`` / ``xy_vec`` / ``readout_vec`` dicts
    (``query → (d_text,)``) for the hier_router head. Queries (or single channels)
    that don't match the template get zero vectors.
    """
    if text_encoder == "t5":
        from transformers import AutoTokenizer, T5EncoderModel
        name      = T5_MODEL
        print(f"  Computing T5 features ({name}) for {len(queries)} unique queries (batch={batch_size}) …")
        tokenizer = AutoTokenizer.from_pretrained(name)
        encoder   = T5EncoderModel.from_pretrained(name).to(device).eval()
    else:
        from transformers import RobertaModel, RobertaTokenizer
        name      = ROBERTA_MODEL
        print(f"  Computing RoBERTa features ({name}) for {len(queries)} unique queries (batch={batch_size}) …")
        tokenizer = RobertaTokenizer.from_pretrained(name)
        encoder   = RobertaModel.from_pretrained(name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    hidden_cache = {}
    mask_cache   = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(queries), batch_size), unit="batch"):
            batch_q = queries[start : start + batch_size]
            enc = tokenizer(
                batch_q,
                max_length  = MAX_TEXT_LEN,
                padding     = "max_length",
                truncation  = True,
                return_tensors = "pt",
            )
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out  = encoder(input_ids=ids, attention_mask=mask)
            hidden = out.last_hidden_state.cpu()   # (B, L, d_text)
            mask_cpu = mask.cpu()                  # (B, L)
            for i, q in enumerate(batch_q):
                hidden_cache[q] = hidden[i]
                mask_cache[q]   = mask_cpu[i]

    out = {"hidden": hidden_cache, "masks": mask_cache}

    if with_spans:
        # Parse each query into its 4 hier_router channels (x / y / "<x> of <y>" /
        # readout — see _span_phrases) and encode every distinct phrase once
        # (mean-pooled over real tokens) → per-query span vectors. The phrase set,
        # and which channels carry the part/object/attribute, are dataset-specific.
        parsed  = {q: _span_phrases(q, span_dataset) for q in queries}
        phrases = sorted(
            {p for ph in parsed.values() if ph is not None for p in ph if p is not None}
        )
        n_unmatched = sum(1 for ph in parsed.values() if ph is None)
        if n_unmatched:
            print(f"  [spans] {n_unmatched}/{len(queries)} queries didn't match the "
                  f"'{span_dataset}' span template → zero span vectors.")
        pvec: dict = {}
        with torch.no_grad():
            for start in tqdm(range(0, len(phrases), batch_size), unit="batch", desc="spans"):
                batch_p = phrases[start : start + batch_size]
                enc = tokenizer(
                    batch_p, max_length=MAX_TEXT_LEN, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
                ids  = enc["input_ids"].to(device)
                m    = enc["attention_mask"].to(device)
                feat = encoder(input_ids=ids, attention_mask=m).last_hidden_state  # (b,l,d)
                w    = m.unsqueeze(-1).to(feat.dtype)
                pooled = (feat * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)        # (b,d)
                for p, v in zip(batch_p, pooled.cpu()):
                    pvec[p] = v
        d_text = int(next(iter(pvec.values())).shape[0]) if pvec else 0
        zero   = torch.zeros(d_text)
        # phrase → vec, mapping a missing/None channel (e.g. CUB whole-bird has no part)
        # to a zero vector.
        vec = lambda p: (pvec[p] if p is not None else zero.clone())
        x_vec, y_vec, xy_vec, readout_vec = {}, {}, {}, {}
        for q, ph in parsed.items():
            if ph is None:
                x_vec[q], y_vec[q] = zero.clone(), zero.clone()
                xy_vec[q], readout_vec[q] = zero.clone(), zero.clone()
            else:
                x_vec[q], y_vec[q]        = vec(ph[0]), vec(ph[1])
                xy_vec[q], readout_vec[q] = vec(ph[2]), vec(ph[3])
        out["x_vec"], out["y_vec"] = x_vec, y_vec
        out["xy_vec"], out["readout_vec"] = xy_vec, readout_vec

    return out


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.csv):
        print(f"CSV not found at {args.csv}. Run from repo root.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if args.attribute_filter is not None and "attribute_type" in df.columns:
        df = df[df["attribute_type"] == args.attribute_filter]
    if args.depth_filter is not None and "depth" in df.columns:
        df = df[df["depth"] == args.depth_filter]
    if args.max_n_objects is not None and "n_objects" in df.columns:
        df = df[df["n_objects"] <= args.max_n_objects]
    unique_images  = df["image_name"].unique().tolist()
    unique_queries = df["query"].unique().tolist()
    print(f"Unique images: {len(unique_images)} | Unique queries: {len(unique_queries)}")

    # ftdinosaur writes to its own default cache path so it never clobbers the
    # OCLF cache (different shape: 256x768 vs 200x384).
    if args.slot_backend == "ftdinosaur" and args.dino_cache_out == DINO_CACHE_OUT:
        args.dino_cache_out = FTDINO_CACHE_OUT
        print(f"slot_backend=ftdinosaur → dino_cache_out defaulted to {args.dino_cache_out}")

    # ── Image (ViT) features ──────────────────────────────────────────────
    if os.path.exists(args.dino_cache_out):
        print(f"Image-feature cache already exists at {args.dino_cache_out} — skipping.")
    elif args.slot_backend == "ftdinosaur":
        from classifier_model import _build_ftdinosaur
        from ftdinosaur_inference import build_dinosaur as _bd
        print(f"Loading ftdinosaur model {args.ftdinosaur_model} …")
        model  = _build_ftdinosaur(args.ftdinosaur_model)
        preproc = _bd.build_preprocessing(args.ftdinosaur_model)

        dino_cache = precompute_ftdinosaur_encoder(
            model, preproc, unique_images, args.image_root, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fp16=args.fp16,
        )
        os.makedirs(os.path.dirname(args.dino_cache_out) or ".", exist_ok=True)
        torch.save(dino_cache, args.dino_cache_out)
        print(f"  Saved → {args.dino_cache_out}  "
              f"({len(dino_cache['features'])} entries, "
              f"feat shape: {next(iter(dino_cache['features'].values())).shape})")
    else:
        from classifier_model import _load_dinosaur_submodules
        print("Loading DINOSAUR feature extractor …")
        fe, _, _ = _load_dinosaur_submodules(args.dino_cfg, args.dino_ckpt, REPO_ROOT, n_slots=args.n_slots)
        fe.eval()

        dino_cache = precompute_dino(
            fe, unique_images, args.image_root, device,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fp16=args.fp16,
            resize_mode=args.resize_mode,
        )
        os.makedirs(os.path.dirname(args.dino_cache_out) or ".", exist_ok=True)
        torch.save(dino_cache, args.dino_cache_out)
        print(f"  Saved → {args.dino_cache_out}  "
              f"({len(dino_cache['features'])} entries, "
              f"feat shape: {next(iter(dino_cache['features'].values())).shape})")

    # ── Text features (RoBERTa or T5) ─────────────────────────────────────
    # T5 hidden states are 768-d vs RoBERTa's 1024-d, so default the t5 run to a
    # *_t5.pt path to avoid clobbering a RoBERTa cache.
    if args.text_encoder == "t5" and args.text_cache_out == TEXT_CACHE_OUT:
        args.text_cache_out = (
            TEXT_CACHE_OUT[:-3] + "_t5.pt" if TEXT_CACHE_OUT.endswith(".pt")
            else TEXT_CACHE_OUT + "_t5"
        )
        print(f"text_encoder=t5 → text_cache_out defaulted to {args.text_cache_out}")

    if args.skip_text:
        print("Skipping text cache (--skip_text).")
    elif os.path.exists(args.text_cache_out):
        print(f"Text cache already exists at {args.text_cache_out} — skipping.")
    else:
        text_cache = precompute_text(unique_queries, device, text_encoder=args.text_encoder,
                                     with_spans=args.with_spans, span_dataset=args.span_dataset)
        os.makedirs(os.path.dirname(args.text_cache_out) or ".", exist_ok=True)
        torch.save(text_cache, args.text_cache_out)
        span_note = "  (+ x_vec/y_vec spans)" if args.with_spans else ""
        print(f"  Saved → {args.text_cache_out}  ({len(text_cache['hidden'])} queries){span_note}")

    print("Done.")


if __name__ == "__main__":
    main()
