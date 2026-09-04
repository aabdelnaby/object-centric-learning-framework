"""Build a part-color VQA CSV for COCO via PACO-LVIS ground-truth part masks.

Mirrors ``build_ade20k_color_vqa_csv.py`` but sources the object->part hierarchy and
part masks from **PACO-LVIS** (human annotations on the COCO-2017 images already on
disk) instead of ADE20K's native hierarchy. Emits unambiguous questions

    "What is the color of the <part> of the <object>?"

for every (object, part) pair that is visually unambiguous:

  * the OBJECT category appears **exactly once** among the image's PACO object
    instances, and
  * the PART category appears **exactly once** among that object instance's parts
    (parts are linked to their parent object via ``obj_ann_id``).

The trained answer (``label``) is filled later by the InternVL VLM
(``annotate_colors_internvl.py``), exactly as for ADE20K; here ``label`` is left
blank. ``geom_label`` holds PACO's dominant-color attribute mapped to the 11 basic
colors (often ``unknown`` — PACO color-labels only ~1/3 of parts), as a second signal.

The CSV schema matches ``parts_color_vqa_internvl_val30.csv`` so the rows concatenate
into the combined hier_router dataset. A mask-index sidecar + per-part binary mask
PNGs let the existing ``extract_part_crops.py`` build the VLM crops.

Outputs under ``--out-dir`` (default ``FG-datset/coco_paco``):
    parts_color_vqa_paco_<split>.csv        — 14-col VQA schema (label/raw_response blank)
    parts_color_vqa_paco_<split>_masks.csv  — question_index, image_path, part_mask_path, names
    part_masks/<split>/<question_index>.png — binary part mask (0/255)

Run from repo root:
    conda run -n oclf_env python build_paco_color_vqa_csv.py --split val
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as maskUtils

# PACO's 30 color attribute names -> the 11 basic colors (+ unknown). Same target
# set as the InternVL prompt, so geom_label and label share a vocabulary.
PACO_COLOR_TO_BASIC = {
    "black": "black", "white": "white",
    "light_blue": "blue", "blue": "blue", "dark_blue": "blue",
    "light_brown": "brown", "brown": "brown", "dark_brown": "brown",
    "light_green": "green", "green": "green", "dark_green": "green",
    "light_grey": "gray", "grey": "gray", "dark_grey": "gray",
    "light_orange": "orange", "orange": "orange", "dark_orange": "orange",
    "light_pink": "pink", "pink": "pink", "dark_pink": "pink",
    "light_purple": "purple", "purple": "purple", "dark_purple": "purple",
    "light_red": "red", "red": "red", "dark_red": "red",
    "light_yellow": "yellow", "yellow": "yellow", "dark_yellow": "yellow",
    "other(color)": "unknown",
}

# A few PACO object names read awkwardly in a question; keep the rest as-is.
OBJ_ALIASES = {
    "laptop computer": "laptop",
    "cellular telephone": "cell phone",
    "television set": "television",
}

# Parts that typically occur MULTIPLE times on a single object instance (bilateral,
# 4x, repeated, or positionally ambiguous). PACO is federated (it may annotate only
# one of them), so a part can pass the "unique within object" count yet still be
# visually ambiguous ("which leg/ear/side?"). The pointer-free question template
# can't disambiguate these, so we drop them (the draft plan's "singleton parts are
# your friend", §11.2). Matched against the raw PACO part token (before cleaning).
MULTI_INSTANCE_PARTS = {
    "leg", "ear", "eye", "arm", "hand", "foot", "wheel", "pedal", "spoke", "step",
    "string", "button", "key", "prong", "lug", "eyelet", "finger_hole", "hole",
    "teeth", "loop", "bar", "light", "side", "slat", "rivet", "wire", "ear_pads",
    "mirror", "headlight", "taillight", "turnsignal", "fender", "runningboard",
    "window", "windowpane", "antenna",
}


def clean_name(name: str, is_object: bool = False) -> str:
    """'car_(automobile)'->'car'; 'laptop_computer:screen'->'screen' (part portion)."""
    name = name.split(":")[-1]            # for 'object:part' categories keep the part
    if "(" in name:
        name = name.split("(")[0]
    out = name.replace("_", " ").strip()
    if is_object:
        out = OBJ_ALIASES.get(out, out)
    return out


def paco_basic_color(ann, id2attr) -> str:
    """Dominant PACO color of an annotation, mapped to a basic class, or 'unknown'."""
    if ann.get("unknown_color"):
        return "unknown"
    basics = []
    for i in ann.get("dom_color_ids") or []:
        b = PACO_COLOR_TO_BASIC.get(id2attr.get(i))
        if b and b != "unknown":
            basics.append(b)
    basics = list(dict.fromkeys(basics))      # dedupe, preserve order
    return basics[0] if len(basics) == 1 else "unknown"   # multi/empty -> unknown


def ann_to_mask(ann, h: int, w: int) -> np.ndarray:
    """Decode a PACO segmentation (polygon list or RLE) to a HxW uint8 {0,1} mask."""
    seg = ann["segmentation"]
    if isinstance(seg, list):                         # polygon(s)
        rle = maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
    elif isinstance(seg.get("counts"), list):         # uncompressed RLE
        rle = maskUtils.frPyObjects(seg, h, w)
    else:                                             # compressed RLE
        rle = seg
    return maskUtils.decode(rle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--paco-json", default=None,
                    help="Default: <out-dir>/paco_annotations/paco_lvis_v1_<split>.json")
    ap.add_argument("--coco-root", default="scripts/datasets/data/coco",
                    help="Dir the PACO file_name (e.g. 'train2017/000..jpg') is relative to.")
    ap.add_argument("--out-dir", default="FG-datset/coco_paco")
    ap.add_argument("--min-area", type=float, default=400.0,
                    help="Minimum part segment area in px (drop tiny parts).")
    ap.add_argument("--keep-multi-parts", action="store_true",
                    help="Keep typically-repeated parts (leg/ear/wheel/side/…); "
                         "default drops them as visually ambiguous.")
    ap.add_argument("--limit", type=int, default=None, help="First N images (debug).")
    args = ap.parse_args()

    paco_json = args.paco_json or os.path.join(
        args.out_dir, "paco_annotations", f"paco_lvis_v1_{args.split}.json")
    masks_dir = os.path.join(args.out_dir, "part_masks", args.split)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"Loading {paco_json} …", flush=True)
    J = json.load(open(paco_json))
    imgs = {im["id"]: im for im in J["images"]}
    cats = {c["id"]: c["name"] for c in J["categories"]}   # 531: objects + object:part
    id2attr = {a["id"]: a["name"] for a in J["attributes"]}
    obj_cat_ids = {cid for cid, n in cats.items() if ":" not in n}
    part_cat_ids = {cid for cid, n in cats.items() if ":" in n}

    objs_by_img = defaultdict(list)
    parts_by_img = defaultdict(list)
    for a in J["annotations"]:
        if a["category_id"] in obj_cat_ids:
            objs_by_img[a["image_id"]].append(a)
        elif a["category_id"] in part_cat_ids:
            parts_by_img[a["image_id"]].append(a)

    img_ids = list(objs_by_img.keys())
    if args.limit:
        img_ids = img_ids[:args.limit]
    print(f"{len(imgs)} images, {len(J['annotations'])} anns; scanning {len(img_ids)} images.",
          flush=True)

    vqa_rows, mask_rows = [], []
    qi = 0
    n_img_q = 0
    n_empty = 0
    n_multi = 0
    for img_id in img_ids:
        im = imgs[img_id]
        H, W = im["height"], im["width"]
        img_path = os.path.abspath(os.path.join(args.coco_root, im["file_name"]))
        ims_objs = objs_by_img[img_id]

        ocount = Counter(o["category_id"] for o in ims_objs)
        uniq_cat = {cid for cid, c in ocount.items() if c == 1}
        if not uniq_cat:
            continue
        obj_by_id = {o["id"]: o for o in ims_objs if o["category_id"] in uniq_cat}

        parts_by_parent = defaultdict(list)
        for pp in parts_by_img.get(img_id, []):
            if pp.get("obj_ann_id") in obj_by_id:
                parts_by_parent[pp["obj_ann_id"]].append(pp)

        before = qi
        for oaid, plist in parts_by_parent.items():
            pcount = Counter(pp["category_id"] for pp in plist)
            on = clean_name(cats[obj_by_id[oaid]["category_id"]], is_object=True)
            for pp in plist:
                if pcount[pp["category_id"]] != 1:          # part unique within object
                    continue
                part_raw = cats[pp["category_id"]].split(":")[-1]
                if not args.keep_multi_parts and part_raw in MULTI_INSTANCE_PARTS:
                    n_multi += 1                            # visually repeated/ambiguous
                    continue
                if float(pp.get("area", 0.0)) < args.min_area:
                    continue
                pn = clean_name(cats[pp["category_id"]])
                if not on or not pn:
                    continue
                m = ann_to_mask(pp, H, W)
                if int(m.sum()) == 0:
                    n_empty += 1
                    continue

                mask_path = os.path.abspath(os.path.join(masks_dir, f"{qi}.png"))
                Image.fromarray((m * 255).astype(np.uint8)).save(mask_path)
                geom = paco_basic_color(pp, id2attr)
                image_index = 1_000_000 + img_id

                vqa_rows.append({
                    "image_index":       image_index,
                    "image_name":        img_path,
                    "split":             args.split,
                    "query":             f"What is the color of the {pn} of the {on}?",
                    "geom_label":        geom,
                    "label_rank":        1,
                    "attribute_type":    "color",
                    "depth":             2,
                    "n_objects":         len(ims_objs),
                    "n_program_nodes":   4,
                    "question_index":    qi,
                    "template_filename": "coco_paco_color_part",
                    "label":             "",     # filled post-VLM
                    "raw_response":      "",
                })
                mask_rows.append({
                    "question_index":  qi,
                    "image_index":     image_index,
                    "split":           args.split,
                    "object_name":     on,
                    "part_name":       pn,
                    "geom_label":      geom,
                    "image_path":      img_path,
                    "part_mask_path":  mask_path,
                })
                qi += 1
        if qi > before:
            n_img_q += 1

    cols = ["image_index", "image_name", "split", "query", "geom_label", "label_rank",
            "attribute_type", "depth", "n_objects", "n_program_nodes", "question_index",
            "template_filename", "label", "raw_response"]
    vqa = pd.DataFrame(vqa_rows)[cols]
    masks = pd.DataFrame(mask_rows)

    out_vqa = os.path.join(args.out_dir, f"parts_color_vqa_paco_{args.split}.csv")
    out_masks = os.path.join(args.out_dir, f"parts_color_vqa_paco_{args.split}_masks.csv")
    vqa.to_csv(out_vqa, index=False)
    masks.to_csv(out_masks, index=False)

    print(f"\n{len(vqa):,} questions from {n_img_q:,} images "
          f"(dropped: {n_multi} multi-instance parts, {n_empty} empty masks).")
    print(f"Distinct queries: {vqa['query'].nunique():,}")
    print("\ngeom_label (PACO color) histogram:")
    for c, k in vqa["geom_label"].value_counts().items():
        print(f"  {c:<10} {k:6,}")
    print("\nTop 20 object/part pairs:")
    pairs = (masks["object_name"] + " / " + masks["part_name"]).value_counts().head(20)
    for p, k in pairs.items():
        print(f"  {p:<32} {k:5,}")
    print(f"\nWrote VQA   -> {out_vqa}")
    print(f"Wrote masks -> {out_masks}  (PNGs in {masks_dir})")


if __name__ == "__main__":
    main()
