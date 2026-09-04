#!/usr/bin/env python3
"""Grounding-faithfulness eval: does the model attend to the *named part*?

Turns "the slot routing gives more meaningful representations" into a number. For
each PACO-val "colour of the <part> of the <object>?" question we have a GT part
mask. We extract each model's spatial attention over the named part and score how
well it lands on the true part region:

  * HierRouter  → path-weighted child attention  Σ_jk w_jk · child_attn_jk  (B,N)
                  (also the argmax-path child mask). This is the "part" the router
                  routed to.
  * Patch-QDot  → the query→patch dot-product attention  (B,196).

Both maps are reshaped to the 14×14 ViT grid, upsampled to the model's square-224
input frame, and compared to the GT part mask (resized the same way) with:

  * mass_in_mask  fraction of attention mass inside the GT part (threshold-free).
  * pointing      argmax attention pixel falls inside the GT part (0/1).
  * iou@mean      IoU of {attention > its mean} with the GT part.
  * chance        the GT part's area fraction = the mass_in_mask / pointing a
                  *uniform* map would score. lift = mass_in_mask − chance.

If the router's region hits the part substantially above chance AND above Patch-QDot,
that substantiates "more meaningful representation". If Patch-QDot localises the part
just as well, the interpretability edge is also a tie (only the symbolic *form*
differs). Colour accuracy on the scored subset is printed as a sanity check.

Example:
    conda run --no-capture-output -n oclf_env python eval_grounding.py \
        --router_ckpt runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt \
        --patch_ckpt  runs/ade20k_patch_qdot_projected/5030505/patches/best_model.pt \
        --patch_ckpt  runs/ade20k_patch_qdot_raw/5030504/patches/best_model.pt \
        --n 400 --device cpu
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from precompute_features import precompute_text, build_image_transform
from visualize_hier_routing import build_model as build_router, load_trainable as load_router
from visualize_patch_qdot import build_model as build_patch, load_trainable as load_patch

MASKS_CSV = "FG-datset/coco_paco/parts_color_vqa_paco_val_masks.csv"


def recon_query(part_name, object_name):
    return f"What is the color of the {part_name} of the {object_name}?"


def att_to_map(att: torch.Tensor, size: int) -> torch.Tensor:
    """(B, N) patch attention → (B, size, size) per-sample prob map (sums to 1).

    Strips the 4 DINOv3 register tokens if N isn't a perfect square but N-4 is."""
    B, N = att.shape
    if math.isqrt(N) ** 2 != N and math.isqrt(N - 4) ** 2 == (N - 4):
        att = att[:, 4:]
        N = att.shape[1]
    side = math.isqrt(N)
    g = att[:, : side * side].reshape(B, 1, side, side).clamp(min=0).float()
    up = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    return up / up.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-9)


def score(maps: torch.Tensor, masks: torch.Tensor):
    """maps (B,H,W) prob, masks (B,H,W) {0,1} → dict of per-sample (B,) metrics."""
    B, H, W = maps.shape
    mf, mm = maps.reshape(B, -1), masks.reshape(B, -1)
    mass = (mf * mm).sum(1)                                    # attention mass in part
    pt   = mm[torch.arange(B), mf.argmax(1)]                   # pointing game (0/1)
    thr  = mf.mean(1, keepdim=True)
    pred = (mf > thr).float()
    inter = (pred * mm).sum(1)
    union = ((pred + mm) > 0).float().sum(1).clamp(min=1)
    iou  = inter / union
    area = mm.mean(1)                                          # chance level
    return {"mass": mass, "point": pt, "iou": iou, "area": area}


@torch.no_grad()
def router_maps(model, img, txt, msk, spans):
    out = model.forward_hier_router_viz(img, txt, msk, spans)
    w, ca = out["w"], out["child_attn"]                       # (B,P,K), (B,P,K,N)
    marg = torch.einsum("bpk,bpkn->bn", w, ca)                # path-weighted part map
    B, P, K, N = ca.shape
    flat = w.reshape(B, -1).argmax(1)
    arg  = ca[torch.arange(B), flat // K, flat % K]           # argmax-path child mask
    return marg, arg, out["logits"]


@torch.no_grad()
def patch_maps(model, img, h_src):
    feat = model.dino_feature_extractor(
        inputs={"input": {"image": img, "batch_size": img.shape[0]}}
    ).features.float()
    h_yx = model.text_projector(h_src.to(feat.dtype))
    logp, attn = model.patch_qdot_head(feat, h_yx)
    return attn, logp


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["config"], ck["label_vocab"], ck["trainable_state"], ck


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router_ckpt", default=None, help="hier_router best_model.pt")
    ap.add_argument("--patch_ckpt", action="append", default=[], help="patch_qdot best_model.pt (repeatable)")
    ap.add_argument("--masks_csv", default=MASKS_CSV)
    ap.add_argument("--n", type=int, default=400, help="# samples (-1 = all val with masks)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_csv", default=None, help="optional per-sample CSV dump")
    args = ap.parse_args()
    device = torch.device(args.device)

    df = pd.read_csv(args.masks_csv)
    df["query"] = [recon_query(p, o) for p, o in zip(df["part_name"], df["object_name"])]
    df["label"] = df["geom_label"].astype(str)
    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed)
    df = df.reset_index(drop=True)
    print(f"Scoring {len(df)} PACO-val questions with GT part masks (device={device}).")

    # one text-encode for all unique queries (with spans → ch1/ch3 for router + readout)
    text_cfg = "t5"  # all these runs use t5; spans needed by both router and patch_qdot
    txt = precompute_text(sorted(df["query"].unique()), device, text_encoder=text_cfg, with_spans=True)

    # square-224 frame shared by image + mask (all these runs: img_size 224, resize square)
    SIZE = 384  # mask/eval resolution (upsample target); independent of model patch grid
    img_tf = build_image_transform(224, "square")

    # ── preload all images, masks, text tensors (CPU) ────────────────────────
    imgs, masks, th, am, sp, valid = [], [], [], [], [], []
    for _, r in df.iterrows():
        try:
            pil = PILImage.open(r["image_path"]).convert("RGB")
            m   = PILImage.open(r["part_mask_path"]).resize((SIZE, SIZE), PILImage.NEAREST)
        except Exception:
            valid.append(False); continue
        mk = (np.array(m) > 0)
        if mk.ndim == 3:
            mk = mk.any(-1)
        if mk.sum() == 0:                     # empty GT part → can't score grounding
            valid.append(False); continue
        imgs.append(img_tf(pil))
        masks.append(torch.from_numpy(mk.astype(np.float32)))
        q = r["query"]
        th.append(txt["hidden"][q]); am.append(txt["masks"][q])
        sp.append(torch.stack([txt["x_vec"][q], txt["y_vec"][q], txt["xy_vec"][q], txt["readout_vec"][q]]))
        valid.append(True)
    df = df[pd.Series(valid).values].reset_index(drop=True)
    imgs = torch.stack(imgs); masks_t = torch.stack(masks)
    th = torch.stack(th); am = torch.stack(am); sp = torch.stack(sp)
    readout = sp[:, 3]
    print(f"  {len(df)} scorable (non-empty mask). mean part area frac={masks_t.mean():.4f}")

    results = {}     # name -> dict of metric -> (N,) tensor
    colacc  = {}     # name -> colour accuracy

    def run_model(name, build_fn, load_fn, cfg, lbl_vocab, ts, kind):
        model = build_fn(cfg, len(lbl_vocab), device); load_fn(model, ts)
        accmaps = {"mass": [], "point": [], "iou": [], "area": []}
        if name.startswith("HierRouter"):
            accmaps_arg = {"mass": [], "point": [], "iou": [], "area": []}
        correct, total = 0, 0
        for s in range(0, len(df), args.batch):
            e = min(s + args.batch, len(df))
            img = imgs[s:e].to(device); mk = masks_t[s:e].to(device)
            if kind == "router":
                marg, arg, logits = router_maps(model, img, th[s:e].to(device),
                                                am[s:e].to(device), sp[s:e].to(device))
                for tgt, store in ((marg, accmaps), (arg, accmaps_arg)):
                    mp = att_to_map(tgt, SIZE)
                    sc = score(mp, mk)
                    for k in accmaps: store[k].append(sc[k])
            else:
                attn, logits = patch_maps(model, img, readout[s:e].to(device))
                mp = att_to_map(attn, SIZE)
                sc = score(mp, mk)
                for k in accmaps: accmaps[k].append(sc[k])
            pred = logits.argmax(1).cpu()
            for i, gi in enumerate(range(s, e)):
                total += 1
                correct += int(lbl_vocab.get(df["label"].iloc[gi], -1) == int(pred[i]))
        results[name] = {k: torch.cat(v) for k, v in accmaps.items()}
        if name.startswith("HierRouter"):
            results[name + " (argmax-path)"] = {k: torch.cat(v) for k, v in accmaps_arg.items()}
        colacc[name] = correct / max(total, 1)

    if args.router_ckpt:
        cfg, lv, ts, _ = load_ckpt(args.router_ckpt)
        assert cfg["pooler"] == "hier_router", f"router_ckpt pooler={cfg['pooler']}"
        print(f"\n[HierRouter] {args.router_ckpt}")
        run_model("HierRouter (path-weighted)", build_router, load_router, cfg, lv, ts, "router")

    for pc in args.patch_ckpt:
        cfg, lv, ts, _ = load_ckpt(pc)
        assert cfg["pooler"] == "patch_qdot", f"patch_ckpt pooler={cfg['pooler']}"
        variant = "projected" if cfg.get("patch_qdot_project_patches") else "raw"
        name = f"Patch-QDot-{variant}"
        print(f"\n[{name}] {pc}")
        run_model(name, build_patch, load_patch, cfg, lv, ts, "patch")

    # ── report ───────────────────────────────────────────────────────────────
    chance = float(masks_t.mean())
    print("\n" + "=" * 86)
    print(f"GROUNDING FAITHFULNESS  (n={len(df)} PACO-val parts, chance mass≈part-area={chance:.4f})")
    print("=" * 86)
    print(f"{'model':<34}{'mass_in_mask':>13}{'lift':>8}{'pointing':>10}{'iou@mean':>10}{'col_acc':>9}")
    print("-" * 86)
    for name, m in results.items():
        mass = float(m["mass"].mean()); area = float(m["area"].mean())
        ca = colacc.get(name.replace(" (argmax-path)", ""), float("nan"))
        print(f"{name:<34}{mass:>13.4f}{mass-area:>8.4f}{float(m['point'].mean()):>10.4f}"
              f"{float(m['iou'].mean()):>10.4f}{ca:>9.4f}")
    print("-" * 86)
    print(f"{'(uniform / chance)':<34}{chance:>13.4f}{0.0:>8.4f}{chance:>10.4f}{'-':>10}{'-':>9}")
    print("=" * 86)

    if args.out_csv:
        rows = []
        for name, m in results.items():
            for i in range(len(df)):
                rows.append(dict(model=name, idx=i, query=df["query"].iloc[i],
                                 mass=float(m["mass"][i]), point=float(m["point"][i]),
                                 iou=float(m["iou"][i]), area=float(m["area"][i])))
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"per-sample metrics → {args.out_csv}")


if __name__ == "__main__":
    main()
