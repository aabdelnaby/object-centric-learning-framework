"""One-off check that the refactored package reproduces the original thesis code.

For each of the six thesis checkpoints the legacy implementation (``classifier_model.py`` etc. in
the original research checkout) and this package load the same weights, receive the same 64
validation rows from the real feature caches and T5 spans, are seeded identically before the
forward pass, and must produce the same log-probabilities (and, for routers, the same routing
trace; for Patch-QDot, the same patch attention).

    OLD_REPO=../object-centric-learning-framework conda run -n oclf_env python tests/equivalence_legacy_vs_new.py

Needs the legacy checkout, ``tools/link_local_data.sh`` and ``tools/collect_thesis_checkpoints.sh``.
Not a pytest test: it loads the 8 GB PACO cache and takes several minutes on CPU.
"""

from __future__ import annotations

import os
import sys
import time

NEW = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD = os.path.abspath(os.environ.get("OLD_REPO", os.path.join(NEW, "..", "object-centric-learning-framework")))
sys.path.insert(0, NEW)
sys.path.insert(1, OLD)
os.chdir(NEW)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(max(1, os.cpu_count() or 1))

import classifier_model as legacy_cm  # noqa: E402  (legacy checkout)
import precompute_features as legacy_pf  # noqa: E402
from cub_dataset import CUBCachedFeatDataset  # noqa: E402
from superclevr3d_dataset import SuperCLEVR3DCachedFeatDataset  # noqa: E402

from hier_dinosaur.data import PRESETS, build_label_vocab, load_frame, split_frame  # noqa: E402
from hier_dinosaur.models import collect_trainable_state, load_checkpoint, normalize_config  # noqa: E402
from hier_dinosaur.text import encode_spans  # noqa: E402

NAMES = ["paco_hier_router", "paco_hier_router_parent_only", "paco_patch_qdot_raw",
         "paco_patch_qdot_projected", "paco_patch_qca", "cub_hier_router"]
LEGACY_DINO_CKPT = os.path.join(OLD, "checkpoints/dinov3/epoch_67-step_500000_dinov3_11_slots.ckpt")
N_ROWS = int(os.environ.get("N_ROWS", 64))
DEVICE = torch.device("cpu")
ATOL, RTOL = 1e-5, 1e-4


def build_legacy(cfg, num_classes):
    common = dict(
        dinosaur_cfg_name=cfg["dinosaur_cfg_name"], dinosaur_ckpt_path=LEGACY_DINO_CKPT, num_classes=num_classes,
        d_slot=cfg["d_slot"], d_text=cfg["d_text"], num_heads=cfg["num_heads"], roberta_model=cfg["roberta_model"],
        text_encoder_type=cfg["text_encoder"], t5_model=cfg["t5_model"], vqa_d_model=cfg["vqa_d_model"],
        load_text_encoder=False, pooler=cfg["pooler"], pooler_layers=cfg["pooler_layers"],
        pooler_dropout=cfg["pooler_dropout"], zero_image_feats=cfg["zero_image_feats"], repo_root=OLD,
    )
    if cfg.get("patch_control", False):
        m = legacy_cm.PatchClassifier(
            d_vit=cfg["d_vit"], patch_qdot_project_patches=cfg.get("patch_qdot_project_patches", False),
            patch_qdot_strip_registers=cfg.get("patch_qdot_strip_registers", True),
            patch_qdot_temperature=cfg.get("patch_qdot_temperature", 1.0),
            patch_qdot_normalize=cfg.get("patch_qdot_normalize", False), **common)
    else:
        m = legacy_cm.SlotClassifier(
            n_slots=cfg["n_slots"], img_size=cfg["img_size"], slot_backend=cfg["slot_backend"],
            ftdinosaur_model=cfg["ftdinosaur_model"], finetune_ckpt_path=cfg.get("finetune_ckpt_path"),
            recursive_infer=cfg["recursive_infer"], recursive_children=cfg["recursive_children"],
            recursive_parents=cfg["recursive_parents"], recursive_spread=cfg["recursive_spread"],
            recursive_include_parents=cfg["recursive_include_parents"], rank_method=cfg["rank_method"],
            router_temp=cfg["router_temp"], router_entropy_weight=cfg["router_entropy_weight"],
            child_scorer=cfg["child_scorer"], router_use_children=not cfg.get("router_parent_only", False),
            router_readout_query=cfg.get("router_readout_query", True),
            router_color_source=cfg.get("router_color_source", "slot"),
            router_qdot_project_patches=cfg.get("router_qdot_project_patches", True),
            router_qdot_dropout=cfg.get("router_qdot_dropout", 0.0), **common)
    return m.eval()


def load_legacy(model, trainable):
    for k, v in trainable.items():
        if hasattr(model, k) and isinstance(getattr(model, k), torch.nn.Module):
            getattr(model, k).load_state_dict(v)


@torch.no_grad()
def legacy_trace(model, dino, spans):
    """Legacy routing trace on cached features (mirrors eval_hier_router.viz_cached, incl. parent-only)."""
    from ocl.typing import FeatureExtractorOutput
    B = dino.shape[0]
    feat_out = FeatureExtractorOutput(features=dino.float(), positions=model._dino_positions)
    slots, slot_masks, embedded = model._slots_feats_from_featout(feat_out, B)
    rank = slot_masks.sum(dim=-1)
    if not model.hier_router.use_children:
        parent_slots, nonempty, top_idx = model._select_parents(slots, slot_masks, rank)
        h_q = model.text_projector(spans[:, 2].to(dtype=parent_slots.dtype))
        h_ro = model._readout_query(spans, parent_slots.dtype)
        logP_a, _ = model.hier_router(parent_slots, None, None, h_q, nonempty, h_ro)
        return {"logits": logP_a, "P_parent": model.hier_router.last_logP_parent.exp(), "top_idx": top_idx,
                "nonempty": nonempty}
    child_slots, info = model._refine_top_slots(embedded, slots, slot_masks, rank)
    P, Ds = info["parent_slots"].shape[1], info["parent_slots"].shape[-1]
    child_slots = child_slots.view(B, P, model.recursive_children, Ds)
    h_x = model.text_projector(spans[:, 3].to(dtype=child_slots.dtype))
    h_y = model.text_projector(spans[:, 1].to(dtype=child_slots.dtype))
    h_ro = h_x if model.hier_router.readout_query else None
    logP_a, _ = model.hier_router(info["parent_slots"], child_slots, h_x, h_y, info["nonempty"], h_ro)
    logw = model.hier_router.last_logP_parent[:, :, None] + model.hier_router.last_logP_child
    return {"logits": logP_a, "P_parent": model.hier_router.last_logP_parent.exp(),
            "P_child": model.hier_router.last_logP_child.exp(), "w": logw.exp(),
            "child_attn": info["child_attn"], "parent_masks": info["parent_masks"],
            "top_idx": info["top_idx"], "nonempty": info["nonempty"]}


def check(name, a, b, exact=False):
    if exact:
        ok = torch.equal(a, b)
        diff = float((a.float() - b.float()).abs().max()) if a.shape == b.shape else float("nan")
    else:
        ok = a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL)
        diff = float((a.float() - b.float()).abs().max()) if a.shape == b.shape else float("nan")
    print(f"    {'OK ' if ok else 'FAIL'} {name:<22s} max|diff|={diff:.2e} shape={tuple(a.shape)}")
    return ok


def main():
    failures = 0
    dino_caches, text_by_dataset = {}, {}
    for name in NAMES:
        t0 = time.time()
        path = f"checkpoints/thesis/{name}/best_model.pt"
        print(f"\n=== {name}  ({path})")
        ck = torch.load(path, map_location="cpu")
        cfg, vocab = ck["config"], ck["label_vocab"]
        ncfg = normalize_config(cfg)
        dataset, spec = ncfg["dataset"], PRESETS[normalize_config(cfg)["dataset"]]
        split = spec.val_split

        # ── data: legacy dataset vs new frame ────────────────────────────
        if spec.dino_cache not in dino_caches:
            print(f"  loading cache {spec.dino_cache} …", flush=True)
            dino_caches[spec.dino_cache] = torch.load(spec.dino_cache, map_location="cpu")
        cache = dino_caches[spec.dino_cache]
        if dataset not in text_by_dataset:
            frame = pd.read_csv(spec.csv)
            frame["label"] = frame["label"].astype(str)
            if spec.category_filter:
                frame = frame[frame["query"].str.contains(spec.category_filter, case=False, na=False)]
            uq = sorted(frame[frame["split"] == split]["query"].unique().tolist())
            print(f"  encoding {len(uq)} {split} queries with legacy precompute_text and new encode_spans …", flush=True)
            legacy_txt = legacy_pf.precompute_text(uq, DEVICE, text_encoder="t5", with_spans=True,
                                                   span_dataset=cfg["dataset"])
            new_spans = encode_spans(uq, dataset, DEVICE)
            text_by_dataset[dataset] = (legacy_txt, new_spans)
        legacy_txt, new_spans = text_by_dataset[dataset]

        if dataset == "cub":
            legacy_ds = CUBCachedFeatDataset(csv_path=spec.csv, dino_cache_path="", text_cache_path="", split=split,
                                             label_vocab=vocab, category_filter=cfg.get("category_filter"),
                                             text_cache=legacy_txt, dino_cache=cache, return_spans=True)
        else:
            legacy_ds = SuperCLEVR3DCachedFeatDataset(csv_path=spec.csv, dino_cache_path="", text_cache_path="",
                                                      split=split, label_vocab=vocab, text_cache=legacy_txt,
                                                      dino_cache=cache, return_spans=True)
        df = load_frame(spec.csv, ncfg.get("category_filter"))
        new_vocab = build_label_vocab(df, spec.train_split)
        rows = split_frame(df, split, vocab)
        same_vocab = new_vocab == vocab
        same_rows = (len(rows) == len(legacy_ds.df)
                     and (rows[["image_name", "query", "label"]].values == legacy_ds.df[["image_name", "query", "label"]].values).all())
        print(f"    {'OK ' if same_vocab else 'FAIL'} label vocab identical ({len(vocab)} classes)")
        print(f"    {'OK ' if same_rows else 'FAIL'} {split} rows identical (n={len(rows)})")
        failures += (not same_vocab) + (not same_rows)

        idx = list(range(min(N_ROWS, len(rows))))
        old_batch = [legacy_ds[i] for i in idx]
        dino = torch.stack([b[0] for b in old_batch])
        th = torch.stack([b[1] for b in old_batch])
        am = torch.stack([b[2] for b in old_batch])
        labels = torch.stack([b[3] for b in old_batch])
        sp_old = torch.stack([b[4] for b in old_batch])
        sp_new = torch.stack([new_spans[q] for q in rows["query"].iloc[idx]])
        feats_new = torch.stack([cache["features"][n] for n in rows["image_name"].iloc[idx]])
        failures += not check("spans", sp_old, sp_new)
        failures += not check("features", dino, feats_new, exact=True)

        # ── models ────────────────────────────────────────────────────────
        old = build_legacy(cfg, len(vocab))
        load_legacy(old, ck["trainable_state"])
        new, info = load_checkpoint(path, device=DEVICE)
        new_state = collect_trainable_state(new)
        same_w = all(torch.equal(v, new_state[k][kk]) for k, sd in ck["trainable_state"].items()
                     if k in new_state for kk, v in sd.items())
        n_old = sum(p.numel() for p in old.trainable_parameters())
        n_new = sum(p.numel() for p in new.trainable_parameters())
        print(f"    {'OK ' if same_w and n_old == n_new else 'FAIL'} trainable weights identical "
              f"(legacy {n_old:,} / new {n_new:,} params)")
        failures += not (same_w and n_old == n_new)

        is_patch = cfg.get("patch_control", False)
        with torch.no_grad():
            torch.manual_seed(0)
            if is_patch:
                logits_old = old.forward_cached(dino, th, am, spans=sp_old)
                attn_old = (old.patch_qdot_head if cfg["pooler"] == "patch_qdot" else old.qca_head).last_attn
            else:
                logits_old = old.forward_recursive_cached(dino, th, am, spans=sp_old)
            torch.manual_seed(0)
            if is_patch:
                logits_new, attn_new = new.attend(feats_new, sp_new)
            else:
                logits_new = new(feats_new, sp_new)
        failures += not check("log P(a)", logits_old, logits_new, exact=is_patch)
        acc_old = float((logits_old.argmax(1) == labels).float().mean())
        acc_new = float((logits_new.argmax(1) == labels).float().mean())
        print(f"    accuracy on these rows: legacy {acc_old:.4f} / new {acc_new:.4f}")
        if is_patch:
            failures += not check("patch attention", attn_old, attn_new, exact=True)
        else:
            with torch.no_grad():
                torch.manual_seed(1)
                tr_old = legacy_trace(old, dino, sp_old)
                torch.manual_seed(1)
                tr_new = new.trace_cached(feats_new, sp_new)
            for key in ("logits", "P_parent", "P_child", "w", "child_attn", "parent_masks"):
                if key in tr_old:
                    failures += not check(f"trace {key}", tr_old[key], tr_new[key])
            for key in ("top_idx", "nonempty"):
                failures += not check(f"trace {key}", tr_old[key], tr_new[key], exact=True)
        print(f"  done in {time.time() - t0:.0f}s")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
