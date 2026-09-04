"""One-off: compute top-1/2/3 (marginal) for a list of checkpoints, loading the
shared DINO cache ONCE. Sanity check = reproduce each ckpt's stored val_acc.
Throwaway driver for the thesis results tables."""
import sys, torch, pandas as pd
from torch.utils.data import DataLoader
from precompute_features import precompute_text
from superclevr3d_dataset import SuperCLEVR3DCachedFeatDataset
from classifier_model import SlotClassifier, PatchClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT  = "val"
CKPTS = [
    ("full-router(slot)",      "runs/ade20k_hier_router_readout/4986866/slots_9/best_model.pt"),
    ("parent-only",            "runs/ade20k_hier_router_parent_only/4985890/slots_9/best_model.pt"),
    ("patch_qdot-raw",         "runs/ade20k_patch_qdot_raw/5030504/patches/best_model.pt"),
    ("patch_qdot-projected",   "runs/ade20k_patch_qdot_projected/5030505/patches/best_model.pt"),
    ("ext:maskpool-scratch",   "runs/ade20k_hier_router_patch_readout/5039798/slots_9/best_model.pt"),
    ("ext:maskpool-frozen",    "runs/ade20k_hier_router_patch_readout_frozen/5040040/slots_9/best_model.pt"),
    ("ext:qdot-frozen",        "runs/ade20k_hier_router_patch_qdot_frozen/5040143/slots_9/best_model.pt"),
    ("ext:qdot-naiveFT",       "runs/ade20k_hier_router_patch_qdot_finetune/5040507/slots_9/best_model.pt"),
    ("ext:qdot-LPFT",          "runs/ade20k_hier_router_patch_qdot_lpft/5040612/slots_9/best_model.pt"),
]

def build(cfg, nc):
    common = dict(
        dinosaur_cfg_name=cfg["dinosaur_cfg_name"], dinosaur_ckpt_path=cfg["dinosaur_ckpt"],
        num_classes=nc, d_slot=cfg["d_slot"], d_text=cfg["d_text"], num_heads=cfg["num_heads"],
        roberta_model=cfg["roberta_model"], text_encoder_type=cfg["text_encoder"],
        t5_model=cfg["t5_model"], vqa_d_model=cfg["vqa_d_model"], load_text_encoder=False,
        pooler=cfg["pooler"], pooler_layers=cfg["pooler_layers"],
        pooler_dropout=cfg["pooler_dropout"], zero_image_feats=cfg["zero_image_feats"],
    )
    if cfg.get("patch_control", False):
        m = PatchClassifier(d_vit=cfg["d_vit"],
            patch_qdot_project_patches=cfg.get("patch_qdot_project_patches", False),
            patch_qdot_strip_registers=cfg.get("patch_qdot_strip_registers", True),
            patch_qdot_temperature=cfg.get("patch_qdot_temperature", 1.0),
            patch_qdot_normalize=cfg.get("patch_qdot_normalize", False), **common)
    else:
        m = SlotClassifier(
            n_slots=cfg["n_slots"], img_size=cfg["img_size"], slot_backend=cfg["slot_backend"],
            ftdinosaur_model=cfg["ftdinosaur_model"], finetune_ckpt_path=cfg.get("finetune_ckpt_path"),
            recursive_infer=cfg["recursive_infer"], recursive_children=cfg["recursive_children"],
            recursive_parents=cfg["recursive_parents"], recursive_spread=cfg["recursive_spread"],
            recursive_include_parents=cfg["recursive_include_parents"], rank_method=cfg["rank_method"],
            router_temp=cfg["router_temp"], router_entropy_weight=cfg["router_entropy_weight"],
            child_scorer=cfg["child_scorer"],
            router_use_children=not cfg.get("router_parent_only", False),
            router_readout_query=cfg.get("router_readout_query", True),
            router_color_source=cfg.get("router_color_source", "slot"),
            router_qdot_project_patches=cfg.get("router_qdot_project_patches", True),
            router_qdot_dropout=cfg.get("router_qdot_dropout", 0.0), **common)
    return m.eval().to(DEVICE)

def load_state(m, tr):
    for k, v in tr.items():
        if hasattr(m, k) and isinstance(getattr(m, k), torch.nn.Module):
            getattr(m, k).load_state_dict(v)

print("Loading shared DINO cache (once) …", flush=True)
DINO = None  # lazy: set from first cfg
text_cache = {}

print(f"{'model':24s} {'stored':>7s} {'top1':>7s} {'top2':>7s} {'top3':>7s} {'|Δ|':>6s}", flush=True)
for name, path in CKPTS:
    try:
        ck = torch.load(path, map_location="cpu")
        cfg, lv = ck["config"], ck["label_vocab"]
        nc = len(lv)
        if DINO is None:
            DINO = torch.load(cfg["dino_cache"], map_location="cpu")
        m = build(cfg, nc); load_state(m, ck["trainable_state"])
        csv_path = cfg["csv_path"]; key = (csv_path, SPLIT)
        if key not in text_cache:
            df = pd.read_csv(csv_path); df["label"] = df["label"].astype(str)
            uq = sorted(df[df["split"] == SPLIT]["query"].unique().tolist())
            text_cache[key] = precompute_text(uq, device=torch.device("cpu"),
                                              text_encoder=cfg["text_encoder"], with_spans=True)
        ds = SuperCLEVR3DCachedFeatDataset(csv_path=csv_path, label_vocab=lv,
            dino_cache_path=cfg["dino_cache"], text_cache_path=cfg.get("text_cache", ""),
            split=SPLIT, return_spans=True, text_cache=text_cache[key], dino_cache=DINO)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
        is_patch = cfg.get("patch_control", False)
        c = {1: 0, 2: 0, 3: 0}; n = 0
        with torch.no_grad():
            for batch in loader:
                dino, th, am, lab, sp = [t.to(DEVICE) for t in batch]
                if is_patch:
                    logits = m.forward_cached(dino, th, am, spans=sp)
                else:
                    logits = m.forward_recursive_cached(dino, th, am, spans=sp)
                t3 = logits.topk(3, dim=1).indices
                for kk in (1, 2, 3):
                    c[kk] += (t3[:, :kk] == lab[:, None]).any(1).sum().item()
                n += lab.size(0)
        acc = {kk: c[kk] / n for kk in c}
        stored = ck["val_acc"]
        print(f"{name:24s} {stored:7.4f} {acc[1]:7.4f} {acc[2]:7.4f} {acc[3]:7.4f} {abs(acc[1]-stored):6.4f}  (n={n})", flush=True)
    except Exception as e:
        print(f"{name:24s}  FAILED: {type(e).__name__}: {e}", flush=True)
print("DONE", flush=True)
