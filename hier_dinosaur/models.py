"""The five thesis models, run configs and checkpoint I/O.

    hier_router               SlotRouterModel  (object→part routing, slot readout)      Table 1 row 1
    hier_router_parent_only   SlotRouterModel  (no part level)                           Table 1 row 5
    patch_qdot_raw            PatchBaselineModel(Patch-QDot, raw 384-d patches)          Table 1 row 3
    patch_qdot_projected      PatchBaselineModel(Patch-QDot, projected to 256-d)         Table 1 row 2
    patch_qca                 PatchBaselineModel(Patch-QCA, multi-head cross-attention)  Table 1 row 4

A run config is a flat dict (``format = "hier_dinosaur-v1"``); ``normalize_config`` also accepts
the config dicts stored in the original thesis checkpoints and translates them.
Trained weights are stored as ``{module_name: state_dict}`` with the same module names as the
thesis checkpoints, so those load unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn

from .baselines import PatchQueryDotProductColorHead, QueryCrossAttentionColorHead
from .dinosaur import D_SLOT, D_VIT, FrozenDinosaur
from .hierarchy import SlotTree, build_tree
from .router import HierRouter, RouterOutput, TextProjector

MODELS = ("hier_router", "hier_router_parent_only", "patch_qdot_raw", "patch_qdot_projected", "patch_qca")
CONFIG_FORMAT = "hier_dinosaur-v1"
DEFAULT_DINOSAUR_CKPT = "checkpoints/dinosaur_dinov3_vits16_coco.ckpt"
TRAINABLE_MODULES = ("text_projector", "patch_projector", "hier_router", "qca_head", "patch_qdot_head")


def is_router(model_name: str) -> bool:
    return model_name.startswith("hier_router")


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class SlotRouterModel(nn.Module):
    """Frozen Hierarchical-DINOSAUR tree + trainable TextProjector + HierRouter."""

    def __init__(
        self,
        dinosaur: FrozenDinosaur,
        num_classes: int,
        d_text: int = 768,
        n_children: int = 5,
        n_parents: Optional[int] = None,
        spread: float = 0.0,
        router_temp: float = 0.95,
        child_scorer: str = "mlp",
        use_children: bool = True,
        readout_query: bool = True,
    ):
        super().__init__()
        self.dinosaur = dinosaur
        self.n_children = int(n_children)
        self.n_parents = n_parents
        self.spread = float(spread)
        self.text_projector = TextProjector(d_text=d_text, d_slot=D_SLOT)
        self.hier_router = HierRouter(
            d_slot=D_SLOT, num_classes=num_classes, temperature=router_temp,
            child_scorer=child_scorer, use_children=use_children, readout_query=readout_query,
        )

    @property
    def use_children(self) -> bool:
        return self.hier_router.use_children

    def trainable_parameters(self):
        return list(self.text_projector.parameters()) + list(self.hier_router.parameters())

    def tree(self, feat_out) -> SlotTree:
        return build_tree(self.dinosaur, feat_out, self.n_parents, self.n_children, self.spread,
                          with_children=self.use_children)

    def route(self, tree: SlotTree, spans: torch.Tensor) -> RouterOutput:
        """Language routing over a built tree. ``spans`` is (B, 4, d_text)."""
        dtype = tree.parent_slots.dtype
        proj = lambda ch: self.text_projector(spans[:, ch].to(dtype=dtype))
        if not self.use_children:
            h_q = proj(2)                                       # "<part> of the <object>"
            h_readout = proj(3) if self.hier_router.readout_query else None
            return self.hier_router(tree.parent_slots, None, None, h_q, tree.nonempty, h_readout)
        h_x = proj(3)                                           # "<object> <part>"
        h_y = proj(1)                                           # "<object>"
        h_readout = h_x if self.hier_router.readout_query else None
        return self.hier_router(tree.parent_slots, tree.child_slots, h_x, h_y, tree.nonempty, h_readout)

    def forward(self, dino_features: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """Cached patch features (B, N, 384) + spans → log P(a) (B, C)."""
        return self.route(self.tree(self.dinosaur.from_cache(dino_features)), spans).logp_answer

    def forward_images(self, images: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        return self.route(self.tree(self.dinosaur.encode_images(images)), spans).logp_answer

    @torch.no_grad()
    def trace(self, feat_out, spans: torch.Tensor) -> Dict:
        """Full routing trace for evaluation and visualisation.

        Keys: ``logits`` (B,C) log P(a); ``P_parent`` (B,P); ``P_child`` (B,P,K); ``w`` (B,P,K)
        path weights; ``p_color`` (B,P,K,C); ``parent_masks`` (B,P,N); ``child_attn`` (B,P,K,N);
        ``top_idx`` (B,P); ``nonempty`` (B,P); plus ``router`` (RouterOutput) and ``tree`` (SlotTree).
        The parent-only model is reported as a degenerate K = 1 tree.
        """
        tree = self.tree(feat_out)
        out = self.route(tree, spans)
        if self.use_children:
            return {
                "logits": out.logp_answer, "P_parent": out.logp_parent.exp(),
                "P_child": out.logp_child.exp(), "w": out.logw.exp(), "p_color": out.logp_color.exp(),
                "parent_masks": tree.parent_masks, "child_attn": tree.child_attn,
                "top_idx": tree.top_idx, "nonempty": tree.nonempty, "router": out, "tree": tree,
            }
        p_parent = out.logp_parent.exp()
        return {
            "logits": out.logp_answer, "P_parent": p_parent,
            "P_child": torch.ones_like(p_parent)[:, :, None], "w": p_parent[:, :, None],
            "p_color": out.logp_color.exp()[:, :, None, :],
            "parent_masks": tree.parent_masks, "child_attn": tree.parent_masks[:, :, None, :],
            "top_idx": tree.top_idx, "nonempty": tree.nonempty, "router": out, "tree": tree,
        }

    def trace_images(self, images: torch.Tensor, spans: torch.Tensor) -> Dict:
        return self.trace(self.dinosaur.encode_images(images), spans)

    def trace_cached(self, dino_features: torch.Tensor, spans: torch.Tensor) -> Dict:
        return self.trace(self.dinosaur.from_cache(dino_features), spans)


class PatchBaselineModel(nn.Module):
    """Frozen DINOv3 patches + trainable TextProjector + a flat patch head (Patch-QDot or Patch-QCA)."""

    def __init__(
        self,
        dinosaur: FrozenDinosaur,
        num_classes: int,
        head: str = "patch_qdot",
        d_text: int = 768,
        num_heads: int = 8,
        project_patches: bool = False,
        legacy_strip_tokens: int = 0,
    ):
        super().__init__()
        if head not in ("patch_qdot", "qca"):
            raise ValueError(f"unknown head {head!r}")
        self.dinosaur = dinosaur
        self.head_type = head
        if head == "qca":
            self.patch_projector = nn.Linear(D_VIT, D_SLOT)
            self.text_projector = TextProjector(d_text=d_text, d_slot=D_SLOT)
            self.qca_head = QueryCrossAttentionColorHead(D_SLOT, num_classes, num_heads=num_heads)
        else:
            self.patch_projector = nn.Identity()
            self.text_projector = TextProjector(d_text=d_text, d_slot=D_SLOT)
            self.patch_qdot_head = PatchQueryDotProductColorHead(
                d_query=D_SLOT, d_vit=D_VIT, d_slot=D_SLOT, num_classes=num_classes,
                project_patches=project_patches, legacy_strip_tokens=legacy_strip_tokens,
            )

    @property
    def head(self) -> nn.Module:
        return self.qca_head if self.head_type == "qca" else self.patch_qdot_head

    def trainable_parameters(self):
        return (list(self.patch_projector.parameters()) + list(self.text_projector.parameters())
                + list(self.head.parameters()))

    def attend(self, dino_features: torch.Tensor, spans: torch.Tensor):
        """→ ``(log P(a) (B, C), attention over patches (B, N'))``."""
        patches = dino_features.float() if dino_features.dtype != torch.float32 else dino_features
        h_yx = self.text_projector(spans[:, 3].to(dtype=patches.dtype))     # "<object> <part>"
        if self.head_type == "qca":
            return self.qca_head(self.patch_projector(patches), h_yx)
        return self.patch_qdot_head(patches, h_yx)

    def forward(self, dino_features: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        return self.attend(dino_features, spans)[0]

    def forward_images(self, images: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        return self.attend(self.dinosaur.encode_images(images).features, spans)[0]

    def attend_images(self, images: torch.Tensor, spans: torch.Tensor):
        return self.attend(self.dinosaur.encode_images(images).features, spans)


# ──────────────────────────────────────────────────────────────────────────────
# Configs
# ──────────────────────────────────────────────────────────────────────────────

def normalize_config(cfg: Dict) -> Dict:
    """Return a ``hier_dinosaur-v1`` config; translate the config dicts of the thesis checkpoints."""
    if cfg.get("format") == CONFIG_FORMAT:
        return dict(cfg)

    pooler, patch = cfg.get("pooler"), bool(cfg.get("patch_control", False))
    if pooler == "hier_router" and not patch:
        model = "hier_router_parent_only" if cfg.get("router_parent_only", False) else "hier_router"
    elif pooler == "patch_qdot" and patch:
        model = "patch_qdot_projected" if cfg.get("patch_qdot_project_patches", False) else "patch_qdot_raw"
    elif pooler == "qca" and patch:
        model = "patch_qca"
    else:
        raise ValueError(f"legacy config (pooler={pooler!r}, patch_control={patch}) is not a thesis model")

    dataset = {"ade20k": "paco", "cub": "cub"}.get(cfg.get("dataset"))
    if dataset is None:
        raise ValueError(f"legacy dataset {cfg.get('dataset')!r} is not a thesis dataset")

    fixed = {  # options that never varied in the thesis; anything else needs code that no longer exists
        "router_color_source": "slot", "router_entropy_weight": 0.0, "patch_qdot_temperature": 1.0,
        "patch_qdot_normalize": False, "optimizer": "adamw", "lr_schedule": "cosine",
        "zero_image_feats": False, "text_encoder": "t5", "slot_backend": "oclf", "router_init_ckpt": None,
    }
    for k, v in fixed.items():
        if cfg.get(k, v) != v:
            raise ValueError(f"legacy config has {k}={cfg.get(k)!r}; only {v!r} is supported")

    n_slots = int(cfg.get("n_slots", 7))
    strip = 4 if (model.startswith("patch_qdot") and cfg.get("patch_qdot_strip_registers", True)) else 0
    return {
        "format": CONFIG_FORMAT, "legacy": True,
        "model": model, "dataset": dataset,
        "csv": cfg.get("csv_path"), "image_root": cfg.get("image_root"), "dino_cache": cfg.get("dino_cache"),
        "dinosaur_ckpt": cfg.get("dinosaur_ckpt"), "category_filter": cfg.get("category_filter"),
        "n_slots": n_slots, "children": int(cfg.get("recursive_children", 4)),
        "parents": int(cfg.get("recursive_parents") or n_slots), "spread": float(cfg.get("recursive_spread", 0.5)),
        "router_temp": float(cfg.get("router_temp", 1.0)), "child_scorer": cfg.get("child_scorer", "bilinear"),
        "readout_query": bool(cfg.get("router_readout_query", True)), "num_heads": int(cfg.get("num_heads", 8)),
        "legacy_strip_tokens": strip, "d_text": int(cfg.get("d_text", 768)),
        "img_size": int(cfg.get("img_size", 224)), "resize_mode": cfg.get("resize_mode", "crop"),
        "lr": cfg.get("lr"), "weight_decay": cfg.get("weight_decay"), "batch_size": cfg.get("batch_size"),
        "warmup_steps": cfg.get("warmup_steps"), "max_steps": cfg.get("max_steps"), "seed": cfg.get("seed"),
    }


def resolve_dinosaur_ckpt(path: Optional[str]) -> str:
    """Use ``path`` if it exists, otherwise fall back to the branch's default checkpoint location."""
    if path and os.path.isfile(path):
        return path
    if os.path.isfile(DEFAULT_DINOSAUR_CKPT):
        if path:
            print(f"  [dinosaur] {path} not found; using {DEFAULT_DINOSAUR_CKPT}")
        return DEFAULT_DINOSAUR_CKPT
    raise FileNotFoundError(f"DINOSAUR checkpoint not found: {path or DEFAULT_DINOSAUR_CKPT} "
                            f"(run tools/link_local_data.sh or experiments/01_train_dinosaur.sh)")


def build_model(cfg: Dict, num_classes: int, device=None, dinosaur_ckpt: Optional[str] = None) -> nn.Module:
    """Instantiate the model described by a (legacy or new) run config; frozen backbone included."""
    cfg = normalize_config(cfg)
    ckpt = resolve_dinosaur_ckpt(dinosaur_ckpt or cfg.get("dinosaur_ckpt"))
    dinosaur = FrozenDinosaur(n_slots=cfg["n_slots"], checkpoint=ckpt)
    name = cfg["model"]
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")
    if is_router(name):
        model = SlotRouterModel(
            dinosaur, num_classes, d_text=cfg["d_text"], n_children=cfg["children"],
            n_parents=cfg["parents"], spread=cfg["spread"], router_temp=cfg["router_temp"],
            child_scorer=cfg["child_scorer"], use_children=(name == "hier_router"),
            readout_query=cfg["readout_query"],
        )
    else:
        model = PatchBaselineModel(
            dinosaur, num_classes, head=("qca" if name == "patch_qca" else "patch_qdot"),
            d_text=cfg["d_text"], num_heads=cfg["num_heads"],
            project_patches=(name == "patch_qdot_projected"),
            legacy_strip_tokens=cfg.get("legacy_strip_tokens", 0),
        )
    if device is not None:
        model = model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ──────────────────────────────────────────────────────────────────────────────

def collect_trainable_state(model: nn.Module) -> Dict[str, Dict]:
    return {name: getattr(model, name).state_dict() for name in TRAINABLE_MODULES if hasattr(model, name)}


def load_trainable_state(model: nn.Module, state: Dict[str, Dict]) -> List[str]:
    """Load every stored module the model has; returns the names loaded (strict per module)."""
    loaded = []
    for name, sd in state.items():
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            module.load_state_dict(sd)
            loaded.append(name)
    return loaded


def classes_from_vocab(label_vocab: Dict[str, int]) -> List[str]:
    classes = [None] * len(label_vocab)
    for name, idx in label_vocab.items():
        classes[idx] = name
    return classes


def save_checkpoint(path, model: nn.Module, cfg: Dict, label_vocab: Dict[str, int], epoch: int,
                    val_acc: float, extra: Optional[Dict] = None) -> None:
    payload = {"epoch": int(epoch), "val_acc": float(val_acc), "label_vocab": label_vocab,
               "config": cfg, "trainable_state": collect_trainable_state(model)}
    if extra:
        payload.update(extra)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu", dinosaur_ckpt: Optional[str] = None):
    """→ ``(model in eval mode, info)`` where ``info`` has config, label_vocab, classes, epoch, val_acc."""
    ck = torch.load(path, map_location="cpu")
    cfg = normalize_config(ck["config"])
    label_vocab = ck["label_vocab"]
    model = build_model(cfg, len(label_vocab), device=device, dinosaur_ckpt=dinosaur_ckpt)
    loaded = load_trainable_state(model, ck["trainable_state"])
    if not any(n in loaded for n in ("hier_router", "qca_head", "patch_qdot_head")):
        raise RuntimeError(f"{path} contains no trained head weights ({loaded})")
    model.eval()
    info = {"config": cfg, "label_vocab": label_vocab, "classes": classes_from_vocab(label_vocab),
            "epoch": ck.get("epoch"), "val_acc": ck.get("val_acc"), "path": str(path), "loaded": loaded}
    return model, info


def write_json(path, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
