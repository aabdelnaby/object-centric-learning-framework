"""Fine-tune DINOSAUR slot attention on CUB-200-2011 images.

Self-supervised: reconstructs frozen DINO ViT-S/16 patch features from learned
slots, using the same MSE-on-normalised-features loss as original DINOSAUR
training.  No labels needed — only raw images.

Frozen:   feature_extractor  (ViT-S/16 DINO backbone)
Trained:  conditioning, perceptual_grouping, object_decoder

Why fine-tune conditioning too?
    RandomConditioning has learnable mu/logsigma that initialise slots.  On
    out-of-domain images (COCO checkpoint → CUB birds) the random draws are
    misaligned with the feature distribution, contributing to slot collapse.
    Letting it adapt costs almost nothing (~512 parameters).

Why fine-tune object_decoder too?
    The autoregressive transformer decoder is trained jointly with slot
    attention.  Freezing it while fine-tuning the slot attention would give
    gradients that push slots toward COCO-shaped reconstructions, counteracting
    the fine-tuning objective.  Training both together is the correct setup.

Usage
-----
    # Default: 30 epochs, lr=2e-4, batch=32, n_slots=7
    python finetune_slots.py

    # Custom
    python finetune_slots.py --n_epochs 50 --lr 1e-4 --batch_size 16

Output
------
    cub_slot_finetune/best_checkpoint.pt   <- conditioning + perceptual_grouping
                                              weights (pass to train.py)
    cub_slot_finetune/metrics.csv
    cub_slot_finetune/viz/epoch_NNNN.png   <- slot segmentation grids
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    # Data
    "csv_path":   "FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv",
    "image_root": "FG-datset/CUB_200_2011/images",
    # Model
    "dinosaur_cfg_name": "projects/bridging/dinosaur/coco_feat_rec_dino_small16_auto_dinov3",
    "dinosaur_ckpt":     None,
    "n_slots":  None,
    # More SA iterations → slots converge to spatially stable regions.
    # The checkpoint was trained with 3; 7 re-uses the same weights but gives
    # the competitive dynamics more time to settle on CUB images.
    "sa_iters": 7,
    # Optimisation
    "lr":           2e-4,
    "weight_decay": 1e-2,
    "batch_size":   32,
    "n_epochs":     30,
    "warmup_steps": 500,
    # Entropy regularisation — breaks slot collapse (uniform slot usage).
    # Lowered from 0.05 now that TV loss handles spatial structure.
    "entropy_weight": 0.02,
    # Total variation on slot masks — penalises the checkerboard pattern by
    # encouraging spatially adjacent patches to belong to the same slot.
    # Acts on the 14×14 patch grid before upsampling.
    "tv_weight": 0.01,
    # Logging
    "out_dir":   "cub_slot_finetune",
    "viz_every": 5,    # produce a slot-seg grid every N epochs (also epoch 1)
    "viz_n":     8,    # number of fixed val images in each viz grid
    # Infra
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]


# ── Dataset — images only, no labels ─────────────────────────────────────────

class CUBImageDataset(Dataset):
    """Unique CUB-200 images for a given split, no attribute labels."""

    def __init__(self, csv_path: str, image_root: str, split: str, transform):
        df = pd.read_csv(csv_path)
        self.image_names = df[df["split"] == split]["image_name"].unique().tolist()
        self.image_root  = image_root
        self.transform   = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = os.path.join(self.image_root, self.image_names[idx])
        return self.transform(PILImage.open(path).convert("RGB"))


def make_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


# ── Load DINOSAUR modules from checkpoint ────────────────────────────────────

def load_dinosaur_for_finetune(cfg_name: str, ckpt_path: str, n_slots: int, sa_iters: int = 7):
    """Instantiate and load feature_extractor, conditioning, perceptual_grouping,
    and object_decoder from the DINOSAUR checkpoint.

    Returns (feature_extractor, conditioning, perceptual_grouping, object_decoder)
    on CPU.  feature_extractor is frozen; the rest have requires_grad=True.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import ocl.cli._config   # registers OmegaConf resolvers
    import ocl.cli.train     # registers 'training_config' in ConfigStore
    import hydra
    import hydra_zen
    from hydra.core.global_hydra import GlobalHydra

    configs_dir  = os.path.join(repo_root, "configs")
    map_location = None if torch.cuda.is_available() else torch.device("cpu")

    GlobalHydra.instance().clear()
    try:
        with hydra.initialize_config_dir(config_dir=configs_dir, version_base="1.1"):
            cfg = hydra.compose(
                config_name="training_config",
                overrides=[
                    f"+experiment={cfg_name}",
                    "models.feature_extractor.pretrained=false",
                    f"models.conditioning.n_slots={n_slots}",
                    f"+models.perceptual_grouping.iters={sa_iters}",
                ],
            )
    finally:
        GlobalHydra.instance().clear()

    feature_extractor   = hydra_zen.instantiate(cfg.models.feature_extractor,  _convert_="all")
    conditioning        = hydra_zen.instantiate(cfg.models.conditioning,        _convert_="all")
    perceptual_grouping = hydra_zen.instantiate(cfg.models.perceptual_grouping, _convert_="all")
    object_decoder      = hydra_zen.instantiate(cfg.models.object_decoder,      _convert_="all")

    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd   = ckpt["state_dict"]

    def filtered_sd(prefix: str) -> dict:
        full = f"models.{prefix}."
        return {k[len(full):]: v for k, v in sd.items() if k.startswith(full)}

    feature_extractor.load_state_dict(filtered_sd("feature_extractor"))
    perceptual_grouping.load_state_dict(filtered_sd("perceptual_grouping"))
    object_decoder.load_state_dict(filtered_sd("object_decoder"))

    # Load conditioning if n_slots matches the checkpoint; else keep random init.
    cond_sd = filtered_sd("conditioning")
    try:
        conditioning.load_state_dict(cond_sd, strict=True)
        print(f"  conditioning    : loaded from checkpoint (n_slots={n_slots})")
    except RuntimeError:
        ckpt_n = next(iter(cond_sd.values())).shape[1]
        print(f"  conditioning    : random init (n_slots={n_slots}, ckpt={ckpt_n})")

    print(f"  perceptual_grouping: loaded from checkpoint  (iters={sa_iters})")
    print(f"  object_decoder      : loaded from checkpoint")

    # Freeze feature extractor only.
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad_(False)

    return feature_extractor, conditioning, perceptual_grouping, object_decoder


# ── Forward pass + MSE reconstruction loss ───────────────────────────────────

def forward_and_loss(
    feature_extractor,
    conditioning,
    perceptual_grouping,
    object_decoder,
    images: torch.Tensor,
    entropy_weight: float = 0.0,
    tv_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Run full pipeline and return (loss, slot_masks, recon_loss_val, entropy_val).

    slot_masks: (B, N_slots, N_patches) — used for visualisation.

    Loss = MSE(reconstruction, normalised_target) - entropy_weight * H(slot_usage)

    The entropy term maximises H over the K-dimensional average slot-usage vector:
        avg_usage[b, k] = mean_p slot_masks[b, k, p]   (fraction of patches slot k owns)
    Maximising entropy pushes all slots toward claiming ~1/K of patches rather
    than one slot monopolising all of them.  Optimal value is log(K) ≈ 1.95 for
    K=7.  Watch the "H=" value during training: a healthy run goes from ~0.3
    (collapsed) toward ~1.9 (uniform utilisation).
    """
    B = images.shape[0]
    routing: dict = {"input": {"image": images, "batch_size": B}}

    with torch.no_grad():
        routing["feature_extractor"] = feature_extractor(inputs=routing)

    routing["conditioning"]        = conditioning(inputs=routing)
    routing["perceptual_grouping"] = perceptual_grouping(inputs=routing)
    routing["object_decoder"]      = object_decoder(inputs=routing)

    reconstruction = routing["object_decoder"].reconstruction   # (B, 196, 384)
    target         = routing["object_decoder"].target.detach()  # (B, 196, 384)

    # Normalise target per patch (exact match to ReconstructionLoss default).
    mean        = target.mean(dim=-1, keepdim=True)
    var         = target.var(dim=-1, keepdim=True)
    target_norm = (target - mean) / (var + 1e-6).sqrt()

    recon_loss = F.mse_loss(reconstruction, target_norm)
    slot_masks = routing["perceptual_grouping"].feature_attributions  # (B, K, P)

    # Slot utilisation entropy regulariser.
    avg_usage    = slot_masks.mean(dim=2)                                      # (B, K)
    slot_entropy = -(avg_usage * (avg_usage + 1e-8).log()).sum(dim=-1).mean()  # scalar

    # Total variation on the slot mask grid.
    # Penalises the checkerboard pattern: adjacent patches should prefer the
    # same slot.  Pad to the next perfect square first — the ViT feature
    # extractor drops register tokens, leaving 192 patches not 196.
    B_sz, K_sz, P_sz = slot_masks.shape
    grid_side = math.isqrt(P_sz)
    if grid_side * grid_side < P_sz:
        grid_side += 1
    n_pad = grid_side * grid_side - P_sz
    if n_pad > 0:
        pad = slot_masks.new_zeros(B_sz, K_sz, n_pad)
        masks_2d = torch.cat([slot_masks, pad], dim=2).view(B_sz, K_sz, grid_side, grid_side)
    else:
        masks_2d = slot_masks.view(B_sz, K_sz, grid_side, grid_side)
    tv_h  = (masks_2d[:, :, 1:, :] - masks_2d[:, :, :-1, :]).abs().mean()
    tv_w  = (masks_2d[:, :, :, 1:] - masks_2d[:, :, :, :-1]).abs().mean()
    tv_loss = tv_h + tv_w

    loss = recon_loss - entropy_weight * slot_entropy + tv_weight * tv_loss
    return loss, slot_masks, recon_loss.item(), slot_entropy.item(), tv_loss.item()


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_viz(
    feature_extractor,
    conditioning,
    perceptual_grouping,
    object_decoder,
    images: torch.Tensor,   # (N, 3, 224, 224) already on device
    epoch: int,
    out_dir: Path,
) -> None:
    """Save a grid of (original image | slot segmentation overlay) rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm as _cm

    mean_t = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGE_STD).view(3, 1, 1)

    def denorm(t: torch.Tensor) -> np.ndarray:
        return (t.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

    conditioning.eval()
    perceptual_grouping.eval()
    object_decoder.eval()

    with torch.no_grad():
        _, slot_masks, _, _, _ = forward_and_loss(
            feature_extractor, conditioning, perceptual_grouping, object_decoder, images
        )

    slot_masks = slot_masks.cpu()   # (N, K, P)
    N, K, P    = slot_masks.shape
    grid_side  = math.isqrt(P)
    if grid_side * grid_side < P:
        grid_side += 1
    n_pad = grid_side ** 2 - P
    if n_pad:
        slot_masks = torch.cat([slot_masks, torch.zeros(N, K, n_pad)], dim=2)

    cmap   = _cm.get_cmap("tab20" if K <= 20 else "turbo", K)
    colors = np.array([cmap(i)[:3] for i in range(K)], dtype=np.float32)

    fig, axes = plt.subplots(N, 2, figsize=(6, N * 3.5))
    if N == 1:
        axes = axes[None, :]

    for i in range(N):
        img_vis  = denorm(images[i])
        masks_2d = slot_masks[i].view(K, grid_side, grid_side)
        masks_up = F.interpolate(
            masks_2d.unsqueeze(0).float(), size=(224, 224),
            mode="bilinear", align_corners=False,
        )[0].numpy()   # (K, 224, 224)

        argmax  = masks_up.argmax(axis=0)          # (224, 224) dominant slot
        seg_rgb = colors[argmax]                   # (224, 224, 3)
        overlay = (0.55 * seg_rgb + 0.45 * img_vis).clip(0, 1)

        axes[i, 0].imshow(img_vis)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay)
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 0].set_title("image", fontsize=7)
            axes[i, 1].set_title("slot segmentation", fontsize=7)

    plt.suptitle(f"Epoch {epoch} — CUB slot fine-tuning", fontsize=9)
    plt.tight_layout()
    (out_dir / "viz").mkdir(exist_ok=True)
    plt.savefig(out_dir / "viz" / f"epoch_{epoch:04d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    device  = torch.device(cfg["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = CUBImageDataset(
        cfg["csv_path"], cfg["image_root"], split="train",
        transform=make_transforms(train=True),
    )
    val_ds = CUBImageDataset(
        cfg["csv_path"], cfg["image_root"], split="test",
        transform=make_transforms(train=False),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Batch: {cfg['batch_size']}")

    # ── Load modules ─────────────────────────────────────────────────────────
    print("Loading DINOSAUR modules …")
    fe, cond, pg, decoder = load_dinosaur_for_finetune(
        cfg["dinosaur_cfg_name"], cfg["dinosaur_ckpt"], cfg["n_slots"],
        sa_iters=cfg["sa_iters"],
    )
    fe      = fe.to(device)
    cond    = cond.to(device)
    pg      = pg.to(device)
    decoder = decoder.to(device)

    # ── Optimiser & LR schedule ──────────────────────────────────────────────
    trainable_params = (
        list(cond.parameters())
        + list(pg.parameters())
        + list(decoder.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,}  (frozen ViT excluded)")

    optimizer   = torch.optim.AdamW(
        trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["n_epochs"] * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        progress = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Fixed val images for visualisation ───────────────────────────────────
    rng         = np.random.RandomState(42)
    viz_indices = rng.choice(len(val_ds), size=min(cfg["viz_n"], len(val_ds)), replace=False)
    viz_images  = torch.stack([val_ds[int(i)] for i in viz_indices]).to(device)

    ew  = cfg["entropy_weight"]
    tvw = cfg["tv_weight"]
    K   = cfg["n_slots"]
    H_max = math.log(K)
    print(f"entropy_weight={ew}  tv_weight={tvw}  sa_iters={cfg['sa_iters']}")
    print(f"H_max={H_max:.3f} (log {K}). Watch H: collapse≈0.3, good≈{H_max:.1f}")

    # ── Metrics CSV ──────────────────────────────────────────────────────────
    metrics_fh = open(out_dir / "metrics.csv", "w", newline="")
    metrics_w  = csv.writer(metrics_fh)
    metrics_w.writerow(["epoch", "train_loss", "train_recon", "train_H", "train_TV",
                         "val_loss", "val_recon", "val_H", "val_TV", "best"])

    # ── Loop ─────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, cfg["n_epochs"] + 1):

        # Train
        cond.train(); pg.train(); decoder.train()
        sum_loss, sum_recon, sum_H, sum_TV, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss, _, recon_val, H_val, TV_val = forward_and_loss(
                fe, cond, pg, decoder, images, entropy_weight=ew, tv_weight=tvw
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            sum_loss  += loss.item()
            sum_recon += recon_val
            sum_H     += H_val
            sum_TV    += TV_val
            n_batches += 1
        train_loss  = sum_loss  / n_batches
        train_recon = sum_recon / n_batches
        train_H     = sum_H     / n_batches
        train_TV    = sum_TV    / n_batches

        # Val
        cond.eval(); pg.eval(); decoder.eval()
        vsum_loss, vsum_recon, vsum_H, vsum_TV, val_n = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                loss, _, recon_val, H_val, TV_val = forward_and_loss(
                    fe, cond, pg, decoder, images, entropy_weight=ew, tv_weight=tvw
                )
                vsum_loss  += loss.item()
                vsum_recon += recon_val
                vsum_H     += H_val
                vsum_TV    += TV_val
                val_n      += 1
        val_loss  = vsum_loss  / val_n
        val_recon = vsum_recon / val_n
        val_H     = vsum_H     / val_n
        val_TV    = vsum_TV    / val_n

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":               epoch,
                    "val_loss":            val_loss,
                    "conditioning":        cond.state_dict(),
                    "perceptual_grouping": pg.state_dict(),
                    "object_decoder":      decoder.state_dict(),
                    "config":              cfg,
                },
                out_dir / "best_checkpoint.pt",
            )

        marker = " ↑ best" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{cfg['n_epochs']} | "
            f"train recon={train_recon:.4f} H={train_H:.3f} TV={train_TV:.4f} | "
            f"val   recon={val_recon:.4f} H={val_H:.3f} TV={val_TV:.4f}{marker}"
        )
        metrics_w.writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_recon:.6f}", f"{train_H:.4f}", f"{train_TV:.4f}",
            f"{val_loss:.6f}", f"{val_recon:.6f}", f"{val_H:.4f}", f"{val_TV:.4f}",
            int(is_best),
        ])
        metrics_fh.flush()

        # Visualisation
        if cfg["viz_every"] > 0 and (epoch % cfg["viz_every"] == 0 or epoch == 1):
            save_viz(fe, cond, pg, decoder, viz_images, epoch, out_dir)
            print(f"  → viz/epoch_{epoch:04d}.png")

    metrics_fh.close()
    print(
        f"\nDone.  Best val loss: {best_val_loss:.4f}\n"
        f"Fine-tuned checkpoint → {out_dir / 'best_checkpoint.pt'}\n\n"
        f"Use it in the classifier:\n"
        f"    python train.py --finetune_ckpt {out_dir / 'best_checkpoint.pt'}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DINOSAUR slot attention on CUB-200")
    p.add_argument("--n_epochs",      type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--n_slots",       type=int,   default=None)
    p.add_argument("--dinosaur_ckpt", type=str,   default=None)
    p.add_argument("--out_dir",       type=str,   default=None)
    p.add_argument("--viz_every",      type=int,   default=None)
    p.add_argument("--warmup_steps",   type=int,   default=None)
    p.add_argument("--entropy_weight", type=float, default=None,
                   help="Slot utilisation entropy weight (default: 0.02)")
    p.add_argument("--tv_weight",      type=float, default=None,
                   help="Total variation weight on slot masks (default: 0.01). "
                        "Increase if checkerboard pattern persists.")
    p.add_argument("--sa_iters",       type=int,   default=None,
                   help="Slot attention iterations (default: 7, checkpoint used 3)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = dict(CONFIG)
    for key in ("n_epochs", "lr", "batch_size", "n_slots", "dinosaur_ckpt",
                "out_dir", "viz_every", "warmup_steps", "entropy_weight",
                "tv_weight", "sa_iters"):
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
    train(cfg)


if __name__ == "__main__":
    main()
