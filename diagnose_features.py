"""Diagnostic: DINOv3 patch features on CUB-200 images.

Three complementary views per image:

  1. PCA of patch features (first 3 PCs → RGB)
     Fitted globally across all sampled images so colours are consistent.
     If the bird region and background map to different colours the features
     are spatially discriminative enough to support slot segmentation.

  2. Self-attention maps from the last ViT block (one subplot per head)
     CLS-token → patch attention.  DINO/DINOv3 heads are known to attend to
     foreground objects; this confirms whether that holds on CUB images.

  3. Cosine similarity from the centre patch
     Pick the patch closest to the image centre, show cosine similarity to
     every other patch.  A good feature should be most similar to nearby
     bird patches and dissimilar to background patches.

Usage:
    python diagnose_features.py
    python diagnose_features.py --model vit_small_patch16_224.dino --n_images 8
    python diagnose_features.py --out_dir my_diagnostics/

Output:
    <out_dir>/feature_pca.png       — PCA grid
    <out_dir>/attention_maps.png    — per-head CLS attention
    <out_dir>/cosine_sim.png        — centre-patch similarity
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image as PILImage
from sklearn.decomposition import PCA
from torchvision import transforms

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "model":      "vit_small_patch16_dinov3.lvd1689m",   # timm model name — matches DINOSAUR DINOv3 checkpoint
    "pretrained": True,
    "csv_path":   "FG-datset/CUB_200_2011/cub200_ranked_classification_dataset.csv",
    "image_root": "FG-datset/CUB_200_2011/images",
    "split":      "test",
    "n_images":   6,
    "seed":       42,
    "out_dir":    "diagnostics",
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, pretrained: bool, device: torch.device):
    """Load timm ViT, disable fused attention on last block so we can hook it."""
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval().to(device)

    # fused_attn uses F.scaled_dot_product_attention which returns no weights.
    # Disable it on the last block only so we can read the attention matrix.
    last_attn = model.blocks[-1].attn
    last_attn.fused_attn = False

    n_heads           = last_attn.num_heads
    num_prefix_tokens = getattr(model, "num_prefix_tokens", 1)  # CLS [+ registers]

    print(f"Model          : {model_name}")
    print(f"Prefix tokens  : {num_prefix_tokens}  (CLS + {num_prefix_tokens-1} registers)")
    print(f"Attention heads: {n_heads}")

    return model, n_heads, num_prefix_tokens


# ── Image loading ─────────────────────────────────────────────────────────────

def load_images(csv_path, image_root, split, n_images, seed):
    df = pd.read_csv(csv_path)
    unique_imgs = df[df["split"] == split]["image_name"].unique().tolist()
    rng = np.random.RandomState(seed)
    chosen = rng.choice(unique_imgs, size=min(n_images, len(unique_imgs)), replace=False)

    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    mean_t = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGE_STD).view(3, 1, 1)

    tensors, originals, names = [], [], []
    for name in chosen:
        path = os.path.join(image_root, name)
        pil  = PILImage.open(path).convert("RGB")
        t    = tf(pil)
        vis  = (t * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        tensors.append(t)
        originals.append(vis)
        names.append(name)

    return torch.stack(tensors), originals, names   # (N, 3, 224, 224)


# ── Feature + attention extraction ───────────────────────────────────────────

def extract(model, images: torch.Tensor, num_prefix_tokens: int, device):
    """Return (patch_features, attn_maps).

    patch_features : (N, P, D)   — spatial patch tokens only
    attn_maps      : (N, H, P, P) — last-block attention among patch tokens
                     (prefix tokens stripped; CLS-to-patches slice = attn[:,:,0,:])
    """
    attn_store = []

    def hook(module, inp, out):
        # inp[0] is the attention weight (B, H, N_tok, N_tok) in the non-fused path
        attn_store.append(inp[0].detach().cpu())

    handle = model.blocks[-1].attn.attn_drop.register_forward_hook(hook)

    images = images.to(device)
    with torch.no_grad():
        feats = model.forward_features(images)   # (N, n_prefix + P, D)

    handle.remove()

    patch_feats = feats[:, num_prefix_tokens:, :].cpu()   # (N, P, D)
    P = patch_feats.shape[1]
    n_patches_side = math.isqrt(P)
    print(f"Patch grid     : {n_patches_side}×{n_patches_side} = {P} patches")

    attn_full   = attn_store[0]                            # (N, H, N_tok, N_tok)

    # Strip prefix rows AND columns so we have patch-only attention (N, H, P+1, P)
    # Row 0 = CLS → patch attention (what we visualise)
    attn_maps = attn_full[:, :, :num_prefix_tokens+1, num_prefix_tokens:]  # keep CLS row

    return patch_feats, attn_maps, n_patches_side


# ── Plot 1: PCA ───────────────────────────────────────────────────────────────

def plot_pca(patch_feats, originals, names, n_patches_side, out_dir):
    """PCA of all patch features, first 3 PCs → RGB per image."""
    N, P, D = patch_feats.shape
    all_feats = patch_feats.reshape(N * P, D).numpy()

    pca   = PCA(n_components=3)
    pcs   = pca.fit_transform(all_feats)                          # (N*P, 3)
    pcs   = pcs.reshape(N, P, 3)
    var   = pca.explained_variance_ratio_
    print(f"PCA var explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2]:.1%}")

    # Normalise each PC across all images jointly → consistent colours
    for c in range(3):
        lo, hi = pcs[:, :, c].min(), pcs[:, :, c].max()
        pcs[:, :, c] = (pcs[:, :, c] - lo) / (hi - lo + 1e-8)

    fig, axes = plt.subplots(N, 2, figsize=(6, N * 3.2))
    if N == 1:
        axes = axes[None, :]

    for i in range(N):
        pca_img = pcs[i].reshape(n_patches_side, n_patches_side, 3)
        pca_up  = torch.tensor(pca_img).permute(2, 0, 1).unsqueeze(0).float()
        pca_up  = F.interpolate(pca_up, size=(224, 224), mode="bilinear",
                                align_corners=False)[0].permute(1, 2, 0).numpy()

        axes[i, 0].imshow(originals[i]);  axes[i, 0].axis("off")
        axes[i, 1].imshow(pca_up.clip(0, 1));  axes[i, 1].axis("off")
        species = names[i].split("/")[0].replace("_", " ")
        axes[i, 0].set_title(species, fontsize=6)
        if i == 0:
            axes[i, 1].set_title(
                f"PCA (PC1={var[0]:.0%} PC2={var[1]:.0%} PC3={var[2]:.0%})",
                fontsize=6,
            )

    plt.suptitle("DINOv3 patch features — PCA (first 3 PCs → RGB)", fontsize=9)
    plt.tight_layout()
    out = out_dir / "feature_pca.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Plot 2: Attention maps ────────────────────────────────────────────────────

def plot_attention(attn_maps, originals, names, n_patches_side, out_dir):
    """CLS → patch attention per head, last ViT block."""
    N, n_heads = attn_maps.shape[0], attn_maps.shape[1]
    # attn_maps: (N, H, n_prefix+1, P) — row 0 = CLS

    n_cols = 1 + n_heads   # original + one per head
    fig, axes = plt.subplots(N, n_cols, figsize=(n_cols * 2.5, N * 3.0))
    if N == 1:
        axes = axes[None, :]

    for i in range(N):
        axes[i, 0].imshow(originals[i]);  axes[i, 0].axis("off")
        species = names[i].split("/")[0].replace("_", " ")
        axes[i, 0].set_title(species, fontsize=6)

        cls_attn = attn_maps[i, :, 0, :]   # (H, P)  — CLS→patches

        for h in range(n_heads):
            a = cls_attn[h].reshape(n_patches_side, n_patches_side).numpy()
            a_up = F.interpolate(
                torch.tensor(a).unsqueeze(0).unsqueeze(0),
                size=(224, 224), mode="bilinear", align_corners=False,
            )[0, 0].numpy()
            axes[i, 1 + h].imshow(a_up, cmap="inferno")
            axes[i, 1 + h].axis("off")
            if i == 0:
                axes[i, 1 + h].set_title(f"head {h}", fontsize=7)

    plt.suptitle(
        "DINOv3 — CLS→patch attention (last block)\n"
        "Bright = patches the CLS token attends to",
        fontsize=9,
    )
    plt.tight_layout()
    out = out_dir / "attention_maps.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Plot 3: Cosine similarity from centre patch ───────────────────────────────

def plot_cosine_sim(patch_feats, originals, names, n_patches_side, out_dir):
    """Cosine similarity from the centre patch to all other patches."""
    N, P, D = patch_feats.shape
    cx = cy  = n_patches_side // 2
    centre_idx = cy * n_patches_side + cx

    fig, axes = plt.subplots(N, 2, figsize=(6, N * 3.2))
    if N == 1:
        axes = axes[None, :]

    for i in range(N):
        feats = F.normalize(patch_feats[i], dim=-1)   # (P, D)
        query = feats[centre_idx]                      # (D,)
        sims  = (feats @ query).numpy()                # (P,)

        sim_2d = sims.reshape(n_patches_side, n_patches_side)
        sim_up = F.interpolate(
            torch.tensor(sim_2d).unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224), mode="bilinear", align_corners=False,
        )[0, 0].numpy()

        # Overlay: blend similarity heatmap with original image
        sim_norm  = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-8)
        cmap_rgba = plt.cm.viridis(sim_norm)[:, :, :3]
        overlay   = (0.5 * originals[i] + 0.5 * cmap_rgba).clip(0, 1)

        axes[i, 0].imshow(originals[i])
        # Mark centre patch
        patch_px = 224 // n_patches_side
        rect_x   = cx * patch_px
        rect_y   = cy * patch_px
        axes[i, 0].add_patch(plt.Rectangle(
            (rect_x, rect_y), patch_px, patch_px,
            fill=False, edgecolor="red", linewidth=2,
        ))
        axes[i, 0].axis("off")
        axes[i, 0].set_title(names[i].split("/")[0].replace("_", " "), fontsize=6)

        axes[i, 1].imshow(overlay)
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("cosine sim from centre patch (viridis)", fontsize=6)

    plt.suptitle(
        "DINOv3 — cosine similarity from centre patch\n"
        "Bright = similar to red-boxed patch  |  If bird ≠ background → features are discriminative",
        fontsize=9,
    )
    plt.tight_layout()
    out = out_dir / "cosine_sim.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_feature_stats(patch_feats, attn_maps):
    """Print basic statistics that indicate feature quality."""
    N, P, D = patch_feats.shape
    feats_norm = F.normalize(patch_feats, dim=-1)

    # Mean pairwise cosine similarity within each image (lower = more diverse)
    sims = torch.bmm(feats_norm, feats_norm.transpose(1, 2))   # (N, P, P)
    mask = ~torch.eye(P, dtype=torch.bool).unsqueeze(0)
    mean_sim = sims[mask.expand(N, -1, -1)].mean().item()

    # Per-image feature variance (higher = richer encoding)
    var_per_img = patch_feats.var(dim=1).mean().item()

    # Attention entropy: low = focused (good), high = diffuse (bad)
    cls_attn    = attn_maps[:, :, 0, :]             # (N, H, P)
    attn_ent    = -(cls_attn * (cls_attn + 1e-8).log()).sum(dim=-1).mean().item()
    attn_ent_max = math.log(P)

    print(f"\nFeature stats across {N} images:")
    print(f"  Mean pairwise cosine sim : {mean_sim:.4f}  "
          f"(→0 = diverse patches, →1 = all patches look the same)")
    print(f"  Mean patch feature var   : {var_per_img:.4f}  (higher = richer)")
    print(f"  CLS attn entropy         : {attn_ent:.4f} / {attn_ent_max:.4f}  "
          f"(lower = more focused attention)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    device  = torch.device(cfg["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model …")
    model, n_heads, num_prefix_tokens = load_model(
        cfg["model"], cfg["pretrained"], device
    )

    print("Loading CUB images …")
    images, originals, names = load_images(
        cfg["csv_path"], cfg["image_root"],
        cfg["split"], cfg["n_images"], cfg["seed"],
    )
    print(f"  {len(names)} images loaded")

    print("Extracting features …")
    patch_feats, attn_maps, n_patches_side = extract(model, images, num_prefix_tokens, device)
    print(f"  patch_feats : {tuple(patch_feats.shape)}   (N, P, D)")
    print(f"  attn_maps   : {tuple(attn_maps.shape)}   (N, H, prefix+1, P)")

    print_feature_stats(patch_feats, attn_maps)

    print("\nSaving plots …")
    plot_pca(patch_feats, originals, names, n_patches_side, out_dir)
    plot_attention(attn_maps, originals, names, n_patches_side, out_dir)
    plot_cosine_sim(patch_feats, originals, names, n_patches_side, out_dir)

    print("\nDone.")


def parse_args():
    p = argparse.ArgumentParser(description="DINOv3 feature diagnostics on CUB-200")
    p.add_argument("--model",     type=str, default=None,
                   help="timm model name (default: vit_small_patch16_dinov3)")
    p.add_argument("--n_images",  type=int, default=None)
    p.add_argument("--split",     type=str, default=None, choices=["train", "test"])
    p.add_argument("--seed",      type=int, default=None)
    p.add_argument("--out_dir",   type=str, default=None)
    p.add_argument("--no_pretrained", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = dict(CONFIG)
    for key in ("model", "n_images", "split", "seed", "out_dir"):
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
    if args.no_pretrained:
        cfg["pretrained"] = False
    run(cfg)


if __name__ == "__main__":
    main()
