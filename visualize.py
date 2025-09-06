import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Directory containing your .npy files
output_dir = "./outputs/val"
save_dir = "./visualization_results/checkpoint_98"
os.makedirs(save_dir, exist_ok=True)

# Get all sample indices
sample_indices = sorted(set([path.stem.split('.')[0] for path in Path(output_dir).glob("*.npy")]))

def colorize_masks(masks: np.ndarray) -> np.ndarray:
    """Convert [K,H,W] soft masks to a color visualization [H,W,3]."""
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [K,H,W] or [H,W], got {masks.shape}")
    k, h, w = masks.shape
    seg_img = np.zeros((h, w, 3), dtype=np.float32)
    cmap = plt.cm.get_cmap('tab20', max(1, k))
    for i in range(k):
        color = cmap(i)[:3]
        for c in range(3):
            seg_img[:, :, c] += masks[i] * color[c]
    return np.clip(seg_img, 0.0, 1.0)


for idx in sample_indices:
    # Create figure with subplots (Original, Parent masks, Child masks, Reconstruction)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Load and display original image
    orig_img_path = Path(output_dir) / f"{idx}.input.orig_image.npy"
    if orig_img_path.exists():
        orig_img = np.load(orig_img_path)
        # Convert from [C,H,W] to [H,W,C] if needed
        if orig_img.shape[0] == 3:
            orig_img = np.transpose(orig_img, (1, 2, 0))
        axes[0].imshow(np.clip(orig_img, 0, 1))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
    
    # 2. Load and display parent masks (colored). Prefer parent_masks_128 if present, else masks_resized
    parent_path = Path(output_dir) / f"{idx}.parent_masks_128.npy"
    fallback_path = Path(output_dir) / f"{idx}.masks_resized.npy"
    if parent_path.exists() or fallback_path.exists():
        parent_masks = np.load(parent_path) if parent_path.exists() else np.load(fallback_path)
        if parent_masks.ndim == 4:
            parent_masks = parent_masks[0]
        seg_img_parent = colorize_masks(parent_masks)
        axes[1].imshow(seg_img_parent)
        axes[1].set_title("Parent Masks")
        axes[1].axis('off')
    else:
        axes[1].set_title("Parent Masks N/A")
        axes[1].axis('off')

    # 3. Load and display child masks (colored)
    child_path = Path(output_dir) / f"{idx}.child_masks_128.npy"
    if child_path.exists():
        child_masks = np.load(child_path)
        if child_masks.ndim == 4:
            child_masks = child_masks[0]
        seg_img_child = colorize_masks(child_masks)
        axes[2].imshow(seg_img_child)
        axes[2].set_title("Child Masks")
        axes[2].axis('off')
    else:
        axes[2].set_title("Child Masks N/A")
        axes[2].axis('off')
    
    # 4. Load and display reconstruction if available
    recon_path = Path(output_dir) / f"{idx}.object_decoder.reconstruction.npy"
    if recon_path.exists():
        recon = np.load(recon_path)
        if recon.shape[0] == 3:  # [C,H,W]
            recon = np.transpose(recon, (1, 2, 0))
        axes[3].imshow(np.clip(recon, 0, 1))
        axes[3].set_title("Reconstruction")
        axes[3].axis('off')
    else:
        axes[3].set_title("Reconstruction Not Available")
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"visualization_{idx}.png"))
    plt.close(fig)
    
    print(f"Saved visualization for sample {idx}")

print(f"All visualizations saved to {save_dir}")
