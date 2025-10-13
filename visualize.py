import os
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

# Means/stds used by ImageNet-style normalization; used to denormalize tensors if needed.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Try to reuse the repo's segmentation colormap and overlay; fall back to matplotlib if unavailable.
try:
    from ocl.visualizations import Segmentation as _Segmentation
    import torch

    def _get_repo_cmap(n_classes: int):
        return _Segmentation()._get_cmap(n_classes)

    _HAS_SEG = True

except Exception:  # pragma: no cover
    from matplotlib import cm as _cm

    def _get_repo_cmap(n_classes: int):
        if n_classes <= 20:
            mpl = _cm.get_cmap("tab20", n_classes)(range(n_classes))
        else:
            mpl = _cm.get_cmap("turbo", n_classes)(range(n_classes))
        return [tuple((255 * c[:3]).astype(int)) for c in mpl]
    _HAS_SEG = False

parser = argparse.ArgumentParser(description="Visualize evaluation outputs (.npy files)")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./outputs/val/epoch_43-step_323897_3_slots_depth_2",
    help="Directory containing .npy outputs (e.g., ./outputs/val/<relevant_name>)",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./visualization_results/epoch_43-step_323897_3_slots_depth_2",
    help="Directory to save visualization PNGs",
)


def gather_sample_indices(output_dir: Union[str, Path]) -> List[str]:
    directory = Path(output_dir)
    all_stems = [path.stem for path in directory.glob("*.npy")]
    return sorted({stem.split(".")[0] for stem in all_stems})


def _ensure_hwc01(img: np.ndarray) -> np.ndarray:
    """Ensure image is HWC and in [0,1] for display, denormalizing ImageNet tensors if needed."""
    arr = img
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] <= arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.float32, copy=False)
    if arr.min() < 0.0:
        # Likely ImageNet-normalized tensor.
        arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    elif arr.max() > 1.01:
        # Likely 0-255 data.
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_khw(masks: np.ndarray) -> np.ndarray:
    """Best-effort conversion of masks to [K,H,W].

    Acceptable inputs: [K,H,W], [B,K,H,W] (takes first), [H,W,K], [K,H,W,1], [H,W,K,1], [H,W].
    """
    arr = masks
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim == 4:
        # Assume [B,K,H,W], take first
        arr = arr[0]
    if arr.ndim == 2:
        # Already labels, convert to one-hot with inferred K
        labels = arr.astype(np.int64)
        k = int(labels.max()) + 1 if labels.size > 0 else 1
        oh = np.eye(k, dtype=np.float32)[labels]  # [H,W,K]
        arr = np.moveaxis(oh, -1, 0)  # [K,H,W]
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")
    # Decide if channel-first or channel-last
    k_first = arr.shape[0]
    h1, w1 = arr.shape[1], arr.shape[2]
    # Heuristic: K is usually the smallest dimension
    if k_first <= min(h1, w1):
        return arr
    else:
        # Probably [H,W,K]
        return np.moveaxis(arr, -1, 0)


def _restrict_children_to_dominant_parent(
    children_masks: np.ndarray, parent_masks: np.ndarray
) -> np.ndarray:
    """Zero children outside the region where their parent is the argmax mask."""

    try:
        children_khw = _to_khw(children_masks)
    except ValueError:
        return children_masks
    try:
        parent_khw = _to_khw(parent_masks)
    except ValueError:
        return children_khw

    if children_khw.shape[1:] != parent_khw.shape[1:]:
        return children_khw

    k_parent = parent_khw.shape[0]
    if k_parent <= 0:
        return children_khw

    total_children = children_khw.shape[0]
    if total_children % k_parent != 0:
        return children_khw

    per_parent = total_children // k_parent
    restricted = children_khw.copy()
    for parent_idx in range(k_parent):
        start = parent_idx * per_parent
        end = start + per_parent
        mask = parent_khw[parent_idx].astype(restricted.dtype, copy=False)
        restricted[start:end] *= mask[None, :, :]

    return restricted


def _resize_masks_to_hw(masks_khw: np.ndarray, target_h: int, target_w: int):
    """Resize [K,H,W] masks to requested spatial size using available backends."""
    if masks_khw.shape[1:] == (target_h, target_w):
        return masks_khw

    torch_mod = globals().get("torch")
    if torch_mod is None:
        try:
            import torch as torch_mod  # type: ignore
        except Exception:  # pragma: no cover
            torch_mod = None
    if torch_mod is not None:
        try:
            import torch.nn.functional as F  # type: ignore

            tensor = torch_mod.from_numpy(masks_khw)[None].to(torch_mod.float32)
            resized = F.interpolate(
                tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            return resized[0].cpu().numpy()
        except Exception:  # pragma: no cover
            pass

    scale_h = target_h / masks_khw.shape[1]
    scale_w = target_w / masks_khw.shape[2]
    if float(scale_h).is_integer() and float(scale_w).is_integer():
        return np.repeat(
            np.repeat(masks_khw, repeats=int(scale_h), axis=1),
            repeats=int(scale_w),
            axis=2,
        )

    try:
        from PIL import Image

        resized_masks = []
        for mask in masks_khw:
            img = Image.fromarray(mask.astype(np.float32), mode="F")
            resized = img.resize((target_w, target_h), resample=Image.BILINEAR)
            resized_masks.append(np.array(resized, dtype=np.float32))
        return np.stack(resized_masks, axis=0)
    except Exception:  # pragma: no cover
        return None


def colorize_masks_info(masks: np.ndarray):
    """Colorize masks and return (segmentation_rgb, num_classes).

    Accepts various input mask shapes and converts to [K,H,W] internally.
    """
    m = _to_khw(masks)
    k, h, w = m.shape
    if k == 0:
        return np.zeros((h, w, 3), dtype=np.float32), 0
    labels = np.argmax(m, axis=0)
    cmap_arr = np.asarray(_get_repo_cmap(k), dtype=np.uint8)
    seg_rgb = cmap_arr[labels]
    return (seg_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0), k


def colorize_masks(masks: np.ndarray) -> np.ndarray:
    return colorize_masks_info(masks)[0]


def render_sample_panels(output_dir: Path, idx: str, axes: Sequence[plt.Axes]) -> None:
    output_dir = Path(output_dir)
    axes = list(axes)
    for ax in axes:
        ax.axis("off")

    # 1. Original image
    orig_img_path = output_dir / f"{idx}.input.orig_image.npy"
    alt_img_paths = [
        output_dir / f"{idx}.input.image.npy",
        output_dir / f"{idx}.input.rgb.npy",
    ]
    orig_img = None
    if orig_img_path.exists():
        orig_img = _ensure_hwc01(np.load(orig_img_path))
        axes[0].set_title("Original Image")
    else:
        for candidate in alt_img_paths:
            if candidate.exists():
                orig_img = _ensure_hwc01(np.load(candidate))
                axes[0].set_title(f"{candidate.stem.split('.')[-1].capitalize()} Image")
                break
    if orig_img is not None:
        axes[0].imshow(orig_img)
    else:
        axes[0].set_title("Original N/A")

    parent_masks_khw = None
    child_masks_khw = None
    grandchild_masks_khw = None

    # 2. Parent masks (prefer resized 128 if present)
    parent_path = output_dir / f"{idx}.parent_masks_128.npy"
    fallback_path = output_dir / f"{idx}.masks_resized.npy"
    if parent_path.exists() or fallback_path.exists():
        parent_masks_khw = _to_khw(np.load(parent_path) if parent_path.exists() else np.load(fallback_path))
        seg_img_parent, k_parent = colorize_masks_info(parent_masks_khw)
        axes[1].imshow(seg_img_parent)
        axes[1].set_title(f"Parent Masks (K={k_parent})")
    else:
        axes[1].set_title("Parent Masks N/A")

    # 3. Child masks
    child_path = output_dir / f"{idx}.child_masks_128.npy"
    if child_path.exists():
        child_masks_khw = _to_khw(np.load(child_path))
        if parent_masks_khw is not None:
            child_masks_khw = _restrict_children_to_dominant_parent(
                child_masks_khw, parent_masks_khw
            )
        seg_img_child, k_child = colorize_masks_info(child_masks_khw)
        axes[2].imshow(seg_img_child)
        axes[2].set_title(f"Child Masks (K={k_child})")
    else:
        axes[2].set_title("Child Masks N/A")

    # 4. Grandchild masks
    grandchild_path = output_dir / f"{idx}.grandchild_masks_128.npy"
    if grandchild_path.exists():
        grandchild_masks_khw = _to_khw(np.load(grandchild_path))
        parent_for_grandchildren = child_masks_khw if child_masks_khw is not None else parent_masks_khw
        if parent_for_grandchildren is not None:
            grandchild_masks_khw = _restrict_children_to_dominant_parent(
                grandchild_masks_khw, parent_for_grandchildren
            )
        seg_img_grandchild, k_grand = colorize_masks_info(grandchild_masks_khw)
        axes[3].imshow(seg_img_grandchild)
        axes[3].set_title(f"Grandchild Masks (K={k_grand})")
    else:
        axes[3].set_title("Grandchild Masks N/A")

    # 5. Overlay predicted grandchild masks on the original image (prefer resized masks if available)
    def _overlay_on_image(image_hwc01: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay masks onto image using repo Segmentation if available, else local alpha blend.

        Accepts masks as [K,H,W], [B,K,H,W] or [H,W] labels.
        """
        H, W = image_hwc01.shape[:2]
        try:
            mkhw = _to_khw(masks)
        except Exception:
            return None
        resized_masks = _resize_masks_to_hw(mkhw, H, W)
        if resized_masks is None:
            return None
        mkhw = resized_masks

        if _HAS_SEG:
            try:
                # Convert image to torch BCHW with batch size 1, values in 0..1
                img_t = torch.from_numpy(image_hwc01).permute(2, 0, 1)[None].to(torch.float32)

                # Convert masks to [B,K,H,W] float tensor (support HWK/KHW/labels)
                m = torch.from_numpy(mkhw)[None].to(torch.float32)

                # Use identity denormalization (input already 0..1)
                seg = _Segmentation(n_instances=1)
                vis = seg(image=img_t, mask=m)
                # Extract CHW tensor, convert to HWC float 0..1
                out = vis.img_tensor
                if out.dim() == 3:
                    out = out
                else:
                    # Unexpected, give up
                    return None
                if out.dtype == torch.uint8:
                    out = out.to(torch.float32) / 255.0
                out_np = out.permute(1, 2, 0).cpu().numpy()
                return np.clip(out_np, 0.0, 1.0)
            except Exception:
                # Fall back to local alpha-blend if anything goes wrong
                pass

        # Fallback: local alpha blend with colormap
        k, mh, mw = mkhw.shape
        labels = np.argmax(mkhw, axis=0)
        weights = np.max(mkhw, axis=0)
        num_classes = k
        cmap = np.asarray(_get_repo_cmap(num_classes), dtype=np.uint8)
        seg_rgb = (cmap[labels].astype(np.float32) / 255.0)
        if weights is not None:
            weights = np.clip(weights, 0.0, 1.0)[..., None]
            out = image_hwc01 * (1.0 - alpha * weights) + seg_rgb * (alpha * weights)
        else:
            out = image_hwc01 * (1.0 - alpha) + seg_rgb * alpha
        return np.clip(out, 0.0, 1.0)

    overlay_shown = False
    if orig_img is not None:
        overlay_source = None
        grandchild_resized_path = output_dir / f"{idx}.grandchild_masks_resized.npy"
        if grandchild_resized_path.exists():
            overlay_source = np.load(grandchild_resized_path)
        elif grandchild_masks_khw is not None:
            overlay_source = grandchild_masks_khw
        elif child_masks_khw is not None:
            overlay_source = child_masks_khw
        elif parent_masks_khw is not None:
            overlay_source = parent_masks_khw
        if overlay_source is not None:
            overlay_img = _overlay_on_image(orig_img, overlay_source, alpha=0.5)
            if overlay_img is not None:
                axes[4].imshow(overlay_img)
                axes[4].set_title("Overlay (Grandchild)")
                overlay_shown = True
    if not overlay_shown:
        axes[4].set_title("Overlay N/A")

    # 6. Ground-truth mask visualization (colorized)
    gt_path = None
    for candidate in (
        output_dir / f"{idx}.input.mask.npy",
        output_dir / f"{idx}.input.instance_mask.npy",
        output_dir / f"{idx}.input.segmentation_mask.npy",
    ):
        if candidate.exists():
            gt_path = candidate
            break
    if gt_path is not None:
        gt = np.load(gt_path)
        gt_vis = None
        # Accept [H,W] integer, [K,H,W], [H,W,K], [H,W,1]
        if gt.ndim == 2:
            labels = gt.astype(np.int64)
            num_classes = int(labels.max()) + 1 if labels.size > 0 else 1
            cmap = np.asarray(_get_repo_cmap(num_classes), dtype=np.uint8)
            gt_vis = (cmap[labels].astype(np.float32) / 255.0)
        elif gt.ndim == 3:
            # Squeeze singleton last channel
            if gt.shape[-1] == 1:
                labels = gt[..., 0]
                num_classes = int(labels.max()) + 1 if labels.size > 0 else 1
                cmap = np.asarray(_get_repo_cmap(num_classes), dtype=np.uint8)
                gt_vis = (cmap[labels.astype(np.int64)].astype(np.float32) / 255.0)
            else:
                # Try to interpret as one-hot either [K,H,W] or [H,W,K]
                if gt.shape[0] < 16 and gt.shape[1] > 16 and gt.shape[2] > 16:
                    # [K,H,W]
                    gt_vis = colorize_masks(gt)
                elif gt.shape[-1] < 16 and gt.shape[0] > 16 and gt.shape[1] > 16:
                    # [H,W,K]
                    gt_vis = colorize_masks(np.moveaxis(gt, -1, 0))
        if gt_vis is not None:
            axes[5].imshow(np.clip(gt_vis, 0.0, 1.0))
            axes[5].set_title("GT Mask")
        else:
            axes[5].set_title("GT Mask (Unsupported format)")
    else:
        axes[5].set_title("GT Mask N/A")

    for ax in axes:
        ax.axis("off")


def visualize_directory(
    output_dir: Path,
    save_dir: Path,
    sample_indices: Optional[Sequence[str]] = None,
) -> None:
    output_dir = Path(output_dir)
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if sample_indices is None:
        sample_indices = gather_sample_indices(output_dir)

    for idx in sample_indices:
        n_cols = 6
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        render_sample_panels(output_dir, idx, axes)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"visualization_{idx}.png"))
        plt.close(fig)
        print(f"Saved visualization for sample {idx}")

    print(f"All visualizations saved to {save_dir}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parser.parse_args(argv)
    visualize_directory(args.output_dir, args.save_dir)


if __name__ == "__main__":
    main()
