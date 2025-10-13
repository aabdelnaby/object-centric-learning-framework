#!/usr/bin/env python3
"""
Create tree-style visualizations for hierarchical slot masks.

Example:
    python visualize_mask_hierarchy.py \
        --input_dir slots_hierarchy/train \
        --index 00 \
        --save_path visualization_results/demo_tree.png
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import visualize  # Reuse normalization utilities and colormaps


BACKGROUND_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)


@dataclass
class Node:
    image: np.ndarray
    label: str


def _load_optional_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    data = np.load(path)
    return visualize._to_khw(data)


def _load_original_image(sample_dir: Path, stem: str) -> Optional[np.ndarray]:
    candidates = [
        sample_dir / f"{stem}.input.orig_image.npy",
        sample_dir / f"{stem}.input.image.npy",
        sample_dir / f"{stem}.input.rgb.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            arr = np.load(candidate)
            return visualize._ensure_hwc01(arr)
    return None


def _resize_masks(masks: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if masks.shape[1:] == (h, w):
        return masks
    tensor = torch.from_numpy(masks.astype(np.float32))[None]
    resized = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
    return resized[0].cpu().numpy()


def _resize_image(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if image.shape[0] == h and image.shape[1] == w:
        return image
    tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None]
    resized = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).cpu().numpy()


def _colorize_with_background(
    masks: np.ndarray,
    colors: Optional[np.ndarray] = None,
    background: np.ndarray = BACKGROUND_COLOR,
) -> np.ndarray:
    arr = np.asarray(masks, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected masks with 2 or 3 dims, got {arr.shape}")

    k, h, w = arr.shape
    if k == 0:
        return np.tile(background.reshape(1, 1, 3), (h, w, 1))

    if colors is None or len(colors) < k:
        cmap = visualize._get_repo_cmap(max(k, 1))
        colors = np.asarray(cmap[:k], dtype=np.float32) / 255.0
    else:
        colors = np.asarray(colors[:k], dtype=np.float32)

    arr = np.clip(arr, 0.0, None)
    max_val = np.max(arr)
    if max_val > 0.0:
        arr = arr / max_val

    stack = np.concatenate([np.zeros((1, h, w), dtype=np.float32), arr], axis=0)
    labels = np.argmax(stack, axis=0)
    strengths = np.max(stack, axis=0)[..., None]

    palette = np.vstack([background.reshape(1, 3), colors.reshape(-1, 3)])
    out = palette[labels]
    blended = background + (out - background) * strengths
    return np.clip(blended, 0.0, 1.0)


def _mask_to_color(mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.float32)
    if color.max() > 1.0:
        color = color / 255.0
    return _colorize_with_background(mask, colors=color[None, ...])


def _overlay_image(image: np.ndarray, masks: np.ndarray, alpha: float) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.0:
        img = np.clip(img / 255.0, 0.0, 1.0)

    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim == 2:
        masks = masks[None, ...]

    resized = _resize_masks(masks, img.shape[:2])
    seg = _colorize_with_background(resized)
    return np.clip((1.0 - alpha) * img + alpha * seg, 0.0, 1.0)


def _assign_children(num_parents: int, num_children: int) -> List[List[int]]:
    if num_parents <= 0 or num_children <= 0:
        return [[] for _ in range(max(num_parents, 0))]

    base = num_children // num_parents
    remainder = num_children % num_parents
    assignments: List[List[int]] = []
    start = 0
    for i in range(num_parents):
        extra = 1 if i < remainder else 0
        end = start + base + extra
        assignments.append(list(range(start, min(end, num_children))))
        start = end
    return assignments


def _build_level_nodes(
    level_name: str,
    masks: np.ndarray,
    next_masks: Optional[np.ndarray],
    level_palette: np.ndarray,
    original: Optional[np.ndarray],
) -> Tuple[List[Node], List[List[int]]]:
    k, h, w = masks.shape
    children: List[List[int]]
    if next_masks is not None:
        children = _assign_children(k, next_masks.shape[0])
    else:
        children = [[] for _ in range(k)]

    nodes: List[Node] = []
    original_resized: Optional[np.ndarray] = None
    if original is not None:
        original_resized = _resize_image(original, (h, w))

    for idx in range(k):
        mask = masks[idx]
        child_indices = children[idx]
        if next_masks is not None and child_indices:
            subset = next_masks[child_indices]
            subset = subset * mask[None, ...]
            palette = (
                np.asarray(
                    visualize._get_repo_cmap(max(len(child_indices), 1)),
                    dtype=np.float32,
                )
                / 255.0
            )[: len(child_indices)]
            image = _colorize_with_background(subset, colors=palette)
            support = np.clip(subset.sum(axis=0), 0.0, 1.0)
        else:
            color = level_palette[idx % len(level_palette)]
            image = _mask_to_color(mask, color)
            support = np.clip(mask, 0.0, 1.0)

        if original_resized is not None:
            support_rgb = support[..., None]
            image = image * (1.0 - support_rgb) + original_resized * support_rgb

        label = f"{level_name.capitalize()} {idx}"
        nodes.append(Node(image=image, label=label))
    return nodes, children


def _layout_tree(
    levels: Sequence[Sequence[Node]],
    connections: Sequence[Sequence[Sequence[int]]],
    save_path: Optional[Path],
    show: bool,
    dpi: int,
) -> None:
    if not levels:
        raise ValueError("No nodes to plot.")

    num_levels = len(levels)
    max_nodes = max(len(level) for level in levels)
    fig_width = max(6.0, 1.8 * max_nodes)
    fig_height = max(4.0, 2.0 * num_levels)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    left_margin = 0.06
    right_margin = 0.06
    top_margin = 0.08
    bottom_margin = 0.08
    vertical_gap = 0.05

    available_height = 1.0 - top_margin - bottom_margin
    node_height = (available_height - vertical_gap * (num_levels - 1)) / num_levels
    node_height = max(min(node_height, 0.22), 0.12)

    available_width = 1.0 - left_margin - right_margin

    centers_per_level: List[List[Tuple[float, float, float]]] = []

    for level_idx, nodes in enumerate(levels):
        count = len(nodes)
        if count == 0:
            centers_per_level.append([])
            continue
        node_width = min(0.25, (available_width / max(count, 1)) * 0.8)
        y_center = 1.0 - top_margin - node_height / 2.0 - level_idx * (
            node_height + vertical_gap
        )

        centers: List[Tuple[float, float, float]] = []
        for item_idx, node in enumerate(nodes):
            x_center = left_margin + available_width * ((item_idx + 0.5) / count)
            ax = fig.add_axes(
                [
                    x_center - node_width / 2.0,
                    y_center - node_height / 2.0,
                    node_width,
                    node_height,
                ]
            )
            ax.imshow(node.image)
            ax.set_title(node.label, fontsize=8)
            ax.axis("off")
            centers.append((x_center, y_center - node_height / 2.0, y_center + node_height / 2.0))
        centers_per_level.append(centers)

    for level_idx, child_lists in enumerate(connections):
        if level_idx + 1 >= len(centers_per_level):
            break
        parents = centers_per_level[level_idx]
        children = centers_per_level[level_idx + 1]
        if not parents or not children:
            continue
        for parent_idx, child_indices in enumerate(child_lists):
            if parent_idx >= len(parents):
                continue
            px, p_bottom, _ = parents[parent_idx]
            for child_idx in child_indices:
                if child_idx >= len(children):
                    continue
                cx, _, c_top = children[child_idx]
                line = plt.Line2D(
                    [px, cx],
                    [p_bottom, c_top],
                    color="black",
                    linewidth=1.0,
                    alpha=0.6,
                    transform=fig.transFigure,
                )
                fig.add_artist(line)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_tree_visualization(
    input_dir: Path,
    sample_index: str,
    save_path: Optional[Path],
    overlay_level: str,
    overlay_alpha: float,
    dpi: int,
    show: bool,
) -> None:
    stem = sample_index
    original = _load_original_image(input_dir, stem)

    level_candidates = [
        ("parent", f"{stem}.parent_masks_128.npy"),
        ("child", f"{stem}.child_masks_128.npy"),
        ("grandchild", f"{stem}.grandchild_masks_128.npy"),
    ]

    mask_levels: List[Tuple[str, np.ndarray]] = []
    for name, filename in level_candidates:
        arr = _load_optional_mask(input_dir / filename)
        if arr is not None and arr.size > 0:
            mask_levels.append((name, arr))

    levels: List[List[Node]] = []
    connections: List[List[List[int]]] = []

    overlay_source: Optional[np.ndarray] = None
    available = {name: masks for name, masks in mask_levels}
    for name in ("grandchild", "child", "parent"):
        if overlay_level == name and name in available:
            overlay_source = available[name]
            break
    if overlay_source is None:
        for name in ("child", "parent", "grandchild"):
            if name in available:
                overlay_source = available[name]
                break

    if original is not None and overlay_source is not None:
        overlay_image = _overlay_image(original, overlay_source, overlay_alpha)
        root_label = "Input + masks"
        levels.append([Node(image=overlay_image, label=root_label)])
    elif original is not None:
        levels.append([Node(image=original, label="Input")])
    elif overlay_source is not None:
        palette = np.asarray(visualize._get_repo_cmap(overlay_source.shape[0]), dtype=np.float32) / 255.0
        image = _colorize_with_background(overlay_source, colors=palette)
        levels.append([Node(image=image, label="Masks")])
    else:
        raise FileNotFoundError(f"No data found for sample {sample_index} in {input_dir}")

    if mask_levels:
        first_count = mask_levels[0][1].shape[0]
        connections.append([list(range(first_count))])

    for idx, (level_name, masks) in enumerate(mask_levels):
        next_masks = mask_levels[idx + 1][1] if idx + 1 < len(mask_levels) else None
        palette = np.asarray(
            visualize._get_repo_cmap(max(masks.shape[0], 1)),
            dtype=np.float32,
        ) / 255.0
        nodes, child_assignments = _build_level_nodes(
            level_name,
            masks,
            next_masks,
            palette,
            original,
        )
        levels.append(nodes)
        if next_masks is not None:
            connections.append(child_assignments)

    _layout_tree(levels, connections, save_path, show, dpi)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize hierarchical masks as a tree.")
    parser.add_argument("--input_dir", required=True, help="Directory with .npy outputs for a sample.")
    parser.add_argument("--index", required=True, help="Sample index/stem (e.g., 00001 or 05).")
    parser.add_argument("--save_path", help="Optional path to save the visualization.")
    parser.add_argument(
        "--overlay_level",
        choices=["parent", "child", "grandchild"],
        default="child",
        help="Mask level to overlay on the root image if available.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.55,
        help="Blend factor for overlaying masks onto the original image.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    save_path = Path(args.save_path) if args.save_path else None
    build_tree_visualization(
        input_dir=input_dir,
        sample_index=args.index,
        save_path=save_path,
        overlay_level=args.overlay_level,
        overlay_alpha=args.overlay_alpha,
        dpi=args.dpi,
        show=args.show,
    )
    if save_path:
        print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    main()
