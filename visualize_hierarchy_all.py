#!/usr/bin/env python3
"""
Create tree-style visualizations for hierarchical slot masks.

Now: nodes show only colors from the original image, with masks used
as transparency (no artificial colormaps).

Example:
    python visualize_mask_hierarchy.py \
        --input_dir runs/slots_hierarchy/train \
        --save_dir visualization_results/
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import visualize  # Reuse normalization utilities etc.


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


def _original_with_mask(
    original: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray = BACKGROUND_COLOR,
) -> np.ndarray:
    """
    Show only the original image colors, using the mask as transparency.
    - background: color outside the mask (default: white)
    """
    img = np.asarray(original, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    m = np.asarray(mask, dtype=np.float32)
    m = np.clip(m, 0.0, None)
    if m.max() > 0:
        m = m / m.max()
    m = m[..., None]  # (H,W,1)

    bg = np.broadcast_to(background.reshape(1, 1, 3), img.shape).astype(np.float32)

    # Where mask is 1 -> original, where 0 -> background
    out = bg * (1.0 - m) + img * m
    return np.clip(out, 0.0, 1.0)


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
    original: Optional[np.ndarray],
) -> Tuple[List[Node], List[List[int]]]:
    k, h, w = masks.shape

    # still compute children indices for drawing edges later
    if next_masks is not None:
        children = _assign_children(k, next_masks.shape[0])
    else:
        children = [[] for _ in range(k)]

    if original is not None:
        original_resized = _resize_image(original, (h, w))
    else:
        original_resized = None

    nodes: List[Node] = []

    for idx in range(k):
        mask = masks[idx]

        # each node uses ONLY its own mask
        support = np.clip(mask, 0.0, 1.0)

        if original_resized is not None:
            image = _original_with_mask(original_resized, support)
        else:
            # fallback if no original image
            m = support
            if m.max() > 0:
                m = m / m.max()
            m = m[..., None]
            bg = np.ones((h, w, 3), dtype=np.float32)
            fg = np.zeros_like(bg)
            image = bg * (1.0 - m) + fg * m

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
            centers.append(
                (x_center, y_center - node_height / 2.0, y_center + node_height / 2.0)
            )
        centers_per_level.append(centers)

    # Draw connecting lines
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
    overlay_alpha: float,  # kept for CLI compatibility, not used for color now
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

    if original is not None:
        levels.append([Node(image=original, label="Input")])
    else:
        # If no original, show nothing at root; we will show mask-only nodes later
        pass

    # Connections from root to first mask level (if root exists)
    if mask_levels and levels:
        first_count = mask_levels[0][1].shape[0]
        connections.append([list(range(first_count))])

    # Build levels for parents / children / grandchildren
    for idx, (level_name, masks) in enumerate(mask_levels):
        next_masks = mask_levels[idx + 1][1] if idx + 1 < len(mask_levels) else None
        nodes, child_assignments = _build_level_nodes(
            level_name,
            masks,
            next_masks,
            original,
        )
        levels.append(nodes)
        if next_masks is not None:
            connections.append(child_assignments)

    if not levels:
        raise FileNotFoundError(f"No data found for sample {sample_index} in {input_dir}")

    _layout_tree(levels, connections, save_path, show, dpi)


def _find_all_sample_indices(input_dir: Path) -> List[str]:
    """
    Infer all sample stems in the directory based on the known filename patterns.
    """
    stems: Set[str] = set()
    suffixes = [
        ".parent_masks_128.npy",
        ".child_masks_128.npy",
        ".grandchild_masks_128.npy",
    ]
    for path in input_dir.glob("*.npy"):
        name = path.name
        for suffix in suffixes:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                stems.add(stem)
                break
    return sorted(stems)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize hierarchical masks as trees (original colors only)."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory with .npy outputs for samples.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory to save all visualizations (one image per sample).",
    )
    parser.add_argument(
        "--overlay_level",
        choices=["parent", "child", "grandchild"],
        default="child",
        help="(Kept for compatibility, not used for colors now).",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.55,
        help="(Kept for compatibility, not used for colors now).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure interactively (not recommended for many samples).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = _find_all_sample_indices(input_dir)
    if not sample_indices:
        raise RuntimeError(f"No sample indices found in {input_dir}")

    print(f"Found {len(sample_indices)} samples in {input_dir}")

    for idx, stem in enumerate(sample_indices):
        out_path = save_dir / f"{stem}.png"
        try:
            build_tree_visualization(
                input_dir=input_dir,
                sample_index=stem,
                save_path=out_path,
                overlay_level=args.overlay_level,
                overlay_alpha=args.overlay_alpha,
                dpi=args.dpi,
                show=args.show,
            )
            print(f"[{idx+1}/{len(sample_indices)}] Saved visualization to {out_path}")
        except FileNotFoundError as e:
            print(f"[{idx+1}/{len(sample_indices)}] Skipping {stem}: {e}")


if __name__ == "__main__":
    main()
