#!/usr/bin/env python3
"""Convenience script: from an image .npy path to a hierarchical tree image.

This wraps `visualize_mask_hierarchy.build_tree_visualization` so you can
pass a single saved input image (from evaluation outputs) and get the
corresponding hierarchical visualization.

Expected input image path examples (produced by eval configs like
`outputs_coco_ccrop_slots_dinov3.yaml`):

    outputs_dir/00001.input.orig_image.npy
    outputs_dir/00001.input.image.npy
    outputs_dir/00001.input.rgb.npy

The script infers:
- `input_dir` = directory of the file (e.g. `outputs_dir`)
- sample index = part before `.input.` (e.g. `00001`)

It then calls `visualize_mask_hierarchy.build_tree_visualization`, which
expects the corresponding mask files in the same directory, e.g.:

    00001.parent_masks_128.npy
    00001.child_masks_128.npy
    00001.grandchild_masks_128.npy

Usage example:

    python visualize_hierarchy_from_image.py \
        --image_path runs/slots_hierarchy/train/00001.input.orig_image.npy \
        --save_path visualization_results/00001_tree.png
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from visualize_mask_hierarchy import build_tree_visualization


def _infer_sample_index_and_dir(image_path: Path) -> tuple[Path, str]:
    """Infer output directory and sample index from an input .npy path.

    Expects filenames like `<stem>.input.orig_image.npy`, `<stem>.input.image.npy`,
    or `<stem>.input.rgb.npy`. Returns `(directory, stem)`.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image path {image_path} does not exist.")

    name = image_path.name
    # We rely on the `.input.` separator used by eval outputs.
    marker = ".input."
    if marker not in name:
        raise ValueError(
            f"Expected filename containing '{marker}', got '{name}'. "
            "Example: '00001.input.orig_image.npy'."
        )

    stem = name.split(marker, 1)[0]
    return image_path.parent, stem


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Given an eval-saved input image .npy, render its hierarchical "
            "slot mask tree visualization."
        )
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help=(
            "Path to an input image .npy from eval outputs (e.g. "
            "'.../00001.input.orig_image.npy')."
        ),
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Where to save the resulting PNG tree visualization.",
    )
    parser.add_argument(
        "--overlay_level",
        choices=["parent", "child", "grandchild"],
        default="child",
        help=(
            "Mask level to overlay on the root image if available "
            "(passed through to visualize_mask_hierarchy)."
        ),
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.55,
        help="Blend factor for overlaying masks onto the original image.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    image_path = Path(args.image_path)
    save_path = Path(args.save_path)

    input_dir, sample_index = _infer_sample_index_and_dir(image_path)

    build_tree_visualization(
        input_dir=input_dir,
        sample_index=sample_index,
        save_path=save_path,
        overlay_level=args.overlay_level,
        overlay_alpha=args.overlay_alpha,
        dpi=args.dpi,
        show=args.show,
    )

    print(f"Saved hierarchical visualization for '{sample_index}' to {save_path}")


if __name__ == "__main__":
    main()
