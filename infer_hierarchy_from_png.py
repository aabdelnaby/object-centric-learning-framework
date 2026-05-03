#!/usr/bin/env python3
"""Run a trained model on a single PNG and visualize hierarchical masks.

This script is a lightweight convenience wrapper that:

1. Loads a model from a training config + checkpoint (same way as `ocl/cli/eval.py`).
2. Runs a single RGB image (PNG/JPEG, etc.) through the model.
3. Looks for hierarchical mask outputs named `parent_masks_128`, `child_masks_128`,
   and `grandchild_masks_128` in the model outputs.
4. Saves these masks and the original image as `.npy` files in a small output
   directory using the naming scheme expected by `visualize_mask_hierarchy.py`.
5. Calls `visualize_mask_hierarchy.build_tree_visualization` to produce a
   hierarchical tree-style PNG visualization.

Notes and assumptions
---------------------
- This script assumes your *model already produces* hierarchical masks under
  the keys `parent_masks_128`, `child_masks_128`, `grandchild_masks_128`.
  That is typically true if you trained with a hierarchical configuration or
  used an evaluation config that adds these modules into the model.
- If those keys are missing in the model outputs, the script will error out
  and tell you which keys it could not find.
- The image preprocessing uses a generic ImageNet-style pipeline
  (resize to 224, center-crop, normalize with ImageNet mean/std). This matches
  the COCO/bridging configs like `outputs_coco_ccrop_slots.yaml`. If your
  model was trained with very different preprocessing, results may degrade.

Example usage
-------------

From the repo root, something like:

    python infer_hierarchy_from_png.py \
        --train_config_path configs/experiment/projects/bridging/dinosaur/coco_feat_rec_dino_base16_auto_dinov3.yaml \
        --checkpoint_path checkpoints/epoch_67-step_500000_hierarchical_gumbel.ckpt \
        --image_path path/to/your_image.png \
        --vis_path visualization_results/your_image_hierarchy.png

Adjust config/checkpoint paths to match your setup.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import hydra
import hydra_zen
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tvT

import ocl.cli._config  # noqa: F401
from ocl.cli import train as train_cli
from visualize_mask_hierarchy import build_tree_visualization


def _build_transform(image_size: int = 224) -> tvT.Compose:
    """ImageNet-style preprocessing used by many configs.

    - Resize (BICUBIC) so smaller side = `image_size`.
    - Center crop to `image_size` x `image_size`.
    - Convert to tensor in [0,1].
    - Normalize with ImageNet statistics.
    """

    return tvT.Compose(
        [
            tvT.Resize(image_size, interpolation=tvT.InterpolationMode.BICUBIC),
            tvT.CenterCrop(image_size),
            tvT.ToTensor(),
            tvT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_and_preprocess_image(path: Path, image_size: int = 224) -> Dict[str, Any]:
    """Load PNG/JPEG from disk and create `input` dict for the model.

    Returns a dict with:
      - "image": 4D tensor [1,3,H,W] ready for the model
      - "orig_image": original HxWx3 numpy array (uint8) for visualization
    """

    if not path.exists():
        raise FileNotFoundError(f"Image path {path} does not exist.")

    img_pil = Image.open(path).convert("RGB")
    orig_np = np.array(img_pil)  # H x W x 3, uint8

    transform = _build_transform(image_size=image_size)
    img_t = transform(img_pil).unsqueeze(0)  # [1,3,H,W]

    return {"image": img_t, "orig_image": orig_np}


def _compose_train_config(train_config_path: str, train_config_overrides: Optional[Sequence[str]]):
    """Compose the training config the same way as in `ocl/cli/eval.py`."""

    abs_path = hydra.utils.to_absolute_path(train_config_path)
    if abs_path.endswith(".yaml"):
        config_dir, config_name = os.path.split(abs_path)
    else:
        config_dir, config_name = abs_path, None

    if not os.path.exists(config_dir):
        raise ValueError(f"Inferred config dir at {config_dir} does not exist.")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir):
        if config_name is None:
            raise ValueError(
                "When `train_config_path` is a directory, you must also "
                "provide `--train_config_name`."
            )
        overrides = list(train_config_overrides) if train_config_overrides else []
        train_config = hydra.compose(os.path.splitext(config_name)[0], overrides=overrides)

    return train_config


def _attach_optional_hierarchical_feature_dim(train_config: Any) -> None:
    """Best-effort fix to ensure hierarchical modules get a feature_dim.

    This mirrors the small adjustment done in `ocl/cli/eval.py` so that
    loading checkpoints with strict shape checking works even if
    `feature_dim` was left as `null` in the config.
    """

    try:
        feature_dim_val = None
        if "experiment" in train_config and "input_feature_dim" in train_config.experiment:
            feature_dim_val = train_config.experiment.input_feature_dim

        if "models" in train_config and feature_dim_val is not None:
            if "hierarchical_refine" in train_config.models:
                train_config.models.hierarchical_refine.feature_dim = feature_dim_val
            if "hierarchical_refine_l2" in train_config.models:
                train_config.models.hierarchical_refine_l2.feature_dim = feature_dim_val
    except Exception:
        # If anything goes wrong here, we silently ignore it.
        pass


def _build_model(train_config: Any, checkpoint_path: str) -> torch.nn.Module:
    checkpoint_abs = hydra.utils.to_absolute_path(checkpoint_path)
    if not os.path.exists(checkpoint_abs):
        raise FileNotFoundError(f"Checkpoint at {checkpoint_abs} does not exist.")

    _attach_optional_hierarchical_feature_dim(train_config)

    model = train_cli.build_model_from_config(train_config, checkpoint_abs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def _extract_mask(outputs: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    """Extract first-batch mask array for a given key if present.

    Returns a numpy array without batch dimension (whatever the remaining
    layout is), or `None` if the key is absent.
    """

    if key not in outputs:
        return None
    value = outputs[key]
    if not isinstance(value, torch.Tensor):
        # Some routed modules may return nested structures; we only support tensors here.
        return None
    if value.dim() == 3:
        # [K,H,W] or [H,W,K] already, just return as-is
        return value.detach().cpu().numpy()
    if value.dim() >= 4:
        # Assume [B,...]; take first element
        return value[0].detach().cpu().numpy()
    # Anything else is unexpected for masks.
    return None


def _save_npy_outputs(
    out_dir: Path,
    stem: str,
    orig_image: np.ndarray,
    parent: Optional[np.ndarray],
    child: Optional[np.ndarray],
    grandchild: Optional[np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{stem}.input.orig_image.npy", orig_image)

    if parent is not None:
        np.save(out_dir / f"{stem}.parent_masks_128.npy", parent)
    if child is not None:
        np.save(out_dir / f"{stem}.child_masks_128.npy", child)
    if grandchild is not None:
        np.save(out_dir / f"{stem}.grandchild_masks_128.npy", grandchild)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained model on a single PNG and visualize hierarchical "
            "slot masks as a tree."
        )
    )
    parser.add_argument(
        "--train_config_path",
        required=True,
        help=(
            "Path to the training config .yaml used to train the model "
            "(same as you would pass to ocl/cli/train.py)."
        ),
    )
    parser.add_argument(
        "--train_config_overrides",
        nargs="*",
        default=None,
        help=(
            "Optional Hydra-style overrides for the training config "
            "(e.g. 'experiment.some_flag=True')."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to the .ckpt file containing trained weights.",
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to an input RGB image (PNG/JPEG).",
    )
    parser.add_argument(
        "--temp_output_dir",
        default="single_image_outputs",
        help=(
            "Directory where intermediate .npy files (orig_image + masks) "
            "will be written."
        ),
    )
    parser.add_argument(
        "--vis_path",
        required=True,
        help="Path where the final hierarchical PNG visualization will be saved.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Square size used for resize/center-crop before feeding to the model.",
    )
    parser.add_argument(
        "--overlay_level",
        choices=["parent", "child", "grandchild"],
        default="child",
        help="Mask level to overlay on the root image in the tree visualization.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.55,
        help="Blend factor for overlaying masks onto the original image.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for the hierarchical visualization.",
    )
    parser.add_argument(
        "--stem",
        default="00000",
        help=(
            "Sample index/stem used to name intermediate .npy files. "
            "Only relevant if you inspect them manually."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    image_path = Path(args.image_path)
    temp_output_dir = Path(args.temp_output_dir)
    vis_path = Path(args.vis_path)

    # 1) Load training config and model
    train_config = _compose_train_config(args.train_config_path, args.train_config_overrides)
    model = _build_model(train_config, args.checkpoint_path)

    # 2) Load and preprocess the image
    inp = _load_and_preprocess_image(image_path, image_size=args.image_size)
    image_tensor = inp["image"].to(next(model.parameters()).device)

    # 3) Forward pass through the model
    with torch.no_grad():
        outputs = model({"image": image_tensor})

    # 4) Extract hierarchical masks (if present)
    parent = _extract_mask(outputs, "parent_masks_128")
    child = _extract_mask(outputs, "child_masks_128")
    grandchild = _extract_mask(outputs, "grandchild_masks_128")

    if parent is None and child is None and grandchild is None:
        raise RuntimeError(
            "Model outputs do not contain any of 'parent_masks_128', 'child_masks_128', "
            "or 'grandchild_masks_128'. Make sure you are using a hierarchical model "
            "or have added the corresponding modules in your config."
        )

    # 5) Save .npy outputs in the expected format
    _save_npy_outputs(
        out_dir=temp_output_dir,
        stem=args.stem,
        orig_image=inp["orig_image"],
        parent=parent,
        child=child,
        grandchild=grandchild,
    )

    # 6) Build hierarchical tree visualization
    build_tree_visualization(
        input_dir=temp_output_dir,
        sample_index=args.stem,
        save_path=vis_path,
        overlay_level=args.overlay_level,
        overlay_alpha=args.overlay_alpha,
        dpi=args.dpi,
        show=args.show,
    )

    print(f"Saved hierarchical visualization to {vis_path}")


if __name__ == "__main__":
    main()
