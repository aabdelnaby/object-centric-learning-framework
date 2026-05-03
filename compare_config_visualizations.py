import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt

from visualize import gather_sample_indices, render_sample_panels


def _infer_output_dir(
    config_path: Union[str, Path],
    split: str,
    outputs_subdir: str,
) -> Path:
    config_path = Path(config_path).resolve()
    run_dir = config_path.parent.parent
    candidates = [
        run_dir / outputs_subdir / split,
        run_dir / outputs_subdir,
        run_dir / split,
        run_dir,
        config_path.parent,
        run_dir.parent / outputs_subdir / split,
    ]

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_dir():
            continue
        if any(candidate.glob("*.npy")):
            return candidate
        sub_candidate = candidate / split
        if sub_candidate.is_dir() and any(sub_candidate.glob("*.npy")):
            return sub_candidate
    raise FileNotFoundError(
        f"Could not locate `{split}` outputs next to {config_path}. "
        "Use --output_dir_* arguments to specify them explicitly."
    )


def _resolve_output_dir(
    config_path: Union[str, Path],
    split: str,
    outputs_subdir: str,
    override: Optional[Union[str, Path]],
) -> Path:
    if override is not None:
        dir_path = Path(override)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Provided output directory {dir_path} does not exist.")
        return dir_path
    return _infer_output_dir(config_path, split, outputs_subdir)


def _choose_sample_index(
    outputs_a: Path,
    outputs_b: Path,
    requested: Optional[str],
) -> str:
    indices_a = set(gather_sample_indices(outputs_a))
    indices_b = set(gather_sample_indices(outputs_b))
    if requested is not None:
        if requested not in indices_a:
            raise ValueError(f"Sample {requested} not found in {outputs_a}.")
        if requested not in indices_b:
            raise ValueError(f"Sample {requested} not found in {outputs_b}.")
        return requested
    common = sorted(indices_a & indices_b)
    if not common:
        raise ValueError("No common sample indices found between the two output directories.")
    return common[0]


def render_comparison(
    output_dir_a: Path,
    output_dir_b: Path,
    sample_index: str,
    label_a: str,
    label_b: str,
    save_path: Path,
    show: bool = False,
) -> None:
    n_cols = 6
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    render_sample_panels(output_dir_a, sample_index, axes[0])
    axes[0][0].text(
        -0.08,
        0.5,
        label_a,
        transform=axes[0][0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )

    render_sample_panels(output_dir_b, sample_index, axes[1])
    axes[1][0].text(
        -0.08,
        0.5,
        label_b,
        transform=axes[1][0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )

    fig.suptitle(f"Sample {sample_index} comparison", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(left=0.12)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the same sample for two evaluation configs using visualize.py panels."
    )
    parser.add_argument("--config_a", required=True, help="Path to first evaluation config.")
    parser.add_argument("--config_b", required=True, help="Path to second evaluation config.")
    parser.add_argument(
        "--sample_index",
        help="Sample index to visualize (e.g., 00001). Defaults to first shared sample.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split to read from within each evaluation outputs directory (default: val).",
    )
    parser.add_argument(
        "--outputs_subdir",
        default="outputs",
        help="Subdirectory name that stores evaluation outputs relative to the config run dir.",
    )
    parser.add_argument(
        "--output_dir_a",
        help="Override the inferred outputs directory for config A (path to directory containing .npy files).",
    )
    parser.add_argument(
        "--output_dir_b",
        help="Override the inferred outputs directory for config B (path to directory containing .npy files).",
    )
    parser.add_argument(
        "--save_dir",
        default="visualization_results/config_comparison",
        help="Directory to store the comparison visualization.",
    )
    parser.add_argument(
        "--filename",
        help="Output filename. Defaults to comparison_<sample_index>.png inside save_dir.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure instead of closing Matplotlib after saving.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir_a = _resolve_output_dir(args.config_a, args.split, args.outputs_subdir, args.output_dir_a)
    output_dir_b = _resolve_output_dir(args.config_b, args.split, args.outputs_subdir, args.output_dir_b)

    sample_index = _choose_sample_index(output_dir_a, output_dir_b, args.sample_index)

    label_a = Path(args.config_a).name
    label_b = Path(args.config_b).name

    save_dir = Path(args.save_dir)
    filename = args.filename or f"comparison_{sample_index}.png"
    save_path = save_dir / filename

    render_comparison(
        output_dir_a,
        output_dir_b,
        sample_index,
        label_a,
        label_b,
        save_path,
        show=args.show,
    )
    print(f"Saved comparison visualization to {save_path}")


if __name__ == "__main__":
    main()
