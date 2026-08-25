#!/usr/bin/env python3
"""Plot classifier validation training histories for the nuc vs trans experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(
    "/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/Jaeger/nuc_vs_trans_experiments/experiments"
)
OUTPUT_DIR = Path(
    "/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/Jaeger/nuc_vs_trans_experiments/figures"
)
OUTPUT_PATH = OUTPUT_DIR / "classifier_comparison.png"

ARCHITECTURES = ["conv", "conv_attention", "conv_bilstm", "conv_hyena"]
INPUT_TYPES = ["trans", "nuc"]

VAL_METRICS = [
    ("val_loss", "Validation Loss", "loss"),
    ("val_categorical_accuracy", "Validation Categorical Accuracy", "accuracy"),
    ("val_macro_f1", "Validation Macro F1", "F1 score"),
]


def parse_experiment_name(name: str) -> dict[str, str] | None:
    """Parse an experiment directory name into architecture, input_type, and seed."""
    if not name.startswith("experiment_"):
        return None
    parts = name.split("_")
    # Drop the leading "experiment" token.
    parts = parts[1:]
    if len(parts) < 3:
        return None
    seed = parts[-1]
    input_type = parts[-2]
    architecture = "_".join(parts[:-2])
    if architecture not in ARCHITECTURES:
        return None
    if input_type not in INPUT_TYPES:
        return None
    if not seed.isdigit():
        return None
    return {
        "architecture": architecture,
        "input_type": input_type,
        "seed": seed,
    }


def discover_experiments(base_dir: Path) -> list[tuple[Path, dict[str, str]]]:
    """Find all experiment directories and parse their names."""
    experiments: list[tuple[Path, dict[str, str]]] = []
    if not base_dir.is_dir():
        print(f"Base directory does not exist: {base_dir}", file=sys.stderr)
        return experiments
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        parsed = parse_experiment_name(path.name)
        if parsed is not None:
            experiments.append((path, parsed))
    return experiments


def build_color_map(architectures: list[str]) -> dict[str, tuple]:
    """Assign a distinct tab10 color to each architecture."""
    cmap = matplotlib.colormaps["tab10"]
    indices = np.linspace(0, 1, len(architectures), endpoint=False)
    return {arch: cmap(i) for arch, i in zip(architectures, indices)}


def plot_histories(experiments: list[tuple[Path, dict[str, str]]]) -> None:
    """Create and save the classifier comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=False, sharey=False)

    architectures = sorted({meta["architecture"] for _, meta in experiments})
    color_map = build_color_map(architectures)
    linestyle_map = {"trans": "-", "nuc": "--"}

    # Track legend entries to avoid duplicate labels.
    legend_handles: dict[str, matplotlib.lines.Line2D] = {}

    for exp_path, meta in experiments:
        architecture = meta["architecture"]
        input_type = meta["input_type"]
        history_path = exp_path / "checkpoints" / "classifier" / "training_history.csv"

        if not history_path.is_file():
            print(f"Skipping {exp_path.name}: missing {history_path}")
            continue

        try:
            df = pd.read_csv(history_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {exp_path.name}: could not read history ({exc})")
            continue

        if "epoch" not in df.columns:
            print(f"Skipping {exp_path.name}: no 'epoch' column in history")
            continue

        color = color_map[architecture]
        linestyle = linestyle_map[input_type]
        label = f"{architecture} {input_type}"

        for ax, (col, title, ylabel) in zip(axes, VAL_METRICS):
            if col not in df.columns:
                print(
                    f"Warning: {exp_path.name} is missing column '{col}'; "
                    f"skipping on '{title}' subplot."
                )
                continue
            (line,) = ax.plot(
                df["epoch"] + 1,
                df[col],
                color=color,
                linestyle=linestyle,
                label=label,
                linewidth=1.5,
                marker="o",
                markersize=4,
            )
            # Store one handle per unique label.
            legend_handles.setdefault(label, line)

    max_epoch = 1
    for exp_path, meta in experiments:
        history_path = exp_path / "checkpoints" / "classifier" / "training_history.csv"
        if not history_path.is_file():
            continue
        try:
            df = pd.read_csv(history_path)
        except Exception:  # noqa: BLE001
            continue
        if "epoch" in df.columns and not df["epoch"].empty:
            max_epoch = max(max_epoch, int(df["epoch"].max()) + 1)

    for ax, (_, title, ylabel) in zip(axes, VAL_METRICS):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(range(1, max_epoch + 1))
        ax.set_xlim(0.5, max_epoch + 0.5)

    if not legend_handles:
        print("No data to plot; skipping figure save.")
        plt.close(fig)
        return

    # Sort legend entries for readability.
    sorted_labels = sorted(legend_handles.keys())
    sorted_handles = [legend_handles[label] for label in sorted_labels]
    legend = fig.legend(
        sorted_handles,
        sorted_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_extra_artists=[legend],
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PATH}")


def main() -> None:
    """Entry point."""
    experiments = discover_experiments(BASE_DIR)
    print(
        f"Found {len(experiments)} experiment directories matching the expected naming."
    )

    if not experiments:
        print("No experiments to plot.")
        return

    # Report discovered experiments.
    for exp_path, meta in experiments:
        print(
            f"  - {exp_path.name}: "
            f"architecture={meta['architecture']}, "
            f"input_type={meta['input_type']}, "
            f"seed={meta['seed']}"
        )

    plot_histories(experiments)


if __name__ == "__main__":
    main()
