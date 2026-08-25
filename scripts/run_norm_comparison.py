#!/usr/bin/env python3
"""Run a local head-to-head comparison of normalization strategies.

Uses `train_config/nn_config_1500bp_nmd_merge_6_class.yaml` as the base
config, swaps in different normalization layers, trains each for a fixed
number of steps on the local GPU, and produces a summary plot.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    print(f"matplotlib not available: {exc}")
    plt = None

BASE_CONFIG = Path("train_config/nn_config_1500bp_nmd_merge_6_class.yaml")
EXPERIMENT_ROOT = Path(
    "/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/training/norm_experiments"
)
EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

# (strategy_label, target_norm_type, alpha_init)
STRATEGIES = [
    ("masked_batchnorm", "masked_batchnorm", None),
    ("masked_layernorm", "masked_layernorm", None),
    ("masked_dyt_a0.5", "masked_dyt", 0.5),
    ("masked_dyt_a0.2", "masked_dyt", 0.2),
    ("masked_dyt_a1.0", "masked_dyt", 1.0),
]


def update_norm_layers(hidden_layers, norm_type, alpha_init):
    """Mutate the representation_learner hidden_layers list in place."""
    for layer in hidden_layers:
        name = layer.get("name", "")
        if name in ("masked_batchnorm", "masked_layernorm", "masked_dyt"):
            layer["name"] = norm_type
            cfg = layer.setdefault("config", {})
            if norm_type == "masked_dyt":
                cfg["alpha_init"] = alpha_init
            else:
                cfg.pop("alpha_init", None)
        elif name == "residual_block":
            cfg = layer.setdefault("config", {})
            cfg["norm_type"] = norm_type
            if norm_type == "masked_dyt":
                cfg["alpha_init"] = alpha_init
            else:
                cfg.pop("alpha_init", None)


def run_strategy(label, norm_type, alpha_init):
    """Train one strategy and return parsed history."""
    print(f"\n{'=' * 60}")
    print(f"Strategy: {label}")
    print(f"{'=' * 60}")

    base = yaml.safe_load(BASE_CONFIG.read_text())
    base["model"]["name"] = f"jaeger_{label}"
    base["model"]["experiment"] = label
    base["model"]["base_dir"] = str(EXPERIMENT_ROOT)

    # Local GPU-friendly settings.
    base["training"]["batch_size"] = 64
    base["training"]["classifier_epochs"] = 1
    base["training"]["classifier_train_steps"] = 500
    base["training"]["classifier_validation_steps"] = 100
    base["training"]["reliability_data_generation"]["inference_batch_size"] = 64

    update_norm_layers(base["model"]["representation_learner"]["hidden_layers"], norm_type, alpha_init)

    config_path = EXPERIMENT_ROOT / f"nn_config_1500bp_nmd_merge_6_class_{label}.yaml"
    config_path.write_text(yaml.dump(base, sort_keys=False))

    log_dir = EXPERIMENT_ROOT / label
    log_dir.mkdir(exist_ok=True)
    stdout_path = log_dir / "train_stdout.log"
    stderr_path = log_dir / "train_stderr.log"

    cmd = [
        "jaeger",
        "train",
        "-c",
        str(config_path),
        "--force",
        "--save_model",
        "--mixed_precision",
        "-v",
    ]
    print("Command:", " ".join(cmd))

    start = time.time()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        result = subprocess.run(cmd, stdout=out, stderr=err)
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.1f}s with return code {result.returncode}")

    history = parse_training_log(label)
    history["label"] = label
    history["norm_type"] = norm_type
    history["alpha_init"] = alpha_init
    history["return_code"] = result.returncode
    history["elapsed_seconds"] = elapsed
    return history


def parse_training_log(label):
    """Read the CSV training log written by the train command."""
    experiment_dir = EXPERIMENT_ROOT / "experiments" / f"experiment_{label}_425425"
    log_file = experiment_dir / "checkpoints" / "classifier" / "training.log"

    rows = []
    fieldnames = [
        "epoch",
        "categorical_accuracy",
        "learning_rate",
        "loss",
        "val_categorical_accuracy",
        "val_loss",
    ]
    if log_file.exists():
        with log_file.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({k: float(row[k]) for k in fieldnames if k in row})
    else:
        print(f"Warning: no training log found at {log_file}", file=sys.stderr)

    return {
        "epochs": [r["epoch"] for r in rows],
        "loss": [r["loss"] for r in rows],
        "val_loss": [r["val_loss"] for r in rows],
        "accuracy": [r["categorical_accuracy"] for r in rows],
        "val_accuracy": [r["val_categorical_accuracy"] for r in rows],
    }


def plot_results(histories):
    """Create summary figures."""
    if plt is None:
        print("matplotlib unavailable; skipping figure generation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for h in histories:
        label = h.get("label", "unknown")
        epochs = h.get("epochs", [])
        if not epochs:
            continue
        axes[0].plot(epochs, h.get("loss", []), marker="o", label=f"{label} train")
        axes[0].plot(epochs, h.get("val_loss", []), linestyle="--", marker="x", label=f"{label} val")
        axes[1].plot(epochs, h.get("accuracy", []), marker="o", label=f"{label} train")
        axes[1].plot(epochs, h.get("val_accuracy", []), linestyle="--", marker="x", label=f"{label} val")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    figure_path = EXPERIMENT_ROOT / "norm_comparison.png"
    fig.savefig(figure_path, dpi=150)
    print(f"Saved figure: {figure_path}")


def main():
    histories = []
    for label, norm_type, alpha_init in STRATEGIES:
        try:
            h = run_strategy(label, norm_type, alpha_init)
            histories.append(h)
        except Exception as exc:
            print(f"ERROR running {label}: {exc}", file=sys.stderr)
            histories.append(
                {
                    "label": label,
                    "norm_type": norm_type,
                    "alpha_init": alpha_init,
                    "error": str(exc),
                }
            )

    summary_path = EXPERIMENT_ROOT / "norm_comparison_summary.json"
    summary_path.write_text(json.dumps(histories, indent=2))
    print(f"\nSaved summary: {summary_path}")

    plot_results(histories)

    print("\nFinal losses:")
    for h in histories:
        label = h["label"]
        if "loss" in h and h["loss"]:
            print(
                f"  {label:25s} train_loss={h['loss'][-1]:.4f} "
                f"val_loss={h['val_loss'][-1]:.4f} "
                f"train_acc={h['accuracy'][-1]:.4f} "
                f"val_acc={h['val_accuracy'][-1]:.4f}"
            )
        else:
            print(f"  {label:25s} NO RESULTS")


if __name__ == "__main__":
    main()
