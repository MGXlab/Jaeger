"""Evaluate a trained Jaeger reliability head at an arbitrary validation crop length.

Loads a saved fragment SavedModel (which exposes a ``reliability`` output) and
scores a reliability validation NPZ with ``validation_crop_sizes`` overridden to
the requested codon count. Reports exact AUROC/AUPRC, calibration (ECE, Brier,
top-bin purity), and both the standard (0-0.95, step 0.05) and a fine
(0.9-0.9995, step 0.0005) threshold sweep, so saturation effects near 1.0 are
visible.

Example:
    python3 scripts/eval_reliability_crop_length.py \
        --config train_config/nn_config.yaml \
        --graph /path/jaeger_XXXX_fragment_graph \
        --val-npz /path/reliability_val_data_5000bp_translated.npz \
        --crop-codons 665 --batch-size 256 \
        --out results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def collect_scores(graph_path: str, dataset, input_type: str):
    """Run the saved graph over the dataset once; return (scores, labels)."""
    import tensorflow as tf

    from jaeger.postprocess.threshold import _to_label_vector, _to_probabilities

    fn = tf.saved_model.load(graph_path).signatures["serving_default"]
    output_names = list(fn.structured_outputs.keys())
    if "reliability" not in output_names:
        raise ValueError(f"Graph {graph_path} has no reliability output: {output_names}")
    # The saved graph's input layer is float32 while the loader yields integer
    # codon ids; the Keras data adapter casts implicitly during training, so we
    # cast explicitly here (integer values are preserved exactly).
    input_spec = fn.structured_input_signature[1]["inputs"]

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in dataset:
        x_in = x[input_type]
        if x_in.dtype != input_spec.dtype:
            x_in = tf.cast(x_in, input_spec.dtype)
        out = fn(inputs=x_in)
        rel = np.asarray(out["reliability"], dtype=np.float32)
        rel = rel.reshape(rel.shape[0], -1)
        if rel.shape[1] != 1:
            raise ValueError(f"reliability output width {rel.shape[1]} != 1")
        scores.append(rel[:, 0])
        labels.append(_to_label_vector(y))
    scores = _to_probabilities(np.concatenate(scores))
    labels = np.concatenate(labels)
    return scores, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training YAML (for string_processor)")
    parser.add_argument("--graph", required=True, help="saved fragment graph dir")
    parser.add_argument("--val-npz", required=True, help="reliability validation NPZ")
    parser.add_argument("--crop-codons", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default=None, help="write JSON results here")
    args = parser.parse_args()

    from jaeger.commands.train import _build_numpy_split
    from jaeger.postprocess.threshold import calibration_summary, tune_reliability_threshold
    from jaeger.utils.misc import load_model_config

    config = load_model_config(Path(args.config))
    # Mirror DynamicModelBuilder._get_string_processor_config: the embedding
    # section (which holds input_type) is merged under string_processor.
    sp = dict(config["model"].get("embedding", {}))
    sp.update(config["model"].get("string_processor", {}))
    sp["validation_crop_sizes"] = [args.crop_codons]
    input_type = sp.get("input_type")

    ds = _build_numpy_split(
        args.val_npz,
        num_classes=2,
        string_processor_config=sp,
        batching_cfg=sp.get("batching", {}),
        batch_size=args.batch_size,
        multi_gpu=False,
        num_replicas=1,
        buffer_size=sp.get("buffer_size", 50000),
        split="validation",
        shuffle=False,
    )

    scores, labels = collect_scores(args.graph, ds, input_type)

    result = {
        "graph": args.graph,
        "val_npz": args.val_npz,
        "crop_codons": args.crop_codons,
        "n_samples": int(labels.size),
        "n_id": int((labels == 1).sum()),
        "n_ood": int((labels == 0).sum()),
    }

    for name, (lo, hi, step) in {
        "coarse": (0.0, 0.95, 0.05),
        "fine": (0.9, 0.9995, 0.0005),
    }.items():
        best, _rows, summary = tune_reliability_threshold(
            scores, labels, metric="f1-id",
            min_threshold=lo, max_threshold=hi, step=step,
        )
        result[name] = {
            "best_threshold": float(best),
            "auroc": float(summary["auroc"]),
            "auprc": float(summary["auprc"]),
        }

    ece, brier, cal_rows = calibration_summary(scores, labels)
    result["ece"] = float(ece)
    result["brier"] = float(brier)
    top = cal_rows[-1]
    result["top_bin"] = {
        "bin_center": float(top["bin_center"]),
        "mean_pred": float(top["mean_pred"]),
        "empirical_id_rate": float(top["empirical_id_rate"]),
        "count": int(top["count"]),
        "count_frac": float(top["count"] / labels.size),
    }

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
