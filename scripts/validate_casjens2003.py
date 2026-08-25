"""Validate Jaeger prophage detection against Casjens 2003 reference set.

Compares Jaeger's prophage predictions with the reference coordinates from
Casjens 2003 (as used in the PHASTEST evaluation) and computes accuracy metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_reference_coordinates(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Load reference prophage coordinates from TSV."""
    df = pd.read_csv(path, sep="\t")
    ref = {}
    for _, row in df.iterrows():
        acc = str(row["accession"])
        if acc not in ref:
            ref[acc] = []
        ref[acc].append((int(row["start"]), int(row["end"])))
    return ref


def load_jaeger_predictions(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Load Jaeger prophage predictions from TSV."""
    df = pd.read_csv(path, sep="\t")
    pred = {}
    for _, row in df.iterrows():
        acc = str(row["contig_id"]).replace("___", ",")
        if acc not in pred:
            pred[acc] = []
        # Use refined boundaries if available, otherwise raw
        start_val = row.get("sstart") or row.get("raw_start") or 0
        end_val = row.get("eend") or row.get("raw_end") or 0
        start = int(start_val) if start_val else 0
        end = int(end_val) if end_val else 0
        if start > 0 and end > 0:
            pred[acc].append((start, end))
    return pred


def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Check if two intervals overlap."""
    return a[0] < b[1] and b[0] < a[1]


def compute_metrics(
    ref: dict[str, list[tuple[int, int]]],
    pred: dict[str, list[tuple[int, int]]],
) -> dict:
    """Compute sensitivity, PPV, and boundary accuracy."""
    total_ref = sum(len(v) for v in ref.values())
    total_pred = sum(len(v) for v in pred.values())

    # Count true positives (predictions that overlap with reference)
    tp = 0
    for acc, pred_coords in pred.items():
        ref_coords = ref.get(acc, [])
        for p in pred_coords:
            if any(overlap(p, r) for r in ref_coords):
                tp += 1

    # Count found reference prophages (reference that overlap with predictions)
    found_ref = 0
    for acc, ref_coords in ref.items():
        pred_coords = pred.get(acc, [])
        for r in ref_coords:
            if any(overlap(r, p) for p in pred_coords):
                found_ref += 1

    sensitivity = found_ref / total_ref if total_ref > 0 else 0
    ppv = tp / total_pred if total_pred > 0 else 0

    # Boundary accuracy for true positives
    boundary_errors = []
    for acc, pred_coords in pred.items():
        ref_coords = ref.get(acc, [])
        for p in pred_coords:
            for r in ref_coords:
                if overlap(p, r):
                    boundary_errors.append(abs(p[0] - r[0]) + abs(p[1] - r[1]))

    mean_boundary_error = (
        sum(boundary_errors) / len(boundary_errors) if boundary_errors else 0
    )

    return {
        "total_reference": total_ref,
        "total_predicted": total_pred,
        "true_positives": tp,
        "found_reference": found_ref,
        "sensitivity": sensitivity,
        "ppv": ppv,
        "mean_boundary_error": mean_boundary_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Jaeger prophage predictions against Casjens 2003"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("casjens2003_prophage_coordinates.tsv"),
        help="Reference prophage coordinates TSV",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Jaeger prophage predictions TSV (prophages_jaeger.tsv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_results.tsv"),
        help="Output TSV for validation results",
    )
    args = parser.parse_args()

    ref = load_reference_coordinates(args.reference)
    pred = load_jaeger_predictions(args.predictions)

    metrics = compute_metrics(ref, pred)

    # Write results
    with open(args.output, "w") as fh:
        fh.write("metric\tvalue\n")
        for k, v in metrics.items():
            fh.write(f"{k}\t{v}\n")

    print("Validation results:")
    print(f"  Reference prophages: {metrics['total_reference']}")
    print(f"  Predicted prophages: {metrics['total_predicted']}")
    print(f"  True positives: {metrics['true_positives']}")
    print(f"  Found reference: {metrics['found_reference']}")
    print(f"  Sensitivity: {metrics['sensitivity']:.2%}")
    print(f"  PPV: {metrics['ppv']:.2%}")
    print(f"  Mean boundary error: {metrics['mean_boundary_error']:.0f} bp")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
