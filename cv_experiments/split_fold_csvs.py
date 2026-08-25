"""Split the pooled train/val CSVs into per-fold CSVs.

Streams ``train.csv`` and ``val.csv`` and writes every row to
``fold{k}.csv`` according to the ``row_folds_{train,val}.npy`` arrays (CSV
row order, produced by ``make_fold_assignments.py``; 255 = dropped).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/data_generation/frag_2000",
    )
    ap.add_argument("--folds-dir", default="cv_experiments/folds")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    data_dir, folds_dir = Path(args.data_dir), Path(args.folds_dir)

    handles = [
        open(folds_dir / f"fold{k}.csv", "w", buffering=1024 * 1024)
        for k in range(args.n_folds)
    ]
    counts = np.zeros(args.n_folds + 1, dtype=np.int64)
    try:
        for split in ("train", "val"):
            rf = np.load(folds_dir / f"row_folds_{split}.npy")
            src = data_dir / f"{split}.csv"
            n = 0
            with open(src, buffering=1024 * 1024) as f:
                for line in f:
                    fold = int(rf[n])
                    n += 1
                    counts[fold if fold < args.n_folds else args.n_folds] += 1
                    if fold < args.n_folds:
                        handles[fold].write(line)
            if n != len(rf):
                raise SystemExit(
                    f"ERROR: {src} has {n} rows but row_folds has {len(rf)}"
                )
            print(f"{split}: wrote {n} rows (running totals: {counts})")
    finally:
        for h in handles:
            h.close()
    print(f"rows per fold: {counts[: args.n_folds]}, dropped: {counts[args.n_folds]}")


if __name__ == "__main__":
    main()
