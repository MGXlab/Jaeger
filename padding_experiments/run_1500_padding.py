#!/usr/bin/env python
"""Accuracy of 1500 bp fragments with 0/100/200/500 nt of M-padding.

Unlike run_padding_experiment.py (which always pads to 2000 nt), here the
total length varies: a 1500 bp prefix is scored native (no padding) and with
100, 200, 500 nt of 'M' padding (total 1600/1700/2000 nt). Fragments are
grouped by total length so each batch is uniform — no implicit padding.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyfastx
import tensorflow as tf

from run_padding_experiment import (
    load_model,
    load_model_from_dir,
    run_inference,
    safe_divide,
)

REAL_LEN = 1500
PAD_AMOUNTS = [0, 100, 200, 500]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--model-dir", default=None)
    p.add_argument("-m", "--model", default="jaeger_f47f36db_1.2M_fragment")
    p.add_argument("-n", "--num-contigs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", required=True)
    return p.parse_args()


def build_records(fasta_path, num_contigs, seed):
    fa = pyfastx.Fasta(fasta_path)
    eligible = [(r.name, r.seq.upper()) for r in fa if len(r.seq) >= REAL_LEN]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligible), size=min(num_contigs, len(eligible)), replace=False)
    records = []
    for i in sorted(idx):
        name, seq = eligible[i]
        contig_id = name.split()[0]
        real = seq[:REAL_LEN]
        g, c, a, t = (real.count(x) for x in "GCAT")
        gc_skew = safe_divide(g - c, g + c)
        for pad in PAD_AMOUNTS:
            padded = real + "M" * pad
            frag_id = f"{contig_id}__pad{pad}"
            records.append(
                {
                    "csv_line": (
                        f"{padded},{frag_id},0,1,0,{REAL_LEN},{g},{c},{a},{t},"
                        f"{gc_skew: .3f}"
                    ),
                    "contig_id": contig_id,
                    "pad_len": pad,
                    "total_len": REAL_LEN + pad,
                }
            )
    return records


def main():
    args = parse_args()
    model = (
        load_model_from_dir(args.model_dir)
        if args.model_dir
        else load_model(args.model)
    )
    class_map = model.class_map
    classes = [
        c
        for _, c in sorted(
            zip(class_map["index"], class_map["class"]), key=lambda t: int(t[0])
        )
    ]

    records = build_records(args.input, args.num_contigs, args.seed)
    print(f"scoring {len(records)} fragments ({len(PAD_AMOUNTS)} pad conditions)")

    # group by total length: uniform shape per batch, no implicit padding
    results = {}
    for total_len in sorted({r["total_len"] for r in records}):
        group = [r for r in records if r["total_len"] == total_len]
        y = run_inference(model, group, batch=96)
        probs = tf.nn.softmax(y["prediction"], axis=-1).numpy()
        rel = tf.nn.sigmoid(y["reliability"]).numpy().squeeze()
        for i, rec in enumerate(group):
            results[rec["csv_line"].split(",")[1]] = (
                classes[int(np.argmax(probs[i]))],
                float(rel[i]),
            )

    out_path = Path(args.output)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["contig_id", "pad_len", "pred_class", "reliability"])
        for rec in records:
            pred, rel = results[rec["csv_line"].split(",")[1]]
            writer.writerow([rec["contig_id"], rec["pad_len"], pred, f"{rel:.6f}"])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
