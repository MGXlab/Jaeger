#!/usr/bin/env python
"""Padding experiment on real short contigs (1000-1999 bp), all virus/phage.

Unlike run_padding_experiment.py (which truncates 2000+ bp contigs), this
scores genuinely short sequences. Every contig is a virus or phage, so the
ground-truth class family is known and accuracy can be measured directly.

For each contig of length L we score nested prefixes L, L-50, ..., 1000,
each right-padded with 'M' to 2000 nt. The full-length prefix (minimal
padding, is_reference=1) is the per-contig reference prediction. Padding
with 'N' is skipped: the previous experiment showed M and N produce
identical model inputs (max prob diff ~2e-4).

Outputs: results_short.csv with one row per fragment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyfastx
import tensorflow as tf

from run_padding_experiment import FRAGSIZE, load_model, run_inference, safe_divide

MIN_REAL_LEN = 1000
STEP = 50


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="input FASTA")
    p.add_argument("-m", "--model", default="jaeger_f47f36db_1.2M_fragment")
    p.add_argument("-o", "--output", default="results_short.csv")
    p.add_argument("--batch", type=int, default=96)
    return p.parse_args()


def build_records(fasta_path: str):
    fa = pyfastx.Fasta(fasta_path)
    records = []
    for rec in fa:
        seq = rec.seq.upper()
        seqlen = len(seq)
        if not (MIN_REAL_LEN <= seqlen < FRAGSIZE):
            continue
        contig_id = rec.name.split()[0]
        for real_len in range(seqlen, MIN_REAL_LEN - 1, -STEP):
            pad_len = FRAGSIZE - real_len
            real_seq = seq[:real_len]
            padded = real_seq + "M" * pad_len
            g = real_seq.count("G")
            c = real_seq.count("C")
            a = real_seq.count("A")
            t = real_seq.count("T")
            gc_skew = safe_divide(g - c, g + c)
            frag_id = f"{contig_id}__M__{pad_len}"
            csv_line = (
                f"{padded},{frag_id},0,1,0,{real_len},{g},{c},{a},{t},"
                f"{gc_skew: .3f}"
            )
            records.append(
                {
                    "csv_line": csv_line,
                    "contig_id": contig_id,
                    "frag_id": frag_id,
                    "pad_len": pad_len,
                    "real_len": real_len,
                    "is_reference": int(real_len == seqlen),
                }
            )
    return records


def main():
    args = parse_args()
    model = load_model(args.model)
    class_map = model.class_map
    classes = [
        c
        for _, c in sorted(
            zip(class_map["index"], class_map["class"]), key=lambda t: int(t[0])
        )
    ]
    print(f"model classes: {classes}")

    records = build_records(args.input)
    n_contigs = len({r["contig_id"] for r in records})
    print(f"scoring {len(records)} fragments from {n_contigs} short contigs")

    y_pred = run_inference(model, records, args.batch)

    probs = tf.nn.softmax(y_pred["prediction"], axis=-1).numpy()
    reliability = None
    if "reliability" in y_pred:
        reliability = tf.nn.sigmoid(y_pred["reliability"]).numpy().squeeze()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "contig_id", "frag_id", "pad_len", "real_len", "is_reference",
                "pred_class", "reliability",
            ]
            + [f"prob_{c}" for c in classes]
        )
        for i, rec in enumerate(records):
            pred_class = classes[int(np.argmax(probs[i]))]
            rel = float(reliability[i]) if reliability is not None else ""
            writer.writerow(
                [
                    rec["contig_id"], rec["frag_id"], rec["pad_len"],
                    rec["real_len"], rec["is_reference"], pred_class, rel,
                ]
                + [f"{p:.6f}" for p in probs[i]]
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
