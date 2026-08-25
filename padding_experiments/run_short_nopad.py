#!/usr/bin/env python
"""Score real short contigs (1000-1999 bp) at native length, no padding.

Batch size 1, so no masked rows are added anywhere: each sequence is
encoded and scored at exactly its own length. This is the strict
'without padding' baseline for results_short.csv. (The old two-pass
short-contig path padded each batch to its longest sequence with zeros,
which is nearly identical to this.)

Outputs: results_short_nopad.csv with one row per contig.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="input FASTA")
    p.add_argument("-m", "--model", default="jaeger_f47f36db_1.2M_fragment")
    p.add_argument("-o", "--output", default="results_short_nopad.csv")
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
        g = seq.count("G")
        c = seq.count("C")
        a = seq.count("A")
        t = seq.count("T")
        gc_skew = safe_divide(g - c, g + c)
        csv_line = f"{seq},{contig_id},0,1,0,{seqlen},{g},{c},{a},{t},{gc_skew: .3f}"
        records.append(
            {
                "csv_line": csv_line,
                "contig_id": contig_id,
                "real_len": seqlen,
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
    print(f"scoring {len(records)} short contigs at native length (batch size 1)")

    y_pred = run_inference(model, records, batch=1)

    probs = tf.nn.softmax(y_pred["prediction"], axis=-1).numpy()
    reliability = None
    if "reliability" in y_pred:
        reliability = tf.nn.sigmoid(y_pred["reliability"]).numpy().squeeze()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["contig_id", "real_len", "pred_class", "reliability"]
            + [f"prob_{c}" for c in classes]
        )
        for i, rec in enumerate(records):
            pred_class = classes[int(np.argmax(probs[i]))]
            rel = float(reliability[i]) if reliability is not None else ""
            writer.writerow(
                [rec["contig_id"], rec["real_len"], pred_class, rel]
                + [f"{p:.6f}" for p in probs[i]]
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
