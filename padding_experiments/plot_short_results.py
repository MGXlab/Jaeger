#!/usr/bin/env python
"""Plot the short-contig padding experiment (results_short.csv).

All contigs are viruses/phages, so 'correct' means predicted virus or phage.

- fig1_accuracy_vs_padding.png     % predicted virus+phage vs padding length,
                                   for natural full-length sequences
                                   (is_reference) and all nested fragments
- fig2_class_probs_vs_padding.png  mean per-class probability vs padding
- fig3_flip_reliability.png        class-flip rate vs each contig's own
                                   minimal-padding reference, and mean
                                   reliability vs padding
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORRECT = {"virus", "phage"}
BIN = 100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", default="results_short.csv")
    p.add_argument("-o", "--outdir", default="figures")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    classes = [c.removeprefix("prob_") for c in prob_cols]
    df["correct"] = df["pred_class"].isin(CORRECT)
    df["pad_bin"] = (df["pad_len"] // BIN) * BIN + BIN // 2

    ref = df[df["is_reference"] == 1].set_index("contig_id")
    df = df.join(ref[["pred_class"]].rename(columns={"pred_class": "ref_class"}),
                 on="contig_id")
    df["flip"] = df["pred_class"] != df["ref_class"]

    bins = sorted(df["pad_bin"].unique())

    # -- Fig 1: accuracy vs padding ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ref_rows = df[df["is_reference"] == 1]
    acc_ref = ref_rows.groupby("pad_bin")["correct"].mean().reindex(bins)
    acc_all = df.groupby("pad_bin")["correct"].mean().reindex(bins)
    n_ref = ref_rows.groupby("pad_bin")["correct"].size().reindex(bins)
    ax.plot(bins, acc_all * 100, "-o", label="all nested fragments")
    ax.plot(bins, acc_ref * 100, "-s",
            label="natural full-length sequences")
    for x, y, n in zip(bins, acc_ref * 100, n_ref):
        if not np.isnan(y):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, alpha=0.7)
    ax.set_xlabel("padding length (nt, bin center)")
    ax.set_ylabel("% predicted virus or phage")
    ax.set_title("Accuracy on real short virus/phage contigs vs padding length\n"
                 "(ground truth: all sequences are viruses or phages)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig1_accuracy_vs_padding.png", dpi=200)
    plt.close(fig)

    # -- Fig 2: per-class probabilities vs padding --------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    colors = plt.cm.tab10.colors
    for ax, cls, color in zip(axes.flat, classes, colors):
        grp = df.groupby("pad_bin")[f"prob_{cls}"]
        mean = grp.mean().reindex(bins)
        q25 = grp.quantile(0.25).reindex(bins)
        q75 = grp.quantile(0.75).reindex(bins)
        ax.plot(bins, mean, "-o", color=color)
        ax.fill_between(bins, q25, q75, color=color, alpha=0.15)
        ax.set_title(cls)
        ax.set_xlabel("padding length (nt, bin center)")
        ax.set_ylabel("mean probability")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-class probability vs padding length — real short "
                 "virus/phage contigs (mean + IQR)")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_class_probs_vs_padding.png", dpi=200)
    plt.close(fig)

    # -- Fig 3: flip rate and reliability vs padding ------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    nonref = df[df["is_reference"] == 0]
    flip = nonref.groupby("pad_bin")["flip"].mean().reindex(bins)
    ax1.plot(bins, flip * 100, "-o")
    ax1.set_xlabel("padding length (nt, bin center)")
    ax1.set_ylabel("class-flip rate (%)")
    ax1.set_title("Flips vs each contig's own minimal-padding reference")
    ax1.grid(alpha=0.3)

    grp = df.groupby("pad_bin")["reliability"]
    ax2.plot(bins, grp.mean().reindex(bins), "-o")
    ax2.fill_between(bins, grp.quantile(0.25).reindex(bins),
                     grp.quantile(0.75).reindex(bins), alpha=0.15)
    ax2.set_xlabel("padding length (nt, bin center)")
    ax2.set_ylabel("reliability probability")
    ax2.set_title("Reliability score vs padding length (mean + IQR)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_flip_reliability.png", dpi=200)
    plt.close(fig)

    # -- Console summary ----------------------------------------------------
    print(f"contigs: {ref.shape[0]}, fragments: {len(df)}")
    print(f"\noverall accuracy (virus+phage): {df['correct'].mean() * 100:.1f}%")
    print(f"natural full-length accuracy:   {ref_rows['correct'].mean() * 100:.1f}%")
    print("\naccuracy (%) by padding bin (natural / all nested):")
    summary = pd.DataFrame({
        "natural_%": (acc_ref * 100).round(1),
        "nested_%": (acc_all * 100).round(1),
        "flip_%": (flip * 100).round(1),
        "reliability": grp.mean().reindex(bins).round(4),
        "n_natural": n_ref,
    })
    print(summary.to_string())
    print("\nclass distribution (%) at extreme padding bins:")
    for b in [bins[0], bins[-1]]:
        dist = df[df["pad_bin"] == b]["pred_class"].value_counts(normalize=True) * 100
        print(f"  pad~{b}: {dict(dist.round(1))}")
    print(f"wrote figures to {outdir}/")


if __name__ == "__main__":
    main()
