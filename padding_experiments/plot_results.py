#!/usr/bin/env python
"""Plot padding-length experiment results.

Reads results.csv produced by run_padding_experiment.py and writes three
figures:

- fig1_class_probs_vs_padding.png  per-class probability vs padding length
- fig2_class_flip_vs_padding.png   class-flip rate vs padding length
                                   (argmax differs from the same contig's
                                   unpadded 2000 nt reference)
- fig3_reliability_vs_padding.png  reliability probability vs padding length
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", default="results.csv")
    p.add_argument("-o", "--outdir", default="figures")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    import pathlib

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    classes = [c.removeprefix("prob_") for c in prob_cols]

    ref = (
        df[df["pad_char"] == "none"]
        .set_index("contig_id")[["pred_class"]]
        .rename(columns={"pred_class": "ref_class"})
    )
    pad = df[df["pad_char"] != "none"].join(ref, on="contig_id")
    pad["flip"] = pad["pred_class"] != pad["ref_class"]

    pad_lengths = sorted(pad["pad_len"].unique())
    pad_chars = sorted(pad["pad_char"].unique())
    char_style = {"M": "-", "N": "--"}
    colors = plt.cm.tab10.colors

    # -- Fig 1: per-class probability vs padding length --------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, cls, color in zip(axes.flat, classes, colors):
        col = f"prob_{cls}"
        for ch in pad_chars:
            sub = pad[pad["pad_char"] == ch]
            grp = sub.groupby("pad_len")[col]
            mean = grp.mean().reindex(pad_lengths)
            q25 = grp.quantile(0.25).reindex(pad_lengths)
            q75 = grp.quantile(0.75).reindex(pad_lengths)
            ax.plot(pad_lengths, mean, char_style.get(ch, "-"), color=color,
                    label=f"{ch} padding")
            ax.fill_between(pad_lengths, q25, q75, color=color, alpha=0.15)
        ax.set_title(cls)
        ax.set_xlabel("padding length (nt)")
        ax.set_ylabel("mean probability")
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Per-class probability vs padding length "
                 "(nested fragments, mean + IQR over contigs)")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_class_probs_vs_padding.png", dpi=200)
    plt.close(fig)

    # -- Fig 2: class-flip rate vs padding length --------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for ch in pad_chars:
        sub = pad[pad["pad_char"] == ch]
        rate = sub.groupby("pad_len")["flip"].mean().reindex(pad_lengths)
        ax.plot(pad_lengths, rate * 100, char_style.get(ch, "-") + "o",
                label=f"{ch} padding")
    ax.set_xlabel("padding length (nt)")
    ax.set_ylabel("class-flip rate (%)")
    ax.set_title("Fraction of fragments whose argmax class differs from\n"
                 "the same contig's unpadded 2000 nt reference")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig2_class_flip_vs_padding.png", dpi=200)
    plt.close(fig)

    # -- Fig 3: reliability vs padding length ------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for ch in pad_chars:
        sub = pad[pad["pad_char"] == ch]
        grp = sub.groupby("pad_len")["reliability"]
        mean = grp.mean().reindex(pad_lengths)
        q25 = grp.quantile(0.25).reindex(pad_lengths)
        q75 = grp.quantile(0.75).reindex(pad_lengths)
        ax.plot(pad_lengths, mean, char_style.get(ch, "-") + "o",
                label=f"{ch} padding")
        ax.fill_between(pad_lengths, q25, q75, alpha=0.15)
    ref_rel = df[df["pad_char"] == "none"]["reliability"].mean()
    ax.axhline(ref_rel, color="k", ls=":", alpha=0.6,
               label=f"unpadded reference (mean={ref_rel:.3f})")
    ax.set_xlabel("padding length (nt)")
    ax.set_ylabel("reliability probability")
    ax.set_title("Reliability score vs padding length (mean + IQR)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig3_reliability_vs_padding.png", dpi=200)
    plt.close(fig)

    # -- Console summary ----------------------------------------------------
    print(f"contigs: {df['contig_id'].nunique()}, fragments: {len(df)}")
    print("\nflip rate (%) by pad char / pad len:")
    print((pad.groupby(["pad_char", "pad_len"])["flip"].mean() * 100)
          .round(1).unstack(0).to_string())
    print("\nmean reliability by pad char / pad len:")
    print(pad.groupby(["pad_char", "pad_len"])["reliability"].mean()
          .round(4).unstack(0).to_string())
    mn = pad[pad["pad_char"] == "M"][prob_cols].to_numpy()
    nn = pad[pad["pad_char"] == "N"][prob_cols].to_numpy()
    print(f"\nmax |prob_M - prob_N| = {np.abs(mn - nn).max():.2e}")
    print(f"wrote figures to {outdir}/")


if __name__ == "__main__":
    main()
