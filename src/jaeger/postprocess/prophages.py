"""

Copyright (c) 2024 Yasas Wijesekara

"""

import os
import logging
import traceback
import pyfastx
import parasail
import numpy as np
import ruptures as rpt
import pandas as pd
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from kneed import KneeLocator
from pycirclize import Circos
from jaeger.postprocess.helpers import (
    calculate_gc_content,
    calculate_percentage_of_n,
    scale_range,
)
from jaeger.seqops.transform import reverse_complement

logger = logging.getLogger("jaeger")


def logits_to_df(config: Any, cmdline_kwargs: dict, **kwargs) -> dict:
    """
    Convert logits to a dict of dataframe for prophage region identification.
    Output of this function serves as the input for change point based
    segmentation.

    Args:
    ----
        logits: list of numpy arrys
        headers: numpy array of sequence identifiers of contigs

    Returns:
    -------
        tmp : dict of [pandas dataframes, str:host, int:lengths]
    """
    lab = {int(k): v for k, v in config["all_labels"].items()}
    tmp = {}
    for key, value, length, gc_skew, gc in zip(
        kwargs.get("headers"),
        kwargs.get("predictions"),
        kwargs.get("lengths"),
        kwargs.get("gc_skews"),
        kwargs.get("gcs"),
    ):
        if length >= cmdline_kwargs.get("lc"):
            # try:
            value = np.exp(value) / np.sum(np.exp(value), axis=1).reshape(-1, 1)
            # bac, phage, euk, arch
            max_class = np.argmax(np.mean(value, axis=0))
            host = lab[max_class]
            t = pd.DataFrame(
                value,
                columns=list(config["all_labels"].values()),
            )
            # window i starts at i * stride (windows overlap when
            # stride < fsize); clamp the last window's x to the true contig
            # length (partial terminal window)
            stride = cmdline_kwargs.get("stride") or cmdline_kwargs.get("fsize")
            t = t.assign(length=[min(i * stride, length) for i in range(len(t))])

            for k, v in lab.items():
                conv = np.convolve(value[:, k], np.ones(4), mode="same")
                # trim/pad to match the number of windows
                if len(conv) > len(t):
                    conv = conv[: len(t)]
                elif len(conv) < len(t):
                    conv = np.pad(conv, (0, len(t) - len(conv)), mode="edge")
                t[v] = conv
            t["gc"] = gc[: len(t)] if len(gc) > len(t) else gc
            gc_skew_conv = np.convolve(np.array(gc_skew), np.ones(10) / 10, mode="same")
            if len(gc_skew_conv) > len(t):
                gc_skew_conv = gc_skew_conv[: len(t)]
            elif len(gc_skew_conv) < len(t):
                gc_skew_conv = np.pad(
                    gc_skew_conv, (0, len(t) - len(gc_skew_conv)), mode="edge"
                )
            t["gc_skew"] = scale_range(gc_skew_conv, min=-1, max=1)

            tmp[f"{key}"] = [t, host, length]
        # except Exception as e:
        #     logger.error(e)
        #     logger.debug(traceback.format_exc())

    return tmp


def logits_to_df_v2(
    class_map: dict,
    cmdline_kwargs: dict,
    headers: np.ndarray,
    predictions: list[np.ndarray],
    lengths: np.ndarray,
    gc_skews: list[np.ndarray],
    gcs: list[np.ndarray],
) -> dict:
    """Convert logits to a dict of dataframes for prophage region identification.

    Compatible with the new SavedModel / TFLite / ONNX pipeline.
    Uses ``class_map`` (with ``class`` and ``index`` keys) instead of the
    legacy ``config["all_labels"]`` dict.

    Returns:
        dict mapping contig_id -> [DataFrame, host_label, length]
    """
    indices = class_map.get("index", [])
    classes = class_map.get("class", [])
    lab = {int(i): c for i, c in zip(indices, classes)}

    tmp = {}
    for key, value, length, gc_skew, gc in zip(
        headers, predictions, lengths, gc_skews, gcs
    ):
        if length >= cmdline_kwargs.get("lc", 500_000):
            value = np.exp(value) / np.sum(np.exp(value), axis=1).reshape(-1, 1)
            max_class = np.argmax(np.mean(value, axis=0))
            host = lab.get(max_class, "unknown")
            t = pd.DataFrame(value, columns=list(lab.values()))
            # window i starts at i * stride (windows overlap when
            # stride < fsize); clamp the last window's x to the true contig
            # length (partial terminal window), which pycirclize rejects as
            # outside the sector when plotting
            stride = cmdline_kwargs.get("stride") or cmdline_kwargs.get("fsize", 2000)
            t = t.assign(length=[min(i * stride, length) for i in range(len(t))])
            for k, v in lab.items():
                conv = np.convolve(value[:, k], np.ones(4), mode="same")
                if len(conv) > len(t):
                    conv = conv[: len(t)]
                elif len(conv) < len(t):
                    conv = np.pad(conv, (0, len(t) - len(conv)), mode="edge")
                t[v] = conv
            t["gc"] = gc[: len(t)] if len(gc) > len(t) else gc
            gc_skew_conv = np.convolve(np.array(gc_skew), np.ones(10) / 10, mode="same")
            if len(gc_skew_conv) > len(t):
                gc_skew_conv = gc_skew_conv[: len(t)]
            elif len(gc_skew_conv) < len(t):
                gc_skew_conv = np.pad(
                    gc_skew_conv, (0, len(t) - len(gc_skew_conv)), mode="edge"
                )
            t["gc_skew"] = scale_range(gc_skew_conv, min=-1, max=1)
            tmp[f"{key}"] = [t, host, length]
    return tmp


def plot_scores(
    logits_df: pd.DataFrame,
    config: Any,
    model: str,
    fsize: int,
    infile_base: str,
    outdir: Path,
    phage_cordinates: dict,
    stride: int | None = None,
) -> None:
    """
    Creates a circos plot of the host genome including putative prophages
    identified by Jaeger.

    Args:
    ----
        logits_df: DataFrame containing the logits.
        args: Dictionary of arguments.
        config: Dictionary containing configuration settings.
        outdir: Output directory for saving the plot.
        phage_cordinates: Dictionary of phage coordinates.
        stride: Sliding-window stride in bp (default: ``fsize``).

    Returns:
    -------
        None
    """
    # quantile cut-off 0.975 (or 0.025 of the right tail)
    lab = {int(k): v for k, v in config["all_labels"].items()}
    step = stride or fsize
    # legend_lines = []

    # Plot outer track with xticks
    major_ticks_interval = 500_000
    minor_ticks_interval = 100_000

    for contig_id in logits_df.keys():
        tmp, host, length = logits_df[contig_id]
        circos = Circos(sectors={contig_id: length})
        sector = circos.get_sector(contig_id)

        outer_track = sector.add_track((98, 100))
        outer_track.axis(fc="lightgrey")
        outer_track.xticks_by_interval(
            major_ticks_interval,
            label_formatter=lambda v: f"{v / 1e6:.1f} Mb",
            show_endlabel=False,
            label_size=11,
        )

        outer_track.xticks_by_interval(
            minor_ticks_interval, tick_length=1, show_label=False, label_size=11
        )
        colors = ["gray", "green", "red", "teal", "brown", "purple", "cyan", "pink"]
        patches = []

        for j, v in enumerate(lab.values()):
            # Plot Forward phage, bacterial, archaeal and eukaryotic scores
            if v == "phage":
                phage_track = sector.add_track((88, 97), r_pad_ratio=0.1)
                phage_track.fill_between(
                    tmp["length"],
                    tmp[v].to_numpy(),
                    vmin=0,
                    vmax=4,
                    color="orange",
                    alpha=1,
                )

                for cords in phage_cordinates[contig_id][0]:
                    # region spans [first window start, last window end];
                    # clamp to the contig length (partial terminal window),
                    # which pycirclize rejects as outside the sector
                    pcs = np.array(
                        [cords[0] * step, (cords[-1] - 1) * step + fsize]
                    ).clip(0, length)
                    phage_track.fill_between(
                        pcs,
                        np.ones_like(pcs) * 4,
                        vmin=0,
                        vmax=4,
                        color="magenta",
                        alpha=0.3,
                        lw=1,
                    )
            else:
                color = colors[j % len(colors)]
                aux_track = sector.add_track((78, 87), r_pad_ratio=0.1)
                aux_track.fill_between(
                    tmp["length"],
                    tmp[v].to_numpy(),
                    vmin=0,
                    vmax=4,
                    color=color,
                    alpha=0.7,
                )
                patches.append(Patch(color=color, label=v))

        # Plot G+C
        gc_content_track = sector.add_track((55, 70))
        tmp["gc"] = tmp["gc"] - tmp["gc"].mean()
        positive_gc_contents = np.where(tmp["gc"] > 0, tmp["gc"], 0)
        negative_gc_contents = np.where(tmp["gc"] < 0, tmp["gc"], 0)
        abs_max_gc_content = np.max(np.abs(tmp["gc"]))

        vmin, vmax = -abs_max_gc_content, abs_max_gc_content
        gc_content_track.fill_between(
            tmp["length"],
            positive_gc_contents,
            0,
            vmin=vmin,
            vmax=vmax,
            color="blue",
            alpha=0.5,
        )
        gc_content_track.fill_between(
            tmp["length"], negative_gc_contents, 0, vmin=vmin, vmax=vmax, color="black"
        )

        # Plot GC skew
        gc_skew_track = sector.add_track((45, 55))
        positive_gc_skews = np.where(tmp["gc_skew"] > 0, tmp["gc_skew"], 0)
        negative_gc_skews = np.where(tmp["gc_skew"] < 0, tmp["gc_skew"], 0)
        abs_max_gc_skew = np.max(np.abs(tmp["gc_skew"]))
        vmin, vmax = -abs_max_gc_skew, abs_max_gc_skew
        gc_skew_track.fill_between(
            tmp["length"], positive_gc_skews, 0, vmin=vmin, vmax=vmax, color="olive"
        )
        gc_skew_track.fill_between(
            tmp["length"], negative_gc_skews, 0, vmin=vmin, vmax=vmax, color="purple"
        )

        _ = circos.plotfig()
        plt.title(
            f"{contig_id.replace('___', ',')}", fontdict={"size": 14, "weight": "bold"}
        )
        # Add legend
        handles = (
            [
                Patch(color="orange", label="phage"),
                Patch(color="magenta", alpha=0.3, label="putative prophage"),
            ]
            + patches
            + [
                Line2D(
                    [],
                    [],
                    color="blue",
                    label="$ > \overline{G+C}$",
                    marker="^",
                    ms=6,
                    ls="None",
                    alpha=0.5,
                ),
                Line2D(
                    [],
                    [],
                    color="black",
                    label="$ < \overline{G+C}$",
                    marker="v",
                    ms=6,
                    ls="None",
                ),
                Line2D(
                    [],
                    [],
                    color="olive",
                    label="Positive GC Skew",
                    marker="^",
                    ms=6,
                    ls="None",
                ),
                Line2D(
                    [],
                    [],
                    color="purple",
                    label="Negative GC Skew",
                    marker="v",
                    ms=6,
                    ls="None",
                ),
            ]
        )
        _ = circos.ax.legend(
            handles=handles, bbox_to_anchor=(0.51, 0.50), loc="center", fontsize=11
        )

        plt.savefig(
            os.path.join(
                outdir / f"{infile_base}_jaeger_{contig_id.split(' ')[0]}.pdf",
            ),
            bbox_inches="tight",
            dpi=300,
        )
        logger.info(
            (
                "prophage plot saved at "
                + os.path.join(
                    outdir / f"{infile_base}_jaeger_{contig_id.split(' ')[0]}.pdf",
                )
            )
        )
        plt.close()


def plot_scores_linear(
    logits_df: pd.DataFrame,
    config: Any,
    model: str,
    fsize: int,
    infile_base: str,
    outdir: Path,
    phage_cordinates: dict,
    stride: int | None = None,
) -> None:
    """
    Creates a linear genome plot of the host genome including putative
    prophages identified by Jaeger.

    Args:
    ----
        logits_df: DataFrame containing the logits.
        config: Dictionary containing configuration settings.
        model: Model identifier string.
        fsize: Fragment size in bp.
        infile_base: Base name of the input file.
        outdir: Output directory for saving the plot.
        phage_cordinates: Dictionary of phage coordinates.
        stride: Sliding-window stride in bp (default: ``fsize``).

    Returns:
    -------
        None
    """
    lab = {int(k): v for k, v in config["all_labels"].items()}
    step = stride or fsize
    colors = ["gray", "green", "red", "teal", "brown", "purple", "pink", "olive"]

    for contig_id in logits_df.keys():
        tmp, host, length = logits_df[contig_id]
        fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(14, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1.5, 1.5], "hspace": 0.05},
        )
        ax_phage, ax_aux, ax_gc, ax_skew = axes

        # --- Phage score track (top) ---
        ax_phage.fill_between(
            tmp["length"],
            tmp["phage"].to_numpy(),
            color="orange",
            alpha=0.8,
            label="phage score",
        )
        # Highlight prophage regions
        for cords in phage_cordinates.get(contig_id, [[], []])[0]:
            # region spans [first window start, last window end];
            # clamp to the contig length (partial terminal window)
            pcs = np.array([cords[0] * step, (cords[-1] - 1) * step + fsize]).clip(
                0, length
            )
            ax_phage.fill_between(
                pcs,
                np.ones_like(pcs) * 4,
                color="magenta",
                alpha=0.25,
                label="putative prophage"
                if cords[0] == phage_cordinates[contig_id][0][0][0]
                else "",
            )
        ax_phage.set_ylim(0, 4)
        ax_phage.set_ylabel("Phage score", fontsize=10)
        ax_phage.set_title(
            f"{contig_id.replace('___', ',')}",
            fontdict={"size": 12, "weight": "bold"},
        )
        ax_phage.legend(loc="upper right", fontsize=8)
        ax_phage.grid(True, linestyle=":", alpha=0.4)

        # --- Auxiliary class scores ---
        patches = []
        for j, v in enumerate(lab.values()):
            if v == "phage":
                continue
            ax_aux.plot(
                tmp["length"],
                tmp[v].to_numpy(),
                color=colors[j % len(colors)],
                alpha=0.7,
                linewidth=1.2,
                label=v,
            )
            patches.append(Patch(color=colors[j % len(colors)], label=v))
        ax_aux.set_ylim(0, 4)
        ax_aux.set_ylabel("Class scores", fontsize=10)
        ax_aux.legend(loc="upper right", fontsize=8)
        ax_aux.grid(True, linestyle=":", alpha=0.4)

        # --- GC content track ---
        gc_centered = tmp["gc"] - tmp["gc"].mean()
        positive_gc = np.where(gc_centered > 0, gc_centered, 0)
        negative_gc = np.where(gc_centered < 0, gc_centered, 0)
        abs_max_gc = np.max(np.abs(gc_centered))
        vmin_gc, vmax_gc = -abs_max_gc, abs_max_gc

        ax_gc.fill_between(
            tmp["length"],
            positive_gc,
            0,
            color="blue",
            alpha=0.5,
            label=r"$> \overline{G+C}$",
        )
        ax_gc.fill_between(
            tmp["length"],
            negative_gc,
            0,
            color="black",
            alpha=0.7,
            label=r"$< \overline{G+C}$",
        )
        ax_gc.set_ylim(vmin_gc, vmax_gc)
        ax_gc.set_ylabel("G+C dev.", fontsize=10)
        ax_gc.legend(loc="upper right", fontsize=8)
        ax_gc.grid(True, linestyle=":", alpha=0.4)

        # --- GC skew track ---
        positive_skew = np.where(tmp["gc_skew"] > 0, tmp["gc_skew"], 0)
        negative_skew = np.where(tmp["gc_skew"] < 0, tmp["gc_skew"], 0)
        abs_max_skew = np.max(np.abs(tmp["gc_skew"]))
        vmin_skew, vmax_skew = -abs_max_skew, abs_max_skew

        ax_skew.fill_between(
            tmp["length"],
            positive_skew,
            0,
            color="olive",
            alpha=0.7,
            label="Positive GC skew",
        )
        ax_skew.fill_between(
            tmp["length"],
            negative_skew,
            0,
            color="purple",
            alpha=0.7,
            label="Negative GC skew",
        )
        ax_skew.set_ylim(vmin_skew, vmax_skew)
        ax_skew.set_ylabel("GC skew", fontsize=10)
        ax_skew.set_xlabel("Genome position (bp)", fontsize=10)
        ax_skew.legend(loc="upper right", fontsize=8)
        ax_skew.grid(True, linestyle=":", alpha=0.4)

        # Format x-axis
        ax_skew.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f} Mb")
        )

        out_path = outdir / f"{infile_base}_jaeger_{contig_id.split(' ')[0]}_linear.pdf"
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        logger.info(f"linear prophage plot saved at {out_path}")
        plt.close()


def _extend_seed(
    seed: int,
    phage_scores: np.ndarray,
    host_scores: np.ndarray,
) -> tuple[int, int]:
    """Extend a seed window to find the maximum scoring region.

    Extends left and right from the seed, adding windows as long as the
    mean phage score of the region exceeds the mean host score. The extension
    is conservative: it stops if adding a window would drop the mean below
    the host mean, or if the window being added has a very low score.
    """
    n = len(phage_scores)
    left = right = seed

    # Extend left
    while left > 0:
        candidate = left - 1
        # Check if the window being added has a reasonable score
        if phage_scores[candidate] < host_scores[candidate] - 0.5:
            break
        # Check if the extended region still has mean phage > mean host
        if (
            phage_scores[candidate : right + 1].mean()
            > host_scores[candidate : right + 1].mean()
        ):
            left = candidate
        else:
            break

    # Extend right
    while right < n - 1:
        candidate = right + 1
        # Check if the window being added has a reasonable score
        if phage_scores[candidate] < host_scores[candidate] - 0.5:
            break
        # Check if the extended region still has mean phage > mean host
        if (
            phage_scores[left : candidate + 1].mean()
            > host_scores[left : candidate + 1].mean()
        ):
            right = candidate
        else:
            break

    return left, right


def _merge_overlapping_regions(
    regions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent regions."""
    if not regions:
        return []
    regions = sorted(regions)
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start <= merged[-1][1] + 1:  # Overlapping or adjacent
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _filter_by_length(
    regions: list[tuple[int, int]],
    fsize: int,
    stride: int,
    min_length: int = 10_000,
    max_length: int = 200_000,
) -> list[tuple[int, int]]:
    """Filter regions by length (prophages are typically 10-200 kb)."""
    filtered = []
    for start_idx, end_idx in regions:
        length = (end_idx - start_idx) * stride + fsize
        if min_length <= length <= max_length:
            filtered.append((start_idx, end_idx))
    return filtered


def _filter_by_score_consistency(
    tmp: pd.DataFrame,
    regions: list[tuple[int, int]],
    identifier: str,
    max_cv: float = 0.5,
) -> list[tuple[int, int]]:
    """Filter regions by score consistency (coefficient of variation)."""
    filtered = []
    for start_idx, end_idx in regions:
        scores = tmp.loc[start_idx:end_idx, identifier]
        if scores.mean() > 0:
            cv = scores.std() / scores.mean()
            if cv < max_cv:
                filtered.append((start_idx, end_idx))
    return filtered


def segment(
    logits_df: pd.DataFrame,
    outdir: Path,
    cutoff_length: int = 500_000,
    sensitivity: float = 1.5,
    identifier: str = "phage",
    host_identifier: str = "bacteria",
    fasta_path: str | None = None,
) -> dict:
    """
    Segments the logit arrays using a combined approach:
    1. Extension from high-scoring seeds for initial detection
    2. Change point detection for boundary refinement
    3. Merge overlapping regions
    4. Apply PPV improvement filters (length, score consistency, phage genes)

    Args:
    ----
        logits_df (dict): Dictionary containing data for segmentation.
        outdir (str): Output directory for saving segmentation results.
        cutoff_length (int, optional): Length threshold for segmenting
                                       data. Defaults to 500,000.
        sensitivity (float, optional): Sensitivity threshold for segmentation.
                                       Defaults to 1.5.
        identifier (str, optional): Column name for phage scores.
        host_identifier (str, optional): Column name for host scores.
        fasta_path (str, optional): Path to FASTA file for phage gene filtering.

    Returns:
    -------
        dict: A dictionary containing segmented data coordinates and scores.
    """
    phage_cordinates = {}
    for key, (tmp, host, length) in logits_df.items():
        if length <= cutoff_length:
            continue

        try:
            phage_scores = tmp[identifier].to_numpy()
            host_scores = (
                tmp[host_identifier].to_numpy()
                if host_identifier in tmp.columns
                else np.zeros_like(phage_scores)
            )

            # Step 1: Extension from high-scoring seeds
            seeds = np.where(phage_scores > sensitivity)[0]
            extension_regions = []
            if len(seeds) > 0:
                for seed in seeds:
                    left, right = _extend_seed(seed, phage_scores, host_scores)
                    extension_regions.append((left, right))
                extension_regions = _merge_overlapping_regions(extension_regions)

            # Step 2: Change point detection for boundary refinement
            cpd_regions = []
            algo = rpt.KernelCPD(kernel="linear", min_size=2, jump=1).fit(
                tmp[identifier].to_numpy()
            )
            if bkpts := [
                algo.predict(pen=i)
                for i in range(1, 10)
                if len(algo.predict(pen=i)) > 1
            ]:
                bkpt_lens = np.array([len(b) for b in bkpts])
                kn = KneeLocator(
                    bkpt_lens,
                    list(range(len(bkpts))),
                    curve="convex",
                    direction="decreasing",
                )
                bkpt_index = (
                    [len(b) for b in bkpts].index(kn.knee)
                    if kn.knee
                    else np.searchsorted(bkpt_lens, 1)
                )
                if bkpt_index == len(bkpt_lens):
                    bkpt_index = None

                ranges = [
                    bkpts[bkpt_index][i : i + 2]
                    for i in range(len(bkpts[bkpt_index]) - 1)
                ]
                range_scores = np.array(
                    [tmp.loc[s:e][identifier].mean() for s, e in ranges]
                )
                range_mask = range_scores > sensitivity
                cpd_regions = [
                    (int(r[0]), int(r[1])) for r in np.array(ranges)[range_mask]
                ]

            # Step 3: Merge extension and CPD regions
            all_regions = extension_regions + cpd_regions
            if not all_regions:
                phage_cordinates[key] = [[], []]
                continue

            merged_regions = _merge_overlapping_regions(all_regions)

            # Step 4: Apply PPV improvement filters
            fsize = 2000
            stride = 1500

            # Length filter (based on smallest prophage: 6 kb)
            merged_regions = _filter_by_length(
                merged_regions, fsize, stride, min_length=6_000, max_length=250_000
            )

            # Score consistency filter (balanced)
            merged_regions = _filter_by_score_consistency(
                tmp, merged_regions, identifier, max_cv=0.6
            )

            # Final filter by mean score > host mean AND mean score > sensitivity
            final_regions = []
            final_scores = []
            for start, end in merged_regions:
                region_phage = phage_scores[start : end + 1].mean()
                region_host = host_scores[start : end + 1].mean()
                if region_phage > region_host and region_phage > sensitivity:
                    final_regions.append([start, end])
                    final_scores.append(region_phage)

            phage_cordinates[key] = [
                np.array(final_regions) if final_regions else [],
                np.array(final_scores) if final_scores else [],
            ]
        except Exception:
            phage_cordinates[key] = [[], []]
            logger.debug(traceback.format_exc())

    return phage_cordinates


def get_prophage_alignment_summary(
    result_object, seq_len, record, cordinates, phage_score, type_="DTR"
) -> dict:
    """
    Generates a summary of the prophage alignment results.

    Args:
    ----
        result_object: The alignment result object.
        seq_len: The length of the DNA sequence.
        record: The DNA sequence record.
        cordinates: The start and end coordinates for alignment.
        phage_score: The score of the prophage.
        type_ (str, optional): The type of prophage repeat. Defaults to "DTR".

    Returns:
    -------
        dict or str: A dictionary containing the prophage
    """

    if result_object is None:
        s_alig_start = cordinates["start"][0]
        e_alig_end = cordinates["end"][0]
        sequence = record[1][s_alig_start:e_alig_end]
        gc_ = calculate_gc_content(sequence)

        return {
            "contig_id": record[0],
            "seq_len": seq_len,
            "region_len": e_alig_end - s_alig_start,
            "phage_score": phage_score,
            "n%": None,
            "gc%": gc_,
            "reject": None,
            "sstart": s_alig_start,
            "send": None,
            "estart": None,
            "eend": e_alig_end,
            "att_alignment_length": None,
            "att_identities": None,
            "att_identity": None,
            "att_score": None,
            "att_type": None,
            "att_fgaps": None,
            "att_rgaps": None,
            "attL": None,
            "attR": None,
        }
    elif result_object.saturated:
        return "saturated"

    else:
        alig_len = len(result_object.traceback.query)
        f_gaps = result_object.traceback.query.count("-")
        rc_gaps = result_object.traceback.ref.count("-")
        iden = result_object.traceback.comp.count("|")

        ltr_cutoff = 250

        if type_ == "ITR":
            s_alig_end = cordinates["start"][0] + result_object.end_query + 1
            s_alig_start = s_alig_end - alig_len
            e_alig_start = cordinates["end"][1] - result_object.end_ref - 1
            e_alig_end = e_alig_start + alig_len
        elif type_ == "DTR":
            s_alig_end = cordinates["start"][0] + result_object.end_query
            s_alig_start = s_alig_end - alig_len + 1
            e_alig_end = cordinates["end"][0] + result_object.end_ref
            e_alig_start = e_alig_end - alig_len + 1

            if (s_alig_end - s_alig_start) >= ltr_cutoff:
                type_ = f"LTR_{type_}"

        sequence = record[1][s_alig_start:e_alig_end]
        percentage_of_N = calculate_percentage_of_n(sequence)
        gc_ = calculate_gc_content(sequence)

        return {
            "contig_id": record[0],
            "seq_len": seq_len,
            "region_len": e_alig_end - s_alig_start,
            "phage_score": phage_score,
            "n%": percentage_of_N,
            "gc%": gc_,
            "reject": percentage_of_N > 0.20,
            "sstart": s_alig_start,
            "send": s_alig_end,
            "estart": e_alig_start,
            "eend": e_alig_end,
            "att_alignment_length": alig_len,
            "att_identities": iden,
            "att_identity": round(iden / alig_len, 2),
            "att_score": result_object.score,
            "att_type": type_,
            "att_fgaps": f_gaps,
            "att_rgaps": rc_gaps,
            "attL": result_object.traceback.query,
            "attR": result_object.traceback.ref,
        }


def _nearest_trna(
    position: int, trnas: list[tuple[int, int, int, str]]
) -> tuple[int | None, str | None]:
    """Return (distance, type) of the nearest tRNA to position, or (None, None)."""
    if not trnas:
        return None, None
    best_dist = None
    best_type = None
    for start, end, strand, trna_type in trnas:
        # Distance to the nearest edge of the tRNA
        if position < start:
            dist = start - position
        elif position >= end:
            dist = position - end
        else:
            dist = 0
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_type = trna_type
    return best_dist, best_type


def prophage_report(
    fsize: int,
    filehandle: Any,
    prophage_cordinates: dict,
    outdir: Path,
    refined_boundaries: dict | None = None,
    stride: int | None = None,
    trna_features: dict[str, list[tuple[int, int, int, str]]] | None = None,
):
    """
    Searches for direct repeats at prophage boundaries and generates
    prophage summaries.

    Args:
    ----
        fsize: Fragment / window size in bp.
        filehandle: File handle for reading DNA sequences.
        prophage_cordinates: Coordinates of prophages for comparison.
        outdir: Output directory for saving prophage summaries.
        refined_boundaries: Optional mapping from contig id to a list of
            ``(raw_start, raw_end, refined_start, refined_end)`` tuples. When
            provided, the att-region search and reported region coordinates use
            the refined boundaries.
        stride: Sliding-window stride in bp (default: ``fsize``).
        trna_features: Optional mapping from contig id to a list of
            ``(start, end, strand, type)`` tRNA tuples. When provided, tRNA
            proximity evidence is added to the output TSV.
    Returns
    -------
        None
    """

    user_matrix = parasail.matrix_create("ACGT", 2, -100)
    step = stride or fsize
    summaries = []

    def append_summary(result, seq_len, record, start, end, j, type_):
        summaries.append(
            get_prophage_alignment_summary(
                result_object=result,
                seq_len=seq_len,
                record=record,
                cordinates={
                    "start": [start, start + off_set],
                    "end": [end - off_set, end + scan_length],
                },
                phage_score=j,
                type_=type_,
            )
        )

    total_contigs = len(
        [h for h in prophage_cordinates if len(prophage_cordinates[h][0]) > 0]
    )
    processed_contigs = 0
    for record in pyfastx.Fasta(filehandle, build_index=False):
        seq_len = len(record[1])
        header = record[0].replace(",", "___")
        logger.debug(f"generating prophage report for {header}")
        if seq_len > 500_000:
            cords, scores = prophage_cordinates.get(f"{header}", [[], []])
            if len(cords) > 0 and len(scores) > 0:
                processed_contigs += 1
                if processed_contigs % 10 == 0:
                    logger.info(
                        f"prophage report: {processed_contigs}/{total_contigs} contigs"
                    )
                contig_refined = (
                    refined_boundaries.get(header) if refined_boundaries else None
                )
                trnas = trna_features.get(header, []) if trna_features else []
                for idx, ((start, end), j) in enumerate(zip(cords, scores)):
                    # region spans [first window start, last window end]
                    raw_start = int(start * step)
                    raw_end = int((end - 1) * step + fsize)
                    if contig_refined is not None and idx < len(contig_refined):
                        _, _, refined_start, refined_end = contig_refined[idx]
                    else:
                        refined_start, refined_end = raw_start, raw_end

                    region_len = refined_end - refined_start
                    scan_length = min(max(int(seq_len * 0.04), 400), 4000)
                    off_set = 2000 if region_len // 2 >= 14000 else region_len // 4

                    search_start = max(refined_start - scan_length, 0)
                    search_end = min(refined_end + scan_length, seq_len)

                    left_seq = str(record[1][search_start : refined_start + off_set])
                    right_seq = str(record[1][refined_end - off_set : search_end])
                    if not left_seq or not right_seq:
                        # degenerate (e.g. zero-width) region: parasail
                        # rejects empty inputs, so report no repeat found
                        summary = get_prophage_alignment_summary(
                            result_object=None,
                            seq_len=seq_len,
                            record=record,
                            cordinates={
                                "start": [refined_start, None],
                                "end": [refined_end, None],
                            },
                            phage_score=j,
                            type_=None,
                        )
                        summary["raw_start"] = raw_start
                        summary["raw_end"] = raw_end
                        summaries.append(summary)
                        continue

                    result_dtr = parasail.sw_trace_scan_16(
                        left_seq,
                        right_seq,
                        100,
                        5,
                        user_matrix,
                    )

                    result_itr = parasail.sw_trace_scan_16(
                        left_seq,
                        reverse_complement(right_seq),
                        100,
                        5,
                        user_matrix,
                    )

                    if (
                        len(result_itr.traceback.query) > 12
                        or len(result_dtr.traceback.query) > 12
                    ):
                        if result_itr.score > result_dtr.score:
                            summary = get_prophage_alignment_summary(
                                result_object=result_itr,
                                seq_len=seq_len,
                                record=record,
                                cordinates={
                                    "start": [search_start, search_start + off_set],
                                    "end": [
                                        refined_end - off_set,
                                        search_end,
                                    ],
                                },
                                phage_score=j,
                                type_="ITR",
                            )
                        else:
                            summary = get_prophage_alignment_summary(
                                result_object=result_dtr,
                                seq_len=seq_len,
                                record=record,
                                cordinates={
                                    "start": [search_start, search_start + off_set],
                                    "end": [
                                        refined_end - off_set,
                                        search_end,
                                    ],
                                },
                                phage_score=j,
                                type_="DTR",
                            )
                    else:
                        summary = get_prophage_alignment_summary(
                            result_object=None,
                            seq_len=seq_len,
                            record=record,
                            cordinates={
                                "start": [refined_start, None],
                                "end": [refined_end, None],
                            },
                            phage_score=j,
                            type_=None,
                        )

                    summary["raw_start"] = raw_start
                    summary["raw_end"] = raw_end
                    # tRNA proximity evidence
                    if trnas:
                        left_dist, left_type = _nearest_trna(refined_start, trnas)
                        right_dist, right_type = _nearest_trna(refined_end, trnas)
                        summary["trna_left_distance"] = left_dist
                        summary["trna_right_distance"] = right_dist
                        summary["trna_left_type"] = left_type
                        summary["trna_right_type"] = right_type
                    summaries.append(summary)

    if summaries:
        df = pd.DataFrame(summaries)
        df["contig_id"] = df["contig_id"].apply(lambda x: x.replace("___", ","))
        df.to_csv(
            outdir / "prophages_jaeger.tsv", sep="\t", index=False, float_format="%.3f"
        )
        logger.info(f"prophage cordinates saved at {outdir / 'prophages_jaeger.tsv'}")
