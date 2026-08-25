import pyfastx
import numpy as np
from pathlib import Path
from rich.progress import track
from jaeger.utils.logging import get_logger

from jaeger.dataops.dataset import build_dataset
from jaeger.dataops.convert import convert_dataset


logger = get_logger(log_file=None, log_path=None, level=3)


def mask_core(**kwargs):
    # Pre-define the alt-nucleotide map once (module scope would also work, but
    # keeping it local avoids polluting the module namespace).
    _ALT = {
        ord("A"): ("T", "G", "C"),
        ord("T"): ("A", "G", "C"),
        ord("G"): ("A", "T", "C"),
        ord("C"): ("A", "T", "G"),
    }
    _DEFAULT_ALTS = ("N", "N", "N")

    input_path = kwargs.get("input")
    output_path = kwargs.get("output")
    min_perc = kwargs.get("minperc", 0.0)
    max_perc = kwargs.get("maxperc", 1.0)
    step = kwargs.get("step", 0.01)  # increment in mutation fraction
    mutate = kwargs.get("mutate", False)  # replace with random nucleotides
    seed = kwargs.get("seed")

    # Single seeded RNG drives both index selection and nucleotide replacement
    # so runs are reproducible when --seed is provided.
    rng = np.random.default_rng(seed)

    def hard_mask(seq: str, indices):
        """Turn seq[i] -> N for each i in indices, leaving other letters intact."""
        ba = bytearray(seq, "ascii")
        for i in indices:
            ba[i] = 0x4E
        return ba.decode("ascii")

    def replacement_mutation(seq: str, indices):
        """Replace seq[i] with one of its 3 alternatives uniformly at random."""
        ba = bytearray(seq, "ascii")
        choices = rng.integers(0, 3, size=len(indices))
        for i, choice in zip(indices, choices):
            alts = _ALT.get(ba[i], _DEFAULT_ALTS)
            ba[i] = ord(alts[choice])
        return ba.decode("ascii")

    f = pyfastx.Fasta(input_path, build_index=False)

    with open(output_path, "w") as fh:
        for name, seq in track(f, description="Processing..."):
            seq = str(seq)
            seqlen = len(seq)
            current_perc = min_perc
            # Shuffle all positions once up front, then consume cumulative
            # chunks. This is O(N) total instead of rebuilding an "available"
            # set every iteration (O(N) per step -> quadratic on long contigs).
            order = rng.permutation(seqlen)
            cursor = 0
            num_mutate = int(seqlen * step)

            while current_perc <= max_perc:
                # Write the entry at the current masking level.
                fh.write(f">{name}_mutperc_{current_perc * 100:.2f}\n")
                for i in range(0, len(seq), 70):
                    fh.write(seq[i : i + 70] + "\n")

                if cursor >= seqlen or num_mutate <= 0:
                    break
                new_indices = order[cursor : cursor + num_mutate]
                cursor += len(new_indices)

                if mutate:
                    seq = replacement_mutation(seq, new_indices)
                else:
                    seq = hard_mask(seq, new_indices)

                current_perc += step


def dataset_core(**kwargs):
    """
    Generate a non-redundant fragment database from a FASTA/CSV file using MMseqs2.

    Required kwargs:
      input      : path to input FASTA/CSV of contigs
      output     : prefix for output FASTA/CSV files
      valperc    : 0.1    # fraction for validation set
      trainperc  : 0.8    # fraction for training set
      testperc   : 0.1    # fraction for test set
      maxiden    : 0.6    # minimum sequence identity for clustering
      maxcov     : 0.6    # minimum coverage fraction for clustering
      method     : "ANI"  # or "AAI" (to do: AAI)
      outtype    : "CSV"  # or "FASTA"
      intype     : "CSV"  # or "FASTA"
      class      : int       # class label as an int
      class_col  : int     # col index of CSV with class id
      seq_col    : int     # col index of CSV with sequence
    """
    build_dataset(**kwargs)


def convert_core(**kwargs):
    """Convert between CSV and FASTA using pandas and pyfastx.

    Parameters
    ----------
    input_path : str
        Path to the input file (CSV or FASTA).
    output_path : str
        Path to the output file (FASTA or CSV).
    input_type : str
        Type of the input file: 'csv' or 'fasta'.
    """
    import pandas as pd

    input_path = Path(kwargs.get("input"))
    output_path = Path(kwargs.get("output"))
    input_type = kwargs.get("itype")
    if input_type == "CSV":
        # CSV -> FASTA
        df = pd.read_csv(
            input_path, usecols=[0, 1, 2], names=["class", "sequence", "id"], dtype=str
        )
        with open(output_path, "w") as fasta_out:
            for cls_id, seq, seq_id in zip(df["class"], df["sequence"], df["id"]):
                fasta_out.write(
                    f">{seq_id.strip()}__class={cls_id.strip()}\n{seq.strip()}\n"
                )
        print(f"[✓] Converted CSV to FASTA: {output_path}")

    elif input_type == "FASTA":
        # FASTA -> CSV
        fasta = pyfastx.Fasta(str(input_path), build_index=False)
        records = []
        for name, seq in fasta:
            if "__class=" in name:
                seq_id, cls_id = name.split("__class=", 1)
            else:
                seq_id, cls_id = name, "0"
            records.append((cls_id, seq, seq_id))
        df = pd.DataFrame(records, columns=["class", "sequence", "id"])
        df.to_csv(output_path, index=False, header=False)
        print(f"[✓] Converted FASTA to CSV: {output_path}")

    else:
        raise ValueError("input_type must be 'CSV' or 'FASTA'")


# ------------------------------------------------------------------
# Stats (late import to avoid heavy deps at module load time)
# ------------------------------------------------------------------
from jaeger.utils.stats import welch_t_one_tailed  # noqa: E402


def stats_core(**kwargs):
    """Calculate stats and create plots from jaeger output/s.

    1. percentage of each class
    2. reliability score distribution
    3. class score distributions
    """
    import matplotlib.pyplot as plt

    import seaborn as sns
    import pandas as pd

    input_path = Path(kwargs.get("input"))
    output_path = Path(kwargs.get("output"))
    output_path.mkdir(exist_ok=True, parents=True)
    pct_class = output_path / "class_percentages.png"
    pct_class_pval = output_path / "class_percentages_pval.png"
    relscore = output_path / "reliability_scores.png"
    relscore_len = output_path / "reliability_scores_by_length.png"
    ent = output_path / "entropy.png"
    eng = output_path / "energy.png"
    clscores = output_path / "class_scores.png"
    tsv_with_pvals = output_path / "jaeger_output_with_pvals.tsv"

    df = pd.read_table(input_path)
    sns.set_context("paper", font_scale=1.2)
    reliability_available = pd.api.types.is_numeric_dtype(df["reliability_score"])
    if not reliability_available:
        logger.warning(
            "Reliability score is unavailable in the input; skipping reliability-related plots."
        )
    if len(df) > 1:
        if reliability_available:
            # Create the count plot
            df["above_threshold"] = df["reliability_score"].apply(
                lambda x: "passed" if x >= 0.8 else "failed"
            )
            ax = sns.countplot(
                data=df,
                x="prediction",
                hue="above_threshold",
                palette="pastel",
                stat="percent",
            )
            # Annotate bars with percentage values (already in percent)
            for p in ax.patches:
                percentage = p.get_height()
                if percentage > 0:
                    ax.text(
                        p.get_x() + p.get_width() / 2,
                        p.get_height(),
                        f"{percentage:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
            # Style tweaks
            ax.set_ylabel("Percentage")
            ax.set_xlabel("Prediction")
            ax.set_title("Class Distribution (%)")
            sns.despine()
            plt.tight_layout()
            plt.savefig(pct_class, dpi=150, bbox_inches="tight")
            plt.close()

            # Calculate per-class distribution of reliability scores
            ax = sns.violinplot(df, x="prediction", y="reliability_score")
            sns.stripplot(
                df,
                x="prediction",
                y="reliability_score",
                s=1,
                alpha=0.1,
                color="gray",
                ax=ax,
            )
            ax.set_ylabel("Reliability score")
            ax.set_xlabel("Class")
            ax.set_title("Per-class distribution of reliability scores")
            sns.despine()
            plt.tight_layout()
            plt.savefig(relscore, dpi=150, bbox_inches="tight")
            plt.close()

        # Calculate per-class distribution of entropy
        ax = sns.violinplot(df, x="prediction", y="entropy")
        sns.stripplot(
            df, x="prediction", y="entropy", s=1, alpha=0.1, color="gray", ax=ax
        )
        ax.set_ylabel("Entropy")
        ax.set_xlabel("Class")
        ax.set_title("Per-class distribution of entropy")
        sns.despine()
        plt.tight_layout()
        plt.savefig(ent, dpi=150, bbox_inches="tight")
        plt.close()

        # Calculate per-class distribution of energy
        if "energy" in df.columns:
            ax = sns.violinplot(df, x="prediction", y="energy")
            sns.stripplot(
                df, x="prediction", y="energy", s=1, alpha=0.1, color="gray", ax=ax
            )
            ax.set_ylabel("Energy")
            ax.set_xlabel("Class")
            ax.set_title("Per-class distribution of Energy")
            sns.despine()
            plt.tight_layout()
            plt.savefig(eng, dpi=150, bbox_inches="tight")
            plt.close()

        # Calculate perclass score distributions
        # Create the grid
        df_long = pd.melt(
            df[
                ["contig_id", "length", "prediction"]
                + [
                    i
                    for i in df.columns
                    if i.endswith("_score") and i != "reliability_score"
                ]
            ],
            id_vars=["contig_id", "length", "prediction"],
            var_name="score_class",
            value_name="scores",
        )
        g = sns.FacetGrid(
            df_long,
            row="prediction",
            hue="score_class",
            margin_titles=False,
            height=2,
            aspect=3.5,
        )
        g.map(
            sns.kdeplot,
            "scores",
            fill=True,
            common_norm=False,
            alpha=0.2,
            linewidth=0.5,
        )
        g.add_legend()
        # Add titles and adjust layout
        g.set_axis_labels("Score", "Density")
        # g.set_titles("Per-class score distributions")
        g.savefig(clscores, dpi=150, bbox_inches="tight")
        plt.close()
        try:
            # quantile bins
            bins = pd.qcut(df["length"], q=5)

            # Extract bin edges
            bin_edges = bins.cat.categories

            # Create labels with numeric min–max
            labels = [
                f"{int(interval.left):,}–{int(interval.right):,}"
                for interval in bin_edges
            ]

            # Recreate qcut with readable labels
            df["length_bin"] = pd.qcut(df["length"], q=5, labels=labels)
            # Calculate per-class distribution of reliability scores
            ax = sns.violinplot(df, x="length_bin", y="reliability_score")
            sns.stripplot(
                df,
                x="length_bin",
                y="reliability_score",
                s=1,
                alpha=0.1,
                color="red",
                ax=ax,
            )
            ax.set_ylabel("Reliability score")
            ax.set_xlabel("Length range")
            ax.set_title("Length-wise (quantile) distribution of reliability scores")
            plt.xticks(rotation=45)
            sns.despine()
            plt.tight_layout()
            plt.savefig(relscore_len, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(e)
            logger.warning("Length-wise (quantile) plot was not created")

    # perform welch t-tests to check if there is a statistically significant difference
    # between the top-k classes
    mean_scores = df[
        [i for i in df.columns if i.endswith("_score") and "reliability" not in i]
    ].to_numpy()
    var_scores = df[[i for i in df.columns if i.endswith("_var")]].to_numpy()
    windows = (
        df[[i for i in df.columns if i.endswith("_windows") and "reliability" not in i]]
        .to_numpy()
        .sum(axis=-1)
    )
    rows = np.arange(mean_scores.shape[0])[:, None]
    sorted_indices = np.flip(np.argsort(mean_scores, axis=-1), axis=-1)
    sorted_means = mean_scores[rows, sorted_indices[:, :2]]
    sorted_vars = var_scores[rows, sorted_indices[:, :2]]
    pvals = []
    for means, vars, n in zip(sorted_means, sorted_vars, windows):
        _, _, p = welch_t_one_tailed(
            mean1=means[0], var1=vars[0], mean2=means[1], var2=vars[1], n1=n, n2=n
        )
        pvals.append(p)
    df["pval"] = pvals

    df.to_csv(tsv_with_pvals, index=None, sep="\t", float_format="%.3f")
    # Create the count plot

    if len(df) > 1:
        df["above_pval_threshold"] = df["pval"].apply(
            lambda x: "passed" if x <= 0.05 else "failed"
        )
        ax = sns.countplot(
            data=df,
            x="prediction",
            hue="above_pval_threshold",
            palette="pastel",
            stat="percent",
        )
        # Annotate bars with percentage values (already in percent)
        for p in ax.patches:
            percentage = p.get_height()
            if percentage > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_height(),
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        # Style tweaks
        ax.set_ylabel("Percentage")
        ax.set_xlabel("Prediction")
        ax.set_title("Class Distribution (%)")
        sns.despine()
        plt.tight_layout()
        plt.savefig(pct_class_pval, dpi=150, bbox_inches="tight")
        plt.close()


# =============================================================================
# Data optimization / format conversion
# =============================================================================


def optimize_data_core(
    input_path: str,
    output_path: str,
    format: str,
    crop_size: tuple[int, ...] = (500,),
    stride: int = 0,
    strides: list[int] | None = None,
    num_classes: int = 3,
    num_workers: int | None = None,
    one_hot: bool = False,
    pad_int: int = 0,
    codon_map: str = "codon_id",
    nucleotide_map: str | None = None,
    compress: str = "default",
    dtype: str = "auto",
    max_length: int = 5000,  # deprecated, ignored
    max_memory_mb: int | None = None,
    pad: bool = False,
    units: str | None = None,
    crop_units: str | None = None,
    overlap: float | None = None,
    balance_classes: bool = False,
    shuffle_seed: int = 42,
    target_num_shards: int | None = None,
):
    """Convert Jaeger CSV training data to an optimized ``.npz`` format.

    This is a thin wrapper around :func:`jaeger.dataops.convert.convert_dataset`.
    See that function for details on supported formats and output contents.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (label,sequence format).
    output_path : str
        Path to output ``.npz`` file.
    format : str
        One of ``nucleotide``, ``translated``, or ``both``.
    crop_size : tuple[int, ...]
        Sequence crop size(s) (default: ``(500,)``).
    stride : int, optional
        Sliding-window stride applied to every crop size (default: 0).
    strides : list[int] | None, optional
        Per-crop-size strides. If given, overrides ``stride``.
    num_classes : int, optional
        Number of classes (default: 3).
    num_workers : int | None, optional
        Number of parallel workers. ``None`` processes in a single worker.
    one_hot : bool, optional
        Encode nucleotide crops as one-hot float tensors (default: False).
    pad_int : int, optional
        Integer padding value for nucleotide crops (default: 0).
    codon_map : str, optional
        Codon map name (default: ``codon_id``).
    nucleotide_map : str | None, optional
        JSON string with mappings for ``A``, ``C``, ``G``, ``T``, ``N``.
    compress : str, optional
        Compression mode for the output archive (default: ``default``).
    dtype : str, optional
        Integer dtype for encoded features: ``int8``, ``uint8``, ``int16``,
        ``int32``, or ``auto``. ``auto`` selects the smallest dtype that fits
        the vocabulary (default: ``auto``).
    max_memory_mb : int | None, optional
        Memory budget in MB for encoded output buffers. ``None`` uses ~75% of
        available RAM. ``0`` disables streaming.
    pad : bool, optional
        If True, pad all crops to the global maximum length. Default is False.
    max_length : int, optional
        Deprecated and ignored. Kept for backward compatibility.
    units : str, optional
        Deprecated alias for ``crop_units``. Accepts ``nuc`` (nucleotides) or
        ``codon``. ``crop_units`` takes precedence when both are given.
    crop_units : str, optional
        Units for ``crop_size`` and ``stride``: ``codon`` (default; crop sizes
        convert to nucleotides via ``3*codons + 5``, strides scale by 3) or
        ``nucleotide``. When neither this nor ``units`` is given, the legacy
        ``nuc`` default is preserved for backward compatibility.
    overlap : float | None, optional
        Overlap between crops as a fraction of each crop size (0.0-1.0).
        If provided, per-crop strides are computed from the (unit-converted)
        crop sizes and ``stride`` is ignored.
    balance_classes : bool, optional
        If True, deal every class round-robin across output shards and
        interleave classes within each shard (default: False).
    shuffle_seed : int, optional
        Seed for the within-class shuffle used when ``balance_classes`` is
        enabled (default: 42).
    """
    # Resolve the crop unit. ``crop_units`` is the canonical flag; the legacy
    # ``units`` (``nuc``/``codon``) is still accepted for backward compatibility.
    if crop_units is not None:
        resolved_units = crop_units.lower()
        if resolved_units not in {"codon", "nucleotide"}:
            raise ValueError("crop_units must be 'codon' or 'nucleotide'")
    elif units is not None:
        legacy = units.lower()
        if legacy not in {"nuc", "codon"}:
            raise ValueError("units must be 'nuc' or 'codon'")
        resolved_units = "nucleotide" if legacy == "nuc" else "codon"
    else:
        # Preserve the historical nucleotide default.
        resolved_units = "nucleotide"

    if resolved_units == "codon":
        from jaeger.seqops.crop import codons_to_nucleotides

        # Codon crops must land on the mod-2 branch so both frame extractors
        # agree: nucleotide length = 3*codons + 5. Stride is a shift, so it
        # scales by 3 without the +5 window offset.
        crop_size = tuple(codons_to_nucleotides(cs) for cs in crop_size)
        stride = stride * 3
        if strides is not None:
            strides = [s * 3 for s in strides]

    if strides is None and overlap is not None:
        strides = [int(cs * (1 - overlap)) for cs in crop_size]

    convert_dataset(
        input_path=input_path,
        output_path=output_path,
        format=format,
        crop_size=crop_size,
        stride=stride,
        strides=strides,
        num_classes=num_classes,
        num_workers=num_workers,
        one_hot=one_hot,
        pad_int=pad_int,
        codon_map=codon_map,
        nucleotide_map=nucleotide_map,
        compress=compress,
        dtype=dtype,
        max_length=max_length,
        max_memory_mb=max_memory_mb,
        pad=pad,
        balance_classes=balance_classes,
        shuffle_seed=shuffle_seed,
        target_num_shards=target_num_shards,
    )


def combine_reliability_data_core(**kwargs):
    """Combine an ID NPZ and an OOD NPZ into a single reliability NPZ."""
    from jaeger.dataops.reliability_generator import _combine_npz_files

    id_path = kwargs.get("id_npz")
    ood_path = kwargs.get("ood_npz")
    output_path = kwargs.get("output")
    shuffle_seed = kwargs.get("shuffle_seed")
    balance_ratio = kwargs.get("balance_ratio")

    _combine_npz_files(
        id_path,
        ood_path,
        output_path,
        shuffle_seed=shuffle_seed,
        id_ood_ratio=balance_ratio,
    )
    logger.info(f"Wrote combined reliability NPZ to {output_path}")


def inspect_npz_core(**kwargs):
    """Print a human-readable summary of a Jaeger NPZ dataset.

    Handles both sharded (streaming converter with ``_jaeger_manifest``) and
    non-sharded NPZ files. Reports keys, shapes, dtypes, label distribution,
    and crop/format metadata without materialising large feature arrays.
    """
    import json

    import numpy as np

    path = kwargs.get("input")
    show_labels = kwargs.get("labels", True)

    with np.load(path, allow_pickle=True) as data:
        files = data.files
        is_sharded = "_jaeger_manifest" in files

        print(f"File: {path}")
        print(f"Sharded: {is_sharded}")

        if is_sharded:
            manifest = json.loads(str(data["_jaeger_manifest"].item()))
            num_shards = int(manifest["num_shards"])
            keys = list(manifest["keys"])
            print(f"Shards: {num_shards}")
            print(f"Format: {manifest.get('format')}")
            print(f"Crop sizes: {manifest.get('crop_sizes')}")
            print(f"Strides: {manifest.get('strides')}")
            print(f"Padded: {manifest.get('padded')}")
            print(f"One-hot: {manifest.get('one_hot')}")
            print(f"Num classes: {manifest.get('num_classes')}")
            print(f"Codon map: {manifest.get('codon_map')}")
            print(f"Nucleotide map: {manifest.get('nucleotide_map')}")

            total = 0
            label_counts: dict[int, int] = {}
            feature_shapes: dict[str, tuple] = {}
            for shard_idx in range(num_shards):
                labels = data[f"labels_{shard_idx:05d}"]
                total += len(labels)
                if show_labels:
                    vals, counts = np.unique(labels, return_counts=True)
                    for v, c in zip(vals.tolist(), counts.tolist()):
                        label_counts[int(v)] = label_counts.get(int(v), 0) + int(c)
                if shard_idx == 0:
                    for key in keys:
                        arr = data[f"{key}_{shard_idx:05d}"]
                        feature_shapes[key] = (arr.shape, str(arr.dtype))
            print(f"Total samples: {total}")
            print("Per-shard-0 array shapes/dtypes:")
            for key in keys:
                shape, dtype = feature_shapes[key]
                print(f"  {key}: shape={shape} dtype={dtype}")
            if show_labels:
                print(f"Label distribution: {dict(sorted(label_counts.items()))}")
        else:
            print(f"Keys: {files}")
            total = None
            for key in files:
                arr = data[key]
                if key.startswith("_"):
                    continue
                if isinstance(arr, np.ndarray):
                    print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
                if key == "labels":
                    total = len(arr)
                    if show_labels:
                        vals, counts = np.unique(arr, return_counts=True)
                        dist = dict(
                            zip(
                                [int(v) for v in vals.tolist()],
                                [int(c) for c in counts.tolist()],
                            )
                        )
                        print(f"  Label distribution: {dict(sorted(dist.items()))}")
            if total is not None:
                print(f"Total samples: {total}")
