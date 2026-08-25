"""Unit tests for prophage boundary refinement with pyrodigal-gv."""

import random
from pathlib import Path

import numpy as np
import pytest

from jaeger.postprocess.prophage_boundaries import (
    filter_low_gene_density_regions,
    find_genes,
    refine_boundary,
    refine_prophage_boundaries,
    refine_region,
)


def test_refine_boundary_keeps_intergenic_left():
    genes = [(100, 200), (300, 400)]
    assert refine_boundary(50, genes, "left") == 50


def test_refine_boundary_keeps_intergenic_right():
    genes = [(100, 200), (300, 400)]
    assert refine_boundary(250, genes, "right") == 250


def test_refine_boundary_left_inside_gene_extends_to_gene_start():
    genes = [(100, 200)]
    assert refine_boundary(150, genes, "left") == 100


def test_refine_boundary_right_inside_gene_extends_to_gene_end():
    genes = [(100, 200)]
    assert refine_boundary(150, genes, "right") == 200


def test_refine_boundary_caps_left_extension():
    genes = [(0, 1000)]
    assert refine_boundary(900, genes, "left", max_extension=50) == 850


def test_refine_boundary_caps_right_extension():
    genes = [(0, 1000)]
    assert refine_boundary(100, genes, "right", max_extension=50) == 150


def test_refine_region_snaps_both_boundaries():
    genes = [(100, 200), (500, 600)]
    assert refine_region(150, 550, genes) == (100, 600)


def test_refine_region_keeps_intergenic_boundaries():
    genes = [(100, 200), (500, 600)]
    assert refine_region(250, 700, genes) == (250, 700)


def test_find_genes_returns_sorted_half_open_intervals_within_sequence():
    # A short random-ish sequence unlikely to contain real genes, but the
    # function should still return sane coordinates.
    seq = "ATG" + "ACGT" * 50 + "TAA"
    genes = find_genes(seq)
    assert isinstance(genes, list)
    for start, end in genes:
        assert 0 <= start < end <= len(seq)
    assert genes == sorted(genes)


def test_refine_boundary_rejects_invalid_side():
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        refine_boundary(50, [(0, 100)], "upstream")


def _make_orf_sequence(num_codons: int = 100, seed: int = 42) -> str:
    """Return a short sequence containing one clean ORF."""
    rng = random.Random(seed)
    stop_codons = {"TAA", "TAG", "TGA"}
    codons = [
        "".join(bases)
        for bases in __import__("itertools").product("ACGT", repeat=3)
        if "".join(bases) not in stop_codons
    ]
    internal = "".join(rng.choice(codons) for _ in range(num_codons))
    return "ATG" + internal + "TAA"


def test_refine_prophage_boundaries_snaps_to_predicted_orf(tmp_path: Path):
    seq = _make_orf_sequence(num_codons=100)
    genes = find_genes(seq)
    assert len(genes) >= 1, "pyrodigal-gv should predict the synthetic ORF"

    orf_start, orf_end = min(genes, key=lambda g: g[0])
    fasta = tmp_path / "test.fa"
    fasta.write_text(f">contig1\n{seq}\n")

    prophage_cordinates = {
        "contig1": [
            np.array([[orf_start + 5, orf_end - 5]]),
            np.array([1.0]),
        ]
    }

    refined = refine_prophage_boundaries(
        prophage_cordinates=prophage_cordinates,
        fasta_path=fasta,
        fsize=1,
        max_extension=1000,
    )

    assert refined == {"contig1": [(orf_start + 5, orf_end - 5, orf_start, orf_end)]}


def test_refine_prophage_boundaries_uses_stride(tmp_path: Path):
    # stop-codon-saturated sequence: no ORFs, so refined == raw boundaries
    seq = ("TAA" * 3400)[:10000]
    fasta = tmp_path / "test.fa"
    fasta.write_text(f">contig1\n{seq}\n")

    prophage_cordinates = {"contig1": [np.array([[2, 5]]), np.array([1.0])]}
    refined = refine_prophage_boundaries(
        prophage_cordinates=prophage_cordinates,
        fasta_path=fasta,
        fsize=2000,
        stride=1500,
    )

    # raw span: [2*1500, (5-1)*1500 + 2000] = [3000, 8000]
    raw_start, raw_end, _, _ = refined["contig1"][0]
    assert (raw_start, raw_end) == (3000, 8000)


def test_prophage_report_skips_degenerate_region(tmp_path: Path):
    # zero-width region (fsize=1, single window): boundary slices are empty
    # and parasail would reject them; the report must skip the alignment
    from jaeger.postprocess.prophages import prophage_report

    seq = "A" * 500_001  # report only processes contigs > 500 kb
    fasta = tmp_path / "test.fa"
    fasta.write_text(f">contig1\n{seq}\n")

    prophage_cordinates = {"contig1": [np.array([[0, 1]]), np.array([2.0])]}
    prophage_report(
        fsize=1,
        filehandle=str(fasta),
        prophage_cordinates=prophage_cordinates,
        outdir=tmp_path,
        stride=1,
    )

    tsv = tmp_path / "prophages_jaeger.tsv"
    assert tsv.exists()
    import pandas as pd

    df = pd.read_csv(tsv, sep="\t")
    assert len(df) == 1
    assert df["raw_start"].iloc[0] == 0
    assert df["raw_end"].iloc[0] == 1


def _make_orf_zone_sequence() -> str:
    """ORF-rich zone followed by a gene-free desert (``CTAG`` repeats carry
    stop codons in every frame on both strands, so no ORFs are predicted)."""
    return _make_orf_sequence(num_codons=100) * 5 + "CTAG" * 1500


def test_filter_low_gene_density_drops_sparse_regions(tmp_path: Path):
    seq = _make_orf_zone_sequence()
    fasta = tmp_path / "test.fa"
    fasta.write_text(f">contig1\n{seq}\n")

    # region A covers the ORF zone [0, 2000), region B the desert [5000, 7000)
    prophage_cordinates = {
        "contig1": [np.array([[0, 4], [10, 14]]), np.array([1.0, 2.0])]
    }
    filtered = filter_low_gene_density_regions(
        prophage_cordinates=prophage_cordinates,
        fasta_path=fasta,
        fsize=500,
        stride=500,
        min_genome_length=0,
    )

    cords, scores = filtered["contig1"]
    assert len(cords) == 1
    # the kept region must be the one spanning the ORF zone
    assert [int(cords[0][0]), int(cords[0][1])] == [0, 4]
    assert scores[0] == pytest.approx(1.0)


def test_filter_low_gene_density_skips_short_contigs(tmp_path: Path):
    seq = _make_orf_zone_sequence()
    fasta = tmp_path / "test.fa"
    fasta.write_text(f">contig1\n{seq}\n")

    prophage_cordinates = {
        "contig1": [np.array([[0, 4], [10, 14]]), np.array([1.0, 2.0])]
    }
    filtered = filter_low_gene_density_regions(
        prophage_cordinates=prophage_cordinates,
        fasta_path=fasta,
        fsize=500,
        stride=500,
        # default 1 Mbp threshold: short contigs pass through unchanged
    )

    cords, scores = filtered["contig1"]
    assert len(cords) == 2
    assert len(scores) == 2


def test_filter_low_gene_density_preserves_empty_and_unmatched(tmp_path: Path):
    seq = _make_orf_zone_sequence()
    fasta = tmp_path / "test.fa"
    fasta.write_text(f">contig1\n{seq}\n>contig2\n{seq}\n")

    prophage_cordinates = {
        "contig1": [np.array([]), np.array([])],
        # contig3 is absent from the FASTA and must not appear in the output
        "contig3": [np.array([[0, 4]]), np.array([1.0])],
    }
    filtered = filter_low_gene_density_regions(
        prophage_cordinates=prophage_cordinates,
        fasta_path=fasta,
        fsize=500,
        stride=500,
        min_genome_length=0,
    )

    assert filtered["contig1"] == [[], []]
    assert "contig2" not in filtered  # no prophage coordinates for it
    assert "contig3" not in filtered
