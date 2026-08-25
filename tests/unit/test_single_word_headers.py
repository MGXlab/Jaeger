"""Regression tests for the single-word FASTA header convention.

Contigs are keyed by the first whitespace-delimited token of the FASTA header
(e.g. ``contig_1``), matching the legacy pipeline. ``fragment_generator`` must
emit that key, every downstream FASTA re-read must look up the same key, and
duplicate keys must be rejected at validation time instead of being silently
renamed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jaeger.postprocess.prophage_boundaries import refine_prophage_boundaries
from jaeger.postprocess.prophages import prophage_report
from jaeger.seqops.io import fragment_generator
from jaeger.utils.fs import validate_fasta_entries

RAW_HEADER = "contig_1 some description, with comma"
SHORT_KEY = "contig_1"


def _write_fasta(tmp_path: Path, seq: str, header: str = RAW_HEADER) -> Path:
    fasta = tmp_path / "test.fa"
    fasta.write_text(f">{header}\n{seq}\n")
    return fasta


def test_fragment_generator_uses_single_word_headers(tmp_path: Path):
    fasta = _write_fasta(tmp_path, ("ACGT" * 600)[:2000])
    records = list(
        fragment_generator(
            file_path=str(fasta), fragsize=200, stride=100, dustmask=False
        )
    )
    assert records
    headers = {r.split(",")[1] for r in records}
    assert headers == {SHORT_KEY}


def test_validate_fasta_entries_rejects_duplicate_headers(tmp_path: Path):
    # distinct full descriptions, same single-word key
    fasta = tmp_path / "dup.fa"
    fasta.write_text(
        ">contig_1 some description\n" + "ACGT" * 600 + "\n"
        ">contig_1 another description\n" + "TGCA" * 600 + "\n"
    )
    with pytest.raises(Exception, match="[Dd]uplicate"):
        validate_fasta_entries(str(fasta), min_len=200)


def test_validate_fasta_entries_rejects_identical_headers(tmp_path: Path):
    fasta = tmp_path / "dup.fa"
    fasta.write_text(
        ">contig_1 some description\n" + "ACGT" * 600 + "\n"
        ">contig_1 some description\n" + "TGCA" * 600 + "\n"
    )
    with pytest.raises(Exception, match="[Dd]uplicate"):
        validate_fasta_entries(str(fasta), min_len=200)


def test_validate_fasta_entries_accepts_unique_headers(tmp_path: Path):
    fasta = tmp_path / "ok.fa"
    fasta.write_text(
        ">contig_1 some description\n" + "ACGT" * 600 + "\n"
        ">contig_2 some description\n" + "TGCA" * 600 + "\n"
    )
    assert validate_fasta_entries(str(fasta), min_len=200) == 2


def test_prophage_report_writes_tsv_for_multiword_headers(tmp_path: Path):
    """End-to-end header matching: keys produced by ``fragment_generator``
    for a multi-word-header FASTA must be found by ``prophage_report``."""
    seq = ("ACGT" * 600)[:2000]
    fasta = _write_fasta(tmp_path, seq)
    records = list(
        fragment_generator(
            file_path=str(fasta), fragsize=200, stride=100, dustmask=False
        )
    )
    key = records[0].split(",")[1]

    prophage_cordinates = {key: [np.array([[1, 3]]), np.array([2.0])]}
    prophage_report(
        fsize=200,
        filehandle=str(fasta),
        prophage_cordinates=prophage_cordinates,
        outdir=tmp_path,
        stride=100,
        cutoff_length=0,
    )

    tsv = tmp_path / "prophages_jaeger.tsv"
    assert tsv.exists(), "prophage report was not written for multi-word headers"
    df = pd.read_csv(tsv, sep="\t")
    assert len(df) == 1
    assert df["contig_id"].iloc[0] == SHORT_KEY
    # raw span: [1*100, (3-1)*100 + 200] = [100, 400]
    assert df["raw_start"].iloc[0] == 100
    assert df["raw_end"].iloc[0] == 400


def test_refine_prophage_boundaries_matches_multiword_headers(tmp_path: Path):
    # stop-codon-saturated sequence: no ORFs, so refined == raw boundaries
    seq = ("TAA" * 3400)[:10000]
    fasta = _write_fasta(tmp_path, seq)

    prophage_cordinates = {SHORT_KEY: [np.array([[2, 5]]), np.array([1.0])]}
    refined = refine_prophage_boundaries(
        prophage_cordinates=prophage_cordinates,
        fasta_path=fasta,
        fsize=2000,
        stride=1500,
    )

    assert SHORT_KEY in refined
    raw_start, raw_end, _, _ = refined[SHORT_KEY][0]
    # raw span: [2*1500, (5-1)*1500 + 2000] = [3000, 8000]
    assert (raw_start, raw_end) == (3000, 8000)
