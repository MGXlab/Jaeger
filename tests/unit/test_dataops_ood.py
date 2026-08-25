"""Tests for jaeger.dataops.ood."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaeger.dataops import ood


def _write_id_csv(path: Path, n: int = 20, seq_len: int = 300, n_labels: int = 6):
    """Write a small ID CSV with balanced labels."""
    import random

    rng = random.Random(42)
    with open(path, "w") as fh:
        for i in range(n):
            seq = "".join(rng.choices("ACGT", k=seq_len))
            fh.write(f"{i % n_labels},{seq}\n")


def _read_ood_csv(path: Path) -> list[tuple[int, str]]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            label, seq = line.split(",", 1)
            records.append((int(label), seq))
    return records


class TestGenerateOodCsvCore:
    def test_basic_generation_count(self, tmp_path: Path):
        inp = tmp_path / "id.csv"
        _write_id_csv(inp, n=20, seq_len=300)
        out = tmp_path / "ood.csv"
        ood.generate_ood_csv_core(
            input=str(inp),
            output=str(out),
            multiplier=1.0,
            generation_workers=1,
        )
        records = _read_ood_csv(out)
        # multiplier 1.0 over 20 records -> ~20 OOD sequences
        assert len(records) == 20
        # All OOD records are labelled 0
        assert all(label == 0 for label, _ in records)

    def test_multiplier_scales_output(self, tmp_path: Path):
        inp = tmp_path / "id.csv"
        _write_id_csv(inp, n=10, seq_len=200)
        out = tmp_path / "ood.csv"
        ood.generate_ood_csv_core(
            input=str(inp),
            output=str(out),
            multiplier=2.0,
            generation_workers=1,
        )
        records = _read_ood_csv(out)
        assert len(records) == 20

    def test_shuffle_preserves_length(self, tmp_path: Path):
        inp = tmp_path / "id.csv"
        _write_id_csv(inp, n=10, seq_len=250)
        out = tmp_path / "ood.csv"
        ood.generate_ood_csv_core(
            input=str(inp),
            output=str(out),
            multiplier=1.0,
            shuffle_proportion=1.0,
            subseq_repeat=False,
            tandem_repeat=False,
            n_stretch=False,
            mix=False,
            generation_workers=1,
        )
        records = _read_ood_csv(out)
        assert len(records) == 10
        # Length-preserving perturbations keep the input length
        assert all(len(seq) == 250 for _, seq in records)

    def test_mix_respects_crop_size_codons(self, tmp_path: Path):
        inp = tmp_path / "id.csv"
        _write_id_csv(inp, n=10, seq_len=500, n_labels=3)
        out = tmp_path / "ood.csv"
        crop_codons = 100
        ood.generate_ood_csv_core(
            input=str(inp),
            output=str(out),
            multiplier=1.0,
            shuffle_proportion=0.0,
            subseq_repeat=False,
            tandem_repeat=False,
            n_stretch=False,
            mix=True,
            crop_size=crop_codons,
            crop_units="codon",
            generation_workers=1,
        )
        records = _read_ood_csv(out)
        expected_nt = 3 * crop_codons + 5
        assert all(len(seq) == expected_nt for _, seq in records)

    def test_empty_input_raises(self, tmp_path: Path):
        inp = tmp_path / "empty.csv"
        inp.write_text("")
        out = tmp_path / "ood.csv"
        with pytest.raises(ValueError, match="No records found"):
            ood.generate_ood_csv_core(
                input=str(inp),
                output=str(out),
                generation_workers=1,
            )

    def test_source_sample_size_subsamples(self, tmp_path: Path):
        inp = tmp_path / "id.csv"
        _write_id_csv(inp, n=100, seq_len=100)
        out = tmp_path / "ood.csv"
        # Subsample to 10 source records; multiplier applied to the original
        # count is not adjusted here, so output reflects the sampled pool size.
        ood.generate_ood_csv_core(
            input=str(inp),
            output=str(out),
            multiplier=1.0,
            source_sample_size=10,
            generation_workers=1,
        )
        records = _read_ood_csv(out)
        assert len(records) == 10

    def test_output_parent_dir_created(self, tmp_path: Path):
        inp = tmp_path / "id.csv"
        _write_id_csv(inp, n=5, seq_len=100)
        out = tmp_path / "nested" / "dir" / "ood.csv"
        ood.generate_ood_csv_core(
            input=str(inp),
            output=str(out),
            multiplier=1.0,
            generation_workers=1,
        )
        assert out.exists()
