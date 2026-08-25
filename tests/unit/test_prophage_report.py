"""Tests for prophage report alignment-summary edge cases."""

from __future__ import annotations

import types

from jaeger.postprocess.prophages import get_prophage_alignment_summary


def test_alignment_summary_handles_saturated_result():
    """A saturated parasail result (16-bit score overflow from a very long
    terminal repeat) must yield a summary dict, not the legacy string
    sentinel that crashed ``prophage_report``."""
    record = ("contig1", "ACGT" * 1000)
    result = types.SimpleNamespace(saturated=True)

    summary = get_prophage_alignment_summary(
        result_object=result,
        seq_len=4000,
        record=record,
        cordinates={"start": [100, 200], "end": [3800, 3900]},
        phage_score=2.0,
        type_="DTR",
    )

    assert isinstance(summary, dict)
    assert summary["att_type"] == "DTR_saturated"
    assert summary["sstart"] == 100
    assert summary["eend"] == 3800
    assert summary["region_len"] == 3700


def test_alignment_summary_without_alignment():
    record = ("contig1", "ACGT" * 1000)

    summary = get_prophage_alignment_summary(
        result_object=None,
        seq_len=4000,
        record=record,
        cordinates={"start": [100, None], "end": [3800, None]},
        phage_score=2.0,
        type_=None,
    )

    assert isinstance(summary, dict)
    assert summary["att_type"] is None
    assert summary["sstart"] == 100
    assert summary["eend"] == 3800
