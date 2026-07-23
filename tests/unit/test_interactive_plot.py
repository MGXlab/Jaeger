"""Unit tests for interactive HTML prophage plots."""

from __future__ import annotations

import pandas as pd

from jaeger.postprocess.interactive_plot import plot_scores_html


def test_plot_scores_html_creates_file(tmp_path) -> None:
    # Build minimal logits_df
    df = pd.DataFrame(
        {
            "length": [0, 2000, 4000],
            "phage": [0.5, 2.5, 0.8],
            "gc": [0.5, 0.52, 0.48],
            "gc_skew": [0.1, -0.05, 0.02],
        }
    )
    logits_df = {"contig1": [df, "bacteria", 5000]}
    phage_cordinates = {"contig1": [[[1, 2]], [2.5]]}

    annotations = {
        "contig1": {
            "cds": [
                {
                    "start": 100,
                    "end": 500,
                    "strand": 1,
                    "product": "terminase",
                    "label": "terminase",
                },
                {
                    "start": 1000,
                    "end": 1500,
                    "strand": -1,
                    "product": "portal",
                    "label": "portal",
                },
            ],
            "trna": [
                {"start": 2000, "end": 2072, "strand": 1, "type": "Ala"},
            ],
        }
    }

    plot_scores_html(
        logits_df,
        phage_cordinates=phage_cordinates,
        annotations=annotations,
        outdir=tmp_path,
        infile_base="test",
        fsize=2000,
        stride=None,
    )

    out_file = tmp_path / "test_jaeger_contig1_interactive.html"
    assert out_file.exists()

    content = out_file.read_text()
    assert '<div id="chart">' in content
    assert '"prophages"' in content
    assert '"genes"' in content
    assert '"trnas"' in content
    assert "tRNA-Ala" in content
    assert "terminase" in content


def test_plot_scores_html_no_annotations(tmp_path) -> None:
    # Without annotations, genes/trnas lists should be empty
    df = pd.DataFrame(
        {
            "length": [0, 2000],
            "phage": [0.5, 2.5],
            "gc": [0.5, 0.52],
            "gc_skew": [0.1, -0.05],
        }
    )
    logits_df = {"contig1": [df, "bacteria", 5000]}
    phage_cordinates = {"contig1": [[[1, 2]], [2.5]]}

    plot_scores_html(
        logits_df,
        phage_cordinates=phage_cordinates,
        annotations=None,
        outdir=tmp_path,
        infile_base="test",
        fsize=2000,
        stride=None,
    )

    out_file = tmp_path / "test_jaeger_contig1_interactive.html"
    assert out_file.exists()

    content = out_file.read_text()
    assert '"genes": []' in content
    assert '"trnas": []' in content
