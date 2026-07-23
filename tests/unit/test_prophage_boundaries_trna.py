"""Unit tests for tRNA-aware prophage boundary refinement."""

from __future__ import annotations

from jaeger.postprocess.prophage_boundaries import (
    refine_boundary_with_trna,
    refine_region_with_trna,
)


def test_refine_boundary_with_trna_left_upstream() -> None:
    # tRNA upstream of left boundary, within max_extension
    trnas = [(100, 200, 1, "Ala")]
    refined, trna_type = refine_boundary_with_trna(
        300, trnas, "left", max_extension=150
    )
    assert refined == 200
    assert trna_type == "Ala"


def test_refine_boundary_with_trna_left_inside() -> None:
    # Boundary inside a tRNA -> snap to tRNA start (keeps tRNA upstream)
    trnas = [(100, 200, 1, "Ala")]
    refined, trna_type = refine_boundary_with_trna(
        150, trnas, "left", max_extension=100
    )
    assert refined == 100
    assert trna_type == "Ala"


def test_refine_boundary_with_trna_left_too_far() -> None:
    # tRNA upstream but beyond max_extension
    trnas = [(100, 200, 1, "Ala")]
    refined, trna_type = refine_boundary_with_trna(
        400, trnas, "left", max_extension=150
    )
    assert refined == 400
    assert trna_type is None


def test_refine_boundary_with_trna_right_downstream() -> None:
    # tRNA downstream of right boundary, within max_extension
    trnas = [(500, 600, 1, "Ile")]
    refined, trna_type = refine_boundary_with_trna(
        400, trnas, "right", max_extension=150
    )
    assert refined == 500
    assert trna_type == "Ile"


def test_refine_boundary_with_trna_right_inside() -> None:
    # Boundary inside a tRNA -> snap to tRNA end (keeps tRNA downstream)
    trnas = [(500, 600, 1, "Ile")]
    refined, trna_type = refine_boundary_with_trna(
        550, trnas, "right", max_extension=100
    )
    assert refined == 600
    assert trna_type == "Ile"


def test_refine_boundary_with_trna_right_too_far() -> None:
    # tRNA downstream but beyond max_extension
    trnas = [(700, 800, 1, "Ile")]
    refined, trna_type = refine_boundary_with_trna(
        500, trnas, "right", max_extension=150
    )
    assert refined == 500
    assert trna_type is None


def test_refine_region_with_trna() -> None:
    trnas = [(100, 200, 1, "Ala"), (500, 600, 1, "Ile")]
    refined_start, refined_end, left_type, right_type = refine_region_with_trna(
        300, 400, trnas, max_extension=150
    )
    assert refined_start == 200
    assert refined_end == 500
    assert left_type == "Ala"
    assert right_type == "Ile"


def test_refine_region_with_trna_no_trnas() -> None:
    refined_start, refined_end, left_type, right_type = refine_region_with_trna(
        300, 400, [], max_extension=150
    )
    assert refined_start == 300
    assert refined_end == 400
    assert left_type is None
    assert right_type is None
