"""Refine prophage boundaries using pyrodigal-gv gene predictions.

The segmentation step in `jaeger.postprocess.prophages` produces prophage
coordinates as multiples of the sliding-window size. This module snaps those
raw coordinates to the nearest intergenic region by running `pyrodigal-gv` on
each contig, so that predicted prophage ends do not fall inside coding genes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyfastx
import pyrodigal_gv

logger = logging.getLogger("jaeger")


# Lazily instantiated metagenomic gene finder.
_FINDER: pyrodigal_gv.ViralGeneFinder | None = None


def _get_gene_finder() -> pyrodigal_gv.ViralGeneFinder:
    """Return a shared `pyrodigal-gv` gene finder instance."""
    global _FINDER
    if _FINDER is None:
        _FINDER = pyrodigal_gv.ViralGeneFinder(meta=True)
    return _FINDER


def find_genes(sequence: str) -> list[tuple[int, int]]:
    """Run pyrodigal-gv on *sequence* and return 0-based half-open gene intervals.

    Args:
        sequence: Nucleotide sequence (IUPAC alphabet is accepted; non-ATGC
            characters are ignored by pyrodigal).

    Returns:
        Sorted list of ``(start, end)`` tuples in 0-based, half-open coordinates.
    """
    finder = _get_gene_finder()
    genes = finder.find_genes(sequence)
    # pyrodigal returns 1-based closed intervals; convert to 0-based half-open.
    intervals = [(int(g.begin) - 1, int(g.end)) for g in genes]
    intervals.sort()
    return intervals


def find_genes_with_strand(sequence: str) -> list[tuple[int, int, int]]:
    """Run pyrodigal-gv on *sequence* and return gene intervals with strand.

    Args:
        sequence: Nucleotide sequence.

    Returns:
        Sorted list of ``(start, end, strand)`` tuples in 0-based half-open
        coordinates. ``strand`` is 1 for forward, -1 for reverse.
    """
    finder = _get_gene_finder()
    genes = finder.find_genes(sequence)
    # pyrodigal returns 1-based closed intervals; convert to 0-based half-open.
    intervals = [(int(g.begin) - 1, int(g.end), int(g.strand)) for g in genes]
    intervals.sort()
    return intervals


def find_trnas(sequence: str) -> list[tuple[int, int, int, str]]:
    """Predict tRNA genes with tRNAscan-SE.

    Args:
        sequence: Nucleotide sequence.

    Returns:
        Sorted list of ``(start, end, strand, type)`` tuples in 0-based
        half-open coordinates. Empty list if tRNAscan-SE is not available.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as fh:
        fh.write(f">seq\n{sequence}\n")
        fasta_path = fh.name

    try:
        result = subprocess.run(
            ["tRNAscan-SE", "-B", "-o", "-", fasta_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning(f"tRNAscan-SE failed: {result.stderr}")
            return []

        trnas = []
        for line in result.stdout.splitlines():
            # Skip header lines and empty lines
            if (
                not line
                or line.startswith("Sequence")
                or line.startswith("Name")
                or line.startswith("--------")
                or line.startswith("Status")
            ):
                continue
            parts = line.split("\t")
            if len(parts) >= 9:
                try:
                    # tRNAscan-SE output: seq_name, tRNA #, Begin, End, Type,
                    # Anticodon, Intron Begin, Intron End, Score
                    start = int(parts[2]) - 1  # 1-based to 0-based
                    end = int(parts[3])
                    strand = 1 if start < end else -1
                    trna_type = parts[4].strip()
                    trnas.append((min(start, end), max(start, end), strand, trna_type))
                except (ValueError, IndexError):
                    continue
        trnas.sort()
        return trnas
    except FileNotFoundError:
        logger.warning("tRNAscan-SE not found; skipping tRNA prediction")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("tRNAscan-SE timed out")
        return []
    finally:
        import os

        os.unlink(fasta_path)


def _is_intergenic(position: int, genes: list[tuple[int, int]]) -> bool:
    """Return True if *position* is not inside any gene interval."""
    for start, end in genes:
        if start <= position < end:
            return False
        if start > position:
            break
    return True


def refine_boundary(
    position: int,
    genes: list[tuple[int, int]],
    side: str,
    max_extension: int | None = None,
) -> int:
    """Snap a single boundary to the nearest intergenic region.

    Args:
        position: Raw boundary coordinate (0-based).
        genes: Sorted list of 0-based half-open gene intervals.
        side: ``"left"`` or ``"right"``. A left boundary is extended leftward,
            a right boundary rightward.
        max_extension: Maximum number of bases the boundary may be moved.
            If ``None``, the extension is unbounded.

    Returns:
        Refined boundary coordinate.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    if _is_intergenic(position, genes):
        return position

    containing_gene = next(
        ((start, end) for start, end in genes if start <= position < end),
        None,
    )
    if containing_gene is None:
        return position

    gene_start, gene_end = containing_gene
    refined = gene_start if side == "left" else gene_end

    if max_extension is not None:
        extension = abs(refined - position)
        if extension > max_extension:
            logger.warning(
                "Boundary refinement exceeded max_extension (%d bp); "
                "capping %s boundary.",
                max_extension,
                side,
            )
            refined = (
                position + max_extension
                if side == "right"
                else position - max_extension
            )

    return refined


def refine_region(
    raw_start: int,
    raw_end: int,
    genes: list[tuple[int, int]],
    max_extension: int | None = None,
) -> tuple[int, int]:
    """Refine both boundaries of a single prophage region.

    Args:
        raw_start: Raw left boundary (0-based).
        raw_end: Raw right boundary (0-based, exclusive).
        genes: Sorted list of 0-based half-open gene intervals.
        max_extension: Maximum extension allowed for each boundary.

    Returns:
        ``(refined_start, refined_end)``.
    """
    refined_start = refine_boundary(
        raw_start, genes, "left", max_extension=max_extension
    )
    refined_end = refine_boundary(raw_end, genes, "right", max_extension=max_extension)
    return refined_start, refined_end


def refine_boundary_with_trna(
    position: int,
    trnas: list[tuple[int, int, int, str]],
    side: str,
    max_extension: int,
) -> tuple[int, str | None]:
    """Snap a boundary to a tRNA edge if a tRNA is within max_extension.

    For ``side="left"``: if a tRNA is upstream and within max_extension
    (``tRNA.end <= position`` and ``position - tRNA.end <= max_extension``),
    snap to ``tRNA.end``. If the boundary is inside a tRNA
    (``tRNA.start <= position < tRNA.end``), snap to ``tRNA.start``.

    For ``side="right"``: if a tRNA is downstream and within max_extension
    (``tRNA.start >= position`` and ``tRNA.start - position <= max_extension``),
    snap to ``tRNA.start``. If the boundary is inside a tRNA, snap to
    ``tRNA.end``.

    Args:
        position: Raw boundary coordinate (0-based).
        trnas: List of ``(start, end, strand, type)`` tuples in 0-based
            half-open coordinates.
        side: ``"left"`` or ``"right"``.
        max_extension: Maximum number of bases the boundary may be moved.

    Returns:
        ``(refined_position, trna_type or None)``.
    """
    for start, end, strand, trna_type in trnas:
        if side == "left":
            # tRNA upstream and within max_extension
            if end <= position and position - end <= max_extension:
                return end, trna_type
            # boundary inside tRNA
            if start <= position < end:
                return start, trna_type
        else:  # right
            # tRNA downstream and within max_extension
            if start >= position and start - position <= max_extension:
                return start, trna_type
            # boundary inside tRNA
            if start <= position < end:
                return end, trna_type
    return position, None


def refine_region_with_trna(
    raw_start: int,
    raw_end: int,
    trnas: list[tuple[int, int, int, str]],
    max_extension: int,
) -> tuple[int, int, str | None, str | None]:
    """Refine both boundaries using tRNA features.

    Returns:
        ``(refined_start, refined_end, left_trna_type, right_trna_type)``.
    """
    refined_start, left_trna = refine_boundary_with_trna(
        raw_start, trnas, "left", max_extension
    )
    refined_end, right_trna = refine_boundary_with_trna(
        raw_end, trnas, "right", max_extension
    )
    return refined_start, refined_end, left_trna, right_trna


def refine_boundary_with_integrase_trna(
    position: int,
    genes: list[tuple[int, int, str]],
    trnas: list[tuple[int, int, int, str]],
    side: str,
    phage_scores: np.ndarray,
    host_scores: np.ndarray,
    window_size: int = 2000,
    stride: int = 1500,
    max_extension: int = 20000,
) -> tuple[int, str | None, str | None, float]:
    """Snap a boundary to the closest integrase or tRNA edge.

    Uses a confidence-based approach:
    - Integrases within 5 kb: high confidence (1.0)
    - Integrases within 10 kb: medium confidence (0.7)
    - Integrases within 20 kb: low confidence (0.4)
    - tRNAs within 5 kb: high confidence (1.0)

    Also checks if phage score drops below host score when extending.
    Only extends if phage score remains above host score.

    Args:
        position: Raw boundary coordinate (0-based).
        genes: List of ``(start, end, product)`` tuples in 0-based half-open
            coordinates.
        trnas: List of ``(start, end, strand, type)`` tuples in 0-based
            half-open coordinates.
        side: ``"left"`` or ``"right"``.
        phage_scores: Array of phage scores for each window.
        host_scores: Array of host scores for each window.
        window_size: Window size in bp.
        stride: Window stride in bp.
        max_extension: Maximum number of bases the boundary may be moved.

    Returns:
        ``(refined_position, integrase_type or None, trna_type or None, confidence)``.
    """
    integrase_type = None
    trna_type = None
    confidence = 0.0

    def _check_phage_score(new_position: int) -> bool:
        """Check if phage score is above host score at the new position."""
        window_idx = new_position // stride
        if 0 <= window_idx < len(phage_scores):
            return phage_scores[window_idx] > host_scores[window_idx]
        return False

    if side == "left":
        # Find closest integrase to the left of the boundary
        best_integrase = None
        best_distance = float("inf")
        for start, end, product in genes:
            if "integrase" in product.lower() and end <= position:
                distance = position - end
                if distance <= max_extension and distance < best_distance:
                    # Check if phage score remains above host score
                    if _check_phage_score(end):
                        best_integrase = (end, product)
                        best_distance = distance

        if best_integrase:
            end, product = best_integrase
            integrase_type = product
            # Assign confidence based on distance
            if best_distance <= 5000:
                confidence = 1.0
            elif best_distance <= 10000:
                confidence = 0.7
            elif best_distance <= 20000:
                confidence = 0.4
            else:
                confidence = 0.1
            return end, integrase_type, trna_type, confidence

        # Find closest tRNA to the right of the boundary
        for start, end, strand, trna_type in trnas:
            if start >= position and start - position <= max_extension:
                # Check if phage score remains above host score
                if _check_phage_score(start):
                    trna_type = trna_type
                    confidence = 1.0
                    return start, integrase_type, trna_type, confidence
    else:  # right
        # Find closest integrase to the right of the boundary
        best_integrase = None
        best_distance = float("inf")
        for start, end, product in genes:
            if "integrase" in product.lower() and start >= position:
                distance = start - position
                if distance <= max_extension and distance < best_distance:
                    # Check if phage score remains above host score
                    if _check_phage_score(start):
                        best_integrase = (start, product)
                        best_distance = distance

        if best_integrase:
            start, product = best_integrase
            integrase_type = product
            # Assign confidence based on distance
            if best_distance <= 5000:
                confidence = 1.0
            elif best_distance <= 10000:
                confidence = 0.7
            elif best_distance <= 20000:
                confidence = 0.4
            else:
                confidence = 0.1
            return start, integrase_type, trna_type, confidence

        # Find closest tRNA to the left of the boundary
        for start, end, strand, trna_type in trnas:
            if end <= position and position - end <= max_extension:
                # Check if phage score remains above host score
                if _check_phage_score(end):
                    trna_type = trna_type
                    confidence = 1.0
                    return end, integrase_type, trna_type, confidence

    return position, integrase_type, trna_type, confidence


def refine_region_with_integrase_trna(
    raw_start: int,
    raw_end: int,
    genes: list[tuple[int, int, str]],
    trnas: list[tuple[int, int, int, str]],
    phage_scores: np.ndarray,
    host_scores: np.ndarray,
    window_size: int = 2000,
    stride: int = 1500,
    max_extension: int = 20000,
) -> tuple[int, int, str | None, str | None, float]:
    """Refine both boundaries using integrase and tRNA features.

    Returns:
        ``(refined_start, refined_end, left_integrase, right_integrase, confidence)``.
    """
    refined_start, left_integrase, _, left_confidence = (
        refine_boundary_with_integrase_trna(
            raw_start,
            genes,
            trnas,
            "left",
            phage_scores,
            host_scores,
            window_size,
            stride,
            max_extension,
        )
    )
    refined_end, right_integrase, _, right_confidence = (
        refine_boundary_with_integrase_trna(
            raw_end,
            genes,
            trnas,
            "right",
            phage_scores,
            host_scores,
            window_size,
            stride,
            max_extension,
        )
    )
    # Overall confidence is the minimum of left and right confidence
    confidence = min(left_confidence, right_confidence)
    return refined_start, refined_end, left_integrase, right_integrase, confidence


def refine_prophage_boundaries(
    prophage_cordinates: dict[str, list[Any]],
    fasta_path: str | Path,
    fsize: int,
    max_extension: int | None = None,
    stride: int | None = None,
    trna_features: dict[str, list[tuple[int, int, int, str]]] | None = None,
    gene_features: dict[str, list[tuple[int, int, str]]] | None = None,
    phage_scores: dict[str, np.ndarray] | None = None,
    host_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Refine window-based prophage boundaries against gene coordinates.

    Args:
        prophage_cordinates: Output of ``segment()``:
            ``{contig_id: [window_index_ranges, scores]}`` where ranges is an
            ``(N, 2)`` array of window indices.
        fasta_path: Path to the input FASTA file.
        fsize: Sliding-window size used by Jaeger.
        max_extension: Maximum boundary extension allowed (default: ``2 * fsize``).
        stride: Sliding-window stride used by Jaeger (default: ``fsize``).
        trna_features: Optional mapping from contig ID to a list of
            ``(start, end, strand, type)`` tRNA tuples. When provided,
            boundaries are further snapped to tRNA edges after gene-based
            refinement.
        gene_features: Optional mapping from contig ID to a list of
            ``(start, end, product)`` gene tuples. When provided, boundaries
            are further snapped to integrase edges.
        phage_scores: Optional mapping from contig ID to phage scores array.
        host_scores: Optional mapping from contig ID to host scores array.

    Returns:
        ``{contig_id: [(raw_start, raw_end, refined_start, refined_end), ...]}``.
    """
    if max_extension is None:
        max_extension = 2 * fsize
    step = stride or fsize

    fasta_path = Path(fasta_path)
    refined: dict[str, list[tuple[int, int, int, int]]] = {}

    fa = pyfastx.Fasta(str(fasta_path), build_index=False)
    total_contigs = len(prophage_cordinates)
    processed_contigs = 0
    for record in fa:
        header = record[0].strip().replace(",", "___")
        if header not in prophage_cordinates:
            continue

        processed_contigs += 1
        if processed_contigs % 10 == 0:
            logger.info(
                f"refining boundaries: {processed_contigs}/{total_contigs} contigs"
            )

        cords, _ = prophage_cordinates[header]
        if len(cords) == 0:
            refined[header] = []
            continue

        sequence = str(record[1])
        genes = find_genes(sequence)
        trnas = trna_features.get(header, []) if trna_features else []
        genes_with_products = gene_features.get(header, []) if gene_features else []
        contig_phage_scores = (
            phage_scores.get(header, np.array([])) if phage_scores else np.array([])
        )
        contig_host_scores = (
            host_scores.get(header, np.array([])) if host_scores else np.array([])
        )

        contig_refined: list[tuple[int, int, int, int]] = []
        for start_idx, end_idx in cords:
            # region spans [first window start, last window end]
            raw_start = int(start_idx * step)
            raw_end = int((end_idx - 1) * step + fsize)
            refined_start, refined_end = refine_region(
                raw_start, raw_end, genes, max_extension=max_extension
            )
            # Further refine with integrase and tRNA features when available.
            if (
                genes_with_products
                and len(contig_phage_scores) > 0
                and len(contig_host_scores) > 0
            ):
                refined_start, refined_end, _, _, _ = refine_region_with_integrase_trna(
                    refined_start,
                    refined_end,
                    genes_with_products,
                    trnas,
                    contig_phage_scores,
                    contig_host_scores,
                    fsize,
                    step,
                    max_extension,
                )
            elif trnas:
                refined_start, refined_end, _, _ = refine_region_with_trna(
                    refined_start, refined_end, trnas, max_extension=max_extension
                )
            refined_start = max(refined_start, 0)
            refined_end = min(refined_end, len(sequence))
            contig_refined.append((raw_start, raw_end, refined_start, refined_end))
        refined[header] = contig_refined

    return refined
