"""GenBank input parsing for Jaeger prophage prediction.

Extracts sequences and feature annotations from GenBank files so that
prophage prediction can use .gb/.gbk input in addition to FASTA.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from Bio import SeqIO

logger = logging.getLogger("jaeger")


# Phage hallmark genes for false positive filtering
PHAGE_HALLMARK_GENES = [
    "integrase",
    "terminase",
    "capsid",
    "tail",
    "portal",
    "protease",
    "lysis",
    "holin",
    "endolysin",
    "repressor",
    "cro",
    "excisionase",
    "phage",
    "prophage",
]


def _is_phage_hallmark(product: str) -> bool:
    """Check if a gene product is a phage hallmark gene."""
    product_lower = product.lower()
    return any(hallmark in product_lower for hallmark in PHAGE_HALLMARK_GENES)


def parse_genbank(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse a GenBank file and return per-contig annotations.

    Args:
        path: Path to a .gb/.gbk file.

    Returns:
        Mapping from contig ID to a dict with keys:
        - ``sequence``: nucleotide sequence string
        - ``cds``: list of CDS feature dicts with ``start``, ``end``,
          ``strand``, ``product``, ``label``
        - ``trna``: list of tRNA feature dicts with ``start``, ``end``,
          ``strand``, ``type``
    """
    path = Path(path)
    annotations: dict[str, dict[str, Any]] = {}

    for record in SeqIO.parse(path, "genbank"):
        contig_id = record.id
        seq = str(record.seq)
        cds_features = []
        trna_features = []

        for feature in record.features:
            start = int(feature.location.start)
            end = int(feature.location.end)
            strand = 1 if feature.location.strand == 1 else -1
            qualifiers = feature.qualifiers

            if feature.type == "CDS":
                product = qualifiers.get("product", [""])[0]
                label = (
                    qualifiers.get("label", [""])[0]
                    or qualifiers.get("gene", [""])[0]
                    or qualifiers.get("locus_tag", [""])[0]
                    or product
                )
                cds_features.append(
                    {
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "product": product,
                        "label": label,
                        "locus_tag": qualifiers.get("locus_tag", [""])[0],
                        "protein_id": qualifiers.get("protein_id", [""])[0],
                        "inference": qualifiers.get("inference", [""])[0],
                        "note": qualifiers.get("note", [""])[0],
                        "gene": qualifiers.get("gene", [""])[0],
                    }
                )
            elif feature.type == "tRNA":
                trna_type = qualifiers.get("product", ["unknown"])[0].replace(
                    "tRNA-", ""
                )
                trna_features.append(
                    {
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "type": trna_type,
                        "anticodon": qualifiers.get("anticodon", [""])[0],
                    }
                )

        annotations[contig_id] = {
            "sequence": seq,
            "cds": cds_features,
            "trna": trna_features,
        }

    logger.info(
        f"parsed {len(annotations)} contigs from {path} "
        f"(CDS: {sum(len(a['cds']) for a in annotations.values())}, "
        f"tRNA: {sum(len(a['trna']) for a in annotations.values())})"
    )
    return annotations


def genbank_to_fasta_and_annotations(
    genbank_path: str | Path, fasta_path: str | Path
) -> dict[str, dict[str, Any]]:
    """Convert a GenBank file to FASTA and return annotations.

    Args:
        genbank_path: Path to the input .gb/.gbk file.
        fasta_path: Path where the FASTA file will be written.

    Returns:
        Same dict as :func:`parse_genbank`.
    """
    genbank_path = Path(genbank_path)
    fasta_path = Path(fasta_path)
    annotations = parse_genbank(genbank_path)

    with open(fasta_path, "w") as fh:
        for contig_id, ann in annotations.items():
            seq = ann["sequence"]
            fh.write(f">{contig_id}\n")
            for i in range(0, len(seq), 70):
                fh.write(seq[i : i + 70] + "\n")

    logger.info(f"converted {genbank_path} to FASTA: {fasta_path}")
    return annotations
