"""Extract Casjens 2003 prophage reference coordinates from PHASTEST evaluation table.

Downloads the PHASTEST evaluation table and parses the reference prophage
coordinates for 54 bacterial genomes from Casjens 2003.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

# PHASTEST evaluation table URL
EVAL_URL = "https://phastest.ca/more_stats_table_4.txt"


def parse_phastest_eval(text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse PHASTEST evaluation table and extract reference prophage coordinates.

    Returns:
        Dict mapping NCBI accession to list of (start, end) prophage coordinates.
    """
    genomes = {}
    current_accession = None
    in_data_section = False

    for line in text.splitlines():
        # New genome section
        if line.startswith("NCBI id:"):
            match = re.search(r"NC_\d+", line)
            if match:
                current_accession = match.group(0)
                genomes[current_accession] = []
            in_data_section = False
            continue

        # Data section starts after the dashed line
        if line.startswith("..."):
            in_data_section = True
            continue

        # Skip empty lines and headers
        if not line.strip() or line.strip().startswith("reference") or line.strip().startswith("PHAST"):
            continue

        # Parse coordinate lines
        if in_data_section and current_accession:
            # The table has a fixed-width format. The reference column is
            # the first ~20 characters. If the line starts with spaces and
            # the first coordinate is after ~20 characters, it's a false
            # positive (empty reference column). If the line starts with
            # "FALSE POSITIVES", it's also a false positive row.
            stripped = line.strip()
            if stripped.startswith("FALSE POSITIVES"):
                continue
            # Check if the first coordinate is within the reference column
            # (first ~20 characters). If not, it's a false positive.
            match = re.search(r"\d+-\d+", line)
            if not match:
                continue
            if match.start() > 15:
                continue
            # Parse the reference coordinate
            coord = match.group(0)
            start, end = coord.split("-")
            genomes[current_accession].append((int(start), int(end)))

    return genomes


def main() -> None:
    # Download evaluation table
    print("Downloading PHASTEST evaluation table...")
    with urllib.request.urlopen(EVAL_URL) as response:
        text = response.read().decode("utf-8")

    # Parse coordinates
    genomes = parse_phastest_eval(text)

    # Write to TSV
    out_path = Path("casjens2003_prophage_coordinates.tsv")
    with open(out_path, "w") as fh:
        fh.write("accession\tstart\tend\n")
        for accession, coords in sorted(genomes.items()):
            for start, end in sorted(coords):
                fh.write(f"{accession}\t{start}\t{end}\n")

    print(f"Extracted {sum(len(c) for c in genomes.values())} prophage coordinates")
    print(f"from {len(genomes)} genomes")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
