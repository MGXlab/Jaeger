"""Download bacterial genomes from NCBI for Casjens 2003 validation set.

Downloads FASTA sequences for the 54 bacterial genomes used in the Casjens 2003
prophage evaluation, based on the accessions extracted from the PHASTEST
evaluation table.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

# NCBI eutils efetch URL for nucleotide FASTA
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def download_genome(accession: str, outdir: Path, retries: int = 3) -> bool:
    """Download a genome FASTA from NCBI.

    Args:
        accession: NCBI accession (e.g., "NC_000913").
        outdir: Output directory.
        retries: Number of retries on failure.

    Returns:
        True if successful, False otherwise.
    """
    out_path = outdir / f"{accession}.fna"
    if out_path.exists():
        print(f"  {accession}: already exists, skipping")
        return True

    params = f"db=nuccore&id={accession}&rettype=fasta&retmode=text"
    url = f"{EFETCH_URL}?{params}"

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read().decode("utf-8")
            if not data.startswith(">"):
                print(f"  {accession}: invalid FASTA response")
                return False
            out_path.write_text(data)
            print(f"  {accession}: downloaded ({len(data)} bytes)")
            return True
        except Exception as e:
            print(f"  {accession}: attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return False


def main() -> None:
    # Read accessions from coordinates file
    coords_path = Path("casjens2003_prophage_coordinates.tsv")
    accessions = set()
    with open(coords_path) as fh:
        next(fh)  # skip header
        for line in fh:
            accession = line.split("\t")[0]
            accessions.add(accession)

    outdir = Path("casjens2003_genomes")
    outdir.mkdir(exist_ok=True)

    print(f"Downloading {len(accessions)} genomes to {outdir}/")
    successful = 0
    failed = []

    for accession in sorted(accessions):
        if download_genome(accession, outdir):
            successful += 1
        else:
            failed.append(accession)
        time.sleep(0.5)  # NCBI rate limit: 3 requests/second

    print(f"\nDownloaded {successful}/{len(accessions)} genomes")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
