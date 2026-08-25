"""Build genome-grouped k-fold CV assignments for the frag_2000 dataset.

Pools train+val genomes from the per-group ``*_split.tsv`` files (test
genomes stay excluded as an untouched holdout), assigns whole genomes to
folds with a fragment-count-balanced greedy split within each group, and
maps every fragment row of the train/val provenance indices (and hence of
the train/val npz files, which share row order) to a fold id.

Provenance genome ids do not always match the split-table ids verbatim
(e.g. ``ACCESSION|AB000296|Viruses`` vs the dereplicated FASTA record ids
used in ``virus_split.tsv``). Ids are therefore resolved to split-table
records first (exact -> ACCESSION-stripped -> version-stripped ->
accession-to-derep-record via the eukaryotic-virus members table), and the
same resolution is used both for counting fragments per genome and for
mapping rows to folds, so the fold balance reflects all fragments.

Outputs (in --out-dir):
  fold_assignments.tsv   genome, group, orig_split, n_fragments, fold
  fold_summary.tsv       per fold x group: n_genomes, n_fragments
  fold_label_summary.tsv per fold x label: n_fragments
  row_folds_train.npy    uint8 fold id per row of train npz / train.csv
  row_folds_val.npy      uint8 fold id per row of val npz / val.csv
                         (255 = row dropped; genome not in the train+val pool)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = [
    "archaea",
    "bacteria",
    "eukfungi",
    "eukmarine",
    "eukprotozoa",
    "euksingle",
    "phages",
    "plasmids",
    "protists",
    "virus",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/data_generation/frag_2000",
    )
    ap.add_argument("--out-dir", default="cv_experiments/folds")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=24876)
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- genome -> (group, original split) ----
    frames = []
    for g in GROUPS:
        t = pd.read_csv(
            data_dir / f"{g}_split.tsv",
            sep="\t",
            header=None,
            names=["genome", "split"],
        )
        t["group"] = g
        frames.append(t)
    genomes = pd.concat(frames, ignore_index=True).drop_duplicates("genome")
    pool = genomes[genomes["split"].isin(["train", "val"])].copy()
    n_test = int((genomes["split"] == "test").sum())
    print(f"pooled train+val genomes: {len(pool)}; test genomes excluded: {n_test}")
    pool_ids = set(pool["genome"])

    # ---- id resolver: provenance genome id -> split-table genome id ----
    def _strip_version(g: str) -> str:
        return g.rsplit(".", 1)[0] if "." in g else g

    def _accession_of(token: str) -> str:
        token = token.strip()
        if token.startswith("ACCESSION|"):
            parts = token.split("|")
            token = parts[1] if len(parts) > 1 else token
        return _strip_version(token)

    bare_to_genome: dict[str, str] = {}
    for g in pool_ids:
        bare_to_genome.setdefault(g, g)
        bare_to_genome.setdefault(_strip_version(g), g)

    members_tsv = (
        data_dir.parent
        / "Eukaryotic_viruses"
        / "evdb_and_virushostdb_eukvir_derep.fna.members.tsv"
    )
    acc_to_rep: dict[str, str] = {}
    if members_tsv.exists():
        rows = []
        with open(members_tsv) as fh:
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 3:
                    rows.append(cols)
        # Pass 1: each record's own accession is authoritative. Pass 2:
        # member accessions only fill gaps (a member can also be a record).
        for cols in rows:
            acc_to_rep[_accession_of(cols[0])] = cols[0]
        for cols in rows:
            for token in cols[2].split(","):
                acc_to_rep.setdefault(_accession_of(token), cols[0])
        print(
            f"loaded {len(acc_to_rep)} accession->record mappings "
            f"from {members_tsv.name}"
        )

    def resolve(g: object) -> object:
        g = str(g)
        if g in pool_ids:
            return g
        if g.startswith("ACCESSION|"):
            parts = g.split("|")
            if len(parts) >= 2 and parts[1] in pool_ids:
                return parts[1]
        hit = bare_to_genome.get(g) or bare_to_genome.get(_strip_version(g))
        if hit is not None:
            return hit
        rep = acc_to_rep.get(_accession_of(g))
        if rep in pool_ids:
            return rep
        return None

    # ---- fragment rows, resolved to split-table genome ids ----
    prov: dict[str, pd.DataFrame] = {}
    resolved: dict[str, pd.Series] = {}
    frag_counts = pd.Series(dtype=np.float64)
    for split in ("train", "val"):
        p = pd.read_csv(
            data_dir / f"provenance_index.{split}.csv",
            header=None,
            names=["label", "genome", "start", "length"],
            usecols=["label", "genome"],
            dtype={"label": np.int64, "genome": str},
            low_memory=False,
        )
        prov[split] = p
        r = p["genome"].map(resolve)
        resolved[split] = r
        frag_counts = frag_counts.add(r.value_counts(), fill_value=0)
        n_unres = int(r.isna().sum())
        print(
            f"{split}: {len(p)} fragment rows from {p['genome'].nunique()} genomes; "
            f"{n_unres} rows unresolved"
        )

    # ---- balanced greedy fold assignment within each group ----
    assign: dict[str, int] = {}
    summary_rows = []
    for g, sub in pool.groupby("group"):
        counts = frag_counts.reindex(sub["genome"].to_numpy()).fillna(0).to_numpy()
        jitter = rng.random(len(sub)) * 1e-9
        order = np.argsort(-(counts + jitter), kind="stable")
        folds = np.zeros(len(sub), dtype=np.int64)
        totals = np.zeros(args.n_folds)
        genome_arr = sub["genome"].to_numpy()
        for i in order:
            f = int(np.argmin(totals))
            folds[i] = f
            totals[f] += counts[i]
        for genome, f, c in zip(genome_arr, folds, counts):
            assign[genome] = int(f)
            summary_rows.append((genome, g, int(c), int(f)))

    assignments = pd.DataFrame(
        summary_rows, columns=["genome", "group", "n_fragments", "fold"]
    )
    assignments = assignments.merge(
        pool[["genome", "split"]], on="genome", how="left"
    ).rename(columns={"split": "orig_split"})
    assignments.to_csv(out_dir / "fold_assignments.tsv", sep="\t", index=False)

    # ---- map fragment rows to folds ----
    label_rows = []
    for split in ("train", "val"):
        rf = resolved[split].map(assign)
        n_missing = int(rf.isna().sum())
        if n_missing:
            missing = prov[split].loc[rf.isna(), "genome"].unique()
            print(
                f"WARNING: dropping {n_missing} {split} rows whose genomes are not "
                f"in the train+val pool (fold id 255); {len(missing)} distinct "
                f"genomes, e.g. {list(missing[:5])}"
            )
        rf = rf.fillna(255).to_numpy().astype(np.uint8)
        np.save(out_dir / f"row_folds_{split}.npy", rf)
        for f in range(args.n_folds):
            counts = prov[split].loc[rf == f, "label"].value_counts()
            for label, c in counts.items():
                label_rows.append((split, f, int(label), int(c)))
        print(f"{split}: rows per fold {np.bincount(rf, minlength=256)[: args.n_folds]}")

    summary = (
        assignments.groupby(["fold", "group"])
        .agg(n_genomes=("genome", "count"), n_fragments=("n_fragments", "sum"))
        .reset_index()
    )
    summary.to_csv(out_dir / "fold_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        label_rows, columns=["split", "fold", "label", "n_fragments"]
    ).to_csv(out_dir / "fold_label_summary.tsv", sep="\t", index=False)

    pivot = summary.pivot_table(
        index="group", columns="fold", values="n_fragments", fill_value=0
    )
    print("\nfragments per group x fold:")
    print(pivot.to_string())
    print(f"\nwrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
