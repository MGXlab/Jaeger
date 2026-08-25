"""Check per-shard label balance and within-shard ordering of a Jaeger NPZ.

Pure stdlib (no numpy) so it can run on hosts whose Python lacks a working
numpy build. A Jaeger sharded NPZ is a zip archive with entries like
``labels_00000``, ``translated_00000``, ... plus a ``_jaeger_manifest``.

Usage:
    python3 scripts/check_npz_shard_balance.py file1.npz [file2.npz ...]
"""

from __future__ import annotations

import ast
import re
import struct
import sys
import zipfile

_DTYPES = {
    "|i1": ("b", 1),
    "|u1": ("B", 1),
    "<i2": ("<h", 2),
    "<i4": ("<i", 4),
    "<i8": ("<q", 8),
    "<u4": ("<I", 4),
    "<u8": ("<Q", 8),
}


def read_npy_labels(zf: zipfile.ZipFile, name: str) -> list[int]:
    """Read a 1-D integer .npy entry without numpy."""
    with zf.open(name) as fh:
        magic = fh.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{name}: not an npy entry")
        major, _minor = fh.read(1)[0], fh.read(1)[0]
        hlen_fmt, hlen_size = ("<H", 2) if major == 1 else ("<I", 4)
        hlen = struct.unpack(hlen_fmt, fh.read(hlen_size))[0]
        header = ast.literal_eval(fh.read(hlen).decode("latin1").strip())
        descr, shape = header["descr"], header["shape"]
        if len(shape) != 1:
            raise ValueError(f"{name}: expected 1-D, got shape {shape}")
        fmt, size = _DTYPES[descr]
        raw = fh.read(shape[0] * size)
        if size == 1:
            return list(struct.unpack(f"{shape[0]}{fmt}", raw))
        return list(struct.unpack(f"{fmt[0]}{shape[0]}{fmt[1:]}", raw))


def main() -> None:
    for path in sys.argv[1:]:
        print(f"\n=== {path} ===")
        with zipfile.ZipFile(path) as zf:
            label_entries = sorted(
                n for n in zf.namelist() if re.fullmatch(r"labels_\d{5}\.npy", n)
            )
            if not label_entries:
                print("  no sharded labels_* entries found")
                continue
            tot: dict[int, int] = {}
            for entry in label_entries:
                labels = read_npy_labels(zf, entry)
                n = len(labels)
                n0 = labels.count(0)
                n1 = labels.count(1)
                other = n - n0 - n1
                half = n // 2
                f0 = labels[:half].count(1) / max(half, 1)
                f1 = labels[half:].count(1) / max(n - half, 1)
                transitions = sum(
                    1 for a, b in zip(labels, labels[1:]) if a != b
                ) / max(n - 1, 1)
                tot[0] = tot.get(0, 0) + n0
                tot[1] = tot.get(1, 0) + n1
                print(
                    f"  {entry}: n={n} ID(1)={n1} OOD(0)={n0} other={other} "
                    f"frac_id={n1 / n:.3f} first_half_id={f0:.3f} "
                    f"second_half_id={f1:.3f} transition_rate={transitions:.3f}"
                )
            total = sum(tot.values())
            print(
                f"  TOTAL: n={total} ID={tot.get(1, 0)} OOD={tot.get(0, 0)} "
                f"frac_id={tot.get(1, 0) / total:.3f}"
            )


if __name__ == "__main__":
    main()
