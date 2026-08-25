from __future__ import annotations

from unittest import mock


def _call(units, crop_size, stride=0, strides=None):
    from jaeger.commands import utils as u

    with mock.patch.object(u, "convert_dataset") as cd:
        u.optimize_data_core(
            input_path="in.csv",
            output_path="out.npz",
            format="translated",
            crop_size=tuple(crop_size),
            stride=stride,
            strides=strides,
            units=units,
        )
    return cd.call_args.kwargs


def test_optimize_data_codon_units_uses_canonical_nt():
    """665 codons must map to 2000 nt (3*665 + 5), not 1995 (3*665)."""
    kw = _call("codon", [665])
    assert kw["crop_size"] == (2000,)
    assert kw["stride"] == 0


def test_optimize_data_codon_units_converts_each_crop_and_stride():
    """Each codon crop -> 3c+5 nt; a codon stride is a shift -> 3*stride bp."""
    kw = _call("codon", [500, 665], stride=100)
    assert kw["crop_size"] == (1505, 2000)
    assert kw["stride"] == 300


def test_optimize_data_codon_units_with_strides_list():
    kw = _call("codon", [665], strides=[100])
    assert kw["crop_size"] == (2000,)
    assert kw["strides"] == [300]


def test_optimize_data_nuc_units_passthrough():
    kw = _call("nuc", [2000], stride=500)
    assert kw["crop_size"] == (2000,)
    assert kw["stride"] == 500


def test_optimize_data_target_shards_passthrough():
    """--target-shards must be forwarded to convert_dataset as target_num_shards."""
    from jaeger.commands import utils as u

    with mock.patch.object(u, "convert_dataset") as cd:
        u.optimize_data_core(
            input_path="in.csv",
            output_path="out.npz",
            format="translated",
            crop_size=(665,),
            crop_units="codon",
            target_num_shards=7,
        )
    assert cd.call_args.kwargs["target_num_shards"] == 7


def test_convert_dataset_target_shards_produces_expected_count(tmp_path):
    """A real small conversion yields ~target_num_shards shards in the manifest."""
    import json
    import random

    np = __import__("numpy")

    rng = random.Random(0)
    csv = tmp_path / "in.csv"
    with open(csv, "w") as fh:
        for i in range(200):
            seq = "".join(rng.choices("ACGT", k=2000))
            fh.write(f"{i % 6},{seq}\n")

    out = tmp_path / "out.npz"
    from jaeger.dataops.convert import convert_dataset

    convert_dataset(
        input_path=str(csv),
        output_path=str(out),
        format="translated",
        crop_size=665,
        num_classes=6,
        num_workers=1,
        target_num_shards=8,
    )
    with np.load(out, allow_pickle=True) as data:
        assert "_jaeger_manifest" in data.files
        manifest = json.loads(str(data["_jaeger_manifest"].item()))
        assert int(manifest["num_shards"]) == 8
