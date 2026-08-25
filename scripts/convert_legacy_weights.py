#!/usr/bin/env python3
"""
Convert a legacy Jaeger .weights.h5 checkpoint into the current builder format.

The legacy checkpoints were produced by an older model graph whose residual blocks
were Keras Functional submodels with inner layers named conv1/conv2/bn1/bn2. The
current DynamicModelBuilder uses explicit rep_* layer names and custom ResidualBlock
layers. Loading the legacy file directly into the current builder therefore misaligns
(or skips) residual-block weights, which is why --generate-reliability-data and
jaeger predict can disagree.

This script walks the legacy HDF5 layout and copies each weight array into the
corresponding layer of a freshly built current-model graph. The resulting
.weights.h5 file can then be loaded by the current builder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import yaml

from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.nnlib.v2.layers import ResidualBlock
from jaeger.utils.misc import load_model_config


def _strip_bias_initializer(cfg: dict) -> None:
    """Replace calculate_from_train_data initializers with zeros so the model
    builds without needing the original training CSV/NPZ files. The real bias
    values are copied from the legacy checkpoint afterward."""
    for branch in ("classifier", "reliability_model"):
        branch_cfg = cfg.get("model", {}).get(branch)
        if branch_cfg is None:
            continue
        for layer_cfg in branch_cfg.get("hidden_layers", []):
            config = layer_cfg.get("config", {})
            if config.get("bias_initializer", "").startswith("calculate_from"):
                config["bias_initializer"] = "zeros"


def _get_h5_datasets(path: Path) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[:]
        f.visititems(visit)
    return data


def _set_layer_weights(layer: tf.keras.layers.Layer, values: list[np.ndarray]) -> None:
    """Assign a list of numpy arrays to a built layer's weights in order."""
    if len(values) != len(layer.weights):
        raise ValueError(
            f"Weight count mismatch for {layer.name}: expected {len(layer.weights)}, "
            f"got {len(values)}"
        )
    for w, v in zip(layer.weights, values):
        w.assign(v)


def convert_legacy_weights(
    project_yaml: Path,
    legacy_weights: Path,
    output: Path,
) -> None:
    print(f"Loading config from {project_yaml}")
    cfg = load_model_config(project_yaml)

    # Avoid touching the original training directories / data files.
    cfg["training"] = {}
    cfg["model"].pop("projection", None)
    _strip_bias_initializer(cfg)

    print("Building current Jaeger model graph")
    builder = DynamicModelBuilder(config=cfg)
    models = builder.build_fragment_classifier()
    jaeger_model = models["jaeger_model"]

    print(f"Reading legacy weights from {legacy_weights}")
    h5_data = _get_h5_datasets(legacy_weights)

    # ------------------------------------------------------------------
    # 1. Top-level layers that have direct name matches in the legacy file
    # ------------------------------------------------------------------
    print("Mapping top-level weights (embedding, masked conv/bn, heads)")

    if "layers/embedding/vars/0" in h5_data:
        emb_layer = jaeger_model.get_layer("embedding")
        _set_layer_weights(emb_layer, [h5_data["layers/embedding/vars/0"]])

    if "layers/masked_conv1d/vars/0" in h5_data:
        conv_layer = jaeger_model.get_layer("rep_masked_conv1d_0")
        _set_layer_weights(
            conv_layer,
            [h5_data["layers/masked_conv1d/vars/0"], h5_data["layers/masked_conv1d/vars/1"]],
        )

    if "layers/masked_batch_norm/vars/0" in h5_data:
        bn_layer = jaeger_model.get_layer("rep_masked_batchnorm_1")
        _set_layer_weights(
            bn_layer,
            [
                h5_data["layers/masked_batch_norm/vars/0"],
                h5_data["layers/masked_batch_norm/vars/1"],
                h5_data["layers/masked_batch_norm/vars/2"],
                h5_data["layers/masked_batch_norm/vars/3"],
            ],
        )

    # Classification head: legacy dense, dense_1, dense_2 -> current dense_0, dense_2, dense_4
    clf_head = jaeger_model.get_layer("classification_head")
    clf_dense_map = ["classifier_dense_0", "classifier_dense_2", "classifier_dense_4"]
    for i, dense_name in enumerate(clf_dense_map):
        group = f"layers/functional_8/layers/dense{'_' + str(i) if i else ''}"
        kernel_key = f"{group}/vars/0"
        bias_key = f"{group}/vars/1"
        if kernel_key in h5_data and bias_key in h5_data:
            dense_layer = clf_head.get_layer(dense_name)
            _set_layer_weights(dense_layer, [h5_data[kernel_key], h5_data[bias_key]])

    # Reliability head: legacy dense, dense_1 -> current dense_0, dense_2
    rel_head = jaeger_model.get_layer("reliability_head")
    rel_dense_map = ["reliability_dense_0", "reliability_dense_2"]
    for i, dense_name in enumerate(rel_dense_map):
        group = f"layers/functional_9/layers/dense{'_' + str(i) if i else ''}"
        kernel_key = f"{group}/vars/0"
        bias_key = f"{group}/vars/1"
        if kernel_key in h5_data and bias_key in h5_data:
            dense_layer = rel_head.get_layer(dense_name)
            _set_layer_weights(dense_layer, [h5_data[kernel_key], h5_data[bias_key]])

    # ------------------------------------------------------------------
    # 2. Residual blocks
    #
    # Legacy layout (one functional_* group per residual block wrapper):
    #   functional      -> rep_residual_block_3   (block_size 1)
    #   functional_1    -> rep_residual_block_4   (block_size 3)
    #   functional_2    -> rep_residual_block_5   (block_size 1)
    #   ...
    #   functional_7    -> rep_residual_block_10  (block_size 3)
    #
    # Inside each functional_* group:
    #   residual_block      -> subblock 0
    #   residual_block_1    -> subblock 1
    #   residual_block_2    -> subblock 2
    #
    # Inside each subblock group:
    #   bn1/vars/0..3, bn2/vars/0..3, [bn3/vars/0..3]
    #   conv1/vars/0..1, conv2/vars/0..1, [conv3/vars/0..1]
    # ------------------------------------------------------------------
    print("Mapping residual block weights")
    residual_group_re = re.compile(r"^layers/functional(?:_(\d+))?/layers/residual_block(_\d+)?/(\w+)/vars/(\w+)$")

    # Collect residual block weights grouped by wrapper index / subblock index.
    residual_weights: dict[tuple[int, int, str, int], np.ndarray] = {}
    for key, value in h5_data.items():
        m = residual_group_re.match(key)
        if not m:
            continue
        wrapper_idx = int(m.group(1) or 0)
        subblock_suffix = m.group(2) or ""
        subblock_idx = int(subblock_suffix.strip("_")) if subblock_suffix else 0
        component = m.group(3)  # bn1, bn2, bn3, conv1, conv2, conv3
        var_idx = int(m.group(4))
        residual_weights[(wrapper_idx, subblock_idx, component, var_idx)] = value

    # Wrapper index 0 corresponds to rep_residual_block_3.
    for (wrapper_idx, subblock_idx, component, var_idx), value in residual_weights.items():
        block_number = 3 + wrapper_idx
        wrapper_name = f"rep_residual_block_{block_number}"
        subblock_name = f"{wrapper_name}_{subblock_idx}"

        wrapper = jaeger_model.get_layer(wrapper_name)
        subblock = wrapper.blocks[subblock_idx]
        if not isinstance(subblock, ResidualBlock):
            raise TypeError(f"Expected ResidualBlock, got {type(subblock)}")

        if component == "conv1":
            layer = subblock.conv1
        elif component == "conv2":
            layer = subblock.conv2
        elif component == "conv3":
            layer = subblock.conv3
        elif component == "bn1":
            layer = subblock.bn1
        elif component == "bn2":
            layer = subblock.bn2
        elif component == "bn3":
            layer = subblock.bn3
        else:
            raise ValueError(f"Unknown residual component: {component}")

        # Gather all var arrays for this component and assign in one go.
        # Convs may have only a kernel when use_bias=False; batch norms always
        # have four moving/statistic variables.
        var_indices = sorted(
            idx
            for (w, s, c, idx), _ in residual_weights.items()
            if w == wrapper_idx and s == subblock_idx and c == component
        )
        values = [residual_weights[(wrapper_idx, subblock_idx, component, idx)] for idx in var_indices]
        _set_layer_weights(layer, values)

    # ------------------------------------------------------------------
    # 3. Save converted weights
    # ------------------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing converted weights to {output}")
    jaeger_model.save_weights(str(output))
    print("Done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-yaml",
        type=Path,
        required=True,
        help="Path to the model's *_project.yaml config file.",
    )
    parser.add_argument(
        "--legacy-weights",
        type=Path,
        required=True,
        help="Path to the legacy .weights.h5 checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the converted .weights.h5 file.",
    )
    args = parser.parse_args()
    convert_legacy_weights(args.project_yaml, args.legacy_weights, args.output)


if __name__ == "__main__":
    main()
