#!/usr/bin/env python3
"""Generate experiment 85 config from the 1500 bp 6-class Zeus template."""

from __future__ import annotations

from pathlib import Path

import yaml

TEMPLATE = Path("train_config/nn_config_1500bp_nmd_merge_6_class_zeus.yaml")
OUTPUT = Path("train_config/jaeger_experiment_85.yaml")

MULTISCALE_AXIAL_LAYERS = [
    {"name": "masked_conv1d", "config": {
        "filters": 96,
        "kernel_size": 7,
        "strides": 1,
        "padding": "same",
        "dilation_rate": 1,
        "use_bias": True,
        "activation": None,
        "kernel_regularizer": "l2",
        "kernel_regularizer_w": 0.00001,
    }},
    {"name": "masked_dyt"},
    {"name": "activation", "config": {"activation": "gelu"}},
    {"name": "parallel_branches", "config": {
        "merge": "concat",
        "branches": [
            {"hidden_layers": [
                {"name": "residual_block", "config": {
                    "use_1x1conv": False,
                    "block_size": 1,
                    "filters": 96,
                    "kernel_size": 5,
                    "dilation_rate": 3,
                    "padding": "same",
                    "use_bias": True,
                    "kernel_regularizer": "l2",
                    "kernel_regularizer_w": 0.00001,
                    "norm_type": "masked_dyt",
                }},
            ], "pooling": None},
            {"hidden_layers": [
                {"name": "residual_block", "config": {
                    "use_1x1conv": False,
                    "block_size": 1,
                    "filters": 96,
                    "kernel_size": 5,
                    "dilation_rate": 9,
                    "padding": "same",
                    "use_bias": True,
                    "kernel_regularizer": "l2",
                    "kernel_regularizer_w": 0.00001,
                    "norm_type": "masked_dyt",
                }},
            ], "pooling": None},
            {"hidden_layers": [
                {"name": "residual_block", "config": {
                    "use_1x1conv": False,
                    "block_size": 1,
                    "filters": 96,
                    "kernel_size": 5,
                    "dilation_rate": 18,
                    "padding": "same",
                    "use_bias": True,
                    "kernel_regularizer": "l2",
                    "kernel_regularizer_w": 0.00001,
                    "norm_type": "masked_dyt",
                }},
            ], "pooling": None},
            {"hidden_layers": [
                {"name": "residual_block", "config": {
                    "use_1x1conv": False,
                    "block_size": 1,
                    "filters": 96,
                    "kernel_size": 5,
                    "dilation_rate": 27,
                    "padding": "same",
                    "use_bias": True,
                    "kernel_regularizer": "l2",
                    "kernel_regularizer_w": 0.00001,
                    "norm_type": "masked_dyt",
                }},
            ], "pooling": None},
        ],
    }},
    # 1x1 projection: 384 -> 96 channels so AxialAttention sees embed_dim == channels
    {"name": "masked_conv1d", "config": {
        "filters": 96,
        "kernel_size": 1,
        "strides": 1,
        "padding": "same",
        "use_bias": True,
        "activation": None,
        "kernel_regularizer": "l2",
        "kernel_regularizer_w": 0.00001,
    }},
    {"name": "masked_dyt"},
    {"name": "activation", "config": {"activation": "gelu"}},
    {"name": "axial_attention", "config": {
        "embed_dim": 96,
        "num_heads": 4,
        "feed_forward_dim": 192,
        "dropout_rate": 0.1,
        "num_blocks": 2,
        "norm_type": "masked_dyt",
    }},
    {"name": "nmd"},
]


def main() -> int:
    cfg = yaml.safe_load(TEMPLATE.read_text())

    # Zeus experiment convention
    cfg["model"]["name"] = "jaeger"
    cfg["model"]["experiment"] = 85

    # The translated NPZ was generated from ~1800 bp nucleotide fragments.
    # Because the input has 6 reading frames, a crop size of 500 codons
    # corresponds to roughly 1500 bp of nucleotide sequence.
    cfg["model"]["string_processor"]["crop_sizes"] = [500]
    cfg["model"]["string_processor"]["validation_crop_sizes"] = [500]

    # Replace the serial residual blocks with parallel branches + 1x1 projection + axial attention
    cfg["model"]["representation_learner"]["hidden_layers"] = MULTISCALE_AXIAL_LAYERS

    # The global-pool output and NMD vector now have 96 dimensions.
    cfg["model"]["classifier"]["input_shape"] = 96
    cfg["model"]["projection"]["input_shape"] = 96
    cfg["model"]["reliability_model"]["input_shape"] = 96

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
