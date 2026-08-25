#!/usr/bin/env python
"""Rebuild a fragment SavedModel graph with a different conv mask_mode.

Loads the model's project yaml + weights, overrides mask_mode on all
masked_conv1d and residual_block layers (weights are unaffected — mask_mode
only changes mask propagation math), and exports a new SavedModel plus the
classes/project yamls in a layout AvailableModels understands.

bias_initializer: calculate_from_train_data entries are replaced with zeros
(we load trained weights anyway; those initializers would try to read
training data paths that only exist on the training cluster).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import tensorflow as tf
import yaml

from jaeger.nnlib.builder import DynamicModelBuilder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True,
                   help="model directory containing *_graph, *.weights.h5, "
                        "*_classes.yaml, *_project.yaml")
    p.add_argument("--mask-mode", required=True, choices=["any", "majority", "strict"])
    p.add_argument("--disable-masking", action="store_true",
                   help="set model.use_masking=false (reproduce legacy graphs "
                        "exported without mask propagation)")
    p.add_argument("-o", "--outdir", required=True, help="output model directory")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    projects = list(src.glob("*_project.yaml"))
    if len(projects) != 1:
        raise SystemExit(f"expected exactly one *_project.yaml in {src}, found {projects}")
    base = projects[0].name.removesuffix("_project.yaml")
    project_path = projects[0]
    weights_path = src / f"{base}.weights.h5"
    classes_path = src / f"{base}_classes.yaml"

    with open(project_path) as fh:
        config = yaml.safe_load(fh)

    n_patched = 0
    for layer in config["model"]["representation_learner"]["hidden_layers"]:
        if layer.get("name") in ("masked_conv1d", "residual_block"):
            layer.setdefault("config", {})["mask_mode"] = args.mask_mode
            n_patched += 1

    # Strip data-derived bias initializers (unavailable outside the cluster).
    for head in ("classifier", "reliability_model"):
        for layer in config["model"][head]["hidden_layers"]:
            cfg = layer.get("config", {})
            if cfg.get("bias_initializer") == "calculate_from_train_data":
                cfg["bias_initializer"] = "zeros"

    if args.disable_masking:
        config["model"]["use_masking"] = False

    # Build the reliability head even though the Zeus training-data paths in
    # the config do not exist locally (we only need inference, not the data).
    config["generate_reliability_data"] = True

    # The project yaml carries unrendered Jinja in checkpoint paths and the
    # builder guards on pre-existing checkpoints; we never save or resume
    # here, so neutralize all checkpoint dirs and force past the guard.
    import tempfile

    tmp_ckpt = tempfile.mkdtemp(prefix="jaeger_rebuild_ckpt_")
    training_cfg = config.setdefault("training", {})
    training_cfg["classifier_dir"] = tmp_ckpt
    training_cfg["reliability_dir"] = tmp_ckpt
    training_cfg["projection_dir"] = tmp_ckpt
    training_cfg.setdefault("callbacks", {})["directories"] = []
    config["force"] = True

    print(f"patched mask_mode={args.mask_mode} on {n_patched} layers")

    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    model = models["jaeger_model"]
    model.load_weights(str(weights_path))
    print(f"loaded weights from {weights_path.name}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{base}_{args.mask_mode}"
    tf.saved_model.save(model, str(outdir / f"{name}_graph"))
    shutil.copy(classes_path, outdir / f"{name}_classes.yaml")
    with open(outdir / f"{name}_project.yaml", "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"wrote {name}_graph, {name}_classes.yaml, {name}_project.yaml to {outdir}")


if __name__ == "__main__":
    main()
