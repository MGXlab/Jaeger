"""
GELU-and-precision comparison for the rebuilt Jaeger classifier vs the legacy SavedModel.

Tests:
- DynamicModelBuilder (current rebuild path) with converted weights, exact GELU
- DynamicModelBuilder with approximate GELU (patched tf.keras.layers.Activation)
- Mixed-precision rebuild
- SavedModel via serving signature and via inner function call_flat

Weight file: /tmp/jaeger_d1754a4e_3.4M_fragment_converted.weights.h5 (current format)
SavedModel:  .../jaeger_d1754a4e_3.4M_fragment_graph
"""

import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path

np.set_printoptions(precision=6, suppress=True)

PROJECT_YAML = Path(
    "/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/models/jaeger_models/"
    "jaeger_d1754a4e_3.4M_fragment/model/jaeger_d1754a4e_3.4M_fragment_project.yaml"
)
WEIGHTS_PATH = Path("/tmp/jaeger_d1754a4e_3.4M_fragment_converted.weights.h5")
GRAPH_PATH = PROJECT_YAML.parent / "jaeger_d1754a4e_3.4M_fragment_graph"
VAL_PATH = Path(
    "/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/training/"
    "data/val_data_2000.csv"
)


def load_config():
    cfg = yaml.safe_load(PROJECT_YAML.read_text())
    cfg["config_path"] = str(PROJECT_YAML)
    cfg["force"] = True
    for layer_cfg in cfg.get("model", {}).get("classifier", {}).get("hidden_layers", []):
        if layer_cfg.get("config", {}).get("bias_initializer") == "calculate_from_train_data":
            layer_cfg["config"]["bias_initializer"] = {"class_name": "Zeros", "config": {}}
    for layer_cfg in cfg.get("model", {}).get("reliability_model", {}).get("hidden_layers", []):
        if layer_cfg.get("config", {}).get("bias_initializer") == "calculate_from_train_data":
            layer_cfg["config"]["bias_initializer"] = {"class_name": "Zeros", "config": {}}
    return cfg


def prepare_input():
    seq = None
    with open(VAL_PATH) as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 2:
                label = int(row[0])
                seq = row[1]
                break
    if seq is None:
        raise ValueError(f"No sequence found in {VAL_PATH}")

    from jaeger.seqops.encode import process_string_train
    from jaeger.nnlib.builder import DynamicModelBuilder

    cfg = load_config()
    builder = DynamicModelBuilder(cfg)
    sp = builder._get_string_processor_config()
    processor = process_string_train(
        codons=sp.get("codon"),
        codon_num=sp.get("codon_id"),
        codon_depth=sp.get("codon_depth") or 64,
        class_label_onehot=False,
        seq_onehot=sp.get("seq_onehot", True),
        num_classes=builder.classifier_out_dim,
        crop_size=sp.get("crop_size"),
        input_type=sp.get("input_type", "translated"),
        masking=sp.get("masking", False),
        ngram_width=sp.get("ngram_width") or 3,
        shuffle_frames=sp.get("shuffle_frames", False),
    )
    inputs, _ = processor(f"{label},{seq}")
    inputs_batch = {k: tf.expand_dims(tf.cast(v, tf.float32), 0) for k, v in inputs.items()}
    return inputs_batch, label


def build_and_load(approximate_gelu: bool = False, mixed_precision: bool = False):
    """Build and load the classifier; optionally force approximate GELU."""
    from jaeger.nnlib.builder import DynamicModelBuilder

    cfg = load_config()

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    # Monkey-patch tf.keras.layers.Activation so that the string "gelu" resolves
    # to approximate GELU everywhere (Activation layers and Dense activations).
    original_init = tf.keras.layers.Activation.__init__

    def patched_init(self, activation, **kwargs):
        if activation == "gelu" and approximate_gelu:
            activation = lambda x: tf.nn.gelu(x, approximate=True)
        original_init(self, activation, **kwargs)

    tf.keras.layers.Activation.__init__ = patched_init

    try:
        builder = DynamicModelBuilder(cfg)
        models = builder.build_fragment_classifier()
        classifier = models["jaeger_classifier"]
        classifier.load_weights(WEIGHTS_PATH)
        return builder, models
    finally:
        tf.keras.layers.Activation.__init__ = original_init


def savedmodel_inner_call(saved, x):
    """Call the SavedModel inner function as described in the task brief."""
    sig = saved.signatures["serving_default"]
    fn_name = [k for k in sig._func_graph._functions if "__inference_serving_default" in k][0]
    fn2 = sig._func_graph._functions[fn_name]
    outputs = fn2.call_flat(x, *sig._captured_inputs)
    return outputs


def summarize(name, built_logits, saved_logits, built_emb=None, saved_emb=None):
    print(f"\n--- {name} ---")
    bl = built_logits.numpy().flatten()
    sl = saved_logits.numpy().flatten()
    print("built logits:", bl)
    print("saved logits:", sl)
    diff = np.abs(bl - sl)
    print("max |built - saved| logits :", np.max(diff))
    print("mean |built - saved| logits:", np.mean(diff))
    if built_emb is not None and saved_emb is not None:
        be = built_emb.numpy()
        se = saved_emb.numpy()
        emb_diff = np.abs(be - se)
        print("max |built - saved| embedding :", np.max(emb_diff))
        print("mean |built - saved| embedding:", np.mean(emb_diff))
        print("built embedding max:", np.max(np.abs(be)), "dtype:", built_emb.dtype)
        print("saved embedding max:", np.max(np.abs(se)), "dtype:", saved_emb.dtype)
    print("built logits dtype:", built_logits.dtype)
    print("saved logits dtype:", saved_logits.dtype)


def main():
    inputs_batch, label = prepare_input()
    print(f"Using label={label}, input shapes:", {k: v.shape.as_list() for k, v in inputs_batch.items()})

    # Load SavedModel once
    saved = tf.saved_model.load(str(GRAPH_PATH))
    infer = saved.signatures["serving_default"]

    print("\n=== SavedModel signature outputs ===")
    print("structured outputs:", infer.structured_outputs)
    out_sig = infer(inputs=inputs_batch["translated"])
    logits_sig = out_sig["prediction"]
    emb_sig = out_sig.get("embedding")
    print("signature logits dtype:", logits_sig.dtype, "shape:", logits_sig.shape)
    if emb_sig is not None:
        print("signature embedding dtype:", emb_sig.dtype, "shape:", emb_sig.shape,
              "max:", np.max(np.abs(emb_sig.numpy())))

    print("\n=== SavedModel inner function outputs ===")
    inner_outs = savedmodel_inner_call(saved, inputs_batch["translated"])
    print("inner output count:", len(inner_outs))
    for i, t in enumerate(inner_outs):
        print(f"  inner[{i}] dtype={t.dtype} shape={t.shape} max={np.max(np.abs(t.numpy()))}")
    # Per task brief, order is [embedding_half, nmd_half, classification_logits_float, reliability_logits_float]
    logits_inner = inner_outs[2]
    emb_inner = inner_outs[0]

    # --- Current code path: exact GELU ---
    print("\n" + "="*70)
    print("CURRENT CODE PATH (DynamicModelBuilder, Activation('gelu') -> exact)")
    print("="*70)
    builder, models = build_and_load(approximate_gelu=False, mixed_precision=False)
    rep_out = models["rep_model"](inputs_batch, training=False)
    emb_built = rep_out[0]
    logits_built = models["jaeger_classifier"](inputs_batch, training=False)
    summarize("Exact-GELU rebuild vs SavedModel signature", logits_built, logits_sig, emb_built, emb_sig)
    summarize("Exact-GELU rebuild vs SavedModel inner function", logits_built, logits_inner, emb_built, emb_inner)

    # --- Approximate GELU path ---
    print("\n" + "="*70)
    print("PATCH TEST (force approximate GELU everywhere)")
    print("="*70)
    builder_a, models_a = build_and_load(approximate_gelu=True, mixed_precision=False)
    rep_out_a = models_a["rep_model"](inputs_batch, training=False)
    emb_built_a = rep_out_a[0]
    logits_built_a = models_a["jaeger_classifier"](inputs_batch, training=False)
    summarize("Approximate-GELU rebuild vs SavedModel signature", logits_built_a, logits_sig, emb_built_a, emb_sig)

    # --- Mixed precision ---
    print("\n" + "="*70)
    print("MIXED PRECISION (mixed_float16)")
    print("="*70)
    builder16, models16 = build_and_load(approximate_gelu=False, mixed_precision=True)
    rep_out16 = models16["rep_model"](inputs_batch, training=False)
    emb_built16 = rep_out16[0]
    logits_built16 = models16["jaeger_classifier"](inputs_batch, training=False)
    summarize("Mixed_float16 rebuild vs SavedModel signature", logits_built16, logits_sig, emb_built16, emb_sig)

    # --- Direct comparison approximate vs exact ---
    print("\n" + "="*70)
    print("DIRECT COMPARISON: APPROXIMATE vs EXACT GELU (rebuilt model)")
    print("="*70)
    print("max |approx logits - exact logits|:", np.max(np.abs(logits_built_a.numpy() - logits_built.numpy())))
    print("max |approx emb - exact emb|:", np.max(np.abs(emb_built_a.numpy() - emb_built.numpy())))
    print("mean |approx logits - exact logits|:", np.mean(np.abs(logits_built_a.numpy() - logits_built.numpy())))
    print("mean |approx emb - exact emb|:", np.mean(np.abs(emb_built_a.numpy() - emb_built.numpy())))


if __name__ == "__main__":
    main()
