import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import yaml
import csv
from pathlib import Path

from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.seqops.encode import process_string_train

PROJECT_YAML = Path(
    "/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/models/jaeger_models/"
    "jaeger_d1754a4e_3.4M_fragment/model/jaeger_d1754a4e_3.4M_fragment_project.yaml"
)
CKPT_PREFIX = PROJECT_YAML.parent / "jaeger_d1754a4e_3.4M_fragment_graph/variables/variables"

cfg = yaml.safe_load(PROJECT_YAML.read_text())
cfg["config_path"] = str(PROJECT_YAML)
cfg["force"] = True
for layer_cfg in cfg.get("model", {}).get("classifier", {}).get("hidden_layers", []):
    if layer_cfg.get("config", {}).get("bias_initializer") == "calculate_from_train_data":
        layer_cfg["config"]["bias_initializer"] = {"class_name": "Zeros", "config": {}}

builder = DynamicModelBuilder(cfg)
models = builder.build_fragment_classifier()
classifier = models["jaeger_classifier"]

# Load checkpoint values directly by matching shapes/values.
ckpt = tf.compat.v1.train.load_checkpoint(str(CKPT_PREFIX))
ckpt_vars = {
    k: ckpt.get_tensor(k)
    for k in ckpt.get_variable_to_shape_map().keys()
    if not k.endswith("_CHECKPOINTABLE_OBJECT_GRAPH")
}

used = set()
for w in models["jaeger_model"].weights:
    arr = w.numpy()
    best_key = None
    best_diff = float("inf")
    for k, v in ckpt_vars.items():
        if k in used:
            continue
        if v.shape != arr.shape:
            continue
        diff = float(np.max(np.abs(v - arr)))
        if diff < best_diff:
            best_diff = diff
            best_key = k
    if best_key is None:
        raise ValueError(f"No matching checkpoint variable for {w.name}")
    if best_diff > 1e-4:
        print(f"WARN large diff for {w.name}: {best_key} diff={best_diff}")
    w.assign(ckpt_vars[best_key])
    used.add(best_key)

print(f"Matched {len(used)}/{len(models['jaeger_model'].weights)} variables")

# Load SavedModel
saved = tf.saved_model.load(str(PROJECT_YAML.parent / "jaeger_d1754a4e_3.4M_fragment_graph"))
infer = saved.signatures["serving_default"]

# Validation sequence
with open("/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/training/data/val_data_2000.csv") as fh:
    reader = csv.reader(fh)
    label, seq = next((int(row[0]), row[1]) for row in reader if len(row) >= 2)
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

logits_built = classifier(inputs_batch, training=False)
emb_built = models["rep_model"](inputs_batch, training=False)[0]
out_saved = infer(inputs=inputs_batch["translated"])
logits_saved = out_saved["prediction"]
emb_saved = out_saved["embedding"]

print("built logits:", logits_built.numpy().flatten())
print("saved logits:", logits_saved.numpy().flatten())
print("max |logits - saved|:", float(tf.reduce_max(tf.abs(logits_built - logits_saved))))
print("max |embedding - saved|:", float(tf.reduce_max(tf.abs(tf.cast(emb_built, tf.float32) - tf.cast(emb_saved, tf.float32)))))
