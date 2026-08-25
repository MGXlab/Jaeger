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
jaeger = models["jaeger_model"]

ckpt = tf.compat.v1.train.load_checkpoint(str(CKPT_PREFIX))

def set_layer_weights(layer, base_key, var_names):
    """Assign checkpoint tensors at base_key/{var_names} to layer.weights in order."""
    values = []
    for name in var_names:
        key = f"{base_key}/{name}/.ATTRIBUTES/VARIABLE_VALUE"
        try:
            values.append(ckpt.get_tensor(key))
        except Exception:
            pass
    if len(values) != len(layer.weights):
        raise ValueError(
            f"Weight count mismatch for {layer.name}: expected {len(layer.weights)}, "
            f"found {len(values)} at {base_key}"
        )
    for w, v in zip(layer.weights, values):
        w.assign(v)

# embedding
set_layer_weights(jaeger.get_layer("embedding"), "_operations/1", ["_embeddings"])

# first conv / bn
set_layer_weights(jaeger.get_layer("rep_masked_conv1d_0"), "_operations/3", ["kernel", "bias"])
set_layer_weights(jaeger.get_layer("rep_masked_batchnorm_1"), "_operations/4", ["gamma", "beta", "moving_mean", "moving_variance"])

# residual stacks: (current wrapper name, checkpoint op number, subblock count, has_bypass)
stack_map = [
    ("rep_residual_block_3", 6, 1, False),
    ("rep_residual_block_4", 7, 3, False),
    ("rep_residual_block_5", 8, 1, True),
    ("rep_residual_block_6", 9, 3, False),
    ("rep_residual_block_7", 10, 1, True),
    ("rep_residual_block_8", 11, 3, False),
    ("rep_residual_block_9", 12, 1, True),
    ("rep_residual_block_10", 13, 3, False),
]

for wrapper_name, op_num, n_blocks, has_bypass in stack_map:
    wrapper = jaeger.get_layer(wrapper_name)
    for idx in range(n_blocks):
        block = wrapper.blocks[idx]
        base = f"_operations/{op_num}/_operations/{idx + 1}"
        set_layer_weights(block.conv1, f"{base}/conv1", ["kernel", "bias"])
        set_layer_weights(block.conv2, f"{base}/conv2", ["kernel", "bias"])
        set_layer_weights(block.bn1, f"{base}/bn1", ["gamma", "beta", "moving_mean", "moving_variance"])
        set_layer_weights(block.bn2, f"{base}/bn2", ["gamma", "beta", "moving_mean", "moving_variance"])
        if has_bypass and idx == 0:
            set_layer_weights(block.conv3, f"{base}/conv3", ["kernel", "bias"])
            set_layer_weights(block.bn3, f"{base}/bn3", ["gamma", "beta", "moving_mean", "moving_variance"])

# classifier head
clf = jaeger.get_layer("classification_head")
set_layer_weights(clf.get_layer("classifier_dense_0"), "_operations/15/_operations/1", ["_kernel", "bias"])
set_layer_weights(clf.get_layer("classifier_dense_2"), "_operations/15/_operations/3", ["_kernel", "bias"])
set_layer_weights(clf.get_layer("classifier_dense_4"), "_operations/15/_operations/5", ["_kernel", "bias"])

# reliability head (optional, not used for classifier output)
# rel = jaeger.get_layer("reliability_head")
# set_layer_weights(rel.get_layer("reliability_dense_0"), "_operations/16/_operations/1", ["_kernel", "bias"])
# set_layer_weights(rel.get_layer("reliability_dense_2"), "_operations/16/_operations/3", ["_kernel", "bias"])

# Compare
saved = tf.saved_model.load(str(PROJECT_YAML.parent / "jaeger_d1754a4e_3.4M_fragment_graph"))
infer = saved.signatures["serving_default"]

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

logits_built = models["jaeger_classifier"](inputs_batch, training=False)
emb_built = models["rep_model"](inputs_batch, training=False)[0]
out_saved = infer(inputs=inputs_batch["translated"])
logits_saved = out_saved["prediction"]
emb_saved = out_saved["embedding"]

print("built logits:", logits_built.numpy().flatten())
print("saved logits:", logits_saved.numpy().flatten())
print("max |logits - saved|:", float(tf.reduce_max(tf.abs(logits_built - logits_saved))))
print("max |embedding - saved|:", float(tf.reduce_max(tf.abs(tf.cast(emb_built, tf.float32) - tf.cast(emb_saved, tf.float32)))))
