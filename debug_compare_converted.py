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

from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.seqops.encode import process_string_train

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

cfg = yaml.safe_load(PROJECT_YAML.read_text())
cfg["config_path"] = str(PROJECT_YAML)
cfg["force"] = True

for layer_cfg in cfg.get("model", {}).get("classifier", {}).get("hidden_layers", []):
    if layer_cfg.get("config", {}).get("bias_initializer") == "calculate_from_train_data":
        layer_cfg["config"]["bias_initializer"] = {"class_name": "Zeros", "config": {}}
for layer_cfg in cfg.get("model", {}).get("reliability_model", {}).get("hidden_layers", []):
    if layer_cfg.get("config", {}).get("bias_initializer") == "calculate_from_train_data":
        layer_cfg["config"]["bias_initializer"] = {"class_name": "Zeros", "config": {}}

print("Building classifier from config...")
builder = DynamicModelBuilder(cfg)
models = builder.build_fragment_classifier()
classifier = models["jaeger_classifier"]
print(f"Loading weights from {WEIGHTS_PATH}")
classifier.load_weights(WEIGHTS_PATH)

print("Loading SavedModel...")
saved = tf.saved_model.load(str(GRAPH_PATH))
infer = saved.signatures["serving_default"]

label = None
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
print(f"Using label={label}, len(seq)={len(seq)}")

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
print("Input shape:", {k: v.shape.as_list() for k, v in inputs_batch.items()})

print("\n--- Built Keras classifier (training=False) ---")
rep_out_built = models["rep_model"](inputs_batch, training=False)
emb_built = rep_out_built[0]
nmd_built = rep_out_built[1]
logits_built = classifier(inputs_batch, training=False)
print("logits:", logits_built.numpy().flatten())
print("softmax:", tf.nn.softmax(logits_built).numpy().flatten())
print("embedding max:", float(tf.reduce_max(tf.abs(emb_built))))

print("\n--- SavedModel ---")
out_saved = infer(inputs=inputs_batch["translated"])
logits_saved = out_saved["prediction"]
emb_saved = out_saved.get("embedding")
print("logits:", logits_saved.numpy().flatten())
print("softmax:", tf.nn.softmax(logits_saved).numpy().flatten())
print("embedding max:", float(tf.reduce_max(tf.abs(emb_saved))))

print("\n--- Differences (training=False) ---")
print("max |built_logits - saved_logits|:", np.max(np.abs(logits_built.numpy() - logits_saved.numpy())))
print("max |built_softmax - saved_softmax|:", np.max(np.abs(tf.nn.softmax(logits_built).numpy() - tf.nn.softmax(logits_saved).numpy())))
if emb_saved is not None:
    print("max |built_embedding - saved_embedding|:", np.max(np.abs(emb_built.numpy() - emb_saved.numpy())))

print("\n--- Built rep_model with training=True ---")
rep_out_train = models["rep_model"](inputs_batch, training=True)
emb_train = rep_out_train[0]
logits_train = classifier(inputs_batch, training=True)
print("logits:", logits_train.numpy().flatten())
print("embedding max:", float(tf.reduce_max(tf.abs(emb_train))))
print("max |built_train_embedding - saved_embedding|:", np.max(np.abs(emb_train.numpy() - emb_saved.numpy())))
print("max |built_train_logits - saved_logits|:", np.max(np.abs(logits_train.numpy() - logits_saved.numpy())))

print("\n--- Rebuild with mixed_float16 policy ---")
tf.keras.mixed_precision.set_global_policy("mixed_float16")
builder16 = DynamicModelBuilder(cfg)
models16 = builder16.build_fragment_classifier()
classifier16 = models16["jaeger_classifier"]
classifier16.load_weights(str(WEIGHTS_PATH))
rep_out16 = models16["rep_model"](inputs_batch, training=False)
emb_built16 = rep_out16[0]
logits_built16 = classifier16(inputs_batch, training=False)
print("logits:", logits_built16.numpy().flatten())
print("embedding max:", float(tf.reduce_max(tf.abs(emb_built16))))
print("max |built16_logits - saved_logits|:", np.max(np.abs(logits_built16.numpy() - logits_saved.numpy())))
print("max |built16_embedding - saved_embedding|:", np.max(np.abs(emb_built16.numpy() - emb_saved.numpy())))
