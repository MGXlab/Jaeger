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

np.set_printoptions(precision=6, suppress=True)

PROJECT_YAML = Path(
    "/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/models/jaeger_models/"
    "jaeger_d1754a4e_3.4M_fragment/model/jaeger_d1754a4e_3.4M_fragment_project.yaml"
)
WEIGHTS_PATH = Path("/tmp/jaeger_d1754a4e_3.4M_fragment_converted.weights.h5")
GRAPH_PATH = PROJECT_YAML.parent / "jaeger_d1754a4e_3.4M_fragment_graph"

cfg = yaml.safe_load(PROJECT_YAML.read_text())
cfg["config_path"] = str(PROJECT_YAML)
cfg["force"] = True
for layer_cfg in cfg.get("model", {}).get("classifier", {}).get("hidden_layers", []):
    if layer_cfg.get("config", {}).get("bias_initializer") == "calculate_from_train_data":
        layer_cfg["config"]["bias_initializer"] = {"class_name": "Zeros", "config": {}}

builder = DynamicModelBuilder(cfg)
models = builder.build_fragment_classifier()
classifier = models["jaeger_classifier"]
classifier.load_weights(str(WEIGHTS_PATH))

saved = tf.saved_model.load(str(GRAPH_PATH))
infer = saved.signatures["serving_default"]

np.random.seed(0)
x = np.random.rand(1, 6, 665).astype(np.float32)
inputs_batch = {"translated": tf.constant(x)}

rep_out = models["rep_model"](inputs_batch, training=False)
emb_built = rep_out[0]
logits_built = classifier(inputs_batch, training=False)

out_saved = infer(inputs=inputs_batch["translated"])
logits_saved = out_saved["prediction"]
emb_saved = out_saved["embedding"]

print("logits built:", logits_built.numpy().flatten())
print("logits saved:", logits_saved.numpy().flatten())
print("max |logits - saved|:", float(tf.reduce_max(tf.abs(logits_built - logits_saved))))
print("max |embedding - saved|:", float(tf.reduce_max(tf.abs(emb_built - emb_saved))))
