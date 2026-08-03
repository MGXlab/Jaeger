"""Custom Keras callbacks used by Jaeger.

The standard Keras ``TerminateOnNaN`` callback overrides ``on_batch_end``.
With the TensorFlow backend Keras may dispatch callbacks asynchronously, so
a NaN detected in ``on_batch_end`` can set ``model.stop_training`` too late
to stop the current epoch. The callbacks below override the training-specific
hooks so they run synchronously.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import sklearn.metrics as skm
import tensorflow as tf

from jaeger.utils.logging import get_logger

logger = get_logger(log_file=None, log_path=None, level=3)


class SyncTerminateOnNaN(tf.keras.callbacks.Callback):
    """Synchronously terminate training when the loss becomes non-finite.

    Parameters
    ----------
    raise_error : bool
        If ``True``, raise ``RuntimeError`` immediately on NaN/Inf loss.
        If ``False`` (default), set ``model.stop_training = True`` so Keras
        exits gracefully at the end of the current batch.
    """

    def __init__(self, raise_error: bool = False):
        super().__init__()
        self.raise_error = raise_error

    def _check_loss(self, batch, logs):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is None:
            return
        if np.isnan(loss) or np.isinf(loss):
            msg = f"Batch {batch}: Invalid loss ({loss}), terminating training"
            if self.raise_error:
                raise RuntimeError(msg)
            tf.print(msg)
            self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        self._check_loss(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        # Also catch NaNs during validation/evaluation.
        self._check_loss(batch, logs)


class ValidationConfusionMatrix(tf.keras.callbacks.Callback):
    """Save a validation confusion matrix at the end of each epoch.

    The callback predicts on a fixed, unshuffled validation dataset, compares
    the predictions with pre-loaded ground-truth labels, and writes the
    confusion matrix (and derived per-class precision/recall/F1) to a JSON or
    CSV file. It is intended for monitoring which classes are being confused
    during training without affecting the model graph.

    Parameters
    ----------
    validation_data :
        Unshuffled ``tf.data.Dataset`` yielding ``(features, one_hot_labels)``.
        The iteration order must match ``labels_path``.
    output_path :
        Destination file. Extension ``.json`` writes JSON, ``.csv`` writes a
        long-form CSV with one row per (true_class, pred_class, count).
    labels_path :
        Path to a Jaeger sharded ``.npz`` archive containing ``labels_*`` keys.
        If omitted, labels are collected by iterating ``validation_data`` once
        at the first epoch (slower but works for non-npz data).
    num_classes :
        Number of classes. Inferred from ``class_names`` if not given.
    class_names :
        Optional class names for the output. Defaults to integer labels.
    batch_size :
        Batch size passed to ``model.predict``.
    include_row_normalized :
        If True (default), also save the row-normalized confusion matrix.
    """

    def __init__(
        self,
        validation_data: tf.data.Dataset,
        output_path: str | Path,
        labels_path: str | Path | None = None,
        num_classes: int | None = None,
        class_names: list[str] | None = None,
        batch_size: int | None = None,
        include_row_normalized: bool = True,
    ):
        super().__init__()
        self.validation_data = validation_data
        self.output_path = Path(output_path)
        self.labels_path = Path(labels_path) if labels_path is not None else None
        self.batch_size = batch_size
        self.include_row_normalized = include_row_normalized

        if class_names is not None:
            self.class_names = list(class_names)
            self.num_classes = len(self.class_names)
        elif num_classes is not None:
            self.num_classes = int(num_classes)
            self.class_names = [str(i) for i in range(self.num_classes)]
        else:
            raise ValueError("Either num_classes or class_names must be provided.")

        self.y_true: np.ndarray | None = None

    def on_train_begin(self, logs=None):
        if self.labels_path is not None:
            data = np.load(str(self.labels_path), allow_pickle=True)
            label_keys = sorted(k for k in data.files if k.startswith("labels_"))
            if not label_keys:
                raise ValueError(f"No labels_* keys found in {self.labels_path}")
            self.y_true = np.concatenate([data[k] for k in label_keys], axis=0)
        elif self.validation_data is not None:
            y_true = []
            for _, y in self.validation_data:
                y_true.append(np.argmax(y.numpy(), axis=-1))
            self.y_true = (
                np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
            )

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        y_pred_logits = self.model.predict(
            self.validation_data,
            batch_size=self.batch_size,
            verbose=0,
        )
        y_pred = np.argmax(y_pred_logits, axis=-1)

        if self.y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Label/prediction count mismatch: "
                f"{self.y_true.shape[0]} vs {y_pred.shape[0]}"
            )

        labels = list(range(self.num_classes))
        cm = skm.confusion_matrix(self.y_true, y_pred, labels=labels)
        prec, rec, f1, _ = skm.precision_recall_fscore_support(
            self.y_true, y_pred, labels=labels, zero_division=0
        )
        row_norm = cm / cm.sum(axis=1, keepdims=True)

        per_class = {
            self.class_names[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i in range(self.num_classes)
        }

        val_accuracy = float(np.mean(y_pred == self.y_true))
        val_loss = self._compute_validation_loss(self.y_true, y_pred_logits)

        # Inject metrics into the shared epoch logs so downstream callbacks
        # (e.g. EarlyStopping, ReduceLROnPlateau, CSVLogger) can see them when
        # Keras validation is disabled to avoid duplicate inference.
        if logs is None:
            logs = {}
        if val_loss is not None and logs.get("val_loss") is None:
            logs["val_loss"] = val_loss
        if logs.get("val_categorical_accuracy") is None:
            logs["val_categorical_accuracy"] = val_accuracy
        if logs.get("val_macro_f1") is None:
            logs["val_macro_f1"] = float(f1.mean())
        for i, name in enumerate(self.class_names):
            if logs.get(f"val_precision_class{i}") is None:
                logs[f"val_precision_class{i}"] = float(prec[i])
            if logs.get(f"val_recall_class{i}") is None:
                logs[f"val_recall_class{i}"] = float(rec[i])
            if logs.get(f"val_f1_class{i}") is None:
                logs[f"val_f1_class{i}"] = float(f1[i])

        payload: dict[str, Any] = {
            "epoch": int(epoch),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "macro_f1": float(f1.mean()),
            "val_categorical_accuracy": val_accuracy,
        }
        if val_loss is not None:
            payload["val_loss"] = val_loss
        if self.include_row_normalized:
            payload["row_normalized_cm"] = np.nan_to_num(row_norm).round(4).tolist()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.suffix.lower() == ".csv":
            self._write_csv(payload)
        else:
            self._write_json(payload)

    def _write_json(self, payload: dict[str, Any]) -> None:
        """Append *payload* to a JSON list, creating the file if needed."""
        history: list[dict[str, Any]] = []
        if self.output_path.exists():
            try:
                data = json.loads(self.output_path.read_text())
                if isinstance(data, list):
                    history = data
            except json.JSONDecodeError:
                pass
        history.append(payload)
        self.output_path.write_text(json.dumps(history, indent=2))

    def _write_csv(self, payload: dict[str, Any]) -> None:
        rows = []
        for i, true_name in enumerate(self.class_names):
            for j, pred_name in enumerate(self.class_names):
                rows.append(
                    {
                        "epoch": payload["epoch"],
                        "true_class": true_name,
                        "pred_class": pred_name,
                        "count": payload["confusion_matrix"][i][j],
                    }
                )
        fieldnames = ["epoch", "true_class", "pred_class", "count"]
        write_header = not self.output_path.exists()
        with self.output_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _compute_validation_loss(
        self, y_true: np.ndarray, y_pred_logits: np.ndarray
    ) -> float | None:
        """Compute validation loss using the model's compiled loss if available."""
        if self.model.loss is None:
            logger.warning(
                "ValidationConfusionMatrix: model.loss is None, cannot compute val_loss"
            )
            return None
        y_true_onehot = np.eye(self.num_classes, dtype=np.float32)[y_true]
        try:
            loss = self.model.loss(
                tf.convert_to_tensor(y_true_onehot, dtype=tf.float32),
                tf.convert_to_tensor(y_pred_logits, dtype=tf.float32),
            )
            if isinstance(loss, tf.Tensor):
                loss = loss.numpy()
            return float(np.mean(loss))
        except Exception as exc:
            logger.warning(
                f"ValidationConfusionMatrix: val_loss computation failed: {exc}"
            )
            return None
