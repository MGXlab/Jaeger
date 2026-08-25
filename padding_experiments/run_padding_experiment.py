#!/usr/bin/env python
"""Measure how M/N padding length affects fragment-level predictions.

For each sampled contig (>= 2000 bp) we take nested prefixes of real length
L = 2000 - pad for pad in {500, 450, ..., 50, 1} and right-pad them to
2000 nt with 'M' (production behaviour) or 'N' (control). The unpadded
2000 nt prefix serves as the per-contig reference prediction. Because
fragments from the same contig are nested, any prediction shift is caused
by the padding ratio alone.

Fragments are encoded and scored through the exact production inference
path (process_string_inference + InferModel on the SavedModel graph).

Outputs: results.csv with one row per fragment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pyfastx
import tensorflow as tf

from jaeger.nnlib.inference import InferModel
from jaeger.seqops.encode import process_string_inference
from jaeger.utils.misc import AvailableModels, json_to_dict

FRAGSIZE = 2000
PAD_LENGTHS = list(range(500, 0, -50)) + [1]  # 500, 450, ..., 50, 1
PAD_CHARS = ["M", "N"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="input FASTA")
    p.add_argument("-m", "--model", default="jaeger_f47f36db_1.2M_fragment")
    p.add_argument("--model-dir", default=None,
                   help="directory containing a rebuilt model (overrides -m)")
    p.add_argument("--pad-chars", default="M,N", help="comma-separated pad chars")
    p.add_argument("-o", "--output", default="results.csv")
    p.add_argument("-n", "--num-contigs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--xla", action="store_true", help="use XLA-compiled inference")
    return p.parse_args()


def load_model(model_name: str) -> InferModel:
    config_path = Path(__file__).resolve().parent.parent / "src/jaeger/data/config.json"
    model_paths = json_to_dict(config_path).get("model_paths")
    info = AvailableModels(path=model_paths).info
    if model_name not in info:
        sys.exit(f"model {model_name} not found; available: {sorted(info)}")
    return InferModel(info[model_name], use_xla=False)


def load_model_from_dir(model_dir: str) -> InferModel:
    info = AvailableModels(path=model_dir).info
    classification = {
        name: meta
        for name, meta in info.items()
        if meta.get("graph") is not None and meta.get("classes") is not None
    }
    if not classification:
        sys.exit(f"no classification model found in {model_dir}")
    name, meta = next(iter(classification.items()))
    print(f"using model {name} from {model_dir}")
    return InferModel(meta, use_xla=False)


def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def build_records(fasta_path: str, num_contigs: int, seed: int, pad_chars):
    """Yield (csv_line, meta_dict) for every padded/reference fragment."""
    fa = pyfastx.Fasta(fasta_path)
    eligible = [(rec.name, rec.seq.upper()) for rec in fa if len(rec.seq) >= FRAGSIZE]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligible), size=min(num_contigs, len(eligible)), replace=False)

    records = []
    for i in sorted(idx):
        name, seq = eligible[i]
        contig_id = name.split()[0]
        prefix = seq[:FRAGSIZE]

        variants = [("none", 0, prefix)]
        for pad in PAD_LENGTHS:
            real = prefix[: FRAGSIZE - pad]
            for char in pad_chars:
                variants.append((char, pad, real + char * pad))

        for pad_char, pad_len, padded in variants:
            real_len = FRAGSIZE - pad_len
            real_seq = padded[:real_len]
            g = real_seq.count("G")
            c = real_seq.count("C")
            a = real_seq.count("A")
            t = real_seq.count("T")
            gc_skew = safe_divide(g - c, g + c)
            frag_id = f"{contig_id}__{pad_char}__{pad_len}"
            csv_line = (
                f"{padded},{frag_id},0,1,0,{real_len},{g},{c},{a},{t},"
                f"{gc_skew: .3f}"
            )
            records.append(
                {
                    "csv_line": csv_line,
                    "contig_id": contig_id,
                    "frag_id": frag_id,
                    "pad_char": pad_char,
                    "pad_len": pad_len,
                    "real_len": real_len,
                }
            )
    return records


def run_inference(model: InferModel, records: list[dict], batch: int):
    spc = model.string_processor_config
    lines = [r["csv_line"] for r in records]

    dataset = tf.data.Dataset.from_generator(
        lambda: iter(lines), output_signature=(tf.TensorSpec(shape=(), dtype=tf.string))
    )
    mapped = dataset.map(
        process_string_inference(
            codons=spc.get("codon"),
            codon_num=spc.get("codon_id"),
            codon_depth=spc.get("codon_depth"),
            ngram_width=spc.get("ngram_width"),
            seq_onehot=spc.get("seq_onehot"),
            crop_size=FRAGSIZE,
            input_type=spc.get("input_type"),
            masking=spc.get("masking"),
            mutate=False,
            shuffle=False,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    batched = mapped.batch(batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(25)
    return model.predict(batched)


def main():
    args = parse_args()
    model = (
        load_model_from_dir(args.model_dir)
        if args.model_dir
        else load_model(args.model)
    )
    if args.xla:
        model.use_xla = True
        model._predict_step = model._build_predict_step()

    pad_chars = [c.strip() for c in args.pad_chars.split(",") if c.strip()]

    class_map = model.class_map
    classes = [
        c for _, c in sorted(zip(class_map["index"], class_map["class"]), key=lambda t: int(t[0]))
    ]
    print(f"model classes: {classes}")

    records = build_records(args.input, args.num_contigs, args.seed, pad_chars)
    print(f"scoring {len(records)} fragments "
          f"({args.num_contigs} contigs x {1 + len(PAD_LENGTHS) * len(pad_chars)} variants)")

    y_pred = run_inference(model, records, args.batch)

    logits = y_pred["prediction"]
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    reliability = None
    if "reliability" in y_pred:
        reliability = tf.nn.sigmoid(y_pred["reliability"]).numpy().squeeze()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = [
            "contig_id", "frag_id", "pad_char", "pad_len", "real_len",
            "pred_class", "reliability",
        ] + [f"prob_{c}" for c in classes]
        writer.writerow(header)
        for i, rec in enumerate(records):
            pred_class = classes[int(np.argmax(probs[i]))]
            rel = float(reliability[i]) if reliability is not None else ""
            writer.writerow(
                [
                    rec["contig_id"], rec["frag_id"], rec["pad_char"],
                    rec["pad_len"], rec["real_len"], pred_class, rel,
                ]
                + [f"{p:.6f}" for p in probs[i]]
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
