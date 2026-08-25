#!/usr/bin/env bash
set -euo pipefail

cd /home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/Jaeger

configs=(
  nuc_vs_trans_experiments/conv_trans.yaml
  nuc_vs_trans_experiments/conv_nuc.yaml
  nuc_vs_trans_experiments/conv_attention_trans.yaml
  nuc_vs_trans_experiments/conv_attention_nuc.yaml
  nuc_vs_trans_experiments/conv_bilstm_trans.yaml
  nuc_vs_trans_experiments/conv_bilstm_nuc.yaml
  nuc_vs_trans_experiments/conv_hyena_trans.yaml
  nuc_vs_trans_experiments/conv_hyena_nuc.yaml
)

mkdir -p nuc_vs_trans_experiments/logs

for cfg in "${configs[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="nuc_vs_trans_experiments/logs/${name}.log"
  exp_dir="nuc_vs_trans_experiments/experiments/experiment_${name}_24876"
  if [[ -f "${exp_dir}/checkpoints/classifier/training_history.csv" ]]; then
    echo "===== Skipping $name (history already exists) =====" | tee -a "$log"
    continue
  fi
  echo "===== Starting $name at $(date -Is) =====" | tee -a "$log"
  jaeger train -c "$cfg" --force 2>&1 | tee -a "$log"
  echo "===== Finished $name at $(date -Is) =====" | tee -a "$log"
done
