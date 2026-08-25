#!/usr/bin/env bash
# Encode the 5 fold CSVs into nuc + translated NPZ files with the same
# parameters as the original frag_2000 NPZ conversion.
set -uo pipefail

cd /home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/Jaeger
FOLDS=/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/data_generation/cv_folds
LOG="${FOLDS}/encode_folds.log"

for k in 0 1 2 3 4; do
  for fmt in nucleotide translated; do
    out="${FOLDS}/fold${k}_${fmt/nucleotide/nuc}.npz"
    if [[ -s "$out" ]]; then
      echo "===== skip fold${k} ${fmt} (exists) =====" | tee -a "$LOG"
      continue
    fi
    echo "===== fold${k} ${fmt} start $(date -Is) =====" | tee -a "$LOG"
    jaeger utils optimize-data \
      -i "${FOLDS}/fold${k}.csv" \
      -o "$out" \
      --format "$fmt" \
      --crop-size 2000 --crop-units nucleotide --stride 0 \
      --num-classes 6 \
      --balance-classes --shuffle-seed 42 \
      --max-memory-mb 16000 \
      >> "$LOG" 2>&1
    rc=$?
    echo "===== fold${k} ${fmt} exit ${rc} $(date -Is) =====" | tee -a "$LOG"
    if [[ $rc -ne 0 ]]; then
      echo "FAILED: fold${k} ${fmt}" | tee -a "$LOG"
      exit $rc
    fi
  done
done
echo "ALL DONE $(date -Is)" | tee -a "$LOG"
