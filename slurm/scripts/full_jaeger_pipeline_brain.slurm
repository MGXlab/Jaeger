#!/usr/bin/env bash
#SBATCH --job-name=jaeger
#SBATCH --output=/home/wijesekary/jaeger/slurm/slurm_logs/%x_%j.out
#SBATCH --error=/home/wijesekary/jaeger/slurm/slurm_logs/%x_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=vision
#SBATCH --gres=gpu:1

# Full Jaeger training-to-evaluation pipeline for Brain (multi-GPU A100).
#
# 1. Train fragment classifier (+ optional self-supervised projection pretraining)
# 2. Generate reliability (OOD) training data
# 3. Train reliability head
# 4. Tune the reliability threshold (inside 'jaeger train') and save the model
# 5. Predict on real-world metagenome assemblies
# 6. Calculate precision/recall/F1/confusion matrices against fraction labels
#
# Usage on Brain:
#   sbatch slurm/scripts/full_jaeger_pipeline_brain.slurm
#
# Local testing (no SLURM / singularity): every configuration variable can be
# overridden from the environment, e.g.
#   MODE=local PROJECT_ROOT=$PWD JAEGER_BIN=/path/to/jaeger \
#     JAEGER_TRAIN_CONFIG=/path/to/config.yaml TEST_DATA="a.fasta b.fasta" \
#     TEST_LABELS_DIR=/path/to/labels bash slurm/scripts/full_jaeger_pipeline_brain.slurm

set -euo pipefail

# ============================================================================
# ---- Configuration (all overridable via environment) ----
# ============================================================================

MODE="${MODE:-brain}"  # brain | local

EXPERIMENT_NUMBER="${EXPERIMENT_NUMBER:-1}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/wijesekary/jaeger/Jaeger}"
CONTAINER="${CONTAINER:-/home/wijesekary/jaeger/container/jaeger_dev.sif}"
JAEGER_TRAIN_CONFIG="${JAEGER_TRAIN_CONFIG:-$PROJECT_ROOT/train_config/jaeger_experiment_${EXPERIMENT_NUMBER}.yaml}"
JAEGER_MODEL_CONFIG="${JAEGER_MODEL_CONFIG:-/home/wijesekary/jaeger/configs/config.json}"
SINGULARITY_METADATA="${SINGULARITY_METADATA:-/home/wijesekary/jaeger/tmp/${EXPERIMENT_NUMBER}_train_meta.json}"

# Executables for local mode (ignored in brain mode, where the container provides them)
JAEGER_BIN="${JAEGER_BIN:-jaeger}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# Training flags
FROM_LAST_CHECKPOINT="${FROM_LAST_CHECKPOINT:-true}"
FORCE="${FORCE:-false}"
ONLY_HEADS="${ONLY_HEADS:-false}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"
PRECISION="${PRECISION:-bf16}"
XLA="${XLA:-true}"
GENERATE_RELIABILITY_DATA="${GENERATE_RELIABILITY_DATA:-true}"
SELF_SUPERVISED_PRETRAINING="${SELF_SUPERVISED_PRETRAINING:-true}"

# Inference settings
STRIDE="${STRIDE:-1500}"
FSIZE="${FSIZE:-2000}"
MIN_LEN="${MIN_LEN:-1500}"
BATCH="${BATCH:-96}"
DYNAMIC_STRIDE="${DYNAMIC_STRIDE:-true}"
DYNAMIC_STRIDE_THRESHOLD="${DYNAMIC_STRIDE_THRESHOLD:-10.0}"
USE_BEST_RC="${USE_BEST_RC:-true}"
WRITE_WINDOW_SCORES="${WRITE_WINDOW_SCORES:-false}"
OVERWRITE="${OVERWRITE:-false}"

# Test data (space-separated FASTA paths; overridable via environment)
TEST_DATA="${TEST_DATA:-/home/wijesekary/jaeger/benchmark_data/gut_scaffolds_gt1500.fasta /home/wijesekary/jaeger/benchmark_data/seawater_scaffolds_gt1500.fasta /home/wijesekary/jaeger/benchmark_data/soil_scaffolds_gt1500.fasta}"
read -ra TEST_FASTAS <<<"$TEST_DATA"

# Real-world fraction labels: one TSV per sample, named <sample>_labels.tsv
TEST_LABELS_DIR="${TEST_LABELS_DIR:-/home/wijesekary/jaeger/benchmark_data}"

# Fallback reliability cutoff, used only when training did not write a tuned
# threshold (reliability_threshold.txt). The tuned threshold is on the
# probability scale and matches the metric configured under
# training.reliability_threshold.metric in the training YAML.
BEST_RC="${BEST_RC:-0}"

# Logs
PIPELINE_LOG="${PIPELINE_LOG:-$(dirname "$SINGULARITY_METADATA")/${EXPERIMENT_NUMBER}_pipeline.log}"

# ============================================================================
# ---- Helper functions ----
# ============================================================================

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >>"$PIPELINE_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

file_exists() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

# Mirror of the prefix matching done by calculate_metrics_realworld.py:
# predictions inherit FASTA names (e.g. gut_scaffolds_gt1500) while labels are
# stored as <sample>_labels.tsv (e.g. gut_labels). Match the longest
# underscore-separated prefix of the FASTA stem that has a label file.
find_label_file() {
  local candidate="$1"
  while true; do
    if [[ -f "$TEST_LABELS_DIR/${candidate}_labels.tsv" ]]; then
      printf '%s\n' "$TEST_LABELS_DIR/${candidate}_labels.tsv"
      return 0
    fi
    [[ "$candidate" == *_* ]] || break
    candidate="${candidate%_*}"
  done
  return 1
}

# Run a step directly on this node (containerized on brain, plain locally).
run_step() {
  "$@" 2>&1 | tee -a "$PIPELINE_LOG"
}

# The sbatch batch shell already runs on the allocated compute node, so every
# step inherits the job's allocation (GPU, CPUs) directly — no nested srun.
log_node_info() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running on node $(hostname): GPU ${CUDA_VISIBLE_DEVICES:-none}, job ${SLURM_JOB_ID:-local}" >&2
}

# ============================================================================
# ---- Pre-flight checks ----
# ============================================================================

mkdir -p "$(dirname "$SINGULARITY_METADATA")"
mkdir -p "$(dirname "$PIPELINE_LOG")"

cd "$PROJECT_ROOT" || die "Could not cd to $PROJECT_ROOT"

log "Starting Jaeger pipeline (mode=$MODE) for experiment: $EXPERIMENT_NUMBER"

if [[ "$MODE" == "brain" ]]; then
  # Load Singularity module inside the batch job
  . /etc/profile.d/modules.sh
  module load singularity/3.11.3
  file_exists "$CONTAINER"
  command -v singularity >/dev/null 2>&1 || die "singularity is required but not installed"
  CONTAINER_BIND="$PROJECT_ROOT/src/jaeger:/usr/local/lib/python3.12/site-packages/jaeger"
  JAEGER_RUN=(singularity run --bind /home/wijesekary/jaeger --bind "$CONTAINER_BIND" --nv "$CONTAINER" jaeger)
  PY_RUN=(singularity exec --bind /home/wijesekary/jaeger --bind "$CONTAINER_BIND" --nv "$CONTAINER" python3)
else
  command -v "$JAEGER_BIN" >/dev/null 2>&1 || die "jaeger executable not found: $JAEGER_BIN"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python not found: $PYTHON_BIN"
  JAEGER_RUN=("$JAEGER_BIN")
  PY_RUN=("$PYTHON_BIN")
fi

file_exists "$JAEGER_TRAIN_CONFIG"
file_exists "$JAEGER_MODEL_CONFIG"

for fasta in "${TEST_FASTAS[@]}"; do
  file_exists "$fasta"
  sample_name=$(basename "$fasta" .fasta)
  find_label_file "$sample_name" >/dev/null \
    || die "No label file matching sample '$sample_name' in $TEST_LABELS_DIR"
done

# ============================================================================
# ---- Training ----
# ============================================================================

TRAIN_FLAGS="-c $JAEGER_TRAIN_CONFIG --save_model --meta $SINGULARITY_METADATA"

if [[ "$MIXED_PRECISION" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --precision $PRECISION"
fi

if [[ "$FORCE" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --force"
fi

if [[ "$FROM_LAST_CHECKPOINT" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --from_last_checkpoint"
fi

if [[ "$ONLY_HEADS" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --only_heads"
fi

if [[ "$XLA" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --xla"
fi

if [[ "$GENERATE_RELIABILITY_DATA" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --generate_reliability_data"
fi

if [[ "$SELF_SUPERVISED_PRETRAINING" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --self_supervised_pretraining"
fi

log "Training Jaeger model with flags: $TRAIN_FLAGS"
log_node_info

# shellcheck disable=SC2086
run_step "${JAEGER_RUN[@]}" train $TRAIN_FLAGS || die "Training failed"

# ============================================================================
# ---- Extract model and experiment paths from metadata ----
# ============================================================================

[[ -f "$SINGULARITY_METADATA" ]] || die "Training metadata file was not created: $SINGULARITY_METADATA"

MODEL_PATH=$("${PY_RUN[@]}" -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$SINGULARITY_METADATA")
EXPERIMENT_PATH=$("${PY_RUN[@]}" -c "import json,sys; print(json.load(open(sys.argv[1]))['experiment_path'])" "$SINGULARITY_METADATA")

[[ -n "$MODEL_PATH" && "$MODEL_PATH" != "null" ]] || die "MODEL_PATH is empty"
[[ -n "$EXPERIMENT_PATH" && "$EXPERIMENT_PATH" != "null" ]] || die "EXPERIMENT_PATH is empty"

log "MODEL_PATH=$MODEL_PATH"
log "EXPERIMENT_PATH=$EXPERIMENT_PATH"

# Resolve the reliability cutoff: when USE_BEST_RC=true, prefer the threshold
# tuned during training and fall back to BEST_RC; otherwise force 0 so the
# reliability head does not filter any predictions.
if [[ "$USE_BEST_RC" == true ]]; then
  THRESHOLD_FILE="$EXPERIMENT_PATH/checkpoints/reliability/reliability_threshold.txt"
  if [[ -f "$THRESHOLD_FILE" ]]; then
    BEST_RC=$(tr -d '[:space:]' <"$THRESHOLD_FILE")
    log "Using tuned reliability threshold: $BEST_RC"
  else
    log "No tuned threshold found at $THRESHOLD_FILE; using BEST_RC=$BEST_RC"
  fi
else
  BEST_RC=0
  log "USE_BEST_RC=false; using reliability cutoff 0"
fi

# ============================================================================
# ---- Inference on test metagenomes ----
# ============================================================================

PREDICTIONS_DIR="$EXPERIMENT_PATH/predictions"
mkdir -p "$PREDICTIONS_DIR"
mkdir -p "$EXPERIMENT_PATH/metrics"

for fasta in "${TEST_FASTAS[@]}"; do
  sample_name=$(basename "$fasta" .fasta)

  if [[ "$OVERWRITE" != true ]] && compgen -G "$PREDICTIONS_DIR/*/${sample_name}.tsv" >/dev/null; then
    log "Skipping prediction for $sample_name (output exists)"
    continue
  fi

  log "Running prediction on: $sample_name"

  WINDOW_FLAG=""
  if [[ "$WRITE_WINDOW_SCORES" == true ]]; then
    WINDOW_FLAG="--window-scores"
  fi

  OVERWRITE_FLAG=""
  if [[ "$OVERWRITE" == true ]]; then
    OVERWRITE_FLAG="--overwrite"
  fi

  DYNAMIC_STRIDE_FLAG=""
  if [[ "$DYNAMIC_STRIDE" == true ]]; then
    DYNAMIC_STRIDE_FLAG="--dynamic-stride --dynamic-stride-threshold $DYNAMIC_STRIDE_THRESHOLD"
  fi

  # shellcheck disable=SC2086
  run_step "${JAEGER_RUN[@]}" predict \
    --model_path "$MODEL_PATH" \
    --config "$JAEGER_MODEL_CONFIG" \
    -i "$fasta" \
    -o "$PREDICTIONS_DIR" \
    --stride "$STRIDE" \
    --fsize "$FSIZE" \
    --min-len "$MIN_LEN" \
    --batch "$BATCH" \
    --rc "$BEST_RC" \
    $DYNAMIC_STRIDE_FLAG \
    $WINDOW_FLAG \
    $OVERWRITE_FLAG \
    || die "Prediction failed for $sample_name"
done

# ============================================================================
# ---- Calculate metrics ----
# ============================================================================

# 'jaeger predict' writes TSVs to <output>/<model_id>/<sample>.tsv; resolve the
# single model subdirectory that was just produced.
mapfile -t PRED_SUBDIRS < <(find "$PREDICTIONS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
[[ ${#PRED_SUBDIRS[@]} -ge 1 ]] || die "No prediction outputs found under $PREDICTIONS_DIR"
if [[ ${#PRED_SUBDIRS[@]} -gt 1 ]]; then
  log "WARNING: multiple prediction subdirs found; using ${PRED_SUBDIRS[0]}"
fi
PRED_OUTPUT_DIR="${PRED_SUBDIRS[0]}"

log "Calculating real-world metrics from $PRED_OUTPUT_DIR"

run_step "${PY_RUN[@]}" "$PROJECT_ROOT/scripts/calculate_metrics_realworld.py" \
  --predictions-dir "$PRED_OUTPUT_DIR" \
  --labels-dir "$TEST_LABELS_DIR" \
  --output-dir "$EXPERIMENT_PATH/metrics" \
  --reliability-cutoff "$BEST_RC" \
  || die "Metric calculation failed"

log "Pipeline completed successfully"
log "Metrics written to: $EXPERIMENT_PATH/metrics"
