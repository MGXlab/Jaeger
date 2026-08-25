#!/usr/bin/env bash
# Zeus CV slurm generator: writes 12 training-only scripts to zeus/slurm/
NAMES="cv_fold0_conv_nuc cv_fold1_conv_nuc cv_fold2_conv_nuc cv_fold3_conv_nuc cv_fold4_conv_nuc cv_fold0_conv_trans cv_fold1_conv_trans cv_fold2_conv_trans cv_fold3_conv_trans cv_fold4_conv_trans cv_fold0_conv_hyena_nuc cv_fold0_conv_hyena_trans"
OUT="$(dirname "$0")/slurm"
mkdir -p "$OUT"
for name in $NAMES; do
cat > "$OUT/jaeger_${name}.slurm" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${name}
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/%x_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node030

# 5-fold CV training run (zeus): ${name}

set -euo pipefail

EXPERIMENT_NAME="${name}"
PROJECT_ROOT="/mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger"
CONTAINER="/mnt/beegfs/bioinf/wijesekara/jaeger/container/jaeger_dev.sif"
CONFIG="/mnt/beegfs/bioinf/wijesekara/jaeger/configs/jaeger_\${EXPERIMENT_NAME}.yaml"
META="/mnt/beegfs/bioinf/wijesekara/jaeger/tmp/\${EXPERIMENT_NAME}_train_meta.json"

XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
SCRATCH="\${SCRATCH:-}"

CONTAINER_BIND="\$PROJECT_ROOT/src/jaeger:/usr/local/lib/python3.12/site-packages/jaeger"

cd "\$PROJECT_ROOT"

TRAIN_FLAGS="-c \$CONFIG --save_model --meta \$META --precision bf16 --xla --from_last_checkpoint"
if [[ -n "\$SCRATCH" ]]; then
  TRAIN_FLAGS="\$TRAIN_FLAGS --tmp \$SCRATCH"
fi

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Running on node \$(hostname): GPU \${CUDA_VISIBLE_DEVICES:-none}, job \${SLURM_JOB_ID:-local}, scratch: \$SCRATCH"

# shellcheck disable=SC2086
apptainer run --bind /scratch --bind /mnt/beegfs --bind "\$CONTAINER_BIND" --env "XLA_FLAGS=\${XLA_FLAGS}" --nv "\$CONTAINER" \
  jaeger train \$TRAIN_FLAGS
EOF
done
ls "$OUT"
