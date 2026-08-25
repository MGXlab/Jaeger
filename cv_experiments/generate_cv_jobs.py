"""Generate 5-fold CV configs and SLURM scripts for brain.

For each of the 8 architecture/input templates (jaeger_experiment_106..113)
and each fold k in 0..4:
  - config: train on the other 4 fold npz files, validate on fold k
  - slurm: training-only job (no reliability data, no predict/metrics)

Outputs:
  cv_experiments/configs/jaeger_cv_fold{k}_{name}.yaml
  cv_experiments/slurm/jaeger_cv_fold{k}_{name}.slurm
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
CONFIGS = HERE / "configs"
SLURM = HERE / "slurm"

TEMPLATE_MAP = {  # template experiment number -> (name, encoding)
    106: ("conv_attention_nuc", "nuc"),
    107: ("conv_attention_trans", "translated"),
    108: ("conv_bilstm_nuc", "nuc"),
    109: ("conv_bilstm_trans", "translated"),
    110: ("conv_hyena_nuc", "nuc"),
    111: ("conv_hyena_trans", "translated"),
    112: ("conv_nuc", "nuc"),
    113: ("conv_trans", "translated"),
}

FOLDS = range(5)
DATA_DIR = "/home/wijesekary/jaeger/data/folds"
EPOCHS = 10
VAL_STEPS = 5000

# bilstm runs are ~4-8x slower -> vision (A100, 3 days); rest -> storm (12 h)
SLOW = {"conv_bilstm_nuc", "conv_bilstm_trans"}

SLURM_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --output=/home/wijesekary/jaeger/slurm/slurm_logs/%x_%j.out
#SBATCH --error=/home/wijesekary/jaeger/slurm/slurm_logs/%x_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1

# 5-fold CV training run: {name} (train folds {train_folds}, val fold {val_fold})

set -euo pipefail

EXPERIMENT_NAME="{name}"
PROJECT_ROOT="/home/wijesekary/jaeger/Jaeger"
CONTAINER="/home/wijesekary/jaeger/container/jaeger_dev.sif"
CONFIG="/home/wijesekary/jaeger/configs/jaeger_${{EXPERIMENT_NAME}}.yaml"
META="/home/wijesekary/jaeger/tmp/${{EXPERIMENT_NAME}}_train_meta.json"

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

. /etc/profile.d/modules.sh
module load singularity/3.11.3

CONTAINER_BIND="$PROJECT_ROOT/src/jaeger:/usr/local/lib/python3.12/site-packages/jaeger"

cd "$PROJECT_ROOT"

singularity run --bind /home/wijesekary/jaeger --bind "$CONTAINER_BIND" --nv "$CONTAINER" \\
  jaeger train -c "$CONFIG" --save_model --meta "$META" \\
  --precision bf16 --xla --from_last_checkpoint
"""


def main() -> None:
    CONFIGS.mkdir(exist_ok=True)
    SLURM.mkdir(exist_ok=True)

    for exp_num, (arch, enc) in TEMPLATE_MAP.items():
        with open(TEMPLATES / f"jaeger_experiment_{exp_num}.yaml") as fh:
            template = yaml.safe_load(fh)

        for k in FOLDS:
            name = f"cv_fold{k}_{arch}"
            cfg = copy.deepcopy(template)

            cfg["model"]["experiment"] = name

            train_folds = [j for j in FOLDS if j != k]
            train_paths = [f"{DATA_DIR}/fold{j}_{enc}.npz" for j in train_folds]
            val_paths = [f"{DATA_DIR}/fold{k}_{enc}.npz"]

            fcd = cfg["training"]["fragment_classifier_data"]
            fcd["train"][0]["path"] = train_paths
            fcd["validation"][0]["path"] = val_paths

            tr = cfg["training"]
            tr["data_dir"] = DATA_DIR
            tr["classifier_epochs"] = EPOCHS
            tr["classifier_train_steps"] = -1  # full data
            tr["classifier_validation_steps"] = VAL_STEPS
            tr["reliability_epochs"] = 0
            tr["projection_epochs"] = 0

            with open(CONFIGS / f"jaeger_{name}.yaml", "w") as fh:
                yaml.dump(cfg, fh, sort_keys=False, default_flow_style=False)

            slow = arch in SLOW
            slurm = SLURM_TEMPLATE.format(
                name=name,
                time="3-00:00:00" if slow else "12:00:00",
                partition="vision" if slow else "storm",
                train_folds=",".join(map(str, train_folds)),
                val_fold=k,
            )
            with open(SLURM / f"jaeger_{name}.slurm", "w") as fh:
                fh.write(slurm)

            print(f"{name}: train folds {train_folds} ({enc}), "
                  f"partition {'vision' if slow else 'storm'}")


if __name__ == "__main__":
    main()
