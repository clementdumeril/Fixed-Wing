#!/bin/bash
# Short run to check the installation on Savio: 20 000 steps, 5 test scenarios, a few minutes.
#SBATCH --job-name=fw-sac-smoke
#SBATCH --account=REPLACE_ME            # your Savio account (fc_..., co_...)
#SBATCH --partition=REPLACE_ME          # e.g. savio3
#SBATCH --qos=REPLACE_ME                # e.g. savio_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-smoke-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.venvs/fixed-wing/bin/activate
python examples/train_sac.py --steps 20000 --seed 0 --n-envs 4 --eval-episodes 5 --out "runs/smoke_${SLURM_JOB_ID}"
