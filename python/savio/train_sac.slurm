#!/bin/bash
# One SAC training per seed (job array), each on its own node share, then evaluation on the 100 test scenarios.
#SBATCH --job-name=fw-sac
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc        # billed per core (savio3 bills whole nodes)
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
python examples/train_sac.py --steps 1000000 --seed "$SLURM_ARRAY_TASK_ID" --n-envs 8 \
    --out "runs/sac_${SLURM_ARRAY_JOB_ID}_seed${SLURM_ARRAY_TASK_ID}"
