#!/bin/bash
# Does a margin to the stall in the reward remove the early stalls? Critic 256 x batch 64, 1 M steps in calm air,
# weight of the angle-of-attack margin term 0 (reference), 0.3 and 1, 3 seeds each, one core each.
# Submit from Fixed-Wing/cpp after savio/build.slurm: sbatch savio/alpha_margin.slurm
#SBATCH --job-name=fw-cpp-alpha
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
WEIGHTS=(0 0.3 1)
WEIGHT=${WEIGHTS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
build/train_sac_c256_b64 build/scenarios_calm.txt "runs/alpha${WEIGHT}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 1000000 \
    --alpha-weight "$WEIGHT"
