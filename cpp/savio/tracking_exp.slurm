#!/bin/bash
# Tracking task with the position out of the saturating cost: weight x (1 - exp(-distance / 30 m)), weight 0.3 and
# 0.4, 3 seeds each, discount factor 0.998 (train_sac_c256_b64_tracking_g998). With the position inside the capped
# cost (savio/tracking.slurm, tracking_gamma.slurm), the quadratic term at the stage 5 weight is flat beyond about 12 m
# and the policy plateaus 25 to 35 m from the reference. Submit from Fixed-Wing/cpp after rebuilding the g998 target:
# sbatch savio/tracking_exp.slurm                               1 M steps
# sbatch --export=ALL,STEPS=3000000 savio/tracking_exp.slurm    any number of steps (about 1 h per million)
#SBATCH --job-name=fw-cpp-exp
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
WEIGHTS=(0.3 0.4)
WEIGHT=${WEIGHTS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
STEPS=${STEPS:-1000000}
build/train_sac_c256_b64_tracking_g998 build/scenarios_calm.txt \
    "runs/tracking_exp${WEIGHT}_$((STEPS / 1000))k_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" "$STEPS" \
    --start random --no-trim-reward --position-exp "$WEIGHT"
