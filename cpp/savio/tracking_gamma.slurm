#!/bin/bash
# Tracking task with a longer critic horizon: discount factor 0.995 (about 4 s) and 0.998 (about 10 s), 3 seeds each,
# with the best reward setting of savio/tracking.slurm (position cost scale 20 m, linear distance cost 0.3). The 0.99
# reference is the tracking_s20_l0.3 runs of that script: same seeds, same machine type.
# Submit from Fixed-Wing/cpp after the train_sac_c256_b64_tracking_g995 and _g998 targets are built:
# sbatch savio/tracking_gamma.slurm                               1 M steps
# sbatch --export=ALL,STEPS=3000000 savio/tracking_gamma.slurm    any number of steps (about 1 h per million)
#SBATCH --job-name=fw-cpp-gamma
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
GAMMAS=(995 998)
GAMMA=${GAMMAS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
STEPS=${STEPS:-1000000}
build/train_sac_c256_b64_tracking_g${GAMMA} build/scenarios_calm.txt \
    "runs/tracking_g${GAMMA}_$((STEPS / 1000))k_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" "$STEPS" \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3
