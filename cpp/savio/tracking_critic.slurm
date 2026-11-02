#!/bin/bash
# Faster tracking training: critics of 256, 128 and 64 neurons, 10 seeds each, 400k steps with a curriculum check
# every 40k steps (instead of 1 M and 100k), on the best setting so far: discount factor 0.995, position cost scale
# 20 m, linear distance cost 0.3, velocity cost toward the point with gain 0.1, margin to the stall with weight 1.
# Almost all the training time is in the network updates, most of it in the 256 x 256 layer of the critic.
# Submit from Fixed-Wing/cpp with the train_sac_c{256,128,64}_b64_tracking_g995 targets built:
# sbatch savio/tracking_critic.slurm
#SBATCH --job-name=fw-cpp-critic
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-29
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
CRITICS=(256 128 64)
CRITIC=${CRITICS[$((SLURM_ARRAY_TASK_ID / 10))]}
SEED=$((SLURM_ARRAY_TASK_ID % 10))
build/train_sac_c${CRITIC}_b64_tracking_g995 build/scenarios_calm.txt \
    "runs/tracking_critic${CRITIC}_400k_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 400000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1 --curriculum-period 40000
