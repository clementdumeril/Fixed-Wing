#!/bin/bash
# The position error also observed on a fine scale (results/near_px4plane: the error left is a steady offset of 3-4 m,
# specific to each actor, with 0.5 m of motion around it; 4 m is only 0.2 in the observation scaled by 20 m). The actor
# and the critic also see tanh(error / S) in the body frame, so that near the point one meter counts 20 / S times more.
# SAC, critic 128, L2F intervals and batch, --reward l2f-penalty, K = 0.3 /s, 4M steps, 10 seeds, best checkpoint every
# 50k steps:
#   tasks 0-9:   fine scale 2 m
#   tasks 10-19: fine scale 5 m
#   tasks 20-29: fine scale 2 m, with the adaptive closeness bonus of the third precision run (the reward also zoomed)
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_fine.slurm
#SBATCH --job-name=fw-cpp-px4plane-fine
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-29
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 10))
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) NAME=fine2m; OPTIONS="--fine-position-scale 2" ;;
    1) NAME=fine5m; OPTIONS="--fine-position-scale 5" ;;
    2) NAME=fine2m_bonus_adaptive; OPTIONS="--fine-position-scale 2 --position-bonus 1 --position-bonus-scale 20 1 --position-bonus-adaptive 0.5" ;;
esac
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine build/scenarios_calm.txt \
    "runs/tracking_px4plane_${NAME}_4M_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.3 \
    $OPTIONS --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 --reward l2f-penalty
