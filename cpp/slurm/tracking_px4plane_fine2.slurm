#!/bin/bash
# Further on the fine observation with the closeness bonus (results/fine_px4plane: S = 2 m and the adaptive bonus took
# the error to 4.9 m; the most precise validation checkpoint with >= 97 % success was at 3.3 m). Every task keeps that
# checkpoint as the best one (--checkpoint-precise 0.97 1000000) instead of the default score (10 m per % failed).
# SAC, critic 128, L2F intervals and batch, --reward l2f-penalty, K = 0.3 /s, 4M steps, 10 seeds, checkpoint every 50k:
#   tasks 0-9:   fine scale 2 m, adaptive bonus (scale half the training distance), weight 1: the fine run again,
#                with the precise checkpoint choice
#   tasks 10-19: fine scale 1 m, same bonus
#   tasks 20-29: fine scale 2 m, bonus weight 2
#   tasks 30-39: fine scale 2 m, bonus scale a quarter of the training distance (--position-bonus-adaptive 0.25)
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_fine2.slurm
#SBATCH --job-name=fw-cpp-px4plane-fine2
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-39
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 10))
FINE=2; BONUS=1; ADAPTIVE=0.5
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) NAME=fine2m_bonus1 ;;
    1) NAME=fine1m_bonus1; FINE=1 ;;
    2) NAME=fine2m_bonus2; BONUS=2 ;;
    3) NAME=fine2m_bonus1_quarter; ADAPTIVE=0.25 ;;
esac
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine build/scenarios_calm.txt \
    "runs/tracking_px4plane_${NAME}_4M_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.3 \
    --fine-position-scale $FINE --position-bonus $BONUS --position-bonus-scale 20 1 --position-bonus-adaptive $ADAPTIVE \
    --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 --checkpoint-precise 0.97 1000000 \
    --reward l2f-penalty
