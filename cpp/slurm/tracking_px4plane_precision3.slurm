#!/bin/bash
# The closeness bonus tested properly (results/precision_px4plane: in the second run its scale shrank on a schedule to
# 1 m while the training flights were still 10-40 m away, so it was about 0 all along). SAC, critic 128, L2F intervals
# and batch, --reward l2f-penalty, K = 0.3 /s, bonus weight 1, 4M steps, 10 seeds, best checkpoint every 50k steps:
#   tasks 0-9:   bonus scale fixed at 10 m
#   tasks 10-19: bonus scale fixed at 5 m
#   tasks 20-29: adaptive scale: half the mean training distance to the point at every curriculum check, from 20 m
#                down to 1 m at the lowest, never growing (--position-bonus-adaptive 0.5)
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_precision3.slurm
#SBATCH --job-name=fw-cpp-px4plane-precision3
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
    0) NAME=bonus10m; BONUS_OPTIONS="--position-bonus-scale 10 10" ;;
    1) NAME=bonus5m; BONUS_OPTIONS="--position-bonus-scale 5 5" ;;
    2) NAME=bonus_adaptive; BONUS_OPTIONS="--position-bonus-scale 20 1 --position-bonus-adaptive 0.5" ;;
esac
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20 build/scenarios_calm.txt \
    "runs/tracking_px4plane_precision_${NAME}_4M_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.3 \
    --position-bonus 1 $BONUS_OPTIONS --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 \
    --reward l2f-penalty
