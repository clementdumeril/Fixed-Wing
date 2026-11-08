#!/bin/bash
# More of the training spent following the point (results/precision_px4plane: reward terms for being close took the
# error to 6.7-7.7 m at best; the training flights start up to 50 m away, so most of the experience is the approach).
# A share of the training starts puts the point within 5 m (horizontally and vertically) of the airplane, whose state is
# drawn as usual (random start); validation and test flights keep the 0-50 m starts. SAC, critic 128, L2F intervals and
# batch, --reward l2f-penalty, K = 0.3 /s, 4M steps, 10 seeds, best checkpoint every 50k steps:
#   tasks 0-9:   half of the starts near the point
#   tasks 10-19: 80 % of the starts near the point
#   tasks 20-29: half of the starts near the point, with the adaptive closeness bonus of the third precision run
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_near.slurm
#SBATCH --job-name=fw-cpp-px4plane-near
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
    0) NAME=near50; OPTIONS="--near-start 0.5 5" ;;
    1) NAME=near80; OPTIONS="--near-start 0.8 5" ;;
    2) NAME=near50_bonus_adaptive; OPTIONS="--near-start 0.5 5 --position-bonus 1 --position-bonus-scale 20 1 --position-bonus-adaptive 0.5" ;;
esac
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20 build/scenarios_calm.txt \
    "runs/tracking_px4plane_${NAME}_4M_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.3 \
    $OPTIONS --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 --reward l2f-penalty
