#!/bin/bash
# The steady lag again (results/precision_px4plane: K = 0.3 /s took it from 5.9 to 3.2 m behind the point, a linear
# distance cost made it worse). A closeness bonus exp(-(d / s)^2) whose scale the curriculum shrinks from 10 m to 1 m
# asks for "be close to the point" whatever the point does (the gain K is tied to the velocity target). SAC, critic
# 128, L2F intervals and batch, --reward l2f-penalty, 4M steps, 10 seeds, best checkpoint every 50k steps:
#   tasks 0-9:   closeness bonus 1, K = 0.3 /s
#   tasks 10-19: closeness bonus 1, K = 0.1 /s (the bonus alone does the work)
#   tasks 20-29: K = 0.5 /s, no bonus
#   tasks 30-39: closeness bonus 1, K = 0.3 /s and the integral of the position error in the observation
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_precision2.slurm
#SBATCH --job-name=fw-cpp-px4plane-precision2
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
TARGET=train_sac_c128_b256_tracking_g995_px4plane_i10a20
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) NAME=bonus_guidance03; K=0.3; BONUS=1 ;;
    1) NAME=bonus_guidance01; K=0.1; BONUS=1 ;;
    2) NAME=guidance05; K=0.5; BONUS=0 ;;
    3) NAME=bonus_guidance03_integral; K=0.3; BONUS=1; TARGET=${TARGET}_integral ;;
esac
build/$TARGET build/scenarios_calm.txt \
    "runs/tracking_px4plane_precision_${NAME}_4M_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance $K \
    --position-bonus $BONUS --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 --reward l2f-penalty
