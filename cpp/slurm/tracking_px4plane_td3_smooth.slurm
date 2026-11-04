#!/bin/bash
# TD3 with the L2F intervals and batch flew bang-bang commands (results/l2f_td3_px4plane): two ways to smooth them,
# 4M steps, 10 seeds each, best checkpoint every 50k steps:
#   tasks 0-9:   10 times the weight of the command-change cost (--action-change-scale 10)
#   tasks 10-19: half the exploration noise (0.05 instead of 0.1)
#   tasks 20-29: both
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_td3_smooth.slurm
#SBATCH --job-name=fw-cpp-px4plane-td3-smooth
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
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
    0) TARGET=train_td3_c64_b256_tracking_g995_px4plane_i10a20; NAME=td3_c64_i10a20_change10_4M; SCALE=10 ;;
    1) TARGET=train_td3_c64_b256_tracking_g995_px4plane_i10a20_n005; NAME=td3_c64_i10a20_noise005_4M; SCALE=1 ;;
    2) TARGET=train_td3_c64_b256_tracking_g995_px4plane_i10a20_n005; NAME=td3_c64_i10a20_noise005_change10_4M; SCALE=10 ;;
esac
build/$TARGET build/scenarios_calm.txt "runs/tracking_px4plane_${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1 --curriculum-period 40000 --checkpoint-interval 50000 --action-change-scale $SCALE
