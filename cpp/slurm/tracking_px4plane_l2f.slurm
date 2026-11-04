#!/bin/bash
# The last two differences with the L2F paper on the advanced_plane, all with its batch (256) and update intervals
# (critics every 10 steps, actor every 20), 4M steps, 10 seeds, best checkpoint every 50k steps:
#   tasks 0-9:   SAC, critic 64 (as L2F) instead of 128
#   tasks 10-19: TD3 (as L2F), critic 64
#   tasks 20-29: SAC, critic 128: the reference (results/l2f_intervals_px4plane), again in the same job for the times
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_l2f.slurm
#SBATCH --job-name=fw-cpp-px4plane-l2f
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
    0) TARGET=train_sac_c64_b256_tracking_g995_px4plane_i10a20; NAME=sac_c64_b256_i10a20_4M ;;
    1) TARGET=train_td3_c64_b256_tracking_g995_px4plane_i10a20; NAME=td3_c64_b256_i10a20_4M ;;
    2) TARGET=train_sac_c128_b256_tracking_g995_px4plane_i10a20; NAME=sac_c128_b256_i10a20_4M ;;
esac
build/$TARGET build/scenarios_calm.txt "runs/tracking_px4plane_${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1 --curriculum-period 40000 --checkpoint-interval 50000
