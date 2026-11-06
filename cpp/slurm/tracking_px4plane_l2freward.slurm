#!/bin/bash
# The reward form of L2F (--reward l2f-penalty: survival minus squared errors, floor -2, termination penalty, the
# L2F schedule every 100k steps) with the L2F intervals and batch, 4M steps, 10 seeds, best checkpoint every 50k
# steps (results/l2f_reward_px4plane: 4 seeds on the lab PC):
#   tasks 0-9:   SAC, critic 128 (compare with results/l2f_td3_px4plane, same recipe with our tanh reward)
#   tasks 10-19: TD3, critic 64: are its commands smooth without the flat part of the tanh?
#   tasks 20-29: SAC, critic 64
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_l2freward.slurm
#SBATCH --job-name=fw-cpp-px4plane-l2freward
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
    0) TARGET=train_sac_c128_b256_tracking_g995_px4plane_i10a20; NAME=sac_c128_l2freward_4M ;;
    1) TARGET=train_td3_c64_b256_tracking_g995_px4plane_i10a20; NAME=td3_c64_l2freward_4M ;;
    2) TARGET=train_sac_c64_b256_tracking_g995_px4plane_i10a20; NAME=sac_c64_l2freward_4M ;;
esac
build/$TARGET build/scenarios_calm.txt "runs/tracking_px4plane_${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 --reward l2f-penalty
