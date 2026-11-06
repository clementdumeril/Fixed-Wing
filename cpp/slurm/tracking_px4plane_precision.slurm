#!/bin/bash
# Closing the ~10 m steady lag of the L2F-reward policies (results/l2f_reward_px4plane): the squared position cost is
# flat near the point and the velocity correction toward it weak (0.1 /s). SAC, critic 128, L2F intervals and batch,
# --reward l2f-penalty, 4M steps, 10 seeds, best checkpoint every 50k steps:
#   tasks 0-9:   + a linear distance cost, 0.1 per meter (--position-l1 0.1)
#   tasks 10-19: velocity correction toward the point 0.3 /s instead of 0.1 (--velocity-guidance 0.3)
#   tasks 20-29: both
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_precision.slurm
#SBATCH --job-name=fw-cpp-px4plane-precision
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
    0) NAME=l1; OPTIONS="--velocity-guidance 0.1 --position-l1 0.1" ;;
    1) NAME=guidance03; OPTIONS="--velocity-guidance 0.3" ;;
    2) NAME=l1_guidance03; OPTIONS="--velocity-guidance 0.3 --position-l1 0.1" ;;
esac
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20 build/scenarios_calm.txt \
    "runs/tracking_px4plane_precision_${NAME}_4M_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 $OPTIONS \
    --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 --reward l2f-penalty
