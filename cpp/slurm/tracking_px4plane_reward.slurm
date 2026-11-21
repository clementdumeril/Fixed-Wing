#!/bin/bash
# The precision recipe on PX4's advanced_plane (results/reward_px4plane, norm_bonus4: 1.4 m position error, 98 % of the
# test flights completed, 16.7 min per run): SAC, critic 128, batch 256, L2F update intervals, the position error also
# observed on a 2 m scale, --reward l2f-norm (the L2F form with the norms of the errors, as RLtools' L2F reward), velocity
# guidance K = 0.3 /s, closeness bonus of weight 4, stall margin, random starts, the most precise checkpoint with at least
# 97 % validation success after 1M steps, 4M steps, 10 seeds.
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_reward.slurm
#SBATCH --job-name=fw-cpp-px4plane-reward
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$SLURM_ARRAY_TASK_ID
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine build/scenarios_calm.txt \
    "runs/tracking_px4plane_reward_norm_bonus4_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --reward l2f-norm --start random --position-scale 20 --velocity-guidance 0.3 --alpha-weight 1 --position-bonus 4 \
    --curriculum-period 100000 --checkpoint-interval 50000 --checkpoint-precise 0.97 1000000
