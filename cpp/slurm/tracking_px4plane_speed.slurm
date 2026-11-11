#!/bin/bash
# Training time against precision, on the best recipe of results/fine_px4plane (third run, fine2m_bonus4: 1.8 m, 4M steps,
# critic 128, updates every 10 steps; 312 us per step on a laptop core, 95 % of it in the network updates, 2 % in the
# airplane). Each variant changes one thing; time per step measured with --profile over 100k steps on the laptop:
#   tasks 0-9:   critic 64 (as in L2F), 4M steps                                   130 us per step
#   tasks 10-19: critic 32 (as RLtools' L2F zoo, rl/zoo/l2f/td3.h), 4M steps       86 us
#   tasks 20-29: critic 64, updates every 16 steps (as that zoo), 4M steps         81 us
#   tasks 30-39: critic 32, updates every 16 steps, 4M steps                        55 us
#   tasks 40-49: critic 128, curriculum gate every 50k steps instead of 100k, 4M    312 us
#   tasks 50-59: critic 128, 8M steps (the error still fell at 4M: 2.2 m at 3M)    312 us
# Fine observation (2 m), adaptive closeness bonus of weight 4, precise checkpoint choice, checkpoint every 50k steps.
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_speed.slurm
#SBATCH --job-name=fw-cpp-px4plane-speed
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-59
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 10))
TARGET=train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine; STEPS=4000000; PERIOD=100000
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) NAME=c64; TARGET=train_sac_c64_b256_tracking_g995_px4plane_i10a20_fine ;;
    1) NAME=c32; TARGET=train_sac_c32_b256_tracking_g995_px4plane_i10a20_fine ;;
    2) NAME=c64_i16; TARGET=train_sac_c64_b256_tracking_g995_px4plane_i16a32_fine ;;
    3) NAME=c32_i16; TARGET=train_sac_c32_b256_tracking_g995_px4plane_i16a32_fine ;;
    4) NAME=c128_gate50k; PERIOD=50000 ;;
    5) NAME=c128_8M; STEPS=8000000 ;;
esac
build/$TARGET build/scenarios_calm.txt \
    "runs/tracking_px4plane_speed_${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" "$STEPS" \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.3 \
    --fine-position-scale 2 --position-bonus 4 --position-bonus-scale 20 1 --position-bonus-adaptive 0.5 \
    --alpha-weight 1 --curriculum-period "$PERIOD" --checkpoint-interval 50000 --checkpoint-precise 0.97 1000000 \
    --reward l2f-penalty
