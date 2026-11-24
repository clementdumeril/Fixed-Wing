#!/bin/bash
#SBATCH --job-name=fw-cpp-px4plane-smooth2
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 4))
VARIANT=$((SLURM_ARRAY_TASK_ID / 4))
NAMES=(rate2_beta5_air beta10_air rate2_beta10_air beta20_air)
OPTIONS=("--rate-weight 2 --sideslip-weight 5 --smooth-ramp 1000000 3000000 --reference-air-relative" "--sideslip-weight 10 --smooth-ramp 1000000 3000000 --reference-air-relative" "--rate-weight 2 --sideslip-weight 10 --smooth-ramp 1000000 3000000 --reference-air-relative" "--sideslip-weight 20 --smooth-ramp 1000000 3000000 --reference-air-relative")
NAME=${NAMES[$VARIANT]}
RUN="runs/tracking_px4plane_smooth2_${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}"
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine_wind build/scenarios_calm.txt "$RUN" "$SEED" 8000000 \
    --reward simple --energy-weight 0 \
    --distance-max 30 --near-start 1 5 --start trimmed --reference-heading-max 20 \
    --curriculum-period 100000 --checkpoint-interval 50000 --checkpoint-precise 0.97 1000000 --eval-duration 120 ${OPTIONS[$VARIANT]}
build/evaluate_smoothness "$RUN/best_actor.h" "$RUN/smoothness_best"
build/evaluate_smoothness "$RUN/best_actor.h" "$RUN/smoothness_best_air" --reference-air-relative
