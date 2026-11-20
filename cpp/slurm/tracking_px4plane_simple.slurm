#!/bin/bash
#SBATCH --job-name=fw-cpp-px4plane-simple
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 4))
ENERGY_WEIGHTS=(0 0.1 0.3)
ENERGY_WEIGHT=${ENERGY_WEIGHTS[$((SLURM_ARRAY_TASK_ID / 4))]}
RUN_TAG=${RUN_TAG:-simple}
EXTRA_OPTIONS=${EXTRA_OPTIONS:-}
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine_wind build/scenarios_calm.txt \
    "runs/tracking_px4plane_${RUN_TAG}_energy${ENERGY_WEIGHT}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 8000000 \
    --reward simple --energy-weight "$ENERGY_WEIGHT" --energy-ramp 1000000 3000000 \
    --distance-max 30 --near-start 1 5 --start trimmed --reference-heading-max 20 --fine-position-scale 2 \
    --curriculum-period 100000 --checkpoint-interval 50000 --checkpoint-precise 0.97 1000000 --eval-duration 120 $EXTRA_OPTIONS
