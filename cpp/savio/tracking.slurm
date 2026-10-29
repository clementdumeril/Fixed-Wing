#!/bin/bash
# Position and velocity tracking: a reference point flies straight at constant velocity, the airplane has to reach it
# and follow it. Random non-stall start, no trim term in the reward, critic 256 x batch 64, 1 M steps in calm air.
# Three reward settings, 3 seeds each, one core each: position cost scale 50 m without a linear term (the default),
# then 20 m with a linear distance cost of weight 0.15 and 0.3 (at most 0.3: the reward of a step stays >= 0, so that
# ending the flight never pays). Test A: 100 normal flights (neutral start); test B: 100 random starts.
# Submit from Fixed-Wing/cpp after the train_sac_c256_b64_tracking target is built:
# sbatch savio/tracking.slurm
#SBATCH --job-name=fw-cpp-tracking
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SCALES=(50 20 20)
LINEAR=(0 0.15 0.3)
VARIANT=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
SCALE=${SCALES[$VARIANT]}
WEIGHT=${LINEAR[$VARIANT]}
build/train_sac_c256_b64_tracking build/scenarios_calm.txt \
    "runs/tracking_s${SCALE}_l${WEIGHT}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 1000000 \
    --start random --no-trim-reward --position-scale "$SCALE" --position-linear "$WEIGHT"
