#!/bin/bash
# Tracking with the integral of the position error in the observation (forgetting time 10 s), 10 seeds, on the best
# setting so far: discount factor 0.995, position cost scale 20 m, linear distance cost 0.3, velocity cost toward the
# point with gain 0.1, margin to the stall with weight 1 (savio/tracking_alpha.slurm, the same runs without the
# integral). Without it, each seed kept its own steady offset to the point, mostly in altitude (-18 to +24 m).
# Submit from Fixed-Wing/cpp with the train_sac_c256_b64_tracking_g995_integral target built:
# sbatch savio/tracking_integral.slurm
#SBATCH --job-name=fw-cpp-integral
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$SLURM_ARRAY_TASK_ID
build/train_sac_c256_b64_tracking_g995_integral build/scenarios_calm.txt \
    "runs/tracking_integral_alpha1_guide0.1_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 1000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1
