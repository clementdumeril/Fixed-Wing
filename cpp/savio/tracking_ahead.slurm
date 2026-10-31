#!/bin/bash
# The best tracking setting with the point starting ahead of the airplane: within 45 deg of its course instead of
# anywhere around it (half the default starts put the point behind, which takes steep turns to reach; the early stalls
# are the same on tests A and B, so they come from the point, not from the airplane's attitude). 10 seeds, to compare
# with the gain 0.1 runs of savio/tracking_seeds.slurm on tests A, B and the new evaluation_ahead.json.
# Submit from Fixed-Wing/cpp after rebuilding the train_sac_c256_b64_tracking_g995 target:
# sbatch savio/tracking_ahead.slurm
#SBATCH --job-name=fw-cpp-ahead
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
build/train_sac_c256_b64_tracking_g995 build/scenarios_calm.txt \
    "runs/tracking_ahead45_guide0.1_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 1000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --reference-bearing-max 45
