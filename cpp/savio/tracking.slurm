#!/bin/bash
# Position and velocity tracking: a reference point flies straight at constant velocity, the airplane has to reach it
# and follow it. Random non-stall start, no trim term in the reward, critic 256 x batch 64, 1 M steps in calm air,
# 10 seeds, one core each. Test A: 100 normal flights (neutral start); test B: 100 random starts.
# Submit from Fixed-Wing/cpp after the train_sac_c256_b64_tracking target is built: sbatch savio/tracking.slurm
#SBATCH --job-name=fw-cpp-tracking
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$SLURM_ARRAY_TASK_ID
build/train_sac_c256_b64_tracking build/scenarios_calm.txt "runs/tracking_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 1000000 \
    --start random --no-trim-reward
