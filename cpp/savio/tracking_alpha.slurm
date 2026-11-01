#!/bin/bash
# Tracking with a margin to the stall in the reward, outside the cap: weights 0.3 and 1, 10 seeds each, on the best
# setting so far (discount factor 0.995, position cost scale 20 m, linear distance cost 0.3, velocity cost toward the
# point with gain 0.1). With gain 0.1 and no margin (savio/tracking_seeds.slurm), most failures were stalls in the
# first second, the policy pulling up to 15 deg of angle of attack; the margin costs within 4 deg of that limit.
# Submit from Fixed-Wing/cpp with the train_sac_c256_b64_tracking_g995 target built:
# sbatch savio/tracking_alpha.slurm
#SBATCH --job-name=fw-cpp-alpha
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
WEIGHTS=(0.3 1)
WEIGHT=${WEIGHTS[$((SLURM_ARRAY_TASK_ID / 10))]}
SEED=$((SLURM_ARRAY_TASK_ID % 10))
build/train_sac_c256_b64_tracking_g995 build/scenarios_calm.txt \
    "runs/tracking_alpha${WEIGHT}_guide0.1_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 1000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight "$WEIGHT"
