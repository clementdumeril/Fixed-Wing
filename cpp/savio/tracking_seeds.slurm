#!/bin/bash
# The two best tracking settings on 10 seeds each, to tell them apart (3 seeds spread from 39 to 97% of completed
# flights): discount factor 0.995, position cost scale 20 m, linear distance cost 0.3, without the velocity cost toward
# the point (gain 0) and with it (gain 0.1). Seeds 0 to 2 repeat the runs of savio/tracking_gamma.slurm and
# savio/tracking_guidance.slurm: on the same machine type they must give the same results.
# Submit from Fixed-Wing/cpp with the train_sac_c256_b64_tracking_g995 target built:
# sbatch savio/tracking_seeds.slurm
#SBATCH --job-name=fw-cpp-seeds
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
GAINS=(0 0.1)
GAIN=${GAINS[$((SLURM_ARRAY_TASK_ID / 10))]}
SEED=$((SLURM_ARRAY_TASK_ID % 10))
build/train_sac_c256_b64_tracking_g995 build/scenarios_calm.txt \
    "runs/tracking_seeds_guide${GAIN}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 1000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance "$GAIN"
