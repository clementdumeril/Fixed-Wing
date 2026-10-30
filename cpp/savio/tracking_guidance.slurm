#!/bin/bash
# Tracking task with the velocity cost on a commanded velocity that leads to the reference point: reference velocity
# - gain x position error, the correction at most 5 m/s; gain 0.1 and 0.2 /s, 3 seeds each, with the setting of
# savio/tracking_gamma.slurm (discount factor 0.995, position cost scale 20 m, linear distance cost 0.3). With the
# reference velocity alone, the policies flew at that velocity 10 to 50 m behind the point, since closing the gap
# costs velocity error. The reference without guidance is the tracking_g995 runs of that script.
# Submit from Fixed-Wing/cpp after rebuilding the train_sac_c256_b64_tracking_g995 target:
# sbatch savio/tracking_guidance.slurm                               1 M steps
# sbatch --export=ALL,STEPS=3000000 savio/tracking_guidance.slurm    any number of steps (about 1 h per million)
#SBATCH --job-name=fw-cpp-guide
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
GAINS=(0.1 0.2)
GAIN=${GAINS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
STEPS=${STEPS:-1000000}
build/train_sac_c256_b64_tracking_g995 build/scenarios_calm.txt \
    "runs/tracking_guide${GAIN}_$((STEPS / 1000))k_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" "$STEPS" \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance "$GAIN"
