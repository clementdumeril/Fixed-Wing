#!/bin/bash
# Time of each part of a training step (airplanes, batch, critics, critic targets, actor and entropy coefficient) for
# the critics of 256, 128 and 64 neurons on the best tracking setting: 30k steps, 10k of warmup then 20k updates.
# Submit from Fixed-Wing/cpp with the train_sac_c{256,128,64}_b64_tracking_g995 targets built:
# sbatch savio/profile.slurm
#SBATCH --job-name=fw-cpp-profile
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
CRITICS=(256 128 64)
CRITIC=${CRITICS[$SLURM_ARRAY_TASK_ID]}
build/train_sac_c${CRITIC}_b64_tracking_g995 build/scenarios_calm.txt \
    "runs/profile_critic${CRITIC}_${SLURM_ARRAY_JOB_ID}" 0 30000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1 --curriculum-period 40000 --profile
