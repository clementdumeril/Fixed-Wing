#!/bin/bash
# One C++ variant, one seed per array task, any number of steps (calm air), then the evaluation on the 100 test
# scenarios. Submit from Fixed-Wing/cpp after savio/build.slurm, e.g. 3 seeds of critic 128 x batch 64 for 3 M steps:
#   sbatch --array=0-2 savio/train.slurm train_sac_c128_b64 3000000
#SBATCH --job-name=fw-cpp-train
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                      # the replay buffer keeps the last 1 M transitions, about 3 GB
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
VARIANT=${1:-train_sac_c128_b64}
STEPS=${2:-3000000}
SEED=$SLURM_ARRAY_TASK_ID
"build/$VARIANT" build/scenarios_calm.txt "runs/${VARIANT}_${STEPS}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" "$STEPS"
