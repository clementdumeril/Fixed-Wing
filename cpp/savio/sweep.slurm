#!/bin/bash
# Network and batch size sweep: 6 variants x 3 seeds, one core each, 1 M steps in calm air, then the evaluation on the
# 100 test scenarios. Submit from Fixed-Wing/cpp after savio/build.slurm: sbatch savio/sweep.slurm
#SBATCH --job-name=fw-cpp-sweep
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                      # the replay buffer of 1 M transitions takes about 3 GB
#SBATCH --time=08:00:00
#SBATCH --array=0-17
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
# critic hidden size x batch size; train_sac is 256 x 256 (the Python run), train_sac_small 64 x 64 (L2F)
VARIANTS=(train_sac train_sac_c256_b64 train_sac_c128_b128 train_sac_c128_b64 train_sac_c64_b256 train_sac_small)
VARIANT=${VARIANTS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
"build/$VARIANT" build/scenarios_calm.txt "runs/${VARIANT}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 1000000
