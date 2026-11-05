#!/bin/bash
# The tracking task on the first model of the E-flite UMX Eratix (docs/eratix_model.md, FW_AIRPLANE=2) with the
# current best recipe: SAC, critic 128, batch 256, critics updated every 10 steps and the actor every 20 (L2F), 4M
# steps, 10 seeds, best checkpoint every 50k steps.
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_eratix.slurm
#SBATCH --job-name=fw-cpp-eratix
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$SLURM_ARRAY_TASK_ID
build/train_sac_c128_b256_tracking_g995_eratix_i10a20 build/scenarios_calm.txt \
    "runs/tracking_eratix_sac_c128_b256_i10a20_4M_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1 --curriculum-period 40000 --checkpoint-interval 50000
