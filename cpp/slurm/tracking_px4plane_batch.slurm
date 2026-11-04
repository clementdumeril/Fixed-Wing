#!/bin/bash
# Batch 256 on the advanced_plane, 10 seeds per variant, best checkpoint every 50k steps:
#   tasks 0-9:   critic 128, batch 256, updates at every step, 400k steps (4 seeds on the lab PC: results/batch_px4plane)
#   tasks 10-19: the same with the update intervals of the L2F paper (critics every 10 steps, actor every 20), 400k
#                steps: 10 times fewer updates
#   tasks 20-29: the L2F intervals for 4M steps: as many critic updates as tasks 0-9
# Submit from Fixed-Wing/cpp with the train_sac_c128_b256_tracking_g995_px4plane{,_i10a20} targets built:
# sbatch slurm/tracking_px4plane_batch.slurm
#SBATCH --job-name=fw-cpp-px4plane-batch
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-29
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 10))
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) TARGET=train_sac_c128_b256_tracking_g995_px4plane; NAME=c128_b256_400k; STEPS=400000 ;;
    1) TARGET=train_sac_c128_b256_tracking_g995_px4plane_i10a20; NAME=c128_b256_i10a20_400k; STEPS=400000 ;;
    2) TARGET=train_sac_c128_b256_tracking_g995_px4plane_i10a20; NAME=c128_b256_i10a20_4M; STEPS=4000000 ;;
esac
build/$TARGET build/scenarios_calm.txt "runs/tracking_px4plane_${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" $STEPS \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1 --curriculum-period 40000 --checkpoint-interval 50000
