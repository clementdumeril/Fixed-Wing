#!/bin/bash
# The closeness bonus weight pushed further (results/fine_px4plane, second run: weight 2 took the error from 3.7 to
# 2.4 m, with test success 97.9 % instead of 98.8 %). How far does the error drop, and when does success fall too low?
# Fine observation, adaptive bonus (scale half the training distance), precise checkpoint choice (--checkpoint-precise
# 0.97 1000000). SAC, critic 128, L2F intervals and batch, --reward l2f-penalty, K = 0.3 /s, 4M steps, 10 seeds:
#   tasks 0-9:   fine scale 2 m, bonus weight 3
#   tasks 10-19: fine scale 2 m, bonus weight 4
#   tasks 20-29: fine scale 1 m, bonus weight 2
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_fine3.slurm
#SBATCH --job-name=fw-cpp-px4plane-fine3
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-29
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 10))
FINE=2; BONUS=1; ADAPTIVE=0.5
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) NAME=fine2m_bonus3; BONUS=3 ;;
    1) NAME=fine2m_bonus4; BONUS=4 ;;
    2) NAME=fine1m_bonus2; FINE=1; BONUS=2 ;;
esac
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine build/scenarios_calm.txt \
    "runs/tracking_px4plane_${NAME}_4M_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.3 \
    --fine-position-scale $FINE --position-bonus $BONUS --position-bonus-scale 20 1 --position-bonus-adaptive $ADAPTIVE \
    --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 --checkpoint-precise 0.97 1000000 \
    --reward l2f-penalty
