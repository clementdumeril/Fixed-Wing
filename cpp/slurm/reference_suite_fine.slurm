#!/bin/bash
# The reference suite (results/reference_suite: circles, helices, figure-eights, speed changes, accelerations swept
# past what the airplane can follow) for the most precise actors on the straight line, trained with the fine position
# observation (results/fine_px4plane). Evaluation only, best checkpoint of each seed, 100 flights per level from level
# and random starts, the first 3 flights of each level recorded:
#   tasks 0-9:   fine2m_bonus4 (third run, job 41045217): 1.8 m on the straight line
#   tasks 10-19: fine2m_bonus2 (second run, job 41037545): 2.4 m, commands twice smoother
# Needs the training runs in runs/ and evaluate_reference_fine. Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/reference_suite_fine.slurm
#SBATCH --job-name=fw-cpp-reference-fine
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-19
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
SEED=$((SLURM_ARRAY_TASK_ID % 10))
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) NAME=fine2m_bonus4; JOB=41045217 ;;
    1) NAME=fine2m_bonus2; JOB=41037545 ;;
esac
RUN="runs/tracking_px4plane_${NAME}_4M_${JOB}_seed${SEED}"
build/evaluate_reference_fine "$RUN/best_actor.h" "runs/reference_suite_fine/${NAME}_seed${SEED}" \
    --flights 100 --record 3 --threads "$SLURM_CPUS_PER_TASK" --fine-position-scale 2
