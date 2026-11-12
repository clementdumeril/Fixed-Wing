#!/bin/bash
# The reward of RLtools' L2F environment at b32d9985 (l2f/parameters/reward_functions/squared): despite the name, its
# terms are norms (distance to the point, velocity error, command change), not squares. Our L2F form squares them, so a
# meter of error near the point costs almost nothing (2.4 x (1 / 20)^2 = 0.006 at the last stage), which the closeness
# bonus and the fine observation had to make up for. --reward l2f-norm: the same form, floor -2 and termination
# penalty as l2f-penalty, with the norms (0.12 per meter at the last stage, the same near the point as far from it).
# Best recipe otherwise (results/fine_px4plane third run, fine2m_bonus4: 1.8 m at 4M steps): SAC, critic 128, L2F
# intervals and batch, K = 0.3 /s, fine observation (2 m), precise checkpoint choice, 4M steps, 10 seeds:
#   tasks 0-9:   l2f-norm with the closeness bonus of weight 4 (the best recipe, linear terms)
#   tasks 10-19: l2f-norm without the bonus (the RLtools form as it is, with the fine observation)
#   tasks 20-29: l2f-norm without the bonus, position scale 10 m instead of 20 (twice the cost per meter)
# Submit from Fixed-Wing/cpp after slurm/build.slurm:
# sbatch slurm/tracking_px4plane_reward.slurm
#SBATCH --job-name=fw-cpp-px4plane-reward
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
SCALE=20; BONUS=""
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) NAME=norm_bonus4; BONUS="--position-bonus 4 --position-bonus-scale 20 1 --position-bonus-adaptive 0.5" ;;
    1) NAME=norm ;;
    2) NAME=norm_scale10; SCALE=10 ;;
esac
build/train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine build/scenarios_calm.txt \
    "runs/tracking_px4plane_reward_${NAME}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 4000000 \
    --start random --no-trim-reward --position-scale "$SCALE" --position-linear 0.3 --velocity-guidance 0.3 \
    --fine-position-scale 2 $BONUS --alpha-weight 1 --curriculum-period 100000 --checkpoint-interval 50000 \
    --checkpoint-precise 0.97 1000000 --reward l2f-norm
