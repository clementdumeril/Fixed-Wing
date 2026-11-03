#!/bin/bash
# Tracking on PX4's advanced_plane (FW_AIRPLANE=1), the airplane of the Gazebo SITL deployment: critic 128, 400k steps
# with a curriculum check every 40k, the best setting of the Aerosonde runs, 10 seeds.
# Submit from Fixed-Wing/cpp with the train_sac_c128_b64_tracking_g995_px4plane target built:
# sbatch savio/tracking_px4plane.slurm
#SBATCH --job-name=fw-cpp-px4plane
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
build/train_sac_c128_b64_tracking_g995_px4plane build/scenarios_calm.txt \
    "runs/tracking_px4plane_c128_400k_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" 400000 \
    --start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 \
    --alpha-weight 1 --curriculum-period 40000
