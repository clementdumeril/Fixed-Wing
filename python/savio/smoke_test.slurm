#!/bin/bash
# Short run to check the installation on Savio: 20 000 steps, 5 test scenarios, a few minutes.
#SBATCH --job-name=fw-sac-smoke
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc        # billed per core (savio3 bills whole nodes)
#SBATCH --qos=savio_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-smoke-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
python examples/train_sac.py --steps 20000 --seed 0 --n-envs 4 --eval-episodes 5 --out "runs/smoke_${SLURM_JOB_ID}"
