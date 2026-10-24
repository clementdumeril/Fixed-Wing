#!/bin/bash
# Compile every C++ target on a compute node (-march=native matches the nodes of the sweep), export the 100 test
# scenarios and check the C++ port against the Python environment. Submit from Fixed-Wing/cpp after savio/setup.sh.
#SBATCH --job-name=fw-cpp-build
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm-build-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 cmake/3.27.7 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
cmake --build build -j "$SLURM_CPUS_PER_TASK"
python tools/export_scenarios.py build/scenarios_calm.txt
python tools/compare_with_python.py build/check
