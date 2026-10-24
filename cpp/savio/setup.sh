#!/bin/bash
# Configure the C++ build on a Savio login node (light: downloads RLtools and its dependencies, compiles nothing).
# Then: sbatch savio/build.slurm, then sbatch savio/sweep.slurm. Run from Fixed-Wing/cpp:  source savio/setup.sh
module load gcc/11.4.0 cmake/3.27.7 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
pip install -q mkl-devel   # Intel MKL for the network computations, as on the laptop
MKL_CONFIG=$(find "$VIRTUAL_ENV" -name MKLConfig.cmake | head -n 1)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DMKL_DIR="$(dirname "$MKL_CONFIG")" -DMKL_ROOT="$VIRTUAL_ENV" -DMKL_THREADING=sequential -DMKL_INTERFACE=lp64 \n    -DRL_TOOLS_DISABLE_TENSORBOARD=ON -DRL_TOOLS_DISABLE_HDF5=ON -DRL_TOOLS_DISABLE_CLI11=ON  # not used, and the system
                                                                                    # protobuf (3.5) is too old
mkdir -p runs
