#!/bin/bash
#SBATCH --job-name=accel-build
#SBATCH --nodelist=node04
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -Eeo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
trap 'rc=$?; echo "$rc" > "$R/accel_build_exit.txt"; bash "$HOME/amcas-pro/tools/publish.sh" "Accel-Sim build feasibility exit=$rc"' EXIT
export CUDA_INSTALL_PATH=/usr/local/cuda-12.6
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$CUDA_INSTALL_PATH/bin:$PATH
cd "$HOME/amcas-pro/tracing/accel-sim-audit-20261117/gpu-simulator"
source setup_environment.sh release
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake -j4
