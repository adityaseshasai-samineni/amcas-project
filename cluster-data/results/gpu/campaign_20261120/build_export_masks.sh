#!/bin/bash
#SBATCH -J accel-mask-export
#SBATCH -w node04
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 00:15:00
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;F=$W/accel-sim-audit-20261117
export PATH=/usr/local/cuda-12.6/bin:$W/mamba/envs/gem5/bin:$PATH LD_LIBRARY_PATH=$W/mamba/envs/gem5/lib:$W/accel-build-deps/lib
python "$R/patch_export_masks.py" "$F/gpu-simulator/gpgpu-sim/src/gpgpu-sim/dram.cc"
cmake --build "$F/gpu-simulator/build-cmake" --target accel-sim.out -j4
cp "$F/gpu-simulator/gpgpu-sim/src/gpgpu-sim/dram.cc" "$R/dram_mask_export.cc"
