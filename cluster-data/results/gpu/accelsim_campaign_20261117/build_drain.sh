#!/bin/bash
#SBATCH -J gpu-drain-build
#SBATCH -w node03
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 00:30:00
set -Eeuo pipefail
A=$HOME/amcas-pro;E=$A/tracing/mamba/envs/gem5;R=$A/results/gpu/accelsim_campaign_20261117
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib
[ -d "$A/tracing/ramulator2-gpu-drain" ] || cp -a "$A/tracing/ramulator2-grid" "$A/tracing/ramulator2-gpu-drain"
python "$R/drain_patch.py"
cd "$A/tracing/ramulator2-gpu-drain"
cmake -S . -B build-drain -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$E" -DPython_EXECUTABLE="$E/bin/python" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build build-drain -j6
