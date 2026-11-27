#!/bin/bash
#SBATCH -J gpu-xworkload
#SBATCH -w node01
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 20:00:00
# Cross-workload fixed-mask test (xworkload.py) over all GPU policy windows; resumable.
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;E=$W/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign cross-workload fixed-mask test"' EXIT
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$W/ramulator2-gpu-drain PYTHONPATH=$W/ramulator2-gpu-drain/python:$A/tools:$A/gpu OPENBLAS_NUM_THREADS=1
mkdir -p "$R/xworkload"
python "$R/xworkload.py" "$R/xworkload/results.json" "/scratch/$USER/amcas-xworkload" 6
