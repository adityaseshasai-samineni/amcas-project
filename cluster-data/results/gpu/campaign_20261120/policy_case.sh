#!/bin/bash
#SBATCH -J gpu-policy-grid
#SBATCH -w node03
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 08:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;E=$W/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign full mapping-policy grid"' EXIT
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$W/ramulator2-gpu-drain PYTHONPATH=$W/ramulator2-gpu-drain/python:$A/tools:$A/gpu OPENBLAS_NUM_THREADS=1
python "$R/policy_grid.py" "$1" "$2" "$3" --size 4000000 --procs 2 --source-metadata "$4"
