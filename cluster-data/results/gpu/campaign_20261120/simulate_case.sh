#!/bin/bash
#SBATCH -J accel-case
#SBATCH -w node04
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 24:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign simulation $1"' EXIT
export PATH=$E/bin:$PATH OPENBLAS_NUM_THREADS=1
python "$R/simulate_case.py" "$1"
