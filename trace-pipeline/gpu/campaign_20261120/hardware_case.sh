#!/bin/bash
#SBATCH -J gpu-hardware
#SBATCH -w node04
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=6G
#SBATCH -t 00:45:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign independent hardware counters $1"' EXIT
"$E/bin/python" "$R/hardware_case.py" "$1"
"$E/bin/python" "$R/reconcile_case.py" "$1"
