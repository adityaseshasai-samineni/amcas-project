#!/bin/bash
#SBATCH -J gpu-fresh-q05
#SBATCH -w node04
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 07:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign fresh0.5B official capture"' EXIT
"$E/bin/python" "$R/run_case.py" q05_short --mode capture
