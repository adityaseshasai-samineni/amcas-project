#!/bin/bash
#SBATCH -J gpu-upload
#SBATCH -w node04
#SBATCH -c 1
#SBATCH --mem=512M
#SBATCH -t 02:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign validated raw admissions assets $1"' EXIT
"$E/bin/python" "$R/upload_case.py" "$1" < "$HOME/.config/amcas/gh_token"
