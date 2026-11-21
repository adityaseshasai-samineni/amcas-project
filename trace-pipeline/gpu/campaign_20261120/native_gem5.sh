#!/bin/bash
#SBATCH -J gpu-legacy-native
#SBATCH -w node04
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 00:30:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "Historical gem5workload GPU feasibility and coverage"' EXIT
"$E/bin/python" "$R/resolve_gem5.py"
while read -r CASE; do "$E/bin/python" "$R/run_case.py" "$CASE" --mode native || true;done < "$R/legacy_cases.txt"
