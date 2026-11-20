#!/bin/bash
#SBATCH -J gpu-legacy-capture
#SBATCH -w node04
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=5G
#SBATCH -t 03:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
CASE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$R/legacy_cases.txt")
trap 'bash "$A/tools/publish.sh" "Historical gem5workload GPU capture $CASE"' EXIT
if ! "$E/bin/python" -c 'import json,sys;assert json.load(open(sys.argv[1]))["passed"]' "$R/runs/$CASE/native.status.json"; then echo "Nativefailed: $CASE";exit 0;fi
"$E/bin/python" "$R/run_case.py" "$CASE" --mode capture
case "$CASE" in g5_s15*) SIMMEM=1G;;*) SIMMEM=4G;;esac
sbatch --mem="$SIMMEM" -J "sim-$CASE" -o "$R/runs/$CASE/simulation_job.log" "$R/simulate_case.sh" "$CASE" | tee "$R/runs/$CASE/simulation_job.txt"
