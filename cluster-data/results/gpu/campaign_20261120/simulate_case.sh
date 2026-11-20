#!/bin/bash
#SBATCH -J accel-case
#SBATCH -w node04
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 24:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign simulation $1"' EXIT
export PATH=$E/bin:$PATH OPENBLAS_NUM_THREADS=1
if [ ! -s "$R/runs/$1/hardware_job.txt" ]; then
 DEV=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["gpu_devices"])' "$R/cases/$1.json")
 if [ "$DEV" = 2 ]; then MEM=22G; else MEM=6G; fi
 sbatch --gres="gpu:$DEV" --mem="$MEM" -J "ncu-$1" -o "$R/runs/$1/hardware_job.log" "$R/hardware_case.sh" "$1" | tee "$R/runs/$1/hardware_job.txt"
fi
python "$R/simulate_case.py" "$1"
python "$R/prepare_policy.py" "$1"
sbatch -J "upload-$1" -o "$R/runs/$1/upload_job.log" "$R/upload_case.sh" "$1" | tee "$R/runs/$1/upload_job.txt"
