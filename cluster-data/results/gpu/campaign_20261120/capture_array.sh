#!/bin/bash
#SBATCH -J gpu-capture-grid
#SBATCH -w node04
#SBATCH -c 4
#SBATCH --mem=10G
#SBATCH -t 07:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
GROUP=$1
CASE=$("$E/bin/python" -c 'import json,sys;d=json.load(open(sys.argv[1]));g=sys.argv[2];xs=[x["cpu_workload"] for x in d["workloads"] if (x["gpu_devices"]==2)==(g=="moe") and x["cpu_workload"]!="q05_short"];print(xs[int(sys.argv[3])])' "$R/coverage_resolved.json" "$GROUP" "$SLURM_ARRAY_TASK_ID")
trap 'bash "$A/tools/publish.sh" "GPU campaign capture $CASE"' EXIT
if ! "$E/bin/python" -c 'import json,sys;assert json.load(open(sys.argv[1]))["passed"]' "$R/runs/$CASE/native.status.json"; then
 echo "Native feasibility failed; retaining explicit failed case, no invalid capture: $CASE";exit 0
fi
"$E/bin/python" "$R/run_case.py" "$CASE" --mode capture
sbatch -J "sim-$CASE" -o "$R/runs/$CASE/simulation_job.log" "$R/simulate_case.sh" "$CASE" | tee "$R/runs/$CASE/simulation_job.txt"
