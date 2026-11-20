#!/bin/bash
#SBATCH -J gpu-native-grid
#SBATCH -w node04
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 04:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign native feasibility and outputs"' EXIT
export OPENBLAS_NUM_THREADS=1
"$E/bin/python" "$R/resolve_cases.py"
while read -r CASE; do
 "$E/bin/python" "$R/run_case.py" "$CASE" --mode native || true
done < <("$E/bin/python" -c 'import json,os;print("\n".join(x["cpu_workload"] for x in json.load(open("'"$R"'/coverage_resolved.json"))["workloads"] if os.environ.get("AMCAS_GROUP","all")=="all" or (x["gpu_devices"]==2)==(os.environ.get("AMCAS_GROUP")=="moe")))')
