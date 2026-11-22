#!/bin/bash
#SBATCH -J accel-case
#SBATCH -w node03
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 3-00:00:00
# Identical to simulate_case.sh except: runs on node03, first copies the capture
# trace from node04:/scratch to the same path on node03 (verified by file count
# and byte total), and pins the upload job to node03 where the outputs live.
# The replay's own output (S/simulation) is excluded from copy and comparison.
# Simulator binary, V100 configs, case JSON and Python steps are unchanged.
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign simulation $1"' EXIT
export PATH=$E/bin:$PATH OPENBLAS_NUM_THREADS=1
S=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["scratch"])' "$R/runs/$1/capture.status.json")
mkdir -p "$S"
rsync -a --partial --exclude=/simulation/ node04:"$S/" "$S/"
src=$(ssh -o BatchMode=yes node04 "find '$S' -path '$S/simulation' -prune -o -type f -printf '%s\n' | awk '{n++;b+=\$1}END{print n,b}'")
dst=$(find "$S" -path "$S/simulation" -prune -o -type f -printf '%s\n' | awk '{n++;b+=$1}END{print n,b}')
echo "trace copy node04 -> node03: src=[$src] dst=[$dst]" | tee "$R/runs/$1/trace_copy_node03.txt"
[ "$src" = "$dst" ] || { echo "trace copy mismatch"; exit 3; }
if [ ! -s "$R/runs/$1/hardware_job.txt" ]; then
 DEV=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["gpu_devices"])' "$R/cases/$1.json")
 if [ "$DEV" = 2 ]; then MEM=22G; else MEM=6G; fi
 sbatch --gres="gpu:$DEV" --mem="$MEM" -J "ncu-$1" -o "$R/runs/$1/hardware_job.log" "$R/hardware_case.sh" "$1" | tee "$R/runs/$1/hardware_job.txt"
fi
python "$R/simulate_case.py" "$1"
python "$R/reconcile_case.py" "$1"
python "$R/prepare_policy.py" "$1"
for t in 1 2 3 4; do sbatch -w node03 -J "upload-$1" -o "$R/runs/$1/upload_job.log" "$R/upload_case.sh" "$1" | tee "$R/runs/$1/upload_job.txt" && break; sleep 30; done
