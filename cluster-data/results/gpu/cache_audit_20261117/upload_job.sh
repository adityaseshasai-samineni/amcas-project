#!/bin/bash
#SBATCH --job-name=accel-upload
#SBATCH --nodelist=node04
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
trap 'rc=$?; bash "$R/publish_context.sh" "V100 trace asset publication exit=$rc; see reconciled simulation summary and trace_asset.json"' EXIT
python "$R/upload_trace.py" > "$R/upload_trace.stdout" 2> "$R/upload_trace.stderr"
