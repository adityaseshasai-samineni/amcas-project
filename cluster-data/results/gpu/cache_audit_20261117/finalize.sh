#!/bin/bash
#SBATCH --job-name=accel-summary
#SBATCH --nodelist=node04
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
trap 'rc=$?; bash "$HOME/amcas-pro/tools/publish.sh" "Accel-Sim final request reconciliation exit=$rc"' EXIT
python "$R/summarize.py" > "$R/final_summary.stdout" 2> "$R/final_summary.stderr"
