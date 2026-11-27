#!/bin/bash
#SBATCH -J gpu-holdout
#SBATCH -w node03
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
# Held-out window check (holdout.py) on the full node03 DRAM streams of the key cases.
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;E=$W/mamba/envs/gem5
trap 'bash "$A/tools/publish.sh" "GPU campaign held-out window check"' EXIT
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$W/ramulator2-gpu-drain PYTHONPATH=$W/ramulator2-gpu-drain/python:$A/tools:$A/gpu OPENBLAS_NUM_THREADS=1
mkdir -p "$R/holdout"
for c in "$@"; do
 for d in /scratch/$USER/amcas-campaign20/$c/capture/simulation/ctx_*; do
  o=$R/holdout/${c}_$(basename $d | tail -c 5).json
  # 72 replays per context: 2 interleaves x (4 + 7 + 7 mappings) x 2 policies
  python -c 'import json,sys;sys.exit(0 if len(json.load(open(sys.argv[1])))>=72 else 1)' "$o" 2>/dev/null && continue
  rm -f "$o"; python "$R/holdout.py" "$d/masked.dram" "$o" 4
 done
done
