#!/bin/bash
#SBATCH -J gpu-qdepth
#SBATCH -c 3
#SBATCH --mem=12G
#SBATCH -t 08:00:00
#SBATCH -x node04,node05
# Ramulator read-buffer depth sensitivity (queue_depth.py) for q7_8k and q30_short (first context).
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;E=$W/mamba/envs/gem5;I=$W/campaign20-grid-input
trap 'bash "$A/tools/publish.sh" "GPU campaign Ramulator queue-depth sensitivity"' EXIT
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$W/ramulator2-gpu-drain PYTHONPATH=$W/ramulator2-gpu-drain/python:$A/tools:$A/gpu OPENBLAS_NUM_THREADS=1
mkdir -p "$R/queue_depth"
for c in q7_8k:ctx_0x55b7730780e0 q30_short:ctx_0x5598806f5f00; do
 n=${c%%:*};x=${c#*:}
 python "$R/queue_depth.py" "$I/$n/$x/window.dram" "$R/runs/$n/capture.tensors.tsv" "$R/queue_depth/$n.json" 3
done
