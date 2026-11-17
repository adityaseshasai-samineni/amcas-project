#!/bin/bash
#SBATCH -J accel-policy
#SBATCH -w node04
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 04:00:00
set -Eeuo pipefail
A=$HOME/amcas-pro
E=$A/tracing/mamba/envs/gem5
R=$A/results/gpu/accelsim_campaign_20261117
D=/scratch/$USER/accel-q7short-20261117
trap 'bash "$A/tools/publish.sh" "Accel-Sim HBM2 translated policy grid"' EXIT
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$A/tracing/ramulator2-gpu-drain PYTHONPATH=$A/gpu:$A/tools:$A/tracing/ramulator2-gpu-drain/python OPENBLAS_NUM_THREADS=1
python "$R/policy_pilot.py" "$D/canonical.dram" "$R/preflight" --size 1000 --windows begin --procs 4
python "$R/policy_pilot.py" "$D/canonical.dram" "$R/window100k" --size 100000 --procs 4
python "$R/policy_pilot.py" "$D/canonical.dram" "$R/window1m" --size 1000000 --procs 4
