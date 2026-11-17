#!/bin/bash
#SBATCH -J drain-fixture
#SBATCH -w node05
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -t 00:05:00
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/accelsim_campaign_20261117;E=$A/tracing/mamba/envs/gem5
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$A/tracing/ramulator2-gpu-drain PYTHONPATH=$A/gpu:$A/tools:$A/tracing/ramulator2-gpu-drain/python OPENBLAS_NUM_THREADS=1
python "$R/policy_pilot.py" "$R/fixture.dram" "$R/fixture" --size 1000 --windows begin --procs 2
