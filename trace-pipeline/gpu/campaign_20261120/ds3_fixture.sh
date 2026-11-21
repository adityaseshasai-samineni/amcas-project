#!/bin/bash
#SBATCH -J ds3-drain-check
#SBATCH -w node03
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:10:00
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;E=$W/mamba/envs/gem5
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$W/ramulator2-gpu-drain:$W/DRAMsim3-gpu-drain PYTHONPATH=$W/ramulator2-gpu-drain/python:$A/tools:$A/gpu:$R OPENBLAS_NUM_THREADS=1
python "$R/ds3_fixture.py"
bash "$A/tools/publish.sh" "Validated true DRAMsim3 write drain and global-feed HBM decoder"
