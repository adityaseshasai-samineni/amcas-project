#!/bin/bash
#SBATCH --job-name=accel-corefix
#SBATCH --nodelist=node04
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
A=$HOME/amcas-pro/tracing/accel-sim-audit-20261117
S=/scratch/$USER/accel-q7short-20261117
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=$HOME/amcas-pro/tracing/accel-build-deps/lib:$HOME/amcas-pro/tracing/mamba/envs/gem5/lib
cd "$A/util/tracer_nvbit/tracer_tool/traces-processing"
make -j2
# The ongoing capture's snapshotted script invokes this rebuilt binary after capture.
touch "$S/.amcas_trace_core" "$S/bench/.amcas_trace_core"
# Validate the repaired full-core format on the completed representative kernel.
bash "$R/bench_retry2.sh"
