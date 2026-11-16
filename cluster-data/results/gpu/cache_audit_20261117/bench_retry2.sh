#!/bin/bash
#SBATCH --job-name=accel-matvec
#SBATCH --nodelist=node04
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
A=$HOME/amcas-pro/tracing/accel-sim-audit-20261117
S=/scratch/$USER/accel-q7short-20261117
export LD_LIBRARY_PATH=$HOME/amcas-pro/tracing/accel-build-deps/lib:$HOME/amcas-pro/tracing/mamba/envs/gem5/lib
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
mkdir -p "$S/bench"; cd "$S/bench"
f=$(find "$S/traces" -name 'kernel-1298-*.trace.xz' | head -1)
ln -sf "$f" "$(basename "$f")"
basename "$f" > kernelslist
/usr/bin/time -v "$A/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing" "$PWD/kernelslist" -j 2 > "$R/bench_post.log" 2> "$R/bench_post.time"
cp "$R/preflight/"*.config "$R/preflight/"*.icnt "$R/preflight/"*.xml .
AMCAS_DRAM_EXPORT="$S/bench_dram.csv" /usr/bin/time -v "$A/gpu-simulator/build-cmake/accel-sim.out" -config gpgpusim.config -config trace.config -trace "$PWD/kernelslist.g" > "$R/bench_sim.log" 2> "$R/bench_sim.time"
