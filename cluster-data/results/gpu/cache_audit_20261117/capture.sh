#!/bin/bash
#SBATCH --job-name=accel-q7short
#SBATCH --nodelist=node04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
W=$HOME/amcas-pro/tracing
A=$W/accel-sim-audit-20261117
S=/scratch/$USER/accel-q7short-20261117
trap 'rc=$?; echo "$rc" > "$R/capture_exit.txt"; bash "$HOME/amcas-pro/tools/publish.sh" "Accel-Sim Qwen short capture exit=$rc"' EXIT
mkdir -p "$S"; cd "$S"
export PATH=/usr/local/cuda-12.6/bin:$W/mamba/envs/gem5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$W/mamba/envs/gem5/lib:$W/accel-build-deps/lib
export GGML_CUDA_DISABLE_GRAPHS=1 CUDA_MODULE_LOADING=EAGER NOBANNER=1
# Custom launches are0-based; officialtracerstarts1. Laststep1295..1832 ->1296..1833 inclusive.
export DYNAMIC_KERNEL_RANGE=1296-1833 TOOL_COMPRESS=1 TOOL_TRACE_CORE=1
/usr/bin/time -v env LD_PRELOAD="$A/util/tracer_nvbit/tracer_tool/tracer_tool.so" "$W/llama.cpp/build-cuda/bin/llama-completion" -m "$W/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" -ngl 99 -n 3 --temp 0 --seed 1 --no-warmup -no-cnv -t 4 -c 4096 -p 'The capital of France is' -fa auto > "$R/capture.stdout" 2> "$R/capture.stderr"
"$A/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing" traces -j 4 > "$R/capture_post.log" 2>&1
ls -lh traces > "$R/capture_inventory.txt"
cp traces/kernelslist.g "$R/capture_kernelslist.g"
