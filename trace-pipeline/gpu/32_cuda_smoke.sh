#!/bin/bash
# Smoke test: model MODEL (default 0.5B), -ngl 99, greedy, 16 tokens. Usage: 32_cuda_smoke.sh [MODEL.gguf] [BUILD]
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export PATH=/usr/local/cuda-12.6/bin:$PATH LD_LIBRARY_PATH=${CUDALIB:-/usr/local/cuda-12.6/lib64}:$E/lib
M=${1:-$W/models/qwen25-05b-q4_0.gguf}; B=${2:-build-cuda}
P="The capital of France is"
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
$W/llama.cpp/$B/bin/llama-completion -m $M -ngl ${NGL:-99} -n 24 --temp 0 --no-warmup -no-cnv -p "$P" ${EXTRA:-} 2>&1 | grep -v -e '^llama_' -e '^load' -e '^print_info' -e '^ggml_' | cut -c1-220 | grep -e "^The" -e "CUDA" -e "eval time" -e "Device" -e "error"
