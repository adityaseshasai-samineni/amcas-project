#!/bin/bash
#SBATCH -J gpu-batch-build
#SBATCH -w node04
#SBATCH -c 4
#SBATCH --mem=6G
#SBATCH -t 00:10:00
set -Eeuo pipefail
A=$HOME/amcas-pro; W=$A/tracing; R=$A/results/gpu/campaign_20261120
export PATH=/usr/local/cuda-12.6/bin:$W/mamba/envs/gem5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$W/mamba/envs/gem5/lib:$W/llama.cpp/build-cuda/bin
B=$W/llama.cpp/build-cuda/bin
c++ -O3 -std=c++17 -rdynamic -I"$W/llama.cpp/include" -I"$W/llama.cpp/ggml/include" -I/usr/local/cuda-12.6/include \
 "$R/gpu_batch_main.cpp" -L"$B" -L/usr/local/cuda-12.6/lib64 -L/usr/local/cuda-12.6/lib64/stubs -Wl,-rpath,"$B":/usr/local/cuda-12.6/lib64 \
 -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -lcudart -lcuda -ldl -lpthread -o "$R/llama-gpu-batch"
