#!/bin/bash
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
W=$HOME/amcas-pro/tracing
export PATH=/usr/local/cuda-12.6/bin:$W/mamba/envs/gem5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$W/mamba/envs/gem5/lib
export GGML_CUDA_DISABLE_GRAPHS=1 CUDA_MODULE_LOADING=EAGER
for fa in auto off; do
 if [ "$fa" = auto ]; then skip=1295; count=538; else skip=1379; count=566; fi
 for replay in application kernel; do
  /opt/nvidia/nsight-compute/2024.3.2/ncu --csv --replay-mode "$replay" --cache-control none --clock-control none --launch-skip "$skip" --launch-count "$count" --metrics dram__bytes_read.sum,dram__bytes_write.sum "$W/llama.cpp/build-cuda/bin/llama-completion" -m "$W/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" -ngl 99 -n 3 --temp 0 --seed 1 --no-warmup -no-cnv -t 4 -c 4096 -p 'The capital of France is' -fa "$fa" > "$R/ncu_${replay}_fa${fa}_two_metrics.raw" 2>&1
 done
 done
