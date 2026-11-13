#!/bin/bash
# llama.cpp CUDA build (build-cuda), sm_70 + sm_86, CUDA 12.6. Run on node04 (has nvcc 12.6).
# Then smoke-test on 0.5B with all layers on GPU.
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export PATH=/usr/local/cuda-12.6/bin:$E/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$E/lib
cd $W/llama.cpp
if [ ! -x build-cuda/bin/llama-completion ]; then
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;86" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DGGML_NATIVE=OFF -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ 2>&1 | tail -5
cmake --build build-cuda -j16 --target llama-completion llama-bench 2>&1 | tail -3 | cut -c1-160
fi
ls build-cuda/bin | head -20
