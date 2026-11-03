#!/bin/bash
#SBATCH -J llama-fast
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -w node01
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/llamafast_%j.log
# Optimized llama.cpp for the full-scale study: native SIMD for node01's Cascade
# Lake (AVX-512 VNNI), OpenMP threads. Built on node01, so it runs only there.
# The SSE2 builds used with gem5 are left untouched.
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib
cd $W/llama.cpp
rm -rf build-fast
cmake -B build-fast -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ 2>&1 | tail -2
cmake --build build-fast -j32 --target llama-completion llama-bench llama-batched-bench 2>&1 | tail -2 | cut -c1-160
ls build-fast/bin
# Smoke test, and the SIMD features the build uses.
build-fast/bin/llama-completion -m $W/models/stories15M.gguf -t 8 -n 8 -p "Once upon a time" -no-cnv 2>&1 \
  | grep -E 'system_info|eval time' | cut -c1-200
bash $HOME/amcas-pro/tools/publish.sh "optimized llama.cpp built on node01 (job $SLURM_JOB_ID)"
