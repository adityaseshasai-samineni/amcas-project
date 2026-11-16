#!/bin/bash
#SBATCH --job-name=accel-test
#SBATCH --nodelist=node04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:03:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
W=$HOME/amcas-pro/tracing/accel-sim-audit-20261117
export PATH=/usr/local/cuda-12.6/bin:$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$HOME/amcas-pro/tracing/mamba/envs/gem5/lib:$HOME/amcas-pro/tracing/accel-build-deps/lib
mkdir -p "$R/preflight_write"; cd "$R/preflight_write"
cat > tiny.cu <<'CU'
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void k(volatile int*p){int i=threadIdx.x;for(int j=0;j<4;j++)p[i+j*8192]=p[i+j*8192]+1;}
int main(){int*p;cudaMalloc(&p,131072);cudaMemset(p,0,131072);k<<<1,32>>>(p);int h[32];cudaMemcpy(h,p,128,cudaMemcpyDeviceToHost);for(int i=0;i<32;i++)if(h[i]!=1)return 2;puts("PASS native write test");cudaFree(p);return cudaDeviceSynchronize()!=cudaSuccess;}

CU
nvcc -arch=sm_70 tiny.cu -o tiny
NOBANNER=1 LD_PRELOAD="$W/util/tracer_nvbit/tracer_tool/tracer_tool.so" ./tiny > native.stdout 2>native.stderr
"$W/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing" traces -j 2 > post.log 2>&1
cp "$W/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_GV100/"* .
cp "$W/gpu-simulator/configs/tested-cfgs/SM7_GV100/trace.config" .
sed -i 's/-gpgpu_cache:dl2 S:32:128:24,/-gpgpu_cache:dl2 S:1:128:1,/' gpgpusim.config
# Deliberately tiny L2 for exporter write-eviction validation, NOT a V100 hardware experiment.
AMCAS_DRAM_EXPORT="$PWD/dram.csv" "$W/gpu-simulator/build-cmake/accel-sim.out" -config gpgpusim.config -config trace.config -trace traces/kernelslist.g > sim.log 2>&1
