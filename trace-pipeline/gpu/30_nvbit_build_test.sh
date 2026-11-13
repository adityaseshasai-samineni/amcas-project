#!/bin/bash
# Build NVBit tools (sm_70;86) and run mem_trace/instr_count on the vectoradd test app.
# usage: sbatch -w node04|node05 --gres=gpu:1 ... 30_nvbit_build_test.sh
export PATH=/usr/local/cuda-12.6/bin:$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
NB=$HOME/amcas-pro/tracing/nvbit/nvbit_release_x86_64
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -2
B=$HOME/amcas-pro/tracing/nvbit/build-sm70_86
if command -v nvcc >/dev/null && [ ! -f $B/mem_trace/mem_trace.so ]; then
  mkdir -p $B
  cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -S $NB/tools -B $B -DNVBIT_TOOL_CUDA_ARCHITECTURES="70;86" >/dev/null && cmake --build $B -j8 2>&1 | tail -5
fi
T=$HOME/amcas-pro/tracing/nvbit/vecadd; mkdir -p $T
[ -x $T/vectoradd ] || ! command -v nvcc >/dev/null || nvcc -O2 -arch=sm_70 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_70,code=sm_70 $NB/test-apps/vectoradd/vectoradd.cu -o $T/vectoradd
find $B -name '*.so' | head
echo "--- native"; $T/vectoradd
for t in instr_count mem_trace; do
  echo "--- $t"; so=$(find $B -name "$t.so" | head -1)
  LD_PRELOAD=$so $T/vectoradd 2>&1 | tail -15
done
