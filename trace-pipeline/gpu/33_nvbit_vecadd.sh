#!/bin/bash
# Run NVBit instr_count / mem_trace on vectoradd (tools already built by 30_*).
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export PATH=/usr/local/cuda-12.6/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$E/lib
B=$W/nvbit/build-sm70_86; T=$W/nvbit/vecadd
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
which nvdisasm || ls /usr/local/cuda-12.6/bin/nvdisasm
echo "--- native"; $T/vectoradd
for t in instr_count mem_trace; do
  echo "--- $t"; LD_PRELOAD=$B/$t/$t.so $T/vectoradd 2>&1 | tail -n 15
done
