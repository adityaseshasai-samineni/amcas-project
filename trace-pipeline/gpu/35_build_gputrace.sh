#!/bin/bash
# Build the amcas GPU trace tool for ARCH (70 or 86) from NVBit 1.7.7.3 mem_trace. usage: 35_build_gputrace.sh [ARCH...]
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export PATH=/usr/local/cuda-12.6/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$E/lib
S=$W/nvbit/v1.7.7.3/nvbit_release_x86_64/tools/mem_trace
for A in ${@:-70 86}; do
  D=$W/nvbit/v1.7.7.3/nvbit_release_x86_64/tools/gputrace2-sm$A; rm -rf $D
  python3 $HOME/amcas-pro/gpu/gputrace_patch.py $S $D || exit 1
  (cd $D && rm -f *.so *.o && make ARCH=sm_$A -j8 2>&1 | tail -n 3; ls -l *.so)
done
