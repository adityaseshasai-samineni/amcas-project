#!/bin/bash
set -eu
P=$HOME/amcas-pro/tracing/accel-sim-audit-20261117/util/tracer_nvbit/nvbit_release/core
N=/usr/local/cuda-12.6/bin/nvcc
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
$N -arch=sm_70 -O3 -I"$P" -Xcompiler -fPIC -Xptxas -astoolspatch --keep-device-functions -c inject_funcs.cu -o inject_funcs.o
$N -arch=sm_70 -O3 -std=c++17 -I"$P" -Xcompiler -fPIC -Xptxas -cloning=no -c tool.cu -o tool.o
$N -arch=sm_70 -shared tool.o inject_funcs.o -L"$P" -lnvbit -L/usr/local/cuda-12.6/lib64/stubs -lcuda -lcudart_static -o latency_probe.so
/usr/local/cuda-12.6/bin/cuobjdump --dump-sass latency_probe.so > observer.sass
