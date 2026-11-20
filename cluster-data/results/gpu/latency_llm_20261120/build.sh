#!/bin/bash
set -eu
P=$HOME/amcas-pro/tracing/accel-sim-audit-20261117/util/tracer_nvbit/nvbit_release/core
N=/usr/local/cuda-12.6/bin/nvcc
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
$N -arch=sm_70 -O3 -I"$P" -Xcompiler -fPIC -Xptxas -astoolspatch --keep-device-functions -c inject_funcs.cu -o inject_funcs.o
$N -arch=sm_70 -O3 -std=c++17 -I"$P" -Xcompiler -fPIC -Xptxas -cloning=no -c tool.cu -o tool.o
$N -arch=sm_70 -shared tool.o inject_funcs.o -L"$P" -lnvbit -L/usr/local/cuda-12.6/lib64/stubs -lcuda -lcudart_static -o latency_probe.so
/usr/local/cuda-12.6/bin/cuobjdump --dump-sass latency_probe.so > observer.sass
# Paired BEFORE/BEFORE control uses the identical callbacks but brackets no original LDG.
# The destination register's old value is passed; original load still runs afterward.
cp ../latency_validation_20261117/latency.cu fixture.cu
$N -O3 -arch=sm_70 fixture.cu -o fixture
timeout 60 env PROBE_CONTROL=1 PROBE_KERNEL=measure_kernel PROBE_OUT=observer_control.csv LD_PRELOAD="$PWD/latency_probe.so" ./fixture l1 1024 8 > control_native.csv 2> control.log
