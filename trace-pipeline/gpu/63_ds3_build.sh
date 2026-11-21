#!/bin/bash
# Build tools/gpu_ds3_drain.cc against libdramsim3.so (gem5 env g++; the system g++ cannot link it).
E=$HOME/amcas-pro/tracing/mamba/envs/gem5; D=$HOME/amcas-pro/tracing/DRAMsim3
PATH=$E/bin:$PATH g++ -O2 -std=c++14 -I$D/src -I$D/ext/headers -I$D/ext/fmt/include $HOME/amcas-pro/tools/gpu_ds3_drain.cc -L$D -ldramsim3 -Wl,-rpath,$D -Wl,-rpath,$E/lib -o $HOME/amcas-pro/tracing/bin/gpu_ds3_drain.new && mv -f $HOME/amcas-pro/tracing/bin/gpu_ds3_drain.new $HOME/amcas-pro/tracing/bin/gpu_ds3_drain && echo built
