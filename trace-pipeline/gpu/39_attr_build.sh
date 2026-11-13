#!/bin/bash
# build libgpuattr.so (tensor dump LD_PRELOAD) against the CUDA llama.cpp headers
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib
L=$W/llama.cpp
g++ -O2 -shared -fPIC -std=c++17 -I$L/ggml/include -I$L/include -o $W/lib/libgpuattr.so $HOME/amcas-pro/gpu/gpu_attr.cpp -L$L/build-cuda/bin -lggml-base -ldl && ls -l $W/lib/libgpuattr.so
