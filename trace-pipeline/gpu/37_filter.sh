#!/bin/bash
# usage: 37_filter.sh PREFIX OUT_PREFIX [gpu_filter args]
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export PATH=$E/bin:$PATH
P=$1; O=$2; shift 2
gcc -O2 -o /tmp/gpu_filter_$$ $HOME/amcas-pro/gpu/gpu_filter.c && /tmp/gpu_filter_$$ $P $O "$@"; rm -f /tmp/gpu_filter_$$
