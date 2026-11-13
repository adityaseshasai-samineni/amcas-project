#!/bin/bash
# usage: 37_stats.sh PREFIX   (compiles gpu_stats.c with the gem5 env gcc on the node, then runs)
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export PATH=$E/bin:$PATH
gcc -O2 -o /tmp/gpu_stats_$$ $HOME/amcas-pro/gpu/gpu_stats.c && /tmp/gpu_stats_$$ $1 $2; rm -f /tmp/gpu_stats_$$
