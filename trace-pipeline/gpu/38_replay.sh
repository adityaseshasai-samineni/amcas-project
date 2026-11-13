#!/bin/bash
# usage: 38_replay.sh DRAM_PREFIX OUT_JSON GPU [gpu_replay args]
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:$W/ramulator2-amcas OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$W/ramulator2-amcas/python
P=$1; O=$2; G=$3; shift 3
$E/bin/python $HOME/amcas-pro/gpu/gpu_replay.py $P $O --gpu $G --tmp /tmp/gpurep_$$ "$@" 2>&1 | grep -v "Ramulator::\|Loading trace\|Loaded"
rm -rf /tmp/gpurep_$$
