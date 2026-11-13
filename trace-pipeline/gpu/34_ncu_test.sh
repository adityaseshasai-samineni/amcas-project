#!/bin/bash
# Can a normal user read DRAM counters with ncu?
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export PATH=/usr/local/cuda-12.6/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$E/lib
hostname; ncu --version | tail -n 1
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum $W/nvbit/vecadd/vectoradd 2>&1 | tail -n 14
