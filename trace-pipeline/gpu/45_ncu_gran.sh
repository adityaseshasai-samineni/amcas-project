#!/bin/bash
# DRAM fetch granularity check: ncu counters on the 0.5B short-context decode step (launch window 1102..1564, graphs off).
# usage: 45_ncu_gran.sh GPU(a16|v100)
GPU=$1; W=$HOME/amcas-pro/tracing; G=$HOME/amcas-pro/gpu; M=$W/models/qwen25-05b-q4_0.gguf
if [ $GPU = v100 ]; then export NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu; else export NCU=/usr/local/cuda-12.6/bin/ncu; fi
WANT="dram__bytes_read.sum dram__bytes_write.sum dram__sectors_read.sum lts__t_sectors_srcunit_tex_op_read.sum lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum lts__t_requests_srcunit_tex_op_read.sum lts__t_sectors_op_read.sum lts__t_sectors_op_read_lookup_miss.sum l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum fbpa__dram_read_sectors.sum"
export LD_LIBRARY_PATH=$W/cudalibs:$W/mamba/envs/gem5/lib
OK=""; for m in $WANT lts__t_sectors_op_atom.sum lts__t_sectors_op_red.sum lts__t_sectors.sum lts__t_sectors_op_write_lookup_miss.sum lts__t_sectors_srcunit_ltcfabric.sum dram__bytes.sum; do
  if $NCU --metrics $m $W/nvbit/vecadd/vectoradd 2>&1 | grep -q "^ *$m "; then OK="$OK,$m"; else echo "metric not available: $m"; fi; done
export METRICS=${OK#,}; echo "using $METRICS"
unset CTX PFILE EXTRA
bash $G/38_ncu_step.sh x $M 3 1102 462 $HOME/amcas-pro/results/gpu/q05_${GPU}_gran 2>&1 | tail -n 14
