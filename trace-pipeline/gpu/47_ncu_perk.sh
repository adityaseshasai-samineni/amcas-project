#!/bin/bash
# Per-kernel ncu L2/DRAM counters for a launch window. usage: 47_ncu_perk.sh ARCH MODEL N SKIP COUNT OUTCSV (env CTX PFILE EXTRA NCU as 38_ncu_step.sh)
export METRICS=${METRICS:-dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum}
bash $HOME/amcas-pro/gpu/38_ncu_step.sh "$@" | tail -n 12
