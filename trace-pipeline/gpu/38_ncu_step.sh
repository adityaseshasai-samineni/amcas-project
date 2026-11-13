#!/bin/bash
# ncu counters for a launch window of llama.cpp. usage: 38_ncu_step.sh ARCH MODEL N SKIP COUNT OUTCSV
A=$1; M=$2; N=$3; SK=$4; CN=$5; OUT=$6
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu
export LD_LIBRARY_PATH=$W/cudalibs:$E/lib GGML_CUDA_DISABLE_GRAPHS=1
$NCU --csv --launch-skip $SK --launch-count $CN \
 --metrics ${METRICS:-dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum} \
 $W/llama.cpp/build-cuda/bin/llama-completion -m $M -ngl 99 -n $N --temp 0 --no-warmup -no-cnv -t 4 -p "The capital of France is" > $OUT.raw 2>&1
python3 - $OUT.raw <<'PY'
import csv,sys,collections
tot=collections.Counter(); n=0
rows=[l for l in open(sys.argv[1]) if l.startswith('"')]
for r in csv.reader(rows):
    if r[0]=="ID": hdr=r; continue
    d=dict(zip(hdr,r)); m=d['Metric Name']; v=float(d['Metric Value'].replace(',','')); u=d['Metric Unit']
    v*= {'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'sector':1}.get(u,1)
    tot[m]+=v
    if m=='dram__bytes_read.sum': n+=1
print("kernels",n)
for k,v in tot.items(): print("%-55s %.2f MB"%(k,v/1e6) if 'bytes' in k else "%-55s %d sectors = %.2f MB"%(k,v,v*32/1e6))
PY
