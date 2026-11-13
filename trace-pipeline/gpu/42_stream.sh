#!/bin/bash
# Streamed NVBit trace: tool -> FIFO -> gpu_filter -> gzip'd DRAM stream. No raw trace on disk.
# usage: 42_stream.sh OUTPREFIX ARCH MODEL N START END [filter args]     (env as 36_gputrace_run.sh: CTX PFILE EXTRA)
# Env TEE=1: also keep OUT.stored.bin (validation of the streamed path against the stored path).
O=$1; A=$2; M=$3; N=$4; S=$5; EN=$6; shift 6
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
mkdir -p $(dirname $O); rm -f $O.bin $O.dram $O.dramlaunch $O.f.bin
mkfifo $O.bin $O.dram $O.dramlaunch
gzip -1 < $O.dram > $O.dram.gz & P1=$!
gzip -1 < $O.dramlaunch > $O.dramlaunch.gz & P2=$!
(PATH=$E/bin:$PATH gcc -O2 -o $O.gf $HOME/amcas-pro/gpu/gpu_filter.c) || exit 1
if [ -n "$TEE" ]; then mkfifo $O.f.bin; tee $O.stored.bin < $O.bin > $O.f.bin & P3=$!; FP=$O.f; else FP=$O; fi
$O.gf $FP $O "$@" & P4=$!
bash $HOME/amcas-pro/gpu/36_gputrace_run.sh $O $A $M $N $S $EN | grep -e wall -e "eval time"
wait $P4 $P1 $P2 $P3 2>/dev/null
rm -f $O.bin $O.dram $O.dramlaunch $O.f.bin $O.gf
ls -l $O.dram.gz $O.dramlaunch.gz
