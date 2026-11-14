#!/bin/bash
# Streamed NVBit trace -> gpu_filter2 (many L2/L1 variants in one pass) -> per-launch tsv (+ optional gz DRAM stream of one variant).
# usage: 46_stream2.sh OUTPREFIX ARCH MODEL N START END "filter2 args..."  (env as 36_gputrace_run.sh: CTX PFILE EXTRA; EMIT=name to keep DRAM stream)
O=$1; A=$2; M=$3; N=$4; S=$5; EN=$6; FA=$7
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
mkdir -p $(dirname $O); rm -f $O.bin $O.dram $O.dramlaunch
mkfifo $O.bin
if [ -n "$EMIT" ]; then mkfifo $O.dram $O.dramlaunch; gzip -1 < $O.dram > $O.dram.gz & P1=$!; gzip -1 < $O.dramlaunch > $O.dramlaunch.gz & P2=$!; fi
(PATH=$E/bin:$PATH gcc -O2 -o $O.gf2 $HOME/amcas-pro/gpu/gpu_filter2.c) || exit 1
if [ -n "$KEEPRAW" ]; then mkfifo $O.f.bin; tee $O.raw.bin < $O.bin > $O.f.bin & P3=$!; mv_ok=1; eval "$O.gf2 $O.f $O $FA ${EMIT:+--emit $EMIT}" & P4=$!; else eval "$O.gf2 $O $O $FA ${EMIT:+--emit $EMIT}" & P4=$!; fi
# gpu_filter2 takes PREFIX not file: it opens PREFIX.bin
bash $HOME/amcas-pro/gpu/36_gputrace_run.sh $O $A $M $N $S $EN | grep -e wall -e "eval time"
wait $P4 $P1 $P2 $P3 2>/dev/null
rm -f $O.bin $O.f.bin $O.dram $O.dramlaunch $O.gf2
ls -l $O.perk.tsv
