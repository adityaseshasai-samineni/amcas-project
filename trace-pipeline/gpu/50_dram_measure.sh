#!/bin/bash
# Measured DRAM behaviour of one llama.cpp CUDA decode step (ncu per kernel) plus an un-profiled token time.
# usage: 50_dram_measure.sh GPU(v100|a16) CTXNAME(short|8k|32k) FA(auto|off) [MODELTAG q7|q05]
GPU=$1; CN=$2; FA=$3; MT=${4:-q7}
W=$HOME/amcas-pro/tracing; G=$HOME/amcas-pro/gpu; E=$W/mamba/envs/gem5; R=$HOME/amcas-pro/results/measure; mkdir -p $R/win
case $MT in q7) M=$W/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf;; q05) M=$W/models/qwen25-05b-q4_0.gguf;; esac
if [ $GPU = v100 ]; then A=70; SCR=/scratch/$USER/gt; NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu
else A=86; SCR=$HOME/gt; NCU=/usr/local/cuda-12.6/bin/ncu; fi
mkdir -p $SCR
case $CN in short) CTX=4096; PC=0;; 8k) CTX=8192; PC=30000;; 32k) CTX=32768; PC=116000;; esac
TAG=${MT}_${GPU}_${CN}_fa${FA}
if [ $PC = 0 ]; then unset PFILE; else PF=$SCR/${TAG}_m.prompt.txt; head -c $PC $HOME/amcas-pro/results/sessions/corpus.txt > $PF; export PFILE=$PF; fi
export CTX EXTRA="-fa $FA" PATH=$W/cudatools:$PATH LD_LIBRARY_PATH=$W/cudalibs:$E/lib
if [ -n "$PFILE" ]; then PA=(-f $PFILE); else PA=(-p "The capital of France is"); fi
# 1. launch window of the last decode step (cached)
if [ ! -s $R/win/$TAG.window ]; then
  LF=$HOME/amcas-pro/results/gpu/${TAG}_launch.txt
  if [ ! -s $LF ]; then
    bash $G/36_gputrace_run.sh $SCR/${TAG}_m_lo $A $M 3 999999999 999999999 | grep -e wall; LF=$SCR/${TAG}_m_lo.launch.txt
  fi
  $E/bin/python $G/find_window.py $LF > $R/win/$TAG.window || exit 1
  rm -f $SCR/${TAG}_m_lo.bin
fi
read S EN < $R/win/$TAG.window; echo "$TAG window $S $EN"
# 2. un-profiled token time (CUDA graphs on, as in normal use)
unset GGML_CUDA_DISABLE_GRAPHS
for i in 1 2; do
$W/llama.cpp/build-cuda/bin/llama-completion -m $M -ngl 99 -n 64 --temp 0 --seed 1 --no-warmup -no-cnv -t 4 -c $CTX "${PA[@]}" -fa $FA 2>&1 | grep -E "eval time" | sed "s/^/$TAG run$i /"
done > $R/$TAG.timing.txt; cat $R/$TAG.timing.txt
# 3. ncu per kernel, clocks not locked
export GGML_CUDA_DISABLE_GRAPHS=1
MET=dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
$NCU --csv --cache-control none --clock-control none --launch-skip $S --launch-count $((EN-S)) --metrics $MET \
 $W/llama.cpp/build-cuda/bin/llama-completion -m $M -ngl 99 -n 3 --temp 0 --seed 1 --no-warmup -no-cnv -t 4 -c $CTX "${PA[@]}" -fa $FA > $R/$TAG.ncu.raw 2>&1
grep -c '^"' $R/$TAG.ncu.raw
$E/bin/python $G/measure_parse.py $R/$TAG.ncu.raw $R/$TAG.timing.txt $TAG > $R/$TAG.summary.json && cat $R/$TAG.summary.json | head -c 3000
rm -f $PF
