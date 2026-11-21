#!/bin/bash
# Capture and KEEP the DRAM stream of one 7B (or other) decode step with the fixed L2 filter (gpu_filter2, XOR-hashed set index x13).
# usage: 60_capture.sh GPU(v100|a16) CTXNAME(short|8k|32k) FA(auto|off)       env: MODEL, MODELTAG (default q7), SCR (stream dir)
# Output (in $SCR): TAG_x13.dram.gz, TAG_x13.dramlaunch.gz, TAG_x13.perk.tsv, TAG_x13.tensors.tsv  (TAG=q7_GPU_CN_faFA)
GPU=$1; CN=$2; FA=$3
W=$HOME/amcas-pro/tracing; G=$HOME/amcas-pro/gpu; E=$W/mamba/envs/gem5; R=$HOME/amcas-pro/results/gpu; mkdir -p $R
MT=${MODELTAG:-q7}; M=${MODEL:-$W/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}
if [ $GPU = v100 ]; then A=70; SCR=${SCR:-/scratch/$USER/gt}; HV="x13:hash=113"
else A=86; SCR=${SCR:-/tmp/$USER/gt}; HV="x13:hash=113,sm=10,l2mb=2"; fi
mkdir -p $SCR; df -BG --output=avail $SCR | tail -1
case $CN in short) CTX=4096; PC=0;; 8k) CTX=8192; PC=30000;; 32k) CTX=32768; PC=116000;; esac
TAG=${MT}_${GPU}_${CN}_fa${FA}_x13
if [ $PC = 0 ]; then unset PFILE; else head -c $PC $HOME/amcas-pro/results/sessions/corpus.txt > $SCR/$TAG.prompt.txt; export PFILE=$SCR/$TAG.prompt.txt; fi
export CTX EXTRA="-fa $FA" PATH=$W/cudatools:$PATH
bash $G/36_gputrace_run.sh $SCR/${TAG}_lo $A $M 3 999999999 999999999 | grep -e wall
read S EN < <($E/bin/python $G/find_window.py $SCR/${TAG}_lo.launch.txt) || exit 1
echo "$TAG window $S $EN"; rm -f $SCR/${TAG}_lo.bin
EMIT=x13 bash $G/46_stream2.sh $SCR/$TAG $A $M 3 $S $EN "--var $HV"
rm -f $SCR/$TAG.prompt.txt $SCR/${TAG}_lo*
ls -l $SCR/$TAG.dram.gz
