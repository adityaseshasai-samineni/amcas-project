#!/bin/bash
# One cell of the GPU experiment matrix: 7B (or other) decode step at a context, FA on/off, on V100 or A16.
# usage: 43_matrix.sh GPU(v100|a16) CTXNAME(short|8k|32k) FA(auto|off) [MODEL_GGUF]
# Streams NVBit -> filter -> gzip (no raw trace). Then: tensor attribution, ncu DRAM check, per-class diag, replay sweeps.
# Env: KEEP=1 keep the .dram.gz; WIN (replay window, default 80000000); MODELTAG (default q7).
GPU=$1; CN=$2; FA=$3
W=$HOME/amcas-pro/tracing; G=$HOME/amcas-pro/gpu; E=$W/mamba/envs/gem5; R=$HOME/amcas-pro/results/gpu; mkdir -p $R
M=${4:-$W/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}; MT=${MODELTAG:-q7}
if [ $GPU = v100 ]; then A=70; SCR=/scratch/$USER/gt; FARGS=""; NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu; ARGS_REPLAY=""
else A=86; SCR=$HOME/gt; FARGS="--sm 10 --l2mb 2"; NCU=/usr/local/cuda-12.6/bin/ncu; fi
mkdir -p $SCR; df -BG --output=avail $SCR | tail -1
case $CN in short) CTX=4096; PC=0;; 8k) CTX=8192; PC=30000;; 32k) CTX=32768; PC=116000;; esac
TAG=${MT}_${GPU}_${CN}_fa${FA}
if [ $PC = 0 ]; then unset PFILE; else head -c $PC $HOME/amcas-pro/results/sessions/corpus.txt > $SCR/$TAG.prompt.txt; export PFILE=$SCR/$TAG.prompt.txt; fi
export CTX EXTRA="-fa $FA" PATH=$W/cudatools:$PATH
bash $G/36_gputrace_run.sh $SCR/${TAG}_lo $A $M 3 999999999 999999999 | grep -e wall
read S EN < <($E/bin/python $G/find_window.py $SCR/${TAG}_lo.launch.txt) || exit 1
echo "$TAG window $S $EN"; rm -f $SCR/${TAG}_lo.bin
bash $G/42_stream.sh $SCR/$TAG $A $M 3 $S $EN $FARGS
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=$W/ramulator2-amcas/python LD_LIBRARY_PATH=$E/lib:$W/ramulator2-amcas
$E/bin/python $G/gpu_attrib.py $SCR/$TAG $SCR/$TAG.tensors.tsv $R/${TAG}_attrib.json | tail -n 12
# ncu DRAM counters of the same window (caches persist between kernels, like the real run)
export NCU METRICS=dram__bytes_read.sum,dram__bytes_write.sum
bash $G/38_ncu_step.sh $A $M 3 $S $((EN-S)) $R/${TAG}_ncu 2>&1 | tail -n 4
$E/bin/python $G/gpu_diag.py $SCR/$TAG $SCR/$TAG.tensors.tsv $R/${TAG}_diag.json --window 60000000 --gpu $GPU 2>&1 | grep -v "^ *search\|bank bit" | head -60
run() { bash $G/38_replay.sh $SCR/$TAG $R/${TAG}_sweep_ib$1.json $GPU --window ${WIN:-80000000} --ib $1 --schemes $2 --procs 4 | grep -v "^  search\|bank bit"; }
run 2 identity,xor,chan,xor_chan
[ $CN = 32k ] && { run 5 identity,xor,chan; run 6 identity,xor,chan; }
[ -n "$KEEP" ] || rm -f $SCR/$TAG.dram.gz $SCR/$TAG.dramlaunch.gz $SCR/$TAG.prompt.txt
