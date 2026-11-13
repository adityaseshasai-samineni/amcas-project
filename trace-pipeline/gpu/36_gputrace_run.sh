#!/bin/bash
# NVBit trace of llama.cpp (CUDA build). usage: 36_gputrace_run.sh OUTPREFIX ARCH MODEL N_PREDICT START END [prompt]
# START/END = kernel launch id window (launch ids count every kernel launch incl. model load).
OUT=$1; A=$2; M=$3; N=$4; S=$5; EN=$6; P=${7:-"The capital of France is"}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export PATH=$W/cudatools:$PATH LD_LIBRARY_PATH=$W/cudalibs:$E/lib
export GGML_CUDA_DISABLE_GRAPHS=1 CUDA_MODULE_LOADING=EAGER NOBANNER=1
export GPUTRACE_OUT=$OUT GPUTRACE_START=$S GPUTRACE_END=$EN
if [ -n "$PFILE" ]; then PA=(-f $PFILE); else PA=(-p "$P"); fi
mkdir -p $(dirname $OUT); hostname
T=$(date +%s.%N)
AMCAS_TENSORS=$OUT.tensors.tsv LD_PRELOAD=$W/nvbit/v1.7.7.3/nvbit_release_x86_64/tools/gputrace2-sm$A/gputrace2-sm$A.so:$W/lib/libgpuattr.so \
 $W/llama.cpp/build-cuda/bin/llama-completion -m $M -ngl 99 -n $N --temp 0 --seed 1 --no-warmup -no-cnv -t 4 ${CTX:+-c $CTX} "${PA[@]}" $EXTRA 2>&1 | grep -e "^The" -e "eval time" | cut -c1-200
echo wall $(echo "$(date +%s.%N) - $T" | bc)
ls -l $OUT*; wc -l $OUT.launch.txt
