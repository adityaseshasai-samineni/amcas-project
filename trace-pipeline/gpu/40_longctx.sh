#!/bin/bash
# One decode step of llama.cpp (CUDA) at a given context: launch-only pass to find the step, NVBit trace of that
# step, L1/L2 filter, tensor attribution, HBM2/GDDR6 replay.
# usage: 40_longctx.sh TAG MODEL_GGUF CTX PROMPT_CHARS ARCH(70|86) GPU(v100|a16) [FA(on|off|auto)] [replay window]
# Env: SCR (scratch dir, default /scratch/$USER/gt), NOREPLAY=1.
TAG=$1; M=$2; CTX=$3; PC=$4; A=$5; GPU=$6; FA=${7:-auto}; WIN=${8:-0}
G=$HOME/amcas-pro/gpu; E=$HOME/amcas-pro/tracing/mamba/envs/gem5; SCR=${SCR:-/scratch/$USER/gt}; mkdir -p $SCR
R=$HOME/amcas-pro/results/gpu; mkdir -p $R
head -c $PC $HOME/amcas-pro/results/sessions/corpus.txt > $SCR/$TAG.prompt.txt
export CTX PFILE=$SCR/$TAG.prompt.txt EXTRA="-fa $FA"
bash $G/36_gputrace_run.sh $SCR/${TAG}_lo $A $M 3 999999999 999999999 | grep -e wall -e "eval time"
read S EN < <($E/bin/python $G/find_window.py $SCR/${TAG}_lo.launch.txt) || exit 1
echo "window $S $EN ($((EN-S)) launches)"; rm -f $SCR/${TAG}_lo.bin
bash $G/36_gputrace_run.sh $SCR/$TAG $A $M 3 $S $EN | grep -e wall -e "eval time" -e "^ "
ls -l $SCR/$TAG.bin
bash $G/37_filter.sh $SCR/$TAG $SCR/$TAG $( [ $GPU = a16 ] && echo "--sm 10 --l2mb 2" )
rm -f $SCR/$TAG.bin
$E/bin/python $G/gpu_attrib.py $SCR/$TAG $SCR/$TAG.tensors.tsv $R/${TAG}_attrib.json
[ -n "${NOREPLAY:-}" ] || bash $G/38_replay.sh $SCR/$TAG $R/${TAG}_replay.json $GPU --window $WIN --procs 4
