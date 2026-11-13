#!/bin/bash
# Diagnostics + mapping sweeps on a kept stream. usage: 41_diag.sh TAG GPU WINDOW SWEEP(full|ib2) [RETRACE "CTX PROMPT_CHARS FA S E MODEL"]
# RETRACE re-traces the window with SM-tagged output (gpu_filter writes sm<<20 into .dramlaunch).
TAG=$1; GPU=$2; WIN=$3; SW=$4; RT=$5
G=$HOME/amcas-pro/gpu; E=$HOME/amcas-pro/tracing/mamba/envs/gem5; SCR=/scratch/$USER/gt; R=$HOME/amcas-pro/results/gpu
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=$HOME/amcas-pro/tracing/ramulator2-amcas/python
if [ -n "$RT" ]; then
  set -- $RT; export CTX=$1 PFILE=$SCR/$TAG.prompt.txt EXTRA="-fa $3"
  bash $G/36_gputrace_run.sh $SCR/${TAG}_sm 70 $6 3 $4 $5 | grep -e wall
  bash $G/37_filter.sh $SCR/${TAG}_sm $SCR/${TAG}_sm; rm -f $SCR/${TAG}_sm.bin
  D=${TAG}_sm
else D=$TAG; fi
export LD_LIBRARY_PATH=$E/lib:$HOME/amcas-pro/tracing/ramulator2-amcas
$E/bin/python $G/gpu_diag.py $SCR/$D $SCR/$D.tensors.tsv $R/${D}_diag.json --window ${DIAGWIN:-60000000} 2>&1 | grep -v "^ *search\|bank bit" | head -150
run() { # ib schemes
  bash $G/38_replay.sh $SCR/$D $R/${D}_sweep_ib$1.json $GPU --window $WIN --ib $1 --schemes $2 --procs 4 | grep -v "^  search\|bank bit"
}
run 2 identity,xor,chan,xor_chan
[ $SW = full ] && { run 5 identity,xor,chan; run 6 identity,xor,chan; }
