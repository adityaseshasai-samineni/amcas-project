#!/bin/bash
# Re-derive the DRAM stream of a kept raw trace with the calibrated L2 hash, then attribution + diag + HBM2 replays (as 43_matrix.sh).
# usage: 50_rerun.sh RAWTAG(e.g. q7_v100_32k_faauto) VARIANT_SPEC("x13:hash=113") GPU(v100)
T=$1; VS=$2; GPU=${3:-v100}; VN=${VS%%:*}
W=$HOME/amcas-pro/tracing; G=$HOME/amcas-pro/gpu; E=$W/mamba/envs/gem5; R=$HOME/amcas-pro/results/gpu; SCR=/scratch/$USER/gt
D=${T}_$VN
bash $G/49_offline.sh $T $D "--var $VS --emit $VN" 2>&1 | tail -n 3
cp $SCR/$T.tensors.tsv $SCR/$D.tensors.tsv
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=$W/ramulator2-amcas/python LD_LIBRARY_PATH=$E/lib:$W/ramulator2-amcas
$E/bin/python $G/gpu_attrib.py $SCR/$D $SCR/$D.tensors.tsv $R/${D}_attrib.json | tail -n 12
$E/bin/python $G/gpu_diag.py $SCR/$D $SCR/$D.tensors.tsv $R/${D}_diag.json --window 60000000 --gpu $GPU 2>&1 | grep -v "^ *search\|bank bit" | head -40
run() { bash $G/38_replay.sh $SCR/$D $R/${D}_sweep_ib$1.json $GPU --window 80000000 --ib $1 --schemes $2 --procs 4 | grep -v "^  search\|bank bit"; }
run 2 identity,xor,chan,xor_chan
run 5 identity,xor,chan
run 6 identity,xor,chan
