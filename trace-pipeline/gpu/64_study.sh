#!/bin/bash
# usage: 64_study.sh PREFIX TAG GPU [IBs "2 5"] [extra gpu_ds3_study args]  -> results/gpu/ds3/TAG_ibN.json
P=$1; T=$2; GPU=$3; IBS=${4:-"2 5"}; shift 4
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; O=$HOME/amcas-pro/results/gpu/ds3; mkdir -p $O
export LD_LIBRARY_PATH=$E/lib:$W/ramulator2-amcas OPENBLAS_NUM_THREADS=1 PYTHONPATH=$W/ramulator2-amcas/python AMCAS_TOOLS=$HOME/amcas-pro/tools
for ib in $IBS; do
  echo "### $T ib$ib $(date)"
  $E/bin/python $HOME/amcas-pro/gpu/gpu_ds3_study.py $P $O/${T}_ib$ib.json --gpu $GPU --ib $ib --tmp ${TMPD:-/tmp} "$@" 2>&1 | grep -v "Ramulator::\|Loading trace\|Loaded\|^  search\|bank bit"
done
