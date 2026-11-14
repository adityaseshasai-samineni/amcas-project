#!/bin/bash
# Measurement cells for one GPU. usage: 51_measure_all.sh GPU [cells...] (default: the long-context cells)
GPU=$1; shift
CELLS=("$@"); [ ${#CELLS[@]} = 0 ] && CELLS=("q7 8k auto" "q7 8k off" "q7 32k auto" "q7 32k off" "q05 8k auto")
for c in "${CELLS[@]}"; do
  set -- $c; bash $HOME/amcas-pro/gpu/50_dram_measure.sh $GPU $2 $3 $1 2>&1 | grep -v '^ \|^{\|^}' | cut -c1-300
done
