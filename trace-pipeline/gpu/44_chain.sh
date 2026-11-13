#!/bin/bash
# Run the matrix cells of one GPU sequentially. usage: 44_chain.sh GPU [cells "short:auto 8k:auto ..."]
GPU=$1; CELLS=${2:-"short:auto short:off 8k:auto 8k:off 32k:auto 32k:off"}
for c in $CELLS; do
  echo "######## $GPU ${c%%:*} fa=${c##*:} $(date)"
  bash $HOME/amcas-pro/gpu/43_matrix.sh $GPU ${c%%:*} ${c##*:} 2>&1 | grep -v "Ramulator::\|Loading trace\|Loaded"
done
echo "######## chain done $(date)"
