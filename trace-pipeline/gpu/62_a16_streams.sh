#!/bin/bash
# A16 streams (6 cells) kept in node05:/tmp/$USER/gt (node05 /scratch is full, home quota is nearly full).
G=$HOME/amcas-pro/gpu
for c in short:auto short:off 8k:auto 8k:off 32k:auto 32k:off; do
  echo "######## a16 ${c%%:*} fa=${c##*:} $(date)"
  bash $G/60_capture.sh a16 ${c%%:*} ${c##*:} 2>&1 | grep -v "^ *search"
done
echo "######## done $(date)"; ls -l /tmp/$USER/gt/*.dram.gz
