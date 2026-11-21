#!/bin/bash
# V100 streams: short/off streamed capture; short/auto, 8k/auto, 8k/off re-filtered offline from the kept raw NVBit traces (same
# NVBit capture, new L2 filter x13); 32k both: gzip the x13 streams of job 124562. Output q7_v100_*_x13.dram.gz in /scratch/$USER/gt.
SCR=/scratch/$USER/gt; G=$HOME/amcas-pro/gpu
bash $G/60_capture.sh v100 short off 2>&1 | grep -v "^ *search"
for t in short_faauto 8k_faauto 8k_faoff; do
  T=q7_v100_$t; echo "### offline $T $(date)"
  bash $G/49_offline.sh $T ${T}_x13 "--var x13:hash=113 --emit x13" 2>&1 | tail -n 4
  cp $SCR/$T.tensors.tsv $SCR/${T}_x13.tensors.tsv 2>/dev/null
  gzip -1 -f $SCR/${T}_x13.dram; gzip -1 -f $SCR/${T}_x13.dramlaunch; ls -l $SCR/${T}_x13.dram.gz
done
for t in 32k_faauto 32k_faoff; do
  T=q7_v100_${t}_x13; echo "### gzip existing $T $(date)"
  [ -f $SCR/$T.dram ] && gzip -1 -f $SCR/$T.dram && gzip -1 -f $SCR/$T.dramlaunch
  ls -l $SCR/$T.dram.gz
done
echo "### done $(date)"
