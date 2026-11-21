#!/bin/bash
# usage: 66_rerun.sh GPU DIR "cell:fa:ib ..."   e.g. "8k:off:5 32k:off:2"
GPU=$1; D=$2; PR=${4:-12}
for c in $3; do IFS=: read cn fa ib <<< "$c"; T=q7_${GPU}_${cn}_fa${fa}
  bash $HOME/amcas-pro/gpu/64_study.sh $D/${T}_x13 $T $GPU "$ib" --procs $PR --tmp /tmp/$USER
done
echo "### reruns done $(date)"
