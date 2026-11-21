#!/bin/bash
# Replay kept streams of one GPU through Ramulator 2 and DRAMsim3 (ib 2 = 256 B and ib 5 = 2 KiB channel interleave).
# usage: 65_replay_cells.sh GPU DIR "short:auto short:off ..." [procs]    (streams DIR/q7_GPU_CN_faFA_x13.dram.gz)
GPU=$1; D=$2; CELLS=$3; PR=${4:-16}
for c in $CELLS; do
  T=q7_${GPU}_${c%%:*}_fa${c##*:}
  bash $HOME/amcas-pro/gpu/64_study.sh $D/${T}_x13 $T $GPU "2 5" --procs $PR --tmp /tmp/$USER
done
echo "### replays done $(date)"
