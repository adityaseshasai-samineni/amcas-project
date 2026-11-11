#!/bin/bash
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=36G
#SBATCH --time=06:00:00
#SBATCH --exclude=node02,node04,node05,node07,node08
# Mapping studies on a kept miss stream, replayed with Ramulator RoBaRaCoCh (Open+ClosedCAP, DDR4x1, DDR5x16).
# Usage: 19_mapstudy.sh MODE OUTNAME ARGS...
#   MODE=moe:     ARGS = MISS_GZ TENSORS_TSV      -> moe_remap.py        -> results/fullscale/moe/OUTNAME
#   MODE=heldout: ARGS = A_MISS A_TSV B_MISS B_TSV -> heldout_remap.py   -> results/fullscale/heldout/OUTNAME
# Env: CAP_GB (modelled DRAM size for shuffled placements, default 32; batched MoE needs >8), WIN (window, default 8000000), PL (placements, default packed,thp)
set -uo pipefail
export AMCAS_CAPACITY_GB=${CAP_GB:-32}
MODE=$1; NAME=$2; shift 2
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
R=$HOME/amcas-pro/results/fullscale/$MODE/$NAME; mkdir -p $R
S=/scratch/$USER/mapstudy/$MODE-$NAME; rm -rf $S; mkdir -p $S; trap 'rm -rf $S' EXIT
TOOL=moe_remap.py; [ $MODE = heldout ] && TOOL=heldout_remap.py
$E/bin/python $HOME/amcas-pro/tools/$TOOL "$@" $S --window ${WIN:-8000000} --placements ${PL:-packed,thp} 2>&1 | tee $R/make.txt
cp $S/manifest.json $R/
replay() {
  f=$1; b=$(basename $f .trc)
  if [[ $b == *_ddr5_* ]]; then OPT=(--dram ddr5 --channels 16 --interleave-bits 2 --issue 4); else OPT=(--dram ddr4); fi
  env LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python $E/bin/python \
    $HOME/amcas-pro/tools/ramulator_replay.py $f $R/replay_$b.json "${OPT[@]}" --mappers RoBaRaCoCh --procs 2 > $R/replay_$b.txt 2>&1
  rm -f $f
}
export -f replay; export R W E
ls $S/*.trc | xargs -P 3 -I{} bash -c 'replay {}'
bash $HOME/amcas-pro/tools/publish.sh "mapstudy $MODE $NAME" || true
