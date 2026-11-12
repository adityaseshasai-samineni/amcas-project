#!/bin/bash
#SBATCH -J frag
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --exclude=node02,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/frag_%j.log
# transfer-owner task 5: fragmentation sensitivity (tools/frag_remap.py), DDR5 x16 Open+ClosedCAP, for 7B 32k and 30B 32k.
# Usage: 28_frag.sh [STREAMS="7B32k 30B32k"]; env WIN (8000000), GRID (frag_remap --grid), SMOKE=1 (no replay)
set -uo pipefail
STREAMS=${1:-"7B32k 30B32k"}
df -h /scratch | tail -1
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
D=$HOME/amcas-pro/results/fullscale/drattr
declare -A SRC=( [7B32k]=$D/qwen2.5-7b-instruct-q4_k_m-00001-of-00002_s1n1t4_attr_c32k [30B32k]=$D/Qwen3-30B-A3B-Q4_K_M_s1n1t4_attr_c32k [05B32k]=$D/qwen25-05b-q4_0_s1n1t4_attr_c32k )
replay() {
  f=$1; b=$(basename $f .trc)
  env LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python $E/bin/python \
    $HOME/amcas-pro/tools/ramulator_replay.py $f $R/replay_$b.json --dram ddr5 --channels 16 --interleave-bits 2 --issue 4 --mappers RoBaRaCoCh --procs 2 > $R/replay_$b.txt 2>&1
  rm -f $f
}
export -f replay; export W E
for s in $STREAMS; do
  R=$HOME/amcas-pro/results/fullscale/frag/$s; mkdir -p $R; export R
  S=/scratch/$USER/frag/$s; rm -rf $S; mkdir -p $S
  A=${SRC[$s]}
  GR=(); [ -n "${GRID:-}" ] && GR=(--grid $GRID)
  $E/bin/python $HOME/amcas-pro/tools/frag_remap.py $A/miss.gz $A/tensors.tsv $S --window ${WIN:-8000000} --cap-gb 32 "${GR[@]}" 2>&1 | tee $R/make.txt
  cp $S/manifest.json $R/
  if [ "${SMOKE:-0}" != 1 ]; then ls $S/*.trc | xargs -P 3 -I{} bash -c 'replay {}'; fi
  rm -rf $S
done
[ "${SMOKE:-0}" = 1 ] || bash $HOME/amcas-pro/tools/publish.sh "frag results" || true
