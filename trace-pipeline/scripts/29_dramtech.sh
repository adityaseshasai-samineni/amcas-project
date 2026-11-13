#!/bin/bash
#SBATCH -J dramtech
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=node02,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/dramtech_%j.log
# Replay an attributed LLC-miss stream on LPDDR5/6, HBM3/4, GDDR6 (dramtech_study.py).
# Usage: 29_dramtech.sh ATTR_RUN TAG SYSTEMS [WINDOW] [O3WINDOW]
# Env: PL (placements), SC (schemes), POINTS, POL, BUB (instructions per read - 1: 52 for 7B 32k, 54 7B 8k, 44 30B), CAP_GB, PROCS
set -uo pipefail
RUN=$1; TAG=$2; SYS=$3; WIN=${4:-8000000}; O3W=${5:-2000000}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:$W/ramulator2-amcas OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$W/ramulator2-amcas/python AMCAS_CAPACITY_GB=${CAP_GB:-32}
A=$HOME/amcas-pro/results/fullscale/drattr/$RUN
R=$HOME/amcas-pro/results/fullscale/dramtech/$TAG; mkdir -p $R
S=/scratch/$USER/dramtech/$TAG; mkdir -p $S; df -h /scratch | tail -1
T=$A/tensors.tsv; [ -s $T ] || T=$(ls $A/../*/tensors.tsv | head -1)
$E/bin/python -u $HOME/amcas-pro/tools/dramtech_study.py $A/miss.gz $T $R --systems $SYS --window $WIN --o3-window $O3W \
  --placements ${PL:-packed,thp,4k} --schemes ${SC:-identity,xor_global,chan,xor_global_chan} --points ${POINTS:-sat,fast,nat4} \
  --policies ${POL:-Open,ClosedCAP} --bubbles ${BUB:-52} --procs ${PROCS:-6} --tmp $S 2>&1 | grep --line-buffered -v "Ramulator::\|Loading trace\|Loaded" | tee $R/log.txt
RC=${PIPESTATUS[0]}
rm -rf $S
[ -n "${NOPUB:-}" ] || bash $HOME/amcas-pro/tools/publish.sh "dramtech results $TAG (rc=$RC)" || true
exit $RC
