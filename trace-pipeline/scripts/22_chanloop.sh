#!/bin/bash
#SBATCH -J chanloop
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/chanloop_%j.log
# Closed-loop (SimpleO3) + saturated replay of identity / xor_global / chan / xor_global_chan for DDR5 x16 and x8.
# Usage: 22_chanloop.sh ATTR_RUN BUBBLES TAG [WINDOW] [PLACEMENTS] [POLICIES]
set -uo pipefail
RUN=$1; BUB=$2; TAG=$3; WIN=${4:-4000000}; PL=${5:-packed,thp,4k}; POL=${6:-Open,ClosedCAP}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:$W/ramulator2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$W/ramulator2/python AMCAS_CAPACITY_GB=${CAP_GB:-32}
A=$HOME/amcas-pro/results/fullscale/drattr/$RUN
R=$HOME/amcas-pro/results/fullscale/chanloop/$TAG; mkdir -p $R
S=/scratch/$USER/chanloop/$TAG; rm -rf $S; mkdir -p $S
trap 'rm -rf $S' EXIT
T=$A/tensors.tsv; [ -s $T ] || T=$(ls $A/../*/tensors.tsv | head -1)
$E/bin/python $HOME/amcas-pro/tools/policy_traces.py $A/miss.gz $T $S --window $WIN --bubbles $BUB \
  --placements $PL --drams ddr5,ddr5x8 --schemes identity,xor_global,chan,xor_global_chan 2>&1 | tee $R/make.txt
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "policy_traces failed"; exit 1; }
cp $S/manifest.json $R/
$E/bin/python $HOME/amcas-pro/tools/policy_replay.py $S $R/replay.json --bubbles $BUB --procs 6 --policies $POL 2>&1 \
  | grep -v "Ramulator::\|Loading trace\|Loaded" > $R/replay.txt
RC=${PIPESTATUS[0]}
bash $HOME/amcas-pro/tools/publish.sh "chanloop results $TAG (rc=$RC)" || true
exit $RC
