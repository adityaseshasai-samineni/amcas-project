#!/bin/bash
#SBATCH -J policy
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/policy_%j.log
# Row-policy study on an attributed LLC-miss stream (policy-dev).
# Usage: 19_policy.sh ATTR_RUN BUBBLES [WINDOW]   (ATTR_RUN as in 18_region.sh; BUBBLES = instructions per DRAM read)
# Needs the private Ramulator build (ramulator-ext/install.sh -> tracing/ramulator2-amcas).
set -uo pipefail
RUN=$1; BUB=$2; WIN=${3:-4000000}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$W/ramulator2-amcas:$E/lib OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$W/ramulator2-amcas/python
A=$HOME/amcas-pro/results/fullscale/drattr/$RUN
R=$HOME/amcas-pro/results/fullscale/policy/$RUN; mkdir -p $R
S=/scratch/$USER/policy/$RUN; rm -rf $S; mkdir -p $S
trap 'rm -rf $S' EXIT
T=$A/tensors.tsv; [ -s $T ] || T=$(ls $A/../*/tensors.tsv | head -1)
$E/bin/python $HOME/amcas-pro/tools/policy_traces.py $A/miss.gz $T $S --window $WIN --bubbles $BUB 2>&1 | tee $R/make.txt
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "policy_traces failed"; exit 1; }
cp $S/manifest.json $R/
$E/bin/python $HOME/amcas-pro/tools/policy_replay.py $S $R/replay.json --bubbles $BUB --procs 6 \
  --policies Open,ClosedCAP,Timeout,HAPPY,Region 2>&1 | grep -v "Ramulator::\|Loading trace\|Loaded" > $R/replay.txt
RC=${PIPESTATUS[0]}
bash $HOME/amcas-pro/tools/publish.sh "policy results $RUN (rc=$RC)" || true
exit $RC
