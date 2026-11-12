#!/bin/bash
#SBATCH -J realmem
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=node02,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/realmem_%j.log
# 7B 32k on the cluster nodes' real memory model: DDR4-2933, N channels (6 = one Cascade Lake socket, 12 = two),
# identity vs xor_global vs region, packed/THP/4k, Open/ClosedCAP, saturated and closed-loop operating points.
# Usage: 25_realmem.sh ATTR_RUN TAG NCH [WINDOW] ; env POINTS (comma list: sat, name:BUBBLES:WINDOW), PL, SC, POL, CAP_GB, KEEP=1
set -uo pipefail
RUN=$1; TAG=$2; NCH=$3; WIN=${4:-3000000}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:$W/ramulator2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$W/ramulator2/python AMCAS_CAPACITY_GB=${CAP_GB:-32}
A=$HOME/amcas-pro/results/fullscale/drattr/$RUN
R=$HOME/amcas-pro/results/fullscale/realmem/$TAG; mkdir -p $R
S=/scratch/$USER/realmem/$TAG; rm -rf $S; mkdir -p $S; df -h /scratch | tail -1
[ -n "${KEEP:-}" ] || trap 'rm -rf $S' EXIT
$E/bin/python $HOME/amcas-pro/tools/realmem_traces.py $A/miss.gz $A/tensors.tsv $S --nch $NCH --window $WIN \
  --placements ${PL:-packed,thp,4k} --schemes ${SC:-identity,xor_global,region} 2>&1 | tee $R/make.txt
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "realmem_traces failed"; exit 1; }
cp $S/manifest.json $R/
$E/bin/python $HOME/amcas-pro/tools/realmem_replay.py $S $R/replay.json --nch $NCH --procs 8 --tmp $S \
  --policies ${POL:-Open,ClosedCAP} --points ${POINTS:-sat} 2>&1 | grep -v "Ramulator::\|Loading trace\|Loaded" > $R/replay.txt
RC=${PIPESTATUS[0]}
if [ -n "${DS3:-}" ]; then  # DRAMsim3 cross-check on the same streams (saturated, DDR4-2933)
  $E/bin/python $HOME/amcas-pro/tools/dramsim3_replay.py $S $R/dramsim3.json --nch $NCH --procs 8 --tmp $S > $R/dramsim3.txt 2>&1 || RC=$?
fi
bash $HOME/amcas-pro/tools/publish.sh "realmem results $TAG (rc=$RC)" || true
exit $RC
