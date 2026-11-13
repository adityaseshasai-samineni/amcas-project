#!/bin/bash
#SBATCH -J mop4
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=90G
#SBATCH --time=06:00:00
#SBATCH --exclude=node02,node04,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/mop4_%j.log
# MOP4CLXOR vs RoBaRaCoCh vs xor_global (and MOP4CLXOR+xor_global, ChRaBaRoCo) on a kept miss stream.
# Usage: 31_mop4.sh RUN_DIR_UNDER_results/fullscale TAG [WINDOW] ; env POINTS (sat,k1,w1024,k4), DRAMS, PL, O3WIN, PROCS, CAP_GB, KEEP=1
# Output: results/fullscale/mop4/TAG/ (make.txt, manifest.json, replay.json, replay.txt)
set -uo pipefail
RUN=$1; TAG=$2; WIN=${3:-8000000}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:$W/ramulator2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$W/ramulator2/python AMCAS_CAPACITY_GB=${CAP_GB:-32}
A=$HOME/amcas-pro/results/fullscale/$RUN
R=$HOME/amcas-pro/results/fullscale/mop4/$TAG; mkdir -p $R
S=/scratch/$USER/mop4/$TAG; rm -rf $S; mkdir -p $S; df -h /scratch | tail -1
[ -n "${KEEP:-}" ] || trap 'rm -rf $S' EXIT
[ -s $A/miss.gz ] && [ -s $A/tensors.tsv ] || { echo "missing inputs in $A"; exit 1; }
$E/bin/python $HOME/amcas-pro/tools/mop4_traces.py $A/miss.gz $A/tensors.tsv $S --window $WIN --o3-window ${O3WIN:-2000000} \
  --placements ${PL:-packed,thp,4k} --drams ${DRAMS:-x16,x8,d4r6,d4} 2>&1 | tee $R/make.txt
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "mop4_traces failed"; exit 1; }
cp $S/manifest.json $R/
$E/bin/python $HOME/amcas-pro/tools/mop4_replay.py $S $R/replay.json --procs ${PROCS:-14} --tmp $S \
  --placements ${PL:-packed,thp,4k} --drams ${DRAMS:-x16,x8,d4r6,d4} --points ${POINTS:-sat,k1,w1024,k4} 2>&1 | grep -v "Ramulator::\|Loading trace\|Loaded" > $R/replay.txt
RC=${PIPESTATUS[0]}
bash $HOME/amcas-pro/tools/publish.sh "mop4 results $TAG (rc=$RC)" || true
