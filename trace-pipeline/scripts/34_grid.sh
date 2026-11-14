#!/bin/bash
#SBATCH -J grid
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=110G
#SBATCH --time=10:00:00
#SBATCH --exclude=node01,node02,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/grid_%j.log
# Mapping x row-policy x trace grid (grid-owner). Usage: 34_grid.sh TAG SPEC [SPEC ...]
#   SPEC = NAME=RELATIVE_RUN_DIR_UNDER_results/fullscale (needs miss.gz + tensors.tsv)
# env: WIN (4000000) PROCS (30) POLICIES MAPPINGS DRAMS PL KEEP=1 GENONLY=1 SKIPDONE=1
# Output per trace: results/fullscale/grid/NAME/{manifest.json,replay.json,replay.txt}; published after each trace.
set -uo pipefail
TAG=$1; shift
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export LD_LIBRARY_PATH=$E/lib:$W/ramulator2-grid OPENBLAS_NUM_THREADS=1 PYTHONPATH=$W/ramulator2-grid/python AMCAS_CAPACITY_GB=${CAP_GB:-32}
for SPEC in "$@"; do
  N=${SPEC%%=*}; D=${SPEC#*=}
  A=$HOME/amcas-pro/results/fullscale/$D
  R=$HOME/amcas-pro/results/fullscale/grid/$N; mkdir -p $R
  if [ -n "${SKIPDONE:-}" ] && [ -s $R/replay.json ] && [ -s $R/done ]; then echo "skip $N"; continue; fi
  S=/scratch/$USER/grid/$TAG/$N; rm -rf $S; mkdir -p $S
  echo "=== $N ($A) $(date)"; df -h /scratch | tail -1
  [ -s $A/miss.gz ] && [ -s $A/tensors.tsv ] || { echo "missing inputs in $A"; continue; }
  $E/bin/python $HOME/amcas-pro/tools/grid_gen.py $A/miss.gz $A/tensors.tsv $S --window ${WIN:-4000000} \
     --placements ${PL:-packed,thp} --drams ${DRAMS:-d4r6,x16} --procs 4 2>&1 | grep -v Warning | tee $R/make.txt
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "grid_gen failed for $N"; rm -rf $S; continue; }
  cp $S/manifest.json $R/
  [ -n "${GENONLY:-}" ] && { [ -n "${KEEP:-}" ] || rm -rf $S; continue; }
  $E/bin/python $HOME/amcas-pro/tools/grid_replay.py $S $R/replay.json --procs ${PROCS:-30} --tmp $S \
     --drams ${DRAMS:-d4r6,x16} --placements ${PL:-packed,thp} ${MAPPINGS:+--mappings $MAPPINGS} ${POLICIES:+--policies $POLICIES} 2>&1 \
     | grep --line-buffered -v "Ramulator::\|Loading trace\|Loaded" > $R/replay.txt
  RC=${PIPESTATUS[0]}; echo "replay rc=$RC for $N $(date)"; [ $RC -eq 0 ] && date > $R/done
  if [ -z "${NODS3:-}" ]; then   # DRAMsim3 DDR4-2933 6ch, OPEN_PAGE and CLOSE_PAGE, every mapping expressible as an address (MOP/ChRa via *3 streams)
    KEYS=$(ls $S | grep '_d4r6_.*\.trc$' | grep -v '_mxg\.trc' | sed 's/\.trc$//' | tr '\n' ',' | sed 's/,$//')
    [ -n "$KEYS" ] && $E/bin/python $HOME/amcas-pro/tools/dramsim3_replay.py $S $R/ds3.json --nch 6 --keys $KEYS --procs ${PROCS:-30} --tmp $S > $R/ds3.txt 2>&1 || echo "ds3 rc=$?"
  fi
  [ -n "${KEEP:-}" ] || rm -rf $S
  bash $HOME/amcas-pro/tools/publish.sh "grid results $N (rc=$RC)" || true
done
