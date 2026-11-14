#!/bin/bash
#SBATCH -J ds3loop
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=02:30:00
#SBATCH --exclude=node02,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/ds3loop_%j.log
# Closed-loop DRAMsim3 (extras-sim-owner): rebuilds the realmem streams (same code/params as 25_realmem.sh) then runs
# tools/dramsim3_loop over them. Usage: 32_ds3loop.sh ATTR_RUN TAG NCH [WINDOW]
# env: POINTS (default sat,fast:52:256,w512:52:512,plat:52:1024,k4:211:1024,k5:264:1280,unl:0:100000000:100000),
#      POL (default OPEN_PAGE), PL, SC, KEYS, SMOKE=1 (tiny window, one placement)
set -uo pipefail
RUN=$1; TAG=$2; NCH=$3; WIN=${4:-3000000}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:$W/ramulator2 OPENBLAS_NUM_THREADS=1 AMCAS_CAPACITY_GB=32
A=$HOME/amcas-pro/results/fullscale/drattr/$RUN
R=$HOME/amcas-pro/results/fullscale/ds3loop/$TAG; mkdir -p $R
S=/scratch/$USER/ds3loop/$TAG; rm -rf $S; mkdir -p $S; trap 'rm -rf $S' EXIT
D=$W/DRAMsim3; BIN=$W/bin/dramsim3_loop
if [ ! -x $BIN ] || [ $HOME/amcas-pro/tools/dramsim3_loop.cc -nt $BIN ]; then
  $E/bin/g++ -O2 -std=c++14 -I$D/src -I$D/ext/headers -I$D/ext/fmt/include $HOME/amcas-pro/tools/dramsim3_loop.cc -L$D -L$E/lib -ldramsim3 -Wl,-rpath,$D -Wl,-rpath,$E/lib -o $BIN || exit 1
fi
$E/bin/python $HOME/amcas-pro/tools/realmem_traces.py $A/miss.gz $A/tensors.tsv $S --nch $NCH --window $WIN \
  --placements ${PL:-packed,thp,4k} --schemes ${SC:-identity,xor_global} 2>&1 | tee $R/make.txt
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "realmem_traces failed"; exit 1; }
cp $S/manifest.json $R/
$E/bin/python $HOME/amcas-pro/tools/dramsim3_loop_run.py $S $R/loop.json --nch $NCH --procs 8 --tmp $S \
  --points ${POINTS:-sat,fast:52:256,w512:52:512,plat:52:1024,k4:211:1024,k5:264:1280,unl:0:100000000:100000} \
  --policies ${POL:-OPEN_PAGE} ${KEYS:+--keys $KEYS} 2>&1 | tee $R/loop.txt
RC=${PIPESTATUS[0]}
[ -n "${SMOKE:-}" ] || bash $HOME/amcas-pro/tools/publish.sh "ds3loop results $TAG (rc=$RC)" || true
exit $RC
