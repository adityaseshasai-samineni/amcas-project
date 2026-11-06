#!/bin/bash
#SBATCH -J drmem-dram
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -w node03
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/drdram_%j.log
# DRAM request streams from the step windows that 14_drmem_phases.sh cut, and their
# Ramulator replay, for comparison with the gem5 run of the same model.
#
# Usage: sbatch [-w NODE] 15_drmem_dram_trace.sh [MODEL] [WINDOWS...]
# Must run on the node whose /scratch holds the windows (node03 for job 117350).
#
# For each window: drcachesim with the prefetcher off and -LL_miss_file, so the miss
# file holds every line that reaches DRAM; tools/drmiss_export.py for stride
# structure, offline row-hit rates under modelled page placements, and a LoadStore
# trace; then tools/ramulator_replay.py over mappings and page policies.
# Writes results/fullscale/drdram/<model>/<window>/.
set -uo pipefail
MODEL=${1:-qwen25-05b-q4_0}; shift || true
WINDOWS=${*:-step0 step1}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
D=$W/dynamorio
S=/scratch/$USER/drphases/$MODEL
R=$HOME/amcas-pro/results/fullscale/drdram/$MODEL
CACHE=(-cores 4 -LL_size 22M -LL_assoc 11 -data_prefetcher none)

for w in $WINDOWS; do
  [ -d $S/win_$w ] || { echo "no window $S/win_$w on $(hostname); run 14_drmem_phases.sh here first"; exit 1; }
done
for w in $WINDOWS; do
  ( O=$R/$w; mkdir -p $O
    /usr/bin/time -f "sim $w: %e s" $D/bin64/drrun -t drcachesim -indir $S/win_$w "${CACHE[@]}" \
      -LL_miss_file $S/win_$w.miss.gz > $O/cachesim_nopf.txt 2>&1
    $E/bin/python $HOME/amcas-pro/tools/drmiss_export.py $S/win_$w.miss.gz $O > $O/export.txt 2>&1
    LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python \
      $E/bin/python $HOME/amcas-pro/tools/ramulator_replay.py $O/dram.ramulator.trc $O/ramulator.json > $O/ramulator.txt 2>&1
    echo "ramulator $w exit=$?" ) &
done
wait
for w in $WINDOWS; do
  O=$R/$w
  echo "=== $w"
  grep -E "^sim " $O/cachesim_nopf.txt
  echo "LL: $(sed -n '/^LL /,$p' $O/cachesim_nopf.txt | grep -m1 ' Misses:' | tr -s ' ')"
  cat $O/export.txt
  cat $O/ramulator.txt
done
bash $HOME/amcas-pro/tools/publish.sh "DynamoRIO DRAM streams and Ramulator replay of $MODEL (job $SLURM_JOB_ID)"
