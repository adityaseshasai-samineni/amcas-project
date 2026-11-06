#!/bin/bash
#SBATCH -J drmem-phases
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -w node01
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/drphases_%j.log
# Split a DynamoRIO trace of native llama.cpp into model load and per-step inference.
#
# Usage: sbatch 14_drmem_phases.sh [MODEL] [N_PREDICT] [THREADS]
#
# Traces with -record_function on llama_decode, so every prompt or token step leaves
# a function marker. Finds each marker's timestamp in the main thread with the view
# tool, then cache-simulates from each step's start to the end of the trace with
# -skip_to_timestamp. The difference between consecutive runs is one step's LLC
# misses. Writes results/fullscale/drphases/<model>/.
set -uo pipefail
MODEL=${1:-qwen25-05b-q4_0}; N=${2:-2}; TH=${3:-4}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
D=$W/dynamorio; B=$W/llama.cpp/build-fast/bin/llama-completion
R=$HOME/amcas-pro/results/fullscale/drphases/$MODEL; mkdir -p $R
S=/scratch/$USER/drphases/$MODEL; rm -rf $S; mkdir -p $S
# --no-warmup: no extra empty decode before the prompt, as in the gem5 runs.
ARGS=(-m $W/models/$MODEL.gguf -t $TH -n $N -p "Once upon a time there was a little girl named Lily." -no-cnv --temp 0 --seed 1 --no-warmup)
CACHE=(-cores $TH -LL_size 22M -LL_assoc 11)

echo "=== native"; /usr/bin/time -f "native %e s" $B "${ARGS[@]}" > $S/native.out 2> $S/native.err; tail -1 $S/native.err
echo "=== traced"
/usr/bin/time -f "traced %e s, max RSS %M KB" $D/bin64/drrun -t drcachesim -offline -outdir $S \
  -record_function 'llama_decode|2' -- $B "${ARGS[@]}" > $S/traced.out 2> $S/traced.err; tail -1 $S/traced.err
echo "same text: $(cmp -s <(tail -c 200 $S/native.out) <(tail -c 200 $S/traced.out) && echo yes || { echo NO; exit 1; })"
T=$(ls -d $S/drmemtrace.llama-completion.*.dir); du -sh $T/raw
/usr/bin/time -f "raw2trace %e s" $D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip > $S/raw2trace.log 2>&1 \
  || { tail -5 $S/raw2trace.log; exit 1; }
tail -3 $S/raw2trace.log; ls -la $T/trace | cut -c30-200

echo "=== basic counts (per thread)"
$D/bin64/drrun -t drcachesim -indir $T -simulator_type basic_counts > $R/basic_counts.txt 2>&1
grep -E "^Thread|instructions|Total counts" $R/basic_counts.txt | head -30 | cut -c1-120

# The main thread is the process: its id is the pid in the directory name.
MAIN=$(basename $T | cut -d. -f3)
echo "=== function markers and timestamps on main thread $MAIN"
# Keep every function-marker line. "<marker: function #ID>" opens both a call and
# a return; a call is the one followed by "function return address". At each call,
# also print STEP, the last timestamp seen (the step's start time) and the main
# thread's instruction ordinal (view's second column).
/usr/bin/time -f "view %e s" $D/bin64/drrun -t drcachesim -indir $T -simulator_type view -only_thread $MAIN 2>&1 \
  | awk '/marker: timestamp/ {ts=$0; next}
         /marker: function/ {print
           if ($0 ~ /function #/) {pend=1; next}
           if (pend && $0 ~ /function return address/ && match(ts, /timestamp [0-9]+/))
             print "STEP " substr(ts, RSTART + 10, RLENGTH - 10) " instr " substr($2, 1, length($2) - 1)
           pend=0}' > $R/markers.txt
head -40 $R/markers.txt | cut -c1-160
grep '^STEP' $R/markers.txt | awk '{print $2}' > $R/step_ts.txt
echo "step starts (timestamp, main-thread instruction):"; grep '^STEP' $R/markers.txt

echo "=== cache simulation from each step's start to the end"
i=0
for ts in $(cat $R/step_ts.txt); do
  /usr/bin/time -f "sim from step $i: %e s" $D/bin64/drrun -t drcachesim -indir $T "${CACHE[@]}" -skip_to_timestamp $ts \
    > $R/cachesim_from_step$i.txt 2>&1
  tail -1 $R/cachesim_from_step$i.txt
  i=$((i+1))
done
echo "=== whole trace"
$D/bin64/drrun -t drcachesim -indir $T "${CACHE[@]}" > $R/cachesim_all.txt 2>&1
for f in $R/cachesim_all.txt $R/cachesim_from_step*.txt; do
  echo "$(basename $f): LL $(sed -n '/^LL /,$p' $f | grep -E '^ *(Misses|Compulsory misses):' | tr -s ' ' | tr '\n' ' ')"
done
cp $S/native.err $S/traced.err $R/ 2>/dev/null
bash $HOME/amcas-pro/tools/publish.sh "DynamoRIO load and per-step split of $MODEL (job $SLURM_JOB_ID)"
