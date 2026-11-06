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
# tools/decode_stamp.c (LD_PRELOAD) logs the wall-clock start and end of every
# llama_process call (one per prompt or token step) in the clock and epoch of
# DynamoRIO's trace timestamps. record_filter then cuts the trace into windows by
# timestamp, model load (before the first step) and each step, and each window is
# cache-simulated on its own, starting cold. (Job 117023 found steps with
# -record_function and the view tool, which did not get through the main thread's
# 2.5G instructions in 5.5 hours; drcachesim -skip_to_timestamp needs an as-traced
# CPU schedule, which these traces do not have.)
# Writes results/fullscale/drphases/<model>/.
set -uo pipefail
MODEL=${1:-qwen25-05b-q4_0}; N=${2:-2}; TH=${3:-4}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
D=$W/dynamorio; B=$W/llama.cpp/build-fast/bin/llama-completion
R=$HOME/amcas-pro/results/fullscale/drphases/$MODEL; mkdir -p $R
S=/scratch/$USER/drphases/$MODEL; rm -rf $S; mkdir -p $S
# --no-warmup: no extra empty decode before the prompt, as in the gem5 runs.
ARGS=(-m $W/models/$MODEL.gguf -t $TH -n $N -p "Once upon a time there was a little girl named Lily." -no-cnv --temp 0 --seed 1 --no-warmup)
CACHE=(-cores $TH -LL_size 22M -LL_assoc 11)
SHIM=$W/lib/libdecode_stamp.so
[ -f $SHIM ] || { mkdir -p $W/lib && $E/bin/gcc -O2 -shared -fPIC -I $W/llama.cpp/include -I $W/llama.cpp/ggml/include \
  $HOME/amcas-pro/tools/decode_stamp.c -o $SHIM -ldl || exit 1; }

echo "=== native"; AMCAS_STEPS=$R/steps_native.txt LD_PRELOAD=$SHIM /usr/bin/time -f "native %e s" $B "${ARGS[@]}" > $S/native.out 2> $S/native.err; tail -1 $S/native.err
echo "=== traced"
AMCAS_STEPS=$R/steps_traced.txt LD_PRELOAD=$SHIM /usr/bin/time -f "traced %e s, max RSS %M KB" \
  $D/bin64/drrun -t drcachesim -offline -outdir $S -- $B "${ARGS[@]}" > $S/traced.out 2> $S/traced.err; tail -1 $S/traced.err
echo "same text: $(cmp -s <(tail -c 200 $S/native.out) <(tail -c 200 $S/traced.out) && echo yes || { echo NO; exit 1; })"
T=$(ls -d $S/drmemtrace.llama-completion.*.dir); du -sh $T/raw
/usr/bin/time -f "raw2trace %e s" $D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip > $S/raw2trace.log 2>&1 \
  || { tail -5 $S/raw2trace.log; exit 1; }
tail -3 $S/raw2trace.log; ls -la $T/trace | cut -c30-200

echo "=== basic counts (per thread)"
$D/bin64/drrun -t drcachesim -indir $T -simulator_type basic_counts > $R/basic_counts.txt 2>&1
grep -E "^Thread|instructions|Total counts" $R/basic_counts.txt | head -30 | cut -c1-120

echo "=== steps (B|E n_tokens time_us), native and traced"; paste $R/steps_native.txt $R/steps_traced.txt
[ -s $R/steps_traced.txt ] || { echo "no steps logged; is the wrapper loaded?"; exit 1; }
# Windows: "load" is everything before the first step; "stepK" is step K's B to E.
awk '$1=="B" {b=$3; if (!k) print "load 0 " b} $1=="E" {printf "step%d %s %s\n", k++, b, $3}' $R/steps_traced.txt > $R/windows.txt
cat $R/windows.txt

echo "=== cut each window and cache-simulate it (windows in parallel)"
while read -r name from to; do
  ( O=$S/win_$name; mkdir -p $O
    $D/bin64/drrun -t drcachesim -indir $T -simulator_type record_filter -outdir $O \
      -trim_before_timestamp $from -trim_after_timestamp $to > $R/filter_$name.txt 2>&1
    $D/bin64/drrun -t drcachesim -indir $O -simulator_type basic_counts > $R/counts_$name.txt 2>&1
    /usr/bin/time -f "sim $name: %e s" $D/bin64/drrun -t drcachesim -indir $O "${CACHE[@]}" > $R/cachesim_$name.txt 2>&1 ) &
done < $R/windows.txt
echo "=== whole trace"
$D/bin64/drrun -t drcachesim -indir $T "${CACHE[@]}" > $R/cachesim_all.txt 2>&1
wait
printf "%-8s %16s %14s %14s\n" window instructions LL_misses LL_compulsory
for name in all $(awk '{print $1}' $R/windows.txt); do
  f=$R/cachesim_$name.txt
  ins=$( [ $name = all ] && grep -m1 "total (fetched) instructions" $R/basic_counts.txt || grep -m1 "total (fetched) instructions" $R/counts_$name.txt )
  printf "%-8s %16s %14s %14s\n" $name "$(echo $ins | awk '{print $1}')" \
    "$(sed -n '/^LL /,$p' $f | grep -m1 ' Misses:' | awk '{print $2}')" "$(sed -n '/^LL /,$p' $f | grep -m1 'Compulsory misses:' | awk '{print $3}')"
done | tee $R/windows_summary.txt
grep -h "^sim " $R/cachesim_*.txt
cp $S/native.err $S/traced.err $R/ 2>/dev/null
bash $HOME/amcas-pro/tools/publish.sh "DynamoRIO load and per-step split of $MODEL (job $SLURM_JOB_ID)"
