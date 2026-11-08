#!/bin/bash
#SBATCH -J drburst
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --exclude=node02,node04,node05,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/drburst_%j.log
# Trace chosen steps of native llama.cpp with llama-drburst and turn them into a DRAM
# request stream with its Ramulator replay. The excluded nodes are GPU, drained or
# without AVX-512; node01, node03 (Cascade Lake) and node06 (Ice Lake) run build-fast.
#
# Usage: sbatch [--mem=..] [-c ..] 16_drburst.sh MODEL SKIP NTRACE [THREADS]
#   SKIP    steps run untraced first: 0 is the prompt, each token one more step
#   NTRACE  steps to trace
# Environment:
#   PROMPT_CHARS  build the prompt from the first N characters of results/sessions/
#                 corpus.txt (llama.cpp docs, about 4.2 characters per Qwen token)
#   PROMPT_FILE   or use this prompt file (default: the short gem5 prompt)
#   CTX           context size (default 4096)
#   TB            threads for the prompt (-tb; default THREADS). Keep it equal to
#                 THREADS: with -tb 16 -t 4 (jobs 118389, 118390) the 12 extra
#                 OpenMP threads are traced too once DynamoRIO attaches, and the
#                 32k run filled node06's /scratch in about 3 minutes
#   RAMU_PROCS    Ramulator configurations at once (default 6; each holds the trace)
#   TAG           run name suffix
# Writes results/fullscale/drburst/<MODEL>_s<SKIP>n<NTRACE>t<THREADS><TAG>/.
#
# Steps:
#   1. untraced run with the same binary, for the reference text;
#   2. traced run, raw2trace (gzip), basic_counts;
#   3. drcachesim with the nextline prefetcher (DRAM reads = LL misses + prefetch
#      misses) and without prefetch with -LL_miss_file (the full read stream);
#   4. the same without prefetch for each compute thread alone (-only_thread): the
#      miss file names the process, not the thread, so this is the only way to get
#      one thread's stream, for comparison with gem5's single core;
#   5. tools/drmiss_export.py, then tools/ramulator_replay.py on one DDR4-2400 channel
#      (comparable with every earlier result) and on 16 DDR5-4800 subchannels
#      (8 DIMM channels, 256 B interleave, the full-scale target).
set -uo pipefail
MODEL=$1; SKIP=$2; NTRACE=$3; TH=${4:-4}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
D=$W/dynamorio; X=$W/drburst-build/llama-drburst
RUN=${MODEL}_s${SKIP}n${NTRACE}t${TH}${TAG:-}
R=$HOME/amcas-pro/results/fullscale/drburst/$RUN; mkdir -p $R
S=/scratch/$USER/drburst/$RUN; rm -rf $S; mkdir -p $S
# /scratch is shared by every user of the node: never leave traces behind (all
# results are written to $R), and do not start on a nearly full disk.
# Keep the logs, so a failed run can still be diagnosed.
trap 'cp $S/*.err $S/raw2trace.log $R/ 2>/dev/null; rm -rf $S' EXIT
FREE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 100 ] || { echo "only ${FREE_GB} GB free on $(hostname):/scratch; not tracing"; exit 1; }
if [ -n "${PROMPT_CHARS:-}" ]; then
  head -c $PROMPT_CHARS $HOME/amcas-pro/results/sessions/corpus.txt > $R/prompt.txt; INPUT=(-f $R/prompt.txt)
elif [ -n "${PROMPT_FILE:-}" ]; then INPUT=(-f "$PROMPT_FILE")
else INPUT=(-p "Once upon a time there was a little girl named Lily."); fi
N=$((SKIP + NTRACE))  # tokens to generate: step k (k >= 1) is the k-th generated token
ARGS=(-m $W/models/$MODEL.gguf -t $TH -tb ${TB:-$TH} -c ${CTX:-4096} -n $N "${INPUT[@]}" -no-cnv --temp 0 --seed 1 --no-warmup)
CACHE=(-LL_size 22M -LL_assoc 11)
echo "run $RUN on $(hostname): ${ARGS[*]}"

echo "=== untraced"
AMCAS_SKIP_STEPS=999999 AMCAS_STEPS=$R/steps_native.txt /usr/bin/time -f "untraced %e s, max RSS %M KB" \
  $X "${ARGS[@]}" > $S/plain.out 2> $S/plain.err; tail -1 $S/plain.err
grep -E "prompt eval time|eval time" $S/plain.err | head -3 | cut -c1-160
echo "=== traced (steps $SKIP to $((SKIP + NTRACE - 1))), raw trace capped at 16 GB per thread"
AMCAS_SKIP_STEPS=$SKIP AMCAS_TRACE_STEPS=$NTRACE AMCAS_STEPS=$R/steps_traced.txt \
  DYNAMORIO_OPTIONS="-stderr_mask 0xc -client_lib ';;-offline -outdir $S -max_trace_size 17179869184'" \
  /usr/bin/time -f "traced %e s, max RSS %M KB" $X "${ARGS[@]}" > $S/traced.out 2> $S/traced.err
grep -E "^drburst|^traced" $S/traced.err
cmp -s <(tail -c 300 $S/plain.out) <(tail -c 300 $S/traced.out) && echo "same text: yes" || { echo "same text: NO"; exit 1; }
paste $R/steps_native.txt $R/steps_traced.txt | tail -$((NTRACE + 2))
T=$(ls -d $S/drmemtrace.llama-drburst.*.dir) || exit 1; du -sh $T/raw
/usr/bin/time -f "raw2trace %e s" $D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip > $S/raw2trace.log 2>&1 \
  || { tail -5 $S/raw2trace.log; exit 1; }
tail -1 $S/raw2trace.log; du -sh $T/trace
$D/bin64/drrun -t drcachesim -indir $T -simulator_type basic_counts > $R/basic_counts.txt 2>&1
grep -E "total \(fetched\)|^Thread|^ +[0-9]+ \(fetched\) instructions" $R/basic_counts.txt | head -20
# Compute threads: those with more than 1M instructions in the traced steps.
THREADS=$(awk '/^Thread [0-9]+ counts:/ {t=$2} /\(fetched\) instructions/ && t && $1 > 1000000 {print t; t=""}' $R/basic_counts.txt)
echo "compute threads: $THREADS"

echo "=== cache simulation: with and without prefetch, and each thread alone (in parallel)"
( /usr/bin/time -f "sim pf: %e s" $D/bin64/drrun -t drcachesim -indir $T -cores $TH "${CACHE[@]}" > $R/cachesim_pf.txt 2>&1 ) &
( /usr/bin/time -f "sim nopf: %e s" $D/bin64/drrun -t drcachesim -indir $T -cores $TH "${CACHE[@]}" -data_prefetcher none \
    -LL_miss_file $S/miss.gz > $R/cachesim_nopf.txt 2>&1 ) &
for t in $THREADS; do
  ( mkdir -p $R/thread_$t
    $D/bin64/drrun -t drcachesim -indir $T -cores 1 "${CACHE[@]}" -data_prefetcher none -only_thread $t \
      -LL_miss_file $S/miss_$t.gz > $R/thread_$t/cachesim_nopf.txt 2>&1
    $E/bin/python $HOME/amcas-pro/tools/drmiss_export.py $S/miss_$t.gz $R/thread_$t --stride-only > $R/thread_$t/export.txt 2>&1 ) &
done
wait
grep -h "^sim " $R/cachesim_pf.txt $R/cachesim_nopf.txt
for f in pf nopf; do
  echo "LL $f: $(sed -n '/^LL /,$p' $R/cachesim_$f.txt | grep -E '^ *(Misses|Prefetch misses):' | tr -s ' ' | tr '\n' ' ')"
done
for t in $THREADS; do echo "thread $t: $(grep -E 'DRAM reads|stride, merged' $R/thread_$t/export.txt | tr '\n' ' ')"; done

echo "=== DRAM stream"
$E/bin/python $HOME/amcas-pro/tools/drmiss_export.py $S/miss.gz $R 2>&1 | tee $R/export.txt
mv $R/dram.ramulator.trc $S/ 2>/dev/null
RAMU=(env LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python $E/bin/python $HOME/amcas-pro/tools/ramulator_replay.py)
echo "=== Ramulator: one DDR4-2400 channel"
"${RAMU[@]}" $S/dram.ramulator.trc $R/ramulator.json --procs ${RAMU_PROCS:-6} 2>&1 | grep -v "LoadStoreTrace" | tee $R/ramulator.txt
echo "=== Ramulator: 16 DDR5-4800 subchannels, 256 B interleave"
"${RAMU[@]}" $S/dram.ramulator.trc $R/ramulator_ddr5x16.json --dram ddr5 --channels 16 --interleave-bits 2 --issue 4 \
  --procs ${RAMU_PROCS:-6} 2>&1 | grep -v "LoadStoreTrace" | tee $R/ramulator_ddr5x16.txt
cp $S/plain.err $S/traced.err $R/ 2>/dev/null
bash $HOME/amcas-pro/tools/publish.sh "DynamoRIO burst trace $RUN (job $SLURM_JOB_ID)"
