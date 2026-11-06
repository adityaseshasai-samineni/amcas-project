#!/bin/bash
#SBATCH -J drburst
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -w node01
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/drburst_%j.log
# Trace chosen steps of native llama.cpp with llama-drburst and turn them into a DRAM
# request stream with its Ramulator replay. Any AVX-512 node (node01, node03) works,
# because nothing is reused from /scratch.
#
# Usage: sbatch [-w NODE] 16_drburst.sh MODEL SKIP NTRACE [THREADS]
#   SKIP    steps run untraced first: 0 is the prompt, each token one more step
#   NTRACE  steps to trace
# Environment: PROMPT_FILE (default: the short gem5 prompt), CTX (context size,
#   default 4096), TAG (run name suffix).
# Writes results/fullscale/drburst/<MODEL>_s<SKIP>n<NTRACE>t<THREADS><TAG>/.
#
# Steps:
#   1. untraced run with the same binary, for the reference text;
#   2. traced run, raw2trace (gzip), basic_counts;
#   3. drcachesim with the nextline prefetcher: DRAM reads = LL misses + prefetch misses;
#   4. drcachesim without prefetch, with -LL_miss_file: the full read stream;
#   5. tools/drmiss_export.py, then tools/ramulator_replay.py.
set -uo pipefail
MODEL=$1; SKIP=$2; NTRACE=$3; TH=${4:-4}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
D=$W/dynamorio; X=$W/drburst-build/llama-drburst
RUN=${MODEL}_s${SKIP}n${NTRACE}t${TH}${TAG:-}
R=$HOME/amcas-pro/results/fullscale/drburst/$RUN; mkdir -p $R
S=/scratch/$USER/drburst/$RUN; rm -rf $S; mkdir -p $S
if [ -n "${PROMPT_FILE:-}" ]; then INPUT=(-f "$PROMPT_FILE"); else INPUT=(-p "Once upon a time there was a little girl named Lily."); fi
N=$((SKIP + NTRACE))  # tokens to generate: step k (k >= 1) is the k-th generated token
ARGS=(-m $W/models/$MODEL.gguf -t $TH -c ${CTX:-4096} -n $N "${INPUT[@]}" -no-cnv --temp 0 --seed 1 --no-warmup)
CACHE=(-cores $TH -LL_size 22M -LL_assoc 11)
echo "run $RUN on $(hostname): ${ARGS[*]}"

echo "=== untraced"
AMCAS_SKIP_STEPS=999999 AMCAS_STEPS=$R/steps_native.txt /usr/bin/time -f "untraced %e s, max RSS %M KB" \
  $X "${ARGS[@]}" > $S/plain.out 2> $S/plain.err; tail -1 $S/plain.err
echo "=== traced (steps $SKIP to $((SKIP + NTRACE - 1)))"
AMCAS_SKIP_STEPS=$SKIP AMCAS_TRACE_STEPS=$NTRACE AMCAS_STEPS=$R/steps_traced.txt \
  DYNAMORIO_OPTIONS="-stderr_mask 0xc -client_lib ';;-offline -outdir $S'" \
  /usr/bin/time -f "traced %e s, max RSS %M KB" $X "${ARGS[@]}" > $S/traced.out 2> $S/traced.err
grep -E "^drburst|^traced" $S/traced.err
cmp -s <(tail -c 300 $S/plain.out) <(tail -c 300 $S/traced.out) && echo "same text: yes" || { echo "same text: NO"; exit 1; }
paste $R/steps_native.txt $R/steps_traced.txt
T=$(ls -d $S/drmemtrace.llama-drburst.*.dir) || exit 1; du -sh $T/raw
/usr/bin/time -f "raw2trace %e s" $D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip > $S/raw2trace.log 2>&1 \
  || { tail -5 $S/raw2trace.log; exit 1; }
tail -1 $S/raw2trace.log; du -sh $T/trace
$D/bin64/drrun -t drcachesim -indir $T -simulator_type basic_counts > $R/basic_counts.txt 2>&1
grep -E "total \(fetched\)|^Thread|^ +[0-9]+ \(fetched\) instructions" $R/basic_counts.txt | head -20

echo "=== cache simulation, with and without prefetch (in parallel)"
( /usr/bin/time -f "sim pf: %e s" $D/bin64/drrun -t drcachesim -indir $T "${CACHE[@]}" > $R/cachesim_pf.txt 2>&1 ) &
( /usr/bin/time -f "sim nopf: %e s" $D/bin64/drrun -t drcachesim -indir $T "${CACHE[@]}" -data_prefetcher none \
    -LL_miss_file $S/miss.gz > $R/cachesim_nopf.txt 2>&1 ) &
wait
grep -h "^sim " $R/cachesim_pf.txt $R/cachesim_nopf.txt
for f in pf nopf; do
  echo "LL $f: $(sed -n '/^LL /,$p' $R/cachesim_$f.txt | grep -E '^ *(Misses|Prefetch misses):' | tr -s ' ' | tr '\n' ' ')"
done

echo "=== DRAM stream and Ramulator"
$E/bin/python $HOME/amcas-pro/tools/drmiss_export.py $S/miss.gz $R 2>&1 | tee $R/export.txt
mv $R/dram.ramulator.trc $S/ 2>/dev/null
LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python \
  $E/bin/python $HOME/amcas-pro/tools/ramulator_replay.py $S/dram.ramulator.trc $R/ramulator.json 2>&1 | tee $R/ramulator.txt
cp $S/plain.err $S/traced.err $R/ 2>/dev/null
bash $HOME/amcas-pro/tools/publish.sh "DynamoRIO burst trace $RUN (job $SLURM_JOB_ID)"
