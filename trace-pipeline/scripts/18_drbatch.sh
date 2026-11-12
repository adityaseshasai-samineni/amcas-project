#!/bin/bash
#SBATCH -J drbatch
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --exclude=node02,node04,node05,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/drbatch_%j.log
# Batched decoding burst: B sequences, each with its own prompt (a different slice of
# results/sessions/corpus.txt, P tokens each), decoded together; one decode step (one
# llama_process call with B tokens, after all prompts) is traced with DynamoRIO.
# Usage: sbatch 18_drbatch.sh MODEL B [THREADS]   (long prompts: P=8192 STRIDE=40000 MAXTRACE_GB=32 ...)
# Env: P (prompt tokens per sequence, default 512), NGEN (decode steps run, default 3),
#      SKIP (decode step to trace; 0 = last-prompt-token step, default 1), NTRACE (1),
#      KVU=1 (unified KV buffer), TAG, RAMU_PROCS (3), SESSIONS (prompt cache dir,
#      default results/sessions/drbatch), NOTRACE_KEEP unused.
# Prompt cache: the first P-1 prompt tokens of each sequence are saved once
# (llama_state_seq_save_file) and restored by every later run (also the traced one).
# Step numbering is by decode step, not by llama_process call, so a restored or a freshly
# evaluated prompt gives the same step index (no prompt-step counting is needed).
# Binary: tracing/drbatch-build/llama-drbatch (cmake -DBATCH=ON, tools/drburst/CMakeLists.txt).
# Steps: untraced reference (token ids), traced step (+ tensor dump), raw2trace gzip,
# basic_counts, drcachesim with prefetch (DRAM reads = LL misses + prefetch misses) and
# without (LL miss file), drmiss_export.py --tensors, Ramulator DDR4 x1 and DDR5 x16.
set -uo pipefail
MODEL=$1; B=$2; TH=${3:-4}; P=${P:-512}; SKIP=${SKIP:-1}; NTRACE=${NTRACE:-1}; NGEN=${NGEN:-3}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
D=$W/dynamorio; X=$W/drbatch-build/llama-drbatch
RUN=${MODEL}_B${B}_P${P}t${TH}${TAG:-}
R=$HOME/amcas-pro/results/fullscale/drbatch/$RUN; mkdir -p $R
S=/scratch/$USER/drbatch/$RUN; rm -rf $S; mkdir -p $S
trap 'cp $S/*.err $S/raw2trace.log $R/ 2>/dev/null; rm -rf $S' EXIT
FREE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 100 ] || { echo "only ${FREE_GB} GB free on $(hostname):/scratch; not tracing"; exit 1; }
export AMCAS_CAPACITY_GB=${CAP_GB:-32}  # drmiss_export placement model; 8 GiB overflows for MoE with B>=4 (sparse 2 MiB pages)
CD=${SESSIONS:-$HOME/amcas-pro/results/sessions/drbatch}; mkdir -p $CD
export AMCAS_MODEL=$W/models/$MODEL.gguf AMCAS_B=$B AMCAS_P=$P AMCAS_THREADS=$TH AMCAS_NGEN=$NGEN AMCAS_KVU=${KVU:-0} \
  AMCAS_CORPUS=$HOME/amcas-pro/results/sessions/corpus.txt AMCAS_CACHE_DIR=$CD
[ -n "${STRIDE:-}" ] && export AMCAS_STRIDE=$STRIDE   # corpus offset between sequences (chars); part of the prompt-cache file name
MAXTR=$(( ${MAXTRACE_GB:-16} * 1073741824 ))   # -max_trace_size per thread
CACHE=(-LL_size 22M -LL_assoc 11)
echo "run $RUN on $(hostname): B=$B P=$P threads=$TH skip=$SKIP ntrace=$NTRACE kvu=${KVU:-0}"

echo "=== untraced"
AMCAS_SKIP_STEPS=999999 AMCAS_OUT=$R/tokens_native.txt AMCAS_STEPS=$R/steps_native.txt \
  /usr/bin/time -f "untraced %e s, max RSS %M KB" $X > $S/plain.out 2> $S/plain.err; grep -E "^drbatch|^untraced" $S/plain.err | tail -$((B + 1))
[ -s $R/tokens_native.txt ] || { tail -5 $S/plain.err; exit 1; }
echo "=== traced (decode step $SKIP, $NTRACE step(s)), raw trace capped at 16 GB per thread"
AMCAS_SKIP_STEPS=$SKIP AMCAS_TRACE_STEPS=$NTRACE AMCAS_OUT=$R/tokens_traced.txt AMCAS_STEPS=$R/steps_traced.txt \
  AMCAS_TENSORS=$R/tensors.tsv \
  DYNAMORIO_OPTIONS="-stderr_mask 0xc -client_lib ';;-offline -outdir $S -max_trace_size $MAXTR'" \
  /usr/bin/time -f "traced %e s, max RSS %M KB" $X > $S/traced.out 2> $S/traced.err
grep -E "^drbatch: (att|det)|^traced" $S/traced.err
# Positional token gate: per sequence (column), all tokens of steps before the traced step ($SKIP) must be
# identical; the first divergence (if any) must be at step >= $SKIP (a near-tie logit flip at the traced step
# under DynamoRIO; later tokens of that sequence then cascade, so their count says nothing).
awk -v skip=$SKIP 'NR==FNR { for (i=1;i<=NF;i++) a[FNR,i]=$i; n[FNR]=NF; rows=FNR; next }
  { for (i=1;i<=NF;i++) if (!((i) in fd) && a[FNR,i]!=$i) fd[i]=FNR-1; cols=NF }
  END { bad=0; for (i=1;i<=cols;i++) { if (i in fd) { printf "seq %d: first divergence at step %d%s\n", i-1, fd[i], (fd[i]<skip ? "  <-- BEFORE traced step" : ""); if (fd[i]<skip) bad=1 } else printf "seq %d: identical\n", i-1 }
    exit bad }' $R/tokens_native.txt $R/tokens_traced.txt | tee $R/token_check.txt
[ ${PIPESTATUS[0]} -eq 0 ] && echo "same tokens (positional check): yes" || { echo "same tokens: NO (divergence before the traced step)"; exit 1; }
paste $R/steps_native.txt $R/steps_traced.txt | head -$((2 * NGEN + 2))
T=$(ls -d $S/drmemtrace.llama-drbatch.*.dir) || exit 1; du -sh $T/raw
/usr/bin/time -f "raw2trace %e s" $D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip > $S/raw2trace.log 2>&1 \
  || { tail -5 $S/raw2trace.log; exit 1; }
tail -1 $S/raw2trace.log; du -sh $T/trace
$D/bin64/drrun -t drcachesim -indir $T -simulator_type basic_counts > $R/basic_counts.txt 2>&1
grep -E "total \(fetched\)|^Thread|^ +[0-9]+ \(fetched\) instructions" $R/basic_counts.txt | head -12

echo "=== cache simulation: with and without prefetch (in parallel)"
( /usr/bin/time -f "sim pf: %e s" $D/bin64/drrun -t drcachesim -indir $T -cores $TH "${CACHE[@]}" > $R/cachesim_pf.txt 2>&1 ) &
( /usr/bin/time -f "sim nopf: %e s" $D/bin64/drrun -t drcachesim -indir $T -cores $TH "${CACHE[@]}" -data_prefetcher none \
    -LL_miss_file $S/miss.gz > $R/cachesim_nopf.txt 2>&1 ) &
wait
grep -h "^sim " $R/cachesim_pf.txt $R/cachesim_nopf.txt
for f in pf nopf; do
  echo "LL $f: $(sed -n '/^LL /,$p' $R/cachesim_$f.txt | grep -E '^ *(Misses|Prefetch misses):' | tr -s ' ' | tr '\n' ' ')"
done

echo "=== DRAM stream and tensor attribution"
$E/bin/python $HOME/amcas-pro/tools/drmiss_export.py $S/miss.gz $R --tensors $R/tensors.tsv 2>&1 | tee $R/export.txt | tail -40
mv $R/dram.ramulator.trc $S/ 2>/dev/null
RAMU=(env LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python $E/bin/python $HOME/amcas-pro/tools/ramulator_replay.py)
echo "=== Ramulator: one DDR4-2400 channel"
"${RAMU[@]}" $S/dram.ramulator.trc $R/ramulator.json --procs ${RAMU_PROCS:-3} 2>&1 | grep -v "LoadStoreTrace" | tee $R/ramulator.txt
echo "=== Ramulator: 16 DDR5-4800 subchannels, 256 B interleave"
"${RAMU[@]}" $S/dram.ramulator.trc $R/ramulator_ddr5x16.json --dram ddr5 --channels 16 --interleave-bits 2 --issue 4 \
  --procs ${RAMU_PROCS:-3} 2>&1 | grep -v "LoadStoreTrace" | tee $R/ramulator_ddr5x16.txt
cp $S/plain.err $S/traced.err $R/ 2>/dev/null
bash $HOME/amcas-pro/tools/publish.sh "DynamoRIO batched decode $RUN (job ${SLURM_JOB_ID:-local})"
