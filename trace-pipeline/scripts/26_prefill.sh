#!/bin/bash
#SBATCH -J prefill
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --exclude=node02,node04,node05,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/prefill_%j.log
# 26_prefill.sh: windowed prefill trace. One sequence, a P-token prompt run as ONE llama_process call split
# into ceil(P/UB) graph computes (n_ubatch=UB); DynamoRIO is attached only for graph computes
# [COMP, COMP+NCOMP) (default: the last one). Keeps the no-prefetch LL miss stream (miss.gz) + tensors.tsv.
# Usage: sbatch 26_prefill.sh MODEL [THREADS]      Env: P (512) UB (64) COMP (last) NCOMP (1) NGEN (3) TAG MAXTRACE_GB (16)
# Untraced and traced runs must give the same greedy tokens after the prompt.
set -uo pipefail
MODEL=$1; B=1; TH=${2:-4}; P=${P:-512}; UB=${UB:-64}; NGEN=${NGEN:-3}; NCOMP=${NCOMP:-1}
COMP=${COMP:-$(( (P + UB - 1) / UB - 1 ))}; SKIP=0; NTRACE=1
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
D=$W/dynamorio; X=$W/drbatch-build/llama-drbatch
RUN=${MODEL}_prefill_P${P}u${UB}c${COMP}n${NCOMP}t${TH}${TAG:-}
R=$HOME/amcas-pro/results/fullscale/prefill/$RUN; mkdir -p $R
S=/scratch/$USER/prefill/$RUN; rm -rf $S; mkdir -p $S
trap 'cp $S/*.err $S/raw2trace.log $R/ 2>/dev/null; rm -rf $S' EXIT
FREE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 100 ] || { echo "only ${FREE_GB} GB free on $(hostname):/scratch; not tracing"; exit 1; }
export AMCAS_CAPACITY_GB=${CAP_GB:-32}  # drmiss_export placement model; 8 GiB overflows for MoE with B>=4 (sparse 2 MiB pages)
CD=${SESSIONS:-$HOME/amcas-pro/results/sessions/drbatch}; mkdir -p $CD
export AMCAS_MODEL=$W/models/$MODEL.gguf AMCAS_B=$B AMCAS_P=$P AMCAS_THREADS=$TH AMCAS_NGEN=$NGEN AMCAS_KVU=${KVU:-0} \
  AMCAS_PREFILL=1 AMCAS_UBATCH=$UB AMCAS_CORPUS=$HOME/amcas-pro/results/sessions/corpus.txt AMCAS_CACHE_DIR=$CD
[ -n "${STRIDE:-}" ] && export AMCAS_STRIDE=$STRIDE   # corpus offset between sequences (chars); part of the prompt-cache file name
MAXTR=$(( ${MAXTRACE_GB:-16} * 1073741824 ))   # -max_trace_size per thread
CACHE=(-LL_size 22M -LL_assoc 11)
echo "run $RUN on $(hostname): B=$B P=$P threads=$TH skip=$SKIP ntrace=$NTRACE kvu=${KVU:-0}"

echo "=== untraced"
AMCAS_TRACE_COMPUTE=999999 AMCAS_SKIP_STEPS=999999 AMCAS_OUT=$R/tokens_native.txt AMCAS_STEPS=$R/steps_native.txt \
  /usr/bin/time -f "untraced %e s, max RSS %M KB" $X > $S/plain.out 2> $S/plain.err; grep -E "^drbatch|^untraced" $S/plain.err | tail -3
[ -s $R/tokens_native.txt ] || { tail -5 $S/plain.err; exit 1; }
echo "=== traced (decode step $SKIP, $NTRACE step(s)), raw trace capped at 16 GB per thread"
AMCAS_TRACE_COMPUTE=$COMP AMCAS_TRACE_COMPUTES=$NCOMP AMCAS_SKIP_STEPS=$SKIP AMCAS_TRACE_STEPS=$NTRACE AMCAS_OUT=$R/tokens_traced.txt AMCAS_STEPS=$R/steps_traced.txt \
  AMCAS_TENSORS=$R/tensors.tsv \
  DYNAMORIO_OPTIONS="-stderr_mask 0xc -client_lib ';;-offline -outdir $S -max_trace_size $MAXTR'" \
  /usr/bin/time -f "traced %e s, max RSS %M KB" $X > $S/traced.out 2> $S/traced.err
grep -E "^drbatch: (att|det|prefill)|^traced" $S/traced.err
cmp -s $R/tokens_native.txt $R/tokens_traced.txt && echo "same tokens: yes" || { echo "same tokens: NO"; exit 1; }
paste $R/steps_native.txt $R/steps_traced.txt | head -$((2 * NGEN + 2))
T=$(ls -d $S/drmemtrace.llama-drbatch.*.dir) || exit 1; du -sh $T/raw
/usr/bin/time -f "raw2trace %e s" $D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip > $S/raw2trace.log 2>&1 \
  || { tail -5 $S/raw2trace.log; exit 1; }
tail -1 $S/raw2trace.log; du -sh $T/trace
$D/bin64/drrun -t drcachesim -indir $T -simulator_type basic_counts > $R/basic_counts.txt 2>&1
grep -E "total \(fetched\)|^Thread|^ +[0-9]+ \(fetched\) instructions" $R/basic_counts.txt | head -12

echo "=== cache simulation without prefetch, keep miss stream"
$D/bin64/drrun -t drcachesim -indir $T -cores $TH "${CACHE[@]}" -data_prefetcher none -LL_miss_file $S/miss.gz > $R/cachesim_nopf.txt 2>&1
cp $S/miss.gz $R/; ls -la $R/miss.gz

echo "=== thread ids in the miss stream"
zcat $R/miss.gz | cut -d, -f1 | sort | uniq -c | sort -rn | head -12 | tee $R/threads_in_miss.txt
zcat $R/miss.gz | wc -l
