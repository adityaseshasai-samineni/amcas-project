#!/bin/bash
#SBATCH -n 1
#SBATCH --exclude=node02,node04,node05,node07,node08
#SBATCH -t 06:00:00
# Region attribution for the DynamoRIO route: trace decode step(s) with llama-drattr,
# which also dumps every tensor of the traced ggml graph(s) (name, address, size, type,
# op) to $R/tensors.tsv; then LLC-miss stream (no prefetch) -> drmiss_export.py --tensors.
# Usage: 17_drattr.sh MODEL SKIP NTRACE [THREADS]   (SKIP/NTRACE as 16_drburst.sh; short prompt)
# Env: CTX (default 4096), KEEP_MISS=1 keeps miss.gz in the results dir (not published: >50 MB),
#      PROMPT_CHARS=N uses the first N characters of results/sessions/corpus.txt as the prompt
#      (as 16_drburst.sh), TAG appended to the run name. Calls publish.sh at the end.
# Binary: tracing/drattr-build/llama-drattr (build: cmake -DATTR=ON, see tools/drburst/CMakeLists.txt)
set -uo pipefail
MODEL=$1; SKIP=$2; NTRACE=$3; TH=${4:-4}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
D=$W/dynamorio; X=$W/drburst-attr-build/llama-drattr
RUN=${MODEL}_s${SKIP}n${NTRACE}t${TH}_attr${TAG:-}
R=$HOME/amcas-pro/results/fullscale/drattr/$RUN; mkdir -p $R
S=/scratch/$USER/drattr/$RUN; rm -rf $S; mkdir -p $S
trap 'cp $S/*.err $S/raw2trace.log $R/ 2>/dev/null; rm -rf $S' EXIT
FREE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 60 ] || { echo "only ${FREE_GB} GB free on /scratch"; exit 1; }
N=$((SKIP + NTRACE))
if [ -n "${PROMPT_CHARS:-}" ]; then
  head -c $PROMPT_CHARS $HOME/amcas-pro/results/sessions/corpus.txt > $R/prompt.txt; INPUT=(-f $R/prompt.txt)
else INPUT=(-p "Once upon a time there was a little girl named Lily."); fi
ARGS=(-m $W/models/$MODEL.gguf -t $TH -tb $TH -c ${CTX:-4096} -n $N "${INPUT[@]}" -no-cnv --temp 0 --seed 1 --no-warmup)
CACHE=(-LL_size 22M -LL_assoc 11)
AMCAS_SKIP_STEPS=999999 AMCAS_STEPS=$R/steps_native.txt $X "${ARGS[@]}" > $S/plain.out 2> $S/plain.err
PROMPT_STEPS=$(( $(grep -c '^B' $R/steps_native.txt) - (N - 1) ))
FIRST=$((PROMPT_STEPS + SKIP - 1))
echo "tracing llama_process calls $FIRST..$((FIRST + NTRACE - 1))"
AMCAS_SKIP_STEPS=$FIRST AMCAS_TRACE_STEPS=$NTRACE AMCAS_STEPS=$R/steps_traced.txt AMCAS_TENSORS=$R/tensors.tsv \
  DYNAMORIO_OPTIONS="-stderr_mask 0xc -client_lib ';;-offline -outdir $S -max_trace_size 17179869184'" \
  $X "${ARGS[@]}" > $S/traced.out 2> $S/traced.err
grep -E "^drburst" $S/traced.err
cmp -s <(tail -c 300 $S/plain.out) <(tail -c 300 $S/traced.out) && echo "same text: yes" || { echo "same text: NO"; diff <(tail -c 300 $S/plain.out) <(tail -c 300 $S/traced.out) | head -20; tail -5 $S/traced.err; exit 1; }
wc -l $R/tensors.tsv
T=$(ls -d $S/drmemtrace.llama-drattr.*.dir) || exit 1
$D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip > $S/raw2trace.log 2>&1 || { tail -5 $S/raw2trace.log; exit 1; }
$D/bin64/drrun -t drcachesim -indir $T -cores $TH "${CACHE[@]}" -data_prefetcher none -LL_miss_file $S/miss.gz > $R/cachesim_nopf.txt 2>&1
$E/bin/python $HOME/amcas-pro/tools/drmiss_export.py $S/miss.gz $R/attr --tensors $R/tensors.tsv --no-trace 2>&1 | tee $R/export.txt | tail -60
[ -n "${KEEP_MISS:-}" ] && cp $S/miss.gz $R/
bash $HOME/amcas-pro/tools/publish.sh "drattr $RUN" || true
