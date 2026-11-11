#!/bin/bash
#SBATCH -J drwb
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --exclude=node02,node04,node05,node07,node08
#SBATCH -t 08:00:00
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/drwb_%j.log
# DRAM writebacks for the DynamoRIO route: trace decode step(s) as 17_drattr.sh, then tools/drwb.c simulates a
# write-back L1D + shared LL over the full trace (drcachesim has no dirty bits) and emits the read+write DRAM stream;
# tools/drwb_replay.py attributes writes to tensor classes and builds Ramulator traces with and without the writes.
# Usage: 20_drwb.sh MODEL SKIP NTRACE [THREADS]    Env: CTX, PROMPT_CHARS, TAG (as 17_drattr.sh), QUANTUM (default 1000
#   instructions per thread turn), WINDOW (default 12000000), CAP_GB (default 32), NOREPLAY=1 (stop after drwb + stats),
#   KEEP_STREAM=1 (copy stream.bin.gz into the results dir; large, not published).
# Binary: tracing/drattr-build/llama-drattr (Release!). Results: results/fullscale/drwb/<MODEL>_s<SKIP>n<N>t<T>_wb<TAG>/
set -uo pipefail
MODEL=$1; SKIP=$2; NTRACE=$3; TH=${4:-4}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:$W/ramulator2 OPENBLAS_NUM_THREADS=1
D=$W/dynamorio; X=$W/drburst-attr-build/llama-drattr
RUN=${MODEL}_s${SKIP}n${NTRACE}t${TH}_wb${TAG:-}
R=$HOME/amcas-pro/results/fullscale/drwb/$RUN; mkdir -p $R
S=/scratch/$USER/drwb/$RUN; rm -rf $S; mkdir -p $S
trap 'cp $S/*.err $S/raw2trace.log $R/ 2>/dev/null; rm -rf $S' EXIT
FREE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 80 ] || { echo "only ${FREE_GB} GB free on /scratch"; exit 1; }
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
cmp -s <(tail -c 300 $S/plain.out) <(tail -c 300 $S/traced.out) && echo "same text: yes" || { echo "same text: NO"; tail -5 $S/traced.err; exit 1; }
T=$(ls -d $S/drmemtrace.llama-drattr.*.dir) || exit 1
$D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip > $S/raw2trace.log 2>&1 || { tail -5 $S/raw2trace.log; exit 1; }
FILES=$(find $T -name '*.trace.gz' | sort); echo "$(echo "$FILES" | wc -l) thread files"; du -sh $T | tail -1
gcc -O2 -o $S/drwb $HOME/amcas-pro/tools/drwb.c -lz || exit 1
$D/bin64/drrun -t drcachesim -indir $T -cores $TH "${CACHE[@]}" -data_prefetcher none -LL_miss_file $S/miss.gz > $R/cachesim_nopf.txt 2>&1
echo "drcachesim LL: $(grep -A3 '^LL ' $R/cachesim_nopf.txt | grep Misses)"
$S/drwb $S/stream.bin ${QUANTUM:-1000} $FILES > $R/drwb.json && cat $R/drwb.json
ls -l $S/stream.bin | awk '{print $5/8 " requests"}'
[ -n "${NOREPLAY:-}" ] && exit 0
[ -n "${KEEP_STREAM:-}" ] && gzip -c $S/stream.bin > $R/stream.bin.gz
export AMCAS_CAPACITY_GB=${CAP_GB:-32}
$E/bin/python $HOME/amcas-pro/tools/drwb_replay.py $S/stream.bin $R/tensors.tsv $S/tr --window ${WINDOW:-12000000} 2>&1 | tee $R/make.txt | tail -50
cp $S/tr/writes.json $S/tr/manifest.json $R/
replay() {
  f=$1; b=$(basename $f .trc)
  if [[ $b == *_ddr5_* ]]; then OPT=(--dram ddr5 --channels 16 --interleave-bits 2 --issue 4); else OPT=(--dram ddr4); fi
  env PYTHONPATH=$W/ramulator2/python $E/bin/python $HOME/amcas-pro/tools/ramulator_replay.py $f $R/replay_$b.json "${OPT[@]}" --mappers RoBaRaCoCh --procs 2 > $R/replay_$b.txt 2>&1
  rm -f $f
}
export -f replay; export R W E
ls $S/tr/*.trc | xargs -P 3 -I{} bash -c 'replay {}'
bash $HOME/amcas-pro/tools/publish.sh "drwb $RUN" || true
