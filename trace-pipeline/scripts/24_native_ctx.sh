#!/bin/bash
#SBATCH -J nativectx
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/nativectx_%j.log
# Native 7B decode tokens/s vs threads at 32k context (prompt = PROMPT_CHARS corpus prefix, as 17_drattr.sh) and at ~0 context.
# The prompt is evaluated once (32 threads) and saved with --prompt-cache; each thread count then reloads it read-only and decodes.
# Usage: 24_native_ctx.sh [MODEL] [THREADS "4 8 16 24 32 48"]   Env: PROMPT_CHARS (116000), CTX (32768), NTOK (32)
set -uo pipefail
MODEL=${1:-qwen2.5-7b-instruct-q4_k_m-00001-of-00002}; THR=${2:-"4 8 16 24 32 48"}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
B=$W/llama.cpp/build-fast/bin; F=$W/models/$MODEL.gguf
R=$HOME/amcas-pro/results/fullscale/native/ctx32k; mkdir -p $R
S=/scratch/$USER/nativectx; mkdir -p $S; trap 'rm -rf $S' EXIT
[ -d /scratch/$USER ] || { S=$HOME/nativectx_tmp; mkdir -p $S; }
lscpu | grep -E 'Model name|Socket|Core|NUMA node[0-9]|^CPU\(s\)' > $R/node.txt; hostname >> $R/node.txt; uptime >> $R/node.txt
head -c ${PROMPT_CHARS:-116000} $HOME/amcas-pro/results/sessions/corpus.txt > $S/prompt.txt
ARGS=(-m $F -c ${CTX:-32768} -b 2048 -ub 512 -f $S/prompt.txt -no-cnv --temp 0 --seed 1 --no-warmup --ignore-eos --prompt-cache $S/s.session)
echo "prefill (32 threads)"; date
$B/llama-completion "${ARGS[@]}" -t 32 -n 1 > $S/save.out 2> $R/save.err
grep -E "prompt eval time|^ *eval time|tokens per second" $R/save.err | tail -3
ls -l $S/s.session
for t in $THR; do
  $B/llama-completion "${ARGS[@]}" --prompt-cache-ro -t $t -n ${NTOK:-32} > $S/n$t.out 2> $R/dec_t$t.err
  echo "t=$t: $(grep -E '^llama_perf_context_print: +eval time' $R/dec_t$t.err | sed 's/.*(//')"
done
$B/llama-bench -m $F -p 0 -n 32 -t $(echo $THR | tr ' ' ,) -r 2 -o json > $R/short_threads.json 2> $R/short_threads.err
bash $HOME/amcas-pro/tools/publish.sh "native ctx32k thread sweep $SLURM_JOB_ID" || true
