#!/bin/bash
#SBATCH -J gem5-trace
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=node04,node05
# Usage: sbatch -o LOG 04_trace.sh NAME MODEL CTX N_PREDICT [llm_trace.py options...]
# Captures one run into results/traces/NAME: dram.trc.gz, stats.txt, phases.tsv,
# amcas_tensors.tsv, and the generated text in simout.
set -uo pipefail
NAME=$1; MODEL=$2; CTX=$3; N=$4; shift 4
W=$HOME/amcas-pro/tracing
E=$W/mamba/envs/gem5
export LD_LIBRARY_PATH=$E/lib
OUT=$HOME/amcas-pro/results/traces/$NAME
mkdir -p $OUT
PROMPT="Once upon a time there was a little girl named Lily. She loved to play outside in the sunshine with her friends."
INPUT="-p '$PROMPT'"
# With SESSION=BASE, restore the prompt BASE.prompt from a session saved by
# 08_long_context_native.sh instead of evaluating it. CTX must match that run.
[ -n "${SESSION:-}" ] && INPUT="-f $SESSION.prompt --prompt-cache $SESSION.session --prompt-cache-ro"
CMD="$W/llama.cpp/build-gem5/bin/llama-completion -m $W/models/$MODEL.gguf -t 1 -c $CTX -b 512 \
$INPUT -n $N --ignore-eos -no-cnv -lm none --no-warmup --temp 0 --seed 1"
date
time $W/gem5/build/X86/${GEM5:-gem5.opt} --outdir=$OUT --redirect-stdout --redirect-stderr \
  $HOME/amcas-pro/configs/llm_trace.py --cmd "$CMD" --cwd $OUT "$@"
echo "gem5 exit=$?"
date
tail -3 $OUT/simout.txt | cut -c1-200
$W/bin/pkt2bin $OUT/dram.trc.gz $OUT/dram.bin > $OUT/dram.hdr && tail -1 $OUT/dram.hdr
ls -la $OUT
