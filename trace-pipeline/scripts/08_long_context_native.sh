#!/bin/bash
#SBATCH -J long-ctx-native
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/longctx_%j.log
# Evaluate long prompts natively and save them as sessions that a gem5 capture can
# restore, and measure the instructions per token, to plan the simulation budget.
# Usage: sbatch 08_long_context_native.sh [MODEL:TOKENS ...]
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
BIN=$W/llama.cpp/build-native/bin/llama-completion
S=$HOME/amcas-pro/results/sessions; mkdir -p $S
SPECS=${*:-stories15M:256 stories15M:1024 stories15M:4096 qwen25-05b-q4_0:256 qwen25-05b-q4_0:1024 tinymoe4:256 tinymoe4:1024}
CORPUS=$S/corpus.txt
(cd $W/llama.cpp && find docs -name '*.md' | sort | xargs cat) > $CORPUS
# A context too small for the prompt makes llama report its length and stop before
# evaluating anything. The binary has no separate tokenizer tool.
count_tokens() {
  $BIN -m $W/models/$1.gguf -t 1 -c 8 -f $2 -n 1 -no-cnv --no-warmup 2>&1 \
    | sed -n 's/.*prompt is too long (\([0-9]*\) tokens.*/\1/p'
}
for spec in $SPECS; do
  m=${spec%%:*}; want=${spec##*:}; base=$S/$m.$want
  # The prompt is the shortest prefix of a fixed text with at least the target
  # number of tokens. The text only needs to be stable and not repetitive.
  lo=1; hi=$(wc -c < $CORPUS)
  while [ $lo -lt $hi ]; do
    mid=$(((lo + hi) / 2)); head -c $mid $CORPUS > $base.prompt
    if [ "$(count_tokens $m $base.prompt)" -ge "$want" ]; then hi=$mid; else lo=$((mid + 1)); fi
  done
  head -c $lo $CORPUS > $base.prompt
  ctx=$(($(count_tokens $m $base.prompt) + 64)); echo $ctx > $base.ctx
  ARGS="-m $W/models/$m.gguf -t 1 -c $ctx -b 512 -f $base.prompt --ignore-eos -no-cnv -lm none --no-warmup --temp 0 --seed 1"
  rm -f $base.session
  echo "### $spec ctx=$ctx"
  timeout 14000 $BIN $ARGS -n 1 --prompt-cache $base.session >$base.save.out 2>$base.save.err
  echo "save exit=$?  $(grep -E 'prompt eval time' $base.save.err | cut -c20-110)"; ls -la $base.session | awk '{print "session bytes", $5}'
  for n in 1 5; do
    timeout 3000 perf stat -x, -e instructions:u,cache-misses:u -- $BIN $ARGS -n $n --prompt-cache $base.session --prompt-cache-ro \
      >$base.n$n.out 2>$base.n$n.err
    grep -E 'exact match|session file' $base.n$n.err | head -1 | cut -c20-100
    awk -F, -v n=$n '/instructions:u/{print "n=" n, "insts", $1} /cache-misses:u/{print "n=" n, "misses", $1}' $base.n$n.err
  done
  echo "tail: $(tail -c 80 $base.n5.out | tr '\n' ' ')"
done
