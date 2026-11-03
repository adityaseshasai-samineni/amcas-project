#!/bin/bash
#SBATCH -J llama-native
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/native_%j.log
# Native run of the same SSE2 binary: reference tokens, instruction count and LLC
# misses to check gem5 against, and the budget needed to plan simulation time.
set -uo pipefail
W=$HOME/amcas-pro/tracing
BIN=$W/llama.cpp/build-native/bin/llama-completion
mkdir -p $HOME/amcas-pro/results/native
cd $W/models
PROMPT="Once upon a time there was a little girl named Lily. She loved to play outside in the sunshine with her friends."
for spec in stories15M:512 tinymoe4:256 qwen25-05b-q4_0:256; do
  m=${spec%%:*}; c=${spec##*:}
  for n in 1 9; do
    echo "### $m n=$n ctx=$c"
    timeout 1500 perf stat -x, -e instructions:u,cycles:u,cache-misses:u,cache-references:u -- \
      $BIN -m $m.gguf -t 1 -c $c -b 512 -p "$PROMPT" -n $n --ignore-eos -no-cnv -lm none --no-warmup \
      --temp 0 --seed 1 >$HOME/amcas-pro/results/native/$m.n$n.out 2>$HOME/amcas-pro/results/native/$m.n$n.err
    grep -E 'instructions:u|cycles:u|cache-|eval time|load time|error|n_expert  |n_layer  |n_embd  |model params|model size' \
      $HOME/amcas-pro/results/native/$m.n$n.err | cut -c1-110
    echo "output: $(tail -c 160 $HOME/amcas-pro/results/native/$m.n$n.out | tr '\n' ' ')"
  done
done
