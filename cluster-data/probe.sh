#!/bin/bash
#SBATCH -J llama-probe
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/probe_%j.log
source /opt/rh/gcc-toolset-13/enable
cd ~/amcas-pro/tracing/models; rm -f tinymix.gguf
BIN=~/amcas-pro/tracing/llama.cpp/build-probe/bin/llama-completion
[ -f tinymoe4.gguf ] || curl -sSL -o tinymoe4.gguf https://huggingface.co/israellaguan/Tinyllama-moe4-Q8_0-GGUF/resolve/main/tinyllama-moe4-q8_0.gguf
ls -la
for m in stories15M tinymoe4; do for n in 1 9; do
 echo "### $m n=$n"
 timeout 900 perf stat -x, -e instructions:u,cycles:u,cache-misses:u,cache-references:u -- $BIN -m $m.gguf -t 1 -p "Once upon a time" -n $n --ignore-eos -no-cnv -lm none --seed 1 2>&1 | grep -E 'instructions|cycles|cache-|eval time|load time|error|n_expert|arch  |model size|model params|n_layer  '
done; done
