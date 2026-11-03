#!/bin/bash
#SBATCH -J native-fullscale
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH -w node01
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/nativefull_%j.log
# Native performance of the full-scale models on node01, alone on the node:
#  - decode speed against thread count, and against context depth;
#  - batched decode (several sequences sharing the weights) at two prompt lengths.
# Speeds give the DRAM bandwidth real inference needs, the regime the traces must
# reproduce. JSON output goes to results/fullscale/native/.
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
B=$W/llama.cpp/build-fast/bin; M=$W/models
O=$HOME/amcas-pro/results/fullscale/native; mkdir -p $O
declare -A MODEL=(
  [qwen05b]=$M/qwen25-05b-q4_0.gguf
  [qwen7b]=$M/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
  [qwen30ba3b]=$M/Qwen3-30B-A3B-Q4_K_M.gguf
)
lscpu | grep -E 'Model name|Socket|Core|NUMA node[0-9]' > $O/node.txt
DEPTH=$($B/llama-bench --help 2>&1 | grep -q -- '--n-depth' && echo yes)
for m in qwen05b qwen7b qwen30ba3b; do
  f=${MODEL[$m]}; [ -f $f ] || { echo "$m: missing $f"; continue; }
  echo "### $m ($(du -h $f | cut -f1))"
  # Decode only (-p 0), 64 tokens, against threads.
  $B/llama-bench -m $f -p 0 -n 64 -t 1,8,16,32 -r 3 -o json > $O/$m.threads.json 2> $O/$m.threads.err
  # Decode against context already in the KV cache, at 32 threads.
  [ -n "$DEPTH" ] && $B/llama-bench -m $f -p 0 -n 64 -t 32 -d 0,1024,8192,32768 -r 2 -o json > $O/$m.depth.json 2> $O/$m.depth.err
  # Batched decode: npl sequences, each with an npp-token prompt then 32 new tokens.
  $B/llama-batched-bench -m $f -c 140000 -b 2048 -ub 512 -t 32 -npp 1024,8192 -ntg 32 -npl 1,4,16 \
    --output-format jsonl > $O/$m.batched.jsonl 2> $O/$m.batched.err
  grep -h -o '"avg_ts": *[0-9.]*' $O/$m.threads.json | head -4 | tr '\n' ' '; echo
done
[ -z "$DEPTH" ] && echo "llama-bench has no --n-depth; depth sweep skipped"
bash $HOME/amcas-pro/tools/publish.sh "native full-scale baselines on node01 (job $SLURM_JOB_ID)"
