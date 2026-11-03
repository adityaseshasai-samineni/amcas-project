#!/bin/bash
#SBATCH -J fetch-models
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/fetch_%j.log
# Download the full-scale models from Hugging Face, resuming interrupted transfers,
# and check each file's SHA-256 against the one Hugging Face lists.
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; M=$W/models
mkdir -p $M
fetch() {
  local repo=$1 file=$2 info sha size
  info=$(curl -s "https://huggingface.co/api/models/$repo/tree/main" | $E/bin/python -c '
import sys, json
f = sys.argv[1]
for x in json.load(sys.stdin):
    if x["path"] == f:
        print(x["lfs"]["oid"], x["size"])' "$file")
  sha=${info% *}; size=${info#* }
  [ -n "$sha" ] || { echo "$file: not listed in $repo"; return 1; }
  for i in $(seq 1 30); do
    [ -f $M/$file ] && [ "$(stat -c %s $M/$file)" = "$size" ] && break
    curl -sL -C - --retry 5 --connect-timeout 30 -o $M/$file "https://huggingface.co/$repo/resolve/main/$file"
    sleep 10
  done
  echo "$sha  $M/$file" | sha256sum -c - || { echo "$file: checksum mismatch, deleting"; rm -f $M/$file; return 1; }
}
fetch Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
fetch Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
fetch Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf
ls -la $M
bash $HOME/amcas-pro/tools/publish.sh "full-scale models downloaded (job $SLURM_JOB_ID)"
