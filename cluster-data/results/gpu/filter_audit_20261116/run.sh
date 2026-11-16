#!/bin/bash
#SBATCH -J filter-audit
#SBATCH -w node04
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 02:00:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/filter_audit_20261116
E=$HOME/amcas-pro/tracing/mamba/envs/gem5
export PATH=$E/bin:$PATH
trap 'rc=$?; echo $rc > "$R/exit_code.txt"; bash "$HOME/amcas-pro/tools/publish.sh" "GPU short filter audit finished status $rc"' EXIT
gcc -O2 "$R/gpu_filter2_audit.c" -o "$R/filter"
/usr/bin/time -v "$R/filter" /scratch/$USER/gt/q7_v100_short_faauto.raw "$R/short" --var baseline:hash=113 --var boundary:hash=113,invalidate=1,bypassstore=1 --var bypass:hash=113,l1kb=0 > "$R/stdout.txt" 2> "$R/filter.log"
sha256sum "$R/gpu_filter2_audit.c" /scratch/$USER/gt/q7_v100_short_faauto.raw.opcodes.txt > "$R/source_hashes.txt"
