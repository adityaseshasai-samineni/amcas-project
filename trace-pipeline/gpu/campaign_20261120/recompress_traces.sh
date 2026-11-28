#!/bin/bash
#SBATCH -J trace-recompress
#SBATCH -w node04
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 12:00:00
# Lossless shrink of a RUNNING capture's finished NVBit traces, to stay under run_case.py's
# 80 GiB scratch trace budget (the tracer writes xz with a 1 MiB dictionary, about -1).
# Each finished kernel-*.trace.xz (not modified for >= 15 min) is recompressed with xz -6,
# largest first. It is replaced (atomic rename on the same filesystem, temp dir outside the
# capture dir) only if the md5 of its decompressed content is unchanged. Repeats passes while
# the capture is running. Log: $R/runs/$CASE/recompress_$SLURM_JOB_ID.tsv
# Usage: sbatch recompress_traces.sh CASE
set -Eeuo pipefail
C=$1; R=$HOME/amcas-pro/results/gpu/campaign_20261120; D=/scratch/$USER/amcas-campaign20/$C/capture/traces
T=/scratch/$USER/recompress_tmp_$SLURM_JOB_ID; mkdir -p "$T"; trap 'rm -rf "$T"' EXIT
LOG=$R/runs/$C/recompress_$SLURM_JOB_ID.tsv; touch "$LOG"
S0=$(date +%s)
running() { [ ! -f "$R/runs/$C/capture.status.json" ] || [ "$(stat -c %Y "$R/runs/$C/capture.status.json")" -lt "$S0" ]; }
while :; do
 n=0
 while read -r sz f; do
  grep -qxF "$f" <(cut -f1 "$LOG") && continue
  a=$(xz -dc "$D/$f" | md5sum | cut -d' ' -f1)
  xz -dc "$D/$f" | xz -6 -T16 -c > "$T/x.xz"
  b=$(xz -dc "$T/x.xz" | md5sum | cut -d' ' -f1)
  new=$(stat -c %s "$T/x.xz")
  if [ "$a" = "$b" ] && [ "$new" -lt "$sz" ]; then mv -f "$T/x.xz" "$D/$f"; st=replaced; else rm -f "$T/x.xz"; st=kept; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$sz" "$new" "$st" "$a" >> "$LOG"; n=$((n+1))
 done < <(find "$D" -maxdepth 1 -name 'kernel-*.trace.xz' -mmin +15 -printf '%s %f\n' | sort -rn)
 echo "$(date -Is) pass done: $n files; dir $(du -sb "$D" | cut -f1) bytes"
 running || break
 [ $n = 0 ] && sleep 600
done
awk -F'\t' '$4=="replaced"{o+=$2;n+=$3;k++}END{printf "replaced %d files: %.2f GB -> %.2f GB\n",k,o/1e9,n/1e9}' "$LOG"
