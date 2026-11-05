#!/bin/bash
#SBATCH -J dynamorio-check
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -w node01
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/drcheck_%j.log
# Can DynamoRIO's drmemtrace record optimized, multi-threaded llama.cpp here?
# Installs the latest Linux x86-64 release (or uses the one installed), traces 2 tokens of Qwen2.5-0.5B on 4
# threads into node01's /scratch, and reports slowdown, trace size and a cache
# simulation of the trace.
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
D=$W/dynamorio
if [ ! -x $D/bin64/drrun ]; then
  # The GitHub API allows 60 unauthenticated requests an hour per address, shared
  # by everyone behind the cluster's NAT, so retry if it refuses.
  for i in $(seq 1 12); do
  url=$(curl -s "https://api.github.com/repos/DynamoRIO/dynamorio/releases?per_page=30" | $E/bin/python -c '
import sys, json, re
for rel in json.load(sys.stdin):
    for a in rel["assets"]:
        if re.fullmatch(r"DynamoRIO-Linux-[0-9.]+\.tar\.gz", a["name"]):
            print(a["browser_download_url"]); sys.exit()' 2>/dev/null)
  [ -n "$url" ] && break; echo "release list unavailable, retrying"; sleep 300
  done
  echo "release: $url"
  mkdir -p $D.tmp && curl -sL "$url" | tar -xz -C $D.tmp --strip-components=1 && mv $D.tmp $D || { echo "install failed"; exit 1; }
fi
$D/bin64/drrun -version
B=$W/llama.cpp/build-fast/bin/llama-completion
# An array, so llama-completion is traced directly: no bash wrapper for DynamoRIO
# to follow into a child process (suspected cause of raw2trace failing in 116659).
ARGS=(-m $W/models/qwen25-05b-q4_0.gguf -t 4 -n 2 -p "Once upon a time there was a little girl named Lily." -no-cnv --temp 0 --seed 1)
S=/scratch/$USER/drcheck
# REUSE=1 keeps a raw trace already on node01's /scratch and only redoes the
# conversion and simulation.
[ "${REUSE:-0}" = 1 ] || rm -rf $S; mkdir -p $S
[ -s $S/native.out ] || { echo "=== native"; /usr/bin/time -f "native %e s, max RSS %M KB" $B "${ARGS[@]}" > $S/native.out 2> $S/native.err; tail -1 $S/native.err; }

# Trace, then convert raw files to the final format as its own step, verbose, so a
# failure shows its cause. The default zip output failed in job 116729 while closing
# a component after about 215 chunks of the main thread (probably zip's 4 GB limit),
# so write gzip. Second attempt only if the first fails: with the model read
# rather than memory-mapped (-lm none; this llama.cpp has no --no-mmap).
attempt() {  # NAME EXTRA_ARGS...
  local N=$1; shift; local O=$S/$N; mkdir -p $O
  if ls -d $O/drmemtrace.llama-completion.*.dir/raw >/dev/null 2>&1 && [ "${REUSE:-0}" = 1 ]; then
    echo "=== reusing raw trace ($N)"
  else
    echo "=== traced ($N)"
    /usr/bin/time -f "traced %e s, max RSS %M KB" $D/bin64/drrun -t drcachesim -offline -outdir $O -- $B "${ARGS[@]}" "$@" \
      > $O/traced.out 2> $O/traced.err; tail -2 $O/traced.err
  fi
  local same=$(cmp -s <(tail -c 200 $S/native.out) <(tail -c 200 $O/traced.out) && echo yes || echo NO)
  echo "same text: $same"; [ $same = yes ] || return 1
  T=$(ls -d $O/drmemtrace.llama-completion.*.dir); du -sh $T/raw
  echo "=== raw2trace ($N)"
  rm -rf $T/trace
  /usr/bin/time -f "raw2trace %e s" $D/tools/bin64/drraw2trace -indir $T -jobs 8 -compress gzip -verbose 1 > $O/raw2trace.log 2>&1
  local rc=$?; tail -15 $O/raw2trace.log | cut -c1-200; echo "raw2trace exit=$rc"
  [ $rc = 0 ] || return 1
  du -sh $T/trace; ls -la $T/trace | head
  echo "=== cache simulation ($N, node01-like LLC, 4 cores)"
  /usr/bin/time -f "simulated %e s" $D/bin64/drrun -t drcachesim -indir $T -cores 4 -LL_size 22M -LL_assoc 11 2>&1 | tail -40 | cut -c1-150
}
attempt mmap || attempt readmodel -lm none
mkdir -p $HOME/amcas-pro/results/fullscale/drcheck

for f in $S/*/raw2trace.log; do n=$(basename $(dirname $f)); cp $f $HOME/amcas-pro/results/fullscale/drcheck/raw2trace_$n.log; cp $(dirname $f)/traced.err $HOME/amcas-pro/results/fullscale/drcheck/traced_$n.err; done
bash $HOME/amcas-pro/tools/publish.sh "DynamoRIO feasibility check on node01 (job $SLURM_JOB_ID)"
