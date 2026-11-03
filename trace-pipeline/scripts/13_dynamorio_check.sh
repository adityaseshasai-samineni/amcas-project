#!/bin/bash
#SBATCH -J dynamorio-check
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -w node01
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/drcheck_%j.log
# Can DynamoRIO's drmemtrace record optimized, multi-threaded llama.cpp here?
# Installs the latest Linux x86-64 release, traces 2 tokens of Qwen2.5-0.5B on 4
# threads into node01's /scratch, and reports slowdown, trace size and a cache
# simulation of the trace.
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
D=$W/dynamorio
if [ ! -x $D/bin64/drrun ]; then
  url=$(curl -s "https://api.github.com/repos/DynamoRIO/dynamorio/releases?per_page=30" | $E/bin/python -c '
import sys, json, re
for rel in json.load(sys.stdin):
    for a in rel["assets"]:
        if re.fullmatch(r"DynamoRIO-Linux-[0-9.]+\.tar\.gz", a["name"]):
            print(a["browser_download_url"]); sys.exit()')
  echo "release: $url"
  mkdir -p $D.tmp && curl -sL "$url" | tar -xz -C $D.tmp --strip-components=1 && mv $D.tmp $D || { echo "install failed"; exit 1; }
fi
$D/bin64/drrun -version
B=$W/llama.cpp/build-fast/bin/llama-completion
ARGS="-m $W/models/qwen25-05b-q4_0.gguf -t 4 -n 2 -p 'Once upon a time there was a little girl named Lily.' -no-cnv --temp 0 --seed 1"
S=/scratch/$USER/drcheck; rm -rf $S; mkdir -p $S
echo "=== native"; /usr/bin/time -f "native %e s, max RSS %M KB" bash -c "$B $ARGS" > $S/native.out 2> $S/native.err; tail -1 $S/native.err
echo "=== traced"; /usr/bin/time -f "traced %e s, max RSS %M KB" $D/bin64/drrun -t drcachesim -offline -outdir $S -- bash -c "$B $ARGS" \
  > $S/traced.out 2> $S/traced.err; tail -3 $S/traced.err
echo "same text: $(cmp -s <(tail -c 200 $S/native.out) <(tail -c 200 $S/traced.out) && echo yes || echo NO)"
du -sh $S/drmemtrace.*
echo "=== cache simulation (node01-like LLC, 4 cores)"
/usr/bin/time -f "simulated %e s" $D/bin64/drrun -t drcachesim -indir $S/drmemtrace.*llama* -cores 4 -LL_size 22M -LL_assoc 11 2>&1 | tail -40 | cut -c1-150
bash $HOME/amcas-pro/tools/publish.sh "DynamoRIO feasibility check on node01 (job $SLURM_JOB_ID)"
