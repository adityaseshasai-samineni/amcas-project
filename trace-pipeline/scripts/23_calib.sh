#!/bin/bash
#SBATCH -J calib
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/calib_%j.log
# Calibrate the closed-loop (SimpleO3) model against native bandwidth: sweep window depth / MSHRs for a short-context proxy
# (7B 8k stream, weights window, bubbles 42) and compute-dilation k for 7B 32k. Output results/fullscale/calib/TAG/.
# Usage: 23_calib.sh TAG
set -uo pipefail
TAG=${1:-calib}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:$W/ramulator2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$W/ramulator2/python AMCAS_CAPACITY_GB=32
D=$HOME/amcas-pro/results/fullscale/drattr
R=$HOME/amcas-pro/results/fullscale/calib/$TAG; mkdir -p $R
S=/scratch/$USER/calib/$TAG; rm -rf $S; mkdir -p $S; trap 'rm -rf $S' EXIT
P=qwen2.5-7b-instruct-q4_k_m-00001-of-00002_s1n1t4_attr
gen() { # name run bubbles startfrac
  mkdir -p $S/$1
  $E/bin/python $HOME/amcas-pro/tools/policy_traces.py $D/$2/miss.gz $D/$2/tensors.tsv $S/$1 --window 1000000 --start-frac $4 \
    --bubbles $3 --placements thp --drams ddr5 --schemes identity 2>&1 | grep "class shares" | sed "s/^/$1 /"
}
gen short ${P}_c8k 42 ${SF_SHORT:-0.5}
for k in 1 6 8 10 12; do gen k32_$k ${P}_c32k $((k*53-1)) 0.3; done
rep() { # dir name window mshr bubbles
  $E/bin/python $HOME/amcas-pro/tools/policy_replay.py $S/$1 $R/$2.json --modes o3 --policies Open --bubbles $5 --procs 1 \
    --window-depth $3 --mshr $4 > $R/$2.txt 2>&1
}
export -f rep; export S R E HOME
{
for w in 128 192 256 384 512; do for m in 16 32; do echo "short short_w${w}_m${m} $w $m 42"; done; done
for k in 1 6 8 10 12; do for w in 128 256 384; do for m in 16 32; do echo "k32_$k k32_k${k}_w${w}_m${m} $((w*k)) $m $((k*53-1))"; done; done; done
} | xargs -P 8 -L1 bash -c 'rep $0 $1 $2 $3 $4'
echo done
bash $HOME/amcas-pro/tools/publish.sh "calib $TAG" || true
