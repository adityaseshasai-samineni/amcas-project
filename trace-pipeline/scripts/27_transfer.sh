#!/bin/bash
#SBATCH -J transfer
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --exclude=node02,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/transfer_%j.log
# transfer-owner task 4: one fixed XOR matrix searched on a train set, applied to other models/contexts (tools/transfer_remap.py).
# Usage: 27_transfer.sh [NAME] ; env WIN (default 8000000), PROCS (6), SMOKE=1 (tiny config, no replay)
set -uo pipefail
NAME=${1:-xfer}
df -h /scratch | tail -1
export AMCAS_CAPACITY_GB=32
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
D=$HOME/amcas-pro/results/fullscale/drattr
R=$HOME/amcas-pro/results/fullscale/transfer/$NAME; mkdir -p $R
S=/scratch/$USER/transfer/$NAME; rm -rf $S; mkdir -p $S; trap 'rm -rf $S' EXIT
Q7=$D/qwen2.5-7b-instruct-q4_k_m-00001-of-00002_s1n1t4_attr; Q30=$D/Qwen3-30B-A3B-Q4_K_M_s1n1t4_attr; Q05=$D/qwen25-05b-q4_0_s1n1t4_attr
$E/bin/python - > $S/cfg.json <<PY
import json
D="$D"
def st(p): return [p+"/miss.gz", p+"/tensors.tsv"]
streams={"7B32k":st("$Q7"+"_c32k"),"7B32k_t2":st("$D/qwen2.5-7b-instruct-q4_k_m-00001-of-00002_s2n1t4_attr_c32k_t2"),
 "30B32k":st("$Q30"+"_c32k"),"05B32k":st("$Q05"+"_c32k"),"7B8k":st("$Q7"+"_c8k"),"7B1k":st("$Q7"+"_c1k"),
 "30Bshort":st("$Q30"+"_miss"),"30B1k":st("$Q30"+"_c1k")}
main={"drams":["ddr5","ddr4"],"placements":["thp","packed"]}
oth={"drams":["ddr5"],"placements":["thp"]}
if "$SMOKE"=="1":
    streams={k:streams[k] for k in ("05B32k","30Bshort")}
    cfg=dict(streams=streams,trains={"T05":["05B32k"],"pool":["05B32k","30Bshort"]},tests={"05B32k":oth,"30Bshort":oth})
else:
    cfg=dict(streams=streams,trains={"T7":["7B32k"],"T30":["30B32k"],"pool":["7B32k","30B32k","05B32k","7B8k"]},
      tests={"7B32k":main,"30B32k":main,"05B32k":main,"7B32k_t2":oth,"7B8k":oth,"7B1k":oth,"30Bshort":oth,"30B1k":oth})
print(json.dumps(cfg))
PY
$E/bin/python $HOME/amcas-pro/tools/transfer_remap.py $S/cfg.json $S --window ${WIN:-8000000} --procs ${PROCS:-6} 2>&1 | tee $R/make.txt
cp $S/manifest.json $R/
[ "${SMOKE:-0}" = 1 ] && exit 0
replay() {
  f=$1; b=$(basename $f .trc)
  if [[ $b == *_ddr5_* ]]; then OPT=(--dram ddr5 --channels 16 --interleave-bits 2 --issue 4); else OPT=(--dram ddr4); fi
  env LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python $E/bin/python \
    $HOME/amcas-pro/tools/ramulator_replay.py $f $R/replay_$b.json "${OPT[@]}" --mappers RoBaRaCoCh --procs 2 > $R/replay_$b.txt 2>&1
  rm -f $f
}
export -f replay; export R W E
ls $S/*.trc | xargs -P 3 -I{} bash -c 'replay {}'
bash $HOME/amcas-pro/tools/publish.sh "transfer results $NAME" || true
