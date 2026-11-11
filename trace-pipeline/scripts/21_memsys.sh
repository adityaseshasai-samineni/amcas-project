#!/bin/bash
#SBATCH -J memsys
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --exclude=node02,node04,node05,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/memsys_%j.log
# Memory-system study on a kept miss stream: DDR5-4800 8 and 16 subchannels, channel hashing, 4 KiB placement, full windows.
# Usage: 21_memsys.sh ATTR_RUN TAG [WINDOW(0=full)] [PLACEMENTS] [DRAMS] [SCHEMES]
# Output: results/fullscale/memsys/TAG/ (manifest.json, replay_<placement>_<dram>_<scheme>.json). Env: PAR (replays at once, default 3), CAP_GB.
set -uo pipefail
RUN=$1; TAG=$2; WIN=${3:-12000000}; PL=${4:-thp,4k}; DR=${5:-ddr5x8}; SC=${6:-identity,chan,xor_global,xor_global_chan,region,region_chan}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
export AMCAS_CAPACITY_GB=${CAP_GB:-32}
A=$HOME/amcas-pro/results/fullscale/drattr/$RUN
MISS=${MISS:-$A/miss.gz}; TSV=${TSV:-$A/tensors.tsv}
R=$HOME/amcas-pro/results/fullscale/memsys/$TAG; mkdir -p $R
S=/scratch/$USER/memsys/$TAG; rm -rf $S; mkdir -p $S
trap 'rm -rf $S' EXIT
[ -s $MISS ] && [ -s $TSV ] || { echo "missing $MISS / $TSV"; exit 1; }
$E/bin/python $HOME/amcas-pro/tools/memsys_study.py $MISS $TSV $S --window $WIN --placements $PL --drams $DR --schemes $SC 2>&1 | tee $R/make.txt
cp $S/manifest.json $R/
replay() {
  f=$1; b=$(basename $f .trc)
  if [[ $b == *_ddr5x8_* ]]; then P=ddr5x8; elif [[ $b == *_ddr5_* ]]; then P=ddr5x16; else P=ddr5x8; fi
  env LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python $E/bin/python \
    $HOME/amcas-pro/tools/ramulator_replay.py $f $R/replay_$b.json --preset $P --mappers RoBaRaCoCh --procs 2 > $R/replay_$b.txt 2>&1
  rm -f $f
}
export -f replay; export R W E
ls $S/*.trc | xargs -P ${PAR:-3} -I{} bash -c 'replay {}'
bash $HOME/amcas-pro/tools/publish.sh "memsys results $TAG" || true
