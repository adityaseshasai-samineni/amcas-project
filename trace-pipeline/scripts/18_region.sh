#!/bin/bash
#SBATCH -J region
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --exclude=node02,node04,node05,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/region_%j.log
# Region-aware mapping vs baselines on an attributed LLC-miss stream.
# Usage: 18_region.sh ATTR_RUN [WINDOW]   (ATTR_RUN = results/fullscale/drattr/<run> with miss.gz and tensors.tsv,
#        from 17_drattr.sh with KEEP_MISS=1)
# tools/region_remap.py writes one Ramulator trace per placement x DRAM x scheme (to /scratch);
# each is replayed (RoBaRaCoCh only, Open and ClosedCAP; DDR4 1 channel, DDR5 16 subchannels at
# 256 B interleave, issue 4). Output: results/fullscale/region/<ATTR_RUN>/ (manifest.json, replay_*.json).
set -uo pipefail
RUN=$1; WIN=${2:-12000000}
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1
A=$HOME/amcas-pro/results/fullscale/drattr/$RUN
R=$HOME/amcas-pro/results/fullscale/region/$RUN; mkdir -p $R
S=/scratch/$USER/region/$RUN; rm -rf $S; mkdir -p $S
trap 'rm -rf $S' EXIT
[ -s $A/miss.gz ] && [ -s $A/tensors.tsv ] || { echo "missing miss.gz/tensors.tsv in $A"; exit 1; }
$E/bin/python $HOME/amcas-pro/tools/region_remap.py $A/miss.gz $A/tensors.tsv $S --window $WIN 2>&1 | tee $R/make.txt
cp $S/manifest.json $R/
replay() {
  f=$1; b=$(basename $f .trc)
  if [[ $b == *_ddr5_* ]]; then OPT=(--dram ddr5 --channels 16 --interleave-bits 2 --issue 4); else OPT=(--dram ddr4); fi
  env LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python $E/bin/python \
    $HOME/amcas-pro/tools/ramulator_replay.py $f $R/replay_$b.json "${OPT[@]}" --mappers RoBaRaCoCh --procs 2 > $R/replay_$b.txt 2>&1
  rm -f $f
}
export -f replay; export R W E
ls $S/*.trc | xargs -P 3 -I{} bash -c 'replay {}'
bash $HOME/amcas-pro/tools/publish.sh "region results $RUN" || true
