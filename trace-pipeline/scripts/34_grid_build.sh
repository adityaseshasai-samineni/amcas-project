#!/bin/bash
#SBATCH -J gridbuild
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --exclude=node02,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/gridbuild_%j.log
# Build a PRIVATE copy ramulator2-grid = ramulator2-amcas + Adaptive and Oracle row policies (grid-owner).
# Does not touch ramulator2-amcas (other agents' jobs use it).
set -euo pipefail
SRC=$HOME/amcas-pro/tracing/ramulator2-amcas; DST=$HOME/amcas-pro/tracing/ramulator2-grid
EXT=$HOME/amcas-pro/ramulator-ext
E=$HOME/amcas-pro/tracing/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
[ -d $DST ] || { mkdir -p $DST; rsync -a --exclude build --exclude libramulator.so $SRC/ $DST/; }
cp $EXT/rowpolicy/*.cpp $EXT/rowpolicy/*.h $DST/src/ramulator/controller/rowpolicy/impl/
C=$DST/src/ramulator/controller/CMakeLists.txt
grep -q adaptive.cpp $C || sed -i 's#^\(\s*\)rowpolicy/impl/timeout.cpp#&\n\1rowpolicy/impl/adaptive.cpp\n\1rowpolicy/impl/oracle.cpp#' $C
cd $DST
BUILD_DIR=$DST/build JOBS=${JOBS:-16} PATH=$E/bin:$PATH sh $DST/build.sh
PYTHONPATH=$DST/python $E/bin/python -m ramulator codegen || true
ls python/ramulator/row_policy; ls -la libramulator.so
