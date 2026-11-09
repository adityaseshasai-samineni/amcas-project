#!/bin/bash
# Build the AMCAS row policies into a private copy of Ramulator 2.1 (the shared tree is untouched).
# Usage: install.sh   (run on a compute node: srun -w node03 -c 16 bash install.sh)
set -euo pipefail
SRC=$HOME/amcas-pro/tracing/ramulator2; DST=$HOME/amcas-pro/tracing/ramulator2-amcas
EXT=$HOME/amcas-pro/ramulator-ext
E=$HOME/amcas-pro/tracing/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
[ -d $DST ] || { mkdir -p $DST; rsync -a --exclude build $SRC/ $DST/; }
cp $EXT/rowpolicy/*.cpp $EXT/rowpolicy/*.h $DST/src/ramulator/controller/rowpolicy/impl/
C=$DST/src/ramulator/controller/CMakeLists.txt
grep -q region.cpp $C || sed -i 's#^\(\s*\)rowpolicy/impl/closed_cap.cpp#&\n\1rowpolicy/impl/region.cpp\n\1rowpolicy/impl/happy.cpp\n\1rowpolicy/impl/timeout.cpp#' $C
cd $DST
BUILD_DIR=$DST/build JOBS=${JOBS:-16} PATH=$E/bin:$PATH sh $DST/build.sh
PYTHONPATH=$DST/python $E/bin/python -m ramulator codegen || true
ls python/ramulator/row_policy
