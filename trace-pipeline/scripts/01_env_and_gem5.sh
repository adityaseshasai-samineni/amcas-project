#!/bin/bash
#SBATCH -J gem5-native
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/gem5native_%j.log
set -uo pipefail
W=$HOME/amcas-pro/tracing; cd $W
export MAMBA_ROOT_PREFIX=$W/mamba
mkdir -p $W/bin
if [ ! -x $W/bin/micromamba ]; then
  curl -sSL --retry 5 https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $W bin/micromamba || exit 1
fi
if [ ! -d $MAMBA_ROOT_PREFIX/envs/gem5 ]; then
  $W/bin/micromamba create -y -n gem5 -c conda-forge --strict-channel-priority \
    python=3.12 scons m4 pkg-config 'libprotobuf=3.21' zlib gcc=13 gxx=13 binutils \
    'sysroot_linux-64=2.28' cmake make libpng 2>&1 | tail -3 || exit 1
  [ -d $MAMBA_ROOT_PREFIX/envs/gem5/bin ] || exit 1
fi
E=$MAMBA_ROOT_PREFIX/envs/gem5
export PATH=$E/bin:$PATH
echo "=== toolchain"; which gcc g++ scons protoc python3-config; gcc --version | head -1; protoc --version; python3 --version
find $E -name libc.a | head; find $E -name 'libstdc++.a' | head -2
cd $W/gem5
export PKG_CONFIG_PATH=$E/lib/pkgconfig LD_LIBRARY_PATH=$E/lib CC=gcc CXX=g++
yes | scons build/X86/gem5.opt -j24 --ignore-style PYTHON_CONFIG=$E/bin/python3-config 2>&1 | tail -25 | cut -c1-200
ls -la build/X86/gem5.opt
build/X86/gem5.opt --version 2>&1 | head -3
