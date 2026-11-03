#!/bin/bash
#SBATCH -J gem5-fast
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/gem5fast_%j.log
# gem5.fast with tcmalloc, for the runs that execute 100B instructions. gem5.opt
# stays in place: the running jobs use it, and it is the one with debug flags.
set -uo pipefail
W=$HOME/amcas-pro/tracing
export MAMBA_ROOT_PREFIX=$W/mamba
$W/bin/micromamba install -y -n gem5 -c conda-forge gperftools 2>&1 | tail -1
E=$MAMBA_ROOT_PREFIX/envs/gem5
export PATH=$E/bin:$PATH PKG_CONFIG_PATH=$E/lib/pkgconfig LD_LIBRARY_PATH=$E/lib CC=gcc CXX=g++
cd $W/gem5
yes | scons build/X86/gem5.fast -j24 --ignore-style PYTHON_CONFIG=$E/bin/python3-config 2>&1 | tail -8 | cut -c1-200
ls -la build/X86/gem5.fast
ldd build/X86/gem5.fast | grep -i tcmalloc
