#!/bin/bash
#SBATCH -J moe4rep
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --exclude=node02,node05,node06,node07,node08
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/moe4rep_%j.log
# transfer-owner task 7: 4-token Qwen3-30B-A3B stream (drattr ..._s1n4t4_attr_n4m): moe mapstudy, region, experts touched.
set -uo pipefail
df -h /scratch | tail -1
A=$HOME/amcas-pro/results/fullscale/drattr/Qwen3-30B-A3B-Q4_K_M_s1n4t4_attr_n4m
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib OPENBLAS_NUM_THREADS=1 AMCAS_CAPACITY_GB=32
mkdir -p $HOME/amcas-pro/results/fullscale/moe/Q30B_n4
$E/bin/python $HOME/amcas-pro/tools/moe_experts_touched.py $A/miss.gz $A/tensors.tsv $HOME/amcas-pro/results/fullscale/moe/Q30B_n4_experts.json 2>&1 | tail -30
WIN=16000000 PL=packed,thp bash $HOME/amcas-pro/scripts/19_mapstudy.sh moe Q30B_n4 $A/miss.gz $A/tensors.tsv
bash $HOME/amcas-pro/scripts/18_region.sh Qwen3-30B-A3B-Q4_K_M_s1n4t4_attr_n4m 12000000
