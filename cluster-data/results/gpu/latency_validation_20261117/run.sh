#!/bin/bash
#SBATCH --job-name=latency-v100
#SBATCH --nodelist=node04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:25:00
set -euo pipefail
cd ~/amcas-pro/results/gpu/latency_validation_20261117
nvidia-smi -q > hardware_before.txt
/usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_70 latency.cu -o latency
/usr/local/cuda-12.6/bin/cuobjdump --dump-sass latency > latency.sass
./latency all 8192 8 > samples.csv 2> native.log
NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu
for regime in l1 l2 dram; do
 for group in dram l1 l2; do
  case "$group" in
   dram) metrics=dram__bytes_read.sum,dram__bytes_write.sum;;
   l1) metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum;;
   l2) metrics=lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum;;
  esac
  "$NCU" --replay-mode application --cache-control none --clock-control none --kernel-name-base demangled --kernel-name 'regex:measure_kernel' --metrics "$metrics" --csv --log-file "ncu_${regime}_${group}.csv" ./latency "$regime" 8192 1 > "profile_${regime}_${group}.csv" 2> "profile_${regime}_${group}.log" || echo "$regime $group failed" >> counter_failures.txt
 done
done
nvidia-smi -q > hardware_after.txt
~/amcas-pro/tracing/mamba/envs/gem5/bin/python analyze.py
bash ~/amcas-pro/tools/publish.sh 'V100 native load latency calibration and held-out classification'
