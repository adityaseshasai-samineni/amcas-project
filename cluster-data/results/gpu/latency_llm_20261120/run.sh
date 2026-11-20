#!/bin/bash
#SBATCH --job-name=nvbit-latency
#SBATCH --nodelist=node04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:12:00
set -euo pipefail
cd ~/amcas-pro/results/gpu/latency_llm_20261120
bash build.sh > build.log 2>&1
cp ../latency_validation_20261117/latency.cu fixture.cu
/usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_70 fixture.cu -o fixture
nvidia-smi -q > hardware.txt
# The unfamiliar injection path must pass a bounded small input before full measurement.
timeout 60 env PROBE_KERNEL=measure_kernel PROBE_OUT=preflight_observer.csv LD_PRELOAD="$PWD/latency_probe.so" ./fixture l1 32 1 > preflight_native.csv 2> preflight.log
for c in l1 l2 dram; do
 /usr/bin/time -f '%e' -o "native_${c}.seconds" ./fixture "$c" 1024 8 > "native_${c}.csv" 2> "native_${c}.log"
 /usr/bin/time -f '%e' -o "observer_${c}.seconds" env PROBE_KERNEL=measure_kernel PROBE_OUT="observer_${c}.csv" LD_PRELOAD="$PWD/latency_probe.so" ./fixture "$c" 1024 8 > "instrumented_${c}.csv" 2> "instrumented_${c}.log"
done
bash ~/amcas-pro/tools/publish.sh 'NVBit paired load-latency observer calibration raw experiment'
