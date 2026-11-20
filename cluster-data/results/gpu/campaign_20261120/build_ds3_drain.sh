#!/bin/bash
#SBATCH -J ds3-real-drain
#SBATCH -w node03
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 00:15:00
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;E=$W/mamba/envs/gem5;D=$W/DRAMsim3-gpu-drain
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib
[ -d "$D" ] || cp -a "$W/DRAMsim3" "$D"
python "$R/patch_ds3_drain.py" "$D"
cmake -S "$D" -B "$D/build-amcas" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$D/build-amcas" --target dramsim3 -j4
g++ -O3 -std=c++14 -I"$D/src" -I"$D/ext/headers" -I"$D/ext/fmt/include" "$R/ds3_demand_drain.cc" -L"$D" -ldramsim3 -Wl,-rpath,"$D" -Wl,-rpath,"$E/lib" -o "$R/ds3_demand_drain"
g++ -O3 -std=c++14 -I"$D/src" -I"$D/ext/headers" -I"$D/ext/fmt/include" "$R/ds3_decode_check.cc" -L"$D" -ldramsim3 -Wl,-rpath,"$D" -Wl,-rpath,"$E/lib" -o "$R/ds3_decode_check"
