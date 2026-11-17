#!/bin/bash
#SBATCH -J accel-convert
#SBATCH -w node04
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:30:00
set -Eeuo pipefail
A=$HOME/amcas-pro
R=$A/results/gpu/accelsim_campaign_20261117
D=/scratch/$USER/accel-q7short-20261117
trap 'bash "$A/tools/publish.sh" "Accel-Sim canonical sector conversion validation"' EXIT
export PATH=$A/tracing/mamba/envs/gem5/bin:$PATH
g++ -O3 "$A/results/gpu/cache_audit_20261117/export_sectors.cc" -lz -o "$R/export_sectors"
"$R/export_sectors" "$D/dram.csv.gz" "$D/canonical.dram" > "$R/conversion.json"
stat -c %s "$D/canonical.dram" > "$R/conversion_binary_bytes.txt"
