#!/bin/bash
#SBATCH --job-name=accel-q7sim
#SBATCH --nodelist=node04
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
A=$HOME/amcas-pro/tracing/accel-sim-audit-20261117
S=/scratch/$USER/accel-q7short-20261117
export LD_LIBRARY_PATH=$HOME/amcas-pro/tracing/accel-build-deps/lib:$HOME/amcas-pro/tracing/mamba/envs/gem5/lib
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
trap 'rc=$?; echo "$rc" > "$R/simulation_exit.txt"; bash "$HOME/amcas-pro/tools/publish.sh" "Accel-Sim Qwen short replay exit=$rc"' EXIT
python "$R/census.py" > "$R/census.stdout"
# Counter validation uses a nonzero-read known-output capture.
mkdir -p "$R/preflight_export";cd "$R/preflight_export"
cp "$R/preflight/"*.config "$R/preflight/"*.icnt "$R/preflight/"*.xml .
AMCAS_DRAM_EXPORT="$PWD/dram.csv" "$A/gpu-simulator/build-cmake/accel-sim.out" -config gpgpusim.config -config trace.config -trace "$R/preflight_read/traces/kernelslist.g" > sim.log 2>&1
python - <<'PY'
import csv,re,json
r=list(csv.DictReader(open('dram.csv')));s=open('sim.log').read();nums=re.findall(r'(?m)^n_req = (\d+)\s*$',s)
assert nums,'controller n_req missing'
# One kernel: one complete controller stats block.
assert len(r)>0 and sum(map(int,nums))==len(r),(nums,len(r))
json.dump({'pass':True,'controller_requests':len(r)},open('validation.json','w'),indent=2)
PY
python - <<'PY'
import pathlib
p=pathlib.Path('/scratch')/pathlib.Path.home().name/'accel-q7short-20261117/traces/kernelslist.g'
r=[s for s in p.read_text().splitlines() if s.startswith('kernel-')]
assert len(r)==538,(len(r),'unexpected decode kernel count')
assert r[0].startswith('kernel-1296-') and r[-1].startswith('kernel-1833-'),(r[0],r[-1])
PY
mkdir -p "$S/sim";cd "$S/sim"
cp "$R/preflight/"*.config "$R/preflight/"*.icnt "$R/preflight/"*.xml .
export AMCAS_DRAM_EXPORT="$S/dram.csv"
/usr/bin/time -v "$A/gpu-simulator/build-cmake/accel-sim.out" -config gpgpusim.config -config trace.config -trace "$S/traces/kernelslist.g" > "$R/simulation.log" 2> "$R/simulation.time"
gzip -1 "$S/dram.csv"
