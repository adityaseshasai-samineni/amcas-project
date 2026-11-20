#!/bin/bash
#SBATCH -J mask-fixture
#SBATCH -w node03
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;OLD=$A/results/gpu/cache_audit_20261117
export PATH=$W/mamba/envs/gem5/bin:$PATH LD_LIBRARY_PATH=$W/mamba/envs/gem5/lib:$W/accel-build-deps/lib
D=$R/mask_fixture;mkdir -p "$D";cd "$D"
cp "$OLD/preflight_write/"*.config "$OLD/preflight_write/"*.icnt "$OLD/preflight_write/"*.xml .
AMCAS_DRAM_EXPORT="$D/dram.csv" AMCAS_DRAM_EXPORT_MASKS=1 "$W/accel-sim-audit-20261117/gpu-simulator/build-cmake/accel-sim.out" -config gpgpusim.config -config trace.config -trace "$OLD/preflight_write/traces/kernelslist.g" > simulation.log 2>&1
g++ -O3 "$R/export_masked_sectors.cc" -lz -o "$R/export_masked_sectors"
gzip -c dram.csv > dram.csv.gz
"$R/export_masked_sectors" dram.csv.gz fixture.dram > conversion.json
python - <<'PY'
import csv,json,re,struct,gzip,subprocess,pathlib
rows=list(csv.DictReader(open('dram.csv')));assert len(rows)==32 and sum(int(x['write']) for x in rows)==16
assert all(int(x['sector_mask']).bit_count()==int(x['bytes'])//32 for x in rows)
nums=re.findall(r'(?m)^n_req = (\d+)\s*$',open('simulation.log').read());assert sum(map(int,nums))==len(rows)
with gzip.open('noncontiguous.csv.gz','wt') as f:f.write('sequence,core_cycle,partition,address_hex,write,bytes,access_type,sector_mask\n0,0,0,0x1000,1,64,9,5\n')
subprocess.check_call(['../export_masked_sectors','noncontiguous.csv.gz','noncontiguous.dram'])
v=struct.unpack('<QQ',pathlib.Path('noncontiguous.dram').read_bytes());assert v==(((0x1000//32)<<1)|1,((0x1040//32)<<1)|1),v
json.dump(dict(passed=True,reads=16,writes=16,controller_packets=32,noncontiguous_mask_test=True,scope='test-only tiny write-throughL2; notV100 workload'),open('validation.json','w'),indent=2)
PY
bash "$A/tools/publish.sh" "Validated masked controller-export and noncontiguous write-sector conversion"
