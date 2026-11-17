#!/bin/bash
#SBATCH --job-name=gpu-raw-counts
#SBATCH --nodelist=node04
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=01:00:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
cd "$R"
trap 'rc=$?; bash "$HOME/amcas-pro/tools/publish.sh" "Exact pre-coalescing lane-reference count audit exit=$rc"' EXIT
gcc -O2 raw_counts.c -o raw_counts
python - <<'PY'
import struct,json,subprocess
with open('raw_count_test.bin','wb') as f:
 f.write(struct.pack('<QIHHHHI',0,4,1,1,0,0xffff,0))
 for i,a in enumerate([[0x1000,0x1004,0x1008],[0x100c,0x1010],[0x1020,0x1024,0x1028],[0x2000,0x2004]]):
  f.write(struct.pack('<QIHHHHI',1,i,0,0,0,0x4004,(1<<len(a))-1));f.write(struct.pack('<'+'Q'*len(a),*a))
 f.write(struct.pack('<QIHHHHI',1,0,0,0,0,0x4104,1));f.write(struct.pack('<Q',0x101e))
subprocess.run(['./raw_counts','raw_count_test.bin','raw_count_test.json'],check=True)
a=json.load(open('raw_count_test.json'));assert a['active_lane_load_references']==10 and a['coalesced_load_32B_sectors']==4 and a['active_lane_store_references']==1 and a['coalesced_store_32B_sectors']==2 and a['launch_markers']==1,a
PY
/usr/bin/time -v ./raw_counts /scratch/$USER/gt/q7_v100_short_faauto.raw.bin raw_counts.json 2>raw_counts.time
python - <<'PY'
import json
a=json.load(open('raw_counts.json'));assert a['warp_load_instructions']+a['warp_store_instructions']==160033156,a
assert a['coalesced_load_32B_sectors']==1075696720,a
PY
