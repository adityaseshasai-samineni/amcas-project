#!/bin/bash
#SBATCH --job-name=cache-audit
#SBATCH --nodelist=node04
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -Eeuo pipefail
R=$HOME/amcas-pro/results/gpu/cache_audit_20261117
cd "$R"
trap 'rc=$?; echo "$rc" > exit_code.txt; bash "$HOME/amcas-pro/tools/publish.sh" "GPU cache audit pilot exit=$rc"' EXIT
export PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/bin:$PATH
gcc -O2 gpu_filter_boundary.c -o filter
python - <<'PY'
import struct,subprocess,csv
# Same sector read by same CTA across dependent kernels must re-access L2 after invalidation.
with open('test.bin','wb') as f:
 for launch in [1,2]:
  f.write(struct.pack('<QIHHHHI',launch,0,0,0,0,0x4004,1));f.write(struct.pack('<Q',4096))
subprocess.run(['./filter','test','test','--var','old:hash=113','--var','boundary:hash=113,invalidate=1'],check=True)
r=list(csv.DictReader(open('test.perk.tsv'),delimiter='\t'))
assert sum(int(x['old_l2rd']) for x in r)==1
assert sum(int(x['boundary_l2rd']) for x in r)==2
assert sum(int(x['boundary_miss']) for x in r)==1
open('test_pass.txt','w').write('PASS: cross-kernel L1 invalidation causes L2 recheck without spurious DRAM read\n')
PY
ln -sf /scratch/$USER/gt/q7_v100_short_faauto.raw.bin input.bin
ln -sf /scratch/$USER/gt/q7_v100_short_faauto.launch.txt input.launch.txt
ln -sf /scratch/$USER/gt/q7_v100_short_faauto.opcodes.txt input.opcodes.txt
/usr/bin/time -v ./filter input pilot --var original:hash=113 --var boundary:hash=113,invalidate=1 --var bypass:l1kb=0,hash=113 > pilot.stdout 2> pilot.stderr
python - <<'PY'
import csv,json
r=list(csv.DictReader(open('pilot.perk.tsv'),delimiter='\t'))
s={k:sum(int(x[k]) for x in r) for k in r[0] if k!='launch'}
assert s['original_miss']==137213135 and s['original_wr']==219930,s
json.dump({'counts':s,'baseline_reproduced':True,'status':'sensitivity experiment; not fully validated'},open('summary.json','w'),indent=2)
PY
