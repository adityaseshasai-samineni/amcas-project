#!/bin/bash
#SBATCH -J gpu-expert-check
#SBATCH -w node03
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;E=$W/mamba/envs/gem5
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$W/ramulator2-gpu-drain PYTHONPATH=$W/ramulator2-gpu-drain/python:$A/tools:$A/gpu OPENBLAS_NUM_THREADS=1
python - "$R" <<'PY'
import pathlib,sys
r=pathlib.Path(sys.argv[1]);s=(r/'policy_fixture.tensors.tsv').read_text().replace('blk.0.weight','blk.0.ffn_gate_exps.weight').replace('\t1024\t1\t1\t1\n','\t1024\t1\t128\t1\n');(r/'expert_fixture.tensors.tsv').write_text(s)
PY
python "$R/policy_grid.py" "$R/mask_fixture/fixture.dram" "$R/expert_fixture.tensors.tsv" "$R/expert_fixture" --size 32 --procs 2 --smoke
python - "$R" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);xs=json.load(open(r/'expert_fixture/results.json'));assert len(xs)==120,len(xs);assert len({x['mapping'] for x in xs})==12
json.dump(dict(passed=True,replays=120,mappings=12,policies=10,scope='synthetic expert-slice plumbing; not measuredGPUperformance'),open(r/'expert_fixture_validation.json','w'),indent=2)
PY
bash "$A/tools/publish.sh" "Validated GPU expert-aware mappings with drained read/write fixture"
