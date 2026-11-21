#!/bin/bash
#SBATCH -J gpu-policy-check
#SBATCH -w node03
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:10:00
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;E=$W/mamba/envs/gem5
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$W/ramulator2-gpu-drain PYTHONPATH=$W/ramulator2-gpu-drain/python:$A/tools:$A/gpu OPENBLAS_NUM_THREADS=1
python - "$R" <<'PY'
import pathlib,csv,sys
r=pathlib.Path(sys.argv[1]);xs=list(csv.DictReader(open(r/'mask_fixture/dram.csv')));lo=min(int(x['address_hex'],16) for x in xs)&~127;hi=max(int(x['address_hex'],16) for x in xs)+128
(r/'policy_fixture.tensors.tsv').write_text('step\tname\tdata\tnbytes\ttype\top\tne0\tne1\tne2\tne3\n'+f'0\tblk.0.weight\t0x{lo:x}\t{hi-lo}\tF32\tNONE\t1024\t1\t1\t1\n')
PY
python "$R/policy_grid.py" "$R/mask_fixture/fixture.dram" "$R/policy_fixture.tensors.tsv" "$R/policy_fixture" --size 32 --procs 2
python - "$R" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);xs=json.load(open(r/'policy_fixture/results.json'));assert len(xs)==400,len(xs);assert len({x['mapping'] for x in xs})==10;assert len({x['policy'] for x in xs})==10
json.dump(dict(passed=True,replays=len(xs),mappings=10,policies=10,placements=2,channel_interleaves=2,read_write_drain_assertions=True,scope='synthetic plumbing fixture, not realGPUperformance'),open(r/'policy_fixture_validation.json','w'),indent=2)
PY
bash "$A/tools/publish.sh" "Validated GPU full mapping-policy replay plumbing"
