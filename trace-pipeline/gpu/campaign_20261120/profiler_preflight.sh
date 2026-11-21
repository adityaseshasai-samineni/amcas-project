#!/bin/bash
#SBATCH -J nvbit-roi-check
#SBATCH -w node04
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=6G
#SBATCH -t 00:15:00
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;F=$W/accel-sim-audit-20261117/util/tracer_nvbit
export PATH=/usr/local/cuda-12.6/bin:$W/mamba/envs/gem5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$W/mamba/envs/gem5/lib
T=$F/tracer_tool-amcas20
[ -d "$T" ] || cp -a "$F/tracer_tool" "$T"
python "$R/patch_profiler_window.py" "$T/tracer_tool.cu"
make -C "$T" ARCH=sm_70 -j4
nvcc -arch=sm_70 -O2 "$R/profiler_fixture.cu" -o "$R/profiler_fixture"
S=/scratch/$USER/campaign20-roi-${SLURM_JOB_ID};mkdir -p "$S";cd "$S";touch .amcas_trace_core
ACTIVE_FROM_START=0 TOOL_TRACE_CORE=1 TOOL_COMPRESS=1 LD_PRELOAD="$T/tracer_tool-amcas20.so" "$R/profiler_fixture" > "$R/roi_fixture.stdout" 2> "$R/roi_fixture.stderr"
"$F/tracer_tool/traces-processing/post-traces-processing" "$S/traces" -j4 > "$R/roi_fixture_post.log" 2>&1
python - "$S" "$R" <<'PY'
import pathlib,sys,json
s,r=map(pathlib.Path,sys.argv[1:]);ks=[x for x in (s/'traces/kernelslist.g').read_text().splitlines() if x.startswith('kernel-')]
assert len(ks)==1,ks
text=(s/'traces'/ks[0]).read_bytes()[:512].decode('utf-8',errors='replace');assert 'inside' in text and 'outside' not in text,text[:1000]
assert 'native_output_pass' in (r/'roi_fixture.stdout').read_text()
json.dump(dict(pass_roi=True,kernels=ks,scratch=str(s)),open(r/'roi_fixture_validation.json','w'),indent=2)
PY
bash "$A/tools/publish.sh" "Validated profiler-window tracing for GPU workload campaign"
