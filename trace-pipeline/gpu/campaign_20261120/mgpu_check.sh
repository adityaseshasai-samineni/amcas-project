#!/bin/bash
#SBATCH -J mgpu-check
#SBATCH -w node04
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=22G
#SBATCH -t 03:00:00
# Validates the multi-GPU tracer fix (patch_tracer_mgpu.py):
#  1. rebuild tracer_tool-amcas20mgpu with ARCH=sm_70 (as the dense tracer was built)
#  2. profiler-window fixture with the patched tracer (only the ROI kernel traced)
#  3. real MoE capture q30_short on two V100s (segfaulted at the first traced kernel before)
#  4. both CUDA contexts must have traced kernels; run_case.py output/token gate must pass
set -Eeuo pipefail
A=$HOME/amcas-pro;W=$A/tracing;R=$A/results/gpu/campaign_20261120;F=$W/accel-sim-audit-20261117/util/tracer_nvbit
T=$F/tracer_tool-amcas20mgpu
trap 'bash "$A/tools/publish.sh" "GPU campaign multi-GPU tracer fix check"' EXIT
export PATH=/usr/local/cuda-12.6/bin:$W/mamba/envs/gem5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$W/mamba/envs/gem5/lib
python -c "import ast;ast.parse(open('$R/run_case.py').read())"
make -C "$T" clean >/dev/null; make -C "$T" ARCH=sm_70 -j4 2>&1 | tail -2
sha256sum "$T/tracer_tool-amcas20mgpu.so" "$F/tracer_tool-amcas20/tracer_tool-amcas20.so" | tee "$R/mgpu_tracer_sha256.txt"

# 2. profiler-window fixture (same as profiler_preflight.sh, patched tracer)
S=/scratch/$USER/campaign20-roi-mgpu-${SLURM_JOB_ID};mkdir -p "$S";cd "$S";touch .amcas_trace_core
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} ACTIVE_FROM_START=0 TOOL_TRACE_CORE=1 TOOL_COMPRESS=1 LD_PRELOAD="$T/tracer_tool-amcas20mgpu.so" "$R/profiler_fixture" > "$R/roi_fixture_mgpu.stdout" 2> "$R/roi_fixture_mgpu.stderr"
"$F/tracer_tool/traces-processing/post-traces-processing" "$S/traces" -j4 > "$R/roi_fixture_mgpu_post.log" 2>&1
python - "$S" "$R" <<'PY'
import pathlib,sys,json
s,r=map(pathlib.Path,sys.argv[1:]);ks=[x for x in (s/'traces/kernelslist.g').read_text().splitlines() if x.startswith('kernel-')]
assert len(ks)==1,ks
text=(s/'traces'/ks[0]).read_bytes()[:512].decode('utf-8',errors='replace');assert 'inside' in text and 'outside' not in text,text[:1000]
assert 'native_output_pass' in (r/'roi_fixture_mgpu.stdout').read_text()
json.dump(dict(pass_roi=True,kernels=ks,scratch=str(s),tracer='tracer_tool-amcas20mgpu'),open(r/'roi_fixture_validation_mgpu.json','w'),indent=2)
print('ROI fixture with mgpu tracer: PASS')
PY

# 3. real two-GPU MoE capture
C=q30_short;CS=/scratch/$USER/amcas-campaign20/$C/capture
if [ -d "$CS" ]; then mv "$CS" "$CS.failed_before_mgpu_fix_$(date +%Y%m%d%H%M)"; fi
cd "$R"
AMCAS_CAPTURE_SECONDS=9000 python "$R/run_case.py" "$C" --mode capture || true
python - "$R" "$C" <<'PY'
import json,pathlib,sys,re,collections
r,c=pathlib.Path(sys.argv[1]),sys.argv[2]
st=json.load(open(r/'runs'/c/'capture.status.json'))
print('capture status:',{k:st.get(k) for k in ('passed','exit_code','stop_reason','seconds')})
t=pathlib.Path(st['scratch'])/'traces'
per=collections.Counter()
g=t/'kernelslist.g'
if g.is_file():
    for x in g.read_text().splitlines():
        m=re.search(r'ctx_0x[0-9a-fA-F]+',x)
        if x.startswith('kernel-') and m: per[m.group(0)]+=1
else:
    for f in t.glob('kernelslist_ctx_0x*'):
        per[f.name]=sum(1 for x in f.read_text().splitlines() if x.startswith('kernel-'))
print('traced kernels per context:',dict(per))
ok=st['passed'] and len([v for v in per.values() if v>0])==2
json.dump(dict(passed=ok,capture=st,kernels_per_context=dict(per)),open(r/'runs'/c/'mgpu_check.json','w'),indent=2)
print('MGPU CHECK:','PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
PY
