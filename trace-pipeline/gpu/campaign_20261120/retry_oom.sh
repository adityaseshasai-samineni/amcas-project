#!/bin/bash
#SBATCH -J gpu-oom-retry
#SBATCH -w node04
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 3-00:00:00
# Memory retry ladder for a two-GPU case whose NVBit capture ran out of device memory
# (q30_B4P32768: 4 x 32k f16 KV + 30B Q4 weights leave ~200 MiB on GPU0 for NVBit).
#  * every rung uses a per-case KV state cache (AMCAS_CACHE_DIR): the native gate run
#    evaluates the 4 prompts once and saves them; the NVBit capture restores them, so the
#    multi-day traced prefill is skipped. Only the decode steps run under NVBit.
#  * rungs go from the exact workload (all 48 layers on GPU) to explicit, recorded
#    deviations (1..4 leading layers kept on the CPU, allowed_offload in the case JSON).
#  * a rung failing with CUDA out-of-memory moves to the next rung; any other failure stops.
#  * the original case JSON is kept as cases/<case>.json.orig_<jobid>.
# Usage: sbatch retry_oom.sh CASE
set -Eeuo pipefail
A=$HOME/amcas-pro;R=$A/results/gpu/campaign_20261120;E=$A/tracing/mamba/envs/gem5;C=$1
trap 'bash "$A/tools/publish.sh" "GPU campaign OOM retry $C"' EXIT
O=$R/runs/$C; J=$R/cases/$C.json; S=/scratch/$USER/amcas-campaign20/$C
cp -n "$J" "$J.orig_$SLURM_JOB_ID"
export AMCAS_CAPTURE_SECONDS=230000
LOG=$O/oom_retry_$SLURM_JOB_ID.jsonl
# rung = GPU_LAYERS:TENSOR_SPLIT ("-" = llama.cpp default split). 99 = all 49 (48 layers + output).
RUNGS="99:- 99:24,25 99:23,26 48:23,25 48:22,26 47:22,25 46:22,24 45:21,24 44:21,23"
for rung in $RUNGS; do
 ngl=${rung%%:*}; ts=${rung#*:}
 "$E/bin/python" - "$J.orig_$SLURM_JOB_ID" "$J" "$ngl" "$ts" "$S/kvcache" <<'EOF'
import json,sys
src,dst,ngl,ts,cache=sys.argv[1:]
d=json.load(open(src)); e=d['environment']
e.update(AMCAS_GPU_LAYERS=ngl,AMCAS_CACHE_DIR=cache,AMCAS_DRIVER_BIN='llama-gpu-batch-v4')
if ts!='-': e['AMCAS_TENSOR_SPLIT']=ts
if int(ngl)<49: d['allowed_offload']=f'{ngl}/49'
d['oom_retry']=dict(gpu_layers=ngl,tensor_split=ts,kv_state_cache=cache,
  note='prompt KV restored from the native run state file; decode steps traced'+('' if int(ngl)>=49 else f'; {49-int(ngl)} leading layer(s) on CPU (not in GPU trace)'))
json.dump(d,open(dst,'w'),indent=2)
EOF
 mkdir -p "$S/kvcache"; rm -rf "$S/capture/traces"; rm -f "$O/capture.stdout" "$O/capture.stderr"
 echo "=== rung ngl=$ngl ts=$ts $(date)"
 rc=0; "$E/bin/python" "$R/run_case.py" "$C" --mode capture || rc=$?
 oom=$(cat "$O/native.stderr" "$O/capture.stdout" "$O/capture.stderr" 2>/dev/null | grep -ci -e "out of memory" -e "cudaMalloc failed" -e "failed to allocate" || true)
 mb=$(grep -h -e "model buffer size" -e "KV buffer size" -e "compute buffer size" -e "offloaded" "$O/native.stderr" 2>/dev/null | grep -v Host | tr -s ' ' | tr '\n' ';' | sed 's/"//g')
 echo "{\"rung\":\"$rung\",\"rc\":$rc,\"oom_lines\":$oom,\"time\":\"$(date -Is)\",\"native_buffers\":\"$mb\"}" | tee -a "$LOG"
 if [ $rc = 0 ]; then
  sbatch -J "sim-$C" -o "$O/simulation_job.log" "$R/simulate_case_node03.sh" "$C" | tee "$O/simulation_job.txt"; exit 0
 fi
 [ "$oom" -gt 0 ] || { echo "rung failed without out-of-memory; stopping (see $O/*.stderr)"; exit 1; }
done
echo "all rungs ran out of memory"; exit 1
