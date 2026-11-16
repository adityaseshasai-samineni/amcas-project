#!/bin/bash
set -Eeuo pipefail
STAGE=$1
A=$HOME/amcas-pro
G=$A/results/gpu/followup_20261116/code
E=$A/tracing/mamba/envs/gem5
ROOT=$A/results/gpu/followup_20261116
mkdir -p "$ROOT" "$A/logs/gpu"
export OPENBLAS_NUM_THREADS=1 AMCAS_TOOLS=$A/tools
export LD_LIBRARY_PATH=$E/lib:$A/tracing/ramulator2-amcas
export PYTHONPATH=$G:$A/gpu:$A/tracing/ramulator2-amcas/python
export GPU_STREAM_DIRS=/tmp/$USER/gt:/scratch/$USER/gt
export TMPDIR=/tmp/$USER/gpufollow_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
SNAP=$ROOT/source_${SLURM_JOB_ID}
mkdir -p "$SNAP"
for f in gpu_followup.py gpu_moe_followup.py gpu_decode_check.cc 68_followup_job.sh 69_schedule_followup.sh 70_publish_followup.sh; do
 cp "$G/$f" "$SNAP/"
done
sha256sum "$G"/gpu_{followup,moe_followup}.py "$A/gpu"/gpu_{ds3_study,replay,io}.py "$A/gpu/gpu_filter2.c" "$A/tools/gpu_ds3_drain.cc" > "$SNAP/dependencies.sha256"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/job_${SLURM_JOB_ID}.txt"
finish(){
 rc=$?
 trap - EXIT
 printf '{"job_id":"%s","stage":"%s","exit_code":%s,"finished":"%s"}\n' "$SLURM_JOB_ID" "$STAGE" "$rc" "$(date -Is)" > "$ROOT/status_${SLURM_JOB_ID}.json"
 printf '\n### GPU follow-up %s, job %s (%s)\n\nExit code %s. Results: results/gpu/followup_20261116.\n' "$STAGE" "$SLURM_JOB_ID" "$(date -Is)" "$rc" >> "$ROOT/context.md"
 bash "$G/70_publish_followup.sh" "$STAGE job $SLURM_JOB_ID exited $rc" || echo "publish failed; results retained on cluster"
 rm -rf "$TMPDIR"
 exit "$rc"
}
trap finish EXIT
echo "stage=$STAGE job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is)"
case "$STAGE" in
 audit|compare|windows|moe-replay)
  "$E/bin/python" "$G/gpu_followup.py" "$STAGE" --root "$ROOT" --procs "$SLURM_CPUS_PER_TASK";;
 moe-smoke)
  "$E/bin/python" "$G/gpu_moe_followup.py" --root "$ROOT";;
 *) echo "unknown stage" >&2; exit 2;;
esac
