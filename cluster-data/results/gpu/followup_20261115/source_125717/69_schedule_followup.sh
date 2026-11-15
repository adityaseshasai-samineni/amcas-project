#!/bin/bash
set -Eeuo pipefail
A=$HOME/amcas-pro
G=$A/gpu
ROOT=$A/results/gpu/followup_20261115
mkdir -p "$ROOT" "$A/logs/gpu"
[ ! -e "$ROOT/submissions.json" ] || { echo "already scheduled; inspect submissions.json" >&2; exit 1; }
mkdir -p "$ROOT/code"
for f in gpu_followup.py gpu_moe_followup.py gpu_decode_check.cc 68_followup_job.sh 69_schedule_followup.sh 70_publish_followup.sh; do
 cp "$G/$f" "$ROOT/code/"
done
submit(){
 stage=$1; shift
 sbatch --parsable -J "gf_$stage" -o "$A/logs/gpu/followup_${stage}_%j.log" "$@" "$G/68_followup_job.sh" "$stage"
}
J1=$(submit audit -w node05 -c 4 --mem=16G -t 01:00:00)
J2=$(submit compare -w node05 -c 8 --mem=64G -t 1-00:00:00 --dependency=afterok:$J1 --kill-on-invalid-dep=yes)
J3=$(submit windows -w node05 -c 8 --mem=64G -t 1-00:00:00 --dependency=afterok:$J2 --kill-on-invalid-dep=yes)
J4=$(submit moe-smoke -w node04 --gres=gpu:2 -c 4 --mem=48G -t 02:00:00 --dependency=afterok:$J1 --kill-on-invalid-dep=yes)
J5=$(submit moe-replay -w node04 --gres=gpu:2 -c 8 --mem=64G -t 1-00:00:00 --dependency=afterok:$J2:$J4 --kill-on-invalid-dep=yes)
python3 - "$ROOT/submissions.json" "$J1" "$J2" "$J3" "$J4" "$J5" <<'PY'
import json,sys,datetime
names=['audit','compare','windows','moe-smoke','moe-replay']
data=dict(created=datetime.datetime.now().isoformat(),jobs=dict(zip(names,sys.argv[2:])),
 dependencies={'compare':['audit'],'windows':['compare'],'moe-smoke':['audit'],'moe-replay':['compare','moe-smoke']})
json.dump(data,open(sys.argv[1],'w'),indent=2)
print(json.dumps(data,indent=2))
PY
bash "$G/70_publish_followup.sh" "scheduled audit=$J1 compare=$J2 windows=$J3 moe-smoke=$J4 moe-replay=$J5"
squeue -u "$USER" -o "%.9i %.22j %.10T %.10M %.12l %.8N %R"
