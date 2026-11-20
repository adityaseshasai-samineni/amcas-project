#!/usr/bin/env python3
"""Retry the failed maximum-context MoE decode using explicit GPU prefill tiles.
B=4,P=32768, full49/49GPUoffload and selected decode shape remain fixed.
Original CPU-like2048/512 preprocessing failure is preserved, not relabeled.
"""
import json,pathlib,subprocess,sys,shutil
R=pathlib.Path.home()/'amcas-pro/results/gpu/campaign_20261120';name='q30_B4P32768';flag=R/'capacity_retry.json'
if flag.is_file():raise SystemExit(0)
p=R/'cases'/f'{name}.json';original=json.load(open(p));status=R/'runs'/name/'native.status.json'
if status.is_file() and json.load(open(status)).get('passed'):json.dump(dict(needed=False,passed=True),open(flag,'w'));raise SystemExit(0)
backup=R/'cases'/f'{name}.original.json'
if not backup.exists():shutil.copy2(p,backup)
attempts=[];passed=False
for batch,ub in [(512,512),(512,128),(128,128),(128,64)]:
 row=json.loads(json.dumps(original));row['environment'].update(AMCAS_N_BATCH=str(batch),AMCAS_UBATCH=str(ub));row['gpu_adaptation']=dict(reason='original2048/512 max-context invocation failed atcuBLASinitialization',prefill_batch=batch,prefill_ubatch=ub,selected_decode_sequences=4,prompt_tokens_per_sequence=32768,full_GPU_offload_required=True)
 json.dump(row,open(p,'w'),indent=2)
 rc=subprocess.call([sys.executable,str(R/'run_case.py'),name,'--mode','native']);d=json.load(open(status)) if status.is_file() else {};attempts.append(dict(batch=batch,ubatch=ub,exit_code=rc,status=d))
 if d.get('passed'):passed=True;break
json.dump(dict(needed=True,passed=passed,attempts=attempts,scope='GPU preprocessing tile adaptation; original failed configuration retained in.original.json and nativehistory'),open(flag,'w'),indent=2)
print('maxcontext_gpu_prefill_adaptation',passed,flush=True)
