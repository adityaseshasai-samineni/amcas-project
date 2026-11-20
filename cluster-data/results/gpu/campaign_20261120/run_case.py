#!/usr/bin/env python3
"""Native/official-SASS workload execution with bounded resources and token gates."""
import argparse,json,os,pathlib,subprocess,time,signal,shutil,re,hashlib,sys
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--mode',choices=['native','capture'],required=True);a=p.parse_args()
A=pathlib.Path.home()/'amcas-pro';R=A/'results/gpu/campaign_20261120';case=json.load(open(R/'cases'/f'{a.case}.json'));out=R/'runs'/a.case;out.mkdir(parents=True,exist_ok=True)
driver=R/('llama-gpu-batch-v2' if (R/'llama-gpu-batch-v2').is_file() else 'llama-gpu-batch');driver_hash=hashlib.sha256(driver.read_bytes()).hexdigest();environment_hash=hashlib.sha256(json.dumps(case['environment'],sort_keys=True).encode()).hexdigest()
if a.mode=='capture' and case['gpu_devices']==2 and not (R/'capacity_retry.json').is_file():
 subprocess.call([sys.executable,str(R/'repair_capacity.py')])
if a.mode=='capture':
 old=json.load(open(out/'native.status.json'))
 if old.get('driver_sha256')!=driver_hash or old.get('environment_sha256')!=environment_hash:
  subprocess.check_call([sys.executable,__file__,a.case,'--mode','native'])
if a.mode=='native' and (out/'native.status.json').is_file():
 previous=json.load(open(out/'native.status.json'));tag=previous.get('driver_sha256','legacy131044')[:8]+'_'+previous.get('environment_sha256','legacy')[:8]
 archive=out/('native_history_'+tag);archive.mkdir(exist_ok=True)
 for f in out.glob('native.*'):
  if f.is_file() and not (archive/f.name).exists():shutil.copy2(f,archive/f.name)

env=os.environ.copy();env.update(case['environment']);
visible=env.get('CUDA_VISIBLE_DEVICES','').split(',');
if visible and visible[0]:env['CUDA_VISIBLE_DEVICES']=','.join(visible[:case['gpu_devices']])
W=A/'tracing';env['PATH']=str(W/'mamba/envs/gem5/bin')+':/usr/local/cuda-12.6/bin:'+env['PATH'];env['LD_LIBRARY_PATH']=':'.join(map(str,[W/'cudalibs',pathlib.Path('/usr/local/cuda-12.6/lib64'),W/'mamba/envs/gem5/lib',W/'llama.cpp/build-cuda/bin']));env.update(GGML_CUDA_DISABLE_GRAPHS='1',CUDA_MODULE_LOADING='EAGER',NOBANNER='1')
if not case['input_present']:raise SystemExit('CPU input missing: '+a.case)
if a.mode=='capture' and not json.load(open(out/'native.status.json'))['passed']:raise SystemExit('native fit/output gate failed')
S=pathlib.Path('/scratch')/pathlib.Path.home().name/'amcas-campaign20'/a.case/a.mode
S.mkdir(parents=True,exist_ok=True);(S/'.amcas_trace_core').touch()
env.update(AMCAS_OUT=str(out/(a.mode+'.tokens')),AMCAS_STEPS=str(out/(a.mode+'.steps')),AMCAS_TENSORS=str(out/(a.mode+'.tensors.tsv')),AMCAS_PROMPT_TOKENS=str(out/(a.mode+'.prompt_tokens')))
if a.mode=='capture':
 assert json.load(open(R/'roi_fixture_validation.json'))['pass_roi']
 if shutil.disk_usage(S).free<90*1024**3:raise SystemExit('scratch free below90GiB; old results preserved')
 env.update(ACTIVE_FROM_START='0',TOOL_TRACE_CORE='1',TOOL_COMPRESS='1',LD_PRELOAD=str(W/'accel-sim-audit-20261117/util/tracer_nvbit/tracer_tool-amcas20/tracer_tool-amcas20.so'))
start=time.time();reason=None;peak_rss_kib=0
with open(out/(a.mode+'.stdout'),'w') as so,open(out/(a.mode+'.stderr'),'w') as se:
 proc=subprocess.Popen([str(driver)],cwd=S,env=env,stdout=so,stderr=se,start_new_session=True)
 while proc.poll() is None:
  time.sleep(5)
  try:
   stat=pathlib.Path(f'/proc/{proc.pid}/status').read_text();m=re.search(r'^VmRSS:\s+(\d+)',stat,re.M)
   if m:peak_rss_kib=max(peak_rss_kib,int(m[1]))
  except FileNotFoundError:pass
  if time.time()-start>(1800 if a.mode=='native' else 6*3600):reason='runtime_limit'
  if a.mode=='capture':
   size=sum(f.stat().st_size for f in S.rglob('*') if f.is_file())
   if size>80*1024**3 or shutil.disk_usage(S).free<25*1024**3:reason='scratch_trace_budget'
  if reason:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   break
rc=proc.returncode;status=dict(environment_sha256=environment_hash,peak_rss_kib=peak_rss_kib,driver=str(driver),driver_sha256=driver_hash,passed=rc==0,exit_code=rc,stop_reason=reason,seconds=time.time()-start,scratch=str(S),gpu_devices=case['gpu_devices'],scope='selected graph windows; native output and official trace are separate executions')
if rc==0:
 log=(out/(a.mode+'.stderr')).read_text(errors='replace')
 off=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',log)
 status['offload_layers']=off
 if not off or any(int(x)!=int(y) for x,y in off):status.update(passed=False,stop_reason='full_GPU_offload_not_confirmed')
 status['cuda_visible_devices']=env.get('CUDA_VISIBLE_DEVICES')
if rc==0 and a.mode=='capture':
 for suffix in ['tokens','prompt_tokens']:
  if (out/f'native.{suffix}').read_bytes()!=(out/f'capture.{suffix}').read_bytes():status.update(passed=False,stop_reason='native_capture_'+suffix+'_mismatch')
if status['passed'] and a.mode=='capture':
 tool=W/'accel-sim-audit-20261117/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing'
 env.pop('LD_PRELOAD',None)
 with open(out/'post.log','w') as log:rc=subprocess.call([str(tool),str(S/'traces'),'-j','4'],cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT)
 status['postprocess_exit']=rc;status['passed']=rc==0
 if rc==0:
  lists=list((S/'traces').glob('kernelslist*.g'));status['kernel_lists']={x.name:x.read_text().splitlines() for x in lists}
  status['trace_bytes']=sum(f.stat().st_size for f in (S/'traces').rglob('*') if f.is_file())
json.dump(status,open(out/(a.mode+'.status.json'),'w'),indent=2)
print(json.dumps(status),flush=True)
raise SystemExit(0 if status['passed'] else 1)
