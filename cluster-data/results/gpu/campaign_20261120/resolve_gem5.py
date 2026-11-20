#!/usr/bin/env python3
import pathlib,json,hashlib
A=pathlib.Path.home()/'amcas-pro';R=A/'results/gpu/campaign_20261120';rows=json.load(open(R/'gem5_cpu_runs.json'));targets={};coverage=[]
def cmds(x):
 if isinstance(x,dict):
  if 'cmd' in x:yield x['cmd']
  for v in x.values():yield from cmds(v)
 elif isinstance(x,list):
  for v in x:yield from cmds(v)
for r in rows:
 root=A/'results/traces'/r['run'];cfg=json.load(open(root/'config.json'));cmd=next(cmds(cfg));arg=lambda x:cmd[cmd.index(x)+1]
 phases=json.load(open(root/'summary.json'))['phases'];ids=[x['id'] for x in phases];model=pathlib.Path(arg('-m')).name;ctx=int(arg('-c'));num=int(arg('-n'));prompt=arg('-p') if '-p' in cmd else None;pfile=arg('-f') if '-f' in cmd else None
 base='g5_'+('s15' if model=='stories15M.gguf' else 'q05' if model.startswith('qwen25') else 'tinymoe4')+'_'+str(r['ctx'])
 slices=[]
 if 0 in ids:slices.append((base+'_prefill','prefill',0,1))
 dec=[i for i in ids if i>0]
 if dec:slices.append((base+'_decode','decode',min(dec),max(dec)-min(dec)+1))
 mapped=[]
 for name,phase,first,n in slices:
  mapped.append(name)
  if name in targets:continue
  env=dict(AMCAS_MODEL=str(A/'tracing/models'/model),AMCAS_B='1',AMCAS_P='512',AMCAS_THREADS='1',AMCAS_NGEN=str(num-1),AMCAS_SKIP_STEPS=str(first),AMCAS_TRACE_STEPS=str(n),AMCAS_GPU_LAYERS='99',AMCAS_FA='-1',AMCAS_CORPUS=str(A/'results/sessions/corpus.txt'),AMCAS_CACHE_DIR='',AMCAS_UBATCH='512',AMCAS_N_BATCH='512',AMCAS_CTX=str(ctx),AMCAS_CAPTURE_SECONDS='9000')
  if prompt:env['AMCAS_PROMPT_TEXT']=prompt
  else:env['AMCAS_LITERAL_FILE']=pfile
  if phase=='prefill':env['AMCAS_PREFILL']='1'
  target=dict(cpu_workload=name,era='gem5',source_cpu_run=r['run'],source_cpu_phase_ids=ids,selected_phase=phase,cpu_generation_limit=num,cpu_source=str(root),gpu_devices=1,input_present=bool(prompt) or pathlib.Path(pfile).is_file(),environment=env,scope='GPU workload translation; original CPUcache/prefetch parameters not GPUhardware settings; split prefill/decode windows cold-modeled independently')
  if prompt:target['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
  else:target['prompt_sha256']=hashlib.sha256(pathlib.Path(pfile).read_bytes()).hexdigest()
  targets[name]=target;json.dump(target,open(R/'cases'/f'{name}.json','w'),indent=2)
 coverage.append(dict(cpu_run=r['run'],model=r['model'],cpu_llc=r['llc'],cpu_prefetch=r['pf'],cpu_context=r['ctx'],cpu_phase_ids=ids,gpu_targets=mapped,cache_knob_translation='not applicable: realV100cachegeometry and cacheoperators replace CPU2/22MiBLLC and stride-prefetch choices; duplicate workloads share GPUtargets'))
json.dump(dict(cpu_rows=coverage,gpu_targets=list(targets.values()),count=len(targets),scope='Earlier gem5era supplement; main20fullscale cases are separate'),open(R/'coverage_gem5.json','w'),indent=2)
(R/'legacy_cases.txt').write_text('\n'.join(targets)+'\n')
