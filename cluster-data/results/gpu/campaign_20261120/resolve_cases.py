#!/usr/bin/env python3
"""Resolve CPU provenance on cluster; never infer missing CPU evidence as completed."""
import json,pathlib,re,hashlib
A=pathlib.Path.home()/'amcas-pro';R=A/'results/gpu/campaign_20261120';d=json.load(open(R/'coverage.json'));D=A/'results/fullscale/drattr'
q7='qwen2.5-7b-instruct-q4_k_m-00001-of-00002';q30='Qwen3-30B-A3B-Q4_K_M';q05='qwen25-05b-q4_0'
for row in d['workloads']:
 name=row['cpu_workload'];m=q30 if name.startswith('q30') else q05 if name.startswith('q05') else q7
 e=dict(AMCAS_MODEL=str(A/'tracing/models'/row['model_file']),AMCAS_B=str(row.get('sequences',1)),AMCAS_P=str(row.get('prompt_tokens_per_sequence',row.get('prompt_tokens',512))),AMCAS_THREADS=str(row.get('cpu_threads',4)),AMCAS_NGEN=str(max(3,row.get('selected_decode_steps',1))),AMCAS_SKIP_STEPS='2' if name.endswith('_t2') else '1',AMCAS_TRACE_STEPS=str(row.get('selected_decode_steps',1)),AMCAS_GPU_LAYERS='99',AMCAS_FA='-1',AMCAS_CORPUS=str(A/'results/sessions/corpus.txt'),AMCAS_CACHE_DIR='',AMCAS_UBATCH=str(row.get('ubatch',512)))
 if row['phase']=='prefill_graph_window':
  e['AMCAS_PREFILL']='1'
  if name.endswith('_pf32'):src=D/(m+'_s0n1t4_attr_pf32');e['AMCAS_LITERAL_FILE']=str(src/'prompt.txt')
  else:src=A/'results/fullscale/prefill'/(m+'_prefill_P512u32c15n1t4');e['AMCAS_TRACE_COMPUTE']='15'
 elif row['phase']=='batched_decode':src=A/'results/fullscale/drbatch'
 else:
  tag=row['context_label'];skip=2 if name.endswith('_t2') else 1; nt=4 if name.endswith('_n4') else 1
  tail='_c32k_t2' if name.endswith('_t2') else '_c'+tag if tag!='short' else '_n4m' if name.endswith('_n4') else '_miss'
  src=D/(m+f'_s{skip}n{nt}t'+e['AMCAS_THREADS']+'_attr'+tail)
  if tag!='short':
   e['AMCAS_LITERAL_FILE']=str(src/'prompt.txt')
   if not pathlib.Path(e['AMCAS_LITERAL_FILE']).is_file():
    # Historical7B32k early stream has identical saved prefix; record actual fallback.
    candidates=list(D.glob(m+'*_c'+tag+'*/prompt.txt'))
    if candidates:e['AMCAS_LITERAL_FILE']=str(sorted(candidates)[0]);src=pathlib.Path(e['AMCAS_LITERAL_FILE']).parent
  else:e['AMCAS_PROMPT_TEXT']='Once upon a time there was a little girl named Lily.'
  log=src/'plain.err'
  if log.is_file():
   match=re.search(r'generate: n_ctx = (\d+)',log.read_text(errors='replace'))
   if match:e['AMCAS_CTX']=match[1]
 row.update(environment=e,cpu_source=str(src),gpu_devices=2 if name.startswith('q30') else 1,cpu_grid_result_present=(A/'results/fullscale/grid'/name/'replay.json').is_file())
 if 'AMCAS_LITERAL_FILE' in e:
  p=pathlib.Path(e['AMCAS_LITERAL_FILE']);row['input_present']=p.is_file()
  if p.is_file():row['prompt_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 else:row['input_present']=True
 (R/'cases').mkdir(exist_ok=True);json.dump(row,open(R/'cases'/f'{name}.json','w'),indent=2)
json.dump(d,open(R/'coverage_resolved.json','w'),indent=2)
