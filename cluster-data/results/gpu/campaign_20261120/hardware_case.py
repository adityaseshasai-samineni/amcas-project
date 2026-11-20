#!/usr/bin/env python3
"""Independent two-counter Nsight reference for the exact GPU case/ROI."""
import pathlib,json,os,subprocess,sys,csv,re,time
A=pathlib.Path.home()/'amcas-pro';W=A/'tracing';R=A/'results/gpu/campaign_20261120';name=sys.argv[1];case=json.load(open(R/'cases'/f'{name}.json'));O=R/'runs'/name;cap=json.load(open(O/'capture.status.json'));assert cap['passed']
env=os.environ.copy();env.update(case['environment']);v=env.get('CUDA_VISIBLE_DEVICES','').split(',')
if v and v[0]:env['CUDA_VISIBLE_DEVICES']=','.join(v[:case['gpu_devices']])
env.update(PATH='/usr/local/cuda-12.6/bin:'+str(W/'mamba/envs/gem5/bin')+':'+env['PATH'],LD_LIBRARY_PATH=':'.join(map(str,[W/'cudalibs',pathlib.Path('/usr/local/cuda-12.6/lib64'),W/'mamba/envs/gem5/lib',W/'llama.cpp/build-cuda/bin'])),GGML_CUDA_DISABLE_GRAPHS='1',CUDA_MODULE_LOADING='EAGER',NOBANNER='1')
env.pop('LD_PRELOAD',None);env.update(AMCAS_OUT=str(O/'hardware.tokens'),AMCAS_STEPS=str(O/'hardware.steps'),AMCAS_TENSORS=str(O/'hardware.tensors.tsv'),AMCAS_PROMPT_TOKENS=str(O/'hardware.prompt_tokens'))
cmd=['/opt/nvidia/nsight-compute/2024.3.2/ncu','--csv','--profile-from-start','off','--replay-mode','application','--cache-control','none','--clock-control','none','--metrics','dram__bytes_read.sum,dram__bytes_write.sum',str(R/'llama-gpu-batch')]
start=time.time()
with open(O/'hardware.raw','w') as out,open(O/'hardware.stderr','w') as err:rc=subprocess.call(cmd,env=env,stdout=out,stderr=err)
rows=[];header=None
for line in (O/'hardware.raw').read_text(errors='replace').splitlines():
 if not line.startswith('"'):continue
 x=next(csv.reader([line]))
 if x[0]=='ID':header=x;continue
 if header and len(x)==len(header):rows.append(dict(zip(header,x)))
units={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9};tot={};ids=set()
for x in rows:
 metric=x.get('Metric Name','')
 if metric not in ['dram__bytes_read.sum','dram__bytes_write.sum']:continue
 value=float(x['Metric Value'].replace(',',''))*units[x.get('Metric Unit','byte')];tot[metric]=tot.get(metric,0)+value;ids.add((x.get('Device',''),x.get('Context',''),x['ID']))
expected=sum(x.startswith('kernel-') for x in cap['kernel_lists']['kernelslist.g'])
match=all((O/f'hardware.{suffix}').is_file() and (O/f'hardware.{suffix}').read_bytes()==(O/f'native.{suffix}').read_bytes() for suffix in ['tokens','prompt_tokens'])
info=dict(passed=rc==0 and match and len(ids)==expected and len(tot)==2,exit_code=rc,kernels=len(ids),expected_kernels=expected,native_output_match=match,read_bytes=tot.get('dram__bytes_read.sum'),write_bytes=tot.get('dram__bytes_write.sum'),seconds=time.time()-start,scope='independent hardware application replay; twoDRAMbytecounters; not ordered hardwareDRAMtrace',profile_from_start=False)
json.dump(info,open(O/'hardware.status.json','w'),indent=2);print(json.dumps(info));raise SystemExit(0 if info['passed'] else 1)
