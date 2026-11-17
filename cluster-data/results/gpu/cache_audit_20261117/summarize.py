from pathlib import Path
import csv,re,json,gzip,hashlib
r=Path.home()/'amcas-pro/results/gpu/cache_audit_20261117';s=Path('/scratch')/Path.home().name/'accel-q7short-20261117'
p=s/'dram.csv.gz'; opener=gzip.open if p.exists() else open
if not p.exists():p=s/'dram.csv'
counts={'read_requests':0,'write_requests':0,'read_bytes':0,'write_bytes':0};types={};sizes={};unaligned=0;last=-1
with opener(p,'rt') as f:
 for x in csv.DictReader(f):
  assert int(x['sequence'])==last+1;last+=1
  k='write' if int(x['write']) else 'read';counts[k+'_requests']+=1;counts[k+'_bytes']+=int(x['bytes']);types[x['access_type']]=types.get(x['access_type'],0)+1; sk=k+'_'+x['bytes'];sizes[sk]=sizes.get(sk,0)+1;unaligned+=int(x['address_hex'],16)%32!=0
log=(r/'simulation.log').read_text(errors='replace')
(r/'simulation_final_stats.log').write_text(log[-1000000:])
nreq=[int(x) for x in re.findall(r'(?m)^n_req = (\d+)\s*$',log)]
assert len(nreq)>=32
controller=sum(nreq[-32:]);assert controller==last+1,(controller,last+1)
assert 'simulation thread exiting' in log
launched=len(re.findall(r'(?m)^launching kernel name:',log));assert launched==538,launched
uids=re.findall(r'(?m)^kernel_launch_uid = (.*?)\s*$',log);assert uids and int(uids[-1].split()[-1])==538,uids[-1:]
stdout=(r/'capture.stdout').read_text(errors='replace');prompt='The capital of France is';stdout=stdout[stdout.index(prompt):]
actual=re.sub(r'Writing results to [^\n]*\n','',stdout).strip();assert actual=='The capital of France is Paris. The',repr(actual)
stderr=(r/'capture.stderr').read_text();pf=int(re.search(r'prompt eval time.*?/\s*(\d+) tokens',stderr).group(1));sampled=int(re.search(r'samplers time.*?/\s*(\d+) tokens',stderr).group(1));assert pf==5 and sampled-pf==3,(pf,sampled)

out={'validated_export_counts':True,'kernels_completed':538,'captured_native_output':actual,'response_text':' Paris. The','generated_tokens':sampled-pf,'controller_requests':controller,'counts':counts,'access_type_counts':types,'request_size_counts':sizes,'addresses_unaligned_to_32B':unaligned,'scope':'one selected decode step; cold-start caches; V100 model; not hardware DRAM trace','prompt':'The capital of France is','prefill_tokens':5,'generation_limit':3,'kernel_window_official_inclusive':[1296,1833],'expected_kernel_count':538,'raw_export':str(p)}
(r/'accelsim_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
