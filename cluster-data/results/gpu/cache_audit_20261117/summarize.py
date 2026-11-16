from pathlib import Path
import csv,re,json,gzip,hashlib
r=Path.home()/'amcas-pro/results/gpu/cache_audit_20261117';s=Path('/scratch')/Path.home().name/'accel-q7short-20261117'
p=s/'dram.csv.gz'; opener=gzip.open if p.exists() else open
if not p.exists():p=s/'dram.csv'
counts={'read_requests':0,'write_requests':0,'read_bytes':0,'write_bytes':0};types={};last=-1
with opener(p,'rt') as f:
 for x in csv.DictReader(f):
  assert int(x['sequence'])==last+1;last+=1
  k='write' if int(x['write']) else 'read';counts[k+'_requests']+=1;counts[k+'_bytes']+=int(x['bytes']);types[x['access_type']]=types.get(x['access_type'],0)+1
log=(r/'simulation.log').read_text(errors='replace')
nreq=[int(x) for x in re.findall(r'(?m)^n_req = (\d+)\s*$',log)]
assert len(nreq)>=32
controller=sum(nreq[-32:]);assert controller==last+1,(controller,last+1)
assert 'simulation thread exiting' in log
out={'validated_export_counts':True,'controller_requests':controller,'counts':counts,'access_type_counts':types,'scope':'one selected decode step; cold-start caches; V100 model; not hardware DRAM trace','prompt':'The capital of France is','prefill_tokens':5,'generation_limit':3,'kernel_window_official_inclusive':[1296,1833],'expected_kernel_count':538,'raw_export':str(p)}
(r/'accelsim_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
