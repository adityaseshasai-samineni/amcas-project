from pathlib import Path
import csv,json,re
r=Path.home()/'amcas-pro/results/gpu/cache_audit_20261117';s=Path('/scratch')/Path.home().name/'accel-q7short-20261117';a={'reads':0,'writes':0,'read_bytes':0,'write_bytes':0};seq=-1
with (s/'bench_dram.csv').open() as f:
 for x in csv.DictReader(f):
  assert int(x['sequence'])==seq+1;seq+=1;k='write' if int(x['write']) else 'read';a[k+'s']+=1;a[k+'_bytes']+=int(x['bytes'])
text=(r/'bench_sim.log').read_text();n=list(map(int,re.findall(r'(?m)^n_req = (\d+)\s*$',text)));assert sum(n[-32:])==seq+1
assert 'simulation thread exiting' in text
out={'pass':True,'kernel_official_id':1298,'counts':a,'controller_requests':seq+1,'wall_seconds':96.23,'simulation_rate_thread_instructions_per_second':949648,'scope':'one isolated cold-start matrix-vector kernel; not full token or hardware accuracy validation'}
(r/'representative_validation.json').write_text(json.dumps(out,indent=2));print(out)
