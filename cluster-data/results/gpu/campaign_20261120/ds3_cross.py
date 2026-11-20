"""Matched timing/geometry/global-feed DRAMsim3 HBM2 cross-check.
64independent PCcontrollers; Ramulator instead couples2PCs/channel.
"""
import pathlib,sys,json,re,subprocess,concurrent.futures,gzip
import numpy as np
A=pathlib.Path.home()/'amcas-pro';R=A/'results/gpu/campaign_20261120'
sys.path.insert(0,str(A/'results/gpu/followup_20261115/code'))
import gpu_followup as F

def run_task(t):
 cfg,out,binary,policy,meta=t;out.mkdir(exist_ok=True)
 r=subprocess.run([str(R/'ds3_demand_drain'),str(cfg),str(out),str(binary),'32'],capture_output=True,text=True,check=True)
 m=re.search(r'DRAIN_CYCLES (\d+) REQS (\d+) WRITES (\d+) MISSING (\d+) UNISSUED (\d+).*DEMAND_EMPTY 1',r.stdout);assert m,r.stdout[-300:]
 assert int(m[2])==meta['reads']+meta['writes'] and int(m[3])==meta['writes'] and int(m[4])==int(m[5])==0
 raw=json.load(open(out/'dramsim3.json'));cs=list(raw.values())
 for key,expected in [('num_reads_done',meta['reads']),('num_writes_done',meta['writes'])]:assert sum(c[key] for c in cs)==expected,(key,expected)
 rd=sum(c['num_read_cmds'] for c in cs);wr=sum(c['num_write_cmds'] for c in cs);hits=sum(c['num_read_row_hits']+c['num_write_row_hits'] for c in cs);cyc=int(m[1])
 result=dict(meta,sim='DRAMsim3',policy=policy,cycles=cyc,time_us=cyc*.00125,wire_read_commands=rd,wire_write_commands=wr,wire_bytes=(rd+wr)*32,row_hit_rate=hits/max(rd+wr,1),accepted_read_callbacks=meta['reads'],accepted_write_callbacks=meta['writes'],demand_drained=True,postdrain_guard_cycles=128,feed='global32requests/controllercycle; in-order head-of-linebackpressure',remaining_difference='64independentPCs vs Ramulator32coupled2PCcontrollers; controller/scheduler/closepolicy implementations differ')
 # Preserve complete native counters compactly; redundant text is regenerable.
 with open(out/'dramsim3.json','rb') as f,gzip.open(out/'dramsim3.json.gz','wb') as z:z.write(f.read())
 for p in out.iterdir():
  if p.name!='dramsim3.json.gz' and p.is_file():p.unlink()
 return result

def cross(view,intra,channels,st,work,meta):
 work=pathlib.Path(work);work.mkdir(exist_ok=True);b,row,col=view.fields(intra);pc=channels.astype(np.int64)*2+(b&1);bank=b>>1
 addr=(((row.astype(np.uint64)<<np.uint64(6))|pc.astype(np.uint64))<<np.uint64(9))|(bank.astype(np.uint64)<<np.uint64(5))|col.astype(np.uint64);addr<<=np.uint64(5)
 rec=(addr<<np.uint64(1))|st.astype(np.uint64);binary=work/'global.bin';rec.tofile(binary);expected=work/'global.expected';np.column_stack((pc,bank,row,col)).astype('<u4').tofile(expected)
 sd=dict(F.G.SYS['v100']);tasks=[];configuration=None;native_configs={}
 for policy,label in [('OPEN_PAGE','Open'),('CLOSE_PAGE','Closed')]:
  cfg=work/(policy+'.ini');config=F.matched_config(cfg,'v100',sd,'matched');s=cfg.read_text()
  for key,value in [('channels',64),('row_buf_policy',policy)]:
   s,n=re.subn(r'(?m)^'+key+r'\s*=.*$',f'{key} = {value}',s);assert n==1,key
  cfg.write_text(s);native_configs[policy]=s;configuration=dict(config,channels=64,tx_bytes=32,issue_per_cycle=32);configuration['ds3_config']=s;configuration['remaining_differences']=[x for x in config['remaining_differences'] if not x.startswith('Global Ramulator feed')]+['Read merging/forwarding and close-page controller implementations differ'];configuration['native_configs']=native_configs
  checked=subprocess.run([str(R/'ds3_decode_check'),str(cfg),str(binary),str(expected)],capture_output=True,text=True,check=True);assert 'DECODE_VALIDATED' in checked.stdout
  tasks.append((cfg,work/policy,binary,label,meta))
 json.dump(configuration,open(work/'configuration.json','w'),indent=2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run_task,tasks))
 binary.unlink();expected.unlink();return results
