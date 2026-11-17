#!/usr/bin/env python3
"""Saturated HBM2 replay of canonical Accel-Sim controller bursts.
GPU translation pilot: synthetic first-touch 2MiB placement, no timestamps,
canonical dirty-sector offsets; output is DRAM-model time, not token time.
"""
import argparse,json,os,sys,time,multiprocessing as mp
import numpy as np
A=os.path.expanduser('~/amcas-pro');sys.path[:0]=[A+'/gpu',A+'/tools']
import gpu_replay as G,remap as R,memsys_study as MS
POLS=['Open','ClosedCAP','Timeout16','Timeout64','Timeout256','HAPPY','HAPPYcnt','Adaptive','Adaptive7_4']
def run(t):
 import ramulator as rm
 path,sc,pol,window,sd=t
 def rp():
  if pol.startswith('Timeout'):return rm.row_policy.Timeout(timeout=int(pol[7:]))
  if pol.startswith('Adaptive'):
   mx,th=(7,4) if pol=='Adaptive7_4' else (15,6)
   return rm.row_policy.Adaptive(max=mx,thresh=th,init=mx)
  if pol.startswith('HAPPY'):return rm.row_policy.HAPPY(mode='count' if pol=='HAPPYcnt' else 'binary')
  return getattr(rm.row_policy,pol)()
 def ctrl():return rm.controller.HBM12(dram=rm.dram.HBM2(org_preset=sd['org'],timing_preset=sd['timing']),scheduler=rm.scheduler.FRFCFS(),refresh_manager=rm.refresh_manager.AllBank(),row_policy=rp(),addr_mapper=rm.addr_mapper.RoBaRaCoCh())
 mem=rm.memory_system.GenericDRAM(clock_ratio=1,controllers=[ctrl() for _ in range(32)],channel_mapper=rm.channel_mapper.CacheLineInterleave(interleave_bits=sd['ib']))
 fe=rm.frontend.LoadStoreTrace(clock_ratio=32,path=path);sim=rm.Simulation(fe,mem);sim.run()
 cs=sim.stats['memory_system']['controller'];tot=lambda k:sum(x[k] for x in cs)
 n=max(1,tot('row_hits')+tot('row_misses')+tot('row_conflicts'))
 return dict(window=window,mapping=sc,policy=pol,reads=tot('num_read_reqs'),writes=tot('num_write_reqs'),cycles=max(x['cycles'] for x in cs),time_us=max(x['cycles'] for x in cs)*1.25e-3,row_hit=tot('row_hits')/n,row_conflict=tot('row_conflicts')/n)
def main():
 p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('out');p.add_argument('--size',type=int,default=1000);p.add_argument('--procs',type=int,default=4);p.add_argument('--ib',type=int,default=2);p.add_argument('--windows',default='begin,middle,end');a=p.parse_args()
 os.makedirs(a.out,exist_ok=True);arr=np.memmap(a.binary,dtype=np.uint64,mode='r');N=len(arr)
 # Freeze placement before selecting windows: first-touch ranks across FULL stream.
 cache=a.binary+'.packed'
 if not os.path.isfile(cache):
  phys,pages=G.pack_pages((np.asarray(arr)>>np.uint64(1))<<np.uint64(5));phys.tofile(cache);json.dump(dict(pages=pages),open(cache+'.json','w'));del phys
 phys=np.memmap(cache,dtype=np.uint64,mode='r');sd=dict(G.SYS['v100']);sd['ib']=a.ib
 R.LAYOUTS['pilot']=dict(G.LAY['hbm2x32'],s=6+a.ib);L=R.Layout('pilot')
 # Train fixed row-only bank XOR using beginning window; evaluate frozen maps.
 R.cost.__defaults__=G.C.cost_params(sd)
 train,_=L.to_intra(phys[:min(1000000,N)]);cands=R.sources(L,False);M=R.search(L,R.sample(train),cands,3)
 chtrain=L.to_intra(phys[:min(1000000,N)])[1]
 Mc,cc,sel,_,_=MS.chan_search(L,train,chtrain,32,3,'chan')
 xtrain=R.apply_matrix(L,train,M,cands)
 Mxc,cxc,selxc,_,_=MS.chan_search(L,xtrain,chtrain,32,3,'xor_global_chan')
 assert all(bit>=L.row_lo for _,bit in cc+cxc)
 info=dict(size=a.size,total_canonical_bursts=N,ib=a.ib,channel_interleave_bytes=64<<a.ib,training='first 1000000 bursts; fixed for middle/end',xor_matrix={j:[cands[c][0] for c in M[j]] for j in range(L.bank)},channel_matrix=sel,xor_channel_matrix=selxc,units='32B canonical bursts; HBM2 1600Mbps tCK1.25ns; saturated open-loop',limitations=['missing dirty masks: contiguous write offsets are convention','synthetic first-touch2MiB placement','cold replay windows; no timestamps','Region/expert mappings require new attribution; MOP/ChRa require HBM decoder audit'])
 json.dump(info,open(a.out+'/manifest.json','w'),indent=2)
 starts={'begin':0,'middle':max(0,N//2-a.size//2),'end':max(0,N-a.size)};tasks=[]
 for name in a.windows.split(','):
  start=starts[name];byte=np.asarray(phys[start:start+a.size]);st=(arr[start:start+a.size]&np.uint64(1)).astype(bool);intra,ch=L.to_intra(byte)
  x=R.apply_matrix(L,intra,M,cands);naive=L.put_bank(intra,L.fields(intra)[0] ^ (L.fields(intra)[1]&((1<<L.bank)-1)))
  banklo=L.bank_lo;nb=max(1,int(intra.max()).bit_length()-banklo);rev=intra&np.uint64((1<<banklo)-1)
  for k in range(nb):rev|=((intra>>np.uint64(banklo+k))&np.uint64(1))<<np.uint64(banklo+nb-1-k)
  maps={'identity':(intra,ch),'xor_global':(x,ch),'naive':(naive,ch),'bitrev':(rev,ch),'chan':(intra,ch^MS.hash_bits(L,intra,Mc,cc).astype(ch.dtype)),'xor_global_chan':(x,ch^MS.hash_bits(L,x,Mxc,cxc).astype(ch.dtype))}
  # Row-only transformations never use offsets[6:5] in bank/channel decisions.
  # Channel search is not admitted if any selected source is a low column bit.
  for sc,(new,chv) in maps.items():
   path=a.out+'/'+name+'_'+sc+'.trc';G.write_trace(path,st,L.from_intra(new,chv))
   for pol in POLS:tasks.append((path,sc,pol,name,sd))
 res=[]
 with mp.get_context('spawn').Pool(a.procs) as pool:
  for r in pool.imap_unordered(run,tasks):
   res.append(r);json.dump(res,open(a.out+'/results.json','w'),indent=1);print(r,flush=True)
 for path in set(t[0] for t in tasks):os.unlink(path)
if __name__=='__main__':main()
