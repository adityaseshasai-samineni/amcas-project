#!/usr/bin/env python3
"""CPU-grid-equivalent mapping/policy sweep on masked GPU32B sectors.
All learned mappings fit the selected evaluation window, matching the CPU grid;
this is an offline characterization, not held-out prediction or token timing.
"""
import argparse,json,os,pathlib,sys,multiprocessing as mp
import numpy as np
A=pathlib.Path.home()/'amcas-pro';sys.path[:0]=[str(A/'tools'),str(A/'gpu')]
import remap as R,region_remap as RR,memsys_study as MS,mop4_traces as MT
import grid_gen as GG,grid_replay as GR,drmiss_export as X,mapprep as P,moe_remap as MR
import gpu_replay as G
POLS=['Open','ClosedCAP','Timeout16','Timeout64','Timeout256','Region','HAPPY','HAPPYcnt','Adaptive','Adaptive7_4']
def replay(t):
 import ramulator as rm
 path,mapper,pol,meta=t;path=pathlib.Path(path)
 def ctrl():return rm.controller.HBM12(dram=rm.dram.HBM2(org_preset='HBM2_2Gb',timing_preset='HBM2_1600Mbps'),scheduler=rm.scheduler.FRFCFS(),refresh_manager=rm.refresh_manager.AllBank(),row_policy=GR.policy(rm,pol,str(path.parent),path.stem,''),addr_mapper=getattr(rm.addr_mapper,mapper)())
 mem=rm.memory_system.GenericDRAM(clock_ratio=1,controllers=[ctrl() for _ in range(32)],channel_mapper=rm.channel_mapper.CacheLineInterleave(interleave_bits=meta['ramulator_interleave_bits']))
 fe=rm.frontend.LoadStoreTrace(clock_ratio=32,path=str(path));sim=rm.Simulation(fe,mem);sim.run();cs=sim.stats['memory_system']['controller'];total=lambda k:sum(c[k] for c in cs)
 assert total('num_read_reqs')==meta['reads'] and total('num_write_reqs')==meta['writes']
 assert total('num_read_reqs_served')+total('num_read_reqs_forwarded')==meta['reads']
 assert total('num_write_reqs_served')+total('num_write_reqs_coalesced')==meta['writes']
 cyc=max(c['cycles'] for c in cs);events=total('row_hits')+total('row_misses')+total('row_conflicts')
 return dict(meta,mapper=mapper,policy=pol,cycles=cyc,time_us=cyc*0.00125,row_hit_rate=total('row_hits')/max(events,1),row_conflict_rate=total('row_conflicts')/max(events,1),read_forwarded=total('num_read_reqs_forwarded'),write_coalesced=total('num_write_reqs_coalesced'),drain_guard_memory_cycles=128)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('binary');ap.add_argument('tensors');ap.add_argument('out');ap.add_argument('--size',type=int,default=4000000);ap.add_argument('--procs',type=int,default=2);ap.add_argument('--smoke',action='store_true');ap.add_argument('--source-metadata');ap.add_argument('--ds3',action='store_true');a=ap.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 arr=np.memmap(a.binary,dtype=np.uint64,mode='r');packed=np.memmap(a.binary+'.packed',dtype=np.uint64,mode='r');assert len(arr)==len(packed)
 n=len(arr);w=min(a.size,n);start=min(int(n*.3),n-w);v=(np.asarray(arr[start:start+w])>>np.uint64(1))<<np.uint64(5);pa=np.asarray(packed[start:start+w]);st=(arr[start:start+w]&np.uint64(1)).astype(bool)
 rows=X.load_tensors(a.tensors);semantic_va=v.view(np.int64);cls,tix=X.classify(semantic_va,rows);grp=np.array([0,1,0,2,2],dtype=np.int8)[cls];coverage=float((tix>=0).mean());features=P.expert_features(dict(rows=rows,tix=tix,cls=cls,va=semantic_va));hasexpert=bool((features[0]>=0).any());assert not hasexpert or features[0].max()<128
 manifest=dict(total_bursts=n,window=[start,start+w],reads=int((~st).sum()),writes=int(st.sum()),tensor_coverage=coverage,expert_accesses=int((features[0]>=0).sum()),request_bytes=32,tck_ps=1250,scope='cold saturated open-loop DRAM-model characterization; not native GPU token speedup',training='learned mappings fit selected evaluation window, as CPU grid; no held-out claim',placement='first-touch2MiB packed and seed1 2MiB shuffled across16GiB',mask_positions='actual exported sector masks; ascending sector order within each controller packet',excluded=[])
 if a.source_metadata:manifest['source']=json.load(open(a.source_metadata))
 if coverage<.5:manifest['excluded'].append('Region mapping/policy: tensor coverage below50%')
 if not hasexpert:manifest['excluded'].append('expert mappings: no attributed expert accesses in selected window')
 json.dump(manifest,open(out/'manifest.json','w'),indent=2);results=[]
 for pl in (['packed'] if a.smoke else ['packed','thp']):
  if pl=='packed':phys=pa
  else:
   # First-touch rank is frozen over the complete stream, not re-packed by window.
   frames=np.random.default_rng(1).permutation(8192).astype(np.uint64);rank=pa>>np.uint64(21);assert rank.max()<8192;phys=(frames[rank]<<np.uint64(21))|(pa&np.uint64((1<<21)-1))
  for shift in ([8] if a.smoke else [8,11]):
   R.LAYOUTS['campaign']=dict(G.LAY['hbm2x32'],s=shift);L=R.Layout('campaign');ML=MT.MopView(L);intra,ch=L.to_intra(phys);ch=ch.astype(np.int64)
   RR.cost2.__defaults__=(2.0,40.0);MS.cost3.__defaults__=(2.0,40.0);R.cost.__defaults__=(2.0,40.0)
   M,sel,_,_=RR.search(L,intra,ch,32,3,'global');xg=RR.apply_named(L,intra,M)
   Mc,cc,sc,_,_=MS.chan_search(L,intra,ch,32,3,'channel');Mxc,cxc,sxc,_,_=MS.chan_search(L,xg,ch,32,3,'joint')
   Mm,sm,_,_=RR.search(ML,intra,ch,32,3,'MOP');mxg=RR.apply_named(ML,intra,Mm)
   maps={'identity':(intra,ch,'RoBaRaCoCh'),'xor_global':(xg,ch,'RoBaRaCoCh'),'chan':(intra,ch^MS.hash_bits(L,intra,Mc,cc),'RoBaRaCoCh'),'xor_global_chan':(xg,ch^MS.hash_bits(L,xg,Mxc,cxc),'RoBaRaCoCh'),'mop':(intra,ch,'MOP4CLXOR'),'mopxor':(mxg,ch,'MOP4CLXOR'),'chra':(intra,ch,'ChRaBaRoCo'),'bitrev':(RR.bitrev(L,intra),ch,'RoBaRaCoCh'),'naive':(RR.apply_named(L,intra,[[(f'row{j}',L.row_lo+j)] for j in range(L.bank)]),ch,'RoBaRaCoCh')}
   matrices=dict(global_bank=sel,channel=sc,joint_channel=sxc,mop_bank=sm)
   if coverage>=.5:
    gran=phys>>np.uint64(21);ug,inv=np.unique(gran,return_inverse=True);cnt=np.zeros((len(ug),3),dtype=np.int64);np.add.at(cnt,(inv,grp),1);tags=cnt.argmax(1)[inv];reg=intra.copy();matrices['region']={}
    for gi in [1,2]:
     mask=tags==gi
     if mask.sum()<5000:continue
     M,selr,_,_=RR.search(L,intra[mask],ch[mask],32,3,'region');reg[mask]=RR.apply_named(L,intra[mask],M);matrices['region'][str(gi)]=selr
    maps['region']=(reg,ch,'RoBaRaCoCh')
   if hasexpert:
    maps['expert_hash']=(MR.expert_xor(L,intra,*features,'hash'),ch,'RoBaRaCoCh');maps['expert_lin']=(MR.expert_xor(L,intra,*features,'lin'),ch,'RoBaRaCoCh')
   json.dump(matrices,open(out/f'{pl}_s{shift}_matrices.json','w'),indent=2)
   # Process one mapping at a time: bounded disk usage and recoverable results.
   for name,(i,c,mapper) in maps.items():
    dest=L.from_intra(i,c.astype(np.uint64));assert np.unique(dest).size==np.unique(phys).size,(pl,shift,name,'not injective')
    if name=='region':assert ((dest>>np.uint64(21))==(phys>>np.uint64(21))).all()
    path=out/f'{pl}_s{shift}_{name}.trc';G.write_trace(path,st,dest);GG.write_ranges(str(path.with_suffix('.ranges')),dest,grp)
    meta=dict(mapping=name,placement=pl,channel_interleave_bytes=1<<shift,ramulator_interleave_bits=shift-5,reads=manifest['reads'],writes=manifest['writes'])
    pols=[p for p in POLS if p!='Region' or coverage>=.5]
    with mp.get_context('spawn').Pool(a.procs) as pool:
     for r in pool.imap_unordered(replay,[(str(path),mapper,p,meta) for p in pols]):results.append(r);json.dump(results,open(out/'results.json','w'),indent=1);print(r,flush=True)
    if a.ds3:
     assert json.load(open(pathlib.Path(__file__).parent/'ds3_fixture_validation.json'))['passed']
     import ds3_cross
     view=ML if mapper=='MOP4CLXOR' else GG.ChRaView(L) if mapper=='ChRaBaRoCo' else L
     ds=ds3_cross.cross(view,i,c,st,out/f'ds3_{pl}_s{shift}_{name}',meta)
     dp=out/'dramsim3_results.json';old=json.load(open(dp)) if dp.is_file() else [];json.dump(old+ds,open(dp,'w'),indent=1)
    path.unlink();path.with_suffix('.ranges').unlink()
 json.dump(dict(passed=True,replays=len(results)),open(out/'complete.json','w'),indent=2)
if __name__=='__main__':main()
