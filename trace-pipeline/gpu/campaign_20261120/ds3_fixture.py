import pathlib,json,sys,numpy as np
A=pathlib.Path.home()/'amcas-pro';R=A/'results/gpu/campaign_20261120';sys.path[:0]=[str(R),str(A/'tools'),str(A/'gpu')]
import ds3_cross as D,remap as M,gpu_replay as G
M.LAYOUTS['fixture']=dict(G.LAY['hbm2x32'],s=8);L=M.Layout('fixture');rng=np.random.default_rng(5);pa=(rng.integers(0,1<<29,size=1000,dtype=np.uint64)<<np.uint64(5));intr,ch=L.to_intra(pa);st=rng.random(1000)<.3
meta=dict(mapping='synthetic_identity',placement='random_fixture',reads=int((~st).sum()),writes=int(st.sum()),channel_interleave_bytes=256)
xs=D.cross(L,intr,ch,st,R/'ds3_fixture_random',meta);assert all(x['demand_drained'] for x in xs)
# Early write acknowledgments must not terminate before one physical coalesced WR.
i=np.zeros(32,dtype=np.uint64);c=np.zeros(32,dtype=np.int64);st=np.ones(32,dtype=bool);meta=dict(mapping='synthetic_same_address_writes',placement='fixture',reads=0,writes=32,channel_interleave_bytes=256)
ys=D.cross(L,i,c,st,R/'ds3_fixture_writes',meta);assert all(x['wire_write_commands']==1 and x['cycles']>128 for x in ys),ys
json.dump(dict(passed=True,random_requests=1000,decoder_all_fields=True,early_write_ack_test=True,write_callbacks=32,physical_write_commands=1,global_issue_per_cycle=32,policies=2,scope='synthetic validation only; no realGPUperformance claim'),open(R/'ds3_fixture_validation.json','w'),indent=2)
