#!/usr/bin/env python3
import pathlib,json,re,subprocess,os,gzip,shutil,time
import sys
name=sys.argv[1];A=pathlib.Path.home()/'amcas-pro';W=A/'tracing';R=A/'results/gpu/campaign_20261120';O=R/'runs'/name;cap=json.load(open(O/'capture.status.json'));assert cap['passed'];assert json.load(open(R/'mask_fixture/validation.json'))['passed']
S=pathlib.Path(cap['scratch']);lists=list((S/'traces').glob('kernelslist.g'));assert len(lists)==1
entries=lists[0].read_text().splitlines();kernels=[x for x in entries if x.startswith('kernel-')];assert kernels,'empty selected GPUwindow'
contexts=sorted(set(re.search(r'ctx_0x[0-9a-fA-F]+',x).group(0) for x in kernels));results=[]
E=W/'mamba/envs/gem5';env=os.environ.copy();env.update(PATH=str(E/'bin')+':'+env['PATH'],LD_LIBRARY_PATH=str(E/'lib')+':'+str(W/'accel-build-deps/lib'))
for ctx in contexts:
 d=S/'simulation'/ctx;d.mkdir(parents=True,exist_ok=True);ro=O/ctx;ro.mkdir(exist_ok=True)
 for pat in ['*.config','*.icnt','*.xml']:
  for p in (A/'results/gpu/cache_audit_20261117/preflight').glob(pat):shutil.copy2(p,d/p.name)
 subset=[x for x in entries if not x.startswith('kernel-') or ctx in x]
 path=d/'kernelslist.g';path.write_text('\n'.join(str((S/'traces'/x).resolve()) if x.startswith('kernel-') else x for x in subset)+'\n')
 n=sum(x.startswith('kernel-') for x in subset);csv=d/'dram.csv';env.update(AMCAS_DRAM_EXPORT=str(csv),AMCAS_DRAM_EXPORT_MASKS='1')
 start=time.time()
 with open(ro/'simulation.log','w') as log,open(ro/'simulation.stderr','w') as err:
  rc=subprocess.call([str(W/'accel-sim-audit-20261117/gpu-simulator/build-cmake/accel-sim.out'),'-config','gpgpusim.config','-config','trace.config','-trace',str(path)],cwd=d,env=env,stdout=log,stderr=err)
 assert rc==0,(name,ctx,rc)
 log=(ro/'simulation.log').read_text(errors='replace');assert 'simulation thread exiting' in log
 launched=len(re.findall(r'(?m)^launching kernel name:',log));assert launched==n,(launched,n)
 uid=re.findall(r'(?m)^kernel_launch_uid = (.*?)\s*$',log);assert uid and int(uid[-1].split()[-1])==n
 counts=[int(x) for x in re.findall(r'(?m)^n_req = (\d+)\s*$',log)];assert len(counts)>=32
 with open(ro/'conversion.json','w') as out:subprocess.check_call([str(R/'export_masked_sectors'),str(csv),str(d/'masked.dram')],stdout=out,env=env)
 conv=json.load(open(ro/'conversion.json'));assert conv['controller_packets']==sum(counts[-32:])
 (ro/'simulation_final_stats.log').write_text(log[-1000000:])
 subprocess.check_call(['gzip','-1',str(csv)])
 result=dict(context=ctx,expected_kernels=n,kernels_completed=launched,seconds=time.time()-start,controller_packets=sum(counts[-32:]),conversion=conv,stream=str(d/'masked.dram'),export=str(csv)+'.gz',scope='independent V100 context model; no interGPU transfer/timing model; cold selectedwindow; source address placement synthetic in downstream replay')
 json.dump(result,open(ro/'simulation.status.json','w'),indent=2);results.append(result)
json.dump(dict(passed=True,contexts=results,kernels_total=sum(x['expected_kernels'] for x in results)),open(O/'simulation.status.json','w'),indent=2)
