#!/usr/bin/env python3
import json,pathlib,sys,hashlib,subprocess
import numpy as np
A=pathlib.Path.home()/'amcas-pro';R=A/'results/gpu/campaign_20261120';name=sys.argv[1];O=R/'runs'/name;sim=json.load(open(O/'simulation.status.json'));assert sim['passed'];assert json.load(open(R/'policy_fixture_validation.json'))['passed']
for c in sim['contexts']:
 src=pathlib.Path(c['stream']);a=np.memmap(src,dtype=np.uint64,mode='r');p=np.memmap(str(src)+'.packed',dtype=np.uint64,mode='r');assert len(a)==len(p);n=len(a);w=min(4000000,n);s=min(int(n*.3),n-w)
 d=A/'tracing/campaign20-grid-input'/name/c['context'];d.mkdir(parents=True,exist_ok=True);dst=d/'window.dram';np.asarray(a[s:s+w]).tofile(dst);np.asarray(p[s:s+w]).tofile(str(dst)+'.packed')
 meta=dict(source_controller_context=c['context'],source_total_32B_sectors=n,source_window=[s,s+w],physical_page_ranks='frozen first-touch over complete source context stream',source_csv=c['export'],input_sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
 mo=O/c['context'];json.dump(meta,open(mo/'policy_source.json','w'),indent=2)
 cmd=['sbatch','--parsable','-J','policy-'+name,'-o',str(mo/'policy_job.log'),str(R/'policy_case.sh'),str(dst),str(O/'capture.tensors.tsv'),str(mo/'policy_grid'),str(mo/'policy_source.json')]
 job=subprocess.check_output(cmd,text=True).strip();(mo/'policy_job.txt').write_text(job+'\n');print(name,c['context'],'policy_job',job,flush=True)
