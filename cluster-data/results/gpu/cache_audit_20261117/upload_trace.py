from pathlib import Path
import http.client,json,os,hashlib,urllib.parse
r=Path.home()/'amcas-pro/results/gpu/cache_audit_20261117';s=Path('/scratch')/Path.home().name/'accel-q7short-20261117';meta=json.loads((r/'accelsim_summary.json').read_text());assert meta['validated_export_counts']
p=s/'dram.csv.gz';assert p.is_file() and p.stat().st_size<2000000000,'gzip asset exceeds GitHub limit; split with manifest before uploading'
tokpath=Path.home()/'.config/amcas/gh_token';assert (tokpath.stat().st_mode&0o777)==0o600;token=tokpath.read_text().strip();repo='adityaseshasai-samineni/amcas-project'
headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'amcas-cluster'}
def api(path):
 c=http.client.HTTPSConnection('api.github.com',timeout=60);c.request('GET',path,headers=headers);v=c.getresponse();b=v.read();assert v.status==200,(v.status,b[:200]);c.close();return json.loads(b)
release=api('/repos/'+repo+'/releases/tags/traces-2026-11-03');assets=api('/repos/'+repo+'/releases/'+str(release['id'])+'/assets?per_page=100');name='accelsim-v100-q7short-dram-admissions-20261117.csv.gz'
sha=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1<<20),b''):sha.update(b)
existing=[a for a in assets if a['name']==name]
if existing:
 a=existing[0];assert a['size']==p.stat().st_size
else:
 c=http.client.HTTPSConnection('uploads.github.com',timeout=600)
 h=dict(headers);h.update({'Content-Type':'application/gzip','Content-Length':str(p.stat().st_size)})
 with p.open('rb') as f:c.request('POST','/repos/'+repo+'/releases/'+str(release['id'])+'/assets?name='+urllib.parse.quote(name),body=f,headers=h)
 v=c.getresponse();b=v.read();assert v.status==201,(v.status,b[:200]);a=json.loads(b);assert a['size']==p.stat().st_size;c.close()
info={'url':a['browser_download_url'],'bytes':p.stat().st_size,'sha256':sha.hexdigest(),'scope':'simulated controller admissions; not hardware DRAM commands','schema':'dram_export_schema.md'};(r/'trace_asset.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
