#!/usr/bin/env python3
"""Stream validated controller CSV assets from cluster; token only on stdin."""
import pathlib,json,sys,http.client,urllib.parse,hashlib,math
A=pathlib.Path.home()/'amcas-pro';R=A/'results/gpu/campaign_20261120';name=sys.argv[1];O=R/'runs'/name;sim=json.load(open(O/'simulation.status.json'));assert sim['passed']
key=pathlib.Path.home()/'.config/amcas/gh_token';assert key.stat().st_mode&0o777==0o600;token=sys.stdin.readline().strip();assert token
repo='adityaseshasai-samineni/amcas-project';headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'amcas-cluster','X-GitHub-Api-Version':'2022-11-28'}
def api(path):
 c=http.client.HTTPSConnection('api.github.com',timeout=120);c.request('GET',path,headers=headers);r=c.getresponse();b=r.read();assert r.status==200,(r.status,b[:200]);c.close();return json.loads(b)
release=api('/repos/'+repo+'/releases/tags/traces-2026-11-03');assets=[];page=1
while True:
 xs=api('/repos/'+repo+'/releases/'+str(release['id'])+'/assets?per_page=100&page='+str(page));assets+=xs
 if len(xs)<100:break
 page+=1
class Segment:
 def __init__(self,f,n):self.f=f;self.left=n
 def read(self,n=8192):
  b=self.f.read(min(n,self.left));self.left-=len(b);return b
allinfo=[]
for ctx in sim['contexts']:
 p=pathlib.Path(ctx['export']);total=p.stat().st_size;limit=1900000000;parts=max(1,math.ceil(total/limit));overall=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):overall.update(b)
 info=dict(context=ctx['context'],source_file=str(p),bytes=total,sha256=overall.hexdigest(),parts=[],schema='sequence,core_cycle,partition,address_hex,write,bytes,access_type,sector_mask',scope='simulated controller admissions, not measured hardwareDRAMcommands')
 for part in range(parts):
  off=part*limit;size=min(limit,total-off);sha=hashlib.sha256()
  with p.open('rb') as f:
   f.seek(off);remain=size
   while remain:
    b=f.read(min(1<<20,remain));assert b;sha.update(b);remain-=len(b)
  asset=f'accelsim20-{name}-{ctx["context"]}-{overall.hexdigest()[:12]}.csv.gz'+(f'.part{part+1:03d}of{parts:03d}' if parts>1 else '')
  existing=[x for x in assets if x['name']==asset]
  if existing:a=existing[0];assert a['size']==size
  else:
   c=http.client.HTTPSConnection('uploads.github.com',timeout=1200);h=dict(headers);h.update({'Content-Type':'application/octet-stream','Content-Length':str(size)})
   with p.open('rb') as f:
    f.seek(off);c.request('POST','/repos/'+repo+'/releases/'+str(release['id'])+'/assets?name='+urllib.parse.quote(asset),body=Segment(f,size),headers=h)
   v=c.getresponse();b=v.read();assert v.status==201,(v.status,b[:200]);a=json.loads(b);assert a['size']==size;c.close();assets.append(a)
  if a.get('digest'):assert a['digest']=='sha256:'+sha.hexdigest(),a['digest']
  info['parts'].append(dict(name=asset,url=a['browser_download_url'],offset=off,bytes=size,sha256=sha.hexdigest()))
 allinfo.append(info);json.dump(dict(validated_simulation=True,assets=allinfo),open(O/'trace_assets.json','w'),indent=2)
# Keep original compressed official SASS. Only remove regenerable postprocessed copies
# after every controller CSV asset is durable and exact simulations passed.
cap=json.load(open(O/'capture.status.json'));s=pathlib.Path(cap['scratch'])/'traces';candidates=list(s.glob('kernel-*.tracez'));removed=[]
if candidates and all(x.with_suffix('.trace.xz').is_file() for x in candidates):
 for x in candidates:removed.append(dict(path=str(x),bytes=x.stat().st_size));x.unlink()
json.dump(dict(removed_regenerable_tracez=removed,original_trace_xz_retained=True,regenerate='post-traces-processing CAPTURE/traces -j4 with .amcas_trace_core in cwd'),open(O/'retention.json','w'),indent=2)
print(json.dumps(dict(case=name,contexts=len(allinfo),uploaded_parts=sum(len(x['parts']) for x in allinfo),regenerable_files_removed=len(removed))))
