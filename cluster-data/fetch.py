import json,subprocess,os,hashlib,sys,time,urllib.request
repo="gem5/ubuntu-24.04_all-dependencies"; tag="latest"; out=os.path.expanduser("~/amcas-pro/tracing/gem5-oci")
os.makedirs(out+"/blobs/sha256",exist_ok=True)
tok=json.load(urllib.request.urlopen(f"https://ghcr.io/token?scope=repository:{repo}:pull"))["token"]
ACC="application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"
def man(ref):
    r=urllib.request.Request(f"https://ghcr.io/v2/{repo}/manifests/{ref}",headers={"Authorization":"Bearer "+tok,"Accept":ACC})
    x=urllib.request.urlopen(r); return x.read(),x.headers["Content-Type"]
b,ct=man(tag); m=json.loads(b)
if "manifests" in m:
    e=[x for x in m["manifests"] if x["platform"]["architecture"]=="amd64"][0]; b,ct=man(e["digest"]); m=json.loads(b)
def save(data):
    h=hashlib.sha256(data).hexdigest(); open(f"{out}/blobs/sha256/{h}","wb").write(data); return "sha256:"+h,len(data)
for d in [m["config"]]+m["layers"]:
    h=d["digest"].split(":")[1]; p=f"{out}/blobs/sha256/{h}"
    print(d["digest"][:20],d["size"],flush=True)
    for a in range(60):
        if os.path.exists(p) and os.path.getsize(p)>=d["size"]: break
        subprocess.call(["curl","-sSL","-C","-","--retry","3","-m","600","-H","Authorization: Bearer "+tok,"-o",p,f"https://ghcr.io/v2/{repo}/blobs/{d['digest']}"]); time.sleep(3)
    s=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<22),b""): s.update(c)
    if s.hexdigest()!=h: sys.exit("checksum mismatch "+h)
dg,sz=save(b)
json.dump({"schemaVersion":2,"manifests":[{"mediaType":ct,"digest":dg,"size":sz,"annotations":{"org.opencontainers.image.ref.name":"latest"}}]},open(out+"/index.json","w"))
json.dump({"imageLayoutVersion":"1.0.0"},open(out+"/oci-layout","w"))
print("OK",len(m["layers"]),"layers")
