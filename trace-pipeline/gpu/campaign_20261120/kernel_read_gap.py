import csv,re,sys,collections,glob
R=sys.argv[1]
# hardware per kernel (ID order within context)
rows=[];hdr=None
for line in open(f"{R}/hardware.raw",errors='replace'):
    if not line.startswith('"'):continue
    x=next(csv.reader([line]))
    if x[0]=='ID':hdr=x;continue
    if hdr and len(x)==len(hdr):rows.append(dict(zip(hdr,x)))
u={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9}
hw=collections.OrderedDict()
for x in rows:
    if x['Metric Name']!='dram__bytes_read.sum':continue
    hw[(x['Context'],int(x['ID']))]=(x['Kernel Name'],float(x['Metric Value'].replace(',',''))*u[x['Metric Unit']])
hwl=list(hw.values())
# simulator per kernel: "total dram reads" appears per kernel (cumulative?)
log=glob.glob(f"{R}/ctx_*/simulation.log")[0]
names=[];reads=[]
for line in open(log,errors='replace'):
    if line.startswith('kernel_name = '):names.append(line.split('=',1)[1].strip())
    elif line.startswith('total dram reads = '):reads.append(int(line.split('=')[1]))
print('hw kernels',len(hwl),'sim kernels',len(names),'reads entries',len(reads))
mono=all(b>=a for a,b in zip(reads,reads[1:]));print('sim reads cumulative?',mono)
per=[b-a for a,b in zip([0]+reads,reads)] if mono else reads
tot_hw=sum(b for _,b in hwl);tot_sim=sum(per)
print(f'hw read GB {tot_hw/1e9:.3f}; sim total dram reads {tot_sim} -> x32B {tot_sim*32/1e9:.3f} GB, x64 {tot_sim*64/1e9:.3f}, x128 {tot_sim*128/1e9:.3f}')
unit=float(sys.argv[2]) if len(sys.argv)>2 else 32
agg=collections.defaultdict(lambda:[0,0.0,0])
for (hn,hb),sn,sr in zip(hwl,names,per):
    k=re.sub(r'\(.*','',hn)[:60]; agg[k][0]+=sr*unit; agg[k][1]+=hb; agg[k][2]+=1
    if not sn.startswith(re.sub(r'\(.*','',hn)[:20].split('<')[0][:10]): pass
print(f"{'kernel':60s} {'n':>4} {'hw MB':>9} {'sim MB':>9} {'sim/hw':>7} {'gap MB':>8}")
for k,(s,h,n) in sorted(agg.items(),key=lambda kv:-(kv[1][1]-kv[1][0]))[:12]:
    print(f"{k:60s} {n:4d} {h/1e6:9.1f} {s/1e6:9.1f} {s/max(h,1):7.3f} {(h-s)/1e6:8.1f}")
