import numpy as np, subprocess, sys, bisect, collections
sys.path.insert(0,'/home/adityaseshasai.s/amcas-pro/tools')
import analyze as A
d=sys.argv[1]
rec=np.memmap(f'{d}/dram.bin',dtype=A.REC,mode='r')
ph=A.read_phases(f'{d}/phases.tsv')
print('phase ms:',[round((p['end']-p['begin'])/1e9,2) for p in ph])
t0,t1=ph[0]['begin'],ph[-1]['end']
roi=rec[(rec['tick']>=t0)&(rec['tick']<t1)]
b=np.array([p['begin'] for p in ph],dtype=np.uint64); e=np.array([p['end'] for p in ph],dtype=np.uint64)
i=np.searchsorted(b,roi['tick'],side='right')-1
inph=roi['tick']<e[i]
wr=np.isin(roi['cmd'],list(A.WRITE_CMDS))
print('in-phase reqs',int(inph.sum()),'between-phase reqs',int((~inph).sum()),'between-phase writes',int((wr&~inph).sum()))
gen=roi['tick']>=e[0]
pp,vp=A.va_map(roi)
va=roi['vaddr'].copy(); need=va==0
pos=np.searchsorted(pp,roi['addr'][need]//A.PAGE); pos[pos>=len(pp)]=0
hit=pp[pos]==roi['addr'][need]//A.PAGE
tmp=np.zeros(int(need.sum()),dtype=np.uint64); tmp[hit]=vp[pos[hit]]*A.PAGE+roi['addr'][need][hit]%A.PAGE; va[need]=tmp
print('writeback VA recovered: %d/%d'%(int(hit.sum()),int(need.sum())))
R=A.label_tensors(f'{d}/amcas_tensors.tsv',None)
cl=np.zeros(len(roi),bool)
for k,v in R.items(): cl|=A.in_ranges(va,A.merge(v))
un=~cl
print('tensor kinds and sizes MiB:',{k:round(sum(b-a for a,b in A.merge(v).tolist())/2**20,2) for k,v in R.items()})
print('unattributed: reads',int((un&~wr).sum()),'writes',int((un&wr).sum()),'of which VA unknown',int((un&(va==0)).sum()))
vb=(va[un]>>20).astype(np.int64)
for (lab,m) in (('reads',~wr[un]),('writes',wr[un])):
    u,c=np.unique(vb[m],return_counts=True); o=np.argsort(-c)[:6]
    print(' top',lab,'VA MiB bins:',[(hex(int(u[j])<<20),int(c[j])) for j in o])
syms=[]
for l in subprocess.run(['nm','-n','--defined-only',sys.argv[2]],capture_output=True,text=True).stdout.splitlines():
    p=l.split()
    if len(p)==3 and p[1] in 'tTwW': syms.append((int(p[0],16),p[2]))
ad=[s[0] for s in syms]
def top(mask,title):
    pcs=roi['pc'][mask]; pcs=pcs[pcs!=0]
    u,c=np.unique(pcs,return_counts=True); cnt=collections.Counter()
    for a,n in zip(u.tolist(),c.tolist()):
        j=bisect.bisect_right(ad,a)-1; cnt[syms[j][1][:70]]+=n
    print(title,sum(cnt.values())); [print('   %8d %s'%(n,k)) for k,n in cnt.most_common(8)]
top(un&~wr&gen,'unattributed reads by function:')
top(cl&~wr&gen,'attributed reads by function:')
