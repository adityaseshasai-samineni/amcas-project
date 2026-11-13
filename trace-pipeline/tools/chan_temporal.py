import numpy as np,sys,glob
for f in sorted(glob.glob(sys.argv[1]+"/*.trc")):
    a=np.array([int(l.split()[1],16) for l in open(f)],dtype=np.uint64)
    nch=8 if "ddr5x8" in f else 16
    ch=((a>>np.uint64(8))&np.uint64(nch-1)).astype(int)
    out=[]
    for W in (32,128):
        n=len(ch)//W; c=ch[:n*W].reshape(n,W)
        cnt=np.stack([(c==k).sum(1) for k in range(nch)],1)
        out.append("W%d max/mean %.2f, chans used %.1f/%d"%(W,(cnt.max(1)/(W/nch)).mean(),(cnt>0).sum(1).mean(),nch))
    print(f.split("/")[-1].ljust(28),len(a)," | ".join(out))
