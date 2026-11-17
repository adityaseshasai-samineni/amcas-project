import csv,json,collections,statistics
D=collections.defaultdict(list); runs=collections.defaultdict(list)
with open('samples.csv') as f:
 for r in csv.DictReader(f):
  r['cycles']=int(r['cycles']);r['seed']=int(r['seed']);D[r['label']].append(r);runs[(r['seed'],r['label'])].append(r['cycles'])
def q(a,p):
 a=sorted(a);return a[int((len(a)-1)*p)]
labels=['l1','l2','dram']; stats={}
for label,rows in D.items():
 a=[r['cycles'] for r in rows];stats[label]={'n':len(a),'p01':q(a,.01),'p05':q(a,.05),'median':q(a,.5),'p95':q(a,.95),'p99':q(a,.99),'min':min(a),'max':max(a)}
train={l:[r['cycles'] for r in D[l] if r['seed']<105] for l in labels};med={l:q(train[l],.5) for l in labels};ordered=sorted(labels,key=lambda l:med[l]);thresholds=[(med[ordered[i]]+med[ordered[i+1]])/2 for i in range(2)];bands={l:[q(train[l],.01),q(train[l],.99)] for l in labels};conf={l:dict.fromkeys(labels,0) for l in labels};abstain={l:dict.fromkeys(labels+['unknown'],0) for l in labels}
for l in labels:
 for r in D[l]:
  if r['seed']<105:continue
  v=r['cycles'];pred=ordered[sum(v>t for t in thresholds)];conf[l][pred]+=1;c=[k for k,(lo,hi) in bands.items() if lo<=v<=hi];abstain[l][c[0] if len(c)==1 else 'unknown']+=1
N=sum(sum(x.values()) for x in conf.values());correct=sum(conf[l][l] for l in labels);accepted=sum(sum(v for k,v in row.items() if k!='unknown') for row in abstain.values());acorrect=sum(abstain[l][l] for l in labels)
result={'quantiles_cycles':stats,'training_seeds':[101,102,103,104],'heldout_seeds':[105,106,107,108],'ordered_classes':ordered,'thresholds':thresholds,'training_01_99_bands':bands,'confusion':conf,'hard_accuracy':correct/N,'abstain_confusion':abstain,'coverage':accepted/N,'accepted_accuracy':acorrect/accepted if accepted else None,'unconditional_correct_fraction':acorrect/N,'runs':{str(k):{'median':q(v,.5),'p99':q(v,.99)} for k,v in runs.items()}}
with open('analysis.json','w') as f:json.dump(result,f,indent=2)
with open('report.md','w') as f:
 f.write('# Native V100 load-latency calibration\n\nControlled intended cache regimes; this is not an exact per-access hardware cache-level oracle or an LLM DRAM trace. Native unprofiled timing samples alone train/test the classifier. Separate Nsight runs check aggregate cache regimes.\n\n')
 f.write('One CTA, one active lane, randomized dependent 128-byte-stride pointer cycle. L1: 8 KiB, ld.global.ca, full-cycle warmup within measurement kernel. L2: 1 MiB, ld.global.cg, full-cycle preparation kernel. DRAM: 64 MiB, ld.global.cg, full-cycle preparation followed by first 8192 nodes; reuse distance exceeds 6 MiB L2. Allocation addresses remain live and distinct across all runs. SM placement is recorded, not pinned. Eight seeds, 8192 samples per regime per seed. Timer-only floor shares output-store instrumentation. Inline PTX predicates the ending clock read on loaded pointer, preserving completion dependency. Pointer outputs are host-validated.\n\n')
 f.write('| Regime | N | p1 | median | p95 | p99 |\n|---|---:|---:|---:|---:|---:|\n')
 for l in ['floor']+labels:
  s=stats[l];f.write('| %s | %d | %d | %d | %d | %d |\n'%(l,s['n'],s['p01'],s['median'],s['p95'],s['p99']))
 f.write('\nAll values are raw inclusive SM-clock cycles. Timer-floor subtraction is not claimed to remove all observer overhead. Global output stores, translation, clock changes, cache replacement and profiler effects remain limitations.\n\n')
 f.write('Training seeds101–104; held-out seeds105–108, distinct allocations. Median-midpoint hard-classifier accuracy: %.4f%%. Conservative unique p1–p99 band classifier coverage %.4f%%, accepted accuracy %.4f%%, unconditional correct fraction %.4f%%. These numbers assess controlled intended regimes, not per-access verified truth.\n\n'%(100*correct/N,100*accepted/N,100*acorrect/accepted if accepted else 0,100*acorrect/N))
 f.write('Confusion matrices and per-run statistics: analysis.json. Raw samples: samples.csv. Separate counter exports: ncu_*.csv. Inspect counter errors before treating regime labels as counter-confirmed. Latency overlap and unknown results must be retained when adapting this method to real kernels. No direct end-to-end LLM performance conclusion follows.\n')
