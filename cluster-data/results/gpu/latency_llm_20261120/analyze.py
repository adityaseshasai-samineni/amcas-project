import csv,json,collections,statistics,os

def read(path):
 d=collections.defaultdict(list)
 with open(path) as f:
  for r in csv.DictReader(f):d[int(r['kernel'])].append(r)
 out=[]
 for index,k in enumerate(sorted(d)):
  rows=d[k]
  if path.endswith('l1.csv') or 'control' in path:rows=rows[64:]
  for r in rows:out.append({'seed':101+index,'cycles':int(r['cycles']),'pc':int(r['pc'])})
 return out

def q(a,p):
 a=sorted(a);return a[int((len(a)-1)*p)]
def stats(a):return dict(n=len(a),p01=q(a,.01),median=q(a,.5),p95=q(a,.95),p99=q(a,.99),min=min(a),max=max(a))
D={k:read('observer_'+k+'.csv') for k in ['l1','l2','dram']};control=read('observer_control.csv')
for k,rows in D.items():assert len(rows)==8192,(k,len(rows))
labels=['l1','l2','dram'];train={k:[r['cycles'] for r in v if r['seed']<105] for k,v in D.items()};test={k:[r['cycles'] for r in v if r['seed']>=105] for k,v in D.items()};med={k:q(v,.5) for k,v in train.items()};order=sorted(labels,key=lambda k:med[k]);thresholds=[(med[order[i]]+med[order[i+1]])/2 for i in range(2)];bands={k:[q(v,.01),q(v,.99)] for k,v in train.items()};conf={k:dict.fromkeys(labels,0) for k in labels};abstain={k:dict.fromkeys(labels+['unknown'],0) for k in labels}
for k,vs in test.items():
 for v in vs:
  pred=order[sum(v>t for t in thresholds)];conf[k][pred]+=1;matches=[l for l,(lo,hi) in bands.items() if lo<=v<=hi];abstain[k][matches[0] if len(matches)==1 else 'unknown']+=1
N=sum(map(len,test.values()));correct=sum(conf[k][k] for k in labels);accepted=sum(sum(v for k,v in row.items() if k!='unknown') for row in abstain.values());acorrect=sum(abstain[k][k] for k in labels)
result=dict(quantiles={k:stats([r['cycles'] for r in v]) for k,v in D.items()},probe_control=stats([r['cycles'] for r in control]),train_seeds=[101,102,103,104],test_seeds=[105,106,107,108],thresholds=thresholds,bands=bands,confusion=conf,abstain_confusion=abstain,hard_accuracy=correct/N,coverage=accepted/N,accepted_accuracy=acorrect/accepted if accepted else None,unconditional_correct=acorrect/N)
with open('analysis.json','w') as f:json.dump(result,f,indent=2)
with open('report.md','w') as f:
 f.write('# NVBit paired load-latency observer calibration\n\nPaired BEFORE/AFTER calls instrument actual GLOBAL LOAD (LDG) instructions only, excluding stores, atomics and LDGSTS. CTA0/thread0 is sampled. AFTER receives the original loaded destination register low32bits, enforcing a scoreboard dependency on the load result before callback execution. Timing includes NVBit calls, register-save/restore, argument delivery, buffer traffic and synchronization; it is not isolated hardware memory latency.\n\n')
 f.write('The BEFORE/BEFORE control uses identical callbacks and old destination value, with the original load executing afterward; it brackets no original load. It measures probe-path overhead rather than a DRAM access. The pre-timestamp membar and scratch writes alter the instruction stream and memory state.\n\n')
 f.write('| Regime | samples | p1 | median | p95 | p99 |\n|---|---:|---:|---:|---:|---:|\n')
 for k,v in list(result['quantiles'].items())+[('paired-call control',result['probe_control'])]:f.write('| %s | %s | %s | %s | %s | %s |\n'%(k,v['n'],v['p01'],v['median'],v['p95'],v['p99']))
 f.write('\nUnits raw inclusive SM cycles. L1 warmup64samples/kernel excluded. Eight distinct allocations/seeds per regime, with training seeds101–104 and testseeds105–108. Class labels indicate intended pointer-chase regimes, not per-access measured cache truth.\n\n')
 f.write('Held-out hard accuracy %.4f%%; unique p1–p99 band coverage %.4f%%; accepted accuracy %s; unconditional correct %.4f%%. Confusion matrices retained in analysis.json.\n\n'%(100*correct/N,100*accepted/N,str(result['accepted_accuracy']),100*acorrect/N))
 f.write('No thresholds from the earlier uninstrumented calibration were reused. Distinct regimes in this single-active-lane fixture do not establish valid classification under concurrent real LLM kernels. LLM application is gated on calibration usefulness and explicit perturbation limits.\n')
