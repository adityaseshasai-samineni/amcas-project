import json,gzip,glob,sys
for g in sys.argv[1:]:
 res={r['mapping']+r['placement']+str(r['channel_interleave_bytes'])+r['policy']:r for r in json.load(open(g+'/dramsim3_results.json'))}
 for pl,s,il in [('packed',8,256),('packed',11,2048)]:
  print(g.split('/')[0],g.split('/')[1][:12],pl,il)
  for m in ['identity','xor_global','naive','mop','mopxor']:
   d=json.load(gzip.open(f'{g}/ds3_{pl}_s{s}_{m}/OPEN_PAGE/dramsim3.json.gz'));cs=[d[k] for k in sorted(d,key=lambda k:d[k]['channel'])]
   T=res[m+pl+str(il)+'Open']['cycles']
   row=[];col=[];act=[]
   for k in range(0,64,2):
    a=cs[k]['num_act_cmds']+cs[k+1]['num_act_cmds'];p=cs[k]['num_pre_cmds']+cs[k+1]['num_pre_cmds']
    c=cs[k]['num_read_cmds']+cs[k+1]['num_read_cmds']+cs[k]['num_write_cmds']+cs[k+1]['num_write_cmds']
    row.append((2*a+p)/T);col.append(c/T);act.append(a)
   A=sum(act);C=sum(c['num_read_cmds']+c['num_write_cmds'] for c in cs)
   # shared-CA lower bound on cycles: max over pairs of row-bus and col-bus busy cycles
   lb=max(max(r*T for r in row),max(x*T for x in col))
   print(f'  {m:11s} cyc {T:7d} ACT/col {A/C:.3f} rowbus/pair mean {sum(row)/32:.2f} max {max(row):.2f} colbus mean {sum(col)/32:.2f} | sharedCA lower bound {lb:7.0f} = {lb/T:.2f}x')
