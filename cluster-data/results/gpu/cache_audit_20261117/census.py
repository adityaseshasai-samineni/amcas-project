from pathlib import Path
import re,csv,json,subprocess
r=Path.home()/'amcas-pro/results/gpu/cache_audit_20261117';s=Path('/scratch')/Path.home().name
stats=list((s/'accel-q7short-20261117/traces').glob('stats_ctx*'));assert len(stats)==1
rows=[]
for x in csv.reader(stats[0].open(),skipinitialspace=True):
 if not x or not x[0].startswith('kernel-'):continue
 k=int(re.match('kernel-(\d+)',x[0]).group(1))
 if 1296<=k<=1833:rows.append((k,x))
assert len(rows)==538,len(rows)
old={}
for l in (s/'gt/q7_v100_short_faauto.launch.txt').open():
 m=re.search(r'Kernel name (.*?) - grid launch id (\d+) - grid size (\d+),(\d+),(\d+) - block size (\d+),(\d+),(\d+)',l)
 if m:old[int(m[2])]=(m[1],tuple(map(int,m.groups()[2:])))
newnames=subprocess.check_output(['c++filt'],input=('\n'.join(x[1][1] for x in rows)+'\n').encode()).decode().splitlines()
name_mismatch=[]
for (k,x),name in zip(rows,newnames):
 grid=tuple(map(int,x[2:5]+x[6:9]));assert grid==old[k-1][1],(k,grid,old[k-1][1])
 if name!=old[k-1][0]:name_mismatch.append({'id':k,'new':name,'old':old[k-1][0]})
# Names can differ only in prettyprinter spelling; record every mismatch for review.
a={'kernels':len(rows),'all_launch_geometries_match':True,'name_mismatches':name_mismatch,'name_match_count':538-len(name_mismatch)}
(r/'capture_census.json').write_text(json.dumps(a,indent=2));print('geometries match',len(rows),'names matching',a['name_match_count'])
