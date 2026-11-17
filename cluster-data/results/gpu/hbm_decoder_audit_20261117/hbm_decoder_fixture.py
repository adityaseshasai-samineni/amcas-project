"""Independent scalar translation of installed ramulator2-grid HBM2_2Gb mappers.
Output fields channel,pseudochannel,sid,bankgroup,bank,row,column.
No existing source is modified. All byte addresses use32B transaction units.
"""
import json,random
BITS=[1,0,2,2,14,5]
def decode(p,mapper,ib=2):
 s=6+ib;ch=(p>>s)&31;i=((p>>(s+5))<<s)|(p&((1<<s)-1));a=i>>5
 if mapper=='mop':
  col=a&3;a>>=2;v=[]
  for w in BITS[:4]:v.append(a&((1<<w)-1));a>>=w
  col|=(a&7)<<2;a>>=3;v+=[a,col];shift=0
  for k,w in enumerate(BITS[:5]):
   if w:v[k]^=(col>>shift)&((1<<w)-1);shift+=w
 elif mapper=='chra':
  col=a&31;a>>=5;row=a&16383;a>>=14;v=[0]*6;v[5]=col;v[4]=row
  # ChRa extracts remaining levels in reverse order, bank then BG then SID then PC.
  for k in range(3,-1,-1):w=BITS[k];v[k]=a&((1<<w)-1);a>>=w
 else:
  col=a&31;a>>=5;v=[0]*6;v[5]=col
  for k,w in enumerate(BITS[:4]):v[k]=a&((1<<w)-1);a>>=w
  v[4]=a
 return [ch]+v
result={'bits':BITS,'tx_bytes':32,'channel_shift':8,'fixtures':{m:{str(p):decode(p,m) for p in [0,32,64,96,128,256,1024,32768,1<<24,1<<30]} for m in ['roba','mop','chra']}}
r=random.Random(173);invariant={}
for m in ['roba','mop','chra']:
 failures=0
 for _ in range(10000):
  base=r.randrange(1<<27)*128
  vals=[decode(base+o,m)[:6] for o in [0,32,64,96]]
  failures+=int(any(x!=vals[0] for x in vals[1:]))
 invariant[m]={'tested_128B_blocks':10000,'noncolumn_field_changes':failures}
assert invariant['roba']['noncolumn_field_changes']==0
assert invariant['chra']['noncolumn_field_changes']==0
assert invariant['mop']['noncolumn_field_changes']==10000
result['invariance']=invariant
print(json.dumps(result,indent=2))
