#!/usr/bin/env python3
"""Per-kernel model vs ncu. usage: l2_compare.py PERK.tsv NCU.raw SKIP [LAUNCH.txt]
ncu rows are in launch order: launch id = SKIP + k-th profiled kernel. Prints per kernel-class table:
ncu L2 read sectors / lookup miss / DRAM read sectors vs model L2 reads / DRAM reads per variant (MB = sectors*32/1e6)."""
import csv, sys, re, collections
perk, raw, skip = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [l for l in open(raw) if l.startswith('"')]
hdr = None; k = {}; names = {}
for r in csv.reader(rows):
    if r[0] == "ID": hdr = r; continue
    d = dict(zip(hdr, r)); i = int(d['ID']); m = d['Metric Name']
    try: v = float(d['Metric Value'].replace(',', ''))
    except ValueError: continue
    u = d['Metric Unit']; v *= {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9}.get(u, 1)
    k.setdefault(i, {})[m] = v; names[i] = d['Kernel Name']
ids = sorted(k); base = min(ids)
f = open(perk); cols = f.readline().strip().split('\t'); vars_ = sorted({c.rsplit('_', 1)[0] for c in cols[1:]}, key=lambda x: cols.index(x + '_l2rd'))
mod = {}
for l in f:
    p = l.strip().split('\t'); mod[int(p[0])] = dict(zip(cols[1:], map(int, p[1:])))
def cls(n):
    n = re.sub(r'\(.*', '', n); n = re.sub(r'<.*', '', n)
    return n.split()[-1] if n.split() else n
agg = collections.OrderedDict()
for i in ids:
    L = skip + (i - base); c = cls(names[i]); a = agg.setdefault(c, collections.Counter()); a['n'] += 1
    m = k[i]
    a['ncu_l2rd'] += m.get('lts__t_sectors_srcunit_tex_op_read.sum', 0); a['ncu_miss'] += m.get('lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum', 0)
    a['ncu_dram'] += m.get('dram__bytes_read.sum', 0) / 32; a['ncu_l1'] += m.get('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', 0)
    mm = mod.get(L, {})
    for v in vars_: a[v + '_l2rd'] += mm.get(v + '_l2rd', 0); a[v + '_miss'] += mm.get(v + '_miss', 0)
mb = lambda x: x * 32 / 1e6
tot = collections.Counter()
for a in agg.values(): tot.update(a)
agg['TOTAL'] = tot
print("%-34s %4s | %8s %8s %8s | " % ("kernel", "n", "ncuL2rd", "ncuMiss", "ncuDRAM") + " ".join("%9s" % v for v in vars_) + "   (MB; variant cols = DRAM-read MB)")
for c, a in sorted(agg.items(), key=lambda x: -x[1]['ncu_dram'])[:int(sys.argv[4]) if len(sys.argv) > 4 else 14]:
    print("%-34s %4d | %8.1f %8.1f %8.1f | " % (c[:34], a['n'], mb(a['ncu_l2rd']), mb(a['ncu_miss']), mb(a['ncu_dram'])) + " ".join("%9.1f" % mb(a[v + '_miss']) for v in vars_))
print("%-34s      | model L2rd (MB) " % "" + " ".join("%s=%.0f" % (v, mb(tot[v + '_l2rd'])) for v in vars_))
