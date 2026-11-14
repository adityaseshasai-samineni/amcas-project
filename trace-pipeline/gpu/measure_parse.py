#!/usr/bin/env python3
"""Parse ncu csv of one decode step: per-kernel tsv (next to raw) + class summary json on stdout. usage: measure_parse.py RAW TIMING TAG"""
import csv, sys, json, re, collections
raw, timing, tag = sys.argv[1:4]
U = {'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1,'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'nsecond':1e-9,'usecond':1e-6,'msecond':1e-3,'second':1,'%':1,'':1}
ks = collections.OrderedDict(); hdr = None
for r in csv.reader([l for l in open(raw, errors='replace') if l.startswith('"')]):
    if r[0] == "ID": hdr = r; continue
    d = dict(zip(hdr, r))
    try: v = float(d['Metric Value'].replace(',', ''))
    except ValueError: continue
    v *= U.get(d['Metric Unit'], 1)
    k = ks.setdefault(d['ID'], {'name': d['Kernel Name']})
    k[d['Metric Name']] = v
def cls(n):
    if 'mul_mat_vec' in n: return 'mul_mat_vec'
    if 'mul_mat_q' in n: return 'mul_mat_q'
    if 'flash_attn' in n: return 'flash_attn'
    if 'quantize' in n: return 'quantize_q8'
    if 'soft_max' in n: return 'soft_max'
    if 'rms_norm' in n: return 'norm'
    return 'other'
C = collections.defaultdict(lambda: collections.Counter()); T = collections.Counter()
f = open(raw.replace('.ncu.raw', '.perk.tsv'), 'w')
f.write("id\tclass\tname\trd_B\twr_B\ttime_s\tdram_pct\tsm_pct\tsol_mem_pct\tl2hit_pct\n")
for i, k in ks.items():
    rd = k.get('dram__bytes_read.sum', 0); wr = k.get('dram__bytes_write.sum', 0); t = k.get('gpu__time_duration.sum', 0)
    dp = k.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0); sp = k.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0)
    sm_ = k.get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 0); hit = k.get('lts__t_sector_hit_rate.pct', 0)
    c = cls(k['name'])
    f.write("%s\t%s\t%s\t%d\t%d\t%.3e\t%.1f\t%.1f\t%.1f\t%.1f\n" % (i, c, k['name'][:60], rd, wr, t, dp, sp, sm_, hit))
    for a, b in (('n',1),('rd',rd),('wr',wr),('t',t),('dp_t',dp*t),('sp_t',sp*t),('hit_t',hit*t)):
        C[c][a] += b; T[a] += b
out = {'tag': tag, 'kernels': int(T['n']), 'dram_read_GB': T['rd']/1e9, 'dram_write_GB': T['wr']/1e9, 'kernel_time_ms': T['t']*1e3,
       'bw_GBps_kernel_time': (T['rd']+T['wr'])/max(T['t'], 1e-12)/1e9, 'dram_pct_time_weighted': T['dp_t']/max(T['t'],1e-12),
       'sm_pct_time_weighted': T['sp_t']/max(T['t'],1e-12), 'l2hit_pct_time_weighted': T['hit_t']/max(T['t'],1e-12), 'classes': {}}
for c, v in C.items():
    out['classes'][c] = {'n': int(v['n']), 'rd_GB': v['rd']/1e9, 'wr_GB': v['wr']/1e9, 'time_ms': v['t']*1e3,
        'time_share': v['t']/max(T['t'],1e-12), 'bw_GBps': (v['rd']+v['wr'])/max(v['t'],1e-12)/1e9,
        'dram_pct': v['dp_t']/max(v['t'],1e-12), 'sm_pct': v['sp_t']/max(v['t'],1e-12), 'l2hit_pct': v['hit_t']/max(v['t'],1e-12)}
for l in open(timing):
    m = re.search(r"\beval time =\s+([\d.]+) ms /\s+(\d+) (runs|tokens).*?([\d.]+) tokens per second", l)
    if m and 'prompt' not in l: out['token_ms_unprofiled'] = float(m.group(1))/int(m.group(2)); out['tok_per_s_unprofiled'] = float(m.group(4))
print(json.dumps(out, indent=1))
