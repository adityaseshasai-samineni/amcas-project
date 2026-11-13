#!/usr/bin/env python3
"""In-order row-hit diagnostics of a gpu_filter DRAM stream (HBM2 V100 layout, identity vs searched bank XOR).
usage: gpu_diag.py DRAM_PREFIX TENSORS.tsv OUT_JSON [--window N]
 (1) per tensor class: hit rate of its requests in the merged stream (previous request to the same (channel,bank), any class)
     and 'own' hit rate (previous request of the same class to that bank);
 (3) if .dramlaunch carries SM ids: hit rate when each SM's sub-stream is considered alone vs the merged stream;
     and per launch (kernel) hit rates of the largest kernels.
In-order = a request hits if the previous request to its (channel, bank) in stream order targeted the same row (remap.inorder_hits model)."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/amcas-pro/tools"))
import remap as R
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_replay as G

pre, tsv, out = sys.argv[1:4]
gpu = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "v100"
win = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import gpu_io
arr = gpu_io.load(pre, "dram", np.uint64)
dl = gpu_io.load(pre, "dramlaunch", np.uint32)
if win: arr, dl = arr[:win], dl[:win]
addr = (arr >> np.uint64(1)) << np.uint64(5)
rows = [l.rstrip("\n").split("\t") for l in open(tsv)][1:]
last = max(int(r[0]) for r in rows)
T = sorted({(int(r[2], 16), int(r[3]), r[1]) for r in rows if int(r[0]) == last})
def cls(n):
    if re.search(r"attn_(q|k|v)\.", n): return "w_attn_qkv"
    if "attn_output" in n: return "w_attn_out"
    if re.search(r"ffn_(gate|up|down)", n) and "weight" in n: return "w_ffn"
    if n.startswith("cache_k"): return "kv_k"
    if n.startswith("cache_v"): return "kv_v"
    if n == "output.weight": return "w_output"
    if n.endswith(".weight"): return "w_other"
    return "activation"
start = np.array([t[0] for t in T], dtype=np.uint64); end = start + np.array([t[1] for t in T], dtype=np.uint64)
names = [cls(t[2]) for t in T]; uc = sorted(set(names)) + ["unknown"]
cid = np.array([uc.index(n) for n in names])
i = np.clip(np.searchsorted(start, addr, side="right") - 1, 0, None)
c = np.where((addr < end[i]) & (addr >= start[i]), cid[i], len(uc) - 1)
phys, _ = G.pack_pages(addr)
SD = G.SYS[gpu]; L = R.Layout(SD["layout"])
intra, ch = L.to_intra(phys)
bus, tmiss = G.C.cost_params(SD); R.cost.__defaults__ = (bus, tmiss)
cands = R.sources(L, False)
Mx = R.search(L, R.sample(intra), cands, 3)
variants = {"identity": intra, "xor_bank": R.apply_matrix(L, intra, Mx, cands)}
def hits(key, row, group=None):
    """hit flag per request: previous request with the same key (and same group) has the same row"""
    k = key if group is None else (group.astype(np.int64) << 40) | key
    o = np.argsort(k, kind="stable"); kk, rr = k[o], row[o]
    h = np.zeros(len(k), dtype=bool); h[1:] = (kk[1:] == kk[:-1]) & (rr[1:] == rr[:-1])
    res = np.empty(len(k), dtype=bool); res[o] = h
    return res
res = dict(xor_matrix={j: [cands[x][0] for x in Mx[j]] for j in range(L.bank)}, layout=SD["layout"] + " " + str(G.LAY[SD["layout"]]) + ", 256 B channel interleave; XOR sources = intra row bits")
sm = (dl >> np.uint32(20)).astype(np.int64); launch = (dl & np.uint32((1 << 20) - 1)).astype(np.int64)
has_sm = sm.max() > 0
for vn, new in variants.items():
    b, r, _ = L.fields(new); key = ch.astype(np.int64) * L.nbanks + b
    h = hits(key, r); v = dict(overall=float(h.mean()))
    v["class"] = {}
    for k, n in enumerate(uc):
        m = c == k
        if not m.any(): continue
        own = hits(key[m], r[m])   # stream of this class alone
        v["class"][n] = dict(share=float(m.mean()), merged_hit=float(h[m].mean()), own_hit=float(own.mean()))
    if has_sm:
        hs = hits(key, r, group=sm); v["per_sm_hit"] = float(hs.mean())
        v["per_sm_class"] = {n: float(hs[c == k].mean()) for k, n in enumerate(uc) if (c == k).any()}
        # per launch: largest launches, merged vs per-sm
        ul, cnt = np.unique(launch, return_counts=True)
        top = ul[np.argsort(-cnt)[:6]]
        v["top_launches"] = {int(l): dict(n=int((launch == l).sum()), merged=float(h[launch == l].mean()), per_sm=float(hs[launch == l].mean())) for l in top}
        # concurrent SMs: mean number of distinct SMs among each consecutive 1000 requests
        v["mean_distinct_sm_per_1000"] = float(np.mean([len(np.unique(sm[j:j+1000])) for j in range(0, len(sm) - 1000, 50000)]))
    res[vn] = v
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1)[:6000])
