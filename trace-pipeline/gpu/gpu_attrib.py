#!/usr/bin/env python3
"""Attribute a gpu_filter DRAM stream to ggml tensor classes. usage: gpu_attrib.py DRAM_PREFIX TENSORS.tsv [OUT_JSON]
Uses the LAST step in the TSV (the traced one). Classes: weights by role (attn_qkv, attn_out, ffn, output, embd),
KV cache, activations (everything else on the device), unknown (address in no listed tensor)."""
import json, re, sys
import numpy as np
pre, tsv = sys.argv[1:3]
rows = [l.rstrip("\n").split("\t") for l in open(tsv)][1:]
last = max(int(r[0]) for r in rows)
T = sorted({(int(r[2], 16), int(r[3]), r[1], r[10] if len(r) > 10 else "") for r in rows if int(r[0]) == last})
def cls(n):
    if re.search(r"attn_(q|k|v)\.", n): return "w_attn_qkv"
    if "attn_output" in n: return "w_attn_out"
    if re.search(r"ffn_(gate|up|down)", n) and "weight" in n: return "w_ffn"
    if n.startswith("cache_k") or n.startswith("cache_v") or "kv" in n.lower(): return "kv_cache"
    if n == "output.weight": return "w_output"
    if n == "token_embd.weight": return "w_embd"
    if n.endswith(".weight"): return "w_other"
    return "activation"
start = np.array([t[0] for t in T], dtype=np.uint64); end = start + np.array([t[1] for t in T], dtype=np.uint64)
names = [cls(t[2]) for t in T]; uc = sorted(set(names)) + ["unknown"]
cid = np.array([uc.index(n) for n in names] + [len(uc) - 1])
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import gpu_io
arr = gpu_io.load(pre, "dram", np.uint64)
addr = (arr >> np.uint64(1)) << np.uint64(5); wr = (arr & np.uint64(1)).astype(bool)
i = np.searchsorted(start, addr, side="right") - 1
ok = (i >= 0) & (addr < end[np.clip(i, 0, None)])
c = np.where(ok, cid[:-1][np.clip(i, 0, None)], len(uc) - 1)
out = {}
for k, n in enumerate(uc):
    m = c == k
    out[n] = dict(read_MB=float((m & ~wr).sum() * 32 / 1e6), write_MB=float((m & wr).sum() * 32 / 1e6), tensors=names.count(n))
print(f"step {last}, {len(T)} tensors")
for n, v in out.items(): print(f"{n:<12} read {v['read_MB']:9.1f} MB  write {v['write_MB']:7.2f} MB  ({v['tensors']} tensors)")
print(f"total read {(~wr).sum()*32/1e6:.1f} MB")
if len(sys.argv) > 3: json.dump(out, open(sys.argv[3], "w"), indent=1)
