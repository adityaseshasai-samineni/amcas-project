#!/usr/bin/env python3
"""Experts touched per layer over a multi-token MoE miss stream (transfer-owner, 4-token sample).
Usage: moe_experts_touched.py MISS_GZ TENSORS_TSV OUT_JSON [--tokens 4]
The stream is cut into --tokens equal read-count parts (token boundaries are not marked in the miss file; decode
tokens have near-equal read counts). Per part: distinct experts per layer (gate tensor), mean over layers; cumulative
union over parts; Jaccard overlap between consecutive parts, per layer averaged."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapprep as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("miss"); ap.add_argument("tensors"); ap.add_argument("out")
ap.add_argument("--tokens", type=int, default=4)
a = ap.parse_args()
d = P.prep(a.miss, a.tensors)
e, ly, kd = P.expert_features(d)
n = d["n"]
sel = (e >= 0) & (kd == 0)
bounds = np.linspace(0, n, a.tokens + 1).astype(int)
sets = []  # per part: dict layer -> set(experts)
for i in range(a.tokens):
    m = sel.copy(); m[:bounds[i]] = False; m[bounds[i + 1]:] = False
    s = {}
    for l, x in set(zip(ly[m].tolist(), e[m].tolist())):
        s.setdefault(l, set()).add(x)
    sets.append(s)
layers = sorted(set().union(*[set(s) for s in sets]))
res = dict(reads=n, parts=a.tokens, layers=len(layers), per_part=[], cumulative=[], jaccard_consecutive=[])
cum = {l: set() for l in layers}
for i, s in enumerate(sets):
    res["per_part"].append(float(np.mean([len(s.get(l, ())) for l in layers])))
    for l in layers:
        cum[l] |= s.get(l, set())
    res["cumulative"].append(float(np.mean([len(cum[l]) for l in layers])))
    if i:
        j = [len(sets[i - 1].get(l, set()) & s.get(l, set())) / max(1, len(sets[i - 1].get(l, set()) | s.get(l, set()))) for l in layers]
        res["jaccard_consecutive"].append(float(np.mean(j)))
print(json.dumps(res, indent=1))
json.dump(res, open(a.out, "w"), indent=1)
