#!/usr/bin/env python3
"""Fragmentation sensitivity of the bank-XOR gain (transfer-owner, task 5).

Usage: frag_remap.py MISS_GZ TENSORS_TSV OUT_DIR [--window N] [--cap-gb 32] [--grid c:L,...] [--drams ddr5]
Placement model (granule = 2 MiB): the stream's distinct VA granules, in VA order, are cut into runs of L consecutive
granules. A fraction c of the runs (chosen at random) is placed CONTIGUOUSLY: the run takes L consecutive physical frames
inside a randomly chosen slot of the modelled DRAM (slots are L frames, aligned; buddy-allocator style), in VA order. All
other granules go to random distinct free frames (the aged-allocator case = the 'thp' placement of drmiss_export).
c=0 is fully random, c=1 with huge L approaches 'packed'. 'packed' (all VA-order, one run) is added as the reference.
Per placement x DRAM writes Ramulator traces: identity, xor_c0 (the matrix searched once on the c=0 placement of this
stream and applied unchanged), xor_self (searched on this placement), same bijection asserts as transfer_remap."""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drmiss_export as X  # noqa: E402
import mapprep as P  # noqa: E402
import region_remap as RR  # noqa: E402
import remap as R  # noqa: E402
from analyze import LINE  # noqa: E402

HUGE = 2 << 20
G = {}


def frag_place(va, c, L, cap_gb, seed=1):
    """Physical addresses for va under (contiguity c, run length L granules); L=0: packed in VA order."""
    rng = np.random.default_rng(seed)
    gr, inv = np.unique(va // HUGE, return_inverse=True)
    n, nfr = len(gr), (cap_gb << 30) // HUGE
    if n > nfr:
        sys.exit("footprint exceeds modelled DRAM")
    if L == 0:
        frames = np.arange(n, dtype=np.int64)
    else:
        nruns = (n + L - 1) // L
        k = int(round(c * nruns))
        contig = rng.choice(nruns, size=k, replace=False) if k else np.zeros(0, dtype=np.int64)
        nslots = nfr // L
        frames = np.full(n, -1, dtype=np.int64)
        used = np.zeros(nfr, dtype=bool)
        if k:
            if k > nslots:
                sys.exit(f"{k} contiguous runs of {L} granules do not fit {nslots} slots")
            slots = rng.choice(nslots, size=k, replace=False)
            for r, s in zip(contig.tolist(), slots.tolist()):
                lo, hi = r * L, min(n, (r + 1) * L)
                frames[lo:hi] = s * L + np.arange(hi - lo)
                used[s * L:s * L + (hi - lo)] = True
        rest = np.flatnonzero(frames < 0)
        free = np.flatnonzero(~used)
        frames[rest] = rng.choice(free, size=len(rest), replace=False)
    assert len(np.unique(frames)) == n
    return (frames[inv] * HUGE + va % HUGE).astype(np.uint64)


def search_task(t):
    key, dk = t
    L, nch = R.Layout(dk), P.NCH[dk]
    i, c = L.to_intra(G["win"][key])
    Mn, sel, c0, c1 = RR.search(L, i, c, nch, G["fanin"], f"{dk} {key}")
    return (key, dk), (Mn, sel, c0, c1)


def emit_task(t):
    key, dk, mats, out = t
    L, nch = R.Layout(dk), P.NCH[dk]
    pb = G["win"][key]
    intra, ch = L.to_intra(pb)
    _b, orow, oc = L.fields(intra)
    nuniq = len(np.unique(pb))
    res = []
    for vk, Mn in mats.items():
        new = intra if Mn is None else RR.apply_named(L, intra, Mn)
        phys = L.from_intra(new, ch)
        _b, nr, nc = L.fields(new)
        assert (nr == orow).all() and (nc == oc).all() and ((phys >> np.uint64(21)) == (pb >> np.uint64(21))).all()
        assert len(np.unique(phys)) == nuniq, "not a bijection"
        fn = f"{key}_{dk}_{vk}.trc"
        P.write_trace(os.path.join(out, fn), phys)
        m = RR.metrics(L, new, ch, nch, np.zeros(len(pb), dtype=np.int8))
        m.update(place=key, dram=dk, scheme=vk, file=fn)
        res.append(m)
        print(f"  {key} {dk} {vk:<10} in-order hit {m['inorder_hit']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss"); ap.add_argument("tensors"); ap.add_argument("out")
    ap.add_argument("--window", type=int, default=8_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--cap-gb", type=int, default=32)
    ap.add_argument("--grid", default="0:1,0.25:8,0.25:64,0.25:512,0.5:8,0.5:64,0.5:512,0.75:8,0.75:64,0.75:512,"
                    "0.9:8,0.9:64,0.9:512,1:8,1:64,1:512")
    ap.add_argument("--drams", default="ddr5")
    ap.add_argument("--fanin", type=int, default=3)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G["fanin"] = a.fanin
    _t, addrs = X.read_misses(a.miss)
    va = (addrs // LINE * LINE).astype(np.int64)
    del addrs
    n = len(va)
    s0, s1 = P.window(n, a.window, a.start_frac)
    ngr = len(np.unique(va // HUGE))
    print(f"{n:,} reads, window [{s0:,},{s1:,}); {ngr} distinct 2 MiB granules of {(a.cap_gb << 30) // HUGE}", flush=True)
    keys = []
    for spec in a.grid.split(","):
        c, L = spec.split(":")
        c, L = float(c), int(L)
        key = f"c{c:g}_L{L}"
        G.setdefault("win", {})[key] = frag_place(va, c, L, a.cap_gb)[s0:s1].copy()
        keys.append(key)
    G["win"]["packed"] = frag_place(va, 0, 0, a.cap_gb)[s0:s1].copy()
    keys.append("packed")
    base = "c0_L1"
    assert base in keys
    drams = a.drams.split(",")
    man = dict(reads=n, window=[s0, s1], granules=ngr, keys=keys, matrices={}, files=[])
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs) as pool:
        mats = dict(pool.map(search_task, [(k, d) for k in keys for d in drams], chunksize=1))
        for (k, d), (_M, sel, c0, c1) in mats.items():
            man["matrices"][f"{k}/{d}"] = dict(matrix=sel, cost=[c0, c1])
        et = [(k, d, {"identity": None, "xor_c0": mats[(base, d)][0], "xor_self": mats[(k, d)][0]}, a.out)
              for k in keys for d in drams]
        for r in pool.map(emit_task, et, chunksize=1):
            man["files"] += r
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
