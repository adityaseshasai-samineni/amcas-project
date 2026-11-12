#!/usr/bin/env python3
"""Cross-model / cross-context transfer of ONE fixed bank-XOR matrix (transfer-owner, task 4).

Generalises heldout_remap.py (imports mapprep / region_remap / remap, edits none of them) from one train/test pair to
train-set and test-set lists. A hardware mapping is fixed for every workload, so matrices searched on one stream (or a
mix) are applied unchanged to other streams.

Usage: transfer_remap.py CONFIG.json OUT_DIR [--window N] [--fanin K]
CONFIG: {"streams": {NAME: [miss_gz, tensors_tsv]},
         "trains": {TRAIN: [STREAM, ...]},                   # matrix searched on the mix (equal windows) of these streams
         "tests":  {STREAM: {"drams": ["ddr5"], "placements": ["thp"]}}}   # which (placement, dram) combos to emit
Per test stream, placement, DRAM, Ramulator traces for:
  identity; naive (bank bit j ^= row bit j, the permutation-based interleaving of Zhang et al.); self (searched on the
  test window itself); x_TRAIN for every train set (the matrix searched on TRAIN); x_TRAIN_loo when the test stream is
  in TRAIN with len(TRAIN)>1 (the same mix without the test stream: no leakage); x_TRAIN is skipped (= self) when
  TRAIN == [test].
Matrices are over physical row bits of the placement, so they apply unchanged to streams from other processes (different
virtual addresses, different random THP frames). Each stream gets its own placement (drmiss_export.placements, seed 1,
AMCAS_CAPACITY_GB modelled DRAM). Bijection: only the bank field changes, as an XOR of row bits, so (a) row/col/channel
and the 2 MiB granule are asserted unchanged and (b) the number of distinct addresses in the window is asserted equal."""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapprep as P  # noqa: E402
import region_remap as RR  # noqa: E402
import remap as R  # noqa: E402

G = {}  # shared (forked) state: windows[stream][pk] = physical addresses of the stream's window


def chunks(x, nch_, size=100_000):
    if len(x) <= nch_ * size:
        return np.arange(len(x))
    return np.concatenate([np.arange(s, s + size) for s in np.linspace(0, len(x) - size, nch_).astype(int)])


def train_sample(L, streams, pk):
    """<= 800k requests: 8 chunks of 100k in total, split equally over the streams (RR.search then uses it as is)."""
    per = max(1, 8 // len(streams))
    ints, chs = [], []
    for s in streams:
        i, c = L.to_intra(G["win"][s][pk])
        ix = chunks(i, per)
        ints.append(i[ix]); chs.append(c[ix])
    return np.concatenate(ints), np.concatenate(chs)


def naive_matrix(L):
    return [[(f"row{j}", L.row_lo + j)] for j in range(L.bank)]


def search_task(t):
    key, pk, dk, streams = t
    L, nch = R.Layout(dk), P.NCH[dk]
    i, c = train_sample(L, streams, pk)
    Mn, sel, c0, c1 = RR.search(L, i, c, nch, G["fanin"], f"{pk}/{dk} {key}")
    return (key, pk, dk), (Mn, sel, c0, c1)


def emit_task(t):
    test, pk, dk, mats, out = t
    L, nch = R.Layout(dk), P.NCH[dk]
    pb = G["win"][test][pk]
    intra, ch = L.to_intra(pb)
    ob, orow, oc = L.fields(intra)
    nuniq = len(np.unique(pb))
    res = []
    for vk, Mn in mats.items():
        new = intra if Mn is None else RR.apply_named(L, intra, Mn)
        phys = L.from_intra(new, ch)
        nb, nr, nc = L.fields(new)
        assert (nr == orow).all() and (nc == oc).all(), "row/col changed"
        assert ((phys >> np.uint64(P.GRAN)) == (pb >> np.uint64(P.GRAN))).all(), "granule changed"
        assert (L.fields(new)[0] >= 0).all() and len(np.unique(phys)) == nuniq, "not a bijection"
        fn = f"{test}__{pk}_{dk}_{vk}.trc"
        P.write_trace(os.path.join(out, fn), phys)
        m = RR.metrics(L, new, ch, nch, np.zeros(len(pb), dtype=np.int8))
        m.update(test=test, placement=pk, dram=dk, scheme=vk, file=fn, bijective=True)
        res.append(m)
        print(f"  {test} {pk} {dk} {vk:<18} in-order hit {m['inorder_hit']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("out")
    ap.add_argument("--window", type=int, default=8_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--fanin", type=int, default=3)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    G["fanin"] = a.fanin
    need = set(cfg["tests"]) | {s for v in cfg["trains"].values() for s in v}
    pks = {pk for t in cfg["tests"].values() for pk in t["placements"]}
    G["win"] = {}
    man = dict(window=a.window, streams={}, matrices={}, files=[])
    for name in cfg["streams"]:
        if name not in need:
            continue
        miss, tsv = cfg["streams"][name]
        d = P.prep(miss, tsv)
        s0, s1 = P.window(d["n"], a.window, a.start_frac)
        G["win"][name] = {pk: d["place"][P.PLACE[pk]].astype(np.uint64)[s0:s1].copy() for pk in pks}
        man["streams"][name] = dict(reads=d["n"], window=[s0, s1])
        print(f"{name}: {d['n']:,} reads, window [{s0:,},{s1:,})", flush=True)
        del d
    combos = sorted({(pk, dk) for t in cfg["tests"].values() for pk in t["placements"] for dk in t["drams"]})
    tasks = []
    for pk, dk in combos:
        tests = [s for s, t in cfg["tests"].items() if pk in t["placements"] and dk in t["drams"]]
        for tr, ss in cfg["trains"].items():
            tasks.append((tr, pk, dk, ss))
        for s in tests:
            tasks.append((f"self:{s}", pk, dk, [s]))
            for tr, ss in cfg["trains"].items():
                if s in ss and len(ss) > 1:
                    tasks.append((f"loo:{tr}:{s}", pk, dk, [x for x in ss if x != s]))
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs) as pool:
        mats = dict(pool.map(search_task, tasks, chunksize=1))
        for (key, pk, dk), (Mn, sel, c0, c1) in mats.items():
            man["matrices"][f"{pk}/{dk}/{key}"] = dict(matrix=sel, cost=[c0, c1])
        etasks = []
        for pk, dk in combos:
            L = R.Layout(dk)
            for s, t in cfg["tests"].items():
                if pk not in t["placements"] or dk not in t["drams"]:
                    continue
                v = {"identity": None, "naive": naive_matrix(L), "self": mats[(f"self:{s}", pk, dk)][0]}
                for tr, ss in cfg["trains"].items():
                    if ss == [s]:
                        continue
                    v[f"x_{tr}"] = mats[(tr, pk, dk)][0]
                    if s in ss and len(ss) > 1:
                        v[f"x_{tr}_loo"] = mats[(f"loo:{tr}:{s}", pk, dk)][0]
                etasks.append((s, pk, dk, v, a.out))
        for r in pool.map(emit_task, etasks, chunksize=1):
            man["files"] += r
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
