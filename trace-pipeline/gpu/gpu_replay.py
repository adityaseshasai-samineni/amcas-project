#!/usr/bin/env python3
"""Replay a gpu_filter DRAM request stream (OUT.dram) through Ramulator 2 HBM2 (V100) or GDDR6 (A16).

usage: gpu_replay.py DRAM_PREFIX OUT_JSON --gpu {v100,a16} [--schemes identity,xor] [--policies Open,ClosedCAP]
                     [--window N] [--procs P] [--tmp DIR]
Needs PYTHONPATH=ramulator2-amcas/python and LD_LIBRARY_PATH=ramulator2-amcas; imports remap.py and
dramtech_cfg.py from ../tools (unchanged).

Assumptions (stated in context.md): the GPU virtual address is made 'physical' by packing 2 MiB pages in
first-touch order (cudaMalloc'd regions are 2 MiB aligned and device pages are 2 MiB); channels are
interleaved every 256 B (address bits [8, 8+W)); RoBaRaCoCh inside a channel; 32 B requests.
Saturated replay (LoadStoreTrace, no timestamps): time = DRAM-bound time of the request stream.
  v100: 32 HBM2 channels (4 stacks x 8 ch, 2 pseudo-channels each), HBM2_2Gb (512 MB/channel, 16 GB), 1.6 Gb/s
        (Ramulator has no 1.75 Gb/s bin; 819 GB/s vs the real 900).
  a16 : 8 GDDR6 16-bit channels = 128-bit, 2 GiB each, 14 Gb/s preset (real A16 12.5 Gb/s, 200 GB/s; model 224).
"""
import argparse, json, multiprocessing, os, sys
import numpy as np

TOOLS = os.path.join(os.environ.get("AMCAS_TOOLS", os.path.expanduser("~/amcas-pro/tools")))
sys.path.insert(0, TOOLS)
import remap as R
import dramtech_cfg as C

SYS = {
    "v100": dict(std="HBM2", org="HBM2_2Gb", timing="HBM2_1600Mbps", ctrl="HBM12", nch=32, ib=2, tx=32, layout="hbm2x32",
                 peak=819.2, tck_ps=1250, tick_mult=1, issue=32, kw={}, over={},
                 what="HBM2, 32 channels x 2 PC, 1.6 Gb/s, 16 GiB"),
    "a16": dict(std="GDDR6", org="GDDR6_32Gb_x16", timing="GDDR6_14000_1350mV_double", ctrl="GenericDDR", nch=8, ib=3, tx=32,
                layout="gddr6x8", peak=8 * 28.0, tck_ps=572, tick_mult=1, issue=8, kw={},
                over=dict(nRFCab=210, nRFCpb=105), what="GDDR6 14 Gb/s, 8 x16 channels (128-bit), 16 GiB"),
}
LAY = {"hbm2x32": dict(col=5, bank=5, row=14, off=5, s=8, W=5), "gddr6x8": dict(col=7, bank=4, row=15, off=5, s=8, W=3)}
R.LAYOUTS.update(LAY)
PAGE = 21


def pack_pages(p):
    """virtual byte address -> packed 'physical' address: 2 MiB pages numbered in first-touch order."""
    pg = p >> np.uint64(PAGE)
    u, first, inv = np.unique(pg, return_index=True, return_inverse=True)
    rank = np.empty(len(u), dtype=np.uint64)
    rank[np.argsort(first)] = np.arange(len(u), dtype=np.uint64)
    return (rank[inv] << np.uint64(PAGE)) | (p & np.uint64((1 << PAGE) - 1)), len(u)


def write_trace(path, st, addrs, chunk=1_000_000):
    with open(path, "w") as f:
        for i in range(0, len(addrs), chunk):
            s = st[i:i + chunk]; a = addrs[i:i + chunk]
            lines = np.where(s, "ST 0x", "LD 0x")
            f.write("".join(f"{l}{x:x}\n" for l, x in zip(lines.tolist(), a.tolist())))


def replay(task):
    import ramulator
    sd, path, scheme, pol = task
    mem = C.build_mem(sd, pol)
    fe = ramulator.frontend.LoadStoreTrace(clock_ratio=sd["issue"], path=path)
    sim = ramulator.Simulation(fe, mem); sim.run()
    cs = sim.stats["memory_system"]["controller"]; cs = cs if isinstance(cs, list) else [cs]
    tot = lambda k: sum(c[k] for c in cs)
    n = max(tot("row_hits") + tot("row_misses") + tot("row_conflicts"), 1)
    served = max(sum(c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs), 1)
    mcyc = max(c["cycles"] for c in cs); tick = C.tick_ps(sd)
    lat = sum(c["avg_read_latency"] * c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs if c["num_read_reqs"] > 0) / served
    nreq = tot("num_read_reqs") + tot("num_write_reqs")
    return dict(scheme=scheme, policy=pol, reads=tot("num_read_reqs"), writes=tot("num_write_reqs"), cycles=mcyc,
                time_us=mcyc * tick / 1e6, GBps=nreq * sd["tx"] / (mcyc * tick / 1e3), hit_rate=tot("row_hits") / n,
                conflict_rate=tot("row_conflicts") / n, avg_read_latency_ns=lat * tick / 1e3,
                chan_reads=[c["num_read_reqs"] for c in cs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix"); ap.add_argument("out")
    ap.add_argument("--gpu", required=True, choices=list(SYS))
    ap.add_argument("--schemes", default="identity,xor"); ap.add_argument("--policies", default="Open,ClosedCAP")
    ap.add_argument("--window", type=int, default=0); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tmp", default="/tmp"); ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    sd = SYS[a.gpu]; L = R.Layout(sd["layout"])
    arr = np.fromfile(a.prefix + ".dram", dtype=np.uint64)
    if a.window: arr = arr[:a.window]
    st = (arr & np.uint64(1)).astype(bool); byte = (arr >> np.uint64(1)) << np.uint64(5)
    phys, npg = pack_pages(byte)
    print(f"{a.gpu}: {len(arr)} requests ({(~st).sum()} rd, {st.sum()} wr), {npg} 2MiB pages = {npg*2} MiB footprint", flush=True)
    bus, tmiss = C.cost_params(sd); R.cost.__defaults__ = (bus, tmiss)
    intra, ch = L.to_intra(phys)
    tasks, info = [], {}
    os.makedirs(a.tmp, exist_ok=True)
    for sc in a.schemes.split(","):
        if sc == "identity":
            new = intra
        else:
            cands = R.sources(L, False)
            M = R.search(L, R.sample(intra), cands, a.fanin)
            new = R.apply_matrix(L, intra, M, cands)
            info["xor_matrix"] = {j: [cands[c][0] for c in M[j]] for j in range(L.bank)}
        b, r, _ = L.fields(new)
        info[sc + "_inorder_hit"] = R.inorder_hits(b, r) / len(b)
        path = f"{a.tmp}/gpu_{os.getpid()}_{sc}.trc"
        write_trace(path, st, L.from_intra(new, ch))
        for pol in a.policies.split(","):
            tasks.append((sd, path, sc, pol))
    with multiprocessing.Pool(min(a.procs, len(tasks))) as pool:
        res = pool.map(replay, tasks)
    for t in set(t[1] for t in tasks): os.remove(t)
    json.dump(dict(system=sd["what"], peak_GBps=sd["peak"], requests=len(arr), pages_2MiB=npg, info=info, results=res), open(a.out, "w"), indent=1)
    print(sd["what"], "peak", sd["peak"], "GB/s", info)
    print(f"{'scheme':<9}{'policy':<11}{'reads':>10}{'writes':>8}{'hit':>7}{'conf':>7}{'lat_ns':>8}{'time_us':>10}{'GB/s':>8}")
    for r in res:
        print(f"{r['scheme']:<9}{r['policy']:<11}{r['reads']:>10}{r['writes']:>8}{r['hit_rate']:>7.3f}{r['conflict_rate']:>7.3f}"
              f"{r['avg_read_latency_ns']:>8.1f}{r['time_us']:>10.1f}{r['GBps']:>8.1f}")


if __name__ == "__main__":
    main()
