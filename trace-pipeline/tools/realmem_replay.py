#!/usr/bin/env python3
"""Ramulator replay of realmem_traces.py output on the cluster-node DDR4 model (validation-owner).

Usage: realmem_replay.py TRACE_DIR OUT_JSON --nch N [--policies Open,ClosedCAP] [--points sat,fast:52:256,...] [--keys ...]
DRAM: DDR4-2933Y (21-21-21), 16Gb x8, 2 ranks, per channel (32 GB DIMM); N used channels in a power-of-two
slot count (idle controllers get no requests). Point syntax: 'sat' = LoadStoreTrace, issue ceil(N*1.5/2) per
cycle... (N=1: 1, 6: 2, 12: 3, aggregate peak 1.5 req/cycle/4 channels); 'name:BUBBLES:WINDOW' = SimpleO3, 4 cores, ipc 4, MSHR 16,
core clock = 2 x memory clock (2.93 GHz), bubbles = non-memory instructions per DRAM read, WINDOW = instruction window.
Needs PYTHONPATH=ramulator2-amcas/python and LD_LIBRARY_PATH=ramulator2-amcas.
"""
import argparse, json, multiprocessing, os, tempfile

SLOTS = lambda n: 1 if n == 1 else 1 << (n - 1).bit_length()


def run(cfg):
    import ramulator
    d, key, point, pol, a = cfg
    nslot = SLOTS(a.nch)

    def ctrl():
        dram = ramulator.dram.DDR4(org_preset="DDR4_16Gb_x8", timing_preset="DDR4_2933Y", rank=2)
        return ramulator.controller.GenericDDR(dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
                                               refresh_manager=ramulator.refresh_manager.AllBank(),
                                               row_policy=getattr(ramulator.row_policy, pol)(),
                                               addr_mapper=ramulator.addr_mapper.RoBaRaCoCh())

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl() for _ in range(nslot)],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=0 if nslot == 1 else 2))
    tmp = None
    if point == "sat":
        fe = ramulator.frontend.LoadStoreTrace(clock_ratio=max(1, (a.nch + 3) // 4), path=f"{d}/{key}.trc")
    else:
        name, bub, win = point.split(":"); bub, win = int(bub), int(win)
        tmp = tempfile.mkdtemp(dir=a.tmp)
        tr = []
        for t in range(4):
            p = f"{tmp}/t{t}.trc"
            with open(p, "w") as f, open(f"{d}/{key}.t{t}.addr") as g:
                f.writelines(f"{bub} {x}" for x in g)
            tr.append(p)
        ns = [sum(1 for _ in open(t)) for t in tr]
        fe = ramulator.frontend.SimpleO3(clock_ratio=2, num_expected_insts=min(ns) * (bub + 1), traces=tr, ipc=4,
                                         inst_window_depth=win, llc_capacity_per_core="4KB", llc_associativity=4,
                                         llc_num_mshr_per_core=16, translation=ramulator.translation.NoTranslation(max_addr=1 << 40))
    sim = ramulator.Simulation(fe, mem)
    sim.run()
    cs = sim.stats["memory_system"]["controller"]
    cs = cs if isinstance(cs, list) else [cs]
    tot = lambda k: sum(c[k] for c in cs)
    n = max(tot("row_hits") + tot("row_misses") + tot("row_conflicts"), 1)
    served = max(sum(c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs), 1)
    cyc = max(c["cycles"] for c in cs)
    mcyc = cyc
    if point != "sat":
        fs = sim.stats["frontend"]
        cyc = max(fs[f"cycles_recorded_core_{i}"] for i in range(4))
    if tmp:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return dict(key=key, point=point, policy=pol, nch=a.nch, reads=tot("num_read_reqs"), cycles=cyc, mem_cycles=mcyc,
                hit_rate=tot("row_hits") / n, conflict_rate=tot("row_conflicts") / n,
                avg_read_latency=sum(c["avg_read_latency"] * c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs if c["num_read_reqs"] > 0) / served,
                throughput_MBps=sum(c.get("total_throughput_MBps", 0.0) for c in cs),
                channel_reads=[c["num_read_reqs"] for c in cs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out")
    ap.add_argument("--nch", type=int, required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--policies", default="Open,ClosedCAP")
    ap.add_argument("--points", default="sat")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tmp", default="/scratch")
    a = ap.parse_args()
    keys = a.keys.split(",") if a.keys else sorted(f[:-4] for f in os.listdir(a.dir) if f.endswith(".trc"))
    cfgs = [(a.dir, k, pt, p, a) for k in keys for pt in a.points.split(",") for p in a.policies.split(",")]
    res = []
    with multiprocessing.Pool(a.procs) as pool:
        for r in pool.imap_unordered(run, cfgs):
            res.append(r)
            print(f"{r['key']:<18}{r['point']:<12}{r['policy']:<10}hit {r['hit_rate']:.3f} lat {r['avg_read_latency']:7.1f} "
                  f"cyc {r['cycles']/1e6:8.3f}M {r['throughput_MBps']/1e3:7.1f}GB/s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
