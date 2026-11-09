#!/usr/bin/env python3
"""Replay policy_traces.py output through the AMCAS Ramulator copy (ramulator2-amcas), for each
row policy, saturated (LoadStoreTrace) and closed-loop (SimpleO3, 4 cores).

Usage: policy_replay.py TRACE_DIR OUT_JSON [--keys K1,K2] [--modes sat,o3] [--policies ...] [--procs N]
Needs PYTHONPATH=ramulator2-amcas/python and LD_LIBRARY_PATH=ramulator2-amcas.

sat: LoadStoreTrace, 4 requests per memory cycle (DDR5) / 1 (DDR4), as ramulator_replay.py.
o3 : SimpleO3 front end, one core per recorded thread, trace "bubbles addr" per miss, ipc 4,
     128-entry window, 16 MSHRs per core (the bound on outstanding misses), 4 KB 4-way front-end
     cache (practically a pass-through: the stream is already post-LLC), core clock = 3x
     (DDR4-2400, 3.6 GHz) or 1x (DDR5-4800, 2.4 GHz) the memory clock. Runs until every core has
     retired its own trace once; cores that finish first keep replaying (small perturbation).
     'cycles' = slowest core's finish time in memory cycles.
"""
import argparse, json, multiprocessing, os

POLS = {
    "Open": ("Open", {}),
    "ClosedCAP": ("ClosedCAP", {}),
    "Timeout": ("Timeout", {"timeout": 64}),
    "HAPPY": ("HAPPY", {"mode": "binary"}),
    "HAPPYcnt": ("HAPPY", {"mode": "count"}),
    "Region": ("Region", {"close_classes": "KO"}),
}


def run(cfg):
    import ramulator
    d, key, mode, pol, a = cfg
    dram_k = key.split("_")[1]
    nch = 1 if dram_k == "ddr4" else 16
    impl, kw = POLS[pol]
    if impl == "Region":
        kw = dict(kw, ranges=f"{d}/{key}.ranges")

    def ctrl():
        dram = (ramulator.dram.DDR4(org_preset="DDR4_8Gb_x8", timing_preset="DDR4_2400R", rank=2) if dram_k == "ddr4"
                else ramulator.dram.DDR5(org_preset="DDR5_16Gb_x8", timing_preset="DDR5_4800B", rank=1))
        return ramulator.controller.GenericDDR(
            dram=dram, scheduler=ramulator.scheduler.FRFCFS(), refresh_manager=ramulator.refresh_manager.AllBank(),
            row_policy=getattr(ramulator.row_policy, impl)(**kw), addr_mapper=ramulator.addr_mapper.RoBaRaCoCh())

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl() for _ in range(nch)],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=0 if nch == 1 else 2))
    if mode == "sat":
        fe = ramulator.frontend.LoadStoreTrace(clock_ratio=1 if nch == 1 else 4, path=f"{d}/{key}.trc")
    else:
        tr = [f"{d}/{key}.t{t}.trc" for t in range(4)]
        ns = [sum(1 for _ in open(t)) for t in tr]
        b = a.bubbles
        fe = ramulator.frontend.SimpleO3(
            clock_ratio=3 if nch == 1 else 1, num_expected_insts=min(ns) * (b + 1), traces=tr, ipc=4,
            inst_window_depth=128, llc_capacity_per_core="4KB", llc_associativity=4, llc_num_mshr_per_core=16,
            translation=ramulator.translation.NoTranslation(max_addr=1 << 40))
    sim = ramulator.Simulation(fe, mem)
    sim.run()
    cs = sim.stats["memory_system"]["controller"]
    cs = cs if isinstance(cs, list) else [cs]
    tot = lambda k: sum(c[k] for c in cs)
    n = max(tot("row_hits") + tot("row_misses") + tot("row_conflicts"), 1)
    served = max(sum(c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs), 1)
    cyc = max(c["cycles"] for c in cs)
    if mode == "o3":
        fs = sim.stats["frontend"]
        cyc = max(fs[f"cycles_recorded_core_{i}"] for i in range(4))
    return dict(key=key, mode=mode, policy=pol, reads=tot("num_read_reqs"), cycles=cyc, mem_cycles=max(c["cycles"] for c in cs),
                hit_rate=tot("row_hits") / n, conflict_rate=tot("row_conflicts") / n,
                avg_read_latency=sum(c["avg_read_latency"] * c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs) / served,
                throughput_MBps=sum(c.get("total_throughput_MBps", 0.0) for c in cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out")
    ap.add_argument("--keys", default="")
    ap.add_argument("--modes", default="sat,o3")
    ap.add_argument("--policies", default=",".join(POLS))
    ap.add_argument("--bubbles", type=int, default=40)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    keys = a.keys.split(",") if a.keys else sorted(f[:-4] for f in os.listdir(a.dir) if f.endswith(".trc") and ".t" not in f[-7:-4] and f.count(".") == 1)
    cfgs = [(a.dir, k, m, p, a) for k in keys for m in a.modes.split(",") for p in a.policies.split(",")]
    with multiprocessing.Pool(a.procs) as pool:
        res = []
        for r in pool.imap_unordered(run, cfgs):
            res.append(r); print(f"{r['key']:<28}{r['mode']:<4}{r['policy']:<10}hit {r['hit_rate']:.3f} lat {r['avg_read_latency']:7.1f} cyc {r['cycles']/1e6:8.2f}M {r['throughput_MBps']/1e3:7.1f}GB/s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
