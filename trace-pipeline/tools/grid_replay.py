#!/usr/bin/env python3
"""Ramulator replay of grid_gen.py output for every mapping x row policy (grid-owner).

Usage: grid_replay.py TRACE_DIR OUT_JSON [--drams d4r6,x16] [--placements packed,thp] [--mappings ...] [--policies ...]
                      [--procs N] [--tmp DIR]
Needs PYTHONPATH=ramulator2-grid/python and LD_LIBRARY_PATH=ramulator2-grid (+ env lib).

DRAM and frontend exactly as mop4_replay.py's saturated point: x16 = DDR5-4800B 16Gb x8 1 rank, 16 controllers, issue 4/cycle;
d4r6 = DDR4-2933Y 16Gb x8 2 ranks, 8 controller slots (6 used), issue 2/cycle (LoadStoreTrace, no timestamps).
Row policies: Open, ClosedCAP, Timeout16/64/256 (idle cycles), Region (K and O granules ClosedCAP, W open), HAPPY (binary),
HAPPYcnt, Adaptive (4-bit per-bank counter, 15/6) and Adaptive7_4 (3-bit, 7/4), Oracle (future-aware: pass 1 'OracleRec' = Open + logging of every bank's row
sequence, pass 2 'Oracle' consumes it; replays of one stream/mapper always run record first).
"""
import argparse, json, multiprocessing, os, shutil, sys

MAPS = {"identity": ("id", "RoBaRaCoCh"), "xor_global": ("xg", "RoBaRaCoCh"), "region": ("region", "RoBaRaCoCh"),
        "chan": ("chan", "RoBaRaCoCh"), "xor_global_chan": ("xgc", "RoBaRaCoCh"), "mop": ("id", "MOP4CLXOR"),
        "mopxor": ("mxg", "MOP4CLXOR"), "bitrev": ("bitrev", "RoBaRaCoCh"), "chra": ("id", "ChRaBaRoCo"),
        "naive": ("naive", "RoBaRaCoCh"), "expert_hash": ("eh", "RoBaRaCoCh"), "expert_lin": ("el", "RoBaRaCoCh")}
POLS = ["Open", "ClosedCAP", "Timeout16", "Timeout64", "Timeout256", "Region", "HAPPY", "HAPPYcnt", "Adaptive", "Adaptive7_4", "OracleRec", "Oracle"]
NCTRL = {"x16": 16, "d4r6": 8}
ISSUE = {"x16": 4, "d4r6": 2}


def policy(ramulator, pol, d, key, orc):
    rp = ramulator.row_policy
    if pol in ("Open", "ClosedCAP"):
        return getattr(rp, pol)()
    if pol.startswith("Adaptive"):  # Adaptive = 4-bit counter (0..15), keep open while >= 6, starts at 15; AdaptiveMAX_THRESH other settings
        mx, th = (int(x) for x in pol[8:].split("_")) if len(pol) > 8 else (15, 6)
        return rp.Adaptive(max=mx, thresh=th, init=mx)
    if pol.startswith("Timeout"):
        return rp.Timeout(timeout=int(pol[7:]))
    if pol == "Region":
        return rp.Region(ranges=f"{d}/{key}.ranges", close_classes="KO")
    if pol == "HAPPY":
        return rp.HAPPY(mode="binary")
    if pol == "HAPPYcnt":
        return rp.HAPPY(mode="count")
    if pol == "OracleRec":
        return rp.Oracle(mode="record", path=orc)
    if pol.startswith("Oracle") and pol != "OracleRec":  # Oracle (horizon 16), OracleH1 / OracleH4 / OracleH64 ...
        h = int(pol[7:]) if pol.startswith("OracleH") else 16
        return rp.Oracle(mode="oracle", path=orc, trace=f"{d}/{key}.trc", horizon=h)
    raise ValueError(pol)


def run(cfg):
    import ramulator
    d, pk, dk, mp, pol, tmpdir = cfg
    stream, mapper = MAPS[mp]
    key = f"{pk}_{dk}_{stream}"
    orc = f"{tmpdir}/{pk}_{dk}_{mp}.seq"
    n = NCTRL[dk]

    def ctrl():
        if dk == "x16":
            dram = ramulator.dram.DDR5(org_preset="DDR5_16Gb_x8", timing_preset="DDR5_4800B", rank=1)
        else:
            dram = ramulator.dram.DDR4(org_preset="DDR4_16Gb_x8", timing_preset="DDR4_2933Y", rank=2)
        return ramulator.controller.GenericDDR(dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
                                               refresh_manager=ramulator.refresh_manager.AllBank(),
                                               row_policy=policy(ramulator, pol, d, key, orc),
                                               addr_mapper=getattr(ramulator.addr_mapper, mapper)())

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl() for _ in range(n)],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=2))
    fe = ramulator.frontend.LoadStoreTrace(clock_ratio=ISSUE[dk], path=f"{d}/{key}.trc")
    sim = ramulator.Simulation(fe, mem)
    sim.run()
    cs = sim.stats["memory_system"]["controller"]
    cs = cs if isinstance(cs, list) else [cs]
    tot = lambda k: sum(c[k] for c in cs)
    nn = max(tot("row_hits") + tot("row_misses") + tot("row_conflicts"), 1)
    served = max(sum(c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs), 1)
    cyc = max(c["cycles"] for c in cs)
    return dict(placement=pk, dram=dk, mapping=mp, mapper=mapper, policy=pol, reads=tot("num_read_reqs"), cycles=cyc,
                hit_rate=tot("row_hits") / nn, miss_rate=tot("row_misses") / nn, conflict_rate=tot("row_conflicts") / nn,
                avg_read_latency=sum(c["avg_read_latency"] * c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs if c["num_read_reqs"] > 0) / served,
                throughput_MBps=sum(c.get("total_throughput_MBps", 0.0) for c in cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out")
    ap.add_argument("--drams", default="d4r6,x16")
    ap.add_argument("--placements", default="packed,thp")
    ap.add_argument("--mappings", default=",".join(MAPS))
    ap.add_argument("--policies", default=",".join(POLS))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tmp", default="/scratch")
    a = ap.parse_args()
    avail = lambda pk, dk, mp: os.path.exists(f"{a.dir}/{pk}_{dk}_{MAPS[mp][0]}.trc")
    pols = a.policies.split(",")
    pass1 = [p for p in pols if p == "OracleRec"]
    pass2 = [p for p in pols if p != "OracleRec"]
    if any(p.startswith("Oracle") for p in pass2) and not pass1:
        pass1 = ["OracleRec"]
    base = [(pk, dk, mp) for dk in a.drams.split(",") for pk in a.placements.split(",") for mp in a.mappings.split(",") if avail(pk, dk, mp)]
    res = []
    with multiprocessing.get_context("spawn").Pool(a.procs) as pool:
        for phase, ps in (("rec", pass1), ("main", pass2)):
            cfgs = [(a.dir, pk, dk, mp, p, a.tmp) for (pk, dk, mp) in base for p in ps]
            print(f"{phase}: {len(cfgs)} replays", flush=True)
            for r in pool.imap_unordered(run, cfgs):
                res.append(r)
                print(f"{r['placement']:<7}{r['dram']:<5}{r['mapping']:<15}{r['policy']:<11}hit {r['hit_rate']:.3f} "
                      f"lat {r['avg_read_latency']:7.1f} cyc {r['cycles']/1e6:8.3f}M", flush=True)
                if len(res) % 20 == 0:
                    json.dump(res, open(a.out, "w"))
    json.dump(res, open(a.out, "w"), indent=1)
    for f in os.listdir(a.tmp):
        if ".seq." in f:
            os.remove(f"{a.tmp}/{f}")


if __name__ == "__main__":
    main()
