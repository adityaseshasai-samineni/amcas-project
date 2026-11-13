#!/usr/bin/env python3
"""Ramulator replay of mop4_traces.py output with a mapper switch (mop-owner).

Usage: mop4_replay.py TRACE_DIR OUT_JSON [--drams x16,x8,d4r6,d4] [--placements ...] [--points sat,k1,w1024,k4]
                      [--policies Open,ClosedCAP] [--schemes ident,xor_global,mop,mopxor,chra] [--procs N] [--tmp DIR]

Scheme -> (stream variant, Ramulator mapper): ident=(id,RoBaRaCoCh) xor_global=(xg,RoBaRaCoCh) mop=(id,MOP4CLXOR)
mopxor=(mxg,MOP4CLXOR) chra=(id,ChRaBaRoCo).
DRAM (as the owners' harnesses): x16/x8 DDR5-4800B 16Gb x8 1 rank, 16/8 controllers, 256 B channel interleave, sat issue 4/2,
closed-loop core clock = memory clock (policy_replay.py); d4r6 DDR4-2933Y 16Gb x8 2 ranks, 8 controller slots (6 used), issue 2,
core = 2x memory clock (realmem_replay.py); d4 DDR4-2400R 8Gb x8 2 ranks, 1 channel, issue 1, core 3x memory clock (policy_replay.py).
Points: sat = LoadStoreTrace; closed loop = SimpleO3 4 cores ipc 4 MSHR 16: k1 = bubbles 52 window 256 (fast core, ~22 GB/s),
w1024 = bubbles 52 window 1024 (~40 GB/s plateau), k4 = bubbles 211 window 1024 (native 4-thread 32k, ~10 GB/s).
Needs PYTHONPATH=ramulator2-amcas/python and LD_LIBRARY_PATH=ramulator2-amcas (+ env lib).
"""
import argparse, json, multiprocessing, os, shutil, tempfile

SCHEMES = {"ident": ("id", "RoBaRaCoCh"), "xor_global": ("xg", "RoBaRaCoCh"), "mop": ("id", "MOP4CLXOR"),
           "mopxor": ("mxg", "MOP4CLXOR"), "chra": ("id", "ChRaBaRoCo")}
POINTS = {"k1": (52, 256), "w1024": (52, 1024), "k4": (211, 1024)}
NCTRL = {"x16": 16, "x8": 8, "d4r6": 8, "d4": 1}
ISSUE = {"x16": 4, "x8": 2, "d4r6": 2, "d4": 1}
CORE = {"x16": 1, "x8": 1, "d4r6": 2, "d4": 3}


def run(cfg):
    import ramulator
    d, pk, dk, sch, point, pol, tmpdir = cfg
    var, mapper = SCHEMES[sch]
    key = f"{pk}_{dk}_{var}"
    n = NCTRL[dk]

    def ctrl():
        if dk in ("x16", "x8"):
            dram = ramulator.dram.DDR5(org_preset="DDR5_16Gb_x8", timing_preset="DDR5_4800B", rank=1)
        elif dk == "d4r6":
            dram = ramulator.dram.DDR4(org_preset="DDR4_16Gb_x8", timing_preset="DDR4_2933Y", rank=2)
        else:
            dram = ramulator.dram.DDR4(org_preset="DDR4_8Gb_x8", timing_preset="DDR4_2400R", rank=2)
        return ramulator.controller.GenericDDR(dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
                                               refresh_manager=ramulator.refresh_manager.AllBank(),
                                               row_policy=getattr(ramulator.row_policy, pol)(),
                                               addr_mapper=getattr(ramulator.addr_mapper, mapper)())

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl() for _ in range(n)],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=0 if n == 1 else 2))
    tmp = None
    if point == "sat":
        fe = ramulator.frontend.LoadStoreTrace(clock_ratio=ISSUE[dk], path=f"{d}/{key}.trc")
    else:
        bub, win = POINTS[point]
        tmp = tempfile.mkdtemp(dir=tmpdir)
        tr = []
        for t in range(4):
            p = f"{tmp}/t{t}.trc"
            with open(p, "w") as f, open(f"{d}/{key}.t{t}.addr") as g:
                f.writelines(f"{bub} {x}" for x in g)
            tr.append(p)
        ns = [sum(1 for _ in open(t)) for t in tr]
        fe = ramulator.frontend.SimpleO3(clock_ratio=CORE[dk], num_expected_insts=min(ns) * (bub + 1), traces=tr, ipc=4,
                                         inst_window_depth=win, llc_capacity_per_core="4KB", llc_associativity=4,
                                         llc_num_mshr_per_core=16, translation=ramulator.translation.NoTranslation(max_addr=1 << 40))
    sim = ramulator.Simulation(fe, mem)
    sim.run()
    cs = sim.stats["memory_system"]["controller"]
    cs = cs if isinstance(cs, list) else [cs]
    tot = lambda k: sum(c[k] for c in cs)
    nn = max(tot("row_hits") + tot("row_misses") + tot("row_conflicts"), 1)
    served = max(sum(c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs), 1)
    cyc = mcyc = max(c["cycles"] for c in cs)
    if point != "sat":
        fs = sim.stats["frontend"]
        cyc = max(fs[f"cycles_recorded_core_{i}"] for i in range(4))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return dict(placement=pk, dram=dk, scheme=sch, mapper=mapper, point=point, policy=pol, reads=tot("num_read_reqs"),
                cycles=cyc, mem_cycles=mcyc, hit_rate=tot("row_hits") / nn, conflict_rate=tot("row_conflicts") / nn,
                avg_read_latency=sum(c["avg_read_latency"] * c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs if c["num_read_reqs"] > 0) / served,
                throughput_MBps=sum(c.get("total_throughput_MBps", 0.0) for c in cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out")
    ap.add_argument("--drams", default="x16,x8,d4r6,d4")
    ap.add_argument("--placements", default="packed,thp,4k")
    ap.add_argument("--points", default="sat,k1,w1024,k4")
    ap.add_argument("--d4-points", default="sat,k4", help="points used for the 1-channel DDR4 (peak 19 GB/s: only k4 is below it)")
    ap.add_argument("--policies", default="Open,ClosedCAP")
    ap.add_argument("--schemes", default="ident,xor_global,mop,mopxor,chra")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tmp", default="/scratch")
    a = ap.parse_args()
    cfgs = []
    for pt in a.points.split(","):  # sat first: they are the longest; keep ordering by point then others
        for pk in a.placements.split(","):
            for dk in a.drams.split(","):
                if dk == "d4" and pt not in a.d4_points.split(","):
                    continue
                for sch in a.schemes.split(","):
                    for pol in a.policies.split(","):
                        cfgs.append((a.dir, pk, dk, sch, pt, pol, a.tmp))
    print(f"{len(cfgs)} replays", flush=True)
    res = []
    with multiprocessing.Pool(a.procs) as pool:
        for r in pool.imap_unordered(run, cfgs):
            res.append(r)
            print(f"{r['placement']:<7}{r['dram']:<5}{r['scheme']:<11}{r['point']:<6}{r['policy']:<10}hit {r['hit_rate']:.3f} "
                  f"lat {r['avg_read_latency']:7.1f} cyc {r['cycles']/1e6:8.3f}M {r['throughput_MBps']/1e3:7.1f}GB/s", flush=True)
            if len(res) % 10 == 0:
                json.dump(res, open(a.out, "w"))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
