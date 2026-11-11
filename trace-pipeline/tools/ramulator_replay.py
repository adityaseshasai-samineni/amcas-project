#!/usr/bin/env python3
"""Replay a LoadStoreTrace through Ramulator 2.1 for each mapping and page policy.

Usage: ramulator_replay.py TRACE OUT_JSON [--dram ddr4|ddr5] [--channels N]
                           [--interleave-bits B] [--issue K] [--procs P]

LoadStoreTrace has no timestamps: it offers up to K requests per memory cycle and
blocks when the controllers are full. The stream is therefore saturated, unlike the
timed DRAMsim3 replay, and latency here is not comparable with it.

  ddr4 (default): 2 ranks of DDR4-2400 8Gb x8, one channel, the same DRAM as the
       DRAMsim3 config and every earlier result. 19.2 GB/s peak.
  ddr5: DDR5-4800B 16Gb x8, one rank per 32-bit subchannel; N controllers, one per
       subchannel. 8 DIMM channels = 16 subchannels, about 307 GB/s peak, the
       planned full-scale target. One request per cycle (64 B / 416 ps, about
       154 GB/s) would be the bottleneck, so the front end issues K per cycle.
Channels are interleaved every 64 B << B (B=2: 256 B, B=6: 4 KiB).
"""

import argparse
import json
import multiprocessing

MAPPERS = ["RoBaRaCoCh", "MOP4CLXOR", "ChRaBaRoCo"]
POLICIES = ["Open", "ClosedCAP"]


def run(cfg):
    import ramulator

    trace, mapper, policy, a = cfg

    def ctrl():
        if a.dram == "ddr4":
            dram = ramulator.dram.DDR4(org_preset="DDR4_8Gb_x8", timing_preset="DDR4_2400R", rank=2)
        else:
            dram = ramulator.dram.DDR5(org_preset="DDR5_16Gb_x8", timing_preset="DDR5_4800B", rank=1)
        return ramulator.controller.GenericDDR(
            dram=dram,
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.AllBank(),
            row_policy=getattr(ramulator.row_policy, policy)(),
            addr_mapper=getattr(ramulator.addr_mapper, mapper)(),
        )

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl() for _ in range(a.channels)],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=a.interleave_bits),
    )
    sim = ramulator.Simulation(ramulator.frontend.LoadStoreTrace(clock_ratio=a.issue, path=trace), mem)
    sim.run()
    cs = sim.stats["memory_system"]["controller"]
    cs = cs if isinstance(cs, list) else [cs]
    tot = lambda k: sum(c[k] for c in cs)
    n = tot("row_hits") + tot("row_misses") + tot("row_conflicts")
    served = sum(c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs)
    return dict(
        mapper=mapper,
        policy=policy,
        dram=a.dram,
        channels=a.channels,
        interleave_bits=a.interleave_bits,
        reads=tot("num_read_reqs"),
        writes=tot("num_write_reqs"),
        cycles=max(c["cycles"] for c in cs),
        row_hits=tot("row_hits"),
        row_misses=tot("row_misses"),
        row_conflicts=tot("row_conflicts"),
        hit_rate=tot("row_hits") / max(n, 1),
        avg_read_latency=sum(c["avg_read_latency"] * c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs)
        / max(served, 1),
        throughput_MBps=sum(c.get("total_throughput_MBps", 0.0) for c in cs),
        channel_reads=[c["num_read_reqs"] for c in cs],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--dram", choices=["ddr4", "ddr5"], default="ddr4")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--interleave-bits", type=int, default=0)
    ap.add_argument("--issue", type=int, default=1, help="requests the front end offers per memory cycle")
    ap.add_argument("--preset", choices=["ddr5x16", "ddr5x8"],
                    help="shorthand for --dram ddr5 with channels/issue: ddr5x16 = 16 subchannels (8 DIMM channels, 307 GB/s), "
                         "interleave 256 B, issue 4; ddr5x8 = 8 subchannels (4 DIMM channels, 154 GB/s), interleave 256 B, issue 2. "
                         "Both DDR5-4800B 16Gb x8.")
    ap.add_argument("--procs", type=int, default=6, help="configurations run at once; each holds the whole trace")
    ap.add_argument("--mappers", default=",".join(MAPPERS), help="comma list; RoBaRaCoCh only for pre-transformed (remap.py) traces")
    a = ap.parse_args()
    if a.preset:
        a.dram, a.interleave_bits = "ddr5", 2
        a.channels, a.issue = (16, 4) if a.preset == "ddr5x16" else (8, 2)
    cfgs = [(a.trace, m, p, a) for m in a.mappers.split(",") for p in POLICIES]
    with multiprocessing.Pool(min(a.procs, len(cfgs))) as pool:
        res = pool.map(run, cfgs)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{a.dram}, {a.channels} channel(s), interleave 64 B << {a.interleave_bits}, issue {a.issue}/cycle")
    print(
        f"{'mapper':<12}{'policy':<11}{'reads':>10}{'writes':>9}{'hit':>7}{'conflict':>10}"
        f"{'rd_lat':>8}{'Mcycles':>9}{'GB/s':>8}"
    )
    for r in res:
        n = max(r["row_hits"] + r["row_misses"] + r["row_conflicts"], 1)
        print(
            f"{r['mapper']:<12}{r['policy']:<11}{r['reads']:>10}{r['writes']:>9}{r['hit_rate']:>7.3f}"
            f"{r['row_conflicts'] / n:>10.3f}{r['avg_read_latency']:>8.1f}{r['cycles'] / 1e6:>9.1f}"
            f"{r['throughput_MBps'] / 1e3:>8.1f}"
        )


if __name__ == "__main__":
    main()
