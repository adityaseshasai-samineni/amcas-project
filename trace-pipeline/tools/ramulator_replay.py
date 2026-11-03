#!/usr/bin/env python3
"""Replay a LoadStoreTrace through Ramulator 2.1 for each mapping and page policy.

Usage: ramulator_replay.py TRACE OUT_JSON

LoadStoreTrace has no timestamps: it offers one request per cycle and blocks when
the controller is full. The stream is therefore saturated, unlike the timed
DRAMsim3 replay, and latency here is not comparable with it.
"""

import json
import multiprocessing
import sys

MAPPERS = ["RoBaRaCoCh", "MOP4CLXOR", "ChRaBaRoCo"]
POLICIES = ["Open", "ClosedCAP"]


def run(cfg):
    import ramulator

    trace, mapper, policy = cfg
    # Same DRAM as the DRAMsim3 config: 2 ranks of DDR4-2400 8Gb x8, one channel.
    ctrl = ramulator.controller.GenericDDR(
        dram=ramulator.dram.DDR4(
            org_preset="DDR4_8Gb_x8", timing_preset="DDR4_2400R", rank=2
        ),
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=getattr(ramulator.row_policy, policy)(),
        addr_mapper=getattr(ramulator.addr_mapper, mapper)(),
    )
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
    sim = ramulator.Simulation(
        ramulator.frontend.LoadStoreTrace(clock_ratio=1, path=trace), mem
    )
    sim.run()
    s = sim.stats["memory_system"]["controller"]
    n = s["row_hits"] + s["row_misses"] + s["row_conflicts"]
    return dict(
        mapper=mapper,
        policy=policy,
        reads=s["num_read_reqs"],
        writes=s["num_write_reqs"],
        cycles=s["cycles"],
        row_hits=s["row_hits"],
        row_misses=s["row_misses"],
        row_conflicts=s["row_conflicts"],
        hit_rate=s["row_hits"] / max(n, 1),
        avg_read_latency=s["avg_read_latency"],
    )


def main():
    trace, out = sys.argv[1:3]
    cfgs = [(trace, m, p) for m in MAPPERS for p in POLICIES]
    with multiprocessing.Pool(len(cfgs)) as pool:
        res = pool.map(run, cfgs)
    json.dump(res, open(out, "w"), indent=1)
    print(f"{'mapper':<12}{'policy':<11}{'reads':>10}{'writes':>9}{'hit':>7}{'conflict':>10}{'rd_lat':>8}{'Mcycles':>9}")
    for r in res:
        n = max(r["row_hits"] + r["row_misses"] + r["row_conflicts"], 1)
        print(
            f"{r['mapper']:<12}{r['policy']:<11}{r['reads']:>10}{r['writes']:>9}{r['hit_rate']:>7.3f}"
            f"{r['row_conflicts'] / n:>10.3f}{r['avg_read_latency']:>8.1f}{r['cycles'] / 1e6:>9.1f}"
        )


if __name__ == "__main__":
    main()
