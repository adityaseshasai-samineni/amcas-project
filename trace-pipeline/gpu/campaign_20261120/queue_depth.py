#!/usr/bin/env python3
"""Ramulator controller queue-depth sensitivity for the GPU mapping disagreement.

Rebuilds exactly the policy_grid.py traces (same window file, packed placement,
256 B channel interleave, same xor_global / mopxor searches) for identity,
xor_global and mopxor, then replays each under Open with HBM12
read_buffer_size = 32 (grid default), 64 (Accel-Sim SM7_QV100
frfcfs_dram_sched_queue_size) and 128. Everything else matches policy_grid.replay.
Usage: queue_depth.py WINDOW_DRAM TENSORS OUT_JSON
"""
import json, pathlib, sys, multiprocessing as mp
import numpy as np
A = pathlib.Path.home() / 'amcas-pro'; sys.path[:0] = [str(A / 'tools'), str(A / 'gpu')]
import remap as R, region_remap as RR, mop4_traces as MT, grid_replay as GR, gpu_replay as G

DEPTHS = [32, 64, 128]


def replay(t):
    import ramulator as rm
    path, mapper, depth, meta = t
    path = pathlib.Path(path)
    def ctrl():
        return rm.controller.HBM12(read_buffer_size=depth, dram=rm.dram.HBM2(org_preset='HBM2_2Gb', timing_preset='HBM2_1600Mbps'),
                                   scheduler=rm.scheduler.FRFCFS(), refresh_manager=rm.refresh_manager.AllBank(),
                                   row_policy=GR.policy(rm, 'Open', str(path.parent), path.stem, ''), addr_mapper=getattr(rm.addr_mapper, mapper)())
    mem = rm.memory_system.GenericDRAM(clock_ratio=1, controllers=[ctrl() for _ in range(32)],
                                       channel_mapper=rm.channel_mapper.CacheLineInterleave(interleave_bits=3))
    fe = rm.frontend.LoadStoreTrace(clock_ratio=32, path=str(path)); sim = rm.Simulation(fe, mem); sim.run()
    cs = sim.stats['memory_system']['controller']; tot = lambda k: sum(c[k] for c in cs)
    assert tot('num_read_reqs') == meta['reads'] and tot('num_write_reqs') == meta['writes']
    assert tot('num_read_reqs_served') + tot('num_read_reqs_forwarded') == meta['reads']
    cyc = max(c['cycles'] for c in cs); ev = tot('row_hits') + tot('row_misses') + tot('row_conflicts')
    return dict(meta, read_buffer_size=depth, cycles=cyc, row_hit_rate=tot('row_hits') / max(ev, 1),
                row_conflict_rate=tot('row_conflicts') / max(ev, 1))


def main():
    binary, tensors, out = sys.argv[1:4]; out = pathlib.Path(out); work = out.parent / (out.stem + '_work'); work.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(binary, dtype=np.uint64, mode='r'); pa = np.asarray(np.memmap(binary + '.packed', dtype=np.uint64, mode='r'))
    n = len(arr); w = min(4000000, n); s = min(int(n * .3), n - w); assert s == 0 and w == n, (s, w, n)
    st = (np.asarray(arr) & np.uint64(1)).astype(bool)
    R.LAYOUTS['campaign'] = dict(G.LAY['hbm2x32'], s=8); L = R.Layout('campaign'); ML = MT.MopView(L); intra, ch = L.to_intra(pa)
    import memsys_study as MS
    RR.cost2.__defaults__ = (2.0, 40.0); MS.cost3.__defaults__ = (2.0, 40.0); R.cost.__defaults__ = (2.0, 40.0)
    M, _, _, _ = RR.search(L, intra, ch.astype(np.int64), 32, 3, 'global'); xg = RR.apply_named(L, intra, M)
    Mm, _, _, _ = RR.search(ML, intra, ch.astype(np.int64), 32, 3, 'MOP'); mxg = RR.apply_named(ML, intra, Mm)
    maps = {'identity': (intra, 'RoBaRaCoCh'), 'xor_global': (xg, 'RoBaRaCoCh'), 'mopxor': (mxg, 'MOP4CLXOR')}
    meta0 = dict(reads=int((~st).sum()), writes=int(st.sum()), placement='packed', channel_interleave_bytes=256, policy='Open', source=binary)
    tasks = []
    for name, (i, mapper) in maps.items():
        dest = L.from_intra(i, ch); assert np.unique(dest).size == np.unique(pa).size, name
        p = work / f'{name}.trc'; G.write_trace(p, st, dest)
        tasks += [(str(p), mapper, d, dict(meta0, mapping=name)) for d in DEPTHS]
    res = []
    with mp.get_context('spawn').Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 3) as pool:
        for r in pool.imap_unordered(replay, tasks):
            res.append(r); json.dump(res, open(out, 'w'), indent=1); print(r, flush=True)
    for p in work.glob('*'): p.unlink()
    work.rmdir()


if __name__ == '__main__':
    main()
