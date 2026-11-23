#!/usr/bin/env python3
"""Does sharing one command bus between two HBM2 pseudo-channels decide the mop ranking?

Same window, placement (packed), 256 B channel interleave and mapping searches as
policy_grid.py / queue_depth.py. Each request's EFFECTIVE (channel, bank5, row, col)
is taken from the mapping's view (Layout for RoBaRaCoCh mappings, MopView for
mopxor, i.e. after the MOP4CLXOR XOR) and re-encoded explicitly, then replayed
with the RoBaRaCoCh mapper (no further hashing) in two organisations:

  shared: 32 HBM12 controllers x 2 PCs (the grid's model; JEDEC: both PCs of a
          channel share one row and one column command bus). bank5 bit0 = PC.
  split : 64 HBM12 controllers x 1 PC (pseudochannel=1), controller = ch*2+PC,
          each PC with its own row and column command bus (DRAMsim3-like).

Buffer depth per channel pair is matched: shared read/write buffer D, split D/2
per PC, for D in 32 and 64. All DRAM timings are identical.
The shared runs must reproduce the grid's identity cycles (encoding check).
Usage: pc_split.py WINDOW_DRAM OUT_JSON [PROCS]
"""
import json, pathlib, sys, multiprocessing as mp
import numpy as np
A = pathlib.Path.home() / 'amcas-pro'; sys.path[:0] = [str(A / 'tools'), str(A / 'gpu')]
import remap as R, region_remap as RR, mop4_traces as MT, grid_replay as GR, gpu_replay as G, memsys_study as MS

U = np.uint64


def encode(org, ch, b, row, col):
    ch, b, row, col = (x.astype(np.uint64) for x in (ch, b, row, col))
    if org == 'shared':  # off5 | col5 | ctrl5 | PC,BG2,BA2 | row
        return (((row << U(5)) | b) << U(15)) | (ch << U(10)) | (col << U(5))
    ctrl = ch * U(2) + (b & U(1))  # off5 | col5 | ctrl6 | BG2,BA2 | row
    return (((row << U(4)) | (b >> U(1))) << U(16)) | (ctrl << U(10)) | (col << U(5))


def replay(t):
    import ramulator as rm
    path, org, depth, meta = t
    path = pathlib.Path(path); n, pcs, buf = (32, 2, depth) if org == 'shared' else (64, 1, depth // 2)
    def ctrl():
        return rm.controller.HBM12(read_buffer_size=buf, write_buffer_size=buf,
                                   dram=rm.dram.HBM2(org_preset='HBM2_2Gb', timing_preset='HBM2_1600Mbps', pseudochannel=pcs),
                                   scheduler=rm.scheduler.FRFCFS(), refresh_manager=rm.refresh_manager.AllBank(),
                                   row_policy=GR.policy(rm, 'Open', str(path.parent), path.stem, ''), addr_mapper=rm.addr_mapper.RoBaRaCoCh())
    mem = rm.memory_system.GenericDRAM(clock_ratio=1, controllers=[ctrl() for _ in range(n)],
                                       channel_mapper=rm.channel_mapper.CacheLineInterleave(interleave_bits=5))
    fe = rm.frontend.LoadStoreTrace(clock_ratio=32, path=str(path)); sim = rm.Simulation(fe, mem); sim.run()
    cs = sim.stats['memory_system']['controller']; tot = lambda k: sum(c[k] for c in cs)
    assert len(cs) == n and tot('num_read_reqs') == meta['reads'] and tot('num_write_reqs') == meta['writes']
    assert tot('num_read_reqs_served') + tot('num_read_reqs_forwarded') == meta['reads']
    cyc = max(c['cycles'] for c in cs); ev = tot('row_hits') + tot('row_misses') + tot('row_conflicts')
    busy = sorted(c['num_read_reqs'] + c['num_write_reqs'] for c in cs)
    return dict(meta, org=org, controllers=n, buffer_per_controller=buf, buffer_per_channel_pair=depth, cycles=cyc,
                row_hit_rate=tot('row_hits') / max(ev, 1), row_conflict_rate=tot('row_conflicts') / max(ev, 1),
                reqs_per_controller_min_max=[busy[0], busy[-1]])


def main():
    binary, out = sys.argv[1:3]; out = pathlib.Path(out); work = out.parent / (out.stem + '_work'); work.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(binary, dtype=np.uint64, mode='r'); pa = np.asarray(np.memmap(binary + '.packed', dtype=np.uint64, mode='r'))
    assert len(arr) == len(pa) <= 4000000
    st = (np.asarray(arr) & U(1)).astype(bool)
    R.LAYOUTS['campaign'] = dict(G.LAY['hbm2x32'], s=8); L = R.Layout('campaign'); ML = MT.MopView(L); intra, ch = L.to_intra(pa)
    RR.cost2.__defaults__ = (2.0, 40.0); MS.cost3.__defaults__ = (2.0, 40.0); R.cost.__defaults__ = (2.0, 40.0)
    M, _, _, _ = RR.search(L, intra, ch.astype(np.int64), 32, 3, 'global'); xg = RR.apply_named(L, intra, M)
    Mm, _, _, _ = RR.search(ML, intra, ch.astype(np.int64), 32, 3, 'MOP'); mxg = RR.apply_named(ML, intra, Mm)
    views = {'identity': (L, intra), 'xor_global': (L, xg), 'mopxor': (ML, mxg)}
    meta0 = dict(reads=int((~st).sum()), writes=int(st.sum()), placement='packed', channel_interleave_bytes=256, policy='Open', source=binary)
    tasks = []
    for name, (view, i) in views.items():
        b, row, col = view.fields(i)
        for org in ('shared', 'split'):
            dest = encode(org, ch, b, row, col); assert np.unique(dest).size == np.unique(pa).size, (name, org)
            p = work / f'{name}_{org}.trc'; G.write_trace(p, st, dest)
            tasks += [(str(p), org, d, dict(meta0, mapping=name)) for d in (32, 64)]
    res = []
    with mp.get_context('spawn').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        for r in pool.imap_unordered(replay, tasks):
            res.append(r); json.dump(res, open(out, 'w'), indent=1); print(r, flush=True)
    for p in work.glob('*'): p.unlink()
    work.rmdir()


if __name__ == '__main__':
    main()
