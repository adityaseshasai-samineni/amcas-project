#!/usr/bin/env python3
"""Held-out window check for the GPU mapping results.

policy_grid.py fits each XOR mapping on one 4M-sector window (at 30% of the
context's DRAM stream) and evaluates it on that same window (in-sample).
This script asks whether the masks generalise along the stream:

  fit window   F = stream[0.3n : 0.3n+4M]          (the grid's window)
  eval windows E in {0.1n, 0.3n, 0.7n} (each 4M sectors)

For every eval window it replays (Ramulator HBM2, same controller, timings,
packed placement, CacheLineInterleave as policy_grid.replay):
  identity                       (baseline on E)
  <map>_transfer                 masks fitted on F, applied to E   (held-out)
  <map>_oracle                   masks refitted on E itself         (in-sample on E)
for map in xor_global, chan, xor_global_chan; policies Open and Timeout256;
channel interleave 256 B (s=8) and 2 KiB (s=11).
On E = F, transfer == oracle (checked: identical cycles to the grid).
Usage: holdout.py MASKED_DRAM OUT_JSON [PROCS]
"""
import json, os, pathlib, sys, multiprocessing as mp
import numpy as np
A = pathlib.Path.home() / 'amcas-pro'; sys.path[:0] = [str(A / 'tools'), str(A / 'gpu')]
import remap as R, region_remap as RR, memsys_study as MS, grid_replay as GR, gpu_replay as G

W = int(os.environ.get('HOLDOUT_W', 4000000))  # smaller only for smoke tests
POLS = ['Open', 'Timeout256']
U = np.uint64


def replay(t):
    import ramulator as rm
    path, pol, meta = t; path = pathlib.Path(path)
    def ctrl():
        return rm.controller.HBM12(dram=rm.dram.HBM2(org_preset='HBM2_2Gb', timing_preset='HBM2_1600Mbps'),
                                   scheduler=rm.scheduler.FRFCFS(), refresh_manager=rm.refresh_manager.AllBank(),
                                   row_policy=GR.policy(rm, pol, str(path.parent), path.stem, ''), addr_mapper=rm.addr_mapper.RoBaRaCoCh())
    mem = rm.memory_system.GenericDRAM(clock_ratio=1, controllers=[ctrl() for _ in range(32)],
                                       channel_mapper=rm.channel_mapper.CacheLineInterleave(interleave_bits=meta['ramulator_interleave_bits']))
    fe = rm.frontend.LoadStoreTrace(clock_ratio=32, path=str(path)); sim = rm.Simulation(fe, mem); sim.run()
    cs = sim.stats['memory_system']['controller']; tot = lambda k: sum(c[k] for c in cs)
    assert tot('num_read_reqs') == meta['reads'] and tot('num_write_reqs') == meta['writes']
    cyc = max(c['cycles'] for c in cs); ev = tot('row_hits') + tot('row_misses') + tot('row_conflicts')
    return dict(meta, policy=pol, cycles=cyc, row_hit_rate=tot('row_hits') / max(ev, 1), row_conflict_rate=tot('row_conflicts') / max(ev, 1))


def fit(L, intra, ch):
    M, _, _, _ = RR.search(L, intra, ch, 32, 3, 'global')
    Mc, cc, _, _, _ = MS.chan_search(L, intra, ch, 32, 3, 'channel')
    xg = RR.apply_named(L, intra, M)
    Mxc, cxc, _, _, _ = MS.chan_search(L, xg, ch, 32, 3, 'joint')
    return dict(M=M, Mc=Mc, cc=cc, Mxc=Mxc, cxc=cxc)


def apply(L, intra, ch, f):
    xg = RR.apply_named(L, intra, f['M'])
    return {'xor_global': (xg, ch),
            'chan': (intra, ch ^ MS.hash_bits(L, intra, f['Mc'], f['cc'])),
            'xor_global_chan': (xg, ch ^ MS.hash_bits(L, xg, f['Mxc'], f['cxc']))}


def main():
    src, out = sys.argv[1:3]; out = pathlib.Path(out); work = out.parent / (out.stem + '_work'); work.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(src, dtype=np.uint64, mode='r'); pk = np.memmap(src + '.packed', dtype=np.uint64, mode='r')
    n = len(arr); assert n == len(pk) and n >= 10 * W, n
    starts = {'10': int(n * .1), '30': min(int(n * .3), n - W), '70': int(n * .7)}
    for s8 in (8, 11):
        R.LAYOUTS['campaign'] = dict(G.LAY['hbm2x32'], s=s8); L = R.Layout('campaign')
        RR.cost2.__defaults__ = (2.0, 40.0); MS.cost3.__defaults__ = (2.0, 40.0); R.cost.__defaults__ = (2.0, 40.0)
        win = {}
        for k, s in starts.items():
            pa = np.asarray(pk[s:s + W]); st = (np.asarray(arr[s:s + W]) & U(1)).astype(bool)
            intra, ch = L.to_intra(pa); win[k] = (pa, st, intra, ch.astype(np.int64))
        fits = {k: fit(L, w[2], w[3]) for k, w in win.items()}
        tasks = []
        for k, (pa, st, intra, ch) in win.items():
            meta0 = dict(source=src, stream_sectors=n, eval_window=k, eval_start=starts[k], fit_window='30', placement='packed',
                         channel_interleave_bytes=1 << s8, ramulator_interleave_bits=s8 - 5,
                         reads=int((~st).sum()), writes=int(st.sum()))
            maps = {'identity': (intra, ch)}
            maps.update({m + '_transfer': v for m, v in apply(L, intra, ch, fits['30']).items()})
            if k != '30':
                maps.update({m + '_oracle': v for m, v in apply(L, intra, ch, fits[k]).items()})
            for name, (i, c) in maps.items():
                dest = L.from_intra(i, c.astype(U)); assert np.unique(dest).size == np.unique(pa).size, (k, name)
                p = work / f's{s8}_w{k}_{name}.trc'; G.write_trace(p, st, dest)
                tasks += [(str(p), pol, dict(meta0, mapping=name)) for pol in POLS]
        res = json.load(open(out)) if out.is_file() else []
        with mp.get_context('spawn').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
            for r in pool.imap_unordered(replay, tasks):
                res.append(r); json.dump(res, open(out, 'w'), indent=1); print(r, flush=True)
        for p in work.glob('*'): p.unlink()
    work.rmdir()


if __name__ == '__main__':
    main()
