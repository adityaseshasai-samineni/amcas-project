#!/usr/bin/env python3
"""Cross-workload fixed-mask test for the GPU mapping proposal.

A real controller has ONE address mapping. policy_grid.py fits the XOR masks on
each case's own window (in-sample oracle). Here the masks are fitted once on a
training set and applied unchanged to every case's policy window:

  trainers: q7_32k                       (dense, long context)
            q30_32k/ctx..e520            (MoE, long context, GPU0)
            pooled = q7_32k + q30_32k/e520 + q05_32k windows concatenated
  eval:     every window in tracing/campaign20-grid-input/*/ctx_*/window.dram
  maps:     identity; for xor_global, chan, xor_global_chan: <map>_oracle
            (refit on the eval window itself) and <map>_<trainer> (fixed mask)
  grid:     placement packed/thp (same THP permutation as policy_grid.py),
            interleave 256 B / 2 KiB, policies Open / Timeout256.
Ramulator HBM2 replay identical to policy_grid.replay. The oracle numbers must
reproduce the grid's in-sample cycles (consistency check).
Usage: xworkload.py OUT_JSON WORKDIR [PROCS]
"""
import json, os, pathlib, sys, multiprocessing as mp
import numpy as np
A = pathlib.Path.home() / 'amcas-pro'; sys.path[:0] = [str(A / 'tools'), str(A / 'gpu'), str(pathlib.Path(__file__).parent)]
import remap as R, region_remap as RR, memsys_study as MS, gpu_replay as G
from holdout import replay, fit, apply

I = A / 'tracing/campaign20-grid-input'
U = np.uint64
POLS = ['Open', 'Timeout256']
TRAIN = {'q7_32k': ['q7_32k/ctx_0x5607d28030e0'],
         'q30_32k': ['q30_32k/ctx_0x55f33f54e520'],
         'pooled': ['q7_32k/ctx_0x5607d28030e0', 'q30_32k/ctx_0x55f33f54e520', 'q05_32k/ctx_0x556960d1c0c0']}
SMOKE = int(os.environ.get('XW_SMOKE', 0))  # >0: truncate every window to this many sectors, first 2 eval windows only
FRAMES = np.random.default_rng(1).permutation(8192).astype(U)


def load(key, placement):
    p = I / key / 'window.dram'
    arr = np.fromfile(p, dtype=U); pa = np.fromfile(str(p) + '.packed', dtype=U); assert len(arr) == len(pa)
    if SMOKE: arr, pa = arr[:SMOKE], pa[:SMOKE]
    st = (arr & U(1)).astype(bool)
    if placement == 'thp':
        rank = pa >> U(21); assert rank.max() < 8192
        pa = (FRAMES[rank] << U(21)) | (pa & U((1 << 21) - 1))
    return pa, st


def main():
    out, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); work.mkdir(parents=True, exist_ok=True)
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    windows = sorted(str(d.relative_to(I)) for d in I.glob('*/ctx_*') if (d / 'window.dram.packed').is_file())
    if SMOKE: windows = windows[:2]
    res = json.load(open(out)) if out.is_file() else []
    done = {(r['eval'], r['placement'], r['channel_interleave_bytes']) for r in res}
    for placement in ('packed', 'thp'):
        for s8 in (8, 11):
            R.LAYOUTS['campaign'] = dict(G.LAY['hbm2x32'], s=s8); L = R.Layout('campaign')
            RR.cost2.__defaults__ = (2.0, 40.0); MS.cost3.__defaults__ = (2.0, 40.0); R.cost.__defaults__ = (2.0, 40.0)
            fits = {}
            for t, keys in TRAIN.items():
                pa = np.concatenate([load(k, placement)[0] for k in keys]); intra, ch = L.to_intra(pa)
                fits[t] = fit(L, intra, ch.astype(np.int64))
            for w in windows:
                if (w, placement, 1 << s8) in done: continue
                pa, st = load(w, placement); intra, ch = L.to_intra(pa); ch = ch.astype(np.int64)
                meta0 = dict(eval=w, case=w.split('/')[0], placement=placement, channel_interleave_bytes=1 << s8,
                             ramulator_interleave_bits=s8 - 5, reads=int((~st).sum()), writes=int(st.sum()))
                maps = {'identity': (intra, ch)}
                maps.update({m + '_oracle': v for m, v in apply(L, intra, ch, fit(L, intra, ch)).items()})
                for t, f in fits.items():
                    maps.update({m + '_' + t: v for m, v in apply(L, intra, ch, f).items()})
                tasks = []
                for name, (i, c) in maps.items():
                    dest = L.from_intra(i, c.astype(U)); assert np.unique(dest).size == np.unique(pa).size, (w, name)
                    p = work / f'{placement}_s{s8}_{name}.trc'; G.write_trace(p, st, dest)
                    tasks += [(str(p), pol, dict(meta0, mapping=name)) for pol in POLS]
                with mp.get_context('spawn').Pool(procs) as pool:
                    for r in pool.imap_unordered(replay, tasks):
                        res.append(r); json.dump(res, open(out, 'w'), indent=1)
                print(w, placement, 1 << s8, len(tasks), 'replays', flush=True)
                for p in work.glob('*.trc'): p.unlink()


if __name__ == '__main__':
    main()
