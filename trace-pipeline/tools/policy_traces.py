#!/usr/bin/env python3
"""Build Ramulator traces for the row-policy study from an attributed LLC-miss stream.

Usage: policy_traces.py MISS_GZ TENSORS_TSV OUT_DIR [--window N] [--start-frac F] [--bubbles B]

Uses (read-only) region_remap/remap/drmiss_export helpers. For placement in {packed, thp} x
DRAM in {ddr4, ddr5} x mapping in {identity, xor_global} (global greedy GF(2) bank XOR, as
region_remap) it writes, for one contiguous window of the miss stream:
  KEY.trc        saturated LoadStoreTrace ("LD 0xADDR")
  KEY.tK.trc     per-thread SimpleO3 traces ("BUBBLES ADDR"), K = thread 0..3 (misses of the same
                 recorded thread, in order)
  KEY.ranges     physical 2 MiB granule ranges "START END CLASS" (W weights, K KV cache, O other),
                 class by majority of the reads the granule receives (as region_remap). Both
                 mappings keep the granule (asserted), so one file serves both.
KEY = <placement>_<dram>_<mapping>.  BUBBLES = non-memory instructions between misses of a thread
(--bubbles, instructions per DRAM read of the token: total instructions / reads).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remap as R
import drmiss_export as X
import region_remap as G
from analyze import LINE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss"); ap.add_argument("tensors"); ap.add_argument("out")
    ap.add_argument("--window", type=int, default=4_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--bubbles", type=int, default=40)
    ap.add_argument("--placements", default="packed,thp")
    ap.add_argument("--drams", default="ddr4,ddr5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tids, addrs = X.read_misses(a.miss)
    va = (addrs // LINE * LINE).astype(np.int64); del addrs
    cls, _ = X.classify(va, X.load_tensors(a.tensors))
    grp_of = np.array([0 if n in ("expert weights", "dense weights") else 1 if n == "KV cache" else 2 for n in X.CLASSES] + [2], dtype=np.int8)
    grp = grp_of[cls]
    place, _ = X.placements(va)
    n = len(va); w = min(a.window, n); s0 = min(int(n * a.start_frac), n - w)
    ut = {t: i for i, t in enumerate(np.unique(tids[s0:s0 + w]))}
    tw_id = np.array([ut[t] for t in tids[s0:s0 + w]], dtype=np.int8)
    print(f"{n:,} reads, window [{s0:,},{s0 + w:,}), threads {len(ut)}, class shares W/K/O "
          f"{[round(float((grp[s0:s0 + w] == i).mean()), 4) for i in range(3)]}")
    man = dict(reads_total=n, window=[s0, s0 + w], files=[])
    for pk in a.placements.split(","):
        pa = place[G.PLACE[pk]].astype(np.uint64)
        gran = (pa >> np.uint64(G.GRAN)).astype(np.int64)
        ug, inv = np.unique(gran, return_inverse=True)
        cnt = np.zeros((len(ug), 3), dtype=np.int64); np.add.at(cnt, (inv, grp.astype(np.int64)), 1)
        gtag = cnt.argmax(1)
        pw = pa[s0:s0 + w]
        wg = np.unique((pw >> np.uint64(G.GRAN)).astype(np.int64))
        for dk in a.drams.split(","):
            L, nch = R.Layout(dk), G.NCH[dk]
            intra, ch = L.to_intra(pw)
            Mg, sel, c0, c1 = G.search(L, intra, ch, nch, 3, f"{pk}/{dk} global")
            for sk, new in (("identity", intra), ("xor_global", G.apply_named(L, intra, Mg))):
                phys = L.from_intra(new, ch)
                assert ((phys >> np.uint64(G.GRAN)) == (pw >> np.uint64(G.GRAN))).all()
                key = f"{pk}_{dk}_{sk}"
                with open(f"{a.out}/{key}.trc", "w") as f:
                    f.writelines(f"LD 0x{x:x}\n" for x in phys.tolist())
                for t in range(len(ut)):
                    with open(f"{a.out}/{key}.t{t}.trc", "w") as f:
                        f.writelines(f"{a.bubbles} {x}\n" for x in phys[tw_id == t].tolist())
                man["files"].append(dict(key=key, placement=pk, dram=dk, mapping=sk, reads=int(w),
                                         per_thread=[int((tw_id == t).sum()) for t in range(len(ut))], bubbles=a.bubbles))
            # ranges: granules touched in the window, merged
            tg = "WKO"
            rows = []
            for g in wg:
                c = tg[gtag[np.searchsorted(ug, g)]]
                if rows and rows[-1][2] == c and rows[-1][1] == (g << G.GRAN):
                    rows[-1][1] = (g + 1) << G.GRAN
                else:
                    rows.append([g << G.GRAN, (g + 1) << G.GRAN, c])
            for sk in ("identity", "xor_global"):
                with open(f"{a.out}/{pk}_{dk}_{sk}.ranges", "w") as f:
                    f.writelines(f"{s} {e} {c}\n" for s, e, c in rows)
            print(f"  {pk}/{dk}: {len(rows)} ranges")
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
