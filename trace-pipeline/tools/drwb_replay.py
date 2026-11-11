#!/usr/bin/env python3
"""Read+write DRAM stream (from tools/drwb.c) -> attribution of writes and Ramulator traces with ST.

Usage: drwb_replay.py STREAM TENSORS_TSV OUT_DIR [--window N] [--start-frac F] [--placements packed,thp] [--drams ddr4,ddr5]

STREAM: uint64 per request, line-aligned virtual address | 1 for a write (drwb output).
Writes OUT_DIR/writes.json (reads/writes per class, bytes, write fraction) and, for a window of N consecutive
requests (default 12M, from F=0.3 of the stream), for each placement x DRAM x {identity, xor_global}
two traces: <p>_<d>_<s>_rw.trc (reads and writes, LD/ST) and <p>_<d>_<s>_ro.trc (the same window's reads only).
The xor_global matrix is searched on the window's read+write requests (region_remap.search, as in 18_region.sh) and
applied to both traces, so rw and ro differ only by the writes. manifest.json lists the files.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_remap as RR  # noqa: E402
import remap as R  # noqa: E402
import drmiss_export as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("tensors")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=12_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--placements", default="packed,thp")
    ap.add_argument("--drams", default="ddr4,ddr5")
    ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    v = np.fromfile(a.stream, dtype="<u8")
    isw = (v & np.uint64(1)).astype(bool)
    va = (v & ~np.uint64(63)).astype(np.int64)
    del v
    n = len(va)
    cls, _ = X.classify(va, X.load_tensors(a.tensors))
    names = X.CLASSES + ["unattributed"]
    stats = dict(requests=n, reads=int((~isw).sum()), writes=int(isw.sum()), write_fraction=float(isw.mean()), classes={})
    for i, nm in enumerate(names):
        r, w = int(((cls == i) & ~isw).sum()), int(((cls == i) & isw).sum())
        stats["classes"][nm] = dict(reads=r, writes=w, write_bytes=w * 64, read_bytes=r * 64)
    json.dump(stats, open(os.path.join(a.out, "writes.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    place, _np = X.placements(va)
    w = min(a.window, n)
    s0 = min(int(n * a.start_frac), n - w)
    sw = isw[s0:s0 + w]
    print(f"window [{s0:,}, {s0 + w:,}): {int(sw.sum()):,} writes of {w:,} requests")
    manifest = dict(requests=n, window=[s0, s0 + w], window_writes=int(sw.sum()), files=[])
    for pk in a.placements.split(","):
        pw = place[RR.PLACE[pk]].astype(np.uint64)[s0:s0 + w]
        for dk in a.drams.split(","):
            L, nch = R.Layout(dk), RR.NCH[dk]
            intra, ch = L.to_intra(pw)
            Mg, sel, c0, c1 = RR.search(L, intra, ch, nch, a.fanin, f"{pk}/{dk} global(rw)")
            for sk, new in (("identity", intra), ("xor_global", RR.apply_named(L, intra, Mg))):
                phys = L.from_intra(new, ch)
                for mode, sel_mask in (("rw", np.ones(w, bool)), ("ro", ~sw)):
                    fn = f"{pk}_{dk}_{sk}_{mode}.trc"
                    ph, ww = phys[sel_mask].tolist(), sw[sel_mask].tolist()
                    with open(os.path.join(a.out, fn), "w") as f:
                        f.writelines(f"{'ST' if x else 'LD'} 0x{p:x}\n" for p, x in zip(ph, ww))
                    manifest["files"].append(dict(file=fn, placement=pk, dram=dk, scheme=sk, mode=mode, requests=len(ph),
                                                  matrix=sel if sk == "xor_global" else None))
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
