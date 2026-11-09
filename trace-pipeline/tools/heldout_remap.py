#!/usr/bin/env python3
"""Held-out test of the XOR / region mappings: matrices searched on stream A (train) are
applied to stream B (test), as in region_remap.py but across tokens.

Usage: heldout_remap.py A_MISS A_TENSORS B_MISS B_TENSORS OUT_DIR [--window N] [--placements thp] [--drams ddr4,ddr5]
Writes for B's window, per placement x DRAM, Ramulator traces:
  identity, xor_train (global matrix searched on A), region_train (K/O matrices searched on A),
  xor_self, region_self (searched on B itself: the in-sample number used before).
Streams A and B come from separate processes (different virtual addresses, different random
THP placement): the matrices are over physical row bits and class tags, which transfer."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapprep as P  # noqa: E402
import region_remap as RR  # noqa: E402
import remap as R  # noqa: E402


def train(L, nch, d, pk, fanin, s0, s1, tag):
    pa = d["place"][P.PLACE[pk]].astype(np.uint64)
    t = P.granule_tags(d, pa)
    intra, ch = L.to_intra(pa[s0:s1])
    Mg = RR.search(L, intra, ch, nch, fanin, f"{tag} global")
    reg = {}
    for gi in (1, 2):
        m = t == gi
        if m.sum() < 5000:
            continue
        si, sc = L.to_intra(pa[m])
        reg[gi] = RR.search(L, si, sc, nch, fanin, f"{tag} region {P.TAGS[gi]}")
    return Mg, reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("am"); ap.add_argument("at"); ap.add_argument("bm"); ap.add_argument("bt"); ap.add_argument("out")
    ap.add_argument("--window", type=int, default=8_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--placements", default="thp")
    ap.add_argument("--drams", default="ddr4,ddr5")
    ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A, B = P.prep(a.am, a.at), P.prep(a.bm, a.bt)
    a0, a1 = P.window(A["n"], a.window, a.start_frac)
    b0, b1 = P.window(B["n"], a.window, a.start_frac)
    print(f"A {A['n']:,} reads window [{a0:,},{a1:,}); B {B['n']:,} reads window [{b0:,},{b1:,})")
    man = dict(a_reads=A["n"], b_reads=B["n"], window=[b0, b1], reads_total=B["n"], files=[])
    for pk in a.placements.split(","):
        pb = B["place"][P.PLACE[pk]].astype(np.uint64)
        tb = P.granule_tags(B, pb)[b0:b1]
        for dk in a.drams.split(","):
            L, nch = R.Layout(dk), P.NCH[dk]
            intra, ch = L.to_intra(pb[b0:b1])
            MgA, regA = train(L, nch, A, pk, a.fanin, a0, a1, f"{pk}/{dk} train(A)")
            MgB, regB = train(L, nch, B, pk, a.fanin, b0, b1, f"{pk}/{dk} self(B)")

            def region(regs):
                out = intra.copy()
                for gi, (Mn, *_r) in regs.items():
                    m = tb == gi
                    if m.any():
                        out[m] = RR.apply_named(L, intra[m], Mn)
                return out
            variants = {"identity": intra, "xor_train": RR.apply_named(L, intra, MgA[0]), "region_train": region(regA),
                        "xor_self": RR.apply_named(L, intra, MgB[0]), "region_self": region(regB)}
            for sk, new in variants.items():
                phys = L.from_intra(new, ch)
                if sk != "identity":
                    assert ((phys >> np.uint64(P.GRAN)) == (pb[b0:b1] >> np.uint64(P.GRAN))).all(), "granule changed"
                    assert (L.fields(new)[1] == L.fields(intra)[1]).all(), "row changed"
                fn = f"{pk}_{dk}_{sk}.trc"
                P.write_trace(os.path.join(a.out, fn), phys)
                m = RR.metrics(L, new, ch, nch, tb)
                m.update(placement=pk, dram=dk, scheme=sk, file=fn)
                man["files"].append(m)
                print(f"  {pk} {dk} {sk:<13} in-order hit {m['inorder_hit']:.3f}")
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
