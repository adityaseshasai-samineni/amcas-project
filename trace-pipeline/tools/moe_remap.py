#!/usr/bin/env python3
"""Expert-aware mapping for MoE expert weights vs identity / global XOR / region, on an
attributed LLC-miss stream (30B-A3B).

Usage: moe_remap.py MISS_GZ TENSORS_TSV OUT_DIR [--window N] [--placements packed,thp] [--drams ddr4,ddr5]

Expert id of a read = (address - tensor base) // (tensor bytes / n_experts). Variants:
  identity, xor_global (remap's GF(2) search on the window), region (K/O XOR, as region_remap)
  expert_hash   bank ^= H(layer, tensor kind, expert) (5-bit multiplicative hash), applied to
                row groups whose reads are expert weights; the hash is a function of the row group
  expert_lin    bank ^= (layer*3+kind + expert) mod 32, a round-robin offset
  region_expert region + expert_hash on the weight granules
Bijection: as region_remap. The value XORed into the bank field must depend only on bits that
are preserved (the row, which includes the granule). An expert slice (864 KiB for 30B gate/up)
does not align to a row group, so H is evaluated on the row group's majority expert (computed
over the stream; in hardware, a table indexed by row group, or slice-aligned allocation).
Channel bits are not changed: the 256 B interleave already spreads every stream over all 16
subchannels, so a channel XOR can only permute equal loads."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapprep as P  # noqa: E402
import region_remap as RR  # noqa: E402
import remap as R  # noqa: E402


def hash5(layer, kind, e, mode):
    k = layer * 3 + kind
    if mode == "lin":
        return (k + e) % 32
    return ((((k * 128 + e) * 2654435761) & 0xFFFFFFFF) >> 15) & 31


def expert_xor(L, intra, e, lay, kd, mode):
    """XOR the bank field by H(majority expert of the row group); non-expert row groups unchanged."""
    bank, row, _ = L.fields(intra)
    isx = e >= 0
    if not isx.any():
        return intra
    # majority expert per row (row field of the intra address; unique per row group of a channel)
    key = row[isx] * 32768 + (lay[isx] * 3 + kd[isx]) * 128 + e[isx]  # (row, layer/kind, expert) cells
    cell, cnt = np.unique(key, return_counts=True)
    r_of = cell // 32768
    order = np.lexsort((-cnt, r_of))  # per row, highest count first
    r_sorted = r_of[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = r_sorted[1:] != r_sorted[:-1]
    win = cell[order][first]
    wrow, wrest = win // 32768, win % 32768
    hv = hash5((wrest // 128) // 3, (wrest // 128) % 3, wrest % 128, mode).astype(np.int64)
    # the XOR constant must depend on the row only (bijection per row): apply it to every access in a row that
    # has a winning expert, including non-expert accesses that share that row
    idx = np.minimum(np.searchsorted(wrow, row), len(wrow) - 1)
    hit = wrow[idx] == row
    new = bank.copy()
    new[hit] ^= hv[idx[hit]]
    return L.put_bank(intra, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss"); ap.add_argument("tensors"); ap.add_argument("out")
    ap.add_argument("--window", type=int, default=12_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--placements", default="packed,thp")
    ap.add_argument("--drams", default="ddr4,ddr5")
    ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = P.prep(a.miss, a.tensors)
    e_all, l_all, k_all = P.expert_features(d)
    s0, s1 = P.window(d["n"], a.window, a.start_frac)
    print(f"{d['n']:,} reads; window [{s0:,},{s1:,}); expert share in window {float((e_all[s0:s1] >= 0).mean()):.3f}; "
          f"distinct experts in window {len(np.unique(e_all[s0:s1][e_all[s0:s1] >= 0]))}")
    man = dict(reads_total=d["n"], window=[s0, s1], files=[])
    for pk in a.placements.split(","):
        pa = d["place"][P.PLACE[pk]].astype(np.uint64)
        tag = P.granule_tags(d, pa)
        pw, tw = pa[s0:s1], tag[s0:s1]
        e, ly, kd = e_all[s0:s1], l_all[s0:s1], k_all[s0:s1]
        for dk in a.drams.split(","):
            L, nch = R.Layout(dk), P.NCH[dk]
            intra, ch = L.to_intra(pw)
            Mg = RR.search(L, intra, ch, nch, a.fanin, f"{pk}/{dk} global")
            reg = intra.copy()
            for gi in (1, 2):
                m = tw == gi
                mf = tag == gi
                if mf.sum() < 5000 or not m.any():
                    continue
                si, sc = L.to_intra(pa[mf])
                Mr = RR.search(L, si, sc, nch, a.fanin, f"{pk}/{dk} region {P.TAGS[gi]}")
                reg[m] = RR.apply_named(L, intra[m], Mr[0])
            ex_h = expert_xor(L, intra, e, ly, kd, "hash")
            reg_e = reg.copy()
            wm = tw == 0
            reg_e[wm] = ex_h[wm]
            variants = {"identity": intra, "xor_global": RR.apply_named(L, intra, Mg[0]), "region": reg,
                        "expert_hash": ex_h, "expert_lin": expert_xor(L, intra, e, ly, kd, "lin"), "region_expert": reg_e}
            for sk, new in variants.items():
                phys = L.from_intra(new, ch)
                assert ((phys >> np.uint64(P.GRAN)) == (pw >> np.uint64(P.GRAN))).all(), "granule changed"
                assert (L.fields(new)[1] == L.fields(intra)[1]).all() and (L.fields(new)[2] == L.fields(intra)[2]).all()
                # bijection check on the used cells: same (bank,row,col) pair count
                ob, orow, oc = L.fields(intra); nb, nr, nc = L.fields(new)
                assert len(np.unique((nb << 40) | (nr << 8) | nc)) == len(np.unique((ob << 40) | (orow << 8) | oc)), "not injective"
                fn = f"{pk}_{dk}_{sk}.trc"
                P.write_trace(os.path.join(a.out, fn), phys)
                m = RR.metrics(L, new, ch, nch, tw)
                m.update(placement=pk, dram=dk, scheme=sk, file=fn)
                man["files"].append(m)
                print(f"  {pk} {dk} {sk:<14} in-order hit {m['inorder_hit']:.3f} banks {m['banks_used']}/{m['banks_total']}")
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
