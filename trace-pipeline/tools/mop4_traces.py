#!/usr/bin/env python3
"""Request streams for the MOP4CLXOR vs xor_global comparison (mop-owner).

Usage: mop4_traces.py MISS_GZ TENSORS_TSV OUT_DIR [--window 8000000] [--o3-window 2000000] [--start-frac 0.3]
           [--placements packed,thp,4k] [--drams x16,x8,d4r6,d4] [--fanin 3]

Imports remap / region_remap / drmiss_export / realmem_traces read-only. For placement x DRAM it writes
  KEY_id   .trc  identity stream (window `--window`): replay with RoBaRaCoCh (identity), MOP4CLXOR (mop) or ChRaBaRoCo
  KEY_xg   .trc  RoBaRaCoCh + xor_global (bank bits ^= <=K row bits, greedy search, as region_remap.search): replay with RoBaRaCoCh
  KEY_mxg  .trc  MOP4CLXOR + xor_global: the same search run on the MOP4CLXOR-decoded (bank, row) fields; replay with MOP4CLXOR
  KEY_*.tK.addr  per-thread decimal addresses of the first --o3-window requests (closed loop; bubbles added by the replayer)
KEY = <placement>_<dram>. DRAM: x16 = DDR5-4800 16 subchannels, x8 = 8 subchannels, d4r6 = DDR4-2933 6 channels
(validation-owner model), d4 = DDR4-2400 1 channel (ramulator_replay.py's DDR4).

MOP4CLXOR (ramulator2 controller/addr_mapper/impl/mop4clxor.cpp), on the intra-channel address after the channel
bits are removed, LSB to MSB: [off=6 line bits] col[1:0] | bank region (rank, bankgroup, bank; 5 bits) | col[hi] | row.
Then the decoded fields are hashed: for each level below Row, in order rank, bankgroup, bank, the level value is XORed
with the next bits of the FULL column value (so the bank index = raw5 ^ (col & 31), the four lines of a 256 B chunk
go to four different bank groups), and the row is XORed with col >> 5 (1 bit DDR5, 2 bits DDR4). Bank-region and row
bits stay at the same address positions as in RoBaRaCoCh for the row; the bank region moves from above the column
field (RoBaRaCoCh) to just above col[1:0].
Composition with xor_global: the hash is a function of (raw bank, col, row); adding bank ^= f(row bits) to the effective
bank is done by XORing f(row) into the raw bank field of the address (put_bank below), so MOP4CLXOR then yields
bank_eff ^ f(row). Row, col and channel never change (asserted), and the map stays injective (asserted).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remap as R
import drmiss_export as X
import region_remap as G
import realmem_traces as RT
from analyze import LINE

U = np.uint64


class MopView:
    """remap.Layout-compatible view of a layout under MOP4CLXOR: fields() returns the EFFECTIVE (bank, row, col)."""

    def __init__(self, L):
        self.L = L
        self.bank_lo = L.off + 2
        self.hi_lo = self.bank_lo + L.bank

    def __getattr__(self, k):
        return getattr(self.L, k)

    def _col(self, i):
        L = self.L
        lo = (i >> U(L.off)) & U(3)
        hi = (i >> U(self.hi_lo)) & U((1 << (L.col - 2)) - 1)
        return (lo | (hi << U(2))).astype(np.int64)

    def fields(self, i):
        L = self.L
        col = self._col(i)
        raw = ((i >> U(self.bank_lo)) & U(L.nbanks - 1)).astype(np.int64)
        row = (i >> U(L.row_lo)).astype(np.int64)
        return raw ^ (col & (L.nbanks - 1)), row ^ (col >> L.bank), col

    def put_bank(self, i, bank):
        L = self.L
        col = self._col(i)
        raw = (bank.astype(np.int64) ^ (col & (L.nbanks - 1))).astype(np.uint64)
        m = U((L.nbanks - 1) << self.bank_lo)
        return (i & ~m) | (raw << U(self.bank_lo))


def make_layout(dk):
    if dk == "x16":
        return R.Layout("ddr5"), 16
    if dk == "x8":
        return R.Layout("ddr5x8"), 8
    if dk == "d4r6":
        return RT.RealLayout(6), 6
    if dk == "d4":
        return R.Layout("ddr4"), 1
    raise ValueError(dk)


def to_phys(L, intra, ch):
    ch = ch.astype(np.uint64)
    return L.to_slot(intra, ch) if hasattr(L, "to_slot") else L.from_intra(intra, ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss"); ap.add_argument("tensors"); ap.add_argument("out")
    ap.add_argument("--window", type=int, default=8_000_000)
    ap.add_argument("--o3-window", type=int, default=2_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--placements", default="packed,thp,4k")
    ap.add_argument("--drams", default="x16,x8,d4r6,d4")
    ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tids, addrs = X.read_misses(a.miss)
    va = (addrs // LINE * LINE).astype(np.int64); del addrs
    place, _ = X.placements(va)
    n = len(va); w = min(a.window, n); s0 = min(int(n * a.start_frac), n - w)
    wc = min(a.o3_window, w)
    NT = 4
    ut = {t: i for i, t in enumerate(np.unique(tids[s0:s0 + w]))}
    if len(ut) == NT:
        tw_id = np.array([ut[t] for t in tids[s0:s0 + wc]], dtype=np.int8)
    else:
        print(f"  WARNING: window has {len(ut)} thread(s); 4 contiguous pseudo-threads")
        tw_id = (np.arange(wc) * NT // wc).astype(np.int8)
    del va
    print(f"{n:,} reads, window [{s0:,},{s0 + w:,}), o3 window {wc:,}", flush=True)
    man = dict(reads_total=n, window=[s0, s0 + w], o3_window=wc, files=[])
    for pk in a.placements.split(","):
        pw = place[G.PLACE[pk]].astype(np.uint64)[s0:s0 + w]
        for dk in a.drams.split(","):
            L, nch = make_layout(dk)
            ML = MopView(L)
            intra, ch = L.to_intra(pw); ch = ch.astype(np.int64)
            Mg, sel_g, g0, g1 = G.search(L, intra, ch, nch, a.fanin, f"{pk}/{dk} RoBa xor_global")
            xg = G.apply_named(L, intra, Mg)
            Mm, sel_m, m0, m1 = G.search(ML, intra, ch, nch, a.fanin, f"{pk}/{dk} MOP4CLXOR xor_global")
            mxg = G.apply_named(ML, intra, Mm)
            # checks: row/col unchanged, channel unchanged, injective
            ob, orow, oc = L.fields(intra)
            nb, nr, nc_ = L.fields(xg)
            assert (nr == orow).all() and (nc_ == oc).all(), "RoBa xor changed row/col"
            mb0, mr0, mc0 = ML.fields(intra)
            mb1, mr1, mc1 = ML.fields(mxg)
            assert (mr0 == mr1).all() and (mc0 == mc1).all(), "MOP xor changed row/col"
            assert (ob != nb).any() and (mb0 != mb1).any(), "search found nothing"
            nuniq = np.unique(pw).size
            for name, arr in (("id", intra), ("xg", xg), ("mxg", mxg)):
                assert np.unique(L.from_intra(arr, ch.astype(np.uint64))).size == nuniq, f"{name} not injective"
            variants = {"id": (L, intra), "xg": (L, xg), "mxg": (ML, mxg)}
            mets = {"id/RoBa": G.metrics(L, intra, ch, nch, np.zeros(w, dtype=np.int8)),
                    "id/MOP": G.metrics(ML, intra, ch, nch, np.zeros(w, dtype=np.int8)),
                    "xg/RoBa": G.metrics(L, xg, ch, nch, np.zeros(w, dtype=np.int8)),
                    "mxg/MOP": G.metrics(ML, mxg, ch, nch, np.zeros(w, dtype=np.int8))}
            load = np.bincount(ch, minlength=nch)
            for vk, (_lay, arr) in variants.items():
                phys = to_phys(L, arr, ch)
                key = f"{pk}_{dk}_{vk}"
                with open(f"{a.out}/{key}.trc", "w") as f:
                    for i in range(0, w, 2_000_000):
                        f.write("".join(map("LD 0x%x\n".__mod__, phys[i:i + 2_000_000].tolist())))
                for t in range(NT):
                    with open(f"{a.out}/{key}.t{t}.addr", "w") as f:
                        f.writelines(f"{x}\n" for x in phys[:wc][tw_id == t].tolist())
                man["files"].append(dict(key=key, placement=pk, dram=dk, variant=vk))
            info = dict(placement=pk, dram=dk, xg_matrix=sel_g, xg_cost=[g0, g1], mxg_matrix=sel_m, mxg_cost=[m0, m1],
                        static=mets, chan_load_max_over_mean=float(load.max() / load.mean()))
            man.setdefault("info", []).append(info)
            print(f"  {pk}/{dk}: in-order hit id/RoBa {mets['id/RoBa']['inorder_hit']:.3f} id/MOP {mets['id/MOP']['inorder_hit']:.3f} "
                  f"xg {mets['xg/RoBa']['inorder_hit']:.3f} mxg {mets['mxg/MOP']['inorder_hit']:.3f}", flush=True)
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
