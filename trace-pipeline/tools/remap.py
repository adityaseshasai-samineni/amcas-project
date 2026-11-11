#!/usr/bin/env python3
"""Baseline DRAM address mappings from the literature as offline address transforms.

Usage: remap.py TRACE OUT --scheme {identity,bitrev,xor_gf2} [--dram ddr4|ddr5]
                [--matrix FILE | --matrix-out FILE] [--max-fanin K] [--allow-col]

TRACE/OUT are Ramulator LoadStoreTrace files ("LD 0xADDR" / "ST 0xADDR"). The output is
meant to be replayed with the unchanged RoBaRaCoCh mapper (ramulator_replay.py): the
transformed address, decoded by RoBaRaCoCh, yields the (rank, bankgroup, bank, row, col)
the scheme intends. Run the replay on OUT and read only the RoBaRaCoCh rows.

Bit layout assumed (read from ramulator2 src/ramulator/controller/addr_mapper/impl/
ro_ba_ra_co_ch.cpp and memory_system/channel_mapper/impl/cache_line_interleave.cpp):
  channel mapper: channel = addr[s+W-1:s], s = 6 + interleave_bits; those W bits are
     removed from the address, the rest closes up (intra-channel address).
  RoBaRaCoCh, on the intra-channel address, LSB to MSB:
     [5:0] line | column (post-prefetch bits) | rank | bankgroup | bank | row
  (the name is misleading: rank is the LOWEST of the bank-region bits).
  DDR4  (DDR4_8Gb_x8, 2 ranks, 1 channel, W=0): col 7 bits [12:6], rank [13],
     bankgroup [15:14], bank [17:16], row [33:18]. Bank region = 5 bits [17:13].
  DDR5  (DDR5_16Gb_x8, 1 rank/subchannel, 16 subchannels, interleave_bits=2 so s=8, W=4):
     intra-channel: col 6 bits [11:6], bankgroup [14:12], bank [16:15], row [32:17].
     Bank region = 5 bits [16:12]. The DDR5 org numbers (8 BG x 4 banks, 2^16 rows,
     1024 columns, prefetch 16) are from JEDEC, not read from the Ramulator tables.
The bank region is treated as one 5-bit "bank index" (it is what the in-order row-hit
model in analyze.py uses too).

Schemes:
  identity  copy.
  bitrev    Shao and Davis (2005), bit-reversal mapping: the bits above the column field
            (bank region and row, in the active width of the trace) are reversed in
            order, so the bank index is drawn from the top address bits and the row from
            the bottom ones. This is our reading of the paper; a bijection.
  xor_gf2   Rotaru-style application-specific GF(2) mapping: bank' = bank XOR M.s where
            s is a vector of candidate source bits (all row bits, optionally the upper
            column bits) and M a binary matrix with at most K ones per row. M is found
            by greedy search on a sample of the trace, minimising a cheap cost model
            (see cost()). Bijective for fixed row/col. Saved with --matrix-out.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAYOUTS = {
    # name: line bits, col_bits, bank_bits(region), row_bits, channel shift s, channel width W
    "ddr4": dict(col=7, bank=5, row=16, s=0, W=0),
    "ddr5": dict(col=6, bank=5, row=16, s=8, W=4),
    # 8 independent 32-bit subchannel controllers (4 DIMM channels x 2 subchannels), 256 B interleave:
    # same intra-channel layout as ddr5, 3 channel bits instead of 4 (added by memsys)
    "ddr5x8": dict(col=6, bank=5, row=16, s=8, W=3),
}


class Layout:
    def __init__(self, name):
        d = LAYOUTS[name]
        self.__dict__.update(d)
        self.bank_lo = 6 + self.col
        self.row_lo = self.bank_lo + self.bank
        self.nbanks = 1 << self.bank

    def to_intra(self, p):
        if self.W == 0:
            return p, np.zeros_like(p)
        low = p & np.uint64((1 << self.s) - 1)
        ch = (p >> np.uint64(self.s)) & np.uint64((1 << self.W) - 1)
        return ((p >> np.uint64(self.s + self.W)) << np.uint64(self.s)) | low, ch

    def from_intra(self, i, ch):
        if self.W == 0:
            return i
        low = i & np.uint64((1 << self.s) - 1)
        return ((i >> np.uint64(self.s)) << np.uint64(self.s + self.W)) | (ch << np.uint64(self.s)) | low

    def fields(self, i):
        """(bank index, row, column) of an intra-channel address, RoBaRaCoCh."""
        u = np.uint64
        bank = (i >> u(self.bank_lo)) & u(self.nbanks - 1)
        row = i >> u(self.row_lo)
        col = (i >> u(6)) & u((1 << self.col) - 1)
        return bank.astype(np.int64), row.astype(np.int64), col.astype(np.int64)

    def put_bank(self, i, bank):
        u = np.uint64
        m = u((self.nbanks - 1) << self.bank_lo)
        return (i & ~m) | (bank.astype(np.uint64) << u(self.bank_lo))


def read_trace(path, chunk=2_000_000):
    ops, addrs, ob, ab = [], [], [], []
    with open(path) as f:
        for line in f:
            o, a = line.split()
            ob.append(o == "ST")
            ab.append(int(a, 16))
            if len(ab) >= chunk:
                ops.append(np.array(ob, dtype=bool)); addrs.append(np.array(ab, dtype=np.uint64)); ob, ab = [], []
    ops.append(np.array(ob, dtype=bool)); addrs.append(np.array(ab, dtype=np.uint64))
    return np.concatenate(ops), np.concatenate(addrs)


def write_trace(path, st, addrs):
    with open(path, "w") as f:
        for s, a in zip(st.tolist(), addrs.tolist()):
            f.write(f"{'ST' if s else 'LD'} 0x{a:x}\n")


def inorder_hits(bank, row):
    """Vectorised equivalent of analyze.open_page_stats (in-order open page, per-bank
    previous row): number of hits."""
    order = np.argsort(bank, kind="stable")
    b, r = bank[order], row[order]
    return int(((b[1:] == b[:-1]) & (r[1:] == r[:-1])).sum())


def cost(bank, row, nbanks, bus=4.0, tmiss=55.0):
    """Cheap cycle estimate for a saturated stream: the larger of the data bus time
    (bus cycles per line) and the busiest bank's occupancy (a row hit holds the bank for
    `bus` cycles, a row change for tRC = tmiss), plus 0.05 x total bank occupancy plus a window-conflict term.
    Rewards both row hits and a balanced spread over banks. Also returns the hit rate."""
    n = len(bank)
    order = np.argsort(bank, kind="stable")
    b, r = bank[order], row[order]
    hit = np.empty(n, dtype=bool)
    hit[0] = False
    hit[1:] = (b[1:] == b[:-1]) & (r[1:] == r[:-1])
    occ = np.bincount(b, weights=np.where(hit, bus, tmiss), minlength=nbanks)
    # The max() alone is flat once banks are plentiful; the small sum term keeps row
    # hits (fewer activates, lower latency) in the objective.
    # Window term: requests that land in the same bank as one of the previous 3 requests
    # but in a different row (a row conflict in the making). It gives the greedy search a
    # gradient when no single XOR changes the hit rate or bank balance yet.
    near = sum(int(((bank[l:] == bank[:-l]) & (row[l:] != row[:-l])).sum()) for l in (1, 2, 3))
    return max(bus * n, occ.max()) + 0.05 * occ.sum() + 0.25 * tmiss * near, hit.sum() / n


def sample(intra, n_chunks=8, size=100_000):
    if len(intra) <= n_chunks * size:
        return intra
    starts = np.linspace(0, len(intra) - size, n_chunks).astype(int)
    return np.concatenate([intra[s : s + size] for s in starts])


def sources(L, allow_col):
    """Candidate XOR source bits as (name, bit position in the intra-channel address)."""
    c = [(f"row{k}", L.row_lo + k) for k in range(L.row)]
    if allow_col:
        c += [(f"col{k}", 6 + k) for k in range(2, L.col)]
    return c


def apply_matrix(L, intra, M, cands):
    bank, row, col = L.fields(intra)
    new = bank.copy()
    for j in range(L.bank):
        acc = np.zeros(len(intra), dtype=np.int64)
        for ci in M[j]:
            acc ^= ((intra >> np.uint64(cands[ci][1])) & np.uint64(1)).astype(np.int64)
        new ^= acc << j
    return L.put_bank(intra, new)


def search(L, samp, cands, fanin):
    M = [[] for _ in range(L.bank)]

    def score(M):
        a = apply_matrix(L, samp, M, cands)
        b, r, _ = L.fields(a)
        return cost(b, r, L.nbanks)[0]

    best = score(M)
    print(f"  search: baseline cost {best:.0f}")
    improved = True
    while improved:
        improved = False
        for j in range(L.bank):
            if len(M[j]) >= fanin:
                continue
            trial = None
            for ci in range(len(cands)):
                if ci in M[j]:
                    continue
                M[j].append(ci)
                s = score(M)
                M[j].pop()
                if s < best * 0.999 and (trial is None or s < trial[0]):
                    trial = (s, ci)
            if trial:
                best = trial[0]
                M[j].append(trial[1])
                improved = True
                print(f"  bank bit {j} ^= {cands[trial[1]][0]}  -> cost {best:.0f}")
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--scheme", required=True, choices=["identity", "bitrev", "xor_gf2"])
    ap.add_argument("--dram", default="ddr4", choices=list(LAYOUTS))
    ap.add_argument("--matrix", help="reuse a matrix saved by --matrix-out")
    ap.add_argument("--matrix-out")
    ap.add_argument("--max-fanin", type=int, default=3)
    ap.add_argument("--allow-col", action="store_true", help="also allow upper column bits as XOR sources")
    a = ap.parse_args()
    L = Layout(a.dram)
    st, p = read_trace(a.trace)
    intra, ch = L.to_intra(p)

    def report(tag, i):
        b, r, _ = L.fields(i)
        s = sample(i)
        sb, sr, _ = L.fields(s)
        c, h = cost(sb, sr, L.nbanks)
        used = np.count_nonzero(np.bincount(b, minlength=L.nbanks))
        print(f"{tag:<8} in-order row hit {inorder_hits(b, r) / len(b):.3f} (sample cost {c:.0f}), banks used {used}/{L.nbanks}")

    report("before", intra)
    if a.scheme == "identity":
        new = intra
    elif a.scheme == "bitrev":
        w = max(int(intra.max()).bit_length() - L.bank_lo, 1)
        u = intra >> np.uint64(L.bank_lo)
        rev = np.zeros_like(u)
        for k in range(w):
            rev |= ((u >> np.uint64(k)) & np.uint64(1)) << np.uint64(w - 1 - k)
        new = (intra & np.uint64((1 << L.bank_lo) - 1)) | (rev << np.uint64(L.bank_lo))
        print(f"  bit-reversal over {w} bits above the column field")
    else:
        cands = sources(L, a.allow_col)
        if a.matrix:
            M = json.load(open(a.matrix))["M"]
        else:
            M = search(L, sample(intra), cands, a.max_fanin)
        if a.matrix_out:
            json.dump(dict(dram=a.dram, M=M, sources=[c[0] for c in cands], bank_bit_positions=[L.bank_lo + j for j in range(L.bank)]),
                      open(a.matrix_out, "w"))
        new = apply_matrix(L, intra, M, cands)
        print("  matrix (bank bit j ^= sources):", {j: [cands[c][0] for c in M[j]] for j in range(L.bank)})
    report("after", new)
    # Self-check: the mapping is a bijection on (bank, row, col) and keeps the channel.
    nb, nr, nc = L.fields(new)
    ob, orow, oc = L.fields(intra)
    assert (nc == oc).all(), "column changed"
    if a.scheme != "bitrev":
        assert (nr == orow).all(), "row changed"
        key = (nb.astype(np.int64) << 40) | (nr << 8) | nc
        okey = (ob.astype(np.int64) << 40) | (orow << 8) | oc
        # distinct inputs stay distinct (injective): compare unique counts
        assert len(np.unique(key)) == len(np.unique(okey)), "not injective"
    write_trace(a.out, st, L.from_intra(new, ch))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
