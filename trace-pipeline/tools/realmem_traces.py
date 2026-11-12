#!/usr/bin/env python3
"""Request streams for the 'real memory system' study (validation-owner).

Usage: realmem_traces.py MISS_GZ TENSORS_TSV OUT_DIR [--nch 6] [--window 3000000] [--placements packed,thp,4k]
                         [--schemes identity,xor_global,region] [--start-frac 0.3]

Models the cluster nodes' DDR4: N channels (6 = one Cascade Lake socket, 12 = two), one 32 GB
dual-rank 16Gb x8 DIMM each (intra-channel RoBaRaCoCh: col 7, bank+rank 5, row 17 bits),
channel = (address >> 8) mod N (256 B interleave; N need not be a power of two). Re-uses
remap / region_remap / drmiss_export read-only (the same placements, the same greedy GF(2) bank-XOR search).

Per KEY = <placement>_<scheme>:
  KEY.trc       "LD 0xADDR" saturated stream for Ramulator; the channel sits in a power-of-two field
                (bits [8, 8+W2)) so ramulator's CacheLineInterleave(interleave_bits=2) picks controller `ch`; controllers
                >= N stay idle.
  KEY.tK.addr   per-thread decimal addresses (same encoding), K = 0..3; bubbles are added by realmem_replay.py.
manifest.json: static metrics (in-order hit, channel load, banks used) and the XOR matrices.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remap as R
import drmiss_export as X
import region_remap as G
from analyze import LINE

R.LAYOUTS["ddr4real"] = dict(col=7, bank=5, row=17, s=8, W=0)


class RealLayout(R.Layout):
    """N channels (any N), 256 B interleave; self.W = bits of the power-of-two slot field used for Ramulator."""
    def __init__(self, n):
        super().__init__("ddr4real")
        self.n = n
        self.W = max(1, (n - 1).bit_length()) if n > 1 else 0

    def to_intra(self, p):
        if self.n == 1:
            return p, np.zeros_like(p)
        g = p >> np.uint64(8)
        n = np.uint64(self.n)
        return ((g // n) << np.uint64(8)) | (p & np.uint64(255)), g % n

    def from_intra(self, i, ch):  # linear (non-slot) form, used for the injectivity check only
        if self.n == 1:
            return i
        return (((i >> np.uint64(8)) * np.uint64(self.n) + ch) << np.uint64(8)) | (i & np.uint64(255))

    def to_slot(self, i, ch):  # ramulator encoding with the channel in a pow2 field
        return R.Layout.from_intra(self, i, ch.astype(np.uint64)) if self.n > 1 else i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss"); ap.add_argument("tensors"); ap.add_argument("out")
    ap.add_argument("--nch", type=int, default=6)
    ap.add_argument("--window", type=int, default=3_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--placements", default="packed,thp,4k")
    ap.add_argument("--schemes", default="identity,xor_global,region")
    ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    schemes = a.schemes.split(",")
    tids, addrs = X.read_misses(a.miss)
    va = (addrs // LINE * LINE).astype(np.int64); del addrs
    cls, _ = X.classify(va, X.load_tensors(a.tensors))
    grp_of = np.array([0 if n in ("expert weights", "dense weights") else 1 if n == "KV cache" else 2 for n in X.CLASSES] + [2], dtype=np.int8)
    grp = grp_of[cls]; del cls
    place, _ = X.placements(va)
    n = len(va); w = min(a.window, n); s0 = min(int(n * a.start_frac), n - w)
    NT = 4
    ut = {t: i for i, t in enumerate(np.unique(tids[s0:s0 + w]))}
    if len(ut) == NT:
        tw_id = np.array([ut[t] for t in tids[s0:s0 + w]], dtype=np.int8)
    else:
        print(f"  WARNING: window has {len(ut)} thread(s); 4 contiguous pseudo-threads")
        tw_id = (np.arange(w) * NT // w).astype(np.int8)
    print(f"{n:,} reads, window [{s0:,},{s0 + w:,}), nch {a.nch}", flush=True)
    L = RealLayout(a.nch); nch = a.nch
    man = dict(reads_total=n, window=[s0, s0 + w], nch=nch, slot_bits=L.W, files=[])
    for pk in a.placements.split(","):
        pa = place[G.PLACE[pk]].astype(np.uint64)
        pw = pa[s0:s0 + w]
        if "region" in schemes:
            gran = (pa >> np.uint64(G.GRAN)).astype(np.int64)
            ug, inv = np.unique(gran, return_inverse=True)
            cnt = np.zeros((len(ug), 3), dtype=np.int64); np.add.at(cnt, (inv, grp.astype(np.int64)), 1)
            gtag = cnt.argmax(1).astype(np.int8); tag = gtag[inv]; tw = tag[s0:s0 + w]; del gran, inv
        intra, ch = L.to_intra(pw); ch = ch.astype(np.int64)
        variants, info = {"identity": intra}, {}
        if "xor_global" in schemes:
            Mg, sel, c0, c1 = G.search(L, intra, ch, nch, a.fanin, f"{pk} global")
            variants["xor_global"] = G.apply_named(L, intra, Mg); info["xor_global"] = dict(matrix=sel, cost=[c0, c1])
        if "region" in schemes:
            reg = intra.copy(); info["region"] = {}
            for gi in (1, 2):
                m = tw == gi; mf = tag == gi
                if mf.sum() < 5000:
                    info["region"][G.TAGS[gi]] = "identity"; continue
                si, sc = L.to_intra(pa[mf])
                Mr, selr, c0, c1 = G.search(L, si, sc.astype(np.int64), nch, a.fanin, f"{pk} region {G.TAGS[gi]}")
                if m.any():
                    reg[m] = G.apply_named(L, intra[m], Mr)
                info["region"][G.TAGS[gi]] = dict(matrix=selr, cost=[c0, c1])
            variants["region"] = reg
        for sk in schemes:
            new = variants[sk]
            ob, orow, oc = L.fields(intra); nb, nr, ncl = L.fields(new)
            assert (nr == orow).all() and (ncl == oc).all(), "row/col changed"
            lin = L.from_intra(new, ch.astype(np.uint64))
            assert np.unique(lin).size == np.unique(pw).size, "not injective"
            # the bank hash is a controller-internal function of the address: with a non-power-of-two channel count the
            # linear (mod N) address of a line may leave its 2 MiB page; only injectivity matters (asserted above)
            phys = L.to_slot(new, ch)
            key = f"{pk}_{sk}"
            with open(f"{a.out}/{key}.trc", "w") as f:
                f.writelines(f"LD 0x{x:x}\n" for x in phys.tolist())
            for t in range(NT):
                with open(f"{a.out}/{key}.t{t}.addr", "w") as f:
                    f.writelines(f"{x}\n" for x in phys[tw_id == t].tolist())
            m = G.metrics(L, new, ch, nch, np.zeros(len(phys), dtype=np.int8))
            load = np.bincount(ch, minlength=nch)
            m.update(key=key, placement=pk, scheme=sk, info=info.get(sk, {}), chan_load_max_over_mean=float(load.max() / load.mean()))
            man["files"].append(m)
            print(f"  {key:<20} in-order hit {m['inorder_hit']:.3f} banks {m['banks_used']}/{m['banks_total']} chan max/mean {m['chan_load_max_over_mean']:.3f}", flush=True)
        del variants
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
