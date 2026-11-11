#!/usr/bin/env python3
"""Memory-system / mapping study on a kept LLC-miss stream (memsys agent).

Usage: memsys_study.py MISS_GZ TENSORS_TSV OUT_DIR [--window N|0] [--start-frac F]
           [--placements packed,thp,4k] [--drams ddr5x8,ddr5]
           [--schemes identity,chan,xor_global,xor_global_chan,region,region_chan]

Like region_remap.py (imported, not edited: search, apply_named, metrics, class tags) but
adds the DDR5 8-subchannel layout (remap.LAYOUTS['ddr5x8']), channel-bit hashing, the 4 KiB
placement and a full-stream window (--window 0). One Ramulator LoadStoreTrace per
placement x DRAM x scheme, named <placement>_<dram>_<scheme>.trc, to be replayed with
ramulator_replay.py --preset ddr5x8|ddr5x16 --mappers RoBaRaCoCh.

Schemes (all offline address transforms; row and column never change):
  identity         unchanged (RoBaRaCoCh, channel = physical bits [10:8] at 256 B interleave)
  chan             channel bits ^= XOR of up to K row bits per channel bit (greedy search), banks unchanged
  xor_global       bank bits ^= XOR of up to K row bits (remap.xor_gf2 search), channel unchanged
  xor_global_chan  both: bank search first, then channel search on the bank-hashed stream
  region           region_remap's per-2MiB-granule class XOR (K and O granules), bank only
  region_chan      region, then the global channel hash searched on the result
The channel hash cost is the remap.cost model with the data-bus term taken from the busiest
channel (so an unbalanced channel load is penalised, not just bank conflicts).
Channel hash source bits are the intra-channel row bits (physical bits >= 20), so in a
bijection: (intra address, channel) -> (intra address, channel ^ f(intra row bits)).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remap as R  # noqa: E402
import region_remap as G  # noqa: E402
import drmiss_export as X  # noqa: E402
from analyze import LINE  # noqa: E402

G.NCH["ddr5x8"] = 8
ALL = ["identity", "chan", "xor_global", "xor_global_chan", "region", "region_chan"]


def cost3(ch, bank, row, nbanks, nch, bus=4.0, tmiss=55.0):
    key = ch.astype(np.int64) * nbanks + bank
    n = len(key)
    order = np.argsort(key, kind="stable")
    b, r = key[order], row[order]
    hit = np.zeros(n, dtype=bool)
    hit[1:] = (b[1:] == b[:-1]) & (r[1:] == r[:-1])
    occ = np.bincount(b, weights=np.where(hit, bus, tmiss), minlength=nbanks * nch)
    near = sum(int(((key[l:] == key[:-l]) & (row[l:] != row[:-l])).sum()) for l in (1, 2, 3))
    busiest = np.bincount(ch, minlength=nch).max()
    return max(bus * busiest, occ.max()) + 0.05 * occ.sum() + 0.25 * tmiss * near


def sample(intra, ch, n_chunks=8, size=100_000):
    if len(intra) <= n_chunks * size:
        return intra, ch
    idx = np.concatenate([np.arange(s, s + size) for s in np.linspace(0, len(intra) - size, n_chunks).astype(int)])
    return intra[idx], ch[idx]


def hash_bits(L, intra, M, cands):
    """XOR value per request: bit j = parity of the row source bits listed in M[j]."""
    out = np.zeros(len(intra), dtype=np.int64)
    for j, ms in enumerate(M):
        acc = np.zeros(len(intra), dtype=np.int64)
        for ci in ms:
            acc ^= ((intra >> np.uint64(cands[ci][1])) & np.uint64(1)).astype(np.int64)
        out ^= acc << j
    return out


def chan_search(L, intra, ch, nch, fanin, tag):
    si, sc = sample(intra, ch)
    sc = sc.astype(np.int64)
    cands = [c for c in R.sources(L, False) if len(np.unique((si >> np.uint64(c[1])) & np.uint64(1))) > 1]
    b, r, _ = L.fields(si)
    W = L.W
    M = [[] for _ in range(W)]

    def score(M):
        return cost3(sc ^ hash_bits(L, si, M, cands), b, r, L.nbanks, nch)

    best = base = score(M)
    improved = True
    while improved:
        improved = False
        for j in range(W):
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
    sel = {j: [cands[c][0] for c in M[j]] for j in range(W)}
    print(f"  [{tag}] channel hash cost {base:.0f} -> {best:.0f}; chan bit ^= {sel}")
    return M, cands, sel, base, best


def write_trace(path, phys, chunk=2_000_000):
    with open(path, "w") as f:
        for i in range(0, len(phys), chunk):
            f.write("".join(map("LD 0x%x\n".__mod__, phys[i:i + chunk].tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss")
    ap.add_argument("tensors")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=12_000_000, help="0 = the whole stream")
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--placements", default="thp,4k")
    ap.add_argument("--drams", default="ddr5x8")
    ap.add_argument("--schemes", default=",".join(ALL))
    ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    schemes = a.schemes.split(",")
    os.makedirs(a.out, exist_ok=True)
    _t, addrs = X.read_misses(a.miss)
    va = (addrs // LINE * LINE).astype(np.int64)
    del addrs
    cls, _ = X.classify(va, X.load_tensors(a.tensors))
    grp_of = np.array([0 if n in ("expert weights", "dense weights") else 1 if n == "KV cache" else 2 for n in X.CLASSES] + [2], dtype=np.int8)
    grp = grp_of[cls]
    del cls
    place, _np = X.placements(va)
    n = len(va)
    w = n if a.window == 0 else min(a.window, n)
    s0 = 0 if w == n else min(int(n * a.start_frac), n - w)
    print(f"{n:,} reads; window [{s0:,}, {s0 + w:,}); class shares (W,K,O) window "
          f"{[round(float((grp[s0:s0 + w] == i).mean()), 4) for i in range(3)]}", flush=True)
    manifest = dict(reads_total=n, window=[s0, s0 + w], files=[])
    for pk in a.placements.split(","):
        pa = place[G.PLACE[pk]].astype(np.uint64)
        need_region = any(s.startswith("region") for s in schemes)
        if need_region:
            gran = (pa >> np.uint64(G.GRAN)).astype(np.int64)
            ug, inv = np.unique(gran, return_inverse=True)
            cnt = np.zeros((len(ug), 3), dtype=np.int64)
            np.add.at(cnt, (inv, grp.astype(np.int64)), 1)
            gtag = cnt.argmax(1).astype(np.int8)
            tag = gtag[inv]
            print(f"placement {pk}: {len(ug)} granules, tags W/K/O = {[int((gtag == i).sum()) for i in range(3)]}; "
                  f"reads whose own class equals granule tag {float((tag == grp).mean()):.4f}", flush=True)
            del gran, inv
        pw = pa[s0:s0 + w]
        tw = tag[s0:s0 + w] if need_region else None
        for dk in a.drams.split(","):
            L, nch = R.Layout(dk), G.NCH[dk]
            intra, ch = L.to_intra(pw)
            ch = ch.astype(np.int64)
            variants, info = {}, {}
            variants["identity"] = (intra, ch)
            bank_new = None
            if any(s in schemes for s in ("xor_global", "xor_global_chan")):
                Mg, sel, c0, c1 = G.search(L, intra, ch, nch, a.fanin, f"{pk}/{dk} global bank")
                bank_new = G.apply_named(L, intra, Mg)
                info["xor_global"] = dict(matrix=sel, cost=[c0, c1])
                variants["xor_global"] = (bank_new, ch)
            if "chan" in schemes:
                M, cands, sel, c0, c1 = chan_search(L, intra, ch, nch, a.fanin, f"{pk}/{dk} chan")
                variants["chan"] = (intra, ch ^ hash_bits(L, intra, M, cands))
                info["chan"] = dict(matrix=sel, cost=[c0, c1])
            if "xor_global_chan" in schemes:
                M, cands, sel, c0, c1 = chan_search(L, bank_new, ch, nch, a.fanin, f"{pk}/{dk} bank+chan")
                variants["xor_global_chan"] = (bank_new, ch ^ hash_bits(L, bank_new, M, cands))
                info["xor_global_chan"] = dict(matrix=sel, cost=[c0, c1], bank=info["xor_global"]["matrix"])
            if need_region:
                reg = intra.copy()
                info["region"] = {}
                for gi in (1, 2):
                    m = tw == gi
                    mf = tag == gi
                    if mf.sum() < 5_000:
                        info["region"][G.TAGS[gi]] = "identity (too few reads)"
                        continue
                    si, sc = L.to_intra(pa[mf])
                    Mr, selr, c0, c1 = G.search(L, si, sc, nch, a.fanin, f"{pk}/{dk} region {G.TAGS[gi]}")
                    if m.any():
                        reg[m] = G.apply_named(L, intra[m], Mr)
                    info["region"][G.TAGS[gi]] = dict(matrix=selr, cost=[c0, c1], reads_full=int(mf.sum()), reads_window=int(m.sum()))
                variants["region"] = (reg, ch)
                if "region_chan" in schemes:
                    M, cands, sel, c0, c1 = chan_search(L, reg, ch, nch, a.fanin, f"{pk}/{dk} region+chan")
                    variants["region_chan"] = (reg, ch ^ hash_bits(L, reg, M, cands))
                    info["region_chan"] = dict(matrix=sel, cost=[c0, c1])
            for sk in schemes:
                new, nc_ = variants[sk]
                phys = L.from_intra(new, nc_.astype(np.uint64))
                if sk in ("xor_global", "region"):
                    assert ((phys >> np.uint64(G.GRAN)) == (pw >> np.uint64(G.GRAN))).all(), "granule changed"
                ob, orow, oc = L.fields(intra)
                nb, nr, ncl = L.fields(new)
                assert (nr == orow).all() and (ncl == oc).all(), "row/col changed"
                # bijection: distinct (channel, bank, row, col) stay distinct
                k_new = np.unique(phys).size
                assert k_new == np.unique(pw).size, "not injective"
                fn = f"{pk}_{dk}_{sk}.trc"
                write_trace(os.path.join(a.out, fn), phys)
                m = G.metrics(L, new, nc_, nch, tw if tw is not None else np.zeros(len(phys), dtype=np.int8))
                load = np.bincount(nc_, minlength=nch)
                m.update(placement=pk, dram=dk, scheme=sk, file=fn, info=info.get(sk, {}), chan_load_max_over_mean=float(load.max() / load.mean()))
                manifest["files"].append(m)
                print(f"  {pk:<7}{dk:<7}{sk:<16} in-order hit {m['inorder_hit']:.3f} banks {m['banks_used']}/{m['banks_total']} "
                      f"chan max/mean {m['chan_load_max_over_mean']:.3f}", flush=True)
            del variants
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
