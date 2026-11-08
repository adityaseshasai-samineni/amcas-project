#!/usr/bin/env python3
"""Region-aware address mapping vs global baselines, on a real LLC-miss stream.

Usage: region_remap.py MISS_GZ TENSORS_TSV OUT_DIR [--window N] [--start-frac F]
                       [--placements packed,thp] [--drams ddr4,ddr5]

Reads drcachesim's miss file (drmiss_export.read_misses) and the llama-drattr tensor
dump, labels every read with its class (drmiss_export.classify), models physical
placement (drmiss_export.placements), cuts a contiguous window of N requests (a few
layers of one token: the stream is periodic over layers), and for each placement x DRAM
writes one Ramulator trace per scheme, to be replayed with the unchanged RoBaRaCoCh mapper
(ramulator_replay.py --mappers RoBaRaCoCh), as remap.py does:

  identity   no change
  bitrev     remap.py's bit reversal (Shao and Davis style), whole memory
  xor_global remap.py's greedy GF(2) bank-XOR search on the whole window, whole memory
  region     REGION-AWARE (the proposal): each 2 MiB physical granule carries a class tag
             W (weights: dense + expert), K (KV cache) or O (activations + unattributed),
             by majority of the reads it receives. W granules use the identity mapping
             (row-interleave). K and O granules use their own bank-index XOR, bank' = bank
             ^ M.rowbits, with M found by the same greedy search on that class's sub-stream
             only.

Why region is a bijection: the XOR changes only the bank-index field of the address, as a
function of row bits and nothing else. Bank field and column and channel lie below bit 21
of the physical address for both layouts (DDR4: bank [17:13]; DDR5 16 subch at 256 B:
intra bank [16:12] = physical [20:16], channel physical [11:8]), so a 2 MiB granule is
closed under the transform: within it (bank, low row bits, col) -> (bank ^ f(row), same
row, same col) is a permutation, the granule number is untouched, and different granules
never interact. Hence the whole map is a bijection of physical addresses for any granule
tagging. The asserts below check granule, channel, row and column are preserved. Hardware
cost: a small table indexed by physical granule (tag, 2 bits) plus the XOR trees. The OS must
keep each class on its own 2 MiB granules (natural with THP; with 4 KiB pages mixed
granules fall back to the majority class's transform, which is still a bijection but
loses benefit). A granule tag is chosen once (e.g. at allocation), not per access.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remap as R  # noqa: E402
import drmiss_export as X  # noqa: E402
from analyze import LINE  # noqa: E402

GRAN = 21  # granule = 2 MiB
NCH = {"ddr4": 1, "ddr5": 16}
TAGS = ["W", "K", "O"]
PLACE = {"packed": "VA-order packed", "thp": "THP 2MiB shuffled", "4k": "4KiB shuffled"}


def cost2(key, row, nkeys, nch, bus=4.0, tmiss=55.0):
    """remap.cost generalised to nch independent channels (key = channel*nbanks + bank):
    the bus term is per channel."""
    n = len(key)
    order = np.argsort(key, kind="stable")
    b, r = key[order], row[order]
    hit = np.zeros(n, dtype=bool)
    hit[1:] = (b[1:] == b[:-1]) & (r[1:] == r[:-1])
    occ = np.bincount(b, weights=np.where(hit, bus, tmiss), minlength=nkeys)
    near = sum(int(((key[l:] == key[:-l]) & (row[l:] != row[:-l])).sum()) for l in (1, 2, 3))
    return max(bus * n / nch, occ.max()) + 0.05 * occ.sum() + 0.25 * tmiss * near


def search(L, intra, ch, nch, fanin, tag):
    n = len(intra)
    if n > 800_000:
        idx = np.concatenate([np.arange(s, s + 100_000) for s in np.linspace(0, n - 100_000, 8).astype(int)])
        intra, ch = intra[idx], ch[idx]
    cands = [c for c in R.sources(L, False) if len(np.unique((intra >> np.uint64(c[1])) & np.uint64(1))) > 1]
    M = [[] for _ in range(L.bank)]

    def score(M):
        b, r, _ = L.fields(R.apply_matrix(L, intra, M, cands))
        return cost2(ch.astype(np.int64) * L.nbanks + b, r, L.nbanks * nch, nch)

    best = base = score(M)
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
    sel = {j: [cands[c][0] for c in M[j]] for j in range(L.bank)}
    print(f"  [{tag}] n={n:,} cost {base:.0f} -> {best:.0f}; bank bit ^= {sel}")
    # Return the matrix over the full candidate list (names resolved by caller).
    return [[cands[c] for c in M[j]] for j in range(L.bank)], sel, base, best


def apply_named(L, intra, Mn):
    """apply a matrix given as lists of (name, bit) candidates"""
    cands, M = [], []
    for j in range(L.bank):
        row = []
        for c in Mn[j]:
            if c not in cands:
                cands.append(c)
            row.append(cands.index(c))
        M.append(row)
    return R.apply_matrix(L, intra, M, cands)


def bitrev(L, intra):
    w = max(int(intra.max()).bit_length() - L.bank_lo, 1)
    u = intra >> np.uint64(L.bank_lo)
    rev = np.zeros_like(u)
    for k in range(w):
        rev |= ((u >> np.uint64(k)) & np.uint64(1)) << np.uint64(w - 1 - k)
    return (intra & np.uint64((1 << L.bank_lo) - 1)) | (rev << np.uint64(L.bank_lo))


def metrics(L, intra, ch, nch, grp):
    b, r, _ = L.fields(intra)
    key = ch.astype(np.int64) * L.nbanks + b
    out = dict(inorder_hit=R.inorder_hits(key, r) / len(key), banks_used=int(len(np.unique(key))), banks_total=L.nbanks * nch)
    for gi, g in enumerate(TAGS):
        m = grp == gi
        if m.sum() > 200:
            out[f"inorder_hit_{g}"] = R.inorder_hits(key[m], r[m]) / int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss")
    ap.add_argument("tensors")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=12_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--placements", default="packed,thp")
    ap.add_argument("--drams", default="ddr4,ddr5")
    ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _t, addrs = X.read_misses(a.miss)
    va = (addrs // LINE * LINE).astype(np.int64)
    del addrs
    cls, _ = X.classify(va, X.load_tensors(a.tensors))
    # class id -> tag: expert/dense weights W, KV K, activations/unattributed O
    names = X.CLASSES  # ["expert weights","KV cache","dense weights","activations"]
    grp_of = np.array([0 if n in ("expert weights", "dense weights") else 1 if n == "KV cache" else 2 for n in names] + [2], dtype=np.int8)
    grp = grp_of[cls]
    place, _np = X.placements(va)
    n = len(va)
    w = min(a.window, n)
    s0 = min(int(n * a.start_frac), n - w)
    print(f"{n:,} reads; window [{s0:,}, {s0 + w:,}); class shares (W,K,O) full "
          f"{[round(float((grp == i).mean()), 4) for i in range(3)]}, window {[round(float((grp[s0:s0 + w] == i).mean()), 4) for i in range(3)]}")
    manifest = dict(reads_total=n, window=[s0, s0 + w], files=[])
    for pk in a.placements.split(","):
        pa = place[PLACE[pk]].astype(np.uint64)
        gran = (pa >> np.uint64(GRAN)).astype(np.int64)
        ug, inv = np.unique(gran, return_inverse=True)
        cnt = np.zeros((len(ug), 3), dtype=np.int64)
        np.add.at(cnt, (inv, grp.astype(np.int64)), 1)
        gtag = cnt.argmax(1).astype(np.int8)
        tag = gtag[inv]  # class tag of the granule each read lands in
        pure = float((tag == grp).mean())
        print(f"placement {pk}: {len(ug)} granules, tags W/K/O = {[int((gtag == i).sum()) for i in range(3)]}; "
              f"reads whose own class equals their granule's tag {pure:.4f}")
        pw, tw = pa[s0:s0 + w], tag[s0:s0 + w]
        for dk in a.drams.split(","):
            L, nch = R.Layout(dk), NCH[dk]
            intra, ch = L.to_intra(pw)
            variants = {"identity": intra, "bitrev": bitrev(L, intra)}
            info = {}
            Mg, sel, c0, c1 = search(L, intra, ch, nch, a.fanin, f"{pk}/{dk} global")
            variants["xor_global"] = apply_named(L, intra, Mg)
            info["xor_global"] = dict(matrix=sel, cost=[c0, c1])
            reg = intra.copy()
            info["region"] = {}
            for gi in (1, 2):
                m = tw == gi
                mf = tag == gi  # search on the class's sub-stream of the WHOLE token, not just the window
                if mf.sum() < 5_000:
                    info["region"][TAGS[gi]] = "identity (too few reads)"
                    continue
                si, sc = L.to_intra(pa[mf])
                Mr, selr, c0, c1 = search(L, si, sc, nch, a.fanin, f"{pk}/{dk} region {TAGS[gi]}")
                if m.any():
                    reg[m] = apply_named(L, intra[m], Mr)
                info["region"][TAGS[gi]] = dict(matrix=selr, cost=[c0, c1], reads_full=int(mf.sum()), reads_window=int(m.sum()))
            variants["region"] = reg
            for sk, new in variants.items():
                phys = L.from_intra(new, ch)
                if sk in ("xor_global", "region"):
                    assert ((phys >> np.uint64(GRAN)) == (pw >> np.uint64(GRAN))).all(), "granule changed"
                    ob, orow, oc = L.fields(intra)
                    nb, nr, nc = L.fields(new)
                    assert (nr == orow).all() and (nc == oc).all(), "row/col changed"
                fn = f"{pk}_{dk}_{sk}.trc"
                with open(os.path.join(a.out, fn), "w") as f:
                    f.writelines(f"LD 0x{x:x}\n" for x in phys.tolist())
                m = metrics(L, new, ch, nch, tw)
                m.update(placement=pk, dram=dk, scheme=sk, file=fn, info=info.get(sk, {}))
                manifest["files"].append(m)
                print(f"  {pk:<7}{dk:<5}{sk:<11} in-order hit {m['inorder_hit']:.3f} banks {m['banks_used']}/{m['banks_total']}"
                      + "".join(f" {g}:{m.get('inorder_hit_' + g, float('nan')):.3f}" for g in TAGS))
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
