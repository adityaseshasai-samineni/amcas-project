#!/usr/bin/env python3
"""Request streams for the full mapping x row-policy x trace grid (grid-owner).

Usage: grid_gen.py MISS_GZ TENSORS_TSV OUT_DIR [--window 4000000] [--start-frac 0.3] [--placements packed,thp]
                   [--drams d4r6,x16] [--fanin 3] [--procs 4]

Imports remap / region_remap / memsys_study / mop4_traces / moe_remap / mapprep / realmem_traces read-only
(nothing edited). One window of --window requests from --start-frac of the step. For every placement x DRAM
(KEY = <placement>_<dram>) it writes one saturated Ramulator trace per STREAM and, for the Region policy, a
ranges file per stream; manifest.json holds the static metrics and OFFLINE ORACLE counts per mapping.

DRAM: d4r6 = DDR4-2933 6 channels (validation-owner's real-node model, channel = (addr>>8) mod 6), x16 = DDR5-4800
16 sub-channels, 256 B channel interleave (as mop4_replay.py).

Streams (address transforms; row/col never change, injective, asserted):
  id         identity            xg    xor_global (bank ^= <=3 row bits, greedy search on the window)
  region     per-2MiB-granule class XOR (K and O granules searched on their own sub-stream, W identity)
  chan       channel ^= row-bit hash (x16: XOR; d4r6: channel = (channel + hash) mod 6, still a permutation)
  xgc        xg then chan search
  bitrev     bit reversal of the bank/row/col field above the line bits (remap.py's / Shao-Davis)
  naive      bank bit j ^= row bit j (permutation-based interleaving, Zhang et al.)
  mxg        MOP4CLXOR + xor_global (search on the MOP4CLXOR-decoded fields, mop4_traces)
  eh, el     expert_hash / expert_lin (MoE streams only, moe_remap.expert_xor)
Mappings = (stream, Ramulator mapper): identity (id,RoBaRaCoCh) xor_global (xg) region chan xor_global_chan (xgc)
  bitrev naive expert_hash expert_lin (all RoBaRaCoCh), mop (id, MOP4CLXOR) mopxor (mxg, MOP4CLXOR) chra (id, ChRaBaRoCo).
"""
import argparse, json, multiprocessing, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remap as R
import drmiss_export as X
import region_remap as G
import memsys_study as M
import mop4_traces as MT
import mapprep as P
from analyze import LINE

U = np.uint64
MAPPINGS = [("identity", "id", "RoBaRaCoCh"), ("xor_global", "xg", "RoBaRaCoCh"), ("region", "region", "RoBaRaCoCh"),
            ("chan", "chan", "RoBaRaCoCh"), ("xor_global_chan", "xgc", "RoBaRaCoCh"), ("mop", "id", "MOP4CLXOR"),
            ("mopxor", "mxg", "MOP4CLXOR"), ("bitrev", "bitrev", "RoBaRaCoCh"), ("chra", "id", "ChRaBaRoCo"),
            ("naive", "naive", "RoBaRaCoCh"), ("expert_hash", "eh", "RoBaRaCoCh"), ("expert_lin", "el", "RoBaRaCoCh")]
# DRAM timing (memory cycles) for the modelled offline latency: tCL (hit), tRCD, tRP (+ burst folded into tCL)
TIM = {"d4r6": dict(cl=21, rcd=21, rp=21), "x16": dict(cl=40, rcd=39, rp=39)}
NCH = {"d4r6": 6, "x16": 16}
SH = {}  # shared (forked) data


class ChRaView:
    """ChRaBaRoCo decode of an intra-channel address: off | col | row | bank(+rank) (bank on top)."""
    def __init__(self, L):
        self.L = L

    def __getattr__(self, k):
        return getattr(self.L, k)

    def fields(self, i):
        L = self.L
        col = (i >> U(L.off)) & U((1 << L.col) - 1)
        row = (i >> U(L.off + L.col)) & U((1 << L.row) - 1)
        bank = (i >> U(L.off + L.col + L.row)) & U(L.nbanks - 1)
        return bank.astype(np.int64), row.astype(np.int64), col.astype(np.int64)


def chan_search(L, intra, ch, nch, fanin, tag, mod6):
    """memsys_study.chan_search with the combine step generalised: ch' = (ch + h) mod nch (nch not a power of two) or ch ^ h."""
    si, sc = M.sample(intra, ch)
    sc = sc.astype(np.int64)
    cands = [c for c in R.sources(L, False) if len(np.unique((si >> U(c[1])) & U(1))) > 1]
    b, r, _ = L.fields(si)
    W = L.W
    Mx = [[] for _ in range(W)]
    comb = (lambda c, h: (c + h) % nch) if mod6 else (lambda c, h: c ^ h)

    def score(Mx):
        return M.cost3(comb(sc, M.hash_bits(L, si, Mx, cands)), b, r, L.nbanks, nch)

    best = base = score(Mx)
    improved = True
    while improved:
        improved = False
        for j in range(W):
            if len(Mx[j]) >= fanin:
                continue
            trial = None
            for ci in range(len(cands)):
                if ci in Mx[j]:
                    continue
                Mx[j].append(ci); s = score(Mx); Mx[j].pop()
                if s < best * 0.999 and (trial is None or s < trial[0]):
                    trial = (s, ci)
            if trial:
                best = trial[0]; Mx[j].append(trial[1]); improved = True
    sel = {j: [cands[c][0] for c in Mx[j]] for j in range(W)}
    print(f"  [{tag}] channel hash cost {base:.0f} -> {best:.0f}; chan bit ^= {sel}", flush=True)
    return Mx, cands, sel, comb


def offline(view, intra, ch, nch, dk):
    """In-order per-bank classification (stream order): Open = hits / conflicts / cold; Oracle (next access of the bank decides
    keep-open vs close): same hits, every other access a miss (no conflict); Closed (always precharge): all misses."""
    b, r, _ = view.fields(intra)
    key = np.asarray(ch).astype(np.int64) * view.nbanks + b
    n = len(key)
    order = np.argsort(key, kind="stable")
    kk, rr = key[order], r[order]
    first = np.ones(n, dtype=bool); first[1:] = kk[1:] != kk[:-1]
    hit = np.zeros(n, dtype=bool); hit[1:] = (kk[1:] == kk[:-1]) & (rr[1:] == rr[:-1])
    nh = int(hit.sum()); nc = int(first.sum()); nf = n - nh - nc
    t = TIM[dk]
    lat = lambda h, m, c: (h * t["cl"] + m * (t["rcd"] + t["cl"]) + c * (t["rp"] + t["rcd"] + t["cl"])) / n
    return dict(n=n, hits=nh, conflicts=nf, cold=nc, hit_rate=nh / n, conflict_rate=nf / n,
                lat_open=lat(nh, nc, nf), lat_closed=lat(0, n, 0), lat_oracle=lat(nh, n - nh, 0),
                oracle_hit_rate=nh / n, banks_used=int(len(np.unique(key))), banks_total=view.nbanks * nch)


def write_trace(path, phys, chunk=2_000_000):
    with open(path, "w") as f:
        for i in range(0, len(phys), chunk):
            f.write("".join(map("LD 0x%x\n".__mod__, phys[i:i + chunk].tolist())))


def write_ranges(path, phys, grp):
    g = (phys >> U(G.GRAN)).astype(np.int64)
    ug, inv = np.unique(g, return_inverse=True)
    cnt = np.zeros((len(ug), 3), dtype=np.int64)
    np.add.at(cnt, (inv, grp.astype(np.int64)), 1)
    tg = "WKO"
    rows = []
    for gi, c in zip(ug.tolist(), cnt.argmax(1).tolist()):
        c = tg[c]
        if rows and rows[-1][2] == c and rows[-1][1] == (gi << G.GRAN):
            rows[-1][1] = (gi + 1) << G.GRAN
        else:
            rows.append([gi << G.GRAN, (gi + 1) << G.GRAN, c])
    with open(path, "w") as f:
        f.writelines(f"{s} {e} {c}\n" for s, e, c in rows)
    return len(rows)


def sub_sample(pa, mf, n_chunks=8, size=100_000):
    idx = np.flatnonzero(mf)
    if len(idx) > n_chunks * size:
        idx = idx[np.concatenate([np.arange(s, s + size) for s in np.linspace(0, len(idx) - size, n_chunks).astype(int)])]
    return pa[idx]


def combo(args):
    pk, dk, out, fanin = args
    a = SH
    L, nch = MT.make_layout(dk)
    mod6 = dk == "d4r6"
    ML, CV = MT.MopView(L), ChRaView(L)
    pa, tag = a["pa"][pk], a["tag"][pk]
    s0, s1 = a["win"]
    pw, tw = pa[s0:s1], tag[s0:s1]
    gw = a["grp"][s0:s1]
    intra, ch = L.to_intra(pw)
    ch = ch.astype(np.int64)
    S = {"id": (L, intra, ch)}
    info = {}
    Mg, sel, c0, c1 = G.search(L, intra, ch, nch, fanin, f"{pk}/{dk} xor_global")
    xg = G.apply_named(L, intra, Mg); S["xg"] = (L, xg, ch); info["xor_global"] = dict(matrix=sel, cost=[c0, c1])
    # region
    reg = intra.copy(); info["region"] = {}
    for gi in (1, 2):
        m = tw == gi
        mf = tag == gi
        if mf.sum() < 5_000 or not m.any():
            info["region"][G.TAGS[gi]] = "identity (too few reads)"; continue
        si, sc = L.to_intra(sub_sample(pa, mf))
        Mr, selr, c0, c1 = G.search(L, si, sc.astype(np.int64), nch, fanin, f"{pk}/{dk} region {G.TAGS[gi]}")
        reg[m] = G.apply_named(L, intra[m], Mr)
        info["region"][G.TAGS[gi]] = dict(matrix=selr, cost=[c0, c1], reads_window=int(m.sum()))
    S["region"] = (L, reg, ch)
    # channel hashes
    Mc, cands, selc, comb = chan_search(L, intra, ch, nch, fanin, f"{pk}/{dk} chan", mod6)
    S["chan"] = (L, intra, comb(ch, M.hash_bits(L, intra, Mc, cands))); info["chan"] = dict(matrix=selc)
    Mc2, cands2, selc2, comb = chan_search(L, xg, ch, nch, fanin, f"{pk}/{dk} bank+chan", mod6)
    S["xgc"] = (L, xg, comb(ch, M.hash_bits(L, xg, Mc2, cands2))); info["xor_global_chan"] = dict(matrix=selc2, bank=info["xor_global"]["matrix"])
    S["bitrev"] = (L, G.bitrev(L, intra), ch)
    S["naive"] = (L, G.apply_named(L, intra, [[(f"row{j}", L.row_lo + j)] for j in range(L.bank)]), ch)
    Mm, selm, m0, m1 = G.search(ML, intra, ch, nch, fanin, f"{pk}/{dk} MOP4CLXOR xor_global")
    S["mxg"] = (ML, G.apply_named(ML, intra, Mm), ch); info["mopxor"] = dict(matrix=selm, cost=[m0, m1])
    if a["moe"]:
        e, ly, kd = a["ef"]
        import moe_remap as MR
        S["eh"] = (L, MR.expert_xor(L, intra, e, ly, kd, "hash"), ch)
        S["el"] = (L, MR.expert_xor(L, intra, e, ly, kd, "lin"), ch)
    # checks
    ob, orow, oc = L.fields(intra)
    nuniq = np.unique(pw).size
    for sk, (lay, arr, c_) in S.items():
        if sk == "bitrev":
            assert np.unique(L.from_intra(arr, c_.astype(U))).size == nuniq, f"{sk} not injective"
            continue
        v = ML if sk == "mxg" else L
        b0, r0, c0_ = v.fields(intra); b1, r1, c1_ = v.fields(arr)
        assert (r0 == r1).all() and (c0_ == c1_).all(), f"{sk} changed row/col"
        if sk in ("chan", "xgc"):
            # channel permuted per intra address: injectivity on (intra, ch) pairs
            kk = arr.astype(np.int64) * 16 + c_
            k0 = intra.astype(np.int64) * 16 + ch
            assert np.unique(kk).size == np.unique(k0).size, f"{sk} not injective"
        else:
            assert np.unique(L.from_intra(arr, c_.astype(U))).size == nuniq, f"{sk} not injective"
    man = dict(key=f"{pk}_{dk}", placement=pk, dram=dk, info=info, mappings={})
    for sk, (lay, arr, c_) in S.items():
        phys = MT.to_phys(L, arr, c_)
        write_trace(f"{out}/{pk}_{dk}_{sk}.trc", phys)
        man.setdefault("ranges", {})[sk] = write_ranges(f"{out}/{pk}_{dk}_{sk}.ranges", phys, gw)
    if dk == "d4r6":
        # DRAMsim3 has no MOP4CLXOR / ChRaBaRoCo: express them as an address transform so its fixed rochrababgco decode sees the
        # EFFECTIVE (bank, row, col) of the mapper (a bijection of the intra-channel address; asserted)
        for name, view, sk in (("mop3", ML, "id"), ("mxg3", ML, "mxg"), ("chra3", CV, "id")):
            arr, c_ = S[sk][1], S[sk][2]
            b_, r_, c2 = view.fields(arr)
            low = arr & U((1 << L.off) - 1)
            new = (r_.astype(U) << U(L.row_lo)) | (b_.astype(U) << U(L.bank_lo)) | (c2.astype(U) << U(L.off)) | low
            assert np.unique(new).size == np.unique(arr).size, f"{name} not a bijection"
            write_trace(f"{out}/{pk}_{dk}_{name}.trc", MT.to_phys(L, new, c_))
    load = {}
    for mname, sk, mapper in MAPPINGS:
        if sk not in S:
            continue
        lay, arr, c_ = S[sk]
        view = ML if mapper == "MOP4CLXOR" else CV if mapper == "ChRaBaRoCo" else L
        o = offline(view, arr, c_, nch, dk)
        cl = np.bincount(np.asarray(c_).astype(np.int64), minlength=nch)
        o["chan_load_max_over_mean"] = float(cl.max() / cl.mean()) if dk != "d4r6" else float(cl[:6].max() / cl[:6].mean())
        o.update(stream=sk, mapper=mapper)
        man["mappings"][mname] = o
    json.dump(man, open(f"{out}/{pk}_{dk}.manifest.json", "w"), indent=1)
    print(f"  done {pk}/{dk}: " + " ".join(f"{k}:{v['hit_rate']:.3f}/{v['oracle_hit_rate']:.3f}" for k, v in man["mappings"].items()), flush=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss"); ap.add_argument("tensors"); ap.add_argument("out")
    ap.add_argument("--window", type=int, default=4_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--placements", default="packed,thp")
    ap.add_argument("--drams", default="d4r6,x16")
    ap.add_argument("--fanin", type=int, default=3)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tids, addrs = X.read_misses(a.miss); del tids
    va = (addrs // LINE * LINE).astype(np.int64); del addrs
    rows = X.load_tensors(a.tensors)
    cls, tix = X.classify(va, rows)
    grp_of = np.array([0 if n in ("expert weights", "dense weights") else 1 if n == "KV cache" else 2 for n in X.CLASSES] + [2], dtype=np.int8)
    grp = grp_of[cls]
    n = len(va); w = min(a.window, n); s0 = min(int(n * a.start_frac), n - w)
    moe = bool(((cls == 0)).any())
    SH["moe"] = moe
    if moe:
        sl = slice(s0, s0 + w)
        SH["ef"] = P.expert_features(dict(rows=rows, cls=cls[sl], tix=tix[sl], va=va[sl]))
    del cls, tix
    try:
        place, _ = X.placements(va)
    except SystemExit:
        X.CAPACITY = 64 << 30
        place, _ = X.placements(va)
        print("  capacity raised to 64 GiB", flush=True)
    del va
    SH.update(win=(s0, s0 + w), grp=grp, pa={}, tag={})
    for pk in a.placements.split(","):
        pa = place[G.PLACE[pk]].astype(U)
        gran = (pa >> U(G.GRAN)).astype(np.int64)
        ug, inv = np.unique(gran, return_inverse=True)
        cnt = np.zeros((len(ug), 3), dtype=np.int64)
        np.add.at(cnt, (inv, grp.astype(np.int64)), 1)
        SH["tag"][pk] = cnt.argmax(1).astype(np.int8)[inv]
        SH["pa"][pk] = pa
        del gran, inv
    del place
    print(f"{n:,} reads, window [{s0:,},{s0 + w:,}), moe={moe}, window class shares W/K/O "
          f"{[round(float((grp[s0:s0 + w] == i).mean()), 4) for i in range(3)]}", flush=True)
    SH["grp"] = grp
    jobs = [(pk, dk, a.out, a.fanin) for pk in a.placements.split(",") for dk in a.drams.split(",")]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(min(a.procs, len(jobs))) as pool:
        mans = pool.map(combo, jobs, chunksize=1)
    json.dump(dict(reads_total=n, window=[s0, s0 + w], moe=moe, combos=mans), open(f"{a.out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
