#!/usr/bin/env python3
"""DRAM-technology study: replay an attributed CPU LLC-miss stream on LPDDR5/6, HBM3/4, GDDR6 (dramtech-owner).

Usage: dramtech_study.py MISS_GZ TENSORS_TSV OUT_DIR --systems lpddr5x32,hbm3x16,... [--placements packed,thp,4k]
         [--schemes identity,xor_global,chan,xor_global_chan,region] [--window 8000000] [--o3-window 2000000]
         [--points sat,fast,wide,nat4] [--policies Open,ClosedCAP] [--bubbles 52] [--procs 6] [--tmp /scratch/..]
Needs PYTHONPATH=ramulator2-amcas/python, LD_LIBRARY_PATH=ramulator2-amcas.

Per system (dramtech_cfg.SYSTEMS) and placement, the same window of the miss stream (start 30%) is mapped with the
system's RoBaRaCoCh layout (remap.LAYOUTS, registered by dramtech_cfg.register), hashed (identity / bank XOR
search / channel hash / both / region; the searches are region_remap.search and memsys_study.chan_search imported
unchanged, with the cost model's bus/tRC terms set from the standard's timing), written as sector-expanded traces
(each 64 B line = two 32 B requests addr, addr+32: the new standards have 32 B transactions) and replayed.

Points: sat = LoadStoreTrace (saturated, issue rate 2x the system's peak request rate); name:k:window = SimpleO3, 4 cores,
ipc 4, core clock = round(2.4 GHz / tick) x tick, instructions per line T = k*(BUBBLES+1) * f_core/2.4 GHz (calibrated
per policy-owner part 2/5: fast = 1:256, wide = 1:1024, nat4 = 4.5:1152 at 2.4 GHz, window scaled by the same factor),
32 B LLC lines, 32 MSHRs per core (two sectors per line). Time = core cycles / ratio, in memory ticks.
"""
import argparse, json, multiprocessing, os, shutil, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remap as R
import drmiss_export as X
import region_remap as G
import memsys_study as M
import dramtech_cfg as C
from analyze import LINE

C.register(R, G)
POINTS = {"fast": (1.0, 256), "wide": (1.0, 1024), "nat4": (4.5, 1152)}
NT = 4
SECT = 32


def write_sat(path, phys, chunk=1_000_000):
    with open(path, "w") as f:
        for i in range(0, len(phys), chunk):
            p = phys[i:i + chunk]
            a = np.empty(2 * len(p), dtype=np.uint64); a[0::2] = p; a[1::2] = p + np.uint64(SECT)
            f.write("".join(map("LD 0x%x\n".__mod__, a.tolist())))


def replay(task):
    import ramulator
    sd, d, key, point, pol, a = task
    mem = C.build_mem(sd, pol)
    tick = C.tick_ps(sd)
    tmp = None
    if point == "sat":
        fe = ramulator.frontend.LoadStoreTrace(clock_ratio=sd["issue"], path=f"{d}/{key}.trc")
        r = 1; T = 0
    else:
        k, win = POINTS[point]
        r = max(1, round(2.4 * tick / 1e3))  # core cycles per memory tick: core clock = r / tick, nearest to 2.4 GHz
        fcore = r * 1e3 / tick  # GHz
        s = k * fcore / 2.4
        T = max(2, round(s * (a.bubbles + 1)))     # instructions per 64 B line (incl. its two memory ops)
        wdepth = max(8, round(win * fcore / 2.4))
        tmp = os.path.join(a.tmp, f"o3_{os.getpid()}_{key}_{point}_{pol}")
        os.makedirs(tmp, exist_ok=True)
        tr = []
        for t in range(NT):
            p = f"{tmp}/t{t}.trc"
            with open(p, "w") as f, open(f"{d}/{key}.t{t}.addr") as g:
                for x in g:
                    x = int(x)
                    f.write(f"{T - 2} {x}\n0 {x + SECT}\n")
            tr.append(p)
        nl = min(sum(1 for _ in open(f"{d}/{key}.t{t}.addr")) for t in range(NT))
        fe = ramulator.frontend.SimpleO3(
            clock_ratio=r, num_expected_insts=nl * T, traces=tr, ipc=4, inst_window_depth=wdepth,
            llc_capacity_per_core="4KB", llc_associativity=4, llc_linesize=SECT, llc_num_mshr_per_core=32,
            translation=ramulator.translation.NoTranslation(max_addr=1 << 40))
    sim = ramulator.Simulation(fe, mem)
    sim.run()
    cs = sim.stats["memory_system"]["controller"]
    cs = cs if isinstance(cs, list) else [cs]
    tot = lambda kk: sum(c[kk] for c in cs)
    n = max(tot("row_hits") + tot("row_misses") + tot("row_conflicts"), 1)
    served = max(sum(c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs), 1)
    mcyc = max(c["cycles"] for c in cs)
    cyc = mcyc
    if point != "sat":
        fs = sim.stats["frontend"]
        cyc = max(fs[f"cycles_recorded_core_{i}"] for i in range(NT)) / r
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    lat = sum(c["avg_read_latency"] * c.get("num_read_reqs_served", c["num_read_reqs"]) for c in cs if c["num_read_reqs"] > 0) / served
    return dict(key=key, point=point, policy=pol, T=T, reads=tot("num_read_reqs"), cycles=cyc, mem_cycles=mcyc, tick_ps=tick,
                time_us=cyc * tick / 1e6, hit_rate=tot("row_hits") / n, conflict_rate=tot("row_conflicts") / n,
                avg_read_latency_ticks=lat, avg_read_latency_ns=lat * tick / 1e3,
                GBps=tot("num_read_reqs") * sd["tx"] / (cyc * tick / 1e3) if cyc else 0.0,
                throughput_MBps=sum(c.get("total_throughput_MBps", 0.0) for c in cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss"); ap.add_argument("tensors"); ap.add_argument("out")
    ap.add_argument("--systems", required=True)
    ap.add_argument("--placements", default="packed,thp,4k")
    ap.add_argument("--schemes", default="identity,xor_global,chan,xor_global_chan")
    ap.add_argument("--window", type=int, default=8_000_000)
    ap.add_argument("--o3-window", type=int, default=2_000_000)
    ap.add_argument("--start-frac", type=float, default=0.3)
    ap.add_argument("--points", default="sat,fast,nat4")
    ap.add_argument("--policies", default="Open,ClosedCAP")
    ap.add_argument("--bubbles", type=int, default=52)
    ap.add_argument("--fanin", type=int, default=3)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tmp", default="/scratch")
    ap.add_argument("--cost", default="auto", choices=["auto", "default"])
    ap.add_argument("--keep", action="store_true")
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
    ut = {t: i for i, t in enumerate(np.unique(tids[s0:s0 + w]))}
    if len(ut) == NT:
        tw_id = np.array([ut[t] for t in tids[s0:s0 + w]], dtype=np.int8)
    else:
        print(f"  WARNING: window has {len(ut)} thread(s); 4 contiguous pseudo-threads", flush=True)
        tw_id = (np.arange(w) * NT // w).astype(np.int8)
    print(f"{n:,} reads, window [{s0:,},{s0 + w:,}), class shares W/K/O {[round(float((grp[s0:s0 + w] == i).mean()), 4) for i in range(3)]}", flush=True)
    o3n = min(a.o3_window, w)
    if len(ut) != NT:  # pseudo-threads must cover the o3 prefix, not the whole window
        tw_id = (np.arange(w) * NT // o3n % NT).astype(np.int8) if False else np.minimum(np.arange(w) * NT // o3n, NT - 1).astype(np.int8)
    for name in a.systems.split(","):
        sd = C.SYSTEMS[name]
        outj = f"{a.out}/{name}.json"
        if os.path.exists(outj):
            print(f"{name}: done already, skipping"); continue
        L = R.Layout(sd["layout"]); nch = sd["nch"]
        if a.cost == "auto":
            bus, tm = C.cost_params(sd)
            G.cost2.__defaults__ = (bus, tm); M.cost3.__defaults__ = (bus, tm)
        print(f"=== {name}: {sd['what']}; layout {sd['layout']} cost bus/tRC = {C.cost_params(sd) if a.cost == 'auto' else 'default'}", flush=True)
        S = f"{a.tmp}/dramtech_{name}_{os.getpid()}"; shutil.rmtree(S, ignore_errors=True); os.makedirs(S)
        man = dict(system=name, what=sd["what"], layout=sd["layout"], nch=nch, window=[s0, s0 + w], files=[])
        t0 = time.time()
        for pk in a.placements.split(","):
            pa = place[G.PLACE[pk]].astype(np.uint64)
            pw = pa[s0:s0 + w]
            tag = tw = None
            if "region" in schemes:
                gran = (pa >> np.uint64(G.GRAN)).astype(np.int64)
                ug, inv = np.unique(gran, return_inverse=True)
                cnt = np.zeros((len(ug), 3), dtype=np.int64); np.add.at(cnt, (inv, grp.astype(np.int64)), 1)
                gtag = cnt.argmax(1).astype(np.int8); tag = gtag[inv]; tw = tag[s0:s0 + w]; del gran, inv
            assert int(pw.max()) < (1 << (L.off + L.col + L.bank + L.row + L.W)), "physical address beyond the modelled capacity"
            intra, ch = L.to_intra(pw); ch = ch.astype(np.int64)
            variants, info = {"identity": (intra, ch)}, {}
            bank_new = None
            if any(s in schemes for s in ("xor_global", "xor_global_chan")):
                Mg, sel, c0, c1 = G.search(L, intra, ch, nch, a.fanin, f"{name}/{pk} global bank")
                bank_new = G.apply_named(L, intra, Mg)
                variants["xor_global"] = (bank_new, ch); info["xor_global"] = dict(matrix=sel, cost=[c0, c1])
            if "chan" in schemes:
                Mc, cands, sel, c0, c1 = M.chan_search(L, intra, ch, nch, a.fanin, f"{name}/{pk} chan")
                variants["chan"] = (intra, ch ^ M.hash_bits(L, intra, Mc, cands)); info["chan"] = dict(matrix=sel, cost=[c0, c1])
            if "xor_global_chan" in schemes:
                Mc, cands, sel, c0, c1 = M.chan_search(L, bank_new, ch, nch, a.fanin, f"{name}/{pk} bank+chan")
                variants["xor_global_chan"] = (bank_new, ch ^ M.hash_bits(L, bank_new, Mc, cands))
                info["xor_global_chan"] = dict(matrix=sel, cost=[c0, c1], bank=info["xor_global"]["matrix"])
            if "region" in schemes:
                reg = intra.copy(); info["region"] = {}
                for gi in (1, 2):
                    m = tw == gi; mf = tag == gi
                    if mf.sum() < 5_000:
                        info["region"][G.TAGS[gi]] = "identity (too few reads)"; continue
                    si, sc = L.to_intra(pa[mf])
                    Mr, selr, c0, c1 = G.search(L, si, sc.astype(np.int64), nch, a.fanin, f"{name}/{pk} region {G.TAGS[gi]}")
                    if m.any():
                        reg[m] = G.apply_named(L, intra[m], Mr)
                    info["region"][G.TAGS[gi]] = dict(matrix=selr, cost=[c0, c1])
                variants["region"] = (reg, ch)
            for sk in schemes:
                new, nc_ = variants[sk]
                phys = L.from_intra(new, nc_.astype(np.uint64))
                ob, orow, oc = L.fields(intra); nb, nr, ncl = L.fields(new)
                assert (nr == orow).all() and (ncl == oc).all(), "row/col changed"
                assert np.unique(phys).size == np.unique(pw).size, "not injective"
                if sk in ("xor_global", "region"):
                    assert ((phys >> np.uint64(G.GRAN)) == (pw >> np.uint64(G.GRAN))).all(), "granule changed"
                key = f"{pk}_{sk}"
                if "sat" in a.points.split(","):
                    write_sat(f"{S}/{key}.trc", phys)
                for t in range(NT):
                    sel_t = (tw_id[:o3n] == t)
                    with open(f"{S}/{key}.t{t}.addr", "w") as f:
                        f.writelines(f"{x}\n" for x in phys[:o3n][sel_t].tolist())
                mm = G.metrics(L, new, nc_, nch, tw if tw is not None else np.zeros(len(phys), dtype=np.int8))
                load = np.bincount(nc_, minlength=nch)
                mm.update(key=key, placement=pk, scheme=sk, info=info.get(sk, {}), chan_load_max_over_mean=float(load.max() / load.mean()))
                man["files"].append(mm)
                print(f"  {key:<24} in-order hit {mm['inorder_hit']:.3f} banks {mm['banks_used']}/{mm['banks_total']} chan max/mean {mm['chan_load_max_over_mean']:.3f}", flush=True)
            del variants
        print(f"  traces written in {time.time() - t0:.0f} s", flush=True)
        keys = [f"{pk}_{sk}" for pk in a.placements.split(",") for sk in schemes]
        tasks = [(sd, S, k, pt, pol, a) for k in keys for pt in a.points.split(",") for pol in a.policies.split(",")]
        res = []
        with multiprocessing.Pool(a.procs) as pool:
            for r in pool.imap_unordered(replay, tasks):
                res.append(r)
                print(f"  {r['key']:<24}{r['point']:<6}{r['policy']:<10}hit {r['hit_rate']:.3f} lat {r['avg_read_latency_ns']:7.1f}ns "
                      f"{r['cycles'] / 1e6:8.3f}Mt {r['GBps']:7.1f}GB/s", flush=True)
        json.dump(dict(manifest=man, results=res, args=vars(a)), open(outj, "w"), indent=1)
        if not a.keep:
            shutil.rmtree(S, ignore_errors=True)
        print(f"  {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
