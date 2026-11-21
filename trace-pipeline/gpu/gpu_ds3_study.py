#!/usr/bin/env python3
"""Replay one kept GPU DRAM stream (PREFIX.dram[.gz]) in Ramulator 2 and DRAMsim3 with the same remapped addresses (gpu-ds3-owner).

usage: gpu_ds3_study.py PREFIX OUT_JSON --gpu {v100,a16} [--sims ram,ds3] [--ib 2] [--schemes identity,xor,chan,xor_chan]
                        [--window 80000000] [--procs 8] [--tmp DIR]
Imports gpu_replay.py (unchanged): same stream window, 2 MiB first-touch page packing, channel interleave 64 B << ib, schemes,
Ramulator systems (HBM2 32 ch x 2 PC for v100, GDDR6 8 x16 ch for a16; Open / ClosedCAP).
DRAMsim3: one instance per HBM2 pseudo-channel (64) or per GDDR6 channel (8), each fed its own request sub-stream in order, one
request per cycle when its transaction queue accepts (saturated, run to drain; tools/gpu_ds3_drain.cc); OPEN_PAGE / CLOSE_PAGE.
Instances are independent: cycles = slowest instance; hit = (read+write row hits)/(requests); latency = request-weighted mean read
latency in cycles.  The address handed to DRAMsim3 is composed from the same (channel, pc, bank, row, column) the Ramulator layout
decodes, so both simulators see identical bank/row/column assignment; its address_mapping 'rochrababgco' = MSB row | bank | bank-group | column.
"""
import argparse, json, multiprocessing, os, re, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_replay as GR
import remap as R
import dramtech_cfg as C
import memsys_study as MS
import gpu_io

D3 = os.path.expanduser("~/amcas-pro/tracing/DRAMsim3")
DRV = os.path.expanduser("~/amcas-pro/tracing/bin/gpu_ds3_drain")
# DRAMsim3 systems. v100: HBM2 pseudo-channel instance (64-bit, BL4 = 32 B, 16 banks = 4 BG x 4, 1 KiB page = 5 column bits, 16384 rows,
# 256 MB each x 64 = 16 GiB), stock HBM2_8Gb_x128 timing in cycles (tRCD 14, tRP 14, tRAS 34, CL 14, tCCD_S/L 1/2, tRFC 260) with tCK = 1.14 ns
# (V100's 877 MHz HBM2 clock; peak 64 x 32 B / 2.28 ns = 898 GB/s).  a16: GDDR6 x16 channel (16-bit, BL16 = 32 B, 16 banks, bank groups
# disabled, 4 KiB page = 7 column bits, 32768 rows, 2 GiB each x 8 = 16 GiB), stock GDDR6_8Gb_x16 timing in cycles, tCK = 0.64 ns
# (12.5 Gb/s pins), tCCD_S = tCCD_L = 2 so one 32 B burst per 1.28 ns = 25 GB/s per channel, 200 GB/s peak.
DS3 = {
    "v100": dict(base="HBM2_8Gb_x128.ini", npc=2, tck=1.14, bus=2.0, tmiss=48.0, peak=64 * 32 / 2.28,
                 sub={"rows": "16384", "columns": "64", "device_width": "64", "bus_width": "64", "BL": "4", "num_dies": "1",
                      "channel_size": "256", "channels": "1", "tCK": "1.14", "address_mapping": "rochrababgco"}),
    "a16": dict(base="GDDR6_8Gb_x16.ini", npc=1, tck=0.64, bus=2.0, tmiss=78.0, peak=8 * 32 / 1.28,
                sub={"rows": "32768", "columns": "128", "device_width": "16", "bus_width": "16", "BL": "16", "channel_size": "2048",
                     "channels": "1", "tCK": "0.64", "tCCD_S": "2", "tCCD_L": "2", "address_mapping": "rochrababgco"}),
}


def ds3_config(path, gpu, pol):
    s = open(f"{D3}/configs/{DS3[gpu]['base']}").read()
    sub = dict(DS3[gpu]["sub"], row_buf_policy=pol, epoch_period="1000000000")
    for k, v in sub.items():
        s, n = re.subn(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", s)
        assert n == 1, k
    open(path, "w").write(s)


def ds3_run(t):
    cfg, out, tr = t
    os.makedirs(out, exist_ok=True)
    r = subprocess.run([DRV, cfg, out, tr], capture_output=True, text=True)
    fb = 0
    mm = re.search(r"UNISSUED (\d+)", r.stdout)
    if mm and int(mm.group(1)) > 0:
        # DRAMsim3 (split read/write queues) can deadlock in write-drain mode; rerun this instance with one unified queue.
        fb = 1
        cfg2 = cfg.replace(".ini", "_uq.ini")
        open(cfg2, "w").write(re.sub(r"(?m)^unified_queue\s*=.*$", "unified_queue = True", open(cfg).read()))
        for f in os.listdir(out): os.remove(f"{out}/{f}")
        r = subprocess.run([DRV, cfg2, out, tr], capture_output=True, text=True)
    m = re.search(r"DRAIN_CYCLES (\d+) REQS (\d+) WRITES (\d+)(?: MISSING (\d+) UNISSUED (\d+))?", r.stdout)
    if not m:
        raise RuntimeError(f"{tr}: {r.stdout[-300:]} {r.stderr[-300:]}")
    j = json.load(open(f"{out}/dramsim3.json"))["0"]
    res = dict(fallback_unified=fb, missing=int(m.group(4) or 0), unissued=int(m.group(5) or 0), cycles=int(m.group(1)), reqs=int(m.group(2)), writes=int(m.group(3)), reads_done=j["num_reads_done"], writes_done=j["num_writes_done"],
               hits=j["num_read_row_hits"] + j["num_write_row_hits"], rd_hits=j["num_read_row_hits"], lat=j["average_read_latency"], acts=j["num_act_cmds"])
    for f in os.listdir(out):
        os.remove(f"{out}/{f}")
    os.rmdir(out)
    return res


def ds3_files(gpu, L, new, chv, st, tmp, tag, nch):
    """split the remapped stream into per-instance binary traces; returns list of paths (empty instances skipped)."""
    b, row, col = L.fields(new)
    npc = DS3[gpu]["npc"]
    if gpu == "v100":
        inst = chv.astype(np.int64) * 2 + (b & 1)
        bank = b >> 1                       # 4 bits: bg = low 2, ba = high 2 (matches DRAMsim3 ro|ba|bg|co)
        addr = ((row.astype(np.uint64) << np.uint64(4 + 5)) | (bank.astype(np.uint64) << np.uint64(5)) | col.astype(np.uint64)) << np.uint64(5)
    else:
        inst = chv.astype(np.int64)
        addr = ((row.astype(np.uint64) << np.uint64(4 + 7)) | (b.astype(np.uint64) << np.uint64(7)) | col.astype(np.uint64)) << np.uint64(5)
    rec = (addr << np.uint64(1)) | st.astype(np.uint64)
    order = np.argsort(inst, kind="stable")
    cnt = np.bincount(inst, minlength=nch * npc)
    ends = np.cumsum(cnt); starts = ends - cnt
    paths = []
    for i in range(nch * npc):
        if cnt[i]:
            p = f"{tmp}/{tag}_i{i}.bin"
            rec[order[starts[i]:ends[i]]].tofile(p)
            paths.append(p)
    return paths, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix"); ap.add_argument("out")
    ap.add_argument("--gpu", required=True, choices=["v100", "a16"])
    ap.add_argument("--sims", default="ram,ds3")
    ap.add_argument("--schemes", default="identity,xor,chan,xor_chan"); ap.add_argument("--ib", type=int, default=2)
    ap.add_argument("--window", type=int, default=80_000_000); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tmp", default="/tmp"); ap.add_argument("--fanin", type=int, default=3)
    a = ap.parse_args()
    sims = a.sims.split(",")
    sd = dict(GR.SYS[a.gpu]); sd["ib"] = a.ib
    R.LAYOUTS[sd["layout"] + "_ib"] = dict(GR.LAY[sd["layout"]], s=6 + a.ib)
    L = R.Layout(sd["layout"] + "_ib")
    arr = gpu_io.load(a.prefix, "dram", np.uint64)
    total = len(arr)
    if a.window: arr = arr[:a.window]
    st = (arr & np.uint64(1)).astype(bool); byte = (arr >> np.uint64(1)) << np.uint64(5)
    phys, npg = GR.pack_pages(byte)
    print(f"{a.gpu} ib{a.ib}: {len(arr)} of {total} requests ({(~st).sum()} rd, {st.sum()} wr), {npg} 2MiB pages", flush=True)
    intra, ch = L.to_intra(phys)
    work = f"{a.tmp}/ds3study_{os.getpid()}"; os.makedirs(work, exist_ok=True)
    nch = sd["nch"]
    out = dict(gpu=a.gpu, ib=a.ib, requests=len(arr), stream_requests=total, reads=int((~st).sum()), writes=int(st.sum()), pages_2MiB=npg, results=[], info={})
    ram_tasks, ds3_tasks = [], []
    for sim in sims:
        if sim == "ram":
            bus, tmiss = C.cost_params(sd)
        else:
            bus, tmiss = DS3[a.gpu]["bus"], DS3[a.gpu]["tmiss"]
        R.cost.__defaults__ = (bus, tmiss)
        MS.cost3.__defaults__ = (bus, tmiss)
        for sc in a.schemes.split(","):
            chv = ch; info = {}
            if sc == "identity":
                new = intra
            elif sc in ("chan", "xor_chan"):
                base = intra
                if sc == "xor_chan":
                    cands = R.sources(L, False); Mx = R.search(L, R.sample(intra), cands, a.fanin); base = R.apply_matrix(L, intra, Mx, cands)
                    info["xor_matrix"] = {j: [cands[c][0] for c in Mx[j]] for j in range(L.bank)}
                Mc, cc, sel, c0, c1 = MS.chan_search(L, base, chv, nch, a.fanin, sc)
                new = base; chv = chv ^ MS.hash_bits(L, base, Mc, cc).astype(chv.dtype)
                info["chan_matrix"] = sel
            else:
                cands = R.sources(L, False); M = R.search(L, R.sample(intra), cands, a.fanin)
                new = R.apply_matrix(L, intra, M, cands)
                info["xor_matrix"] = {j: [cands[c][0] for c in M[j]] for j in range(L.bank)}
            bb, rr, _ = L.fields(new)
            cm = np.bincount(chv.astype(np.int64), minlength=nch)
            info["chan_max_over_mean"] = float(cm.max() / cm.mean()); info["inorder_hit"] = R.inorder_hits(bb, rr) / len(bb)
            out["info"][f"{sim}:{sc}"] = info
            tag = f"{sim}_{sc}"
            if sim == "ram":
                p = f"{work}/{tag}.trc"; GR.write_trace(p, st, L.from_intra(new, chv))
                for pol in ("Open", "ClosedCAP"):
                    ram_tasks.append((sd, p, sc, pol))
            else:
                paths, cnt = ds3_files(a.gpu, L, new, chv, st, work, tag, nch)
                for pol in ("OPEN_PAGE", "CLOSE_PAGE"):
                    cfg = f"{work}/{tag}_{pol}.ini"; ds3_config(cfg, a.gpu, pol)
                    ds3_tasks.append((sc, pol, [(cfg, f"{work}/o_{tag}_{pol}_{os.path.basename(q)}", q) for q in paths], cnt))
    del arr, byte, phys, intra, ch, st
    with multiprocessing.Pool(a.procs) as pool:
        if ram_tasks:
            res = pool.map(GR.replay, ram_tasks, chunksize=1)
            for r in res:
                r["sim"] = "ram"; r["cycles_unit_ns"] = sd["tck_ps"] / 1000.0; out["results"].append(r)
        flat = [(i, t) for i, x in enumerate(ds3_tasks) for t in x[2]]
        if flat:
            rs = pool.map(ds3_run, [t for _, t in flat], chunksize=1)
            for i, (sc, pol, _, cnt) in enumerate(ds3_tasks):
                x = [r for (j, _), r in zip(flat, rs) if j == i]
                reqs = sum(r["reqs"] for r in x); rd = sum(r["reads_done"] for r in x); tck = DS3[a.gpu]["tck"]
                cyc = max(r["cycles"] for r in x)
                out["results"].append(dict(sim="ds3", scheme=sc, policy=pol, reads=rd, writes=sum(r["writes_done"] for r in x), cycles=cyc,
                    time_us=cyc * tck / 1e3, GBps=reqs * 32 / (cyc * tck), hit_rate=sum(r["hits"] for r in x) / max(reqs, 1),
                    avg_read_latency_cycles=sum(r["lat"] * r["reads_done"] for r in x) / max(rd, 1), cycles_unit_ns=tck,
                    avg_read_latency_ns=tck * sum(r["lat"] * r["reads_done"] for r in x) / max(rd, 1),
                    fallback_unified_instances=sum(r["fallback_unified"] for r in x), missing_callbacks=sum(r["missing"] for r in x), unissued=sum(r["unissued"] for r in x), acts=sum(r["acts"] for r in x), inst_max_over_mean=float(cnt.max() / cnt.mean())))
    for r in out["results"]:
        if r["sim"] == "ram":
            r["avg_read_latency_cycles"] = r["avg_read_latency_ns"] / r["cycles_unit_ns"]
    import shutil; shutil.rmtree(work, ignore_errors=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'sim':<5}{'scheme':<9}{'policy':<11}{'hit':>7}{'lat_cyc':>9}{'lat_ns':>8}{'cycles':>11}{'time_us':>10}{'GB/s':>8}")
    for r in out["results"]:
        print(f"{r['sim']:<5}{r['scheme']:<9}{r['policy']:<11}{r['hit_rate']:>7.3f}{r['avg_read_latency_cycles']:>9.1f}{r['avg_read_latency_ns']:>8.1f}"
              f"{r['cycles']:>11}{r['time_us']:>10.1f}{r['GBps']:>8.1f}")


if __name__ == "__main__":
    main()
