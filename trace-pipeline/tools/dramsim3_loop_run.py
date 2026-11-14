#!/usr/bin/env python3
"""Run tools/dramsim3_loop (closed-loop DRAMsim3, extras-sim-owner) over realmem_traces.py streams.

Usage: dramsim3_loop_run.py TRACE_DIR OUT_JSON --nch N [--keys K,..] [--points sat,fast:52:256,...] [--policies OPEN_PAGE,CLOSE_PAGE]
Point 'sat' = per-channel in-order saturated (== dramsim3_drain, validates the driver); 'name:BUBBLES:WINDOW[:MSHR]' = 4 cores
(KEY.t0..t3.addr), ipc 4, MSHR 16, core clock = 2 x memory clock. DRAM config as in dramsim3_replay.config (DDR4-2933, 16Gb x8, 2 ranks).
"""
import argparse, json, multiprocessing, os, re, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dramsim3_replay as DR

DRV = f"{DR.HOME}/tracing/bin/dramsim3_loop"
TCK_NS = 2000.0 / 2933  # memory clock period (ns)


def run(cfg):
    d, key, point, pol, nch, tmp = cfg
    out = f"{tmp}/{key}_{point.split(':')[0]}_{pol}"
    os.makedirs(out, exist_ok=True)
    DR.config(f"{out}/config.ini", pol)
    if point == "sat":
        args = [DRV, f"{out}/config.ini", out, str(nch), "sat", "0", "0", "0", "0", "0", f"{d}/{key}.trc"]
    else:
        parts = point.split(":"); bub, win = parts[1], parts[2]; mshr = parts[3] if len(parts) > 3 else "16"
        args = [DRV, f"{out}/config.ini", out, str(nch), "closed", bub, win, mshr, "4", "2"] + [f"{d}/{key}.t{t}.addr" for t in range(4)]
    r = subprocess.run(args, capture_output=True, text=True)
    m = re.search(r"RESULT mode=(\w+) cycles=(\d+) (?:reads_issued|reads)=(\d+)(?: minlen=(\d+))?", r.stdout)
    if not m:
        raise RuntimeError(f"{key} {point} {pol}: {r.stdout[-300:]} {r.stderr[-300:]}")
    cyc, issued = int(m.group(2)), int(m.group(3))
    rd = hits = lat = acts = 0
    for c in range(nch):
        s = json.load(open(f"{out}/ch{c}/dramsim3.json"))["0"]
        n = s["num_reads_done"]; rd += n; hits += s["num_read_row_hits"]; lat += s["average_read_latency"] * n; acts += s["num_act_cmds"]
    minlen = int(m.group(4)) if m.group(4) else issued
    nread = 4 * minlen if point != "sat" else issued
    shutil.rmtree(out, ignore_errors=True)
    return dict(key=key, point=point, policy=pol, nch=nch, cycles=cyc, reads_done=rd, reads_counted=nread, hit_rate=hits / max(rd, 1),
                avg_read_latency=lat / max(rd, 1), acts=acts, GBps=nread * 64 / (cyc * TCK_NS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out")
    ap.add_argument("--nch", type=int, required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--points", default="sat")
    ap.add_argument("--policies", default="OPEN_PAGE")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tmp", default="/scratch")
    a = ap.parse_args()
    keys = a.keys.split(",") if a.keys else sorted(f[:-4] for f in os.listdir(a.dir) if f.endswith(".trc"))
    tmp = f"{a.tmp}/ds3loop_{os.getpid()}"; os.makedirs(tmp, exist_ok=True)
    # long jobs first
    cfgs = [(a.dir, k, pt, p, a.nch, tmp) for pt in a.points.split(",")[::-1] for k in keys for p in a.policies.split(",")]
    res = []
    with multiprocessing.Pool(a.procs) as pool:
        for r in pool.imap_unordered(run, cfgs):
            res.append(r)
            print(f"{r['key']:<18}{r['point']:<14}{r['policy']:<11}hit {r['hit_rate']:.3f} lat {r['avg_read_latency']:7.1f} cyc {r['cycles']/1e6:8.3f}M {r['GBps']:7.1f}GB/s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
