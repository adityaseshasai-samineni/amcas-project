#!/usr/bin/env python3
"""DRAMsim3 cross-check of the realmem_traces.py request streams (validation-owner).

Usage: dramsim3_replay.py TRACE_DIR OUT_JSON --nch N [--keys K1,K2] [--policies OPEN_PAGE,CLOSE_PAGE] [--procs P]

TRACE_DIR holds KEY.trc ("LD 0xADDR", channel in the power-of-two field at bits [8, 8+W2), W2 = ceil(log2 N)).
Each channel's requests are split out, the channel field is removed (256 B interleave), and the stream is replayed in
order, saturated (all timestamps 0, one request per cycle when the transaction queue accepts) through one
DRAMsim3 instance per channel; channels are independent, so total cycles = slowest channel, row hits and
read latency are request-weighted. DRAM: DDR4-2933 (21-21-21), 16Gb x8, 2 ranks, 32 GB/channel (DRAMsim3's DDR4_8Gb_x8_2933
config with rows=131072, channel_size=32768, tRFC for 16Gb), address_mapping rochrababgco (MSB..LSB: row, channel(width 0),
rank, bank, bankgroup, column; with the 6 line-offset bits dropped = remap.Layout 'bank' field = {rank, bank, bg}, 7 column bits).
The driver tools/dramsim3_drain.cc (tracing/bin/dramsim3_drain) runs until the last read completes (dramsim3main only runs a fixed -c).
"""
import argparse, json, multiprocessing, os, re, subprocess, sys
import numpy as np

HOME = os.path.expanduser("~/amcas-pro")
D3 = f"{HOME}/tracing/DRAMsim3"
DRV = f"{HOME}/tracing/bin/dramsim3_drain"


def config(path, policy, rate="2933"):
    s = open(f"{D3}/configs/DDR4_8Gb_x8_{rate}.ini").read()
    sub = {"rows": "131072", "channel_size": "32768", "tRFC": "807", "tRFC2": "610", "tRFC4": "381",
           "address_mapping": "rochrababgco", "row_buf_policy": policy, "epoch_period": "1000000000"}
    for k, v in sub.items():
        s, n = re.subn(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", s)
        assert n == 1, k
    open(path, "w").write(s)


def run(cfg):
    d, key, pol, c, nch, tmp = cfg
    w2 = 0 if nch == 1 else max(1, (nch - 1).bit_length())
    out = f"{tmp}/{key}_{pol}_{c}"
    os.makedirs(out, exist_ok=True)
    config(f"{out}/config.ini", pol)
    tr = f"{tmp}/{key}.ch{c}.trc"
    r = subprocess.run([DRV, f"{out}/config.ini", out, tr], capture_output=True, text=True)
    m = re.search(r"DRAIN_CYCLES (\d+) READS (\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"{key} {pol} ch{c}: {r.stdout[-300:]} {r.stderr[-300:]}")
    s = json.load(open(f"{out}/dramsim3.json"))["0"]
    res = dict(key=key, policy=pol, ch=c, cycles=int(m.group(1)), reads=int(m.group(2)), reads_done=s["num_reads_done"],
               row_hits=s["num_read_row_hits"], avg_read_latency=s["average_read_latency"], acts=s["num_act_cmds"])
    for f in os.listdir(out):
        os.remove(f"{out}/{f}")
    os.rmdir(out)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out")
    ap.add_argument("--nch", type=int, required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--policies", default="OPEN_PAGE,CLOSE_PAGE")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tmp", default="/scratch")
    a = ap.parse_args()
    nch = a.nch; w2 = 0 if nch == 1 else max(1, (nch - 1).bit_length())
    keys = a.keys.split(",") if a.keys else sorted(f[:-4] for f in os.listdir(a.dir) if f.endswith(".trc"))
    tmp = f"{a.tmp}/ds3_{os.getpid()}"; os.makedirs(tmp, exist_ok=True)
    for k in keys:  # split per channel, strip the channel field
        p = np.array([int(l.split()[1], 16) for l in open(f"{a.dir}/{k}.trc")], dtype=np.uint64)
        if nch == 1:
            parts = {0: p}
        else:
            ch = ((p >> np.uint64(8)) & np.uint64((1 << w2) - 1)).astype(np.int64)
            intra = ((p >> np.uint64(8 + w2)) << np.uint64(8)) | (p & np.uint64(255))
            parts = {c: intra[ch == c] for c in range(nch)}
        for c, v in parts.items():
            with open(f"{tmp}/{k}.ch{c}.trc", "w") as f:
                f.writelines(f"0x{x:x}\n" for x in v.tolist())
        print(f"split {k}: channel reads {[len(v) for v in parts.values()]}", flush=True)
    cfgs = [(a.dir, k, pol, c, nch, tmp) for k in keys for pol in a.policies.split(",") for c in range(nch)]
    with multiprocessing.Pool(a.procs) as pool:
        rs = pool.map(run, cfgs, chunksize=1)
    agg = []
    for k in keys:
        for pol in a.policies.split(","):
            x = [r for r in rs if r["key"] == k and r["policy"] == pol]
            n = sum(r["reads_done"] for r in x)
            agg.append(dict(key=k, policy=pol, nch=nch, reads=n, cycles=max(r["cycles"] for r in x),
                            hit_rate=sum(r["row_hits"] for r in x) / max(n, 1),
                            avg_read_latency=sum(r["avg_read_latency"] * r["reads_done"] for r in x) / max(n, 1),
                            acts=sum(r["acts"] for r in x), channel_cycles=[r["cycles"] for r in x]))
            g = agg[-1]
            print(f"{k:<18}{pol:<11}reads {g['reads']:>9} hit {g['hit_rate']:.3f} lat {g['avg_read_latency']:7.1f} cyc {g['cycles']/1e6:8.3f}M", flush=True)
    json.dump(agg, open(a.out, "w"), indent=1)
    for f in os.listdir(tmp):
        os.remove(f"{tmp}/{f}")
    os.rmdir(tmp)


if __name__ == "__main__":
    main()
