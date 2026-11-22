#!/usr/bin/env python3
"""GPU campaign: read latency and simulator agreement for every finished policy grid.

Reads cluster-data/results/gpu/campaign_20261120/runs/*/ctx_*/policy_grid/
  results.json          Ramulator 2 grid (cycles, row hit / conflict rates)
  dramsim3_results.json DRAMsim3 grid (cycles, row hit)
  ds3_<placement>_s<bits>_<mapping>/<POLICY>/dramsim3.json.gz  per-channel stats
  ds3_*/OPEN_PAGE.ini   HBM2 timing (CL, tRCDRD, tRP, tCK)

Latency:
  DRAMsim3  measured: read-weighted mean of per-channel average_read_latency
            (memory cycles, includes queueing in the saturated replay).
  Ramulator modelled: the CPU grid's offline method (tools/grid_gen.py),
            hit*CL + miss*(tRCD+CL) + conflict*(tRP+tRCD+CL), no queueing.
Writes analysis/gpu/latency.md and analysis/gpu/latency.csv. Do not hand-edit.
"""
import csv, glob, gzip, json, os, re, sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "cluster-data/results/gpu/campaign_20261120/runs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis/gpu"
os.makedirs(OUT, exist_ok=True)
BITS = {256: 8, 2048: 11}


def ini(path):
    d = {}
    for line in open(path):
        m = re.match(r"\s*(\w+)\s*=\s*([0-9.]+)", line)
        if m:
            d[m.group(1)] = float(m.group(2))
    return d


def ds3_latency(stats_gz):
    ch = json.load(gzip.open(stats_gz))
    n = sum(c["num_reads_done"] for c in ch.values())
    return sum(c["average_read_latency"] * c["num_reads_done"] for c in ch.values()) / max(n, 1)


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
        for k, i in enumerate(o): r[i] = k
        return r
    ra, rb = rank(a), rank(b); n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1)) if n > 2 else float("nan")


rows, summary = [], []
for grid in sorted(glob.glob(f"{ROOT}/*/ctx_*/policy_grid")):
    case = grid.split("/")[-3]
    if not os.path.exists(f"{grid}/results.json"):
        continue
    ram = json.load(open(f"{grid}/results.json")); ram = ram if isinstance(ram, list) else ram["results"]
    ds3 = json.load(open(f"{grid}/dramsim3_results.json")) if os.path.exists(f"{grid}/dramsim3_results.json") else []
    t = None
    for r in ram:
        k = (r["placement"], r["channel_interleave_bytes"])
        if t is None:
            f = glob.glob(f"{grid}/ds3_*_identity/OPEN_PAGE.ini")
            t = ini(f[0]) if f else dict(CL=12, tRCDRD=12, tRP=12, tCK=1.25)
        h, c = r["row_hit_rate"], r["row_conflict_rate"]; m = max(0.0, 1 - h - c)
        lat = h * t["CL"] + m * (t["tRCDRD"] + t["CL"]) + c * (t["tRP"] + t["tRCDRD"] + t["CL"])
        rows.append(dict(case=case, sim="Ramulator", placement=k[0], interleave=k[1], mapping=r["mapping"],
                         policy=r["policy"], cycles=r["cycles"], row_hit=h, read_latency_cyc=lat, latency_kind="modelled"))
    for r in ds3:
        k = (r["placement"], r["channel_interleave_bytes"])
        pol = "OPEN_PAGE" if r["policy"] == "Open" else "CLOSE_PAGE"
        g = f"{grid}/ds3_{k[0]}_s{BITS[k[1]]}_{r['mapping']}/{pol}/dramsim3.json.gz"
        lat = ds3_latency(g) if os.path.exists(g) else float("nan")
        rows.append(dict(case=case, sim="DRAMsim3", placement=k[0], interleave=k[1], mapping=r["mapping"],
                         policy=r["policy"], cycles=r["cycles"], row_hit=r["row_hit_rate"], read_latency_cyc=lat,
                         latency_kind="measured"))

    for pl in ("packed", "thp"):
        for il in (256, 2048):
            cell = [x for x in rows if x["case"] == case and x["placement"] == pl and x["interleave"] == il]
            R = {(x["mapping"], x["policy"]): x for x in cell if x["sim"] == "Ramulator"}
            D = {(x["mapping"], x["policy"]): x for x in cell if x["sim"] == "DRAMsim3"}
            if ("identity", "Open") not in R:
                continue
            base = R[("identity", "Open")]
            best = min(R.values(), key=lambda x: x["cycles"])
            s = dict(case=case, placement=pl, interleave=il, base_hit=base["row_hit"], base_lat=base["read_latency_cyc"],
                     best=f"{best['mapping']} + {best['policy']}", best_ratio=best["cycles"] / base["cycles"],
                     best_hit=best["row_hit"], best_lat=best["read_latency_cyc"])
            for nm, key in (("xg", ("xor_global", "Open")), ("mopxor", ("mopxor", "Open")), ("closed", ("identity", "ClosedCAP"))):
                if key in R:
                    s[nm + "_ratio"] = R[key]["cycles"] / base["cycles"]; s[nm + "_lat"] = R[key]["read_latency_cyc"]
            if ("identity", "Open") in D:
                db = D[("identity", "Open")]
                s["ds3_base_lat"] = db["read_latency_cyc"]
                for nm, key in (("ds3_mopxor", ("mopxor", "Open")), ("ds3_xg", ("xor_global", "Open")), ("ds3_closed", ("identity", "Closed"))):
                    if key in D:
                        s[nm + "_lat"] = D[key]["read_latency_cyc"]; s[nm + "_ratio"] = D[key]["cycles"] / db["cycles"]
                common = [m for (m, p) in D if p == "Open" and (m, "Open") in R and m not in ("bitrev", "chra")]
                if len(common) > 2:
                    s["rho_open"] = spearman([R[(m, "Open")]["cycles"] for m in common], [D[(m, "Open")]["cycles"] for m in common])
                    s["best_open_ram"] = min(common, key=lambda m: R[(m, "Open")]["cycles"])
                    s["best_open_ds3"] = min(common, key=lambda m: D[(m, "Open")]["cycles"])
            summary.append(s)

with open(f"{OUT}/latency.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

tck = 1.25
f = lambda x, d=3: "—" if x is None or x != x else f"{x:.{d}f}"
L = ["# GPU read latency and simulator agreement (V100 Accel-Sim campaign)", "",
     "Generated by `trace-pipeline/gpu/gpu_latency_report.py` from the finished policy grids. Do not hand-edit.", "",
     "HBM2 model, tCK 1.25 ns, 4M-sector window per case, saturated replay. Cycle ratio = cycles / identity + Open "
     "(same case, placement, interleave; <1 is better). Ramulator latency is modelled from hit/miss/conflict counts "
     "(CPU grid method, no queueing); DRAMsim3 latency is measured (includes queueing). Latencies in memory cycles; ns = cycles x 1.25.", "",
     "## Ramulator: row hit, cycles and modelled read latency", "",
     "| case | place | intlv | hit id+Open | lat id+Open | best (cycles) | ratio | hit | lat | mopxor+Open ratio / lat | xor_global+Open ratio / lat | id+ClosedCAP ratio / lat |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s in summary:
    L.append(f"| {s['case']} | {s['placement']} | {s['interleave']} | {f(s['base_hit'])} | {f(s['base_lat'],1)} | {s['best']} | {f(s['best_ratio'])} | "
             f"{f(s['best_hit'])} | {f(s['best_lat'],1)} | {f(s.get('mopxor_ratio'))} / {f(s.get('mopxor_lat'),1)} | "
             f"{f(s.get('xg_ratio'))} / {f(s.get('xg_lat'),1)} | {f(s.get('closed_ratio'))} / {f(s.get('closed_lat'),1)} |")
L += ["", "## DRAMsim3: measured read latency (memory cycles) and agreement with Ramulator", "",
      "rho = Spearman rank correlation of cycles across mappings under Open page (bitrev, chra excluded).", "",
      "| case | place | intlv | id+Open lat | mopxor+Open ratio / lat | xor_global+Open ratio / lat | id+Closed ratio / lat | rho | best Open mapping: Ramulator / DRAMsim3 |",
      "|---|---|---|---|---|---|---|---|---|"]
for s in summary:
    if "ds3_base_lat" not in s:
        continue
    L.append(f"| {s['case']} | {s['placement']} | {s['interleave']} | {f(s['ds3_base_lat'],1)} | {f(s.get('ds3_mopxor_ratio'))} / {f(s.get('ds3_mopxor_lat'),1)} | "
             f"{f(s.get('ds3_xg_ratio'))} / {f(s.get('ds3_xg_lat'),1)} | {f(s.get('ds3_closed_ratio'))} / {f(s.get('ds3_closed_lat'),1)} | "
             f"{f(s.get('rho_open'),2)} | {s.get('best_open_ram','—')} / {s.get('best_open_ds3','—')} |")
open(f"{OUT}/latency.md", "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}/latency.md ({len(summary)} cells, {len(rows)} replays)")
