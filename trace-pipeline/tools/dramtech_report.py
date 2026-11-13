#!/usr/bin/env python3
"""Tabulate dramtech_study.py results: cycles ratio vs identity (<1 = gain), row hit, read latency, GB/s.
Usage: dramtech_report.py RESULT_DIR [RESULT_DIR ...]   (each holds <system>.json)"""
import glob, json, os, sys

ORDER = ["lpddr5x8", "lpddr5x32", "lpddr6x8", "lpddr6x32", "gddr6x16", "hbm3x16", "hbm4x32"]
for d in sys.argv[1:]:
    print(f"\n### {d}\n")
    for sysn in ORDER:
        p = f"{d}/{sysn}.json"
        if not os.path.exists(p): continue
        j = json.load(open(p)); R = {(r["key"], r["point"], r["policy"]): r for r in j["results"]}
        print(f"**{sysn}**: {j['manifest']['what']}\n")
        pts = sorted({k[1] for k in R}, key=lambda x: ["sat", "fast", "nat4"].index(x) if x in ("sat", "fast", "nat4") else 9)
        schemes = sorted({k[0].split("_", 1)[1] for k in R}, key=lambda s: ["identity", "xor_global", "chan", "xor_global_chan", "region"].index(s))
        print("| placement | point | policy | identity Mt (hit, lat ns, GB/s) | " + " | ".join(f"{s} ratio (hit)" for s in schemes if s != "identity") + " |")
        print("|---|---|---|---|" + "---|" * (len(schemes) - 1))
        for pl in ("packed", "thp", "4k"):
            for pt in pts:
                for pol in ("Open", "ClosedCAP"):
                    b = R.get((f"{pl}_identity", pt, pol))
                    if not b: continue
                    row = [pl, pt, pol, f"{b['cycles']/1e6:.3f} ({b['hit_rate']:.2f}, {b['avg_read_latency_ns']:.0f}, {b['GBps']:.0f})"]
                    for s in schemes:
                        if s == "identity": continue
                        r = R.get((f"{pl}_{s}", pt, pol))
                        row.append(f"{r['cycles']/b['cycles']:.3f} ({r['hit_rate']:.2f}, lat {r['avg_read_latency_ns']/b['avg_read_latency_ns']:.2f}x)" if r else "-")
                    print("| " + " | ".join(row) + " |")
        print()
