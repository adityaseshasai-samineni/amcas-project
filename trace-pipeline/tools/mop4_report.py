#!/usr/bin/env python3
"""Tables for the MOP4CLXOR comparison. Usage: mop4_report.py REPLAY_JSON [REPLAY_JSON...] > out.md
Per (placement, DRAM, point, policy): Mcycles of each scheme (ratio vs ident), row hit, read latency, GB/s, winner
(lowest cycles; 'tie' if the best two are within 0.5%). Schemes: ident, xor_global, mop, mopxor, chra."""
import json, sys
from collections import defaultdict

S = ["ident", "xor_global", "mop", "mopxor", "chra"]
ORD_D = ["x16", "x8", "d4r6", "d4"]; ORD_P = ["sat", "k1", "w1024", "k4"]; ORD_PL = ["packed", "thp", "4k"]


def main():
    for path in sys.argv[1:]:
        res = json.load(open(path))
        g = defaultdict(dict)
        for r in res:
            g[(r["placement"], r["dram"], r["point"], r["policy"])][r["scheme"]] = r
        print(f"\n#### {path}\n")
        print("| pl | DRAM | point | pol | ident Mc | xor_global | mop | mopxor | chra | hit id/xg/mop | lat id/xg/mop | GB/s id/xg/mop | winner |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in sorted(g, key=lambda k: (ORD_PL.index(k[0]), ORD_D.index(k[1]), ORD_P.index(k[2]), k[3])):
            d = g[k]
            if "ident" not in d:
                continue
            base = d["ident"]["cycles"]
            cells = []
            for s in S[1:]:
                cells.append(f"{d[s]['cycles'] / 1e6:.2f} ({d[s]['cycles'] / base:.3f})" if s in d else "-")
            cyc = sorted((d[s]["cycles"], s) for s in S if s in d)
            win = cyc[0][1] + (" (tie " + cyc[1][1] + ")" if len(cyc) > 1 and cyc[1][0] < cyc[0][0] * 1.005 else "")
            f = lambda key, fmt: "/".join(format(d[s][key] / (1e3 if key == "throughput_MBps" else 1), fmt) if s in d else "-" for s in ("ident", "xor_global", "mop"))
            print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {base / 1e6:.2f} | " + " | ".join(cells) +
                  f" | {f('hit_rate', '.2f')} | {f('avg_read_latency', '.0f')} | {f('throughput_MBps', '.0f')} | {win} |")


if __name__ == "__main__":
    main()
