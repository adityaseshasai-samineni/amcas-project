#!/usr/bin/env python3
"""Tables for transfer_remap.py / frag_remap.py replays. Usage: transfer_report.py transfer DIR | frag DIR...
Ratios are Ramulator cycles relative to identity (<1 = XOR helps); 'self' = matrix searched on the stream itself."""
import glob, json, os, re, sys

def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "replay_*.json")):
        b = os.path.basename(f)[7:-5]
        for r in json.load(open(f)):
            out[(b, r["policy"])] = r
    return out

def transfer(d):
    R = load(d)
    keys = sorted({re.match(r"(.+?)__(\w+?)_(ddr\d)_(.+)", b).groups() for b, _ in R})
    combos = sorted({(k[0], k[1], k[2]) for k in keys})
    vars_ = ["naive", "self", "x_T7", "x_T30", "x_pool", "x_pool_loo"]
    for pol in ("Open", "ClosedCAP"):
        print(f"\n### {pol}: cycles relative to identity\n")
        for pk in ("thp", "packed"):
            for dk in ("ddr5", "ddr4"):
                rows = [c for c in combos if c[1] == pk and c[2] == dk]
                if not rows: continue
                print(f"\n{pk} {dk}\n\n| test stream | identity Mcyc | hit | " + " | ".join(vars_) + " |\n|---|---|---|" + "---|" * len(vars_))
                for t, _, _ in rows:
                    base = R.get((f"{t}__{pk}_{dk}_identity", pol))
                    if not base: continue
                    cells = []
                    for v in vars_:
                        r = R.get((f"{t}__{pk}_{dk}_{v}", pol))
                        cells.append(f"{r['cycles'] / base['cycles']:.3f}" if r else "-")
                    print(f"| {t} | {base['cycles'] / 1e6:.2f} | {base['hit_rate']:.3f} | " + " | ".join(cells) + " |")

def frag(dirs):
    for d in dirs:
        R = load(d)
        print(f"\n### {os.path.basename(d)} (DDR5 x16)\n")
        keys = sorted({re.match(r"(.+)_ddr5_(\w+)", b).group(1) for b, _ in R})
        def kk(k):
            m = re.match(r"c([\d.]+)_L(\d+)", k)
            return (0, 0, 0) if not m and k == "packed" and False else ((9, 0, 0) if k == "packed" else (float(m.group(1)), int(m.group(2)), 0))
        keys.sort(key=kk)
        for pol in ("Open", "ClosedCAP"):
            print(f"\n{pol}\n\n| placement | identity Mcyc | hit | xor_c0 / id | xor_self / id | hit self |\n|---|---|---|---|---|---|")
            for k in keys:
                g = lambda v: R.get((f"{k}_ddr5_{v}", pol))
                i, x0, xs = g("identity"), g("xor_c0"), g("xor_self")
                if not i: continue
                f = lambda r: f"{r['cycles'] / i['cycles']:.3f}" if r else "-"
                print(f"| {k} | {i['cycles'] / 1e6:.2f} | {i['hit_rate']:.3f} | {f(x0)} | {f(xs)} | {xs['hit_rate'] if xs else 0:.3f} |")

if sys.argv[1] == "transfer": transfer(sys.argv[2])
else: frag(sys.argv[2:])
