#!/usr/bin/env python3
"""Compare the DRAM bytes of kept streams (PREFIX.dram.gz) with the ncu per-step totals (results/gpu/*_ncu.raw, dram__bytes_read/write.sum).
usage: gpu_ds3_validate.py OUT_JSON STREAMDIR[:STREAMDIR...]    (looks for q7_{v100,a16}_{short,8k,32k}_fa{auto,off}_x13.dram.gz)"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_io
RES = os.path.expanduser("~/amcas-pro/results/gpu")
UN = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9}
def ncu(path):
    tot = {}
    if not os.path.exists(path): return None
    hdr = None
    for r in csv.reader([l for l in open(path) if l.startswith('"')]):
        if r[0] == "ID": hdr = r; continue
        d = dict(zip(hdr, r))
        try: v = float(d["Metric Value"].replace(",", ""))
        except ValueError: continue
        tot[d["Metric Name"]] = tot.get(d["Metric Name"], 0) + v * UN.get(d["Metric Unit"], 1)
    return tot
out = {}
dirs = sys.argv[2].split(":")
for gpu in ("v100", "a16"):
    for cn in ("short", "8k", "32k"):
        for fa in ("auto", "off"):
            tag = f"q7_{gpu}_{cn}_fa{fa}"; pre = None
            for d in dirs:
                if os.path.exists(f"{d}/{tag}_x13.dram.gz"): pre = f"{d}/{tag}_x13"
            if not pre: continue
            a = gpu_io.load(pre, "dram", np.uint64); w = (a & np.uint64(1)).astype(bool)
            n = ncu(f"{RES}/{tag}_ncu.raw"); r = dict(requests=len(a), model_read_GB=float((~w).sum()) * 32e-9, model_write_GB=float(w.sum()) * 32e-9)
            if n:
                r["ncu_read_GB"] = n.get("dram__bytes_read.sum", 0) / 1e9; r["ncu_write_GB"] = n.get("dram__bytes_write.sum", 0) / 1e9
                r["read_model_over_ncu"] = r["model_read_GB"] / r["ncu_read_GB"] if r["ncu_read_GB"] else None
            out[tag] = r
            print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
