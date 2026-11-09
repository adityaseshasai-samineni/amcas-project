"""Shared preparation for heldout_remap.py and moe_remap.py: load an attributed LLC-miss
stream, label classes (W weights / K KV / O other), model placements, expert ids."""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drmiss_export as X  # noqa: E402
import remap as R  # noqa: E402
from analyze import LINE  # noqa: E402

GRAN = 21
NCH = {"ddr4": 1, "ddr5": 16}
TAGS = ["W", "K", "O"]
PLACE = {"packed": "VA-order packed", "thp": "THP 2MiB shuffled", "4k": "4KiB shuffled"}


def prep(miss, tensors):
    _t, addrs = X.read_misses(miss)
    va = (addrs // LINE * LINE).astype(np.int64)
    del addrs
    rows = X.load_tensors(tensors)
    cls, tix = X.classify(va, rows)
    names = X.CLASSES
    grp_of = np.array([0 if n in ("expert weights", "dense weights") else 1 if n == "KV cache" else 2 for n in names] + [2], dtype=np.int8)
    d = dict(n=len(va), va=va, cls=cls, tix=tix, grp=grp_of[cls], rows=rows)
    d["place"], _ = X.placements(va)
    return d


def expert_features(d):
    """Per read: expert id (-1 if not an expert read), layer id, tensor kind (gate/up/down = 0/1/2)."""
    rows = d["rows"]
    nt = len(rows)
    sl = np.zeros(nt, dtype=np.int64)
    ly = np.zeros(nt, dtype=np.int64)
    kd = np.zeros(nt, dtype=np.int64)
    ok = np.zeros(nt, dtype=bool)
    for i, r in enumerate(rows):
        if "_exps" in r[0] and r[5][2] > 1 and r[2] > 0:
            ok[i] = True
            sl[i] = r[2] // r[5][2]
            m = re.search(r"blk\.(\d+)\.", r[0])
            ly[i] = int(m.group(1)) if m else 0
            kd[i] = 0 if "gate" in r[0] else 1 if "up" in r[0] else 2
    data = np.array([r[1] for r in rows], dtype=np.int64)
    t = np.maximum(d["tix"], 0)
    isx = (d["cls"] == 0) & (d["tix"] >= 0) & ok[t]
    e = np.where(isx, (d["va"] - data[t]) // np.maximum(sl[t], 1), -1)
    return e, np.where(isx, ly[t], 0), np.where(isx, kd[t], 0)


def granule_tags(d, pa):
    gran = (pa >> np.uint64(GRAN)).astype(np.int64)
    ug, inv = np.unique(gran, return_inverse=True)
    cnt = np.zeros((len(ug), 3), dtype=np.int64)
    np.add.at(cnt, (inv, d["grp"].astype(np.int64)), 1)
    return cnt.argmax(1).astype(np.int8)[inv]


def window(n, w, frac):
    w = min(w, n)
    s0 = min(int(n * frac), n - w)
    return s0, s0 + w


def write_trace(path, phys):
    with open(path, "w") as f:
        f.writelines(f"LD 0x{x:x}\n" for x in phys.tolist())
