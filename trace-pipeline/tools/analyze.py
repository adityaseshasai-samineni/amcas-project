#!/usr/bin/env python3
"""Validate and characterise a captured DRAM trace.

Usage: analyze.py RUN_DIR [--export-limit N]

Reads dram.bin (from pkt2bin), phases.tsv, amcas_tensors.tsv, stats.txt.
Writes summary.json, and the portable traces:
  dram.dramsim3.trc   0xADDR READ|WRITE CYCLE   (DRAMSim3 trace format, 1 cycle = 1 tCK)
  dram.ramulator.trc  LD|ST 0xADDR              (Ramulator2 LoadStoreTrace frontend)
"""

import argparse
import json
import os
import re
import sys

import numpy as np

REC = np.dtype(
    [
        ("tick", "<u8"),
        ("addr", "<u8"),
        ("vaddr", "<u8"),
        ("pc", "<u8"),
        ("size", "<u4"),
        ("cmd", "<u2"),
        ("req", "<u2"),
    ]
)
# gem5 src/mem/packet.hh MemCmd::Command values for the writes a controller sees.
WRITE_CMDS = {4, 7, 8, 9}  # WriteReq, WritebackDirty, WritebackClean, WriteClean
PAGE = 4096
LINE = 64


def read_phases(path):
    """Completed phases simulated in detail. Atomic fast-forward phases have no
    meaningful timing. The detailed CPU's counter is 0 at the switch, and the
    atomic CPU's never is at a phase begin, because the program has loaded by then."""
    begin, out = {}, []
    detailed = False
    with open(path) as f:
        next(f)
        for line in f:
            ev, wid, tick, insts = line.split("\t")
            wid, tick, insts = int(wid), int(tick), int(insts)
            detailed = detailed or insts == 0
            if not detailed:
                continue
            if ev == "begin":
                begin[wid] = (tick, insts)
            elif wid in begin:
                out.append(
                    dict(
                        id=wid,
                        begin=begin[wid][0],
                        end=tick,
                        insts=insts - begin[wid][1],
                    )
                )
    return out


def read_stats(path):
    """One dict per dump, keeping only the DRAM and CPU counters we check."""
    want = re.compile(
        r"^(simInsts|simTicks|system\.mem_ctrl\.(readReqs|writeReqs|dram\.(readRowHits|writeRowHits|readBursts|writeBursts|avgRdBWSys|rowHitRate|bytesRead\S*))|system\.mem_ctrl\.avgMemAccLat|system\.mem_ctrl\.dram\.avgMemAccLat|system\.l3cache\.overallMisses::total|system\.l3cache\.overallAccesses::total)\s+(\S+)"
    )
    dumps, cur = [], {}
    with open(path) as f:
        for line in f:
            if line.startswith("---------- Begin"):
                cur = {}
            elif line.startswith("---------- End"):
                dumps.append(cur)
            else:
                m = want.match(line)
                if m:
                    cur[m.group(1)] = float(m.group(m.lastindex))
    return dumps


def bits(a, lo, n):
    return (a >> lo) & ((1 << n) - 1)


def mapping_ro_ba_co(a, row_kb=8, banks=16, ranks=2):
    """gem5 RoRaBaCoCh on a 2-rank DDR4: row | rank | bank | column | line."""
    col = int(np.log2(row_kb * 1024))
    bb = int(np.log2(banks))
    rb = int(np.log2(ranks))
    bank = a >> col & ((1 << (bb + rb)) - 1)
    row = a >> (col + bb + rb)
    return bank, row


def mapping_ro_co_ba(a, row_kb=8, banks=16, ranks=2):
    """Cache-line bank interleave: row | column | rank | bank | line. Spreads
    sequential scans across every bank, at the cost of revisiting each row."""
    bb = int(np.log2(banks))
    rb = int(np.log2(ranks))
    bank = a >> 6 & ((1 << (bb + rb)) - 1)
    col = int(np.log2(row_kb * 1024))
    row = a >> (col + bb + rb)
    return bank, row


def mapping_xor(a, row_kb=8, banks=16, ranks=2):
    """RoRaBaCoCh with the bank bits XORed with low row bits, in the style of
    Zhang et al. (MICRO 2000): removes row conflicts between aligned tensors."""
    bank, row = mapping_ro_ba_co(a, row_kb, banks, ranks)
    n = banks * ranks
    return bank ^ (row & (n - 1)), row


def open_page_stats(bank, row, nbanks=32):
    """Row hit rate seen by an in-order open-page model. This ignores scheduler
    reordering, so it compares mappings against each other. It is not a
    replacement for Ramulator or DRAMSim, which model queues and timing."""
    open_row = np.full(nbanks, -1, dtype=np.int64)
    hits = misses = conflicts = 0
    for b, r in zip(bank.tolist(), row.tolist()):
        cur = open_row[b]
        if cur == r:
            hits += 1
        elif cur < 0:
            misses += 1
        else:
            conflicts += 1
        open_row[b] = r
    n = max(hits + misses + conflicts, 1)
    return dict(hit=hits / n, empty=misses / n, conflict=conflicts / n)


def va_map(rec):
    """Infer the VA->PA page table from requests that carry a virtual address.
    SE mode never remaps a page, so the mapping is a function."""
    m = rec["vaddr"] != 0
    vp = rec["vaddr"][m] // PAGE
    pp = rec["addr"][m] // PAGE
    uniq, idx = np.unique(pp, return_index=True)
    return uniq, vp[idx]


def label_tensors(path, phase_ids):
    """Long-lived tensor address ranges: weights and KV cache. Everything else in
    the graph is an activation in the graph allocator's arena."""
    ranges = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8 or p[5] in ("(nil)", "0x0"):
                continue
            name, data, nbytes = p[4], int(p[5], 16), int(p[6])
            if nbytes == 0:
                continue
            base = name.split(" (")[0]
            if base.endswith(".weight"):
                kind = "weights.experts" if "_exps" in base else "weights"
            elif base.startswith("cache_k") or base.startswith("cache_v"):
                kind = "kv_cache"
            else:
                kind = "activation"
            key = (kind, data)
            ranges[key] = max(ranges.get(key, 0), nbytes)
    out = {}
    for (kind, data), nbytes in ranges.items():
        out.setdefault(kind, []).append((data, data + nbytes))
    return out


def merge(iv):
    iv = sorted(iv)
    res = []
    for a, b in iv:
        if res and a <= res[-1][1]:
            res[-1][1] = max(res[-1][1], b)
        else:
            res.append([a, b])
    return np.array(res, dtype=np.uint64)


def in_ranges(va, r):
    if len(r) == 0:
        return np.zeros(len(va), dtype=bool)
    i = np.searchsorted(r[:, 0], va, side="right") - 1
    ok = i >= 0
    ok[ok] = va[ok] < r[i[ok], 1]
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    d = a.run

    rec = np.memmap(f"{d}/dram.bin", dtype=REC, mode="r")
    phases = read_phases(f"{d}/phases.tsv")
    dumps = read_stats(f"{d}/stats.txt")
    print(f"records {len(rec):,}  phases {len(phases)}  stat dumps {len(dumps)}")
    if not phases:
        sys.exit("no completed detailed phases")

    t0, t1 = phases[0]["begin"], phases[-1]["end"]
    roi = rec[(rec["tick"] >= t0) & (rec["tick"] < t1)]
    is_wr = np.isin(roi["cmd"], list(WRITE_CMDS))
    print("cmd histogram:", dict(zip(*[x.tolist() for x in np.unique(roi["cmd"], return_counts=True)])))

    # Integrity: every request in the trace must be one the controller counted.
    rows = []
    for ph in phases:
        m = (rec["tick"] >= ph["begin"]) & (rec["tick"] < ph["end"])
        n_r = int((~np.isin(rec["cmd"][m], list(WRITE_CMDS))).sum())
        n_w = int(m.sum()) - n_r
        rows.append(dict(ph, trace_reads=n_r, trace_writes=n_w))

    # Dump 0 is taken at the switch to the detailed CPU, then one per detailed phase.
    # The final dump at exit may be missing, so index from the front.
    per_phase = []
    for r, st in zip(rows, dumps[1 : len(phases) + 1]):
        hits = st.get("system.mem_ctrl.dram.readRowHits", 0) + st.get("system.mem_ctrl.dram.writeRowHits", 0)
        bursts = st.get("system.mem_ctrl.dram.readBursts", 0) + st.get("system.mem_ctrl.dram.writeBursts", 0)
        r["stat_readReqs"] = st.get("system.mem_ctrl.readReqs", 0)
        r["stat_writeReqs"] = st.get("system.mem_ctrl.writeReqs", 0)
        r["row_hit_rate"] = hits / bursts if bursts else None
        r["mpki"] = (r["trace_reads"] + r["trace_writes"]) * 1000.0 / max(r["insts"], 1)
        r["read_lat_ns"] = st.get("system.mem_ctrl.dram.avgMemAccLat")
        per_phase.append(r)

    print(f"\n{'ph':>3} {'Minsts':>9} {'reads':>10} {'stat':>10} {'writes':>9} {'stat':>9} {'MPKI':>6} {'rowhit':>7}")
    for r in per_phase:
        rh = "-" if r["row_hit_rate"] is None else f"{r['row_hit_rate']:.3f}"
        print(
            f"{r['id']:>3} {r['insts'] / 1e6:>9.1f} {r['trace_reads']:>10,} {int(r['stat_readReqs']):>10,} "
            f"{r['trace_writes']:>9,} {int(r['stat_writeReqs']):>9,} {r['mpki']:>6.2f} {rh:>7}"
        )

    # Attribute accesses to tensors. Writebacks have no VA, so translate their PA
    # back through the page table inferred from the demand misses.
    pp, vp = va_map(roi)
    va = roi["vaddr"].copy()
    need = va == 0
    pos = np.searchsorted(pp, roi["addr"][need] // PAGE)
    pos[pos >= len(pp)] = 0
    hit = pp[pos] == roi["addr"][need] // PAGE
    tmp = np.zeros(int(need.sum()), dtype=np.uint64)
    tmp[hit] = vp[pos[hit]] * PAGE + roi["addr"][need][hit] % PAGE
    va[need] = tmp

    attribution = {}
    if os.path.exists(f"{d}/amcas_tensors.tsv"):
        ranges = label_tensors(f"{d}/amcas_tensors.tsv", [p["id"] for p in phases])
        claimed = np.zeros(len(roi), dtype=bool)
        print("\nregion            reads%   writes%  footprint_MiB")
        for kind in ("kv_cache", "weights.experts", "weights", "activation"):
            if kind not in ranges:
                continue
            m = in_ranges(va, merge(ranges[kind])) & ~claimed
            claimed |= m
            rd, wr = int((m & ~is_wr).sum()), int((m & is_wr).sum())
            fp = len(np.unique(roi["addr"][m] // LINE)) * LINE / 2**20
            attribution[kind] = dict(reads=rd, writes=wr, footprint_mib=fp)
            print(
                f"{kind:<16} {100 * rd / max((~is_wr).sum(), 1):>6.1f} {100 * wr / max(is_wr.sum(), 1):>8.1f} {fp:>12.1f}"
            )
        m = ~claimed
        attribution["unattributed"] = dict(reads=int((m & ~is_wr).sum()), writes=int((m & is_wr).sum()))
        print(
            f"{'unattributed':<16} {100 * (m & ~is_wr).sum() / max((~is_wr).sum(), 1):>6.1f} "
            f"{100 * (m & is_wr).sum() / max(is_wr.sum(), 1):>8.1f}"
        )

    # Line-address stride structure of the stream.
    la = (roi["addr"] // LINE).astype(np.int64)
    stride = np.diff(la)
    stride_hist = {}
    if len(stride):
        for label, cond in [
            ("+1 line", stride == 1),
            ("-1 line", stride == -1),
            ("within 16 lines", (np.abs(stride) <= 16) & (np.abs(stride) != 1)),
            ("within 8 KiB row", (np.abs(stride) > 16) & (np.abs(stride) <= 128)),
            ("beyond", np.abs(stride) > 128),
        ]:
            stride_hist[label] = float(cond.mean())
        print("\nstride distribution:", {k: round(v, 3) for k, v in stride_hist.items()})

    # Offline comparison of mappings, including the page placement sensitivity.
    addr = roi["addr"].astype(np.int64)
    mappings = {"RoRaBaCoCh": mapping_ro_ba_co, "RoCoRaBaCh": mapping_ro_co_ba, "XOR-bank": mapping_xor}
    # Scatter across the whole address range, as an aged allocator would. Packing
    # the frames into a small range instead would invent row locality.
    rng = np.random.default_rng(1)
    capacity = 8 << 30
    pages, inv = np.unique(addr // PAGE, return_inverse=True)
    frames = rng.choice(capacity // PAGE, size=len(pages), replace=False).astype(np.int64)
    scattered = frames[inv] * PAGE + addr % PAGE
    huge_size = 2 << 20
    huge_pages, huge_inv = np.unique(addr // huge_size, return_inverse=True)
    huge_frames = rng.choice(capacity // huge_size, size=len(huge_pages), replace=False).astype(np.int64)
    huge = huge_frames[huge_inv] * huge_size + addr % huge_size
    place = {"SE first-touch": addr, "THP 2MiB shuffled": huge, "4KiB shuffled": scattered}
    offline = {}
    print("\noffline in-order open-page row hit rate (DDR4 8 KiB row, 2 ranks x 16 banks)")
    print(f"{'placement':<20}" + "".join(f"{k:>13}" for k in mappings))
    for pname, pa in place.items():
        offline[pname] = {}
        line = f"{pname:<20}"
        for mname, fn in mappings.items():
            bank, row = fn(pa)
            s = open_page_stats(bank, row)
            offline[pname][mname] = s
            line += f"{s['hit']:>13.3f}"
        print(line)

    summary = dict(
        records=len(rec),
        roi_records=len(roi),
        roi_reads=int((~is_wr).sum()),
        roi_writes=int(is_wr.sum()),
        phases=per_phase,
        attribution=attribution,
        stride=stride_hist,
        offline_row_hit=offline,
    )
    json.dump(summary, open(f"{d}/summary.json", "w"), indent=1)

    if not a.no_export:
        # 2400 MT/s DDR4 runs tCK = 833 ps. Ticks are picoseconds.
        cycle = ((roi["tick"] - t0) // 833).astype(np.int64)
        with open(f"{d}/dram.dramsim3.trc", "w") as f3, open(f"{d}/dram.ramulator.trc", "w") as fr:
            for ad, w, c in zip(roi["addr"].tolist(), is_wr.tolist(), cycle.tolist()):
                f3.write(f"0x{ad:x} {'WRITE' if w else 'READ'} {c}\n")
                fr.write(f"{'ST' if w else 'LD'} 0x{ad:x}\n")
        print(f"\nwrote {d}/dram.dramsim3.trc and dram.ramulator.trc ({len(roi):,} requests)")


if __name__ == "__main__":
    main()
