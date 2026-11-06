#!/usr/bin/env python3
"""Turn drcachesim's LLC miss file into a DRAM request stream, comparable to a gem5 run.

Usage: drmiss_export.py MISS_FILE OUT_DIR [--placement NAME]

MISS_FILE is from `drcachesim -LL_miss_file F.gz -data_prefetcher none`: one line
"tid,pc,address" per demand LLC miss, in simulation order. With the prefetcher off,
every line that reaches DRAM is a demand miss, so the file is the whole read stream.
drcachesim does not model dirty lines, so there are no writebacks; store misses
appear as reads. For gem5's Qwen 0.5B, writes were about 1% of reads per token.

Addresses are virtual. Physical placement is modelled, as in analyze.py:
  VA-order packed    pages get frames in virtual-address order from frame 0. Close
                     to gem5 SE mode, where the model's pages were first touched,
                     and given frames, in file order while loading.
  4KiB shuffled      every 4 KiB page gets a random frame in 8 GiB (an aged allocator).
  THP 2MiB shuffled  every 2 MiB region gets a random 2 MiB frame (transparent huge pages).

Writes OUT_DIR/summary.json (stride structure and offline open-page row-hit rate for
each placement and mapping, with analyze.py's definitions) and, for the chosen
placement, OUT_DIR/dram.ramulator.trc (LD 0xADDR, line-aligned).
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import LINE, PAGE, mapping_ro_ba_co, mapping_ro_co_ba, mapping_xor, open_page_stats  # noqa: E402

CAPACITY = 8 << 30


def read_misses(path, chunk=4_000_000):
    """Read in chunks into numpy arrays: a 7B token is about 75M lines, too many for
    Python lists."""
    op = gzip.open if path.endswith(".gz") else open
    tids, addrs, t_buf, a_buf = [], [], [], []

    def flush():
        tids.append(np.array(t_buf, dtype=np.int64))
        addrs.append(np.array(a_buf, dtype=np.uint64))
        t_buf.clear()
        a_buf.clear()

    with op(path, "rt") as f:
        for line in f:
            t, _pc, a = line.rstrip().split(",")
            t_buf.append(int(t))
            a_buf.append(int(a, 16))
            if len(t_buf) >= chunk:
                flush()
    flush()
    return np.concatenate(tids), np.concatenate(addrs)


def placements(va):
    rng = np.random.default_rng(1)
    pages, inv = np.unique(va // PAGE, return_inverse=True)  # sorted: VA order
    if len(pages) * PAGE > CAPACITY:
        sys.exit(f"footprint {len(pages) * PAGE >> 20} MiB exceeds the modelled {CAPACITY >> 30} GiB")
    off = va % PAGE
    packed = np.arange(len(pages), dtype=np.int64)[inv] * PAGE + off
    frames = rng.choice(CAPACITY // PAGE, size=len(pages), replace=False).astype(np.int64)
    shuffled = frames[inv] * PAGE + off
    huge = 2 << 20
    hp, hinv = np.unique(va // huge, return_inverse=True)
    hframes = rng.choice(CAPACITY // huge, size=len(hp), replace=False).astype(np.int64)
    thp = hframes[hinv] * huge + va % huge
    return {"VA-order packed": packed, "4KiB shuffled": shuffled, "THP 2MiB shuffled": thp}, len(pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss_file")
    ap.add_argument("out_dir")
    ap.add_argument("--placement", default="VA-order packed")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tids, addrs = read_misses(a.miss_file)
    va = (addrs // LINE * LINE).astype(np.int64)
    print(f"{len(va):,} DRAM reads ({len(va) * LINE / 1e6:.1f} MB), {len(np.unique(tids))} threads")

    # Stride structure, exactly as analyze.py: on the merged stream, and per thread
    # (the merged stream interleaves the cores, which gem5's one core did not).
    def strides(x):
        s = np.diff(x // LINE)
        return {
            "+1 line": float((s == 1).mean()),
            "-1 line": float((s == -1).mean()),
            "within 16 lines": float(((np.abs(s) <= 16) & (np.abs(s) != 1)).mean()),
            "within 8 KiB row": float(((np.abs(s) > 16) & (np.abs(s) <= 128)).mean()),
            "beyond": float((np.abs(s) > 128).mean()),
        }

    stride = strides(va)
    per_thread = {str(t): strides(va[tids == t]) for t in np.unique(tids) if (tids == t).sum() > 1000}
    print("stride, merged:", {k: round(v, 3) for k, v in stride.items()})
    for t, s in per_thread.items():
        print(f"stride, thread {t}:", {k: round(v, 3) for k, v in s.items()})

    place, n_pages = placements(va)
    mappings = {"RoRaBaCoCh": mapping_ro_ba_co, "RoCoRaBaCh": mapping_ro_co_ba, "XOR-bank": mapping_xor}
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

    json.dump(
        dict(
            reads=int(len(va)),
            footprint_pages=int(n_pages),
            threads={str(t): int((tids == t).sum()) for t in np.unique(tids)},
            stride=stride,
            stride_per_thread=per_thread,
            offline_row_hit=offline,
            exported_placement=a.placement,
        ),
        open(f"{a.out_dir}/summary.json", "w"),
        indent=1,
    )
    with open(f"{a.out_dir}/dram.ramulator.trc", "w") as fr:
        for ad in place[a.placement].tolist():
            fr.write(f"LD 0x{ad:x}\n")
    print(f"\nwrote {a.out_dir}/summary.json and dram.ramulator.trc ({len(va):,} requests, {a.placement})")


if __name__ == "__main__":
    main()
