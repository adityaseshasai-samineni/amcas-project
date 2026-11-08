#!/usr/bin/env python3
"""Turn drcachesim's LLC miss file into a DRAM request stream, comparable to a gem5 run.

Usage: drmiss_export.py MISS_FILE OUT_DIR [--placement NAME] [--stride-only]
                       [--tensors TSV] [--no-trace]

--tensors TSV (from llama-drattr, AMCAS_TENSORS) labels every read by its virtual address:
dense weights / expert weights / KV cache / activations / unattributed, and writes
attribution.json (reads, stride and in-order open-page row hit per class and per placement,
plus per-expert read counts for 3-D *_exps tensors). --no-trace skips dram.ramulator.trc.

MISS_FILE is from `drcachesim -LL_miss_file F.gz -data_prefetcher none`: one line
"pid,pc,address" per demand LLC miss, in simulation order. The first field is the
process id, not the thread (one value for a 4-thread run), so the stream cannot be
split by thread; the "per thread" stride lines are then the merged stream again. With the prefetcher off,
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


def load_tensors(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows.append((p[1], int(p[2], 16), int(p[3]), p[4], p[5], [int(x) for x in p[6:10]]))
    return rows


def tensor_class(name, op):
    if "_exps" in name:
        return "expert weights"
    if name.startswith("cache_k") or name.startswith("cache_v"):
        return "KV cache"
    if op == "NONE" and (name.startswith("blk.") or name.startswith("token_embd") or name.startswith("output")
                         or name.startswith("rope_freqs")):
        return "dense weights"
    return "activations"


CLASSES = ["expert weights", "KV cache", "dense weights", "activations"]


def classify(va, rows):
    """Class id per read (index into CLASSES, len(CLASSES) = unattributed) and the tensor
    index (-1 if none). Priority: experts, KV, dense weights, activations."""
    cls = np.full(len(va), len(CLASSES), dtype=np.int8)
    tix = np.full(len(va), -1, dtype=np.int32)
    for ci in reversed(range(len(CLASSES))):  # lowest priority first, overwritten by higher
        sel = [i for i, r in enumerate(rows) if tensor_class(r[0], r[4]) == CLASSES[ci] and r[2] > 0]
        if not sel:
            continue
        sel.sort(key=lambda i: rows[i][1])
        st = np.array([rows[i][1] for i in sel], dtype=np.int64)
        en = np.array([rows[i][1] + rows[i][2] for i in sel], dtype=np.int64)
        k = np.searchsorted(st, va, side="right") - 1
        ok = (k >= 0) & (va < en[np.maximum(k, 0)])
        cls[ok] = ci
        tix[ok] = np.array(sel)[k[ok]]
    return cls, tix


def class_row_hits(cls, bank, row, nbanks=32):
    """In-order open-page model over the whole stream (shared bank state), hits counted per class."""
    open_row = [-1] * nbanks
    n = len(CLASSES) + 1
    hit, tot = [0] * n, [0] * n
    for c, b, r in zip(cls.tolist(), bank.tolist(), row.tolist()):
        tot[c] += 1
        if open_row[b] == r:
            hit[c] += 1
        open_row[b] = r
    return {(CLASSES + ["unattributed"])[i]: hit[i] / tot[i] for i in range(n) if tot[i]}


def attribution(va, tids, rows, place, strides, out_dir):
    names = CLASSES + ["unattributed"]
    cls, tix = classify(va, rows)
    res = {"reads": int(len(va)), "classes": {}}
    print("\nattribution by class (reads, share, MB; stride +1/within16/beyond, per class stream)")
    for ci, nm in enumerate(names):
        m = cls == ci
        n = int(m.sum())
        if not n:
            continue
        d = dict(reads=n, share=n / len(va), MB=n * LINE / 1e6, stride=strides(va[m]) if n > 1 else {})
        d["tensors_touched"] = int(len(np.unique(tix[m & (tix >= 0)])))
        res["classes"][nm] = d
        st = d["stride"]
        print(f"  {nm:<15}{n:>12,}{d['share']:>8.3f}{d['MB']:>10.1f}  +1 {st.get('+1 line', 0):.2f} w16 {st.get('within 16 lines', 0):.2f} beyond {st.get('beyond', 0):.2f}")
    res["offline_row_hit_by_class"] = {}
    for pname, pa in place.items():
        for mname, fn in (("RoRaBaCoCh", mapping_ro_ba_co), ("XOR-bank", mapping_xor)):
            bank, row = fn(pa)
            h = class_row_hits(cls, bank, row)
            res["offline_row_hit_by_class"][f"{pname} / {mname}"] = h
            print(f"  row hit {pname} / {mname}: " + ", ".join(f"{k} {v:.3f}" for k, v in h.items()))
    # Per-expert traffic: 3-D tensors named *_exps, slice e = bytes [e*S, (e+1)*S), S = nbytes / ne2.
    exp = {}
    per_layer = {}
    for ti, r in enumerate(rows):
        name, data, nb, _t, op, ne = r
        if "_exps" not in name or ne[2] < 2 or nb <= 0:
            continue
        m = tix == ti
        if not m.any():
            continue
        S = nb // ne[2]
        e = (va[m] - data) // S
        cnt = np.bincount(e, minlength=ne[2])[: ne[2]]
        lines_per_slice = S // LINE
        exp.setdefault("per_expert_reads", np.zeros(ne[2], dtype=np.int64))
        exp["per_expert_reads"] += cnt
        per_layer[name] = dict(
            n_expert=int(ne[2]), slice_bytes=int(S),
            experts_touched=int((cnt > 0).sum()),
            experts_mostly_read=int((cnt >= 0.5 * lines_per_slice).sum()),
            reads=int(cnt.sum()),
        )
    if per_layer:
        tot = exp["per_expert_reads"]
        res["experts"] = dict(
            per_expert_reads_all_layers=tot.tolist(),
            per_layer=per_layer,
            mean_experts_touched_per_tensor=float(np.mean([v["experts_touched"] for v in per_layer.values()])),
            mean_experts_mostly_read_per_tensor=float(np.mean([v["experts_mostly_read"] for v in per_layer.values()])),
        )
        print(f"  experts: {len(per_layer)} expert tensors, mean touched/tensor "
              f"{res['experts']['mean_experts_touched_per_tensor']:.1f} of {next(iter(per_layer.values()))['n_expert']}, "
              f"mostly read {res['experts']['mean_experts_mostly_read_per_tensor']:.1f}")
    json.dump(res, open(f"{out_dir}/attribution.json", "w"), indent=1)


def strides(x):
    s = np.diff(x // LINE)
    if len(s) == 0:
        return {}
    return {
        "+1 line": float((s == 1).mean()),
        "-1 line": float((s == -1).mean()),
        "within 16 lines": float(((np.abs(s) <= 16) & (np.abs(s) != 1)).mean()),
        "within 8 KiB row": float(((np.abs(s) > 16) & (np.abs(s) <= 128)).mean()),
        "beyond": float((np.abs(s) > 128).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("miss_file")
    ap.add_argument("out_dir")
    ap.add_argument("--placement", default="VA-order packed")
    ap.add_argument("--tensors", help="tensor TSV from llama-drattr: attribute reads to classes")
    ap.add_argument("--no-trace", action="store_true", help="do not write dram.ramulator.trc")
    ap.add_argument("--stride-only", action="store_true", help="write only stride.json (for one thread's stream)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tids, addrs = read_misses(a.miss_file)
    va = (addrs // LINE * LINE).astype(np.int64)
    print(f"{len(va):,} DRAM reads ({len(va) * LINE / 1e6:.1f} MB), {len(np.unique(tids))} threads")

    # Stride structure, exactly as analyze.py: on the merged stream, and per thread
    # (the merged stream interleaves the cores, which gem5's one core did not).
    stride = strides(va)
    per_thread = {str(t): strides(va[tids == t]) for t in np.unique(tids) if (tids == t).sum() > 1000}
    print("stride, merged:", {k: round(v, 3) for k, v in stride.items()})
    for t, s in per_thread.items():
        print(f"stride, thread {t}:", {k: round(v, 3) for k, v in s.items()})

    if a.stride_only:
        json.dump(dict(reads=int(len(va)), stride=stride), open(f"{a.out_dir}/stride.json", "w"), indent=1)
        return

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
    if a.tensors:
        attribution(va, tids, load_tensors(a.tensors), place, strides, a.out_dir)
    if a.no_trace:
        return
    with open(f"{a.out_dir}/dram.ramulator.trc", "w") as fr:
        for ad in place[a.placement].tolist():
            fr.write(f"LD 0x{ad:x}\n")
    print(f"\nwrote {a.out_dir}/summary.json and dram.ramulator.trc ({len(va):,} requests, {a.placement})")


if __name__ == "__main__":
    main()
