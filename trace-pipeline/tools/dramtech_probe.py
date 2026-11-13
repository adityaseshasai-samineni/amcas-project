#!/usr/bin/env python3
"""Empirically classify each physical address bit of a dramtech system (is it a channel, row, bank-region
or column/offset bit?) with tiny synthetic Ramulator replays, and compare with the layout in remap.LAYOUTS.

Usage: dramtech_probe.py SYSTEM [--n 600]       (PYTHONPATH=ramulator2-amcas/python, LD_LIBRARY_PATH=ramulator2-amcas)
For each bit k: stream A,B,A,B,... with B = A ^ (1<<k) (test 1) and B = A ^ (1<<k) ^ (1<<top) (test 2),
top = highest address bit (always a row bit). Per-channel read counts show channel bits; row conflicts in test 1
show row bits; hits in test 1 but conflicts in test 2 show column/offset bits; hits in both show bank-region bits.
"""
import argparse, multiprocessing, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dramtech_cfg as C
import remap as R
C.register(R)


def run(args):
    name, k, mode, n = args
    import ramulator
    sd = C.SYSTEMS[name]
    L = R.Layout(sd["layout"])
    top = L.off + L.col + L.bank + L.row + L.W - 1
    a, b = 0, (1 << k) | ((1 << top) if mode == 2 else 0)
    with tempfile.NamedTemporaryFile("w", suffix=".trc", delete=False) as f:
        f.write("".join(f"LD 0x{(a, b)[i % 2]:x}\n" for i in range(n)))
        p = f.name
    mem = C.build_mem(sd, "Open")
    sim = ramulator.Simulation(ramulator.frontend.LoadStoreTrace(clock_ratio=2, path=p), mem)
    sim.run(); os.unlink(p)
    cs = sim.stats["memory_system"]["controller"]
    cs = cs if isinstance(cs, list) else [cs]
    return k, mode, [c["num_read_reqs"] for c in cs], sum(c["row_conflicts"] for c in cs), sum(c["row_hits"] for c in cs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("system"); ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    sd = C.SYSTEMS[a.system]; L = R.Layout(sd["layout"])
    top = L.off + L.col + L.bank + L.row + L.W - 1
    jobs = [(a.system, k, m, a.n) for k in range(0, top) for m in (1, 2)]
    with multiprocessing.Pool(8) as pool:
        res = {(k, m): (rc, cf, h) for k, m, rc, cf, h in pool.map(run, jobs)}
    cls = {}
    for k in range(0, top):
        rc1, cf1, h1 = res[(k, 1)]; rc2, cf2, h2 = res[(k, 2)]
        nz = sum(1 for x in rc1 if x)
        if nz > 1: c = "chan"
        elif cf1 >= 5: c = "row"
        elif cf2 >= 5: c = "col/off"
        else: c = "bank"
        cls[k] = c
    runs = []
    for k in range(0, top):
        if runs and runs[-1][0] == cls[k]: runs[-1][2] = k
        else: runs.append([cls[k], k, k])
    print(f"{a.system}: {sd['what']}; address bits 0..{top}")
    print("  measured:", " ".join(f"{c}[{lo}:{hi}]" for c, lo, hi in runs))
    exp = [("col/off", 0, L.off + L.col - 1)]  # channel bits are cut out of this range
    s, W = L.s, L.W
    print(f"  layout : off[{0}:{L.off-1}] col[{L.off}:{L.off+L.col-1}] (intra) bank[{L.bank_lo}:{L.row_lo-1}] row[{L.row_lo}:{L.row_lo+L.row-1}] (intra); "
          f"channel = phys[{s}:{s+W-1}], intra bit k>={s} is phys bit k+{W}")
    # physical expectation
    phys = {}
    for k in range(top + 1):
        if W and s <= k < s + W: phys[k] = "chan"
        else:
            i = k if k < s else k - W
            phys[k] = "col/off" if i < L.bank_lo else "bank" if i < L.row_lo else "row"
    bad = [k for k in range(top) if phys[k] != cls[k]]
    print("  MATCH" if not bad else f"  MISMATCH at bits {bad}: expected {[phys[k] for k in bad]} got {[cls[k] for k in bad]}")


if __name__ == "__main__":
    main()
