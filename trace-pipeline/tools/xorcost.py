#!/usr/bin/env python3
"""Gate-level cost of the searched XOR bank/channel hashes (extras-sim-owner).

Usage: xorcost.py RESULTS_DIR [--md]   (RESULTS_DIR = cluster-data/results/fullscale)
Parses every "bank bit ^= {k: [rowN, ...]}" / "chan bit ^= {...}" line in make.txt of region/memsys/transfer/dramtech/realmem.
For an output bit b_k' = b_k ^ r_i ^ r_j ^ ...: m row terms -> m 2-input XOR gates, depth ceil(log2(m+1)) levels (m+1 inputs).
'shared' = same count after greedy common-pair elimination across the output bits of one matrix.
"""
import ast, collections, math, os, re, sys, statistics as st

LINE = re.compile(r"^\s*\[([^\]]*?)\s+(?:global(?: bank)?|bank\+chan|chan|RoBa xor_global|MOP4CLXOR xor_global|xor_global[^\]]*)\]\s.*?(bank|chan) bit \^= (\{.*\})\s*$")
LINE2 = re.compile(r"\[([^\]]+)\].*?(bank|chan) bit \^= (\{.*\})\s*$")


def cost(mat):
    terms = [frozenset(v) | {f"b{k}"} for k, v in mat.items()]  # base bit participates
    n = sum(len(v) for v in mat.values())
    depth = max([math.ceil(math.log2(len(v) + 1)) if v else 0 for v in mat.values()] + [0])
    # greedy common-pair elimination
    sets = [set(t) for t in terms]; gates = 0; nxt = 0
    while True:
        c = collections.Counter()
        for s in sets:
            l = sorted(s)
            for i in range(len(l)):
                for j in range(i + 1, len(l)):
                    c[(l[i], l[j])] += 1
        if not c or max(c.values()) < 2:
            break
        (a, b), _ = c.most_common(1)[0]; nxt += 1; new = f"n{nxt}"; gates += 1
        for s in sets:
            if a in s and b in s:
                s -= {a, b}; s.add(new)
    gates += sum(max(len(s) - 1, 0) for s in sets)
    return dict(xor2=n, shared=gates, depth=depth, fanin_max=max([len(v) for v in mat.values()] + [0]),
                bits_hashed=sum(1 for v in mat.values() if v), nbits=len(mat))


def main():
    root = sys.argv[1]
    rows = []
    for d in ("region", "memsys", "transfer", "dramtech", "realmem", "mop4", "chanloop"):
        for dp, _, fs in os.walk(os.path.join(root, d)):
            if "make.txt" not in fs:
                continue
            tag = os.path.relpath(dp, root)
            for l in open(os.path.join(dp, "make.txt")):
                m = LINE2.search(l)
                if not m or "region" in m.group(1):
                    continue
                label, kind, mat = m.groups()
                mat = {int(k): v for k, v in ast.literal_eval(mat).items()}
                rows.append(dict(tag=tag, label=label, kind=kind, **cost(mat)))
    by = collections.defaultdict(list)
    for r in rows:
        mem = ("ddr5x8" if "x8" in r["label"] else "ddr5x16" if ("x16" in r["label"] or "ddr5" in r["label"]) else "ddr4" if "ddr4" in r["label"] else "other")
        by[(r["kind"], r["label"].split()[0].split("/")[0] if False else mem)].append(r)
    def agg(rs, key):
        v = [r[key] for r in rs]; return f"{min(v)}/{st.median(v):g}/{max(v)}"
    print(f"{len(rows)} matrices (bank {sum(r['kind']=='bank' for r in rows)}, chan {sum(r['kind']=='chan' for r in rows)})\n")
    print("| hash | memory model | n matrices | output bits | XOR2 gates min/med/max | after pair sharing | depth levels (max) | max row bits per output |")
    print("|---|---|---|---|---|---|---|---|")
    for (kind, mem), rs in sorted(by.items()):
        print(f"| {kind} | {mem} | {len(rs)} | {rs[0]['nbits']} | {agg(rs,'xor2')} | {agg(rs,'shared')} | {agg(rs,'depth')} | {max(r['fanin_max'] for r in rs)} |")
    allb = [r for r in rows if r["kind"] == "bank"]
    print(f"\nall bank matrices: XOR2 {min(r['xor2'] for r in allb)}..{max(r['xor2'] for r in allb)} (median {st.median(r['xor2'] for r in allb):g}), "
          f"depth max {max(r['depth'] for r in allb)}, bits hashed per matrix {min(r['bits_hashed'] for r in allb)}..{max(r['bits_hashed'] for r in allb)}")
    allc = [r for r in rows if r["kind"] == "chan"]
    if allc:
        print(f"all chan matrices: XOR2 {min(r['xor2'] for r in allc)}..{max(r['xor2'] for r in allc)}, depth max {max(r['depth'] for r in allc)}")
    # distinct row bits used (needed for the programmable mask register)
    print("\nexamples (largest bank matrices):")
    for r in sorted(allb, key=lambda r: -r["xor2"])[:3]:
        print(f"  {r['tag']} {r['label']}: xor2 {r['xor2']} shared {r['shared']} depth {r['depth']}")

main()
