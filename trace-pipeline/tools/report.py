#!/usr/bin/env python3
"""Build the cross-run analysis tables and figures from the per-run result files.

Usage: report.py RESULTS_DIR OUT_DIR

RESULTS_DIR is a copy of the cluster's results/ (traces/, native/, sessions/).
Writes OUT_DIR/tables.md and OUT_DIR/fig_*.png. Reads only files written by
analyze.py, 06_dram_sims.sh and the native runs, so it can be re-run at any time.
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Every run, in the order tables list them. `native` is the native output to check
# generated text against, and `prompt` the prompt whose end marks where generation
# starts in both outputs.
P0 = "Once upon a time there was a little girl named Lily. She loved to play outside in the sunshine with her friends."
RUNS = [
    dict(run="s15_l3-22m_stride", model="stories15M", ctx="short", llc="22 MiB", pf="stride", native="native/stories15M.n9.out"),
    dict(run="s15_l3-2m_stride", model="stories15M", ctx="short", llc="2 MiB", pf="stride", native="native/stories15M.n9.out"),
    dict(run="s15_l3-2m_nopf", model="stories15M", ctx="short", llc="2 MiB", pf="none", native="native/stories15M.n9.out"),
    dict(run="s15_c1024_l3-2m", model="stories15M", ctx="1024", llc="2 MiB", pf="stride", native="sessions/stories15M.1024.n5.out", prompt="sessions/stories15M.1024.prompt"),
    dict(run="s15_c4096_l3-2m", model="stories15M", ctx="4096", llc="2 MiB", pf="stride", native="sessions/stories15M.4096.n5.out", prompt="sessions/stories15M.4096.prompt"),
    dict(run="s15_c4096_l3-22m", model="stories15M", ctx="4096", llc="22 MiB", pf="stride", native="sessions/stories15M.4096.n5.out", prompt="sessions/stories15M.4096.prompt"),
    dict(run="qwen_l3-22m_stride", model="Qwen2.5-0.5B", ctx="short", llc="22 MiB", pf="stride", native="native/qwen25-05b-q4_0.n9.out"),
    dict(run="qwen_c1024_l3-22m", model="Qwen2.5-0.5B", ctx="1024", llc="22 MiB", pf="stride", native="sessions/qwen25-05b-q4_0.1024.n5.out", prompt="sessions/qwen25-05b-q4_0.1024.prompt"),
    dict(run="moe_c1024_l3-22m", model="tinymoe4", ctx="1024", llc="22 MiB", pf="stride", native="sessions/tinymoe4.1024.n5.out", prompt="sessions/tinymoe4.1024.prompt"),
    dict(run="moe_l3-22m_stride", model="tinymoe4", ctx="short", llc="22 MiB", pf="stride", native="native/tinymoe4.n9.out"),
]
# Traces replayed in both DRAM simulators, short to long context.
REPLAYED = ["s15_l3-2m_stride", "s15_c1024_l3-2m", "s15_c4096_l3-2m", "s15_c4096_l3-22m"]
LABEL = {"s15_l3-2m_stride": "short, 2 MiB", "s15_c1024_l3-2m": "1024, 2 MiB", "s15_c4096_l3-2m": "4096, 2 MiB", "s15_c4096_l3-22m": "4096, 22 MiB"}
REGIONS = [("weights", "Dense weights"), ("weights.experts", "Expert weights"), ("kv_cache", "KV cache"), ("activation", "Activations"), ("unattributed", "Unattributed")]

# Reference categorical palette (dataviz skill, slots 1-5 in fixed order), light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def generated(path, prompt):
    """Text after the prompt, without gem5's own messages."""
    s = open(path, errors="replace").read()
    s = re.sub(r"\n?Exiting @ tick.*", "", s).replace("switching cpus\n", "")
    i = s.rfind(prompt[-60:])
    return None if i < 0 else s[i + 60 :].rstrip()


def validate(d, r):
    s = json.load(open(f"{d}/traces/{r['run']}/summary.json"))
    counts_ok = all(p["trace_reads"] == p["stat_readReqs"] and p["trace_writes"] == p["stat_writeReqs"] for p in s["phases"])
    prompt = open(f"{d}/{r['prompt']}").read() if "prompt" in r else P0
    g = generated(f"{d}/traces/{r['run']}/simout.txt", prompt)
    n = generated(f"{d}/{r['native']}", prompt)
    text_ok = bool(g) and n is not None and n.startswith(g)
    return s, counts_ok, text_ok, g


def dramsim3(d, run, trace_reads):
    out = {}
    for cfg in sorted(os.listdir(f"{d}/traces/{run}/dramsim3")):
        f = f"{d}/traces/{run}/dramsim3/{cfg}/dramsim3.json"
        if not os.path.exists(f):
            continue
        s = json.load(open(f))["0"]
        hits = s["num_read_row_hits"] + s["num_write_row_hits"]
        n = s["num_reads_done"] + s["num_writes_done"]
        out[cfg] = dict(
            complete=s["num_reads_done"] >= trace_reads,
            hit=hits / max(n, 1),
            rd_lat=s["average_read_latency"],
            act=s["num_act_cmds"],
            act_energy=s["act_energy"],
            total_energy=s["total_energy"],
            requests=n,
            bw=s["average_bandwidth"],
        )
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x" if ax.name == "rectilinear" else "both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_composition(rows, path):
    """100% stacked bars: where each run's DRAM reads go."""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    names = [f"{r['model']}, {r['ctx']}, {r['llc']}" + (", no pf" if r["pf"] == "none" else "") for r in rows]
    for y, r in enumerate(rows):
        left = 0.0
        for k, (key, label) in enumerate(REGIONS):
            v = r["share"].get(key, 0.0)
            if v <= 0:
                continue
            ax.barh(y, v, left=left, height=0.62, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=label if y == 0 or key not in [x for x, _ in REGIONS[:k]] else None)
            if v >= 0.08:
                ax.text(left + v / 2, y, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8, color=INK if k in (2, 3, 4) else "white")
            left += v
    ax.set_yticks(range(len(rows)), names, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of DRAM reads in detailed phases", color=INK2, fontsize=9)
    handles = [matplotlib.patches.Patch(color=SERIES[k], label=label) for k, (_, label) in enumerate(REGIONS)]
    ax.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title("Where DRAM reads go, by model and context length", color=INK, fontsize=11, loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_mapping(ds, path):
    """Small multiples, one measure each: row-hit rate, read latency, ACT energy."""
    xs = list(range(len(REPLAYED)))
    panels = [("hit", "Row-hit rate"), ("rd_lat", "Read latency (cycles)"), ("act_req", "ACT energy per request, relative")]
    maps = [("rochrababgco_OPEN_PAGE", "Row-interleaved (rochrababgco)"), ("rocorababgch_OPEN_PAGE", "Line-interleaved (rocorababgch)")]
    for run in REPLAYED:
        for v in ds[run].values():
            v["act_req"] = v["act_energy"] / v["requests"]
    base = ds[REPLAYED[0]]["rochrababgco_OPEN_PAGE"]["act_req"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for k, (cfg, label) in enumerate(maps):
            ys = [ds[r][cfg][key] / (base if key == "act_req" else 1) for r in REPLAYED]
            ax.plot(xs, ys, color=SERIES[k], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
            ax.annotate(f"{ys[-2]:.2f}" if key == "hit" else f"{ys[-2]:.1f}", (xs[-2], ys[-2]), textcoords="offset points", xytext=(0, -14 if k == 0 and key != "hit" else 8), ha="center", fontsize=8, color=INK)
        ax.set_xticks(xs, [LABEL[r] for r in REPLAYED], fontsize=8, color=INK, rotation=15)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        if key == "hit":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    fig.suptitle("DRAMsim3, open page, at the trace's own timing (stories15M)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_saturated(ram, path):
    """Completion time under saturation, relative to RoBaRaCoCh, open page."""
    mappers = [("RoBaRaCoCh", "RoBaRaCoCh"), ("MOP4CLXOR", "MOP4CLXOR (XOR hash)"), ("ChRaBaRoCo", "ChRaBaRoCo")]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    w = 0.26
    for k, (m, label) in enumerate(mappers):
        ys = []
        for r in REPLAYED:
            by = {(x["mapper"], x["policy"]): x for x in ram[r]}
            ys.append(by[(m, "Open")]["cycles"] / by[("RoBaRaCoCh", "Open")]["cycles"])
        pos = [i + (k - 1) * (w + 0.02) for i in range(len(REPLAYED))]
        ax.bar(pos, ys, width=w, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=label)
        for p, y in zip(pos, ys):
            ax.text(p, y + 0.03, f"{y:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.axhline(1.0, color=INK2, linewidth=1)
    ax.set_xticks(range(len(REPLAYED)), [LABEL[r] for r in REPLAYED], fontsize=8.5, color=INK)
    ax.set_ylabel("Cycles to drain trace, relative", color=INK2, fontsize=9)
    ax.set_ylim(0, 2.0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Ramulator 2.1, saturated replay (no timestamps), open page", color=INK, fontsize=11, loc="left", pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    d, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    md = ["# Cross-run tables", "", f"Generated by `tools/report.py` from `{d}`. Do not edit by hand.", ""]

    # 1. Validation and per-token behaviour.
    rows = []
    md += ["## Runs and validation", "",
           "| Run | Model | Context | LLC | Prefetch | Detailed tokens | Counts match | Text matches native | Minsts/token | DRAM reads/token | DRAM writes/token | MPKI | gem5 row-hit | gem5 read latency (ns) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    present = [r for r in RUNS if os.path.exists(f"{d}/traces/{r['run']}/summary.json")]
    for r in present:
        s, counts_ok, text_ok, g = validate(d, r)
        ph = s["phases"]
        tok = [p for p in ph if p["id"] > 0] or ph  # phase 0 of a short-prompt run is the prompt
        f = lambda k: sum(p[k] for p in tok) / len(tok)
        hit = [p["row_hit_rate"] for p in tok if p["row_hit_rate"] is not None]
        reads = sum(v.get("reads", 0) for v in s["attribution"].values())
        share = {k: v.get("reads", 0) / reads for k, v in s["attribution"].items()}
        rows.append(dict(r, share=share, summary=s))
        md.append(
            f"| `{r['run']}` | {r['model']} | {r['ctx']} | {r['llc']} | {r['pf']} | {len(tok)} | {'yes' if counts_ok else 'NO'} | {'yes' if text_ok else 'NO'} "
            f"| {f('insts') / 1e6:,.0f} | {f('trace_reads'):,.0f} | {f('trace_writes'):,.0f} | {f('mpki'):.2f} | {sum(hit) / len(hit):.3f} | {f('read_lat_ns') / 1000:.1f} |"
        )
    md += ["", "Per-token figures average the generated-token phases. `analyze.py` stores the gem5 read latency in ticks (ps) under `read_lat_ns`; it is converted to ns here.", ""]

    # 2. Read and write composition.
    md += ["## Where DRAM traffic goes", "", "| Run | " + " | ".join(f"{l} reads" for _, l in REGIONS) + " | KV writes | Activation writes | Unattributed writes | Weights MiB | Experts MiB | KV MiB |", "|---|" + "---|" * (len(REGIONS) + 6)]
    for r in rows:
        a = r["summary"]["attribution"]
        wtot = max(sum(v.get("writes", 0) for v in a.values()), 1)
        cell = lambda k: f"{100 * r['share'].get(k, 0):.1f}%" if k in a else "-"
        wcell = lambda k: f"{100 * a[k]['writes'] / wtot:.1f}%" if k in a else "-"
        mib = lambda k: f"{a[k]['footprint_mib']:,.1f}" if k in a and "footprint_mib" in a[k] else "-"
        md.append(f"| `{r['run']}` | " + " | ".join(cell(k) for k, _ in REGIONS) + f" | {wcell('kv_cache')} | {wcell('activation')} | {wcell('unattributed')} | {mib('weights')} | {mib('weights.experts')} | {mib('kv_cache')} |")
    md += ["", "## Stream structure and offline mapping estimate", "",
           "Offline estimate: in-order open-page model, no timing or scheduling; only for comparing mappings.", "",
           "| Run | +1 line | Within 16 lines | Within 8 KiB row | Beyond | RoRaBaCoCh | RoCoRaBaCh | XOR-bank | RoRaBaCoCh, 4 KiB shuffled |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st, off = r["summary"]["stride"], r["summary"]["offline_row_hit"]
        ft, sh = off["SE first-touch"], off["4KiB shuffled"]
        md.append(f"| `{r['run']}` | {st['+1 line']:.2f} | {st['within 16 lines']:.2f} | {st['within 8 KiB row']:.2f} | {st['beyond']:.2f} | {ft['RoRaBaCoCh']['hit']:.3f} | {ft['RoCoRaBaCh']['hit']:.3f} | {ft['XOR-bank']['hit']:.3f} | {sh['RoRaBaCoCh']['hit']:.3f} |")
    fig_composition(rows, f"{out}/fig_read_composition.png")

    # 3. DRAM simulators.
    ds, ram = {}, {}
    md += ["", "## DRAMsim3, at trace timing", "", "Latency in DRAM cycles (tCK 0.83 ns). Energy from DRAMsim3's model, in its units, per request.", "",
           "| Trace | Mapping | Policy | Finished | Row-hit | Read latency | ACTs | ACT energy / request | Total energy / request | Avg bandwidth (GB/s) | Utilisation of 19.2 GB/s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    # Every analysed run with DRAM replays, stories15M first; the figures stay on stories15M.
    dram_runs = [r["run"] for r in present if os.path.isdir(f"{d}/traces/{r['run']}/dramsim3")]
    name = lambda run: LABEL.get(run, f"`{run}`")
    for run in dram_runs:
        s = json.load(open(f"{d}/traces/{run}/summary.json"))
        ds[run] = dramsim3(d, run, s["roi_reads"])
        for cfg, v in ds[run].items():
            m, p = cfg.split("_", 1)
            md.append(f"| {name(run)} | `{m}` | {p.replace('_PAGE', '').lower()} | {'yes' if v['complete'] else 'NO'} | {v['hit']:.3f} | {v['rd_lat']:.1f} | {v['act']:,} | {v['act_energy'] / v['requests']:,.0f} | {v['total_energy'] / v['requests']:,.0f} | {v['bw']:.2f} | {100 * v['bw'] / 19.2:.1f}% |")
        if os.path.exists(f"{d}/traces/{run}/ramulator.json"):
            ram[run] = json.load(open(f"{d}/traces/{run}/ramulator.json"))
    md += ["", "## Ramulator 2.1, saturated (no timestamps)", "", "| Trace | Mapper | Policy | Row-hit | Conflict | Read latency | Mcycles | Cycles vs RoBaRaCoCh open |", "|---|---|---|---|---|---|---|---|"]
    for run in [r for r in dram_runs if r in ram]:
        by = {(x["mapper"], x["policy"]): x for x in ram[run]}
        base = by[("RoBaRaCoCh", "Open")]["cycles"]
        for x in ram[run]:
            n = max(x["row_hits"] + x["row_misses"] + x["row_conflicts"], 1)
            md.append(f"| {name(run)} | {x['mapper']} | {x['policy']} | {x['hit_rate']:.3f} | {x['row_conflicts'] / n:.3f} | {x['avg_read_latency']:.1f} | {x['cycles'] / 1e6:.1f} | {x['cycles'] / base:.2f} |")
    fig_mapping(ds, f"{out}/fig_dramsim3_mapping.png")
    fig_saturated(ram, f"{out}/fig_ramulator_saturated.png")

    open(f"{out}/tables.md", "w").write("\n".join(md) + "\n")
    bad = [r["run"] for r in present if not validate(d, r)[1] or not validate(d, r)[2]]
    print(f"wrote {out}/tables.md and 3 figures; validation failures: {bad or 'none'}")


if __name__ == "__main__":
    main()
