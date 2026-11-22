#!/usr/bin/env python3
"""Make the campaign scripts read one post-processed kernel list per CUDA context.

post-traces-processing writes kernelslist.g for the first context and
kernelslist_ctx_0x<ctx>.g for each further context (two-V100 MoE runs).
simulate_case.py and hardware_case.py read only kernelslist.g, so a MoE replay
would simulate one GPU and the Nsight gate would count one GPU's kernels.
kernelslist_sim_*.g files are replay inputs written by simulate_case.py and are
excluded. With a single context every result is unchanged.

Usage: patch_multictx_lists.py CAMPAIGN_DIR
"""
import pathlib, sys

R = pathlib.Path(sys.argv[1])


def patch(name, pairs):
    p = R / name
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (name, old[:80])
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", name)


patch("simulate_case.py", [
    ("S=pathlib.Path(cap['scratch']);lists=list((S/'traces').glob('kernelslist.g'));assert len(lists)==1\n"
     "entries=lists[0].read_text().splitlines();kernels=[x for x in entries if x.startswith('kernel-')];assert kernels,'empty selected GPUwindow'\n"
     "contexts=sorted(set(re.search(r'ctx_0x[0-9a-fA-F]+',x).group(0) for x in kernels));results=[]\n",
     "S=pathlib.Path(cap['scratch']);lists=sorted(p for p in (S/'traces').glob('kernelslist*.g') if not p.name.startswith('kernelslist_sim_'));assert S/'traces'/'kernelslist.g' in lists\n"
     "# AMCAS: one post-processed list per CUDA context (kernelslist.g, kernelslist_ctx_0x*.g).\n"
     "ctx_entries={}\n"
     "for lp in lists:\n"
     " es=lp.read_text().splitlines()\n"
     " for c in set(re.search(r'ctx_0x[0-9a-fA-F]+',x).group(0) for x in es if x.startswith('kernel-')):assert c not in ctx_entries,c;ctx_entries[c]=es\n"
     "entries=[x for es in ctx_entries.values() for x in es];kernels=[x for x in entries if x.startswith('kernel-')];assert kernels,'empty selected GPUwindow'\n"
     "contexts=sorted(ctx_entries);results=[]\n"),
    (" subset=[x for x in entries if not x.startswith('kernel-') or ctx in x]\n",
     " subset=[x for x in ctx_entries[ctx] if not x.startswith('kernel-') or ctx in x]\n"),
])

patch("hardware_case.py", [
    ("expected=sum(x.startswith('kernel-') for x in cap['kernel_lists']['kernelslist.g'])",
     "expected=sum(x.startswith('kernel-') for k,v in cap['kernel_lists'].items() if not k.startswith('kernelslist_sim_') for x in v)"),
])

patch("simulate_case_node03.sh", [
    ("""find '$S' -path '$S/simulation' -prune -o -type f -printf""",
     """find '$S' -path '$S/simulation' -prune -o -type f ! -name 'kernelslist_sim_*' -printf"""),
    ("""find "$S" -path "$S/simulation" -prune -o -type f -printf""",
     """find "$S" -path "$S/simulation" -prune -o -type f ! -name 'kernelslist_sim_*' -printf"""),
])
