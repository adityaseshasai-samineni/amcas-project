#!/usr/bin/env python3
"""Find the launch window of the last decode step: smallest period L (>=50) such that the last two
L-launch blocks have identical kernel-name sequences. Prints "START END" (END exclusive). usage: find_window.py PREFIX.launch.txt"""
import re, sys
names = {}
for l in open(sys.argv[1]):
    m = re.search(r"Kernel name (.*) - grid launch id (\d+) - grid size", l)
    if m: names[int(m.group(2))] = m.group(1)
T = max(names) + 1
seq = [names.get(i, "") for i in range(T)]
for L in range(50, T // 2):
    if seq[T - L:] == seq[T - 2 * L:T - L]:
        print(T - L, T); sys.exit(0)
sys.exit("no period found")
