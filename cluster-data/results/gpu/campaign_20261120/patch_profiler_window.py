#!/usr/bin/env python3
"""Repair pinned tracer's documented ACTIVE_FROM_START=0 profiler window.
Run only on an isolated tracer source copy; requires two-kernel ROI preflight.
"""
import pathlib,sys
p=pathlib.Path(sys.argv[1]);s=p.read_text()
old='''  if (active_from_start && should_trace_kernel(ctx_kernelid[ctx], func_name))
    active_region = nvbit_instrumentation_enabled;
  else
    active_region = false;'''
new='''  // AMCAS: profiler mode owns active_region across kernel launches.
  if (active_from_start) {
    active_region = should_trace_kernel(ctx_kernelid[ctx], func_name) &&
                    nvbit_instrumentation_enabled;
  }'''
if new in s:raise SystemExit(0)
assert s.count(old)==1,'pinned tracer source differs; inspect before editing'
p.write_text(s.replace(old,new))
