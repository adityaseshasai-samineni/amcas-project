"""Capture the SoC<->DRAM request stream of a workload in gem5 SE mode.

The trace point is a CommMonitor between the coherent memory bus and the DRAM
controller, so it records exactly what a controller would receive: LLC misses,
LLC writebacks and prefetches, in arrival order, with physical addresses.

Fast-forwards in atomic mode with caches on (so they are warm), and switches to
the detailed CPU at the first m5_work_begin issued by the patched llama.cpp.
Every work item boundary is logged to phases.tsv and dumped to stats.txt.
"""

import argparse
import shlex

import m5
from m5.objects import *

ap = argparse.ArgumentParser()
ap.add_argument("--cmd", required=True, help="binary and its arguments")
ap.add_argument("--cwd", default=None)
ap.add_argument("--cpu", choices=["o3", "timing"], default="o3")
ap.add_argument("--clock", default="3GHz")
ap.add_argument("--mem-size", default="8GiB")
ap.add_argument("--l1d", default="32KiB")
ap.add_argument("--l1i", default="32KiB")
ap.add_argument("--l2", default="1MiB")
ap.add_argument("--l3", default="22MiB")
ap.add_argument("--l3-assoc", type=int, default=11)
ap.add_argument("--prefetch", choices=["none", "stride"], default="stride")
ap.add_argument("--dram", default="DDR4_2400_8x8")
ap.add_argument("--no-trace", action="store_true")
ap.add_argument(
    "--detail-from",
    type=int,
    default=0,
    help="work item at which to switch to the detailed CPU; earlier ones "
    "run in atomic mode, so 1 skips detailed simulation of the prompt",
)
ap.add_argument(
    "--max-work-items",
    type=int,
    default=0,
    help="stop after this many completed work items (0 = run to exit)",
)
args = ap.parse_args()


class Cache_(Cache):
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    tgts_per_mshr = 20


def cache(size, assoc, lat, mshrs, **kw):
    return Cache_(
        size=size,
        assoc=assoc,
        tag_latency=lat,
        data_latency=lat,
        response_latency=lat,
        mshrs=mshrs,
        **kw,
    )


system = System(
    mem_mode="atomic",
    mem_ranges=[AddrRange(args.mem_size)],
    cache_line_size=64,
    exit_on_work_items=True,
    clk_domain=SrcClockDomain(clock=args.clock, voltage_domain=VoltageDomain()),
)

# Fast-forward CPU. The detailed CPU takes over at the first work item.
system.cpu = X86AtomicSimpleCPU(cpu_id=0)
argv = shlex.split(args.cmd)
system.workload = SEWorkload.init_compatible(argv[0])
# llama.cpp falls back to getpwuid() when HOME is unset. In a static binary that
# makes glibc dlopen the host's NSS libraries, whose TLS does not work in SE mode.
cwd = args.cwd or m5.options.outdir
process = Process(
    pid=100, cmd=argv, executable=argv[0], cwd=cwd, env=[f"HOME={cwd}"]
)
system.cpu.workload = process
system.cpu.createThreads()
system.cpu.createInterruptController()

system.cpu.addPrivateSplitL1Caches(
    cache(args.l1i, 8, 1, 8, is_read_only=True, writeback_clean=True),
    cache(args.l1d, 8, 2, 16),
)
system.l2bus = L2XBar()
system.l2cache = cache(args.l2, 16, 7, 48)
if args.prefetch == "stride":
    # Server cores run a stream prefetcher at L2. Without one the DRAM stream
    # is demand-only and its row locality does not match real hardware.
    system.l2cache.prefetcher = StridePrefetcher(degree=8, queue_size=64)
system.l3bus = L2XBar()
system.l3cache = cache(args.l3, args.l3_assoc, 20, 64)

system.cpu.connectCachedPorts(system.l2bus.cpu_side_ports)
system.l2cache.cpu_side = system.l2bus.mem_side_ports
system.l2cache.mem_side = system.l3bus.cpu_side_ports
system.l3cache.cpu_side = system.l3bus.mem_side_ports

system.membus = SystemXBar()
system.l3cache.mem_side = system.membus.cpu_side_ports
system.cpu.connectUncachedPorts(
    system.membus.cpu_side_ports, system.membus.mem_side_ports
)
system.system_port = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = getattr(m5.objects, args.dram)(
    range=system.mem_ranges[0]
)

# The trace point: membus -> CommMonitor -> DRAM controller.
system.monitor = CommMonitor()
system.membus.mem_side_ports = system.monitor.cpu_side_port
system.monitor.mem_side_port = system.mem_ctrl.port
if not args.no_trace:
    system.monitor.trace = MemTraceProbe(
        trace_file="dram.trc.gz", trace_compress=True, with_pc=True
    )

Detailed = X86O3CPU if args.cpu == "o3" else X86TimingSimpleCPU
system.detailed = Detailed(switched_out=True, cpu_id=0)
system.detailed.workload = process
system.detailed.clk_domain = system.cpu.clk_domain
system.detailed.isa = system.cpu.isa
if args.cpu == "o3":
    # Cascade Lake sized window, to match the node used for native validation.
    system.detailed.numROBEntries = 224
    system.detailed.LQEntries = 72
    system.detailed.SQEntries = 56
system.detailed.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()

phases = open(f"{m5.options.outdir}/phases.tsv", "w")
print("event\twork_id\ttick\tinsts", file=phases)


def log_phase(event, work_id):
    cpu = system.detailed if switched else system.cpu
    print(
        f"{event}\t{work_id}\t{m5.curTick()}\t{cpu.totalInsts()}",
        file=phases,
        flush=True,
    )


switched = False
completed = 0
while True:
    ev = m5.simulate()
    cause = ev.getCause()
    if cause == "workbegin":
        if not switched and ev.getCode() >= args.detail_from:
            m5.stats.dump()
            m5.switchCpus(system, [(system.cpu, system.detailed)])
            switched = True
        if switched:
            m5.stats.reset()
        log_phase("begin", ev.getCode())
    elif cause == "workend":
        log_phase("end", ev.getCode())
        if switched:
            m5.stats.dump()
            completed += 1
        if args.max_work_items and completed >= args.max_work_items:
            cause = f"reached {completed} work items"
            break
    else:
        break

m5.stats.dump()
print(f"Exiting @ tick {m5.curTick()} because {cause} (code {ev.getCode()})")
