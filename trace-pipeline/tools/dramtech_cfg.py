#!/usr/bin/env python3
"""Sized DRAM-technology systems for the dramtech study (LPDDR5, LPDDR6, HBM3, HBM4, GDDR6; DDR5 as anchor).

Shared by ramulator_replay.py (--system), dramtech_traces.py, dramtech_replay.py, dramtech_probe.py.
Needs the Ramulator Python package (PYTHONPATH=ramulator2-amcas/python) only inside build_mem().

All systems use RoBaRaCoCh on the intra-channel address and CacheLineInterleave at 256 B
(channel bits at address [8, 8+W)), like the DDR5 x16 study. Transactions of the new standards
are 32 B (LPDDR5 x16 BL16, LPDDR6 payload, HBM pseudo-channel BL8, GDDR6 x16 BL16), not 64 B, so a
64 B LLC miss is replayed as TWO sector requests (addr, addr+32) in the same channel, same row
(`sectors = 2`). Peak GB/s is the data-bus rate of the whole system (raw for LPDDR6: payload is
32 of 36 B per burst).

Layout fields (see remap.LAYOUTS): col = column bits after removing the prefetch (including the
sector bit 5), bank = all bank-region bits between column and row (rank, pseudo-channel, stack-id,
bank group, bank), row = row bits, off = transaction offset bits, s/W = channel shift/width.
"""
import math

# name -> system description
SYSTEMS = {
    # anchor: the existing DDR5-4800 16-subchannel system (307 GB/s), 64 B transactions
    "ddr5x16": dict(std="DDR5", org="DDR5_16Gb_x8", timing="DDR5_4800B", ctrl="GenericDDR", nch=16, ib=2, tx=64,
                    layout="ddr5", peak=307.2, tck_ps=416, tick_mult=1, issue=4, kw={}, over={},
                    what="DDR5-4800, 16 x 32-bit subchannels (8 DIMM channels)"),
    # LPDDR5-6400 x16 channels. 32 channels = Grace-like 512-bit interface (Grace uses LPDDR5X-8533 = 546 GB/s;
    # ramulator2 has only the 5500/6400 presets).
    "lpddr5x32": dict(std="LPDDR5", org="LPDDR5_16Gb_x16", timing="LPDDR5_6400", ctrl="LPDDR5", nch=32, ib=3, tx=32,
                      layout="lpddr5x32", peak=32 * 12.8, tck_ps=1250, tick_mult=1, issue=32, kw={}, over={},
                      what="LPDDR5-6400, 32 x16 channels (Grace-like, 410 GB/s)"),
    # laptop/phone-class: 8 x16 channels, 2 ranks to reach a 32 GiB address space (like the 32 GB pool)
    "lpddr5x8": dict(std="LPDDR5", org="LPDDR5_16Gb_x16", timing="LPDDR5_6400", ctrl="LPDDR5", nch=8, ib=3, tx=32,
                     layout="lpddr5x8", peak=8 * 12.8, tck_ps=1250, tick_mult=1, issue=8, kw={}, over=dict(rank=2),
                     what="LPDDR5-6400, 8 x16 channels, 2 ranks (laptop-class, 102 GB/s)"),
    # LPDDR6-10667 BL24, 12-bit sub-channels (payload 32 B per BL24 burst: 14.2 GB/s payload of 16 GB/s raw)
    "lpddr6x32": dict(std="LPDDR6", org="LPDDR6_16Gb_x12", timing="LPDDR6_10667_BL24", ctrl="LPDDR6", nch=32, ib=3, tx=32,
                      layout="lpddr6x32", peak=32 * 14.22, tck_ps=375, tick_mult=1, issue=12, kw={}, over={},
                      what="LPDDR6-10667, 32 x12 sub-channels (455 GB/s payload)"),
    "lpddr6x8": dict(std="LPDDR6", org="LPDDR6_16Gb_x12", timing="LPDDR6_10667_BL24", ctrl="LPDDR6", nch=8, ib=3, tx=32,
                     layout="lpddr6x8", peak=8 * 14.22, tck_ps=375, tick_mult=1, issue=4, kw={}, over=dict(rank=2),
                     what="LPDDR6-10667, 8 x12 sub-channels, 2 ranks (laptop-class, 114 GB/s)"),
    # one HBM3 stack: 16 channels x 2 pseudo-channels x 32 bit at 6.4 Gb/s = 819 GB/s, 32 GiB (8-high x 32 Gb)
    "hbm3x16": dict(std="HBM3", org="HBM3_32Gb_8hi", timing="HBM3_6400Mbps", ctrl="HBM34", nch=16, ib=3, tx=32,
                    layout="hbm3x16", peak=819.2, tck_ps=625, tick_mult=2, issue=16, kw=dict(read_buffer_size=128, write_buffer_size=64),
                    over={}, what="HBM3 one stack, 16 ch x 2 PC, 6.4 Gb/s (819 GB/s), 32 GiB"),
    # one HBM4 stack: 32 channels x 2 pseudo-channels at 8 Gb/s = 2.05 TB/s, 32 GiB (8-high x 32 Gb)
    "hbm4x32": dict(std="HBM4", org="HBM4_32Gb_8Hi", timing="HBM4_8000Mbps", ctrl="HBM34", nch=32, ib=3, tx=32,
                    layout="hbm4x32", peak=2048.0, tck_ps=500, tick_mult=2, issue=32, kw=dict(read_buffer_size=64, write_buffer_size=64),
                    over={}, what="HBM4 one stack, 32 ch x 2 PC, 8 Gb/s (2.05 TB/s), 32 GiB"),
    # GDDR6: ramulator2 models 16-bit channels (2 per device); preset speed is 14 Gb/s (not 16);
    # 16 channels = 8 x 32-bit = 448 GB/s, 32 GiB with the 32 Gb-density preset (4 KiB rows in the model)
    "gddr6x16": dict(std="GDDR6", org="GDDR6_32Gb_x16", timing="GDDR6_14000_1350mV_double", ctrl="GenericDDR", nch=16, ib=3, tx=32,
                     layout="gddr6x16", peak=16 * 28.0, tck_ps=572, tick_mult=1, issue=16, kw={},
                     over=dict(nRFCab=210, nRFCpb=105),  # 8 Gb-device tRFC (120/60 ns): the preset resolves no refresh timing for 32 Gb
                     what="GDDR6 14 Gb/s, 16 x16 channels (= 8 x 32-bit, 448 GB/s)"),
}

# layouts to register in remap.LAYOUTS: col/bank/row bits, tx offset bits, channel shift s, width W
LAYOUT_DEFS = {
    "lpddr5x32": dict(col=6, bank=4, row=16, off=5, s=8, W=5),
    "lpddr5x8": dict(col=6, bank=5, row=16, off=5, s=8, W=3),
    "lpddr6x32": dict(col=6, bank=4, row=16, off=5, s=8, W=5),
    "lpddr6x8": dict(col=6, bank=5, row=16, off=5, s=8, W=3),
    "hbm3x16": dict(col=5, bank=6, row=15, off=5, s=8, W=4),
    "hbm4x32": dict(col=5, bank=6, row=14, off=5, s=8, W=5),
    "gddr6x16": dict(col=7, bank=4, row=15, off=5, s=8, W=4),
}


def register(R, G=None):
    """Add the layouts to remap.LAYOUTS (and channel counts to region_remap.NCH)."""
    R.LAYOUTS.update(LAYOUT_DEFS)
    if G is not None:
        for k, v in SYSTEMS.items():
            if v["layout"] in LAYOUT_DEFS:
                G.NCH[v["layout"]] = v["nch"]


def tick_ps(sysd):
    return sysd["tck_ps"] / sysd["tick_mult"]


def build_mem(sysd, policy, nch=None, ib=None, ramulator=None):
    """GenericDRAM with nch controllers of the system's standard."""
    import ramulator as rm
    nch = sysd["nch"] if nch is None else nch
    ib = sysd["ib"] if ib is None else ib

    def ctrl():
        dram = getattr(rm.dram, sysd["std"])(org_preset=sysd["org"], timing_preset=sysd["timing"], **sysd["over"])
        return getattr(rm.controller, sysd["ctrl"])(
            dram=dram, scheduler=rm.scheduler.FRFCFS(), refresh_manager=rm.refresh_manager.AllBank(),
            row_policy=getattr(rm.row_policy, policy)(), addr_mapper=rm.addr_mapper.RoBaRaCoCh(), **sysd["kw"])

    return rm.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl() for _ in range(nch)],
        channel_mapper=rm.channel_mapper.CacheLineInterleave(interleave_bits=ib))


def cost_params(sysd):
    """(bus, tmiss) for the XOR-search cost models (remap.cost / region_remap.cost2 / memsys_study.cost3):
    bus = data-bus time of one 64 B line on the busiest data bus of a channel, tmiss = row-cycle time (tRC), both in
    the spec's own time unit. The DDR5 anchor keeps the original (4, 55)."""
    if sysd["std"] == "DDR5":
        return 4.0, 55.0
    import ramulator as rm
    org, t = getattr(rm.dram, sysd["std"])(org_preset=sysd["org"], timing_preset=sysd["timing"], **sysd["over"]).resolve()
    nbl = t.get("nBL_min", t.get("nBL"))
    npc = org.get("pseudochannel", 1)
    trc = t.get("nRC", t["nRAS"] + t["nRP"])
    return 2.0 * nbl / npc, float(trc)
