# Hardware cost of the searched XOR bank (and channel) hash

Owner: extras-sim-owner, 2026-11-14. Numbers come from `trace-pipeline/tools/xorcost.py` over every `make.txt` in
`cluster-data/results/fullscale/{region,memsys,transfer,dramtech,realmem,mop4,chanloop}` (458 matrices: 335 bank, 123 channel).

## What the hash is
`xor_global` replaces each bank-field bit `b_k` by `b_k ^ r_i ^ r_j ^ r_l`, with at most 3 row-address bits per output bit (search fan-in limit,
`region_remap.search`), 5 bank bits on DDR4 (2 ranks x 4 BG x 4 banks) and 5 on the DDR5 models, plus (for `xor_global_chan`) 3-4 channel bits with the same form.
The remap never changes row, column or 2 MiB-granule bits, so it is a bijection that sits in the address-decode path between the physical address and
the (channel, rank, bank group, bank, row, column) fields, in front of the command queues.

## Gate counts (measured on the searched matrices)
A bank bit with m row terms needs m 2-input XOR gates in a tree of ceil(log2(m+1)) levels (m+1 inputs including the original bit).

| hash | memory model | matrices | output bits | XOR2 gates min/median/max | after pair sharing | tree depth (levels) | row bits per output |
|---|---|---|---|---|---|---|---|
| bank | DDR4 (ddr4 and realmem 6/12ch) | 50 | 5 | 0 / 5 / 10 | 0 / 5 / 10 | <= 2 | <= 3 |
| bank | DDR5 x16 | 130 | 5 | 1 / 6 / 13 | 1 / 6 / 11 | <= 2 | <= 3 |
| bank | DDR5 x8 | 70 | 5 | 1 / 5 / 12 | 1 / 5 / 11 | <= 2 | <= 3 |
| bank | other labels (thp/x8,x16 MOP4 etc.) | 85 | 5 | 1 / 5 / 9 | 1 / 5 / 9 | <= 2 | <= 3 |
| channel | DDR5 x16 (4 channel bits) | 59 | 4 | 1 / 4 / 12 | 1 / 4 / 11 | <= 2 | <= 3 |
| channel | DDR5 x8 (3 channel bits) | 64 | 3 | 0 / 3 / 8 | 0 / 3 / 8 | <= 2 | <= 3 |

Per memory system the typical full bank hash is **5-6 XOR2 gates (median), at most 13; critical path 1-2 XOR2 levels.** Worst case: 13 gates
(transfer/xfer packed DDR5), 11 after sharing common row-bit pairs between outputs. A bank hash plus channel hash together is below ~25 XOR2 gates.

## Latency
A 2-input XOR is roughly 2x the delay of a NAND2 in a typical standard-cell library (teaching library figures: NAND2 0.10 ns vs XOR2 0.21 ns, i.e. ~2x; smallest
XOR2 cells are slower still), i.e. about 1.5-2.5 FO4 per level; a transmission-gate or AOI XOR is 1.5-2 FO4. So 2 levels = **~3-5 FO4, about 20-50 ps in a 7-16 nm
process (FO4 ~ 8-15 ps) and ~100-150 ps at the 28-45 nm nodes memory controllers often use for the PHY-side logic**. The address arrives from the on-chip network many cycles
before it is needed and the decoded fields feed queue insertion, so the hash is off the DRAM-timing critical path: the controller clock is the DRAM command clock
(1.0-1.5 GHz for DDR4-2933, 2.4 GHz for DDR5-4800 class, i.e. 400-830 ps per cycle), and 2 XOR levels fit in the existing decode stage. The existing decode
(slice/offset extraction, rank/bank-group select, channel interleave with a modulo for non-power-of-two channel counts) already costs several gate levels;
the hash adds 1-2 levels, typically absorbed in the same pipeline stage or one extra stage (1 controller cycle) in the worst case. Because it is applied once per request at queue
entry (not per DRAM command), an extra pipeline cycle adds ~0.7 ns to read latency of ~50-100 ns (below 1%), which the simulations (read latency -5 to -22% on THP) dwarf.

## Area
XOR2 is about 2-3 NAND2 equivalents in a standard-cell library (teaching library: 27 vs 10 um^2 = 2.7x; SkyWater 130 nm smallest XOR2 12.8 um^2). 13 XOR2 = 26-40
gate equivalents; including the channel hash, under 100 GE per memory controller, or **well under 0.001 mm^2 at 7-28 nm**; negligible next to a controller's queues and
scheduler (hundreds of thousands of GE) and to the PHY.

## Configurability (if the matrix must be programmable)
A fixed-wire hash is cheapest but the searched matrices differ per DRAM organisation and workload (region: x8 vs x16 vs DDR4 give different row bits; see
table), so a product would expose a small register: per bank/channel output bit, a mask over the candidate row bits. The searched matrices only use row bits 0-15
(16 candidates; bits 0-13 on most). Programmable cost per output bit: a 16-bit AND-mask (16 AND2 + 16 flops/latch) feeding an XOR reduction tree of 16 inputs
(15 XOR2, depth 4). For 5 bank + 4 channel bits: 9 x (16 AND2 + 15 XOR2 + 16 register bits) = ~144 register bits and ~280 gates (~800 GE), depth 4 XOR2 + 1 AND = 5 levels (~7-12 FO4;
still one controller cycle at 1.5 GHz in modern nodes, or one extra pipeline stage). Restricting the mask to <=3 set bits per output (as searched) allows a cheaper form: three
4-bit row-index selectors per output (mux 16:1, 4 levels) = 5 x 3 x 4 = 60 register bits plus 45 16:1 muxes; the AND-mask form is simpler and smaller. Either way it is
a boot-time (BIOS/firmware) setting, like the existing programmable interleave/hash registers in server memory controllers; changing it on a live system requires the usual
memory re-initialisation or data migration since it remaps physical placement.

## Existing practice and sources
- XOR-based bank hashing is old and cheap: Zhang, Zhu and Zhang, "A permutation-based page interleaving scheme to reduce row-buffer conflicts and exploit data locality" (MICRO-33, 2000) XORs row/cache-index bits into the bank index; they state the hardware cost is trivial (https://home.engineering.iastate.edu/zzhang/publications/abs00-micro-interleave.htm, tech report https://home.engineering.iastate.edu/zzhang/publications/tr00-micro-interleave.pdf), and Sun adopted it per secondary sources.
- Shipping Intel controllers already use linear XOR functions of physical-address bits for channel, rank and bank selection (several bits XOR-ed, Haswell-EP DDR4 servers included); DRAMA reverse engineered them (Pessl et al., USENIX Security 2016, https://arxiv.org/pdf/1511.08756). That is, a 2-3-input XOR per bank/channel bit in the decode path is what current servers already pay for; the proposal asks for the row-bit set to be chosen for transformer streams and for large pages.
- Gate figures: relative XOR2 vs NAND2 area/delay from a teaching standard-cell library (https://users.ece.utexas.edu/~mcdermot/arch/0LD/labs_old/board_instructions/basic_block_library.pdf: NAND2 10 um^2, 0.10 ns; XOR2 27 um^2, 0.21 ns); XOR2 cell families with FO4-normalised delays (https://www.vlsitechnology.org/html/cells/vxlib013/xor2.html); SkyWater 130 nm XOR2 (https://foss-eda-tools.googlesource.com/skywater-src-nda/+/d2c5021a49fe2d18e1092970a921217aa66f4c1b/scs8ls_ds/datasheets/scs8ls_tt_1.8v_25C/XOR2x.html). The absolute ps and mm^2 values above are my scaled estimates (FO4 ~ 8-15 ps at 7-16 nm is an assumption, not looked up) and carry a factor ~2 uncertainty; the gate counts and depths are exact for the searched matrices.

## Verdict
The searched hash is 5-6 XOR2 gates (max 13) and 1-2 gate levels per memory system (plus a similar channel hash): hardware cost is negligible in area and does not lengthen
any DRAM-timing path. A programmable version costs ~144 register bits and ~800 GE with a 4-5-level tree. Hardware cost is not the obstacle to the proposal; the gain (THP, memory-bound regime) is.
