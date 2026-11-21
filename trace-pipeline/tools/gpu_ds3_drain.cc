// Saturated DRAMsim3 driver for one GPU DRAM channel / pseudo-channel instance (gpu-ds3-owner).
// Trace = binary file of u64: (byte_address << 1) | is_write, in order. One request per cycle when the transaction queue
// accepts it; ticks until every read and every write transaction has completed. Prints cycles and counts; DRAMsim3 writes
// dramsim3.json (row hits, average read latency, ...) into OUTDIR.
// Build: see scripts in gpu/63_ds3_build.sh.   Usage: gpu_ds3_drain CONFIG.ini OUTDIR TRACE.bin
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "memory_system.h"
int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s config outdir trace.bin\n", argv[0]); return 1; }
    uint64_t done = 0, added = 0, wdone = 0, last_prog = 0, stamp = 0;
    dramsim3::MemorySystem ms(argv[1], argv[2], [&](uint64_t) { done++; last_prog = 1; }, [&](uint64_t) { done++; wdone++; last_prog = 1; });
    FILE *f = std::fopen(argv[3], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    std::vector<uint64_t> buf(1 << 16);
    size_t n = 0, pos = 0;
    uint64_t clk = 0, nw = 0, unissued = 0;
    bool have = false, eof = false;
    uint64_t addr = 0; bool wr = false;
    while (true) {
        ms.ClockTick();
        if (!have && !eof) {
            if (pos == n) { n = std::fread(buf.data(), 8, buf.size(), f); pos = 0; if (n == 0) eof = true; }
            if (!eof) { uint64_t v = buf[pos++]; wr = v & 1; addr = v >> 1; have = true; }
        }
        if (have && ms.WillAcceptTransaction(addr, wr)) { ms.AddTransaction(addr, wr); added++; nw += wr; have = false; last_prog = 1; }
        clk++;
        if (last_prog) { stamp = clk; last_prog = 0; }
        if (eof && !have && done == added) break;
        // DRAMsim3 merges a write whose address is already pending (one completion callback for two writes): the drain would
        // never see done == added. After 200000 idle cycles accept the stall; the cycle count reported is that of the last event.
        if (clk - stamp > 200000) {  // stalled before EOF (a request that is never accepted): count what was never issued
            clk = stamp; unissued = have ? 1 : 0; if (have) std::fprintf(stderr, "STALL at request %llu: addr 0x%llx write %d\n", (unsigned long long)added, (unsigned long long)addr, (int)wr);
            if (!eof) { uint64_t rest = (n - pos); while (true) { size_t m = std::fread(buf.data(), 8, buf.size(), f); if (m == 0) break; rest += m; } unissued += rest; }
            break;
        }
    }
    ms.PrintStats();
    std::printf("DRAIN_CYCLES %llu REQS %llu WRITES %llu MISSING %llu UNISSUED %llu\n", (unsigned long long)clk, (unsigned long long)added, (unsigned long long)nw, (unsigned long long)(added - done), (unsigned long long)unissued);
    return 0;
}
