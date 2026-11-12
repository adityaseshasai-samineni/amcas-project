// Saturated DRAMsim3 driver: feeds "0xADDR" (read) lines back to back, one per cycle when the transaction queue
// accepts, ticks until every read has completed, prints the drain cycle count. (dramsim3main only runs a fixed -c.)
// Build: g++ -O2 -std=c++14 -I$D/src -I$D/ext/headers -I$D/ext/fmt/include dramsim3_drain.cc -L$D -ldramsim3 -Wl,-rpath,$D -o dramsim3_drain
// Usage: dramsim3_drain CONFIG.ini OUTDIR TRACE
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include "memory_system.h"
int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s config outdir trace\n", argv[0]); return 1; }
    uint64_t done = 0, added = 0;
    dramsim3::MemorySystem ms(argv[1], argv[2], [&](uint64_t) { done++; }, [&](uint64_t) {});
    std::ifstream f(argv[3]);
    std::string line;
    uint64_t addr = 0, clk = 0;
    bool have = false, eof = false;
    while (true) {
        ms.ClockTick();
        if (!have && !eof) {
            if (std::getline(f, line)) { addr = std::strtoull(line.c_str(), nullptr, 0); have = true; } else eof = true;
        }
        if (have && ms.WillAcceptTransaction(addr, false)) { ms.AddTransaction(addr, false); added++; have = false; }
        clk++;
        if (eof && !have && done == added) break;
    }
    ms.PrintStats();
    std::printf("DRAIN_CYCLES %llu READS %llu\n", (unsigned long long)clk, (unsigned long long)added);
    return 0;
}
