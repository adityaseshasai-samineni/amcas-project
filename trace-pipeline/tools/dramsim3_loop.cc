// Closed-loop DRAMsim3 driver (extras-sim-owner): N DRAMsim3 channel instances fed by NC trace-driven cores, the same
// core model as Ramulator's SimpleO3 setup used for the closed-loop studies.
//   Core: ipc instructions/core-cycle retire in order (head load must be complete) and ipc issue/core-cycle into an
//   instruction window of WINDOW entries; each DRAM read is preceded by BUBBLES non-memory instructions (always ready);
//   a load needs a free MSHR (MSHR per core) and a free slot in the channel's DRAMsim3 transaction queue, else the core
//   stalls. Core clock = CR x memory clock (DDR4-2933: CR=2). Loads complete via the DRAMsim3 read callback.
//   Reads only (the stream has >99.6% reads); the LLC is not modelled (the traces are post-LLC misses).
// Modes:
//   closed: per-core files (decimal or "LD 0x.." addresses, channel in the pow2 field at bits [8,8+W2) as written by
//           realmem_traces.py, N channels, channel = field value, intra address = field removed).
//   sat   : a single file, per channel in-order, one request per channel per memory cycle (== dramsim3_drain per channel).
// A run ends when every core has retired min(len)*(BUBBLES+1) instructions (like SimpleO3's num_expected_insts); cores that
// finish early keep running until all are done. Cycles are memory-clock cycles.
// Build: g++ -O2 -std=c++14 -I$D/src -I$D/ext/headers -I$D/ext/fmt/include dramsim3_loop.cc -L$D -ldramsim3 -Wl,-rpath,$D -o dramsim3_loop
// Usage: dramsim3_loop CONFIG.ini OUTDIR NCH MODE BUBBLES WINDOW MSHR IPC CR TRACE...   (MODE closed|sat; sat ignores the rest)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "memory_system.h"
typedef unsigned long long u64;
struct Req { int ch; u64 a; };
static std::vector<Req> load(const char *path, int nch) {
    int w2 = nch == 1 ? 0 : 0; while ((1 << w2) < nch) w2++;
    std::vector<Req> v; std::ifstream f(path); std::string l;
    while (std::getline(f, l)) {
        const char *s = l.c_str();
        if (!strncmp(s, "LD ", 3)) s += 3;
        u64 p = std::strtoull(s, nullptr, 0);
        int ch = nch == 1 ? 0 : (int)((p >> 8) & ((1ull << w2) - 1));
        u64 in = nch == 1 ? p : (((p >> (8 + w2)) << 8) | (p & 255));
        v.push_back({ch, in});
    }
    return v;
}
struct Ld { u64 seq; bool done; };
struct Core {
    std::vector<Req> tr; size_t pos = 0; u64 next_seq = 0, retired = 0, bub_left = 0, target = 0, fin_cycle = 0; bool fin = false;
    std::deque<Ld> infl; u64 issued = 0;
};
int main(int argc, char **argv) {
    if (argc < 10) { std::fprintf(stderr, "usage: %s config outdir nch mode bubbles window mshr ipc cr trace...\n", argv[0]); return 1; }
    int nch = atoi(argv[3]); std::string mode = argv[4];
    u64 bub = strtoull(argv[5], 0, 0), W = strtoull(argv[6], 0, 0), M = strtoull(argv[7], 0, 0);
    int ipc = atoi(argv[8]), cr = atoi(argv[9]);
    std::vector<std::unique_ptr<dramsim3::MemorySystem>> ms;
    std::vector<Core> cores;
    std::unordered_map<u64, std::deque<std::pair<int, u64>>> pend;  // (ch,addr) -> (core,seq)
    u64 sat_done = 0;
    auto cb = [&](int ch) { return [&, ch](uint64_t a) {
        if (mode == "sat") { sat_done++; return; }
        auto it = pend.find(a | ((u64)ch << 52)); if (it == pend.end() || it->second.empty()) return;
        auto cs = it->second.front(); it->second.pop_front();
        for (auto &l : cores[cs.first].infl) if (l.seq == cs.second) { l.done = true; break; }
    }; };
    for (int c = 0; c < nch; c++) {
        std::string od = std::string(argv[2]) + "/ch" + std::to_string(c); std::string mk = "mkdir -p " + od; if (system(mk.c_str())) return 2;
        ms.emplace_back(new dramsim3::MemorySystem(argv[1], od, cb(c), [](uint64_t) {}));
    }
    u64 clk = 0, total = 0;
    if (mode == "sat") {
        std::vector<std::vector<u64>> q(nch); std::vector<size_t> qp(nch, 0);
        for (auto &r : load(argv[10], nch)) q[r.ch].push_back(r.a);
        for (auto &v : q) total += v.size();
        u64 added = 0;
        while (true) {
            for (int c = 0; c < nch; c++) {
                ms[c]->ClockTick();
                if (qp[c] < q[c].size() && ms[c]->WillAcceptTransaction(q[c][qp[c]], false)) { ms[c]->AddTransaction(q[c][qp[c]], false); qp[c]++; added++; }
            }
            clk++;
            if (added == total && sat_done == total) break;
        }
        for (auto &m : ms) m->PrintStats();
        std::printf("RESULT mode=sat cycles=%llu reads=%llu\n", clk, total);
        return 0;
    }
    cores.resize(argc - 10);
    size_t minlen = ~0ull;
    for (size_t i = 0; i < cores.size(); i++) { cores[i].tr = load(argv[10 + i], nch); minlen = std::min(minlen, cores[i].tr.size()); }
    for (auto &c : cores) { c.target = minlen * (bub + 1); c.bub_left = bub; }
    size_t nfin = 0;
    while (nfin < cores.size()) {
        for (auto &m : ms) m->ClockTick();
        for (int t = 0; t < cr; t++) {
            for (size_t ci = 0; ci < cores.size(); ci++) {
                Core &c = cores[ci];
                for (int i = 0; i < ipc; i++) {  // retire
                    if (!c.infl.empty() && c.infl.front().seq == c.retired) { if (c.infl.front().done) { c.infl.pop_front(); c.retired++; } else break; }
                    else if (c.retired < c.next_seq) c.retired++;
                    else break;
                }
                for (int i = 0; i < ipc; i++) {  // issue
                    if (c.pos >= c.tr.size() || c.next_seq - c.retired >= W) break;
                    if (c.bub_left > 0) { c.bub_left--; c.next_seq++; continue; }
                    if (c.infl.size() >= M) break;
                    const Req &r = c.tr[c.pos];
                    if (!ms[r.ch]->WillAcceptTransaction(r.a, false)) break;
                    ms[r.ch]->AddTransaction(r.a, false);
                    pend[r.a | ((u64)r.ch << 52)].push_back({(int)ci, c.next_seq});
                    c.infl.push_back({c.next_seq, false}); c.next_seq++; c.pos++; c.issued++; c.bub_left = bub;
                }
                if (!c.fin && c.retired >= c.target) { c.fin = true; c.fin_cycle = clk + 1; nfin++; }
            }
        }
        clk++;
    }
    u64 fc = 0, iss = 0; for (auto &c : cores) { fc = std::max(fc, c.fin_cycle); iss += c.issued; }
    for (auto &m : ms) m->PrintStats();
    std::printf("RESULT mode=closed cycles=%llu reads_issued=%llu minlen=%zu cores=%zu per_core_fin=", fc, iss, minlen, cores.size());
    for (auto &c : cores) std::printf("%llu,", c.fin_cycle);
    std::printf("\n");
    return 0;
}
