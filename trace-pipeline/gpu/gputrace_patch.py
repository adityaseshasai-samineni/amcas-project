#!/usr/bin/env python3
"""Derive the amcas GPU trace tool from NVBit v1.7.7.3 tools/mem_trace (not redistributed here).
usage: gputrace_patch.py SRC_MEM_TRACE_DIR DST_DIR
Changes vs mem_trace: launch window GPUTRACE_START/END (only those launches are instrumented; every
launch is logged to GPUTRACE_OUT.launch.txt), skip LOCAL/SHARED accesses, record the true active+pred
lane mask, binary output GPUTRACE_OUT.bin (gzip later), opcode table GPUTRACE_OUT.opcodes.txt.
Record: u64 launch | u32 cta_x | u16 cta_y | u16 cta_z | u16 warp | u16 opcode | u32 mask | popcount(mask) x u64 addr.
"""
import shutil, sys, os
src, dst = sys.argv[1:3]
shutil.copytree(src, dst)
def sub(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) >= 1, (path, old[:50])
    s = s.replace(old, new, count)
    open(path, 'w').write(s)
c = dst + '/common.h'
sub(c, "    uint64_t addrs[32];\n", "    uint64_t addrs[32];\n    uint32_t mask;\n")
i = dst + '/inject_funcs.cu'
sub(i, """    /* if thread is predicated off, return */
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);""",
"""    int active_mask = __ballot_sync(__activemask(), pred);
    if (!pred) {
        return;
    }""")
sub(i, "    ma.opcode_id = opcode_id;\n", "    ma.opcode_id = opcode_id;\n    ma.mask = (uint32_t)active_mask;\n")
m = dst + '/mem_trace.cu'
sub(m, '// #define USE_ASYNC_STREAM', '#define USE_ASYNC_STREAM')  # flush channel + device sync after every launch: no lost records
sub(m, "uint64_t global_grid_launch_id = 0;\n", """uint64_t global_grid_launch_id = 0;
uint64_t win_start = 0, win_end = UINT64_MAX;
FILE* fbin = nullptr; FILE* flog = nullptr; std::string out_prefix = "gputrace";
""")
sub(m, "    GET_VAR_INT(verbose, \"TOOL_VERBOSE\", 0, \"Enable verbosity inside the tool\");\n",
"""    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    if (getenv("GPUTRACE_OUT")) out_prefix = getenv("GPUTRACE_OUT");
    if (getenv("GPUTRACE_START")) win_start = strtoull(getenv("GPUTRACE_START"), 0, 10);
    if (getenv("GPUTRACE_END")) win_end = strtoull(getenv("GPUTRACE_END"), 0, 10);
    fbin = fopen((out_prefix + ".bin").c_str(), "wb");
    flog = fopen((out_prefix + ".launch.txt").c_str(), "w");
""")
sub(m, """                instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {""",
"""                instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT ||
                instr->getMemorySpace() == InstrType::MemorySpace::LOCAL ||
                instr->getMemorySpace() == InstrType::MemorySpace::SHARED) {""")
# enter_kernel_launch: instrument only in window
sub(m, """    /* instrument */
    instrument_function_if_needed(ctx, func);
""", """    bool in_win = !(stream_capture || build_graph) && grid_launch_id >= win_start && grid_launch_id < win_end;
    if (in_win) instrument_function_if_needed(ctx, func);
""")
sub(m, "            printf(\n                \"MEMTRACE: CTX 0x%016lx - LAUNCH", "            fprintf(flog,\n                \"MEMTRACE: CTX 0x%016lx - LAUNCH", 2)
sub(m, "    nvbit_enable_instrumented(ctx, func, true);\n\n}", "    nvbit_enable_instrumented(ctx, func, in_win);\n\n}")
# recv: binary
s = open(m).read()
a = s.index("                std::stringstream ss;")
b = s.index("                num_processed_bytes += sizeof(mem_access_t);")
s = s[:a] + """                uint16_t hdr_y = ma->cta_id_y, hdr_z = ma->cta_id_z, w = ma->warp_id, op = ma->opcode_id;
                fwrite(&ma->grid_launch_id, 8, 1, fbin); fwrite(&ma->cta_id_x, 4, 1, fbin);
                fwrite(&hdr_y, 2, 1, fbin); fwrite(&hdr_z, 2, 1, fbin);
                fwrite(&w, 2, 1, fbin); fwrite(&op, 2, 1, fbin); fwrite(&ma->mask, 4, 1, fbin);
                for (int i = 0; i < 32; i++) if (ma->mask >> i & 1) fwrite(&ma->addrs[i], 8, 1, fbin);
""" + s[b:]
s = s.replace("    ctx_state->channel_host.destroy(false);\n", """    ctx_state->channel_host.destroy(false);
    if (fbin) { fclose(fbin); fbin = nullptr; }
    { FILE* fo = fopen((out_prefix + ".opcodes.txt").c_str(), "w");
      for (auto& kv : id_to_opcode_map) fprintf(fo, "%d %s\\n", kv.first, kv.second.c_str()); fclose(fo); }
    if (flog) { fclose(flog); flog = nullptr; }
""", 1)
# graph node launch log
s = s.replace("    printf(\n        \"MEMTRACE: CTX 0x%016lx - LAUNCH", "    fprintf(flog,\n        \"MEMTRACE: CTX 0x%016lx - LAUNCH")
open(m, 'w').write(s)
print("patched", dst)
