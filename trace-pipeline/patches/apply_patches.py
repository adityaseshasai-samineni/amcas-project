#!/usr/bin/env python3
"""Apply the tracing patches to the gem5 and llama.cpp checkouts. Idempotent.

gem5:  record the virtual address of each request in the packet trace, so that
       physical accesses can be attributed to tensors afterwards.
llama: wrap each graph compute in m5_work_begin/end, and dump the address of
       every tensor in the graph just before it runs. Only active with
       -DAMCAS_M5, so the native validation binary is the same code.
"""

import os
import sys

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/amcas-pro/tracing")


def patch(path, old, new, marker):
    path = os.path.join(root, path)
    s = open(path).read()
    if marker in s:
        print("already patched:", path)
        return
    if s.count(old) != 1:
        sys.exit(f"anchor found {s.count(old)} times in {path}: {old!r}")
    open(path, "w").write(s.replace(old, new))
    print("patched:", path)


patch(
    "gem5/src/proto/packet.proto",
    "  optional uint64 pc = 7;\n",
    "  optional uint64 pc = 7;\n  optional uint64 vaddr = 8;\n",
    "optional uint64 vaddr",
)
patch(
    "gem5/src/sim/probe/mem.hh",
    "    Addr pc;\n    RequestorID id;\n",
    "    Addr pc;\n    Addr vaddr;\n    RequestorID id;\n",
    "Addr vaddr;",
)
patch(
    "gem5/src/sim/probe/mem.hh",
    "        pc(pkt->req->hasPC() ? pkt->req->getPC() : 0),\n",
    "        pc(pkt->req->hasPC() ? pkt->req->getPC() : 0),\n"
    "        vaddr(pkt->req->hasVaddr() ? pkt->req->getVaddr() : 0),\n",
    "vaddr(pkt->req->hasVaddr()",
)
patch(
    "gem5/src/mem/probes/mem_trace.cc",
    "        pkt_msg.set_pc(pkt_info.pc);\n",
    "        pkt_msg.set_pc(pkt_info.pc);\n"
    "    if (withPC && pkt_info.vaddr != 0)\n"
    "        pkt_msg.set_vaddr(pkt_info.vaddr);\n",
    "set_vaddr",
)

# gem5 SE mode with one CPU has no spare thread context, so clone() fails. The
# logger's worker thread is the only thread created with -t 1: print inline.
patch(
    "llama.cpp/common/log.cpp",
    "        running = true;\n\n        thrd = std::thread([this]() {\n",
    "        running = true;\n\n"
    "#ifdef AMCAS_M5\n"
    "        return;\n"
    "#endif\n"
    "        thrd = std::thread([this]() {\n",
    "#ifdef AMCAS_M5",
)
patch(
    "llama.cpp/common/log.cpp",
    "        tail = (tail + 1) % queue.size();\n"
    "        cv_new.notify_one();\n"
    "    }\n\n"
    "    void add_json(",
    "        tail = (tail + 1) % queue.size();\n"
    "#ifdef AMCAS_M5\n"
    "        size_t next_head;\n"
    "        flush_queue(head, tail, next_head);\n"
    "        head = next_head;\n"
    "        return;\n"
    "#endif\n"
    "        cv_new.notify_one();\n"
    "    }\n\n"
    "    void add_json(",
    "flush_queue(head, tail, next_head);\n        head = next_head;\n        return;",
)
patch(
    "llama.cpp/common/log.cpp",
    "        thrd.join();\n    }\n\n    void set_file(",
    "        if (thrd.joinable()) {\n"
    "            thrd.join();\n"
    "        } else {\n"
    "            head = tail;\n"
    "        }\n"
    "    }\n\n"
    "    void set_file(",
    "thrd.joinable()",
)

patch(
    "llama.cpp/src/llama-context.cpp",
    '#include "llama-moe-cache.h"\n',
    '#include "llama-moe-cache.h"\n'
    "\n"
    "#ifdef AMCAS_M5\n"
    "#include <gem5/m5ops.h>\n"
    "\n"
    "// One line per tensor reference: phase, node index, role, op, name, data, bytes, view_src.\n"
    "static void amcas_dump_graph(ggml_cgraph * gf, uint64_t phase) {\n"
    '    static FILE * f = fopen("amcas_tensors.tsv", "w");\n'
    "    if (f == nullptr) {\n"
    "        return;\n"
    "    }\n"
    "    const auto emit = [&](int i, const char * role, const ggml_tensor * t) {\n"
    '        fprintf(f, "%llu\\t%d\\t%s\\t%s\\t%s\\t%p\\t%zu\\t%s\\n",\n'
    "                (unsigned long long) phase, i, role, ggml_op_name(t->op), ggml_get_name(t), t->data,\n"
    '                ggml_nbytes(t), t->view_src ? ggml_get_name(t->view_src) : "");\n'
    "    };\n"
    "    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {\n"
    "        const ggml_tensor * node = ggml_graph_node(gf, i);\n"
    '        emit(i, "dst", node);\n'
    "        for (int j = 0; j < GGML_MAX_SRC; ++j) {\n"
    "            if (node->src[j]) {\n"
    '                emit(i, "src", node->src[j]);\n'
    "            }\n"
    "        }\n"
    "    }\n"
    "    fflush(f);\n"
    "}\n"
    "#endif\n",
    "amcas_dump_graph",
)
patch(
    "llama.cpp/src/llama-context.cpp",
    "    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);\n"
    "    if (status != GGML_STATUS_SUCCESS) {\n"
    '        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed',
    "#ifdef AMCAS_M5\n"
    "    static uint64_t amcas_phase = 0;\n"
    "    amcas_dump_graph(gf, amcas_phase);\n"
    "    m5_work_begin(amcas_phase, 0);\n"
    "#endif\n"
    "    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);\n"
    "#ifdef AMCAS_M5\n"
    "    m5_work_end(amcas_phase, 0);\n"
    "    ++amcas_phase;\n"
    "#endif\n"
    "    if (status != GGML_STATUS_SUCCESS) {\n"
    '        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed',
    "m5_work_begin(amcas_phase",
)
