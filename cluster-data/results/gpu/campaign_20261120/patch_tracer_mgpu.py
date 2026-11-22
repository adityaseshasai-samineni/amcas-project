#!/usr/bin/env python3
"""Multi-GPU fix for the Accel-Sim NVBit tracer (tracer_tool-amcas20).

Bug: one receiver thread is created with the first CUDA context and writes
every traced instruction to ctx_resultsFile[<that context>]. When a traced
kernel runs on a second GPU (Qwen3-30B split over two V100s), its records are
written through the first context's FILE*, which is closed or never opened in
the traced window -> SIGSEGV at the first traced kernel. nvbit_tool_init may
also run per context, re-initialising the shared channel and starting a second
receiver; and the first nvbit_at_ctx_term stops the receiver for everyone.

Fix: the receiver writes to the context of the kernel currently being traced
(set in enter_kernel_launch before recv_thread_receiving; traced kernels are
serialised by leave_kernel_launch), tool init runs once, and the receiver stops
when the last context terminates. With one context the behaviour is unchanged.

Usage: patch_tracer_mgpu.py SRC_tracer_tool.cu DST_tracer_tool.cu
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count} occurrence(s), found {n}: {old[:70]!r}"
    s = s.replace(old, new)


sub("volatile bool recv_thread_started = false;",
    "volatile bool recv_thread_started = false;\n"
    "// AMCAS multi-GPU: context of the kernel currently being traced.\n"
    "#include <atomic>\n"
    "static std::atomic<CUcontext> g_trace_ctx{nullptr};\n"
    "static bool g_tool_inited = false;\n"
    "static int g_live_ctx = 0;")

sub("  recv_thread_receiving = true;\n}",
    "  g_trace_ctx.store(ctx);\n  recv_thread_receiving = true;\n}")

sub("  CUcontext ctx = (CUcontext)args;\n  char *recv_buffer",
    "  (void)args;\n  char *recv_buffer")

sub("        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {\n",
    "        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {\n"
    "      CUcontext ctx = g_trace_ctx.load();\n")

sub("void nvbit_tool_init(CUcontext ctx) {\n  ctx_current_kernel_name[ctx] = \"\";\n",
    "void nvbit_tool_init(CUcontext ctx) {\n  ctx_current_kernel_name[ctx] = \"\";\n"
    "  if (g_tool_inited)\n    return;\n  g_tool_inited = true;\n")

sub("void nvbit_at_ctx_term(CUcontext ctx) {\n  if (recv_thread_started) {",
    "void nvbit_at_ctx_term(CUcontext ctx) {\n  if (--g_live_ctx > 0)\n    return;\n  if (recv_thread_started) {")

sub("  ctx_kernelid[ctx] = 1;\n}",
    "  ctx_kernelid[ctx] = 1;\n  ++g_live_ctx;\n}")

open(dst, "w").write(s)
print("patched", dst)
