// llama-completion with DynamoRIO's drmemtrace linked in statically, tracing only
// chosen steps (prompt or token) instead of the whole run.
//
// Model loading was 65% of the instructions of a short Qwen 0.5B run, and grows with
// the model, so tracing from process start does not scale. Here DynamoRIO is set up
// at start but only attached (dr_app_start) at the start of step AMCAS_SKIP_STEPS,
// and detached (dr_app_stop_and_cleanup, which writes the raw trace) after
// AMCAS_TRACE_STEPS steps. A step is one llama_process call: the prompt is step 0,
// each generated token one more step.
//
// Environment:
//   AMCAS_SKIP_STEPS   steps to run untraced first (default 0)
//   AMCAS_TRACE_STEPS  steps to trace (default: all remaining)
//   AMCAS_STEPS        file for "B|E step dr_us" lines, as tools/decode_stamp.c
//   DYNAMORIO_OPTIONS  e.g. "-stderr_mask 0xc -client_lib ';;-offline -outdir DIR'"
//
// llama_process is interposed: this executable exports it (-rdynamic), so calls from
// llama's own libraries bind here, and the real one is found with RTLD_NEXT.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "dr_api.h"
#include "llama.h"

int llama_completion(int argc, char **argv);

static long skip_steps, trace_steps = -1, step;
static bool tracing, finished;
static FILE *out;

static long env_long(const char *name, long dflt) {
    const char *v = getenv(name);
    return v && *v ? atol(v) : dflt;
}

static void stamp(char what) {
    if (!out) {
        const char *p = getenv("AMCAS_STEPS");
        if (!p || !(out = fopen(p, "w"))) return;
        setvbuf(out, NULL, _IOLBF, 0);
    }
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long us = (unsigned long long)tv.tv_sec * 1000000ULL + tv.tv_usec;
    fprintf(out, "%c %ld %llu\n", what, step, us + 11644473600ULL * 1000000ULL);
}

extern "C" int32_t llama_process(struct llama_context *ctx, enum llama_process_type type, struct llama_batch_ext *batch) {
    typedef int32_t (*fn_t)(struct llama_context *, enum llama_process_type, struct llama_batch_ext *);
    static fn_t real = (fn_t)dlsym(RTLD_NEXT, "llama_process");
    if (!tracing && !finished && step >= skip_steps) {
        fprintf(stderr, "drburst: attaching at step %ld\n", step);
        dr_app_start();
        tracing = true;
    }
    stamp('B');
    int32_t r = real(ctx, type, batch);
    stamp('E');
    step++;
    if (tracing && trace_steps >= 0 && step >= skip_steps + trace_steps) {
        dr_app_stop_and_cleanup();
        tracing = false;
        finished = true;
        fprintf(stderr, "drburst: detached after step %ld\n", step - 1);
    }
    return r;
}

int main(int argc, char **argv) {
    skip_steps = env_long("AMCAS_SKIP_STEPS", 0);
    trace_steps = env_long("AMCAS_TRACE_STEPS", -1);
    dr_app_setup();
    int r = llama_completion(argc, argv);
    if (tracing)
        dr_app_stop_and_cleanup();
    else if (!finished)
        dr_app_cleanup();
    return r;
}
