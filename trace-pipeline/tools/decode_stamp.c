// LD_PRELOAD wrapper around llama_process (and llama_decode, for older llama.cpp)
// that logs when each call starts and ends. This llama.cpp's completion tool calls
// llama_process(ctx, type, llama_batch_ext *) and never llama_decode.
//
// Each line of $AMCAS_STEPS is "B|E n dr_us". n is the batch size for llama_decode, or
// the process type (encode or decode) for llama_process, whose batch is opaque.
// dr_us is wall-clock time in microseconds since 1601-01-01 UTC: the clock and epoch
// of DynamoRIO's trace timestamp markers, so a trace can be cut at a step by time.
//
// Build: gcc -O2 -shared -fPIC -I llama.cpp/include -I llama.cpp/ggml/include \
//          decode_stamp.c -o libdecode_stamp.so -ldl

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "llama.h"

static FILE *out;

static void stamp(char what, int n_tokens) {
    if (!out) {
        const char *p = getenv("AMCAS_STEPS");
        if (!p || !(out = fopen(p, "w"))) return;
        setvbuf(out, NULL, _IOLBF, 0);
    }
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long us = (unsigned long long)tv.tv_sec * 1000000ULL + tv.tv_usec;
    fprintf(out, "%c %d %llu\n", what, n_tokens, us + 11644473600ULL * 1000000ULL);
}

int32_t llama_decode(struct llama_context *ctx, struct llama_batch batch) {
    static int32_t (*real)(struct llama_context *, struct llama_batch);
    if (!real) real = (int32_t (*)(struct llama_context *, struct llama_batch))dlsym(RTLD_NEXT, "llama_decode");
    stamp('B', batch.n_tokens);
    int32_t r = real(ctx, batch);
    stamp('E', batch.n_tokens);
    return r;
}

int32_t llama_process(struct llama_context *ctx, enum llama_process_type type, struct llama_batch_ext *batch) {
    static int32_t (*real)(struct llama_context *, enum llama_process_type, struct llama_batch_ext *);
    if (!real) real = (int32_t (*)(struct llama_context *, enum llama_process_type, struct llama_batch_ext *))dlsym(RTLD_NEXT, "llama_process");
    stamp('B', (int)type);
    int32_t r = real(ctx, type, batch);
    stamp('E', (int)type);
    return r;
}
