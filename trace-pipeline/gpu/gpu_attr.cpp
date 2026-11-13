// LD_PRELOAD tensor dumper for the CUDA llama.cpp build (GPU variant of tools/drburst/drattr_main.cpp).
// Interposes ggml_backend_sched_graph_compute_async; after the real call (the scheduler assigns data
// pointers inside it) writes every distinct tensor reachable from the graph: call index ("step"), name,
// data (device address for CUDA buffers), nbytes, type, op, ne0..ne3, and the buffer's name.
// Env: AMCAS_TENSORS = output TSV. Step 0 = first graph compute (the prompt), then one per token.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <utility>
#include "ggml.h"
#include "ggml-backend.h"
static long step;
static FILE *tsv;
static std::set<std::pair<long, void *>> seen;
static void dump_tensor(const struct ggml_tensor *t) {
    if (!t) return;
    while (t->view_src) t = t->view_src;
    if (!t->data || !seen.insert({step, t->data}).second) return;
    const char *bn = t->buffer ? ggml_backend_buffer_name(t->buffer) : "none";
    fprintf(tsv, "%ld\t%s\t0x%llx\t%zu\t%s\t%s\t%ld\t%ld\t%ld\t%ld\t%s\n", step, t->name, (unsigned long long)(uintptr_t)t->data,
            ggml_nbytes(t), ggml_type_name(t->type), ggml_op_name(t->op), (long)t->ne[0], (long)t->ne[1], (long)t->ne[2],
            (long)t->ne[3], bn);
}
extern "C" enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph *g) {
    typedef enum ggml_status (*fn_t)(ggml_backend_sched_t, struct ggml_cgraph *);
    static fn_t real = (fn_t)dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async");
    enum ggml_status r = real(sched, g);
    if (!tsv) {
        const char *p = getenv("AMCAS_TENSORS");
        if (p) tsv = fopen(p, "w");
        if (tsv) fprintf(tsv, "step\tname\tdata\tnbytes\ttype\top\tne0\tne1\tne2\tne3\tbuffer\n");
    }
    if (tsv) {
        for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
            struct ggml_tensor *n = ggml_graph_node(g, i);
            dump_tensor(n);
            for (int k = 0; k < GGML_MAX_SRC; k++) dump_tensor(n->src[k]);
        }
        fflush(tsv);
    }
    step++;
    return r;
}
