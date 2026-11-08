// Batched-decode burst driver: B sequences, each with its own prompt, decoded together;
// DynamoRIO is attached only for chosen decode steps (one llama_process call with B
// tokens). Own main() on libllama.so (no llama-completion), so there is no prompt-batch
// step counting: "decode step k" is the k-th batched call after all prompts are in.
//
// Prompt k = tokens of corpus slice k (distinct slices, offset k*AMCAS_STRIDE chars).
// Prompt cache: the first P-1 prompt tokens of each sequence are saved with
// llama_state_seq_save_file (AMCAS_CACHE_DIR) and restored on later runs, so a long
// prompt is evaluated once. The last prompt token is always run live, batched over all
// sequences (decode step 0, untraced), which also yields the logits for token 1.
//
// Env: AMCAS_MODEL AMCAS_B AMCAS_P(512) AMCAS_THREADS AMCAS_CORPUS AMCAS_STRIDE
//   AMCAS_CACHE_DIR AMCAS_NGEN(4 decode steps after step 0) AMCAS_SKIP_STEPS(1)
//   AMCAS_TRACE_STEPS(1) AMCAS_KVU(0: per-sequence KV streams) AMCAS_OUT (token ids)
//   AMCAS_STEPS (B/E timestamps) AMCAS_TENSORS (tensor dump of traced graphs)
//   DYNAMORIO_OPTIONS as drburst.
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <string>
#include <vector>
#include <set>
#include <utility>

#include "dr_api.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

static long step = -1;
static bool want_trace, finished;
static long g_skip, g_ntrace;
static bool tracing;
static FILE *out, *tsv;
static std::set<std::pair<long, void *>> seen;

static long env_long(const char *n, long d) { const char *v = getenv(n); return v && *v ? atol(v) : d; }
static std::string env_str(const char *n, const char *d = "") { const char *v = getenv(n); return v && *v ? v : d; }

static void stamp(char what) {
    if (!out) {
        const char *p = getenv("AMCAS_STEPS");
        if (!p || !(out = fopen(p, "w"))) return;
        setvbuf(out, NULL, _IOLBF, 0);
    }
    struct timeval tv; gettimeofday(&tv, NULL);
    unsigned long long us = (unsigned long long)tv.tv_sec * 1000000ULL + tv.tv_usec;
    fprintf(out, "%c %ld %llu\n", what, step, us + 11644473600ULL * 1000000ULL);
}

static void dump_tensor(const struct ggml_tensor *t) {
    if (!t) return;
    while (t->view_src) t = t->view_src;
    if (!t->data || !seen.insert({step, t->data}).second) return;
    fprintf(tsv, "%ld\t%s\t0x%llx\t%zu\t%s\t%s\t%ld\t%ld\t%ld\t%ld\n", step, t->name,
            (unsigned long long)(uintptr_t)t->data, ggml_nbytes(t), ggml_type_name(t->type),
            ggml_op_name(t->op), (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3]);
}

// Same interposition as drattr_main.cpp: dump after the real call (data pointers are
// assigned inside it), only while tracing.
extern "C" enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph *g) {
    typedef enum ggml_status (*fn_t)(ggml_backend_sched_t, struct ggml_cgraph *);
    static fn_t real = (fn_t)dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async");
    // Attach just before the graph compute, not in llama_process: with B >= 13
    // sequences DynamoRIO crashed (SIGSEGV in std::unordered_map<bitset<256>,..>::
    // operator[] after a rehash, inside llama_batch_allocr::init) when started earlier.
    if (want_trace && !tracing && !finished) { fprintf(stderr, "drbatch: attaching at step %ld\n", step); dr_app_start(); tracing = true; }
    enum ggml_status r = real(sched, g);
    if (tracing && !tsv) {
        const char *p = getenv("AMCAS_TENSORS");
        if (p) tsv = fopen(p, "w");
        if (tsv) fprintf(tsv, "step\tname\tdata\tnbytes\ttype\top\tne0\tne1\tne2\tne3\n");
    }
    if (tracing && tsv) {
        for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
            struct ggml_tensor *n = ggml_graph_node(g, i);
            dump_tensor(n);
            for (int k = 0; k < GGML_MAX_SRC; k++) dump_tensor(n->src[k]);
        }
        fflush(tsv);
    }
    if (tracing && step + 1 >= g_skip + g_ntrace) { dr_app_stop_and_cleanup(); tracing = false; finished = true; fprintf(stderr, "drbatch: detached after step %ld\n", step); }
    return r;
}

static void on_sig(int sig, siginfo_t *si, void *) {
    fprintf(stderr, "drbatch: signal %d at %p (si_code %d)\n", sig, si->si_addr, si->si_code);
    void *bt[30]; int n = backtrace(bt, 30); backtrace_symbols_fd(bt, n, 2); _exit(99);
}
// Same shape as drburst's wrapper: attach and detach happen inside llama_process.
extern "C" int32_t llama_process(struct llama_context *ctx, enum llama_process_type type, struct llama_batch_ext *batch) {
    typedef int32_t (*fn_t)(struct llama_context *, enum llama_process_type, struct llama_batch_ext *);
    static fn_t real = (fn_t)dlsym(RTLD_NEXT, "llama_process");
    if (step >= 0) stamp('B');
    int32_t r = real(ctx, type, batch);
    if (step >= 0) stamp('E');
    return r;
}
static void die(const char *m) { fprintf(stderr, "drbatch: %s\n", m); exit(1); }

int main() {
    const int B = env_long("AMCAS_B", 1), P = env_long("AMCAS_P", 512), T = env_long("AMCAS_THREADS", 4);
    const int NGEN = env_long("AMCAS_NGEN", 4);
    const long skip = g_skip = env_long("AMCAS_SKIP_STEPS", 1), ntrace = g_ntrace = env_long("AMCAS_TRACE_STEPS", 1);
    std::string model = env_str("AMCAS_MODEL"), corpus_path = env_str("AMCAS_CORPUS"), cdir = env_str("AMCAS_CACHE_DIR");
    long stride = env_long("AMCAS_STRIDE", 30000);
    if (model.empty() || corpus_path.empty()) die("AMCAS_MODEL and AMCAS_CORPUS required");
    if (getenv("AMCAS_SIGDBG")) {
        struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = on_sig; sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, 0); sigaction(SIGILL, &sa, 0); sigaction(SIGBUS, &sa, 0); sigaction(SIGFPE, &sa, 0);
    }
    dr_app_setup();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model *m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) die("model load failed");
    const llama_vocab *vocab = llama_model_get_vocab(m);
    const int cs = ((P + NGEN + 8 + 255) / 256) * 256;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = B * cs; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_seq_max = B;
    cp.n_threads = cp.n_threads_batch = T; cp.no_perf = true; cp.kv_unified = env_long("AMCAS_KVU", 0) != 0;
    llama_context *ctx = llama_init_from_model(m, cp);
    if (!ctx) die("context failed");

    FILE *cf = fopen(corpus_path.c_str(), "rb"); if (!cf) die("corpus");
    std::string corpus; { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, cf)) > 0) corpus.append(buf, n); fclose(cf); }
    std::vector<std::vector<llama_token>> prompt(B);
    for (int b = 0; b < B; b++) {
        size_t off = (size_t)b * stride % (corpus.size() > 8u * P ? corpus.size() - 8u * P : 1);
        std::string s = corpus.substr(off, (size_t)P * 8);
        prompt[b].resize(P * 8);
        int n = llama_tokenize(vocab, s.data(), s.size(), prompt[b].data(), prompt[b].size(), true, false);
        if (n < P) die("prompt too short");
        prompt[b].resize(P);
    }
    llama_batch_ext *batch = llama_batch_ext_init(ctx);
    auto run = [&](bool logits_last_each) {
        int32_t r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        (void)logits_last_each; return r;
    };
    // 1. prompt part (P-1 tokens), cached per sequence
    size_t slash = model.find_last_of('/'); std::string mb = model.substr(slash == std::string::npos ? 0 : slash + 1);
    for (int b = 0; b < B; b++) {
        std::string f = cdir.empty() ? "" : cdir + "/" + mb + "_P" + std::to_string(P) + "_s" + std::to_string(stride) + "_b" + std::to_string(b) + ".seq";
        bool ok = false;
        if (!f.empty()) {
            std::vector<llama_token> tk(P); size_t nt = 0;
            size_t r = llama_state_seq_load_file(ctx, f.c_str(), b, tk.data(), tk.size(), &nt);
            if (r > 0 && nt == (size_t)P - 1 && memcmp(tk.data(), prompt[b].data(), nt * sizeof(llama_token)) == 0) {
                ok = true; fprintf(stderr, "drbatch: seq %d restored from %s\n", b, f.c_str());
            }
        }
        if (!ok) {
            for (int i = 0; i < P - 1; i += 2048) {
                int n = std::min(2048, P - 1 - i);
                llama_batch_ext_clear(batch);
                for (int j = 0; j < n; j++) {
                    int idx = llama_batch_ext_add_token(batch, b, prompt[b][i + j]);
                    llama_pos p = i + j; llama_batch_ext_set_pos(batch, idx, &p);
                }
                if (run(false) != 0) die("prompt eval failed");
            }
            llama_synchronize(ctx);
            if (!f.empty()) { llama_state_seq_save_file(ctx, f.c_str(), b, prompt[b].data(), P - 1); fprintf(stderr, "drbatch: seq %d evaluated and saved\n", b); }
            else fprintf(stderr, "drbatch: seq %d evaluated\n", b);
        }
    }
    // 2. decode steps: step 0 = last prompt token of every sequence; then greedy tokens
    const int nv = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> cur(B); std::vector<int> pos(B, P - 1);
    for (int b = 0; b < B; b++) cur[b] = prompt[b][P - 1];
    FILE *tok = fopen(env_str("AMCAS_OUT", "/dev/null").c_str(), "w");
    for (step = 0; step <= NGEN; step++) {
        llama_batch_ext_clear(batch);
        for (int b = 0; b < B; b++) {
            int idx = llama_batch_ext_add_token(batch, b, cur[b]);
            llama_pos p = pos[b]; llama_batch_ext_set_pos(batch, idx, &p);
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
        want_trace = step >= skip && step < skip + ntrace;
        if (run(true) != 0) die("decode failed");
        want_trace = false;
        for (int b = 0; b < B; b++) {
            const float *l = llama_get_logits_ith(ctx, b);
            int best = 0; for (int i = 1; i < nv; i++) if (l[i] > l[best]) best = i;
            cur[b] = best; pos[b]++;
            fprintf(tok, "%d%c", best, b == B - 1 ? '\n' : ' ');
        }
        fflush(tok);
    }
    if (tracing) dr_app_stop_and_cleanup(); else if (!finished) dr_app_cleanup();
    return 0;
}
