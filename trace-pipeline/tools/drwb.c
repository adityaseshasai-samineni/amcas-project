/* drwb: offline write-back cache simulation over a DynamoRIO drmemtrace (drraw2trace -compress gzip),
 * to recover the DRAM WRITEBACKS that drcachesim does not model (it has no dirty bits).
 *
 * Usage: drwb OUT_STREAM QUANTUM TRACE_FILE...  (one .trace.gz per thread; stats JSON on stdout)
 *
 * Model (matches drcachesim's default hierarchy used by 17_drattr.sh: -cores 4 -LL_size 22M -LL_assoc 11):
 *  per thread a 32 KiB 8-way LRU write-back L1D, one shared 22 MiB 11-way LRU write-back, write-allocate LL,
 *  non-inclusive. Loads, stores and software prefetches (as loads) allocate. Instruction fetches are ignored.
 *  A dirty L1D victim is written into the LL (allocating if absent); a dirty LL victim is a DRAM write.
 *  Threads are interleaved round-robin, QUANTUM instructions each (an approximation of parallel execution).
 * OUT_STREAM: little-endian uint64 per DRAM request, in order: line-aligned virtual address | 1 for a write.
 * Residual dirty lines at the end of the trace (still in L1/LL, written later) are reported, not emitted.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <zlib.h>

#pragma pack(push, 1)
typedef struct { uint16_t type; uint16_t size; uint64_t addr; } ent_t;
#pragma pack(pop)
enum { T_READ = 0, T_WRITE = 1, T_PF_LAST = 8, T_INSTR = 10, T_BUNDLE = 17, T_MARKER = 28 };

typedef struct { uint64_t *tag; uint64_t *lru; uint8_t *dirty; unsigned sets, ways; } cache_t;
static void cinit(cache_t *c, unsigned sets, unsigned ways) {
    c->sets = sets; c->ways = ways;
    c->tag = calloc((size_t)sets * ways, 8); c->lru = calloc((size_t)sets * ways, 8); c->dirty = calloc((size_t)sets * ways, 1);
    if (!c->tag || !c->lru || !c->dirty) { perror("calloc"); exit(1); }
}
static uint64_t tick = 1;
static FILE *out; static uint64_t n_rd, n_wr, n_ll_acc, n_ll_hit, n_l1_acc, n_l1_miss, n_l1_wb, n_wb_hit, n_wb_alloc, n_stores, n_loads, n_pf, n_instr;

static void emit(uint64_t line, int w) { uint64_t v = (line << 6) | (uint64_t)w; fwrite(&v, 8, 1, out); if (w) n_wr++; else n_rd++; }

/* find line (stored as line+1) in cache; returns way index or -1 */
static int find(cache_t *c, unsigned set, uint64_t key) {
    uint64_t *t = c->tag + (size_t)set * c->ways;
    for (unsigned w = 0; w < c->ways; w++) if (t[w] == key) return (int)w;
    return -1;
}
static unsigned victim(cache_t *c, unsigned set) {
    uint64_t *t = c->tag + (size_t)set * c->ways, *l = c->lru + (size_t)set * c->ways;
    unsigned v = 0;
    for (unsigned w = 0; w < c->ways; w++) { if (!t[w]) return w; if (l[w] < l[v]) v = w; }
    return v;
}
static cache_t LL;
/* insert line into LL (miss path or writeback allocate); handles DRAM write of a dirty victim */
static void ll_fill(uint64_t line, int dirty) {
    unsigned set = line % LL.sets; unsigned v = victim(&LL, set); size_t i = (size_t)set * LL.ways + v;
    if (LL.tag[i] && LL.dirty[i]) emit(LL.tag[i] - 1, 1);
    LL.tag[i] = line + 1; LL.lru[i] = tick++; LL.dirty[i] = dirty;
}
static void ll_access(uint64_t line) {
    n_ll_acc++;
    unsigned set = line % LL.sets; int w = find(&LL, set, line + 1);
    if (w >= 0) { LL.lru[(size_t)set * LL.ways + w] = tick++; n_ll_hit++; return; }
    emit(line, 0); ll_fill(line, 0);
}
static void ll_writeback(uint64_t line) {
    unsigned set = line % LL.sets; int w = find(&LL, set, line + 1);
    if (w >= 0) { size_t i = (size_t)set * LL.ways + w; LL.dirty[i] = 1; LL.lru[i] = tick++; n_wb_hit++; return; }
    n_wb_alloc++; ll_fill(line, 1);
}
static void l1_access(cache_t *c, uint64_t line, int is_store) {
    n_l1_acc++;
    unsigned set = line % c->sets; int w = find(c, set, line + 1);
    if (w >= 0) { size_t i = (size_t)set * c->ways + w; c->lru[i] = tick++; if (is_store) c->dirty[i] = 1; return; }
    n_l1_miss++;
    ll_access(line);
    unsigned v = victim(c, set); size_t i = (size_t)set * c->ways + v;
    if (c->tag[i] && c->dirty[i]) { n_l1_wb++; ll_writeback(c->tag[i] - 1); }
    c->tag[i] = line + 1; c->lru[i] = tick++; c->dirty[i] = is_store;
}

typedef struct { gzFile f; ent_t *buf; int n, pos; int done; cache_t l1; uint64_t ts; } th_t;
static int fill(th_t *t) {
    if (t->done) return 0;
    int r = gzfread(t->buf, sizeof(ent_t), 65536, t->f);
    if (r <= 0) { t->done = 1; return 0; }
    t->n = r; t->pos = 0; return 1;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: drwb OUT QUANTUM FILES...\n"); return 2; }
    out = fopen(argv[1], "wb"); uint64_t Q = strtoull(argv[2], 0, 10);
    int nt = argc - 3; th_t *T = calloc(nt, sizeof(th_t));
    cinit(&LL, 32768, 11);
    for (int i = 0; i < nt; i++) {
        T[i].f = gzopen(argv[3 + i], "rb"); if (!T[i].f) { fprintf(stderr, "cannot open %s\n", argv[3 + i]); return 1; }
        T[i].buf = malloc(65536 * sizeof(ent_t)); cinit(&T[i].l1, 64, 8);
    }
    int alive = nt;
    for (int i = 0; i < nt; i++) if (!fill(&T[i])) alive--;
    while (alive > 0) {
        for (int i = 0; i < nt; i++) {
            th_t *t = &T[i]; uint64_t q = 0;
            while (!t->done && q < Q) {
                if (t->pos >= t->n && !fill(t)) { alive--; break; }
                ent_t *e = &t->buf[t->pos++];
                if (e->type == T_INSTR || (e->type >= 11 && e->type <= 16) || (e->type >= 29 && e->type <= 31) || e->type == 48 || e->type == 49) { q++; n_instr++; continue; }
                if (e->type > T_PF_LAST) continue;
                int st = e->type == T_WRITE;
                if (st) n_stores++; else if (e->type == T_READ) n_loads++; else n_pf++;
                uint64_t a = e->addr, b = a + (e->size ? e->size - 1 : 0);
                for (uint64_t l = a >> 6; l <= (b >> 6); l++) l1_access(&t->l1, l, st);
            }
        }
    }
    uint64_t dirty_ll = 0, dirty_l1 = 0;
    for (size_t i = 0; i < (size_t)LL.sets * LL.ways; i++) if (LL.tag[i] && LL.dirty[i]) dirty_ll++;
    for (int k = 0; k < nt; k++) for (size_t i = 0; i < 64 * 8; i++) if (T[k].l1.tag[i] && T[k].l1.dirty[i]) dirty_l1++;
    fclose(out);
    printf("{\"threads\": %d, \"quantum\": %llu, \"instr\": %llu, \"loads\": %llu, \"stores\": %llu, \"prefetches\": %llu,\n"
           " \"l1_accesses\": %llu, \"l1_misses\": %llu, \"l1_dirty_evictions\": %llu, \"ll_accesses\": %llu, \"ll_hits\": %llu,\n"
           " \"dram_reads\": %llu, \"dram_writes\": %llu, \"wb_hit_in_ll\": %llu, \"wb_alloc_in_ll\": %llu,\n"
           " \"residual_dirty_ll\": %llu, \"residual_dirty_l1\": %llu}\n",
           nt, (unsigned long long)Q, (unsigned long long)n_instr, (unsigned long long)n_loads, (unsigned long long)n_stores,
           (unsigned long long)n_pf, (unsigned long long)n_l1_acc, (unsigned long long)n_l1_miss, (unsigned long long)n_l1_wb,
           (unsigned long long)n_ll_acc, (unsigned long long)n_ll_hit, (unsigned long long)n_rd, (unsigned long long)n_wr,
           (unsigned long long)n_wb_hit, (unsigned long long)n_wb_alloc, (unsigned long long)dirty_ll, (unsigned long long)dirty_l1);
    return 0;
}
