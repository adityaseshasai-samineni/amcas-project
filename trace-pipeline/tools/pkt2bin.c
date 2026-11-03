// Decode a gem5 packet trace (dram.trc.gz) to fixed-size binary records.
//
// Parses the protobuf wire format by hand, so it needs only zlib and handles
// hundreds of millions of records, which the Python decoder in gem5/util cannot.
//
// Usage: pkt2bin dram.trc.gz out.bin > header.txt
// Record: see `struct rec`, little endian, 40 bytes.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

struct __attribute__((packed)) rec {
    uint64_t tick, addr, vaddr, pc;
    uint32_t size;
    uint16_t cmd, req;
};

static int rd_varint(gzFile f, uint64_t *v)
{
    uint64_t r = 0;
    for (int s = 0; s < 64; s += 7) {
        int c = gzgetc(f);
        if (c < 0)
            return 0;
        r |= (uint64_t)(c & 0x7f) << s;
        if (!(c & 0x80)) {
            *v = r;
            return 1;
        }
    }
    return 0;
}

static const uint8_t *buf_varint(const uint8_t *p, const uint8_t *e, uint64_t *v)
{
    uint64_t r = 0;
    for (int s = 0; s < 64 && p < e; s += 7) {
        uint8_t c = *p++;
        r |= (uint64_t)(c & 0x7f) << s;
        if (!(c & 0x80)) {
            *v = r;
            return p;
        }
    }
    return NULL;
}

// Print the PacketHeader: tick frequency and the requestor id to name table.
static void parse_header(const uint8_t *p, const uint8_t *e)
{
    while (p < e) {
        uint64_t key, v;
        if (!(p = buf_varint(p, e, &key)))
            return;
        if ((key & 7) == 0) {
            if (!(p = buf_varint(p, e, &v)))
                return;
            if (key >> 3 == 3)
                printf("tick_freq\t%llu\n", (unsigned long long)v);
        } else if ((key & 7) == 2) {
            if (!(p = buf_varint(p, e, &v)))
                return;
            if (key >> 3 == 4) {
                const uint8_t *q = p, *qe = p + v;
                uint64_t id = 0, k2, l2;
                while (q < qe) {
                    if (!(q = buf_varint(q, qe, &k2)))
                        break;
                    if ((k2 & 7) == 0) {
                        if (!(q = buf_varint(q, qe, &id)))
                            break;
                    } else {
                        if (!(q = buf_varint(q, qe, &l2)))
                            break;
                        printf("requestor\t%llu\t%.*s\n", (unsigned long long)id, (int)l2, q);
                        q += l2;
                    }
                }
            }
            p += v;
        } else {
            return;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s dram.trc.gz out.bin\n", argv[0]);
        return 2;
    }
    gzFile in = gzopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (!in || !out) {
        perror("open");
        return 1;
    }
    gzbuffer(in, 1 << 20);
    static char obuf[1 << 22];
    setvbuf(out, obuf, _IOFBF, sizeof(obuf));

    uint32_t magic;
    if (gzread(in, &magic, 4) != 4 || magic != 0x356d6567) {
        fprintf(stderr, "bad magic\n");
        return 1;
    }

    uint8_t msg[4096];
    uint64_t len, n = 0;
    int first = 1;
    while (rd_varint(in, &len)) {
        if (len > sizeof(msg)) {
            fprintf(stderr, "message too long: %llu\n", (unsigned long long)len);
            return 1;
        }
        if (gzread(in, msg, len) != (int)len) {
            fprintf(stderr, "truncated trace after %llu records\n", (unsigned long long)n);
            break;
        }
        if (first) {
            parse_header(msg, msg + len);
            first = 0;
            continue;
        }
        struct rec r;
        memset(&r, 0, sizeof(r));
        const uint8_t *p = msg, *e = msg + len;
        while (p < e) {
            uint64_t key, v;
            if (!(p = buf_varint(p, e, &key)) || (key & 7) != 0 || !(p = buf_varint(p, e, &v))) {
                fprintf(stderr, "malformed record %llu\n", (unsigned long long)n);
                return 1;
            }
            switch (key >> 3) {
            case 1: r.tick = v; break;
            case 2: r.cmd = v; break;
            case 3: r.addr = v; break;
            case 4: r.size = v; break;
            case 6: r.req = v; break;
            case 7: r.pc = v; break;
            case 8: r.vaddr = v; break;
            }
        }
        fwrite(&r, sizeof(r), 1, out);
        n++;
    }
    printf("records\t%llu\n", (unsigned long long)n);
    fclose(out);
    gzclose(in);
    return 0;
}
