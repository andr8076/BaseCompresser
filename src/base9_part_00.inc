#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define BASE9_VERSION 3u
#define BASE9_MIN_VERSION 1u
#define DEFAULT_BLOCK_MIB 64u
#define DEFAULT_MIN_REGION_KIB 256u
#define RANS_SCALE_BITS 12u
#define RANS_TOTFREQ (1u << RANS_SCALE_BITS)
#define RANS_L (1u << 23)
#define CONTEXT_SCALE_BITS 8u
#define CONTEXT_TOTFREQ (1u << CONTEXT_SCALE_BITS)
#define REGION_HEADER_SIZE 16u
#define BLOCK_HEADER_SIZE 24u
#define FILE_HEADER_SIZE 40u
#define SPLIT_MIN_GAIN 64.0

static const uint8_t FILE_MAGIC[8] = {'B','A','S','E','9','\r','\n',0x1a};

enum Method {
    METHOD_RAW = 0,
    METHOD_BASE = 1,
    METHOD_RANS = 2,
    METHOD_DELTA_RANS = 3,
    METHOD_CONTEXT_RANS = 4,
    METHOD_TOKEN_BASE = 5,
    METHOD_COUNT = 6,
};

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} ByteVec;

typedef struct {
    uint64_t raw[256];
    uint64_t delta[256];
    size_t off;
    size_t len;
} LeafStat;

typedef struct PlanNode {
    size_t off;
    size_t len;
    double cost;
    enum Method method;
    struct PlanNode *left;
    struct PlanNode *right;
} PlanNode;

static void free_plan(PlanNode *node);

typedef struct {
    uint8_t *input;
    size_t input_len;
    size_t min_region;
    ByteVec encoded;
    uint32_t region_count;
    uint32_t method_counts[METHOD_COUNT];
    int error;
    char error_msg[160];
} EncodeJob;

typedef struct {
    uint8_t *encoded;
    size_t encoded_len;
    uint32_t expected_raw_len;
    uint32_t expected_regions;
    uint32_t expected_crc;
    ByteVec decoded;
    int error;
    char error_msg[160];
} DecodeJob;

static uint32_t crc_table[256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;

static void crc_init(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j)
            c = (c & 1u) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
        crc_table[i] = c;
    }
}

static uint32_t crc32_bytes(const uint8_t *data, size_t len) {
    pthread_once(&crc_once, crc_init);
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < len; ++i)
        c = crc_table[(c ^ data[i]) & 0xffu] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int bv_reserve(ByteVec *v, size_t extra) {
    if (extra > SIZE_MAX - v->len) return -1;
    size_t need = v->len + extra;
    if (need <= v->cap) return 0;
    size_t cap = v->cap ? v->cap : 4096;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) { cap = need; break; }
        cap *= 2;
    }
    uint8_t *p = realloc(v->data, cap);
    if (!p) return -1;
    v->data = p;
    v->cap = cap;
    return 0;
}

static int bv_append(ByteVec *v, const void *src, size_t n) {
    if (n == 0) return 0;
    if (!src) return -1;
    if (bv_reserve(v, n) != 0) return -1;
    memcpy(v->data + v->len, src, n);
    v->len += n;
    return 0;
}

static int bv_append_u8(ByteVec *v, uint8_t x) { return bv_append(v, &x, 1); }

static int bv_append_u16(ByteVec *v, uint16_t x) {
    uint8_t b[2] = {(uint8_t)x, (uint8_t)(x >> 8)};
    return bv_append(v, b, 2);
}

static int bv_append_u32(ByteVec *v, uint32_t x) {
    uint8_t b[4] = {(uint8_t)x, (uint8_t)(x >> 8), (uint8_t)(x >> 16), (uint8_t)(x >> 24)};
    return bv_append(v, b, 4);
}

static int bv_append_u64(ByteVec *v, uint64_t x) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t)(x >> (8 * i));
    return bv_append(v, b, 8);
}

static uint16_t rd_u16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static uint32_t rd_u32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t rd_u64(const uint8_t *p) {
    uint64_t x = 0;
    for (int i = 7; i >= 0; --i) x = (x << 8) | p[i];
    return x;
}

static void bv_free(ByteVec *v) {
    free(v->data);
    v->data = NULL; v->len = v->cap = 0;
}

static size_t count_used(const uint64_t counts[256]) {
    size_t used = 0;
    for (int i = 0; i < 256; ++i) used += counts[i] != 0;
    return used;
}

static void normalize_freq(const uint64_t counts[256], uint64_t total, uint16_t freq[256], uint16_t cum[256]) {
    size_t used = count_used(counts);
    memset(freq, 0, 256 * sizeof(freq[0]));
    if (!used || !total) {
        memset(cum, 0, 256 * sizeof(cum[0]));
        return;
    }
    uint32_t remaining = RANS_TOTFREQ - (uint32_t)used;
    uint64_t remainders[256] = {0};
    uint32_t assigned = 0;
    for (int i = 0; i < 256; ++i) {
        if (!counts[i]) continue;
        __uint128_t prod = (__uint128_t)counts[i] * remaining;
        uint32_t extra = (uint32_t)(prod / total);
        remainders[i] = (uint64_t)(prod % total);
        freq[i] = (uint16_t)(1u + extra);
        assigned += 1u + extra;
    }
    while (assigned < RANS_TOTFREQ) {
        int best = -1;
        uint64_t best_rem = 0;
        for (int i = 0; i < 256; ++i) {
            if (!counts[i]) continue;
            if (best < 0 || remainders[i] > best_rem) {
                best = i; best_rem = remainders[i];
            }
        }
        if (best < 0) break;
        ++freq[best];
        remainders[best] = 0;
        ++assigned;
    }
    uint32_t c = 0;
    for (int i = 0; i < 256; ++i) {
        cum[i] = (uint16_t)c;
        c += freq[i];
    }
