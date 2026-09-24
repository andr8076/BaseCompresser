}

static size_t rans_estimate_payload(const uint64_t counts[256], uint64_t total) {
    if (!total) return 0;
    uint16_t freq[256], cum[256];
    normalize_freq(counts, total, freq, cum);
    long double bits = 32.0L;
    for (int i = 0; i < 256; ++i) {
        if (!counts[i]) continue;
        bits += (long double)counts[i] *
                ((long double)RANS_SCALE_BITS - base9_log2_frequency[freq[i]]);
    }
    return (size_t)ceill(bits / 8.0L);
}

static unsigned ceil_log2_u128(__uint128_t x) {
    if (x <= 1) return 0;
    --x;
    unsigned bits = 0;
    while (x) { ++bits; x >>= 1; }
    return bits;
}

static unsigned group_width_bytes(unsigned base, unsigned digits) {
    if (digits == 0 || base <= 1) return 0;
    __uint128_t p = 1;
    for (unsigned i = 0; i < digits; ++i) p *= base;
    unsigned bits = ceil_log2_u128(p);
    return (bits + 7u) / 8u;
}

static void choose_base_group(unsigned base, unsigned *best_digits, unsigned *best_bytes) {
    *best_digits = 1; *best_bytes = 1;
    if (base <= 1) { *best_digits = 0; *best_bytes = 0; return; }
    const __uint128_t limit = ((__uint128_t)1) << 64;
    __uint128_t p = 1;
    unsigned bd = 0, bw = 0;
    for (unsigned d = 1; d <= 64; ++d) {
        if (p > limit / base) break;
        p *= base;
        unsigned bits = ceil_log2_u128(p);
        unsigned bytes = (bits + 7u) / 8u;
        if (!bytes) bytes = 1;
        if (!bd || (uint64_t)bytes * bd < (uint64_t)bw * d || ((uint64_t)bytes * bd == (uint64_t)bw * d && d > bd)) {
            bd = d; bw = bytes;
        }
    }
    *best_digits = bd; *best_bytes = bw;
}

static size_t base_payload_size(uint64_t n, unsigned base, unsigned *group_digits, unsigned *group_bytes) {
    if (base <= 1 || n == 0) {
        if (group_digits) *group_digits = 0;
        if (group_bytes) *group_bytes = 0;
        return 0;
    }
    unsigned gd, gb;
    choose_base_group(base, &gd, &gb);
    uint64_t full = n / gd;
    unsigned rem = (unsigned)(n % gd);
    size_t size = (size_t)full * gb + group_width_bytes(base, rem);
    if (group_digits) *group_digits = gd;
    if (group_bytes) *group_bytes = gb;
    return size;
}

static double method_estimate(const uint64_t raw[256], const uint64_t delta[256], uint64_t len, enum Method *best_method) {
    double best = (double)REGION_HEADER_SIZE + (double)len;
    enum Method method = METHOD_RAW;
    size_t used = count_used(raw);
    if (used > 0 && used < 256) {
        size_t payload = base_payload_size(len, (unsigned)used, NULL, NULL);
        double cost = REGION_HEADER_SIZE + 34.0 + (double)payload;
        if (cost < best) { best = cost; method = METHOD_BASE; }
    }
    if (len > 0) {
        size_t p = rans_estimate_payload(raw, len);
        double cost = REGION_HEADER_SIZE + 512.0 + (double)p;
        if (cost < best) { best = cost; method = METHOD_RANS; }
        p = rans_estimate_payload(delta, len);
        cost = REGION_HEADER_SIZE + 512.0 + (double)p;
        if (cost < best) { best = cost; method = METHOD_DELTA_RANS; }
    }
    if (best_method) *best_method = method;
    return best;
}

static void sum_leaf_counts(const LeafStat *leaves, size_t first, size_t last, uint64_t raw[256], uint64_t delta[256]) {
    memset(raw, 0, 256 * sizeof(raw[0]));
    memset(delta, 0, 256 * sizeof(delta[0]));
    for (size_t l = first; l < last; ++l) {
        for (int i = 0; i < 256; ++i) {
            raw[i] += leaves[l].raw[i];
            delta[i] += leaves[l].delta[i];
        }
    }
}

static PlanNode *plan_range(const uint8_t *block, const LeafStat *leaves, size_t first, size_t last) {
    PlanNode *node = calloc(1, sizeof(*node));
    if (!node) return NULL;
    node->off = leaves[first].off;
    size_t end = leaves[last - 1].off + leaves[last - 1].len;
    node->len = end - node->off;
    uint64_t raw[256], delta[256];
    sum_leaf_counts(leaves, first, last, raw, delta);
    if (node->off > 0 && node->len > 0) {
        uint8_t old_delta = (uint8_t)(block[node->off] - block[node->off - 1]);
        uint8_t region_first = block[node->off];
        if (delta[old_delta]) --delta[old_delta];
        ++delta[region_first];
    }
    node->cost = method_estimate(raw, delta, node->len, &node->method);
    if (last - first > 1) {
        size_t mid = first + (last - first) / 2;
        PlanNode *left = plan_range(block, leaves, first, mid);
        PlanNode *right = plan_range(block, leaves, mid, last);
        if (!left || !right) {
            free(left); free(right); free(node); return NULL;
        }
        double split_cost = left->cost + right->cost;
        if (split_cost + SPLIT_MIN_GAIN < node->cost) {
            node->left = left; node->right = right; node->cost = split_cost;
        } else {
            free_plan(left);
            free_plan(right);
        }
    }
    return node;
}

static void free_plan(PlanNode *node) {
    if (!node) return;
    free_plan(node->left); free_plan(node->right); free(node);
}

static size_t count_plan_regions(const PlanNode *node) {
    if (!node) return 0;
    if (!node->left) return 1;
    return count_plan_regions(node->left) + count_plan_regions(node->right);
}

static int encode_base(const uint8_t *data, size_t len, ByteVec *meta, ByteVec *payload) {
    uint8_t present[256] = {0};
    for (size_t i = 0; i < len; ++i) present[data[i]] = 1;
    unsigned base = 0;
    int16_t map[256];
    for (int i = 0; i < 256; ++i) map[i] = -1;
    uint8_t bitmap[32] = {0};
    for (int i = 0; i < 256; ++i) {
        if (present[i]) {
            map[i] = (int16_t)base++;
            bitmap[i >> 3] |= (uint8_t)(1u << (i & 7));
        }
    }
    unsigned gd = 0, gb = 0;
    base_payload_size(len, base, &gd, &gb);
    if (bv_append(meta, bitmap, sizeof(bitmap)) != 0 || bv_append_u8(meta, (uint8_t)gd) != 0 || bv_append_u8(meta, (uint8_t)gb) != 0) return -1;
    if (base <= 1 || len == 0) return 0;
    size_t pos = 0;
    while (pos < len) {
        unsigned take = (unsigned)((len - pos) < gd ? (len - pos) : gd);
        uint64_t value = 0;
        for (unsigned j = 0; j < take; ++j) value = value * base + (uint64_t)map[data[pos + j]];
        unsigned bytes = take == gd ? gb : group_width_bytes(base, take);
        uint8_t tmp[8];
        for (unsigned j = 0; j < bytes; ++j) tmp[j] = (uint8_t)(value >> (8u * j));
        if (bv_append(payload, tmp, bytes) != 0) return -1;
        pos += take;
    }
    return 0;
}

static int rans_encode(const uint8_t *data, size_t len, bool delta_mode, ByteVec *meta, ByteVec *payload) {
    uint64_t counts[256] = {0};
    if (!delta_mode) {
        for (size_t i = 0; i < len; ++i) ++counts[data[i]];
    } else if (len) {
        ++counts[data[0]];
        for (size_t i = 1; i < len; ++i) ++counts[(uint8_t)(data[i] - data[i - 1])];
    }
    uint16_t freq[256], cum[256];
    normalize_freq(counts, len, freq, cum);
    for (int i = 0; i < 256; ++i) if (bv_append_u16(meta, freq[i]) != 0) return -1;
    if (!len) return 0;
    if (len > (SIZE_MAX - 32) / 2) return -1;
    size_t cap = len * 2 + 32;
    uint8_t *buf = malloc(cap);
    if (!buf) return -1;
    uint8_t *ptr = buf + cap;
    uint32_t state = RANS_L;
    for (size_t ii = len; ii-- > 0;) {
        uint8_t sym = delta_mode ? (ii == 0 ? data[0] : (uint8_t)(data[ii] - data[ii - 1])) : data[ii];
        uint32_t f = freq[sym], c = cum[sym];
        if (!f) { free(buf); return -1; }
        uint32_t x_max = ((RANS_L >> RANS_SCALE_BITS) << 8) * f;
        while (state >= x_max) {
            if (ptr == buf) { free(buf); return -1; }
