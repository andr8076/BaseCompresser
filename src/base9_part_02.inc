            *--ptr = (uint8_t)state;
            state >>= 8;
        }
        state = (state / f << RANS_SCALE_BITS) + (state % f) + c;
    }
    if ((size_t)(ptr - buf) < 4) { free(buf); return -1; }
    ptr -= 4;
    ptr[0] = (uint8_t)state; ptr[1] = (uint8_t)(state >> 8); ptr[2] = (uint8_t)(state >> 16); ptr[3] = (uint8_t)(state >> 24);
    size_t out_len = (size_t)((buf + cap) - ptr);
    int rc = bv_append(payload, ptr, out_len);
    free(buf);
    return rc;
}

static int append_region(ByteVec *body, const uint8_t *data, size_t len, enum Method planned, uint32_t method_counts[4]) {
    ByteVec meta = {0}, payload = {0};
    enum Method method = planned;
    int rc = 0;
    if (method == METHOD_BASE) rc = encode_base(data, len, &meta, &payload);
    else if (method == METHOD_RANS) rc = rans_encode(data, len, false, &meta, &payload);
    else if (method == METHOD_DELTA_RANS) rc = rans_encode(data, len, true, &meta, &payload);
    if (rc != 0) { bv_free(&meta); bv_free(&payload); return -1; }

    size_t encoded_total = REGION_HEADER_SIZE + meta.len + payload.len;
    size_t raw_total = REGION_HEADER_SIZE + len;
    if (method != METHOD_RAW && encoded_total >= raw_total) {
        bv_free(&meta); bv_free(&payload);
        method = METHOD_RAW;
    }
    if (method == METHOD_RAW) {
        if (bv_append_u8(body, (uint8_t)METHOD_RAW) != 0 || bv_append_u8(body, 0) != 0 || bv_append_u16(body, 0) != 0 ||
            bv_append_u32(body, (uint32_t)len) != 0 || bv_append_u32(body, (uint32_t)len) != 0 || bv_append_u32(body, crc32_bytes(data, len)) != 0 ||
            bv_append(body, data, len) != 0) return -1;
        ++method_counts[METHOD_RAW];
        return 0;
    }
    if (meta.len > UINT16_MAX || payload.len > UINT32_MAX || len > UINT32_MAX) { bv_free(&meta); bv_free(&payload); return -1; }
    if (bv_append_u8(body, (uint8_t)method) != 0 || bv_append_u8(body, 0) != 0 || bv_append_u16(body, (uint16_t)meta.len) != 0 ||
        bv_append_u32(body, (uint32_t)len) != 0 || bv_append_u32(body, (uint32_t)payload.len) != 0 || bv_append_u32(body, crc32_bytes(data, len)) != 0 ||
        bv_append(body, meta.data, meta.len) != 0 || bv_append(body, payload.data, payload.len) != 0) {
        bv_free(&meta); bv_free(&payload); return -1;
    }
    ++method_counts[method];
    bv_free(&meta); bv_free(&payload);
    return 0;
}

static int encode_plan_regions(ByteVec *body, const uint8_t *block, const PlanNode *node, uint32_t method_counts[4]) {
    if (node->left) {
        if (encode_plan_regions(body, block, node->left, method_counts) != 0) return -1;
        return encode_plan_regions(body, block, node->right, method_counts);
    }
    return append_region(body, block + node->off, node->len, node->method, method_counts);
}

static int build_leaf_stats(const uint8_t *block, size_t len, size_t min_region, LeafStat **out_leaves, size_t *out_count) {
    size_t count = (len + min_region - 1) / min_region;
    if (!count) count = 1;
    LeafStat *leaves = calloc(count, sizeof(*leaves));
    if (!leaves) return -1;
    for (size_t l = 0; l < count; ++l) {
        size_t off = l * min_region;
        size_t n = len > off ? len - off : 0;
        if (n > min_region) n = min_region;
        leaves[l].off = off; leaves[l].len = n;
    }
    for (size_t i = 0; i < len; ++i) {
        size_t l = i / min_region;
        ++leaves[l].raw[block[i]];
        uint8_t d = i == 0 ? block[i] : (uint8_t)(block[i] - block[i - 1]);
        ++leaves[l].delta[d];
    }
    *out_leaves = leaves; *out_count = count;
    return 0;
}

static int encode_block(EncodeJob *job) {
    LeafStat *leaves = NULL; size_t leaf_count = 0;
    if (build_leaf_stats(job->input, job->input_len, job->min_region, &leaves, &leaf_count) != 0) return -1;
    PlanNode *plan = plan_range(job->input, leaves, 0, leaf_count);
    free(leaves);
    if (!plan) return -1;
    uint32_t regions = (uint32_t)count_plan_regions(plan);
    ByteVec body = {0};
    if (encode_plan_regions(&body, job->input, plan, job->method_counts) != 0) { free_plan(plan); bv_free(&body); return -1; }
    free_plan(plan);
    if (bv_append_u32(&job->encoded, (uint32_t)job->input_len) != 0 || bv_append_u32(&job->encoded, regions) != 0 ||
        bv_append_u64(&job->encoded, (uint64_t)body.len) != 0 || bv_append_u32(&job->encoded, crc32_bytes(job->input, job->input_len)) != 0 ||
        bv_append_u32(&job->encoded, 0) != 0 || bv_append(&job->encoded, body.data, body.len) != 0) { bv_free(&body); return -1; }
    bv_free(&body); job->region_count = regions; return 0;
}

static void *encode_thread(void *arg) {
    EncodeJob *job = arg;
    if (encode_block(job) != 0) {
        job->error = 1;
        snprintf(job->error_msg, sizeof(job->error_msg), "block encode failed (out of memory or internal error)");
    }
    return NULL;
}

static int decode_base_region(const uint8_t *meta, size_t meta_len, const uint8_t *payload, size_t payload_len, size_t out_len, ByteVec *out) {
    if (meta_len != 34) return -1;
    const uint8_t *bitmap = meta;
    unsigned gd = meta[32], gb = meta[33];
    uint8_t alphabet[256]; unsigned base = 0;
    for (int i = 0; i < 256; ++i) if (bitmap[i >> 3] & (1u << (i & 7))) alphabet[base++] = (uint8_t)i;
    if (!out_len) return 0;
    if (!base) return -1;
    if (base == 1) {
        if (payload_len != 0 || bv_reserve(out, out_len) != 0) return -1;
        memset(out->data + out->len, alphabet[0], out_len); out->len += out_len; return 0;
    }
    if (!gd || !gb) return -1;
    size_t start = out->len;
    if (bv_reserve(out, out_len) != 0) return -1;
    out->len += out_len;
    size_t pos = 0, ppos = 0;
    while (pos < out_len) {
        unsigned take = (unsigned)((out_len - pos) < gd ? (out_len - pos) : gd);
        unsigned bytes = take == gd ? gb : group_width_bytes(base, take);
        if (ppos + bytes > payload_len || bytes > 8) return -1;
        uint64_t v = 0;
        for (unsigned j = 0; j < bytes; ++j) v |= (uint64_t)payload[ppos + j] << (8u * j);
        for (unsigned j = take; j-- > 0;) {
            unsigned digit = (unsigned)(v % base); v /= base;
            out->data[start + pos + j] = alphabet[digit];
        }
        if (v != 0) return -1;
        pos += take; ppos += bytes;
    }
    return ppos == payload_len ? 0 : -1;
}

static int rans_decode_region(const uint8_t *meta, size_t meta_len, const uint8_t *payload, size_t payload_len, size_t out_len, bool delta_mode, ByteVec *out) {
    if (meta_len != 512 || payload_len < 4) return -1;
    uint16_t freq[256], cum[256];
    uint32_t total = 0;
    for (int i = 0; i < 256; ++i) { freq[i] = rd_u16(meta + 2*i); cum[i] = (uint16_t)total; total += freq[i]; }
    if (total != RANS_TOTFREQ) return -1;
    uint8_t lookup[RANS_TOTFREQ];
    for (int s = 0; s < 256; ++s) for (uint32_t j = 0; j < freq[s]; ++j) lookup[cum[s] + j] = (uint8_t)s;
    uint32_t state = rd_u32(payload);
    const uint8_t *ptr = payload + 4, *end = payload + payload_len;
    size_t start = out->len;
    if (bv_reserve(out, out_len) != 0) return -1;
    out->len += out_len;
    uint8_t prev = 0;
    for (size_t i = 0; i < out_len; ++i) {
        uint32_t slot = state & (RANS_TOTFREQ - 1u);
        uint8_t sym = lookup[slot];
        uint32_t f = freq[sym], c = cum[sym];
        state = f * (state >> RANS_SCALE_BITS) + slot - c;
        while (state < RANS_L) {
            if (ptr >= end) return -1;
            state = (state << 8) | *ptr++;
        }
        uint8_t value = delta_mode ? (i == 0 ? sym : (uint8_t)(prev + sym)) : sym;
        out->data[start + i] = value; prev = value;
    }
    return ptr == end ? 0 : -1;
}

static int decode_block(DecodeJob *job) {
    const uint8_t *p = job->encoded, *end = job->encoded + job->encoded_len;
    uint32_t regions = 0;
    for (; regions < job->expected_regions; ++regions) {
        if ((size_t)(end - p) < REGION_HEADER_SIZE) return -1;
        enum Method method = (enum Method)p[0];
        uint16_t meta_len = rd_u16(p + 2);
        uint32_t raw_len = rd_u32(p + 4);
        uint32_t payload_len = rd_u32(p + 8);
        uint32_t expected_crc = rd_u32(p + 12);
        p += REGION_HEADER_SIZE;
        if ((size_t)(end - p) < (size_t)meta_len + payload_len) return -1;
        const uint8_t *meta = p; p += meta_len;
        const uint8_t *payload = p; p += payload_len;
        size_t before = job->decoded.len;
        int rc = -1;
        if (method == METHOD_RAW) {
            if (meta_len == 0 && payload_len == raw_len) rc = bv_append(&job->decoded, payload, raw_len);
        } else if (method == METHOD_BASE) rc = decode_base_region(meta, meta_len, payload, payload_len, raw_len, &job->decoded);
        else if (method == METHOD_RANS) rc = rans_decode_region(meta, meta_len, payload, payload_len, raw_len, false, &job->decoded);
        else if (method == METHOD_DELTA_RANS) rc = rans_decode_region(meta, meta_len, payload, payload_len, raw_len, true, &job->decoded);
        if (rc != 0 || job->decoded.len - before != raw_len) return -1;
        if (crc32_bytes(job->decoded.data + before, raw_len) != expected_crc) return -1;
    }
    if (p != end || job->decoded.len != job->expected_raw_len || crc32_bytes(job->decoded.data, job->decoded.len) != job->expected_crc) return -1;
    return 0;
}

static void *decode_thread(void *arg) {
    DecodeJob *job = arg;
    if (decode_block(job) != 0) {
        job->error = 1;
        snprintf(job->error_msg, sizeof(job->error_msg), "block decode/checksum failed");
    }
    return NULL;
}

