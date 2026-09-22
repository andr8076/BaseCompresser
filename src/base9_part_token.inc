/*
 * Token-base codec.
 *
 * Repeated adjacent symbols are merged for a few bounded BPE-style rounds.
 * A token expands to 2..8 original bytes. The final stream contains literal
 * bytes plus dictionary tokens and is packed as digits in one adaptive base.
 *
 * This keeps the central BASE idea explicit:
 *
 *     repeated byte sequences -> fewer symbols -> base-N digit stream
 *
 * Discovery is bounded (three rounds, at most eight merges per round), and
 * token mode is kept only if dictionary + radix payload beats the best method
 * already found for the region.
 */

#define TOKEN_MAX_LEN 8u
#define TOKEN_ROUNDS 3u
#define TOKEN_MERGES_PER_ROUND 8u
#define TOKEN_MAX_CREATED (TOKEN_ROUNDS * TOKEN_MERGES_PER_ROUND)
#define TOKEN_SYMBOL_CAP (256u + TOKEN_MAX_CREATED)
#define TOKEN_META_FIXED 40u
#define TOKEN_MIN_GAIN_BITS 128.0L

typedef struct {
    uint8_t len;
    uint8_t bytes[TOKEN_MAX_LEN];
} TokenDef;

typedef struct {
    uint16_t a;
    uint16_t b;
    uint32_t count;
    long double gain_bits;
    TokenDef def;
} TokenMerge;

typedef struct {
    uint8_t len;
    uint8_t bytes[TOKEN_MAX_LEN];
} TokenDecodeSymbol;

static unsigned token_symbol_expansion(
    uint16_t sym,
    const TokenDef defs[TOKEN_MAX_CREATED],
    unsigned created,
    uint8_t out[TOKEN_MAX_LEN]
) {
    if (sym < 256u) {
        out[0] = (uint8_t)sym;
        return 1;
    }
    unsigned idx = (unsigned)sym - 256u;
    if (idx >= created || defs[idx].len == 0 || defs[idx].len > TOKEN_MAX_LEN)
        return 0;
    memcpy(out, defs[idx].bytes, defs[idx].len);
    return defs[idx].len;
}

static bool token_def_exists(
    const TokenDef defs[TOKEN_MAX_CREATED],
    unsigned created,
    const TokenDef *candidate
) {
    for (unsigned i = 0; i < created; ++i) {
        if (defs[i].len == candidate->len &&
            memcmp(defs[i].bytes, candidate->bytes, candidate->len) == 0)
            return true;
    }
    return false;
}

static bool token_pair_to_def(
    uint16_t a,
    uint16_t b,
    const TokenDef defs[TOKEN_MAX_CREATED],
    unsigned created,
    TokenDef *out
) {
    uint8_t left[TOKEN_MAX_LEN], right[TOKEN_MAX_LEN];
    unsigned llen = token_symbol_expansion(a, defs, created, left);
    unsigned rlen = token_symbol_expansion(b, defs, created, right);
    if (!llen || !rlen || llen + rlen > TOKEN_MAX_LEN)
        return false;
    out->len = (uint8_t)(llen + rlen);
    memcpy(out->bytes, left, llen);
    memcpy(out->bytes + llen, right, rlen);
    return true;
}

static unsigned token_choose_merges(
    const uint32_t *pair_counts,
    size_t symbol_count,
    unsigned distinct,
    const TokenDef defs[TOKEN_MAX_CREATED],
    unsigned created,
    TokenMerge out[TOKEN_MERGES_PER_ROUND]
) {
    if (symbol_count < 2 || distinct < 2 || created >= TOKEN_MAX_CREATED)
        return 0;

    unsigned chosen = 0;
    long double projected_symbols = (long double)symbol_count;
    unsigned projected_base = distinct;
    bool used_pair[TOKEN_MERGES_PER_ROUND] = {0};
    (void)used_pair;

    while (chosen < TOKEN_MERGES_PER_ROUND &&
           created + chosen < TOKEN_MAX_CREATED) {
        bool found = false;
        TokenMerge best = {0};
        unsigned max_sym = 256u + created;
        long double log_before = log2l((long double)projected_base);
        long double log_after = log2l((long double)(projected_base + 1u));
        long double base_change =
            projected_symbols * (log_before - log_after);

        for (unsigned a = 0; a < max_sym; ++a) {
            for (unsigned b = 0; b < max_sym; ++b) {
                uint32_t count = pair_counts[(size_t)a * TOKEN_SYMBOL_CAP + b];
                if (count < 2)
                    continue;

                bool already = false;
                for (unsigned j = 0; j < chosen; ++j) {
                    if (out[j].a == a && out[j].b == b) {
                        already = true;
                        break;
                    }
                }
                if (already)
                    continue;

                TokenDef def = {0};
                if (!token_pair_to_def((uint16_t)a, (uint16_t)b, defs, created, &def))
                    continue;
                if (token_def_exists(defs, created, &def))
                    continue;

                uint32_t effective = (a == b) ? (count / 2u) : count;
                if (effective < 2)
                    continue;

                long double dict_bits = (long double)(def.len + 1u) * 8.0L;
                long double gain =
                    base_change + (long double)effective * log_after - dict_bits;

                if (gain <= TOKEN_MIN_GAIN_BITS)
                    continue;
                if (!found || gain > best.gain_bits) {
                    found = true;
                    best.a = (uint16_t)a;
                    best.b = (uint16_t)b;
                    best.count = effective;
                    best.gain_bits = gain;
                    best.def = def;
                }
            }
        }

        if (!found)
            break;

        out[chosen++] = best;
        projected_symbols =
            projected_symbols > best.count ?
            projected_symbols - best.count : 1.0L;
        ++projected_base;
    }

    return chosen;
}

typedef struct {
    uint16_t digit_map[TOKEN_SYMBOL_CAP];
    unsigned base;
    unsigned gd;
    unsigned gb;
    size_t packed_size;
} TokenRadixLayout;

static int token_prepare_layout(
    const uint16_t *symbols,
    size_t symbol_count,
    const TokenDef defs[TOKEN_MAX_CREATED],
    unsigned created,
    ByteVec *meta,
    TokenRadixLayout *layout
) {
    if (!symbol_count || !layout)
        return -1;

    bool used[TOKEN_SYMBOL_CAP] = {0};
    for (size_t i = 0; i < symbol_count; ++i) {
        if (symbols[i] >= TOKEN_SYMBOL_CAP)
            return -1;
        used[symbols[i]] = true;
    }

    uint8_t literal_bitmap[32] = {0};
    for (unsigned i = 0; i < TOKEN_SYMBOL_CAP; ++i)
        layout->digit_map[i] = UINT16_MAX;

    unsigned base = 0;
    for (unsigned b = 0; b < 256; ++b) {
        if (!used[b])
            continue;
        layout->digit_map[b] = (uint16_t)base++;
        literal_bitmap[b >> 3] |= (uint8_t)(1u << (b & 7));
    }

    unsigned used_tokens = 0;
    for (unsigned t = 0; t < created; ++t) {
        unsigned sym = 256u + t;
        if (!used[sym])
            continue;
        layout->digit_map[sym] = (uint16_t)base++;
        ++used_tokens;
    }

    if (!base || used_tokens > 255u || symbol_count > UINT32_MAX)
        return -1;

    unsigned gd = 0, gb = 0;
    size_t packed_size =
        base_payload_size(symbol_count, base, &gd, &gb);
    if (gd > 255u || gb > 255u)
        return -1;

    if (meta) {
        if (bv_append(meta, literal_bitmap, sizeof(literal_bitmap)) != 0 ||
            bv_append_u8(meta, (uint8_t)used_tokens) != 0 ||
            bv_append_u8(meta, (uint8_t)gd) != 0 ||
            bv_append_u8(meta, (uint8_t)gb) != 0 ||
            bv_append_u8(meta, 0) != 0 ||
            bv_append_u32(meta, (uint32_t)symbol_count) != 0)
            return -1;

        for (unsigned t = 0; t < created; ++t) {
            if (!used[256u + t])
                continue;
            if (defs[t].len < 2 || defs[t].len > TOKEN_MAX_LEN ||
                bv_append_u8(meta, defs[t].len) != 0 ||
                bv_append(meta, defs[t].bytes, defs[t].len) != 0)
                return -1;
        }

        if (meta->len > UINT16_MAX)
            return -1;
    }

    layout->base = base;
    layout->gd = gd;
    layout->gb = gb;
    layout->packed_size = packed_size;
    return 0;
}

static int token_pack_stream(
    const uint16_t *symbols,
    size_t symbol_count,
    const TokenDef defs[TOKEN_MAX_CREATED],
    unsigned created,
    ByteVec *meta,
    ByteVec *payload
) {
    TokenRadixLayout layout;
    if (token_prepare_layout(
            symbols, symbol_count, defs, created, meta, &layout) != 0)
        return -1;

    if (layout.base == 1u)
        return 0;

    size_t packed_start = payload->len;
    if (bv_reserve(payload, layout.packed_size) != 0)
        return -1;
    payload->len += layout.packed_size;

    if (base9_gpu_pack_radix_u16(
            symbols, symbol_count, layout.digit_map, TOKEN_SYMBOL_CAP,
            layout.base, layout.gd, layout.gb,
            payload->data + packed_start, layout.packed_size) == 0)
        return 0;

    size_t pos = 0, out_pos = packed_start;
    while (pos < symbol_count) {
        unsigned take =
            (unsigned)((symbol_count - pos) < layout.gd ?
                       (symbol_count - pos) : layout.gd);
        uint64_t value = 0;
        for (unsigned j = 0; j < take; ++j) {
            uint16_t digit = layout.digit_map[symbols[pos + j]];
            if (digit == UINT16_MAX || digit >= layout.base)
                return -1;
            value = value * layout.base + digit;
        }
        unsigned bytes = take == layout.gd ?
                         layout.gb : group_width_bytes(layout.base, take);
        if (bytes > 8 || out_pos + bytes > payload->len)
            return -1;
        for (unsigned j = 0; j < bytes; ++j)
            payload->data[out_pos + j] = (uint8_t)(value >> (8u * j));
        out_pos += bytes;
        pos += take;
    }

    return out_pos == payload->len ? 0 : -1;
}

static unsigned token_count_distinct_symbols(
    const uint16_t *symbols,
    size_t symbol_count,
    unsigned max_sym
) {
    bool used[TOKEN_SYMBOL_CAP] = {0};
    unsigned distinct = 0;
    for (size_t i = 0; i < symbol_count; ++i) {
        unsigned sym = symbols[i];
        if (sym >= max_sym || sym >= TOKEN_SYMBOL_CAP)
            continue;
        if (!used[sym]) {
            used[sym] = true;
            ++distinct;
        }
    }
    return distinct;
}

static void token_count_pairs(
    const uint16_t *symbols,
    size_t symbol_count,
    uint32_t *pair_counts
) {
    if (symbol_count >= 2 &&
        base9_gpu_pair_count_u16(
            symbols, symbol_count, pair_counts, TOKEN_SYMBOL_CAP) == 0)
        return;

    memset(pair_counts, 0,
           (size_t)TOKEN_SYMBOL_CAP * TOKEN_SYMBOL_CAP * sizeof(*pair_counts));
    if (symbol_count < 2)
        return;
    /*
     * token_base_try_encode() rejects regions larger than UINT32_MAX.
     * Later token streams only get shorter, so a single pair can occur at
     * most UINT32_MAX-1 times and these uint32_t counters cannot overflow.
     *
     * Four-way unrolling exposes independent histogram updates to the CPU,
     * which benchmarks faster here than AVX2 gather/scatter emulation.
     */
    size_t i = 0;
    for (; i + 4 < symbol_count; i += 4) {
        unsigned a0 = symbols[i],     b0 = symbols[i + 1];
        unsigned a1 = symbols[i + 1], b1 = symbols[i + 2];
        unsigned a2 = symbols[i + 2], b2 = symbols[i + 3];
        unsigned a3 = symbols[i + 3], b3 = symbols[i + 4];

        if (a0 < TOKEN_SYMBOL_CAP && b0 < TOKEN_SYMBOL_CAP)
            ++pair_counts[(size_t)a0 * TOKEN_SYMBOL_CAP + b0];
        if (a1 < TOKEN_SYMBOL_CAP && b1 < TOKEN_SYMBOL_CAP)
            ++pair_counts[(size_t)a1 * TOKEN_SYMBOL_CAP + b1];
        if (a2 < TOKEN_SYMBOL_CAP && b2 < TOKEN_SYMBOL_CAP)
            ++pair_counts[(size_t)a2 * TOKEN_SYMBOL_CAP + b2];
        if (a3 < TOKEN_SYMBOL_CAP && b3 < TOKEN_SYMBOL_CAP)
            ++pair_counts[(size_t)a3 * TOKEN_SYMBOL_CAP + b3];
    }
    for (; i + 1 < symbol_count; ++i) {
        unsigned a = symbols[i], b = symbols[i + 1];
        if (a < TOKEN_SYMBOL_CAP && b < TOKEN_SYMBOL_CAP)
            ++pair_counts[(size_t)a * TOKEN_SYMBOL_CAP + b];
    }
}

static int token_base_try_encode(
    const uint8_t *data,
    size_t len,
    size_t target_total,
    ByteVec *best_meta,
    ByteVec *best_payload
) {
    if (len < 32 || len > UINT32_MAX)
        return 1;

    size_t pair_slots = (size_t)TOKEN_SYMBOL_CAP * TOKEN_SYMBOL_CAP;
    uint32_t *pair_counts = calloc(pair_slots, sizeof(*pair_counts));
    if (!pair_counts)
        return -1;

    bool raw_used[256] = {0};
    unsigned raw_distinct = 0;
    for (size_t i = 0; i < len; ++i) {
        if (!raw_used[data[i]]) {
            raw_used[data[i]] = true;
            ++raw_distinct;
        }
    }

    if (base9_gpu_pair_count_u8(
            data, len, pair_counts, TOKEN_SYMBOL_CAP) != 0) {
        /*
         * len <= UINT32_MAX, so the same no-overflow proof applies to the
         * initial byte-pair histogram.
         */
        size_t i = 0;
        for (; i + 4 < len; i += 4) {
            ++pair_counts[(size_t)data[i]     * TOKEN_SYMBOL_CAP + data[i + 1]];
            ++pair_counts[(size_t)data[i + 1] * TOKEN_SYMBOL_CAP + data[i + 2]];
            ++pair_counts[(size_t)data[i + 2] * TOKEN_SYMBOL_CAP + data[i + 3]];
            ++pair_counts[(size_t)data[i + 3] * TOKEN_SYMBOL_CAP + data[i + 4]];
        }
        for (; i + 1 < len; ++i)
            ++pair_counts[(size_t)data[i] * TOKEN_SYMBOL_CAP + data[i + 1]];
    }

    if (raw_distinct < 2) {
        free(pair_counts);
        return 1;
    }

    TokenDef defs[TOKEN_MAX_CREATED] = {{0}};
    TokenMerge choices[TOKEN_MERGES_PER_ROUND] = {{0}};
    TokenMerge history[TOKEN_ROUNDS][TOKEN_MERGES_PER_ROUND] = {{{0}}};
    unsigned history_counts[TOKEN_ROUNDS] = {0};

    unsigned first_choices = token_choose_merges(
        pair_counts, len, raw_distinct, defs, 0, choices);
    if (!first_choices) {
        free(pair_counts);
        return 1;
    }

    if (len > SIZE_MAX / sizeof(uint16_t)) {
        free(pair_counts);
        return -1;
    }
    uint16_t *symbols = malloc(len * sizeof(*symbols));
    uint16_t *pair_map = malloc(pair_slots * sizeof(*pair_map));
    if (!symbols || !pair_map) {
        free(symbols);
        free(pair_map);
        free(pair_counts);
        return -1;
    }
    for (size_t i = 0; i < len; ++i)
        symbols[i] = data[i];

    size_t symbol_count = len;
    unsigned created = 0;
    size_t best_total = target_total;
    int best_round = -1;
    int last_round = -1;
    unsigned best_created = 0;

    for (unsigned round = 0; round < TOKEN_ROUNDS; ++round) {
        unsigned distinct;
        unsigned choice_count;

        if (round == 0) {
            distinct = raw_distinct;
            choice_count = first_choices;
        } else {
            token_count_pairs(symbols, symbol_count, pair_counts);
            distinct = token_count_distinct_symbols(
                symbols, symbol_count, 256u + created);
            memset(choices, 0, sizeof(choices));
            choice_count = token_choose_merges(
                pair_counts, symbol_count, distinct, defs, created, choices);
        }

        if (!choice_count)
            break;

        if (choice_count > TOKEN_MERGES_PER_ROUND)
            choice_count = TOKEN_MERGES_PER_ROUND;
        memcpy(history[round], choices,
               choice_count * sizeof(history[round][0]));
        history_counts[round] = choice_count;

        memset(pair_map, 0xff, pair_slots * sizeof(*pair_map));
        for (unsigned c = 0; c < choice_count; ++c) {
            if (created >= TOKEN_MAX_CREATED)
                break;
            defs[created] = choices[c].def;
            uint16_t token_sym = (uint16_t)(256u + created);
            pair_map[(size_t)choices[c].a * TOKEN_SYMBOL_CAP + choices[c].b] =
                token_sym;
            ++created;
        }

        size_t write = 0;
        for (size_t read = 0; read < symbol_count;) {
            if (read + 1 < symbol_count) {
                unsigned a = symbols[read], b = symbols[read + 1];
                uint16_t merged =
                    pair_map[(size_t)a * TOKEN_SYMBOL_CAP + b];
                if (merged != UINT16_MAX) {
                    symbols[write++] = merged;
                    read += 2;
                    continue;
                }
            }
            symbols[write++] = symbols[read++];
        }
        symbol_count = write;
        last_round = (int)round;

        ByteVec estimate_meta = {0};
        TokenRadixLayout estimate_layout;
        if (token_prepare_layout(
                symbols, symbol_count, defs, created,
                &estimate_meta, &estimate_layout) != 0) {
            bv_free(&estimate_meta);
            free(symbols);
            free(pair_map);
            free(pair_counts);
            return -1;
        }

        size_t total =
            REGION_HEADER_SIZE + estimate_meta.len +
            estimate_layout.packed_size;
        bv_free(&estimate_meta);

        if (total < best_total) {
            best_total = total;
            best_round = (int)round;
            best_created = created;
        }
    }

    if (best_round < 0) {
        free(symbols);
        free(pair_map);
        free(pair_counts);
        return 1;
    }

    /*
     * Usually every profitable merge round improves the radix representation,
     * so the live symbol stream is already the winning stream. If an earlier
     * round won, replay only the saved merge decisions; this is still much
     * cheaper than radix-packing every candidate round.
     */
    if (best_round != last_round) {
        for (size_t i = 0; i < len; ++i)
            symbols[i] = data[i];
        symbol_count = len;
        unsigned replay_created = 0;

        for (int round = 0; round <= best_round; ++round) {
            memset(pair_map, 0xff, pair_slots * sizeof(*pair_map));
            for (unsigned c = 0; c < history_counts[round]; ++c) {
                TokenMerge *merge = &history[round][c];
                if (replay_created >= TOKEN_MAX_CREATED)
                    break;
                pair_map[(size_t)merge->a * TOKEN_SYMBOL_CAP + merge->b] =
                    (uint16_t)(256u + replay_created);
                ++replay_created;
            }

            size_t write = 0;
            for (size_t read = 0; read < symbol_count;) {
                if (read + 1 < symbol_count) {
                    unsigned a = symbols[read], b = symbols[read + 1];
                    uint16_t merged =
                        pair_map[(size_t)a * TOKEN_SYMBOL_CAP + b];
                    if (merged != UINT16_MAX) {
                        symbols[write++] = merged;
                        read += 2;
                        continue;
                    }
                }
                symbols[write++] = symbols[read++];
            }
            symbol_count = write;
        }

        if (replay_created != best_created) {
            free(symbols);
            free(pair_map);
            free(pair_counts);
            return -1;
        }
    }

    ByteVec final_meta = {0}, final_payload = {0};
    if (token_pack_stream(
            symbols, symbol_count, defs, best_created,
            &final_meta, &final_payload) != 0) {
        bv_free(&final_meta);
        bv_free(&final_payload);
        free(symbols);
        free(pair_map);
        free(pair_counts);
        return -1;
    }

    size_t actual_total =
        REGION_HEADER_SIZE + final_meta.len + final_payload.len;
    if (actual_total != best_total) {
        bv_free(&final_meta);
        bv_free(&final_payload);
        free(symbols);
        free(pair_map);
        free(pair_counts);
        return -1;
    }

    bv_free(best_meta);
    bv_free(best_payload);
    *best_meta = final_meta;
    *best_payload = final_payload;

    free(symbols);
    free(pair_map);
    free(pair_counts);
    return 0;
}

static int token_base_decode_region(
    const uint8_t *meta,
    size_t meta_len,
    const uint8_t *payload,
    size_t payload_len,
    size_t out_len,
    ByteVec *out
) {
    if (meta_len < TOKEN_META_FIXED)
        return -1;

    const uint8_t *literal_bitmap = meta;
    unsigned token_count = meta[32];
    unsigned stored_gd = meta[33];
    unsigned stored_gb = meta[34];
    uint32_t symbol_count = rd_u32(meta + 36);
    size_t mpos = TOKEN_META_FIXED;

    TokenDecodeSymbol alphabet[TOKEN_SYMBOL_CAP];
    unsigned base = 0;
    for (unsigned b = 0; b < 256; ++b) {
        if (!(literal_bitmap[b >> 3] & (1u << (b & 7))))
            continue;
        alphabet[base].len = 1;
        alphabet[base].bytes[0] = (uint8_t)b;
        ++base;
    }

    for (unsigned t = 0; t < token_count; ++t) {
        if (mpos >= meta_len || base >= TOKEN_SYMBOL_CAP)
            return -1;
        unsigned n = meta[mpos++];
        if (n < 2 || n > TOKEN_MAX_LEN || meta_len - mpos < n)
            return -1;
        alphabet[base].len = (uint8_t)n;
        memcpy(alphabet[base].bytes, meta + mpos, n);
        mpos += n;
        ++base;
    }

    if (mpos != meta_len || !base || !symbol_count)
        return -1;

    unsigned gd = 0, gb = 0;
    size_t expected_payload =
        base_payload_size(symbol_count, base, &gd, &gb);
    if (expected_payload != payload_len ||
        gd != stored_gd || gb != stored_gb)
        return -1;

    size_t start = out->len;
    if (bv_reserve(out, out_len) != 0)
        return -1;

    if (base == 1u) {
        size_t need = (size_t)symbol_count * alphabet[0].len;
        if (need != out_len || payload_len != 0)
            return -1;
        for (uint32_t i = 0; i < symbol_count; ++i) {
            memcpy(out->data + out->len,
                   alphabet[0].bytes, alphabet[0].len);
            out->len += alphabet[0].len;
        }
        return 0;
    }

    size_t spos = 0, ppos = 0;
    uint16_t digits[64];
    while (spos < symbol_count) {
        unsigned take =
            (unsigned)((symbol_count - spos) < gd ?
                       (symbol_count - spos) : gd);
        unsigned bytes = take == gd ? gb : group_width_bytes(base, take);
        if (bytes > 8 || ppos + bytes > payload_len)
            goto fail;

        uint64_t value = 0;
        for (unsigned j = 0; j < bytes; ++j)
            value |= (uint64_t)payload[ppos + j] << (8u * j);

        for (unsigned j = take; j-- > 0;) {
            digits[j] = (uint16_t)(value % base);
            value /= base;
        }
        if (value != 0)
            goto fail;

        for (unsigned j = 0; j < take; ++j) {
            TokenDecodeSymbol *sym = &alphabet[digits[j]];
            if (out->len - start > out_len ||
                sym->len > out_len - (out->len - start))
                goto fail;
            memcpy(out->data + out->len, sym->bytes, sym->len);
            out->len += sym->len;
        }

        spos += take;
        ppos += bytes;
    }

    if (ppos != payload_len || out->len - start != out_len)
        goto fail;
    return 0;

fail:
    out->len = start;
    return -1;
}
