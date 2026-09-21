static uint64_t file_size_of(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < 0) return UINT64_MAX;
    return (uint64_t)st.st_size;
}

static uint64_t available_memory_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char key[64], unit[16]; unsigned long long kb;
    uint64_t out = 0;
    while (fscanf(f, "%63s %llu %15s", key, &kb, unit) == 3) {
        if (strcmp(key, "MemAvailable:") == 0) { out = (uint64_t)kb * 1024u; break; }
    }
    fclose(f); return out;
}

static unsigned auto_threads(size_t block_size, unsigned requested, uint64_t blocks) {
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    unsigned n = requested ? requested : (unsigned)(cpus > 0 ? cpus : 1);
    if (!requested) {
        uint64_t avail = available_memory_bytes();
        if (avail) {
            uint64_t budget = avail * 3 / 5;
            uint64_t per = (uint64_t)block_size * 5 + 32 * 1024 * 1024;
            unsigned mem_n = per ? (unsigned)(budget / per) : 1;
            if (mem_n < 1) mem_n = 1;
            if (n > mem_n) n = mem_n;
        }
    }
    if (blocks && n > blocks) n = (unsigned)blocks;
    if (n < 1) n = 1;
    return n;
}

static char *default_encoded_path(const char *src) {
    size_t n = strlen(src);
    char *out = malloc(n + 7);
    if (!out) return NULL;
    memcpy(out, src, n); memcpy(out + n, ".base9", 7);
    return out;
}

static char *default_decoded_path(const char *src) {
    size_t n = strlen(src);
    const char *suffix = ".base9";
    if (n >= 6 && strcmp(src + n - 6, suffix) == 0) {
        char *out = malloc(n + 3);
        if (!out) return NULL;
        memcpy(out, src, n - 6); memcpy(out + n - 6, ".decoded", 9);
        return out;
    }
    char *out = malloc(n + 9);
    if (!out) return NULL;
    memcpy(out, src, n); memcpy(out + n, ".decoded", 9);
    return out;
}

static int write_file_header(FILE *out, uint64_t original_size, uint32_t block_size, uint32_t min_region) {
    ByteVec h = {0};
    int rc = bv_append(&h, FILE_MAGIC, 8) || bv_append_u32(&h, BASE9_VERSION) || bv_append_u32(&h, FILE_HEADER_SIZE) ||
             bv_append_u64(&h, original_size) || bv_append_u32(&h, block_size) || bv_append_u32(&h, min_region) ||
             bv_append_u32(&h, 0) || bv_append_u32(&h, 0);
    if (rc || h.len != FILE_HEADER_SIZE) { bv_free(&h); return -1; }
    size_t wrote = fwrite(h.data, 1, h.len, out); bv_free(&h); return wrote == FILE_HEADER_SIZE ? 0 : -1;
}

static int encode_file(const char *src, const char *dst, size_t block_size, size_t min_region, unsigned requested_threads) {
    uint64_t total_size = file_size_of(src);
    if (total_size == UINT64_MAX) { fprintf(stderr, "Cannot stat input: %s\n", src); return 1; }
    if (block_size == 0 || block_size > UINT32_MAX || min_region == 0 || min_region > block_size) {
        fprintf(stderr, "Invalid block/min-region size.\n"); return 1;
    }
    uint64_t blocks = total_size ? (total_size + block_size - 1) / block_size : 0;
    unsigned threads = auto_threads(block_size, requested_threads, blocks);
    FILE *in = fopen(src, "rb"); if (!in) { perror("open input"); return 1; }
    FILE *out = fopen(dst, "wb"); if (!out) { perror("open output"); fclose(in); return 1; }
    if (write_file_header(out, total_size, (uint32_t)block_size, (uint32_t)min_region) != 0) { fprintf(stderr, "header write failed\n"); fclose(in); fclose(out); return 1; }
    printf("BASE9 encode: block=%zu MiB, min-region=%zu KiB, threads=%u\n", block_size >> 20, min_region >> 10, threads);
    double start = now_seconds(); uint64_t done = 0; uint64_t total_regions = 0; uint64_t methods[METHOD_COUNT] = {0};
    while (done < total_size) {
        unsigned batch = threads;
        uint64_t remain_blocks = (total_size - done + block_size - 1) / block_size;
        if (batch > remain_blocks) batch = (unsigned)remain_blocks;
        EncodeJob *jobs = calloc(batch, sizeof(*jobs)); pthread_t *tids = calloc(batch, sizeof(*tids));
        if (!jobs || !tids) { fprintf(stderr, "out of memory\n"); free(jobs); free(tids); fclose(in); fclose(out); return 1; }
        unsigned prepared = 0;
        for (; prepared < batch; ++prepared) {
            size_t want = (size_t)((total_size - done) < block_size ? (total_size - done) : block_size);
            jobs[prepared].input = malloc(want ? want : 1); jobs[prepared].input_len = want; jobs[prepared].min_region = min_region;
            if (!jobs[prepared].input || fread(jobs[prepared].input, 1, want, in) != want) { fprintf(stderr, "input read failed\n"); break; }
            done += want;
        }
        if (prepared != batch) { for (unsigned i=0;i<prepared;++i) free(jobs[i].input); free(jobs); free(tids); fclose(in); fclose(out); return 1; }
        if (batch == 1) encode_thread(&jobs[0]); else for (unsigned i=0;i<batch;++i) pthread_create(&tids[i], NULL, encode_thread, &jobs[i]);
        if (batch > 1) for (unsigned i=0;i<batch;++i) pthread_join(tids[i], NULL);
        for (unsigned i=0;i<batch;++i) {
            if (jobs[i].error || fwrite(jobs[i].encoded.data, 1, jobs[i].encoded.len, out) != jobs[i].encoded.len) {
                fprintf(stderr, "encode/write failed: %s\n", jobs[i].error_msg); for (unsigned j=0;j<batch;++j){free(jobs[j].input);bv_free(&jobs[j].encoded);} free(jobs);free(tids);fclose(in);fclose(out);return 1;
            }
            total_regions += jobs[i].region_count; for (int m=0;m<METHOD_COUNT;++m) methods[m] += jobs[i].method_counts[m];
            free(jobs[i].input); bv_free(&jobs[i].encoded);
        }
        free(jobs); free(tids);
        double pct = total_size ? 100.0 * (double)done / (double)total_size : 100.0;
        printf("\rEncoding: %6.2f%% | regions=%" PRIu64 " | raw=%" PRIu64 " base=%" PRIu64 " rans=%" PRIu64 " delta=%" PRIu64 " ctx=%" PRIu64 " token=%" PRIu64, pct, total_regions, methods[0], methods[1], methods[2], methods[3], methods[4], methods[5]); fflush(stdout);
    }
    printf("\n");
    if (fclose(in) != 0 || fclose(out) != 0) { perror("close"); return 1; }
    uint64_t out_size = file_size_of(dst); double elapsed = now_seconds() - start;
    printf("BASE9 encode complete\nInput:       %" PRIu64 " bytes\nOutput:      %" PRIu64 " bytes\n", total_size, out_size);
    if (total_size) printf("Ratio:       %.4fx (%.2f%% of original)\n", (double)out_size/total_size, 100.0*(double)out_size/total_size);
    printf("Regions:     %" PRIu64 " (raw=%" PRIu64 ", base=%" PRIu64 ", rans=%" PRIu64 ", delta-rans=%" PRIu64 ", context-rans=%" PRIu64 ", token-base=%" PRIu64 ")\nThreads:     %u\nElapsed:     %.3f s\nThroughput:  %.2f MiB/s\nCreated:     %s\n", total_regions, methods[0], methods[1], methods[2], methods[3], methods[4], methods[5], threads, elapsed, elapsed>0 ? (double)total_size/(1024.0*1024.0)/elapsed : 0.0, dst);
    base9_gpu_report_usage();
    return 0;
}

static int read_exact(FILE *f, void *buf, size_t n) { return fread(buf, 1, n, f) == n ? 0 : -1; }

static int decode_file(const char *src, const char *dst, unsigned requested_threads) {
    FILE *in = fopen(src, "rb"); if (!in) { perror("open input"); return 1; }
    uint8_t h[FILE_HEADER_SIZE]; if (read_exact(in, h, sizeof(h)) != 0 || memcmp(h, FILE_MAGIC, 8) != 0 || rd_u32(h+12) != FILE_HEADER_SIZE) { fprintf(stderr, "Not a supported BASE9 file.\n"); fclose(in); return 1; }
    uint32_t version = rd_u32(h+8);
    if (version < BASE9_MIN_VERSION || version > BASE9_VERSION) { fprintf(stderr, "Unsupported BASE9 version: %u\n", version); fclose(in); return 1; }
    uint64_t original_size = rd_u64(h+16); uint32_t block_size = rd_u32(h+24); uint32_t min_region = rd_u32(h+28); (void)min_region;
    uint64_t blocks = original_size ? (original_size + block_size - 1) / block_size : 0;
    unsigned threads = auto_threads(block_size, requested_threads, blocks);
    FILE *out = fopen(dst, "wb"); if (!out) { perror("open output"); fclose(in); return 1; }
    double start = now_seconds(); uint64_t written = 0, block_index = 0;
    while (block_index < blocks) {
        unsigned batch = threads; if (batch > blocks - block_index) batch = (unsigned)(blocks - block_index);
        DecodeJob *jobs = calloc(batch, sizeof(*jobs)); pthread_t *tids = calloc(batch, sizeof(*tids));
        if (!jobs || !tids) { fprintf(stderr, "out of memory\n"); free(jobs); free(tids); fclose(in); fclose(out); return 1; }
        unsigned prepared = 0;
        for (; prepared < batch; ++prepared) {
            uint8_t bh[BLOCK_HEADER_SIZE]; if (read_exact(in, bh, sizeof(bh)) != 0) break;
            jobs[prepared].expected_raw_len = rd_u32(bh); jobs[prepared].expected_regions = rd_u32(bh+4); uint64_t body_len = rd_u64(bh+8); jobs[prepared].expected_crc = rd_u32(bh+16);
            if (body_len > SIZE_MAX) break;
            jobs[prepared].encoded_len = (size_t)body_len; jobs[prepared].encoded = malloc(body_len ? (size_t)body_len : 1);
            if (!jobs[prepared].encoded || read_exact(in, jobs[prepared].encoded, (size_t)body_len) != 0) break;
        }
        if (prepared != batch) { fprintf(stderr,"truncated block\n"); for(unsigned i=0;i<prepared;++i) free(jobs[i].encoded); free(jobs);free(tids);fclose(in);fclose(out);return 1; }
        if (batch == 1) decode_thread(&jobs[0]); else for(unsigned i=0;i<batch;++i) pthread_create(&tids[i],NULL,decode_thread,&jobs[i]);
        if (batch > 1) for(unsigned i=0;i<batch;++i) pthread_join(tids[i],NULL);
        for(unsigned i=0;i<batch;++i){
            if(jobs[i].error || fwrite(jobs[i].decoded.data,1,jobs[i].decoded.len,out)!=jobs[i].decoded.len){fprintf(stderr,"decode/write failed: %s\n",jobs[i].error_msg);for(unsigned j=0;j<batch;++j){free(jobs[j].encoded);bv_free(&jobs[j].decoded);}free(jobs);free(tids);fclose(in);fclose(out);return 1;}
            written += jobs[i].decoded.len; free(jobs[i].encoded); bv_free(&jobs[i].decoded); ++block_index;
            printf("\rDecoding: %6.2f%% | blocks=%" PRIu64 "/%" PRIu64, blocks?100.0*(double)block_index/blocks:100.0,block_index,blocks); fflush(stdout);
        }
        free(jobs);free(tids);
    }
    printf("\n"); fclose(in); fclose(out);
    if (written != original_size) { fprintf(stderr,"decoded size mismatch\n"); return 1; }
    double elapsed=now_seconds()-start; printf("BASE9 decode complete\nOutput:      %" PRIu64 " bytes\nThreads:     %u\nElapsed:     %.3f s\nThroughput:  %.2f MiB/s\nCreated:     %s\n",written,threads,elapsed,elapsed>0?(double)written/(1024.0*1024.0)/elapsed:0.0,dst); return 0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n  %s encode INPUT [-o OUTPUT] [--block-mib N] [--min-region-kib N] [--threads N]\n  %s decode INPUT [-o OUTPUT] [--threads N]\n  %s gpu-info\n\n"
        "BASE9 is a bounded-RAM adaptive compressor. It chooses per region among\n"
        "RAW, adaptive-base packing, token-base, static rANS, delta+rANS, and context rANS,\n"
        "and recursively splits large blocks when local statistics compress better.\n"
        "GPU token discovery uses OpenCL when available and always has CPU fallback.\n", argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    base9_gpu_maybe_reexec_private_intel(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "gpu-info") == 0) {
        base9_gpu_print_info();
        return base9_gpu.available ? 0 : 1;
    }
    if (argc < 3) { usage(argv[0]); return 2; }
    bool enc = strcmp(argv[1], "encode") == 0, dec = strcmp(argv[1], "decode") == 0;
    if (!enc && !dec) { usage(argv[0]); return 2; }
    const char *src = argv[2], *dst_arg = NULL; unsigned block_mib = DEFAULT_BLOCK_MIB, min_kib = DEFAULT_MIN_REGION_KIB, threads = 0;
    for (int i = 3; i < argc; ++i) {
        if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) dst_arg = argv[++i];
        else if (strcmp(argv[i], "--block-mib") == 0 && i + 1 < argc) block_mib = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--min-region-kib") == 0 && i + 1 < argc) min_kib = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = (unsigned)strtoul(argv[++i], NULL, 10);
        else { fprintf(stderr, "Unknown/incomplete option: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    char *owned = NULL; const char *dst = dst_arg;
    if (!dst) { owned = enc ? default_encoded_path(src) : default_decoded_path(src); dst = owned; }
    if (!dst) { fprintf(stderr,"out of memory\n"); return 1; }
    int rc;
    if (enc) {
        size_t block = (size_t)block_mib * 1024u * 1024u, minr = (size_t)min_kib * 1024u;
        rc = encode_file(src, dst, block, minr, threads);
    } else rc = decode_file(src, dst, threads);
    free(owned); return rc;
}
