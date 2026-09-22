#include <fcntl.h>
#include <sys/wait.h>

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

enum EncodeSlotState {
    ENCODE_SLOT_EMPTY = 0,
    ENCODE_SLOT_QUEUED = 1,
    ENCODE_SLOT_RUNNING = 2,
    ENCODE_SLOT_DONE = 3,
};

typedef struct {
    EncodeJob *jobs;
    unsigned *states;
    unsigned *queue;
    unsigned slot_count;
    unsigned q_head;
    unsigned q_tail;
    unsigned q_count;
    bool stop;
    pthread_mutex_t mutex;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
} EncodePool;

static void *encode_pool_worker(void *arg) {
    EncodePool *pool = arg;
    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (!pool->stop && pool->q_count == 0)
            pthread_cond_wait(&pool->work_cv, &pool->mutex);
        if (pool->stop) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }

        unsigned slot = pool->queue[pool->q_head];
        pool->q_head = (pool->q_head + 1u) % pool->slot_count;
        --pool->q_count;
        pool->states[slot] = ENCODE_SLOT_RUNNING;
        pthread_mutex_unlock(&pool->mutex);

        encode_thread(&pool->jobs[slot]);

        pthread_mutex_lock(&pool->mutex);
        pool->states[slot] = ENCODE_SLOT_DONE;
        pthread_cond_broadcast(&pool->done_cv);
        pthread_mutex_unlock(&pool->mutex);
    }
}

static int encode_pool_enqueue(EncodePool *pool, unsigned slot) {
    pthread_mutex_lock(&pool->mutex);
    if (pool->stop || pool->q_count >= pool->slot_count ||
        pool->states[slot] != ENCODE_SLOT_EMPTY) {
        pthread_mutex_unlock(&pool->mutex);
        return -1;
    }

    pool->states[slot] = ENCODE_SLOT_QUEUED;
    pool->queue[pool->q_tail] = slot;
    pool->q_tail = (pool->q_tail + 1u) % pool->slot_count;
    ++pool->q_count;
    pthread_cond_signal(&pool->work_cv);
    pthread_mutex_unlock(&pool->mutex);
    return 0;
}

static int encode_pool_wait_done(EncodePool *pool, unsigned slot) {
    pthread_mutex_lock(&pool->mutex);
    while (!pool->stop && pool->states[slot] != ENCODE_SLOT_DONE)
        pthread_cond_wait(&pool->done_cv, &pool->mutex);
    int rc = pool->states[slot] == ENCODE_SLOT_DONE ? 0 : -1;
    pthread_mutex_unlock(&pool->mutex);
    return rc;
}

static void encode_pool_mark_empty(EncodePool *pool, unsigned slot) {
    pthread_mutex_lock(&pool->mutex);
    pool->states[slot] = ENCODE_SLOT_EMPTY;
    pthread_mutex_unlock(&pool->mutex);
}

static void encode_pool_stop(EncodePool *pool) {
    pthread_mutex_lock(&pool->mutex);
    pool->stop = true;
    pthread_cond_broadcast(&pool->work_cv);
    pthread_cond_broadcast(&pool->done_cv);
    pthread_mutex_unlock(&pool->mutex);
}

static int encode_read_job(
    FILE *in,
    EncodeJob *job,
    uint64_t block_index,
    uint64_t total_size,
    size_t block_size,
    size_t min_region
) {
    uint64_t off = block_index * (uint64_t)block_size;
    if (off >= total_size)
        return -1;

    size_t want = (size_t)((total_size - off) < block_size ?
                           (total_size - off) : block_size);
    memset(job, 0, sizeof(*job));
    job->input = malloc(want ? want : 1);
    job->input_len = want;
    job->min_region = min_region;

    if (!job->input)
        return -1;
    if (fread(job->input, 1, want, in) != want) {
        free(job->input);
        job->input = NULL;
        return -1;
    }
    return 0;
}

static void encode_job_release(EncodeJob *job) {
    free(job->input);
    job->input = NULL;
    bv_free(&job->encoded);
}

static unsigned encode_pipeline_slots(
    size_t block_size,
    unsigned threads,
    uint64_t blocks
) {
    if (!blocks)
        return 0;

    uint64_t base_slots = threads;
    if (base_slots > blocks)
        base_slots = blocks;

    unsigned desired_extra = 4;
    const char *env = getenv("BASECOMPRESSER_PIPELINE_EXTRA");
    if (env && *env) {
        unsigned long parsed = strtoul(env, NULL, 10);
        if (parsed <= 64)
            desired_extra = (unsigned)parsed;
    }

    uint64_t max_extra = blocks > base_slots ? blocks - base_slots : 0;
    if (desired_extra > max_extra)
        desired_extra = (unsigned)max_extra;
    if (!desired_extra)
        return (unsigned)base_slots;

    /*
     * Active workers already use the conservative auto_threads() estimate.
     * A look-ahead slot can simultaneously hold an input and an encoded block,
     * so budget two block sizes per extra slot.
     */
    uint64_t avail = available_memory_bytes();
    if (!avail && desired_extra > 1)
        desired_extra = 1;
    if (avail) {
        uint64_t budget = avail * 3u / 5u;
        uint64_t active_per =
            (uint64_t)block_size * 5u + 32u * 1024u * 1024u;
        uint64_t extra_per =
            (uint64_t)block_size * 2u + 1024u * 1024u;

        uint64_t active_need =
            active_per > UINT64_MAX / threads ?
            UINT64_MAX : active_per * threads;
        uint64_t headroom =
            budget > active_need ? budget - active_need : 0;
        uint64_t mem_extra = extra_per ? headroom / extra_per : 0;
        if ((uint64_t)desired_extra > mem_extra)
            desired_extra = (unsigned)mem_extra;
    }

    return (unsigned)(base_slots + desired_extra);
}

static int encode_file(
    const char *src,
    const char *dst,
    size_t block_size,
    size_t min_region,
    unsigned requested_threads
) {
    uint64_t total_size = file_size_of(src);
    if (total_size == UINT64_MAX) {
        fprintf(stderr, "Cannot stat input: %s\n", src);
        return 1;
    }
    if (block_size == 0 || block_size > UINT32_MAX ||
        min_region == 0 || min_region > block_size) {
        fprintf(stderr, "Invalid block/min-region size.\n");
        return 1;
    }

    uint64_t blocks = total_size ?
        (total_size + block_size - 1) / block_size : 0;
    unsigned threads = auto_threads(block_size, requested_threads, blocks);

    FILE *in = fopen(src, "rb");
    if (!in) {
        perror("open input");
        return 1;
    }
    FILE *out = fopen(dst, "wb");
    if (!out) {
        perror("open output");
        fclose(in);
        return 1;
    }
    if (write_file_header(
            out, total_size, (uint32_t)block_size,
            (uint32_t)min_region) != 0) {
        fprintf(stderr, "header write failed\n");
        fclose(in);
        fclose(out);
        return 1;
    }

    printf("BASE9 encode: block=%zu MiB, min-region=%zu KiB, threads=%u\n",
           block_size >> 20, min_region >> 10, threads);

    if (blocks == 0) {
        fclose(in);
        fclose(out);
        printf("BASE9 encode complete\n"
               "Input:       0 bytes\n"
               "Output:      %" PRIu64 " bytes\n"
               "Regions:     0 (raw=0, base=0, rans=0, delta-rans=0, context-rans=0, token-base=0)\n"
               "Threads:     %u\n"
               "Pipeline:    0 slots\n"
               "Elapsed:     0.000 s\n"
               "Throughput:  0.00 MiB/s\n"
               "Created:     %s\n",
               file_size_of(dst), threads, dst);
        return 0;
    }

    /*
     * A small RAM-aware look-ahead queue keeps workers fed without multiplying
     * the large per-block working set. Workers persist for the whole file, so
     * later blocks can run CPU analysis while another block is inside the
     * serialized GPU queue.
     */
    unsigned slot_count =
        encode_pipeline_slots(block_size, threads, blocks);
    if (slot_count < 1)
        slot_count = 1;

    EncodePool pool = {0};
    pool.slot_count = slot_count;
    pool.jobs = calloc(slot_count, sizeof(*pool.jobs));
    pool.states = calloc(slot_count, sizeof(*pool.states));
    pool.queue = calloc(slot_count, sizeof(*pool.queue));
    pthread_t *workers = calloc(threads, sizeof(*workers));

    if (!pool.jobs || !pool.states || !pool.queue || !workers) {
        fprintf(stderr, "out of memory\n");
        free(pool.jobs);
        free(pool.states);
        free(pool.queue);
        free(workers);
        fclose(in);
        fclose(out);
        return 1;
    }

    pthread_mutex_init(&pool.mutex, NULL);
    pthread_cond_init(&pool.work_cv, NULL);
    pthread_cond_init(&pool.done_cv, NULL);

    unsigned workers_started = 0;
    for (; workers_started < threads; ++workers_started) {
        if (pthread_create(
                &workers[workers_started], NULL,
                encode_pool_worker, &pool) != 0)
            break;
    }
    if (workers_started != threads) {
        fprintf(stderr, "failed to start encode worker pool\n");
        encode_pool_stop(&pool);
        for (unsigned i = 0; i < workers_started; ++i)
            pthread_join(workers[i], NULL);
        pthread_cond_destroy(&pool.done_cv);
        pthread_cond_destroy(&pool.work_cv);
        pthread_mutex_destroy(&pool.mutex);
        free(pool.jobs);
        free(pool.states);
        free(pool.queue);
        free(workers);
        fclose(in);
        fclose(out);
        return 1;
    }

    double start_time = now_seconds();
    uint64_t next_read_block = 0;
    uint64_t next_write_block = 0;
    uint64_t written_bytes = 0;
    uint64_t total_regions = 0;
    uint64_t methods[METHOD_COUNT] = {0};
    int failed = 0;

    for (; next_read_block < slot_count; ++next_read_block) {
        unsigned slot = (unsigned)(next_read_block % slot_count);
        if (encode_read_job(
                in, &pool.jobs[slot], next_read_block,
                total_size, block_size, min_region) != 0 ||
            encode_pool_enqueue(&pool, slot) != 0) {
            fprintf(stderr, "input read/queue failed\n");
            failed = 1;
            break;
        }
    }

    while (!failed && next_write_block < blocks) {
        unsigned slot = (unsigned)(next_write_block % slot_count);
        if (encode_pool_wait_done(&pool, slot) != 0) {
            fprintf(stderr, "encode pipeline interrupted\n");
            failed = 1;
            break;
        }

        EncodeJob *job = &pool.jobs[slot];
        if (job->error ||
            fwrite(job->encoded.data, 1, job->encoded.len, out) !=
                job->encoded.len) {
            fprintf(stderr, "encode/write failed: %s\n", job->error_msg);
            failed = 1;
            break;
        }

        written_bytes += job->input_len;
        total_regions += job->region_count;
        for (int m = 0; m < METHOD_COUNT; ++m)
            methods[m] += job->method_counts[m];

        encode_job_release(job);
        encode_pool_mark_empty(&pool, slot);
        ++next_write_block;

        if (next_read_block < blocks) {
            if (encode_read_job(
                    in, job, next_read_block,
                    total_size, block_size, min_region) != 0 ||
                encode_pool_enqueue(&pool, slot) != 0) {
                fprintf(stderr, "input read/queue failed\n");
                failed = 1;
                break;
            }
            ++next_read_block;
        }

        double pct = total_size ?
            100.0 * (double)written_bytes / (double)total_size : 100.0;
        printf("\rEncoding: %6.2f%% | regions=%" PRIu64
               " | raw=%" PRIu64 " base=%" PRIu64
               " rans=%" PRIu64 " delta=%" PRIu64
               " ctx=%" PRIu64 " token=%" PRIu64,
               pct, total_regions, methods[0], methods[1], methods[2],
               methods[3], methods[4], methods[5]);
        fflush(stdout);
    }

    encode_pool_stop(&pool);
    for (unsigned i = 0; i < workers_started; ++i)
        pthread_join(workers[i], NULL);

    for (unsigned i = 0; i < slot_count; ++i)
        encode_job_release(&pool.jobs[i]);

    pthread_cond_destroy(&pool.done_cv);
    pthread_cond_destroy(&pool.work_cv);
    pthread_mutex_destroy(&pool.mutex);
    free(pool.jobs);
    free(pool.states);
    free(pool.queue);
    free(workers);

    printf("\n");

    int close_error = 0;
    if (fclose(in) != 0)
        close_error = 1;
    if (fclose(out) != 0)
        close_error = 1;

    if (failed || close_error) {
        if (close_error)
            perror("close");
        return 1;
    }

    uint64_t out_size = file_size_of(dst);
    double elapsed = now_seconds() - start_time;

    printf("BASE9 encode complete\n"
           "Input:       %" PRIu64 " bytes\n"
           "Output:      %" PRIu64 " bytes\n",
           total_size, out_size);
    if (total_size)
        printf("Ratio:       %.4fx (%.2f%% of original)\n",
               (double)out_size / total_size,
               100.0 * (double)out_size / total_size);

    printf("Regions:     %" PRIu64
           " (raw=%" PRIu64 ", base=%" PRIu64
           ", rans=%" PRIu64 ", delta-rans=%" PRIu64
           ", context-rans=%" PRIu64 ", token-base=%" PRIu64 ")\n"
           "Threads:     %u\n"
           "Pipeline:    %u slots (persistent workers)\n"
           "Elapsed:     %.3f s\n"
           "Throughput:  %.2f MiB/s\n"
           "Created:     %s\n",
           total_regions, methods[0], methods[1], methods[2],
           methods[3], methods[4], methods[5],
           threads, slot_count, elapsed,
           elapsed > 0 ?
               (double)total_size / (1024.0 * 1024.0) / elapsed : 0.0,
           dst);

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

static int gpu_bench_generate_input(const char *path, size_t bytes) {
    uint8_t *buf = malloc(bytes);
    if (!buf)
        return -1;

    static const uint8_t prefix[] = "RECORD:";
    static const uint8_t middle[] = ":VALUE:";
    size_t pos = 0;
    uint32_t i = 0;

    while (pos + 23u <= bytes) {
        memcpy(buf + pos, prefix, sizeof(prefix) - 1);
        pos += sizeof(prefix) - 1;

        uint32_t a = i;
        buf[pos++] = (uint8_t)a;
        buf[pos++] = (uint8_t)(a >> 8);
        buf[pos++] = (uint8_t)(a >> 16);
        buf[pos++] = (uint8_t)(a >> 24);

        memcpy(buf + pos, middle, sizeof(middle) - 1);
        pos += sizeof(middle) - 1;

        uint32_t b = i * 2654435761u;
        buf[pos++] = (uint8_t)b;
        buf[pos++] = (uint8_t)(b >> 8);
        buf[pos++] = (uint8_t)(b >> 16);
        buf[pos++] = (uint8_t)(b >> 24);
        buf[pos++] = '\n';
        ++i;
    }

    while (pos < bytes) {
        buf[pos] = (uint8_t)((pos * 37u + 11u) & 255u);
        ++pos;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        free(buf);
        return -1;
    }
    size_t wrote = fwrite(buf, 1, bytes, f);
    int rc = fclose(f);
    free(buf);
    return wrote == bytes && rc == 0 ? 0 : -1;
}

static bool gpu_bench_files_equal(const char *a_path, const char *b_path) {
    FILE *a = fopen(a_path, "rb");
    FILE *b = fopen(b_path, "rb");
    if (!a || !b) {
        if (a) fclose(a);
        if (b) fclose(b);
        return false;
    }

    uint8_t ab[64 * 1024], bb[64 * 1024];
    bool equal = true;
    for (;;) {
        size_t an = fread(ab, 1, sizeof(ab), a);
        size_t bn = fread(bb, 1, sizeof(bb), b);
        if (an != bn || (an && memcmp(ab, bb, an) != 0)) {
            equal = false;
            break;
        }
        if (an == 0) {
            if (ferror(a) || ferror(b))
                equal = false;
            break;
        }
    }

    fclose(a);
    fclose(b);
    return equal;
}

static int gpu_bench_child(
    const char *input,
    const char *output,
    unsigned threads,
    unsigned mode,
    double *elapsed
) {
    char threads_s[32];
    snprintf(threads_s, sizeof(threads_s), "%u", threads);

    double start = now_seconds();
    pid_t pid = fork();
    if (pid < 0)
        return -1;

    if (pid == 0) {
        setenv("BASECOMPRESSER_PIPELINE_EXTRA", "4", 1);
        setenv("BASECOMPRESSER_GPU_ZERO_COPY", "off", 1);

        if (mode == 0) {
            setenv("BASECOMPRESSER_GPU", "off", 1);
            unsetenv("BASECOMPRESSER_GPU_HIST");
            unsetenv("BASECOMPRESSER_GPU_PACK");
        } else {
            setenv("BASECOMPRESSER_GPU", "force", 1);
            if (mode == 1) {
                setenv("BASECOMPRESSER_GPU_HIST", "sharded", 1);
                setenv("BASECOMPRESSER_GPU_PACK", "off", 1);
            } else if (mode == 2) {
                setenv("BASECOMPRESSER_GPU_HIST", "cpu", 1);
                setenv("BASECOMPRESSER_GPU_PACK", "force", 1);
            } else {
                setenv("BASECOMPRESSER_GPU_HIST", "sharded", 1);
                setenv("BASECOMPRESSER_GPU_PACK", "force", 1);
            }
        }

        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            close(devnull);
        }

        execl(
            "/proc/self/exe",
            "basecompresser",
            "encode",
            input,
            "-o", output,
            "--block-mib", "8",
            "--min-region-kib", "8192",
            "--threads", threads_s,
            (char *)NULL);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid)
        return -1;
    *elapsed = now_seconds() - start;

    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1;
}

static int gpu_bench_mode(
    const char *input,
    const char *output,
    unsigned threads,
    unsigned mode,
    double *average
) {
    double sum = 0.0;
    for (unsigned run = 0; run < 2; ++run) {
        double elapsed = 0.0;
        if (gpu_bench_child(
                input, output, threads, mode, &elapsed) != 0)
            return -1;
        sum += elapsed;
    }
    *average = sum / 2.0;
    return 0;
}

static int gpu_bench(void) {
    if (base9_gpu_init() != 0) {
        fprintf(stderr, "No usable OpenCL GPU is available.\n");
        return 1;
    }

    printf("BASE9 GPU calibration\n"
           "Vendor:  %s\n"
           "Device:  %s\n"
           "Driver:  %s\n",
           base9_gpu.vendor, base9_gpu.name, base9_gpu.driver);

    char input_path[] = "/tmp/basecompresser-gpu-bench-input-XXXXXX";
    char cpu_path[] = "/tmp/basecompresser-gpu-bench-cpu-XXXXXX";
    char test_path[] = "/tmp/basecompresser-gpu-bench-test-XXXXXX";

    int input_fd = mkstemp(input_path);
    int cpu_fd = mkstemp(cpu_path);
    int test_fd = mkstemp(test_path);
    if (input_fd < 0 || cpu_fd < 0 || test_fd < 0) {
        if (input_fd >= 0) close(input_fd);
        if (cpu_fd >= 0) close(cpu_fd);
        if (test_fd >= 0) close(test_fd);
        if (input_fd >= 0) unlink(input_path);
        if (cpu_fd >= 0) unlink(cpu_path);
        if (test_fd >= 0) unlink(test_path);
        perror("gpu-bench temporary file");
        return 1;
    }
    close(input_fd);
    close(cpu_fd);
    close(test_fd);

    const size_t bench_bytes = 64u * 1024u * 1024u;
    if (gpu_bench_generate_input(input_path, bench_bytes) != 0) {
        fprintf(stderr, "Failed to generate GPU benchmark input.\n");
        unlink(input_path);
        unlink(cpu_path);
        unlink(test_path);
        return 1;
    }

    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    unsigned threads = (unsigned)(cpus > 0 ? cpus : 1);
    if (threads > 4)
        threads = 4;
    if (threads < 1)
        threads = 1;

    static const char *names[4] = {
        "CPU only",
        "GPU histograms",
        "GPU radix",
        "GPU histograms + radix"
    };
    double times[4] = {0};

    printf("Corpus:  64 MiB token-heavy, 8 MiB blocks, %u workers\n",
           threads);

    int failed = 0;
    for (unsigned mode = 0; mode < 4; ++mode) {
        const char *out = mode == 0 ? cpu_path : test_path;
        if (gpu_bench_mode(
                input_path, out, threads, mode, &times[mode]) != 0) {
            fprintf(stderr, "Calibration mode failed: %s\n", names[mode]);
            failed = 1;
            break;
        }

        if (mode != 0 && !gpu_bench_files_equal(cpu_path, test_path)) {
            fprintf(stderr,
                    "Calibration aborted: %s produced different BASE9 bytes.\n",
                    names[mode]);
            failed = 1;
            break;
        }

        printf("%-24s %.3f s  (%.2f MiB/s)\n",
               names[mode],
               times[mode],
               64.0 / times[mode]);
    }

    if (failed) {
        unlink(input_path);
        unlink(cpu_path);
        unlink(test_path);
        return 1;
    }

    unsigned best = 0;
    for (unsigned mode = 1; mode < 4; ++mode) {
        if (times[mode] < times[best])
            best = mode;
    }

    /*
     * Require a 3% end-to-end improvement before AUTO changes away from CPU.
     * This avoids turning tiny benchmark jitter into a persistent policy.
     */
    if (best != 0 && times[best] > times[0] * 0.97)
        best = 0;

    bool hist_gpu = best == 1 || best == 3;
    bool pack_gpu = best == 2 || best == 3;

    if (base9_gpu_policy_save(
            hist_gpu, pack_gpu,
            times[0], times[1], times[2], times[3]) != 0) {
        fprintf(stderr, "Calibration succeeded but policy cache write failed.\n");
        unlink(input_path);
        unlink(cpu_path);
        unlink(test_path);
        return 1;
    }

    printf("AUTO policy: histograms=%s, radix=%s\n",
           hist_gpu ? "GPU" : "CPU",
           pack_gpu ? "GPU" : "CPU");
    if (best == 0) {
        printf("Reason: GPU combinations did not beat CPU by at least 3%%.\n");
    } else {
        printf("Measured end-to-end gain: %.2f%%\n",
               (times[0] / times[best] - 1.0) * 100.0);
    }

    unlink(input_path);
    unlink(cpu_path);
    unlink(test_path);
    return 0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n  %s encode INPUT [-o OUTPUT] [--block-mib N] [--min-region-kib N] [--threads N]\n  %s decode INPUT [-o OUTPUT] [--threads N]\n  %s gpu-info\n  %s gpu-bench\n\n"
        "BASE9 is a bounded-RAM adaptive compressor. It chooses per region among\n"
        "RAW, adaptive-base packing, token-base, static rANS, delta+rANS, and context rANS,\n"
        "and recursively splits large blocks when local statistics compress better.\n"
        "GPU token discovery uses OpenCL when available and always has CPU fallback.\n", argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    base9_gpu_maybe_reexec_private_intel(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "gpu-info") == 0) {
        base9_gpu_print_info();
        return base9_gpu.available ? 0 : 1;
    }
    if (argc >= 2 && strcmp(argv[1], "gpu-bench") == 0)
        return gpu_bench();
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
