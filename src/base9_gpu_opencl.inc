/*
 * Optional OpenCL accelerator for BASE9 token discovery.
 *
 * One OpenCL kernel source serves NVIDIA, AMD and Intel GPUs.  The OpenCL
 * loader is resolved at runtime, so BaseCompresser keeps building and running
 * on machines with no GPU runtime at all.
 *
 * The current accelerator targets the hot pair-histogram step used by
 * token-base.  CPU remains the authority for token selection and radix packing.
 */

#include <dlfcn.h>
#include <limits.h>
#include <dirent.h>
#include <strings.h>
#include <ctype.h>

typedef int32_t bc_cl_int;
typedef uint32_t bc_cl_uint;
typedef uint64_t bc_cl_ulong;
typedef uint64_t bc_cl_bitfield;
typedef bc_cl_bitfield bc_cl_device_type;
typedef bc_cl_bitfield bc_cl_mem_flags;
typedef bc_cl_bitfield bc_cl_command_queue_properties;
typedef intptr_t bc_cl_context_properties;
typedef struct _cl_platform_id *bc_cl_platform_id;
typedef struct _cl_device_id *bc_cl_device_id;
typedef struct _cl_context *bc_cl_context;
typedef struct _cl_command_queue *bc_cl_command_queue;
typedef struct _cl_mem *bc_cl_mem;
typedef struct _cl_program *bc_cl_program;
typedef struct _cl_kernel *bc_cl_kernel;
typedef struct _cl_event *bc_cl_event;

#define BC_CL_SUCCESS 0
#define BC_CL_TRUE 1
#define BC_CL_DEVICE_TYPE_GPU (1ULL << 2)
#define BC_CL_MEM_READ_ONLY (1ULL << 2)
#define BC_CL_MEM_READ_WRITE (1ULL << 0)
#define BC_CL_DEVICE_NAME 0x102B
#define BC_CL_DEVICE_VENDOR 0x102C
#define BC_CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define BC_CL_PROGRAM_BUILD_LOG 0x1183

typedef struct {
    bc_cl_int (*GetPlatformIDs)(bc_cl_uint, bc_cl_platform_id *, bc_cl_uint *);
    bc_cl_int (*GetDeviceIDs)(bc_cl_platform_id, bc_cl_device_type, bc_cl_uint, bc_cl_device_id *, bc_cl_uint *);
    bc_cl_int (*GetDeviceInfo)(bc_cl_device_id, bc_cl_uint, size_t, void *, size_t *);
    bc_cl_context (*CreateContext)(const bc_cl_context_properties *, bc_cl_uint, const bc_cl_device_id *, void *, void *, bc_cl_int *);
    bc_cl_command_queue (*CreateCommandQueue)(bc_cl_context, bc_cl_device_id, bc_cl_command_queue_properties, bc_cl_int *);
    bc_cl_program (*CreateProgramWithSource)(bc_cl_context, bc_cl_uint, const char **, const size_t *, bc_cl_int *);
    bc_cl_int (*BuildProgram)(bc_cl_program, bc_cl_uint, const bc_cl_device_id *, const char *, void *, void *);
    bc_cl_int (*GetProgramBuildInfo)(bc_cl_program, bc_cl_device_id, bc_cl_uint, size_t, void *, size_t *);
    bc_cl_kernel (*CreateKernel)(bc_cl_program, const char *, bc_cl_int *);
    bc_cl_mem (*CreateBuffer)(bc_cl_context, bc_cl_mem_flags, size_t, void *, bc_cl_int *);
    bc_cl_int (*SetKernelArg)(bc_cl_kernel, bc_cl_uint, size_t, const void *);
    bc_cl_int (*EnqueueWriteBuffer)(bc_cl_command_queue, bc_cl_mem, bc_cl_uint, size_t, size_t, const void *, bc_cl_uint, const bc_cl_event *, bc_cl_event *);
    bc_cl_int (*EnqueueNDRangeKernel)(bc_cl_command_queue, bc_cl_kernel, bc_cl_uint, const size_t *, const size_t *, const size_t *, bc_cl_uint, const bc_cl_event *, bc_cl_event *);
    bc_cl_int (*EnqueueReadBuffer)(bc_cl_command_queue, bc_cl_mem, bc_cl_uint, size_t, size_t, void *, bc_cl_uint, const bc_cl_event *, bc_cl_event *);
    bc_cl_int (*Finish)(bc_cl_command_queue);
    bc_cl_int (*ReleaseMemObject)(bc_cl_mem);
    bc_cl_int (*ReleaseKernel)(bc_cl_kernel);
    bc_cl_int (*ReleaseProgram)(bc_cl_program);
    bc_cl_int (*ReleaseCommandQueue)(bc_cl_command_queue);
    bc_cl_int (*ReleaseContext)(bc_cl_context);
} Base9OpenCLApi;

typedef struct {
    void *loader;
    Base9OpenCLApi api;
    bc_cl_device_id device;
    bc_cl_context context;
    bc_cl_command_queue queue;
    bc_cl_program program;
    bc_cl_kernel pair_u8;
    bc_cl_kernel pair_u16;
    bc_cl_mem input;
    size_t input_cap;
    bc_cl_mem counts;
    size_t counts_cap;
    bc_cl_uint compute_units;
    char vendor[128];
    char name[256];
    int available;
    int permanently_failed;
    uint64_t successful_jobs;
    uint64_t successful_input_bytes;
    pthread_mutex_t lock;
} Base9GpuState;

static Base9GpuState base9_gpu = {
    .lock = PTHREAD_MUTEX_INITIALIZER
};
static pthread_once_t base9_gpu_once = PTHREAD_ONCE_INIT;

static const char *base9_opencl_source =
"__kernel void pair_u8(__global const uchar *in, ulong n, uint cap, __global uint *counts) {\n"
"  size_t gid=get_global_id(0), stride=get_global_size(0);\n"
"  for (ulong i=(ulong)gid; i+1<n; i+=(ulong)stride) {\n"
"    uint a=(uint)in[i], b=(uint)in[i+1];\n"
"    atomic_inc((volatile __global unsigned int *)&counts[a*cap+b]);\n"
"  }\n"
"}\n"
"__kernel void pair_u16(__global const ushort *in, ulong n, uint cap, __global uint *counts) {\n"
"  size_t gid=get_global_id(0), stride=get_global_size(0);\n"
"  for (ulong i=(ulong)gid; i+1<n; i+=(ulong)stride) {\n"
"    uint a=(uint)in[i], b=(uint)in[i+1];\n"
"    if (a<cap && b<cap) atomic_inc((volatile __global unsigned int *)&counts[a*cap+b]);\n"
"  }\n"
"}\n";

static bool base9_gpu_env_off(void) {
    const char *v = getenv("BASECOMPRESSER_GPU");
    return v && (!strcasecmp(v, "off") || !strcasecmp(v, "no") ||
                 !strcmp(v, "0") || !strcasecmp(v, "false"));
}

static bool base9_gpu_env_force(void) {
    const char *v = getenv("BASECOMPRESSER_GPU");
    return v && (!strcasecmp(v, "force") || !strcasecmp(v, "on") ||
                 !strcmp(v, "1") || !strcasecmp(v, "true"));
}

static bool base9_intel_skylake_host(void) {
#if defined(__linux__) && defined(__x86_64__)
    static const unsigned ids[] = {
        0x1902,0x1906,0x190a,0x190b,0x190e,0x1912,0x1913,0x1915,
        0x1916,0x1917,0x191a,0x191b,0x191d,0x191e,0x1921,0x1923,
        0x1926,0x1927,0x192a,0x192b,0x192d,0x1932,0x193a,0x193b,0x193d
    };
    for (unsigned card = 128; card < 256; ++card) {
        char p[PATH_MAX], vendor[32] = {0}, device[32] = {0};
        snprintf(p, sizeof(p), "/sys/class/drm/renderD%u/device/vendor", card);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        if (!fgets(vendor, sizeof(vendor), f)) { fclose(f); continue; }
        fclose(f);
        if (strtoul(vendor, NULL, 0) != 0x8086u) continue;
        snprintf(p, sizeof(p), "/sys/class/drm/renderD%u/device/device", card);
        f = fopen(p, "r");
        if (!f) continue;
        if (!fgets(device, sizeof(device), f)) { fclose(f); continue; }
        fclose(f);
        unsigned id = (unsigned)strtoul(device, NULL, 0);
        for (size_t i = 0; i < sizeof(ids)/sizeof(ids[0]); ++i)
            if (id == ids[i]) return true;
    }
#endif
    return false;
}

static int base9_gpu_private_paths(char *icd_dir, size_t icd_n, char *lib_dir, size_t lib_n) {
    const char *cache = getenv("XDG_CACHE_HOME");
    char root[PATH_MAX];
    if (cache && *cache) {
        if (snprintf(root, sizeof(root), "%s/BaseCompresser/intel-legacy-opencl/runtime", cache) >= (int)sizeof(root))
            return -1;
    } else {
        const char *home = getenv("HOME");
        if (!home || !*home) return -1;
        if (snprintf(root, sizeof(root), "%s/.cache/BaseCompresser/intel-legacy-opencl/runtime", home) >= (int)sizeof(root))
            return -1;
    }
    if (snprintf(icd_dir, icd_n, "%s/etc/OpenCL/vendors", root) >= (int)icd_n ||
        snprintf(lib_dir, lib_n, "%s/usr/lib/x86_64-linux-gnu", root) >= (int)lib_n)
        return -1;
    char icd[PATH_MAX], lib[PATH_MAX];
    static const char icd_suffix[] = "/intel.icd";
    static const char lib_suffix[] = "/intel-opencl/libigdrcl.so";
    size_t ilen = strlen(icd_dir), llen = strlen(lib_dir);
    if (ilen + sizeof(icd_suffix) > sizeof(icd) ||
        llen + sizeof(lib_suffix) > sizeof(lib))
        return -1;
    memcpy(icd, icd_dir, ilen);
    memcpy(icd + ilen, icd_suffix, sizeof(icd_suffix));
    memcpy(lib, lib_dir, llen);
    memcpy(lib + llen, lib_suffix, sizeof(lib_suffix));
    return access(icd, R_OK) == 0 && access(lib, R_OK) == 0 ? 0 : -1;
}

/*
 * The ICD loader and Intel compiler dependencies read their library paths when
 * the process starts.  Re-exec once with the private cache in the environment,
 * mirroring 265Encode's private legacy-runtime approach.
 */
static void base9_gpu_maybe_reexec_private_intel(int argc, char **argv) {
    if (base9_gpu_env_off()) return;
    if (getenv("BASECOMPRESSER_PRIVATE_OPENCL_ACTIVE")) return;
    bool info = argc >= 2 && strcmp(argv[1], "gpu-info") == 0;
    if (!info && !base9_gpu_env_force()) return;
    if (!base9_intel_skylake_host()) return;

    /* Prefer any already configured system ICD. */
    if (access("/etc/OpenCL/vendors", R_OK) == 0) {
        DIR *d = opendir("/etc/OpenCL/vendors");
        if (d) {
            struct dirent *e;
            bool found = false;
            while ((e = readdir(d))) {
                size_t n = strlen(e->d_name);
                if (n > 4 && strcmp(e->d_name + n - 4, ".icd") == 0) {
                    found = true; break;
                }
            }
            closedir(d);
            if (found) return;
        }
    }

    char icd_dir[PATH_MAX], lib_dir[PATH_MAX];
    if (base9_gpu_private_paths(icd_dir, sizeof(icd_dir), lib_dir, sizeof(lib_dir)) != 0)
        return;

    const char *old_ld = getenv("LD_LIBRARY_PATH");
    char ld[PATH_MAX * 2];
    if (old_ld && *old_ld)
        snprintf(ld, sizeof(ld), "%s:%s", lib_dir, old_ld);
    else
        snprintf(ld, sizeof(ld), "%s", lib_dir);
    setenv("OCL_ICD_VENDORS", icd_dir, 1);
    setenv("LD_LIBRARY_PATH", ld, 1);
    setenv("BASECOMPRESSER_PRIVATE_OPENCL_ACTIVE", "1", 1);
    execv("/proc/self/exe", argv);
}

static int base9_ocl_sym(void **dst, void *lib, const char *name) {
    *dst = dlsym(lib, name);
    return *dst ? 0 : -1;
}

static bool base9_ascii_contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle || !*needle) return true;
    size_t nn = strlen(needle);
    for (const char *p = haystack; *p; ++p) {
        size_t i = 0;
        while (i < nn && p[i] &&
               tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i]))
            ++i;
        if (i == nn) return true;
    }
    return false;
}

static bool base9_gpu_vendor_allowed(const char *vendor) {
    const char *want = getenv("BASECOMPRESSER_GPU_VENDOR");
    if (!want || !*want || !strcasecmp(want, "any") || !strcasecmp(want, "auto"))
        return true;
    if (!strcasecmp(want, "nvidia"))
        return base9_ascii_contains_ci(vendor, "nvidia");
    if (!strcasecmp(want, "amd"))
        return base9_ascii_contains_ci(vendor, "amd") ||
               base9_ascii_contains_ci(vendor, "advanced micro");
    if (!strcasecmp(want, "intel"))
        return base9_ascii_contains_ci(vendor, "intel");
    return true;
}

static void base9_gpu_init_once(void) {
    if (base9_gpu_env_off()) return;

    void *lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return;
    Base9OpenCLApi *a = &base9_gpu.api;
#define BC_LOAD(member, name) do { if (base9_ocl_sym((void **)&a->member, lib, name) != 0) goto fail; } while (0)
    BC_LOAD(GetPlatformIDs, "clGetPlatformIDs");
    BC_LOAD(GetDeviceIDs, "clGetDeviceIDs");
    BC_LOAD(GetDeviceInfo, "clGetDeviceInfo");
    BC_LOAD(CreateContext, "clCreateContext");
    BC_LOAD(CreateCommandQueue, "clCreateCommandQueue");
    BC_LOAD(CreateProgramWithSource, "clCreateProgramWithSource");
    BC_LOAD(BuildProgram, "clBuildProgram");
    BC_LOAD(GetProgramBuildInfo, "clGetProgramBuildInfo");
    BC_LOAD(CreateKernel, "clCreateKernel");
    BC_LOAD(CreateBuffer, "clCreateBuffer");
    BC_LOAD(SetKernelArg, "clSetKernelArg");
    BC_LOAD(EnqueueWriteBuffer, "clEnqueueWriteBuffer");
    BC_LOAD(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
    BC_LOAD(EnqueueReadBuffer, "clEnqueueReadBuffer");
    BC_LOAD(Finish, "clFinish");
    BC_LOAD(ReleaseMemObject, "clReleaseMemObject");
    BC_LOAD(ReleaseKernel, "clReleaseKernel");
    BC_LOAD(ReleaseProgram, "clReleaseProgram");
    BC_LOAD(ReleaseCommandQueue, "clReleaseCommandQueue");
    BC_LOAD(ReleaseContext, "clReleaseContext");
#undef BC_LOAD

    bc_cl_uint np = 0;
    if (a->GetPlatformIDs(0, NULL, &np) != BC_CL_SUCCESS || !np)
        goto fail;
    bc_cl_platform_id *platforms = calloc(np, sizeof(*platforms));
    if (!platforms) goto fail;
    if (a->GetPlatformIDs(np, platforms, NULL) != BC_CL_SUCCESS) {
        free(platforms); goto fail;
    }

    bc_cl_device_id chosen = NULL;
    bc_cl_uint chosen_cu = 0;
    char chosen_vendor[128] = {0}, chosen_name[256] = {0};

    for (bc_cl_uint pi = 0; pi < np; ++pi) {
        bc_cl_uint nd = 0;
        if (a->GetDeviceIDs(platforms[pi], BC_CL_DEVICE_TYPE_GPU, 0, NULL, &nd) != BC_CL_SUCCESS || !nd)
            continue;
        bc_cl_device_id *devs = calloc(nd, sizeof(*devs));
        if (!devs) continue;
        if (a->GetDeviceIDs(platforms[pi], BC_CL_DEVICE_TYPE_GPU, nd, devs, NULL) != BC_CL_SUCCESS) {
            free(devs); continue;
        }
        for (bc_cl_uint di = 0; di < nd; ++di) {
            char vendor[128] = {0}, name[256] = {0};
            bc_cl_uint cu = 1;
            a->GetDeviceInfo(devs[di], BC_CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
            a->GetDeviceInfo(devs[di], BC_CL_DEVICE_NAME, sizeof(name), name, NULL);
            a->GetDeviceInfo(devs[di], BC_CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
            if (!base9_gpu_vendor_allowed(vendor)) continue;
            if (!chosen || cu > chosen_cu) {
                chosen = devs[di];
                chosen_cu = cu;
                snprintf(chosen_vendor, sizeof(chosen_vendor), "%s", vendor);
                snprintf(chosen_name, sizeof(chosen_name), "%s", name);
            }
        }
        free(devs);
    }
    free(platforms);
    if (!chosen) goto fail;

    bc_cl_int err = 0;
    base9_gpu.context = a->CreateContext(NULL, 1, &chosen, NULL, NULL, &err);
    if (!base9_gpu.context || err != BC_CL_SUCCESS) goto fail;
    base9_gpu.queue = a->CreateCommandQueue(base9_gpu.context, chosen, 0, &err);
    if (!base9_gpu.queue || err != BC_CL_SUCCESS) goto fail;

    const char *src = base9_opencl_source;
    size_t src_len = strlen(src);
    base9_gpu.program = a->CreateProgramWithSource(base9_gpu.context, 1, &src, &src_len, &err);
    if (!base9_gpu.program || err != BC_CL_SUCCESS) goto fail;
    err = a->BuildProgram(base9_gpu.program, 1, &chosen, "-cl-std=CL1.2", NULL, NULL);
    if (err != BC_CL_SUCCESS) {
        size_t n = 0;
        a->GetProgramBuildInfo(base9_gpu.program, chosen, BC_CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
        if (n) {
            char *log = malloc(n + 1);
            if (log) {
                a->GetProgramBuildInfo(base9_gpu.program, chosen, BC_CL_PROGRAM_BUILD_LOG, n, log, NULL);
                log[n] = 0;
                fprintf(stderr, "BASE9 GPU OpenCL build failed: %s\n", log);
                free(log);
            }
        }
        goto fail;
    }
    base9_gpu.pair_u8 = a->CreateKernel(base9_gpu.program, "pair_u8", &err);
    if (!base9_gpu.pair_u8 || err != BC_CL_SUCCESS) goto fail;
    base9_gpu.pair_u16 = a->CreateKernel(base9_gpu.program, "pair_u16", &err);
    if (!base9_gpu.pair_u16 || err != BC_CL_SUCCESS) goto fail;

    base9_gpu.loader = lib;
    base9_gpu.device = chosen;
    base9_gpu.compute_units = chosen_cu ? chosen_cu : 1;
    snprintf(base9_gpu.vendor, sizeof(base9_gpu.vendor), "%s", chosen_vendor);
    snprintf(base9_gpu.name, sizeof(base9_gpu.name), "%s", chosen_name);
    base9_gpu.available = 1;
    return;

fail:
    if (base9_gpu.pair_u16 && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.pair_u16);
    if (base9_gpu.pair_u8 && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.pair_u8);
    if (base9_gpu.program && a->ReleaseProgram) a->ReleaseProgram(base9_gpu.program);
    if (base9_gpu.queue && a->ReleaseCommandQueue) a->ReleaseCommandQueue(base9_gpu.queue);
    if (base9_gpu.context && a->ReleaseContext) a->ReleaseContext(base9_gpu.context);
    memset(&base9_gpu.api, 0, sizeof(base9_gpu.api));
    dlclose(lib);
    base9_gpu.loader = NULL;
    base9_gpu.permanently_failed = 1;
}

static int base9_gpu_init(void) {
    pthread_once(&base9_gpu_once, base9_gpu_init_once);
    return base9_gpu.available ? 0 : -1;
}

static int base9_gpu_ensure_buffers(size_t input_bytes, size_t count_bytes) {
    Base9OpenCLApi *a = &base9_gpu.api;
    bc_cl_int err = 0;
    if (input_bytes > base9_gpu.input_cap) {
        if (base9_gpu.input) a->ReleaseMemObject(base9_gpu.input);
        base9_gpu.input = a->CreateBuffer(base9_gpu.context, BC_CL_MEM_READ_ONLY, input_bytes, NULL, &err);
        if (!base9_gpu.input || err != BC_CL_SUCCESS) return -1;
        base9_gpu.input_cap = input_bytes;
    }
    if (count_bytes > base9_gpu.counts_cap) {
        if (base9_gpu.counts) a->ReleaseMemObject(base9_gpu.counts);
        base9_gpu.counts = a->CreateBuffer(base9_gpu.context, BC_CL_MEM_READ_WRITE, count_bytes, NULL, &err);
        if (!base9_gpu.counts || err != BC_CL_SUCCESS) return -1;
        base9_gpu.counts_cap = count_bytes;
    }
    return 0;
}

static int base9_gpu_pair_count_common(
    const void *input,
    size_t n,
    size_t elem_size,
    uint32_t *counts,
    size_t pair_cap,
    bc_cl_kernel kernel
) {
    if (n < 2 || !counts || pair_cap == 0 || pair_cap > UINT32_MAX)
        return 1;
    /* The Gen9/P530 compatibility path is supported, but its shared-memory
       global-atomic histogram is slower than the optimized CPU loop on our
       validated P530 host. Keep it opt-in unless explicitly forced. */
    if (getenv("BASECOMPRESSER_PRIVATE_OPENCL_ACTIVE") &&
        !base9_gpu_env_force())
        return 1;
    if (base9_gpu_init() != 0)
        return 1;

    size_t min_bytes = 1024u * 1024u;
    const char *threshold = getenv("BASECOMPRESSER_GPU_MIN_KIB");
    if (threshold && *threshold) {
        unsigned long kib = strtoul(threshold, NULL, 10);
        if (kib && kib <= SIZE_MAX / 1024u) min_bytes = (size_t)kib * 1024u;
    }
    if (!base9_gpu_env_force() && n * elem_size < min_bytes)
        return 1;

    if (n > SIZE_MAX / elem_size || pair_cap > SIZE_MAX / pair_cap ||
        pair_cap * pair_cap > SIZE_MAX / sizeof(uint32_t))
        return 1;
    size_t input_bytes = n * elem_size;
    size_t count_entries = pair_cap * pair_cap;
    size_t count_bytes = count_entries * sizeof(uint32_t);

    pthread_mutex_lock(&base9_gpu.lock);
    int rc = 1;
    Base9OpenCLApi *a = &base9_gpu.api;
    if (base9_gpu_ensure_buffers(input_bytes, count_bytes) != 0)
        goto done;

    memset(counts, 0, count_bytes);
    if (a->EnqueueWriteBuffer(base9_gpu.queue, base9_gpu.input, BC_CL_TRUE, 0, input_bytes, input, 0, NULL, NULL) != BC_CL_SUCCESS ||
        a->EnqueueWriteBuffer(base9_gpu.queue, base9_gpu.counts, BC_CL_TRUE, 0, count_bytes, counts, 0, NULL, NULL) != BC_CL_SUCCESS)
        goto done;

    bc_cl_ulong nn = (bc_cl_ulong)n;
    bc_cl_uint cap = (bc_cl_uint)pair_cap;
    if (a->SetKernelArg(kernel, 0, sizeof(base9_gpu.input), &base9_gpu.input) != BC_CL_SUCCESS ||
        a->SetKernelArg(kernel, 1, sizeof(nn), &nn) != BC_CL_SUCCESS ||
        a->SetKernelArg(kernel, 2, sizeof(cap), &cap) != BC_CL_SUCCESS ||
        a->SetKernelArg(kernel, 3, sizeof(base9_gpu.counts), &base9_gpu.counts) != BC_CL_SUCCESS)
        goto done;

    size_t local = 128;
    size_t target = (size_t)base9_gpu.compute_units * 4096u;
    if (target < local) target = local;
    size_t useful = n - 1;
    if (target > useful) target = useful;
    size_t global = ((target + local - 1) / local) * local;
    if (global < local) global = local;

    if (a->EnqueueNDRangeKernel(base9_gpu.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL) != BC_CL_SUCCESS ||
        a->EnqueueReadBuffer(base9_gpu.queue, base9_gpu.counts, BC_CL_TRUE, 0, count_bytes, counts, 0, NULL, NULL) != BC_CL_SUCCESS ||
        a->Finish(base9_gpu.queue) != BC_CL_SUCCESS)
        goto done;

    rc = 0;
    ++base9_gpu.successful_jobs;
    base9_gpu.successful_input_bytes += input_bytes;
done:
    pthread_mutex_unlock(&base9_gpu.lock);
    return rc;
}

static int base9_gpu_pair_count_u8(
    const uint8_t *input, size_t n, uint32_t *counts, size_t pair_cap
) {
    if (base9_gpu_init() != 0) return 1;
    return base9_gpu_pair_count_common(
        input, n, sizeof(*input), counts, pair_cap, base9_gpu.pair_u8);
}

static int base9_gpu_pair_count_u16(
    const uint16_t *input, size_t n, uint32_t *counts, size_t pair_cap
) {
    if (base9_gpu_init() != 0) return 1;
    return base9_gpu_pair_count_common(
        input, n, sizeof(*input), counts, pair_cap, base9_gpu.pair_u16);
}

static void base9_gpu_report_usage(void) {
    if (!base9_gpu.available || !base9_gpu.successful_jobs) return;
    printf("GPU scans:   %" PRIu64 " pair histograms, %.2f MiB via %s %s\n",
           base9_gpu.successful_jobs,
           (double)base9_gpu.successful_input_bytes / (1024.0 * 1024.0),
           base9_gpu.vendor, base9_gpu.name);
}

static void base9_gpu_print_info(void) {
    if (base9_gpu_init() == 0) {
        printf("GPU backend: OpenCL\nVendor:      %s\nDevice:      %s\nCompute:     %u units\n",
               base9_gpu.vendor, base9_gpu.name, base9_gpu.compute_units);
        if (getenv("BASECOMPRESSER_PRIVATE_OPENCL_ACTIVE")) {
            printf("Runtime:     private Intel Gen9 compatibility runtime\n");
            printf("Auto policy: CPU pair counting (validated faster on P530); set BASECOMPRESSER_GPU=force to use Intel GPU\n");
        } else {
            printf("Auto policy: GPU pair counting for scans >= 1024 KiB\n");
        }
    } else {
        printf("GPU backend: unavailable (CPU fallback active)\n");
    }
}
