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
#define BC_CL_MEM_USE_HOST_PTR (1ULL << 3)
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
    bc_cl_kernel pair_u8_sharded;
    bc_cl_kernel pair_u16_sharded;
    bc_cl_kernel pair_u8_tiled;
    bc_cl_kernel pair_u16_tiled;
    bc_cl_kernel clear_u32;
    bc_cl_kernel reduce_hist;
    bc_cl_kernel pack_radix_u16;
    bc_cl_mem input;
    size_t input_cap;
    bc_cl_mem counts;
    size_t counts_cap;
    bc_cl_mem partial_counts;
    size_t partial_counts_cap;
    bc_cl_mem pack_map;
    size_t pack_map_cap;
    bc_cl_mem pack_output;
    size_t pack_output_cap;
    bc_cl_uint compute_units;
    char vendor[128];
    char name[256];
    int available;
    int permanently_failed;
    uint64_t successful_jobs;
    uint64_t successful_input_bytes;
    uint64_t successful_pack_jobs;
    uint64_t successful_pack_input_bytes;
    pthread_mutex_t lock;
} Base9GpuState;

static Base9GpuState base9_gpu = {
    .lock = PTHREAD_MUTEX_INITIALIZER
};
static pthread_once_t base9_gpu_once = PTHREAD_ONCE_INIT;

static const char *base9_opencl_hist_source =
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
"}\n"
"__kernel void clear_u32(__global uint *buf, ulong n) {\n"
"  size_t gid=get_global_id(0), stride=get_global_size(0);\n"
"  for (ulong i=(ulong)gid; i<n; i+=(ulong)stride) buf[i]=0u;\n"
"}\n"
"__kernel void pair_u8_sharded(__global const uchar *in, ulong n, uint cap, uint bins, __global uint *partial) {\n"
"  size_t gid=get_global_id(0), stride=get_global_size(0);\n"
"  size_t group=get_group_id(0);\n"
"  __global uint *row=partial + group*(size_t)bins;\n"
"  for (ulong i=(ulong)gid; i+1<n; i+=(ulong)stride) {\n"
"    uint a=(uint)in[i], b=(uint)in[i+1];\n"
"    atomic_inc((volatile __global unsigned int *)&row[a*cap+b]);\n"
"  }\n"
"}\n"
"__kernel void pair_u16_sharded(__global const ushort *in, ulong n, uint cap, uint bins, __global uint *partial) {\n"
"  size_t gid=get_global_id(0), stride=get_global_size(0);\n"
"  size_t group=get_group_id(0);\n"
"  __global uint *row=partial + group*(size_t)bins;\n"
"  for (ulong i=(ulong)gid; i+1<n; i+=(ulong)stride) {\n"
"    uint a=(uint)in[i], b=(uint)in[i+1];\n"
"    if (a<cap && b<cap) atomic_inc((volatile __global unsigned int *)&row[a*cap+b]);\n"
"  }\n"
"}\n"
"__kernel void reduce_hist(__global const uint *partial, uint bins, uint groups, __global uint *out) {\n"
"  size_t gid=get_global_id(0), stride=get_global_size(0);\n"
"  for (size_t bin=gid; bin<(size_t)bins; bin+=stride) {\n"
"    uint sum=0u;\n"
"    for (uint g=0; g<groups; ++g) sum += partial[(size_t)g*bins+bin];\n"
"    out[bin]=sum;\n"
"  }\n"
"}\n"
"#define BC_TILE_ROWS 16u\n"
"__kernel void pair_u8_tiled(__global const uchar *in, ulong n, uint cap, uint rows, __global uint *out, __local uint *hist) {\n"
"  uint lid=(uint)get_local_id(0), lsize=(uint)get_local_size(0);\n"
"  uint tile=(uint)get_group_id(0), row0=tile*BC_TILE_ROWS;\n"
"  uint local_bins=BC_TILE_ROWS*cap;\n"
"  for (uint j=lid; j<local_bins; j+=lsize) hist[j]=0u;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for (ulong i=(ulong)lid; i+1<n; i+=(ulong)lsize) {\n"
"    uint a=(uint)in[i], b=(uint)in[i+1];\n"
"    if (a>=row0 && a<row0+BC_TILE_ROWS && a<rows)\n"
"      atomic_inc((volatile __local unsigned int *)&hist[(a-row0)*cap+b]);\n"
"  }\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for (uint j=lid; j<local_bins; j+=lsize) {\n"
"    uint r=j/cap, c=j-r*cap, global_r=row0+r;\n"
"    if (global_r<rows) out[global_r*cap+c]=hist[j];\n"
"  }\n"
"}\n"
"__kernel void pair_u16_tiled(__global const ushort *in, ulong n, uint cap, uint rows, __global uint *out, __local uint *hist) {\n"
"  uint lid=(uint)get_local_id(0), lsize=(uint)get_local_size(0);\n"
"  uint tile=(uint)get_group_id(0), row0=tile*BC_TILE_ROWS;\n"
"  uint local_bins=BC_TILE_ROWS*cap;\n"
"  for (uint j=lid; j<local_bins; j+=lsize) hist[j]=0u;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for (ulong i=(ulong)lid; i+1<n; i+=(ulong)lsize) {\n"
"    uint a=(uint)in[i], b=(uint)in[i+1];\n"
"    if (a>=row0 && a<row0+BC_TILE_ROWS && a<rows && b<cap)\n"
"      atomic_inc((volatile __local unsigned int *)&hist[(a-row0)*cap+b]);\n"
"  }\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for (uint j=lid; j<local_bins; j+=lsize) {\n"
"    uint r=j/cap, c=j-r*cap, global_r=row0+r;\n"
"    if (global_r<rows) out[global_r*cap+c]=hist[j];\n"
"  }\n"
"}\n";
static const char *base9_opencl_pack_source =
"__kernel void pack_radix_u16(__global const ushort *symbols, ulong n, __global const ushort *map, uint map_len, uint base, uint gd, uint gb, uint rem_bytes, __global uchar *out) {\n"
"  ulong group=(ulong)get_global_id(0);\n"
"  ulong full=n/(ulong)gd, rem=n%(ulong)gd;\n"
"  ulong groups=full+(rem?1ul:0ul);\n"
"  if (group>=groups) return;\n"
"  ulong pos=group*(ulong)gd;\n"
"  uint take=(group<full)?gd:(uint)rem;\n"
"  ulong value=0ul;\n"
"  for (uint j=0; j<take; ++j) {\n"
"    uint sym=(uint)symbols[pos+j];\n"
"    if (sym>=map_len) return;\n"
"    value=value*(ulong)base+(ulong)map[sym];\n"
"  }\n"
"  uint bytes=(group<full)?gb:rem_bytes;\n"
"  ulong off=group*(ulong)gb;\n"
"  for (uint j=0; j<bytes; ++j) out[off+j]=(uchar)(value>>(8u*j));\n"
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

    const char *sources[2] = {
        base9_opencl_hist_source,
        base9_opencl_pack_source
    };
    size_t source_lengths[2] = {
        strlen(base9_opencl_hist_source),
        strlen(base9_opencl_pack_source)
    };
    base9_gpu.program = a->CreateProgramWithSource(
        base9_gpu.context, 2, sources, source_lengths, &err);
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
    base9_gpu.pair_u8_sharded = a->CreateKernel(base9_gpu.program, "pair_u8_sharded", &err);
    if (!base9_gpu.pair_u8_sharded || err != BC_CL_SUCCESS) goto fail;
    base9_gpu.pair_u16_sharded = a->CreateKernel(base9_gpu.program, "pair_u16_sharded", &err);
    if (!base9_gpu.pair_u16_sharded || err != BC_CL_SUCCESS) goto fail;
    base9_gpu.pair_u8_tiled = a->CreateKernel(base9_gpu.program, "pair_u8_tiled", &err);
    if (!base9_gpu.pair_u8_tiled || err != BC_CL_SUCCESS) goto fail;
    base9_gpu.pair_u16_tiled = a->CreateKernel(base9_gpu.program, "pair_u16_tiled", &err);
    if (!base9_gpu.pair_u16_tiled || err != BC_CL_SUCCESS) goto fail;
    base9_gpu.clear_u32 = a->CreateKernel(base9_gpu.program, "clear_u32", &err);
    if (!base9_gpu.clear_u32 || err != BC_CL_SUCCESS) goto fail;
    base9_gpu.reduce_hist = a->CreateKernel(base9_gpu.program, "reduce_hist", &err);
    if (!base9_gpu.reduce_hist || err != BC_CL_SUCCESS) goto fail;
    base9_gpu.pack_radix_u16 = a->CreateKernel(base9_gpu.program, "pack_radix_u16", &err);
    if (!base9_gpu.pack_radix_u16 || err != BC_CL_SUCCESS) goto fail;

    base9_gpu.loader = lib;
    base9_gpu.device = chosen;
    base9_gpu.compute_units = chosen_cu ? chosen_cu : 1;
    snprintf(base9_gpu.vendor, sizeof(base9_gpu.vendor), "%s", chosen_vendor);
    snprintf(base9_gpu.name, sizeof(base9_gpu.name), "%s", chosen_name);
    base9_gpu.available = 1;
    return;

fail:
    if (base9_gpu.pack_radix_u16 && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.pack_radix_u16);
    if (base9_gpu.reduce_hist && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.reduce_hist);
    if (base9_gpu.clear_u32 && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.clear_u32);
    if (base9_gpu.pair_u16_tiled && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.pair_u16_tiled);
    if (base9_gpu.pair_u8_tiled && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.pair_u8_tiled);
    if (base9_gpu.pair_u16_sharded && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.pair_u16_sharded);
    if (base9_gpu.pair_u8_sharded && a->ReleaseKernel) a->ReleaseKernel(base9_gpu.pair_u8_sharded);
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

static int base9_gpu_ensure_buffers(size_t input_bytes, size_t count_bytes, size_t partial_bytes) {
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
    if (partial_bytes > base9_gpu.partial_counts_cap) {
        if (base9_gpu.partial_counts) a->ReleaseMemObject(base9_gpu.partial_counts);
        base9_gpu.partial_counts = a->CreateBuffer(base9_gpu.context, BC_CL_MEM_READ_WRITE, partial_bytes, NULL, &err);
        if (!base9_gpu.partial_counts || err != BC_CL_SUCCESS) return -1;
        base9_gpu.partial_counts_cap = partial_bytes;
    }
    return 0;
}

static bool base9_gpu_zero_copy_input(void) {
    const char *mode = getenv("BASECOMPRESSER_GPU_ZERO_COPY");
    if (mode && *mode) {
        if (!strcasecmp(mode, "on") || !strcasecmp(mode, "yes") ||
            !strcmp(mode, "1") || !strcasecmp(mode, "true"))
            return true;
        if (!strcasecmp(mode, "off") || !strcasecmp(mode, "no") ||
            !strcmp(mode, "0") || !strcasecmp(mode, "false"))
            return false;
    }
    return false;
}

enum Base9GpuHistMode {
    BASE9_GPU_HIST_DIRECT = 0,
    BASE9_GPU_HIST_SHARDED = 1,
    BASE9_GPU_HIST_TILED = 2,
    BASE9_GPU_HIST_CPU = 3,
};

static enum Base9GpuHistMode base9_gpu_hist_mode(void) {
    const char *mode = getenv("BASECOMPRESSER_GPU_HIST");
    if (!mode || !*mode || !strcasecmp(mode, "auto") || !strcasecmp(mode, "sharded"))
        return BASE9_GPU_HIST_SHARDED;
    if (!strcasecmp(mode, "direct") || !strcasecmp(mode, "global"))
        return BASE9_GPU_HIST_DIRECT;
    if (!strcasecmp(mode, "tiled") || !strcasecmp(mode, "local"))
        return BASE9_GPU_HIST_TILED;
    if (!strcasecmp(mode, "cpu") || !strcasecmp(mode, "off"))
        return BASE9_GPU_HIST_CPU;
    return BASE9_GPU_HIST_SHARDED;
}

static size_t base9_gpu_local_size(void) {
    size_t local = 128;
    const char *v = getenv("BASECOMPRESSER_GPU_LOCAL_SIZE");
    if (v && *v) {
        unsigned long parsed = strtoul(v, NULL, 10);
        if (parsed == 32 || parsed == 64 || parsed == 128 || parsed == 256)
            local = (size_t)parsed;
    }
    return local;
}

static unsigned base9_gpu_shard_groups(void) {
    const char *absolute = getenv("BASECOMPRESSER_GPU_GROUPS");
    if (absolute && *absolute) {
        unsigned long parsed = strtoul(absolute, NULL, 10);
        if (parsed >= 1 && parsed <= 1024)
            return (unsigned)parsed;
    }

    if (getenv("BASECOMPRESSER_PRIVATE_OPENCL_ACTIVE"))
        return 8;

    unsigned mult = 2;
    const char *v = getenv("BASECOMPRESSER_GPU_SHARDS_PER_CU");
    if (v && *v) {
        unsigned long parsed = strtoul(v, NULL, 10);
        if (parsed >= 1 && parsed <= 16) mult = (unsigned)parsed;
    }
    uint64_t groups = (uint64_t)(base9_gpu.compute_units ? base9_gpu.compute_units : 1) * mult;
    if (groups < 1) groups = 1;
    if (groups > 128) groups = 128;
    return (unsigned)groups;
}

static int base9_gpu_pair_count_common(
    const void *input,
    size_t n,
    size_t elem_size,
    uint32_t *counts,
    size_t pair_cap,
    size_t active_rows,
    bc_cl_kernel direct_kernel,
    bc_cl_kernel sharded_kernel,
    bc_cl_kernel tiled_kernel
) {
    if (n < 2 || !counts || pair_cap == 0 || pair_cap > UINT32_MAX ||
        active_rows == 0 || active_rows > pair_cap)
        return 1;

    /* The private P530 path remains opt-in in AUTO until the optimized
       histogram actually wins there. Forced mode is used by benchmarking and
       the equivalence test. */
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
    enum Base9GpuHistMode hist_mode = base9_gpu_hist_mode();
    if (hist_mode == BASE9_GPU_HIST_CPU)
        return 1;
    unsigned groups = hist_mode == BASE9_GPU_HIST_SHARDED ?
                      base9_gpu_shard_groups() : 0;

    size_t partial_entries = 0, partial_bytes = 0;
    if (hist_mode == BASE9_GPU_HIST_SHARDED) {
        if (count_entries > SIZE_MAX / groups)
            return 1;
        partial_entries = count_entries * groups;
        if (partial_entries > SIZE_MAX / sizeof(uint32_t))
            return 1;
        partial_bytes = partial_entries * sizeof(uint32_t);
    }

    pthread_mutex_lock(&base9_gpu.lock);
    int rc = 1;
    Base9OpenCLApi *a = &base9_gpu.api;
    bc_cl_mem host_input = NULL;
    bool zero_copy = base9_gpu_zero_copy_input();
    if (base9_gpu_ensure_buffers(zero_copy ? 0 : input_bytes,
                                 count_bytes, partial_bytes) != 0)
        goto done;

    bc_cl_mem input_mem = base9_gpu.input;
    if (zero_copy) {
        bc_cl_int create_err = BC_CL_SUCCESS;
        host_input = a->CreateBuffer(
            base9_gpu.context, BC_CL_MEM_READ_ONLY | BC_CL_MEM_USE_HOST_PTR,
            input_bytes, (void *)input, &create_err);
        if (!host_input || create_err != BC_CL_SUCCESS)
            goto done;
        input_mem = host_input;
    } else if (a->EnqueueWriteBuffer(
                   base9_gpu.queue, base9_gpu.input, BC_CL_TRUE, 0,
                   input_bytes, input, 0, NULL, NULL) != BC_CL_SUCCESS) {
        goto done;
    }

    size_t local = base9_gpu_local_size();

    if (hist_mode == BASE9_GPU_HIST_DIRECT) {
        memset(counts, 0, count_bytes);
        if (a->EnqueueWriteBuffer(base9_gpu.queue, base9_gpu.counts, BC_CL_TRUE, 0,
                                  count_bytes, counts, 0, NULL, NULL) != BC_CL_SUCCESS)
            goto done;

        bc_cl_ulong nn = (bc_cl_ulong)n;
        bc_cl_uint cap = (bc_cl_uint)pair_cap;
        if (a->SetKernelArg(direct_kernel, 0, sizeof(input_mem), &input_mem) != BC_CL_SUCCESS ||
            a->SetKernelArg(direct_kernel, 1, sizeof(nn), &nn) != BC_CL_SUCCESS ||
            a->SetKernelArg(direct_kernel, 2, sizeof(cap), &cap) != BC_CL_SUCCESS ||
            a->SetKernelArg(direct_kernel, 3, sizeof(base9_gpu.counts), &base9_gpu.counts) != BC_CL_SUCCESS)
            goto done;

        size_t target = (size_t)base9_gpu.compute_units * 4096u;
        if (target < local) target = local;
        size_t useful = n - 1;
        if (target > useful) target = useful;
        size_t global = ((target + local - 1) / local) * local;
        if (global < local) global = local;

        if (a->EnqueueNDRangeKernel(base9_gpu.queue, direct_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL) != BC_CL_SUCCESS)
            goto done;
    } else if (hist_mode == BASE9_GPU_HIST_SHARDED) {
        bc_cl_ulong clear_n = (bc_cl_ulong)partial_entries;
        if (a->SetKernelArg(base9_gpu.clear_u32, 0, sizeof(base9_gpu.partial_counts), &base9_gpu.partial_counts) != BC_CL_SUCCESS ||
            a->SetKernelArg(base9_gpu.clear_u32, 1, sizeof(clear_n), &clear_n) != BC_CL_SUCCESS)
            goto done;
        size_t clear_global = (size_t)base9_gpu.compute_units * 1024u;
        if (clear_global < local) clear_global = local;
        clear_global = ((clear_global + local - 1) / local) * local;
        if (a->EnqueueNDRangeKernel(base9_gpu.queue, base9_gpu.clear_u32, 1, NULL,
                                    &clear_global, &local, 0, NULL, NULL) != BC_CL_SUCCESS)
            goto done;

        bc_cl_ulong nn = (bc_cl_ulong)n;
        bc_cl_uint cap = (bc_cl_uint)pair_cap;
        bc_cl_uint bins = (bc_cl_uint)count_entries;
        if (a->SetKernelArg(sharded_kernel, 0, sizeof(input_mem), &input_mem) != BC_CL_SUCCESS ||
            a->SetKernelArg(sharded_kernel, 1, sizeof(nn), &nn) != BC_CL_SUCCESS ||
            a->SetKernelArg(sharded_kernel, 2, sizeof(cap), &cap) != BC_CL_SUCCESS ||
            a->SetKernelArg(sharded_kernel, 3, sizeof(bins), &bins) != BC_CL_SUCCESS ||
            a->SetKernelArg(sharded_kernel, 4, sizeof(base9_gpu.partial_counts), &base9_gpu.partial_counts) != BC_CL_SUCCESS)
            goto done;

        size_t global = (size_t)groups * local;
        if (a->EnqueueNDRangeKernel(base9_gpu.queue, sharded_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL) != BC_CL_SUCCESS)
            goto done;

        bc_cl_uint group_count = groups;
        if (a->SetKernelArg(base9_gpu.reduce_hist, 0, sizeof(base9_gpu.partial_counts), &base9_gpu.partial_counts) != BC_CL_SUCCESS ||
            a->SetKernelArg(base9_gpu.reduce_hist, 1, sizeof(bins), &bins) != BC_CL_SUCCESS ||
            a->SetKernelArg(base9_gpu.reduce_hist, 2, sizeof(group_count), &group_count) != BC_CL_SUCCESS ||
            a->SetKernelArg(base9_gpu.reduce_hist, 3, sizeof(base9_gpu.counts), &base9_gpu.counts) != BC_CL_SUCCESS)
            goto done;

        size_t reduce_global = ((count_entries + local - 1) / local) * local;
        if (a->EnqueueNDRangeKernel(base9_gpu.queue, base9_gpu.reduce_hist, 1, NULL,
                                    &reduce_global, &local, 0, NULL, NULL) != BC_CL_SUCCESS)
            goto done;
    } else {
        bc_cl_ulong nn = (bc_cl_ulong)n;
        bc_cl_uint cap = (bc_cl_uint)pair_cap;
        bc_cl_uint rows = (bc_cl_uint)active_rows;
        size_t local_hist_bytes = 16u * pair_cap * sizeof(uint32_t);
        if (a->SetKernelArg(tiled_kernel, 0, sizeof(input_mem), &input_mem) != BC_CL_SUCCESS ||
            a->SetKernelArg(tiled_kernel, 1, sizeof(nn), &nn) != BC_CL_SUCCESS ||
            a->SetKernelArg(tiled_kernel, 2, sizeof(cap), &cap) != BC_CL_SUCCESS ||
            a->SetKernelArg(tiled_kernel, 3, sizeof(rows), &rows) != BC_CL_SUCCESS ||
            a->SetKernelArg(tiled_kernel, 4, sizeof(base9_gpu.counts), &base9_gpu.counts) != BC_CL_SUCCESS ||
            a->SetKernelArg(tiled_kernel, 5, local_hist_bytes, NULL) != BC_CL_SUCCESS)
            goto done;

        size_t tiles = (active_rows + 15u) / 16u;
        size_t global = tiles * local;
        if (a->EnqueueNDRangeKernel(base9_gpu.queue, tiled_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL) != BC_CL_SUCCESS)
            goto done;
    }

    if (a->EnqueueReadBuffer(base9_gpu.queue, base9_gpu.counts, BC_CL_TRUE, 0,
                             count_bytes, counts, 0, NULL, NULL) != BC_CL_SUCCESS ||
        a->Finish(base9_gpu.queue) != BC_CL_SUCCESS)
        goto done;

    rc = 0;
    ++base9_gpu.successful_jobs;
    base9_gpu.successful_input_bytes += input_bytes;
done:
    if (host_input)
        a->ReleaseMemObject(host_input);
    pthread_mutex_unlock(&base9_gpu.lock);
    return rc;
}

static int base9_gpu_pair_count_u8(
    const uint8_t *input, size_t n, uint32_t *counts, size_t pair_cap
) {
    if (base9_gpu_init() != 0) return 1;
    size_t rows = pair_cap < 256u ? pair_cap : 256u;
    return base9_gpu_pair_count_common(
        input, n, sizeof(*input), counts, pair_cap, rows,
        base9_gpu.pair_u8, base9_gpu.pair_u8_sharded, base9_gpu.pair_u8_tiled);
}

static int base9_gpu_pair_count_u16(
    const uint16_t *input, size_t n, uint32_t *counts, size_t pair_cap
) {
    if (base9_gpu_init() != 0) return 1;
    return base9_gpu_pair_count_common(
        input, n, sizeof(*input), counts, pair_cap, pair_cap,
        base9_gpu.pair_u16, base9_gpu.pair_u16_sharded, base9_gpu.pair_u16_tiled);
}

static bool base9_gpu_pack_enabled(size_t input_bytes) {
    const char *mode = getenv("BASECOMPRESSER_GPU_PACK");
    if (mode && *mode) {
        if (!strcasecmp(mode, "off") || !strcasecmp(mode, "no") ||
            !strcmp(mode, "0") || !strcasecmp(mode, "false"))
            return false;
        if (!strcasecmp(mode, "force") || !strcasecmp(mode, "on") ||
            !strcmp(mode, "1") || !strcasecmp(mode, "true"))
            return true;
    }
    if (base9_gpu_env_off())
        return false;
    if (base9_gpu_env_force())
        return true;

    size_t min_bytes = 4u * 1024u * 1024u;
    const char *threshold = getenv("BASECOMPRESSER_GPU_PACK_MIN_KIB");
    if (threshold && *threshold) {
        unsigned long kib = strtoul(threshold, NULL, 10);
        if (kib && kib <= SIZE_MAX / 1024u)
            min_bytes = (size_t)kib * 1024u;
    }
    return input_bytes >= min_bytes;
}

static int base9_gpu_ensure_pack_buffers(size_t map_bytes, size_t output_bytes) {
    Base9OpenCLApi *a = &base9_gpu.api;
    bc_cl_int err = BC_CL_SUCCESS;
    if (map_bytes > base9_gpu.pack_map_cap) {
        if (base9_gpu.pack_map) a->ReleaseMemObject(base9_gpu.pack_map);
        base9_gpu.pack_map = a->CreateBuffer(
            base9_gpu.context, BC_CL_MEM_READ_ONLY, map_bytes, NULL, &err);
        if (!base9_gpu.pack_map || err != BC_CL_SUCCESS) return -1;
        base9_gpu.pack_map_cap = map_bytes;
    }
    if (output_bytes > base9_gpu.pack_output_cap) {
        if (base9_gpu.pack_output) a->ReleaseMemObject(base9_gpu.pack_output);
        base9_gpu.pack_output = a->CreateBuffer(
            base9_gpu.context, BC_CL_MEM_READ_WRITE, output_bytes, NULL, &err);
        if (!base9_gpu.pack_output || err != BC_CL_SUCCESS) return -1;
        base9_gpu.pack_output_cap = output_bytes;
    }
    return 0;
}

static int base9_gpu_pack_radix_u16(
    const uint16_t *symbols,
    size_t symbol_count,
    const uint16_t *digit_map,
    size_t map_len,
    unsigned base,
    unsigned gd,
    unsigned gb,
    uint8_t *output,
    size_t output_bytes
) {
    if (!symbols || !digit_map || !output || !symbol_count ||
        map_len == 0 || map_len > UINT32_MAX ||
        base <= 1 || gd == 0 || gb == 0 || gb > 8)
        return 1;
    if (symbol_count > SIZE_MAX / sizeof(*symbols) ||
        map_len > SIZE_MAX / sizeof(*digit_map))
        return 1;

    size_t input_bytes = symbol_count * sizeof(*symbols);
    size_t map_bytes = map_len * sizeof(*digit_map);
    if (!base9_gpu_pack_enabled(input_bytes) || base9_gpu_init() != 0)
        return 1;

    size_t full = symbol_count / gd;
    unsigned rem = (unsigned)(symbol_count % gd);
    size_t groups = full + (rem ? 1u : 0u);
    unsigned rem_bytes = rem ? group_width_bytes(base, rem) : 0u;
    size_t expected = full * (size_t)gb + rem_bytes;
    if (expected != output_bytes || !groups)
        return 1;

    pthread_mutex_lock(&base9_gpu.lock);
    int rc = 1;
    Base9OpenCLApi *a = &base9_gpu.api;

    if (base9_gpu_ensure_buffers(input_bytes, 0, 0) != 0 ||
        base9_gpu_ensure_pack_buffers(map_bytes, output_bytes) != 0)
        goto done;

    if (a->EnqueueWriteBuffer(
            base9_gpu.queue, base9_gpu.input, BC_CL_TRUE, 0,
            input_bytes, symbols, 0, NULL, NULL) != BC_CL_SUCCESS ||
        a->EnqueueWriteBuffer(
            base9_gpu.queue, base9_gpu.pack_map, BC_CL_TRUE, 0,
            map_bytes, digit_map, 0, NULL, NULL) != BC_CL_SUCCESS)
        goto done;

    bc_cl_ulong nn = (bc_cl_ulong)symbol_count;
    bc_cl_uint map_n = (bc_cl_uint)map_len;
    bc_cl_uint b = base, group_digits = gd, group_bytes = gb;
    bc_cl_uint tail_bytes = rem_bytes;

    if (a->SetKernelArg(base9_gpu.pack_radix_u16, 0,
                        sizeof(base9_gpu.input), &base9_gpu.input) != BC_CL_SUCCESS ||
        a->SetKernelArg(base9_gpu.pack_radix_u16, 1,
                        sizeof(nn), &nn) != BC_CL_SUCCESS ||
        a->SetKernelArg(base9_gpu.pack_radix_u16, 2,
                        sizeof(base9_gpu.pack_map), &base9_gpu.pack_map) != BC_CL_SUCCESS ||
        a->SetKernelArg(base9_gpu.pack_radix_u16, 3,
                        sizeof(map_n), &map_n) != BC_CL_SUCCESS ||
        a->SetKernelArg(base9_gpu.pack_radix_u16, 4,
                        sizeof(b), &b) != BC_CL_SUCCESS ||
        a->SetKernelArg(base9_gpu.pack_radix_u16, 5,
                        sizeof(group_digits), &group_digits) != BC_CL_SUCCESS ||
        a->SetKernelArg(base9_gpu.pack_radix_u16, 6,
                        sizeof(group_bytes), &group_bytes) != BC_CL_SUCCESS ||
        a->SetKernelArg(base9_gpu.pack_radix_u16, 7,
                        sizeof(tail_bytes), &tail_bytes) != BC_CL_SUCCESS ||
        a->SetKernelArg(base9_gpu.pack_radix_u16, 8,
                        sizeof(base9_gpu.pack_output), &base9_gpu.pack_output) != BC_CL_SUCCESS)
        goto done;

    size_t local = base9_gpu_local_size();
    size_t global = ((groups + local - 1) / local) * local;
    if (a->EnqueueNDRangeKernel(
            base9_gpu.queue, base9_gpu.pack_radix_u16, 1,
            NULL, &global, &local, 0, NULL, NULL) != BC_CL_SUCCESS ||
        a->EnqueueReadBuffer(
            base9_gpu.queue, base9_gpu.pack_output, BC_CL_TRUE, 0,
            output_bytes, output, 0, NULL, NULL) != BC_CL_SUCCESS ||
        a->Finish(base9_gpu.queue) != BC_CL_SUCCESS)
        goto done;

    rc = 0;
    ++base9_gpu.successful_pack_jobs;
    base9_gpu.successful_pack_input_bytes += input_bytes;

done:
    pthread_mutex_unlock(&base9_gpu.lock);
    return rc;
}

static void base9_gpu_report_usage(void) {
    if (!base9_gpu.available) return;
    if (base9_gpu.successful_jobs) {
        printf("GPU scans:   %" PRIu64 " pair histograms, %.2f MiB via %s %s\n",
               base9_gpu.successful_jobs,
               (double)base9_gpu.successful_input_bytes / (1024.0 * 1024.0),
               base9_gpu.vendor, base9_gpu.name);
    }
    if (base9_gpu.successful_pack_jobs) {
        printf("GPU radix:   %" PRIu64 " higher-base packs, %.2f MiB symbols via %s %s\n",
               base9_gpu.successful_pack_jobs,
               (double)base9_gpu.successful_pack_input_bytes / (1024.0 * 1024.0),
               base9_gpu.vendor, base9_gpu.name);
    }
}

static void base9_gpu_print_info(void) {
    if (base9_gpu_init() == 0) {
        printf("GPU backend: OpenCL\nVendor:      %s\nDevice:      %s\nCompute:     %u units\n",
               base9_gpu.vendor, base9_gpu.name, base9_gpu.compute_units);
        if (getenv("BASECOMPRESSER_PRIVATE_OPENCL_ACTIVE")) {
            printf("Runtime:     private Intel Gen9 compatibility runtime\n");
            printf("Auto policy: CPU on validated P530; BASECOMPRESSER_GPU=force enables sharded histograms + GPU radix packing\n");
        } else {
            printf("Auto policy: sharded GPU histograms for scans >= 1024 KiB; GPU radix packing for symbol streams >= 4096 KiB\n");
        }
    } else {
        printf("GPU backend: unavailable (CPU fallback active)\n");
    }
}
