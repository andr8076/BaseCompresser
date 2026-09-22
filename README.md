# BaseCompresser

Experimental **BASE9** lossless compressor built around the adaptive-base idea, redesigned for huge binary files.

BASE9 does not turn an entire file into one giant integer. It streams the file in bounded blocks, recursively finds regions with different statistics, and chooses the smallest representation it knows for each region.

Current region methods:

- **RAW** — incompressible data is copied without pointless expansion.
- **Adaptive base** — if a region uses a small byte alphabet, remap those bytes to base-N digits and pack groups with native 64-bit arithmetic.
- **Token base** — discover repeated 2–8 byte sequences, turn them into super-symbols, and pack the shorter symbol stream as digits in an adaptive base.
- **rANS** — static frequency coding lets a region compress even when all 256 byte values occur, as long as their frequencies are uneven.
- **Delta + rANS** — a reversible first-order delta transform exposes structure in counters, samples, geometry, and other locally correlated byte streams before frequency coding.
- **Order-1 context rANS** — models the next-byte distribution separately for each previous byte. This can compress data whose global byte histogram is almost uniform when byte-to-byte transitions remain predictable.

The encoder starts with large blocks (64 MiB by default) and uses a dynamic split plan down to 256 KiB regions. A split is kept only when the estimated encoded representation of the children beats the parent after metadata/header costs.

## Build

```bash
make
make test
```

The default release build uses `-O3 -march=native -flto` and pthread block parallelism.

## Use

```bash
./basecompresser encode huge-file.bin
./basecompresser decode huge-file.bin.base9
```

Useful controls:

```bash
./basecompresser encode input.bin --block-mib 64 --min-region-kib 256 --threads 16
```

If `--threads` is omitted, BASE9 chooses a worker count from CPU availability and available RAM. Memory scales with block size × active workers, not total file size.

## GPU acceleration

BASE9 has an optional OpenCL backend shared by NVIDIA, AMD, and Intel GPUs.
It accelerates two operations that fit the adaptive-base design:

- **Token pair histograms** — the default GPU path uses workgroup-sharded
  histograms to reduce global atomic contention.
- **Higher-base radix packing** — once the winning token vocabulary is known,
  independent radix groups can be packed in parallel on the GPU.

Token selection itself remains deterministic on the CPU, and every GPU path has
an exact CPU fallback. CPU and GPU encodes are regression-tested to produce
byte-for-byte identical `.base9` files.

```bash
./basecompresser gpu-info
```

Useful controls:

```bash
BASECOMPRESSER_GPU=off ./basecompresser encode input.bin
BASECOMPRESSER_GPU=force ./basecompresser encode input.bin
BASECOMPRESSER_GPU_VENDOR=nvidia ./basecompresser encode input.bin
BASECOMPRESSER_GPU_VENDOR=amd ./basecompresser encode input.bin
BASECOMPRESSER_GPU_VENDOR=intel ./basecompresser encode input.bin
BASECOMPRESSER_GPU_HIST=sharded ./basecompresser encode input.bin
BASECOMPRESSER_GPU_HIST=direct ./basecompresser encode input.bin
BASECOMPRESSER_GPU_HIST=cpu ./basecompresser encode input.bin
BASECOMPRESSER_GPU_PACK=off ./basecompresser encode input.bin
```

`sharded` is the normal histogram implementation. `direct` keeps the original
single global-atomic histogram for comparison, while `tiled` remains an
experimental local-memory implementation. Large discrete GPUs have their
default shard count bounded so histogram scratch memory cannot grow without
limit; advanced testing can override it with `BASECOMPRESSER_GPU_GROUPS`.

For the same Skylake/Gen9 hosts supported by 265Encode's legacy Intel path,
BaseCompresser can prepare a private OpenCL compute runtime without installing
packages into `/usr` or `/etc`:

```bash
./tools/setup-intel-legacy-opencl.sh
./basecompresser gpu-info
```

The Intel HD P530 path was measured directly. Sharded histograms are
substantially faster than the original GPU global-atomic kernel, and GPU radix
packing is close to CPU speed, but the complete CPU path remains faster on this
legacy iGPU. AUTO therefore stays CPU-first on P530; forcing the GPU remains
available for testing and future kernels. Normal NVIDIA/AMD OpenCL devices use
the GPU automatically for sufficiently large work.

## Why BASE9 differs from BASE8

BASE8 only gained from a reduced alphabet. If a block contained all 256 byte values it normally fell back to RAW, even when one value was vastly more common than another.

BASE9 adds probability/frequency coding. A region can therefore use all 256 byte values and still compress when the distribution is predictable. Adaptive splitting also prevents a large heterogeneous block from hiding smaller compressible regions.

## Format status

The BASE9 container is experimental and versioned. Current encodes use format version 3; the decoder remains backward-compatible with versions 1 and 2. Exact round-trip integrity is protected with CRC32 at both region and block level. The format may change while the compression model is being developed.

## Token-base direction

Token-base is the project-specific path: repeated byte sequences become digits.
The current implementation uses three bounded pair-merge rounds, so useful
2-byte tokens can recursively become 4-byte and then 8-byte super-symbols.
Token-base is accepted only when its dictionary plus radix-packed payload beats
the best competing representation for that region. Candidate merge rounds are
sized exactly without generating temporary payloads; only the winning round is
actually radix-packed. This keeps the higher-base method identical while
avoiding repeated conversion work.

## Next performance/compression work

- token-aware region split costs and content-defined token boundaries
- batch/pipeline GPU work across independent blocks so transfer and CPU analysis overlap
- SIMD CPU token scans for systems where GPU offload is not profitable
- wider token dictionaries and longer super-symbols when they prove profitable
- SIMD histogram and transform kernels (AVX2 first)
- persistent worker pool instead of one pthread batch per set of blocks
- denser rANS model serialization
- integrate context-cost estimates directly into the adaptive split planner
- denser/higher-resolution context probability models
- higher-order and mixed context models
- content-defined region boundaries rather than boundaries limited to the minimum-region grid
- benchmark corpus and regression thresholds
