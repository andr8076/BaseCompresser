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

## GPU token discovery

BASE9 can accelerate token pair histograms with OpenCL while keeping token
selection and adaptive-base packing on the CPU. The same kernel path is used
for NVIDIA, AMD, and Intel OpenCL devices. GPU support is optional: if no
usable OpenCL GPU exists, compression automatically keeps using the CPU.

```bash
./basecompresser gpu-info
```

Useful environment controls:

```bash
BASECOMPRESSER_GPU=off ./basecompresser encode input.bin
BASECOMPRESSER_GPU=force ./basecompresser encode input.bin
BASECOMPRESSER_GPU_VENDOR=nvidia ./basecompresser encode input.bin
BASECOMPRESSER_GPU_VENDOR=amd ./basecompresser encode input.bin
BASECOMPRESSER_GPU_VENDOR=intel ./basecompresser encode input.bin
BASECOMPRESSER_GPU_MIN_KIB=2048 ./basecompresser encode input.bin
```

AUTO uses GPU pair counting for sufficiently large scans on normal OpenCL
devices. The validated Skylake/P530 private-compatibility path is intentionally
CPU-first in AUTO because its current global-atomic histogram kernel measured
slower than the CPU implementation; `BASECOMPRESSER_GPU=force` still enables
it for testing or future kernels.

For the same Skylake/Gen9 hosts supported by 265Encode's legacy Intel path,
BaseCompresser can prepare a private OpenCL compute runtime without installing
packages into `/usr` or `/etc`:

```bash
./tools/setup-intel-legacy-opencl.sh
./basecompresser gpu-info
```

The helper uses the same Intel Gen9/i915 PCI-device detection policy as
265Encode and stores the extracted runtime under the user's cache directory.

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
the best competing representation for that region.

## Next performance/compression work

- token-aware region split costs and content-defined token boundaries
- faster token discovery / pair counting (SIMD and persistent scratch buffers)
- wider token dictionaries and longer super-symbols when they prove profitable
- SIMD histogram and transform kernels (AVX2 first)
- persistent worker pool instead of one pthread batch per set of blocks
- denser rANS model serialization
- integrate context-cost estimates directly into the adaptive split planner
- denser/higher-resolution context probability models
- higher-order and mixed context models
- content-defined region boundaries rather than boundaries limited to the minimum-region grid
- benchmark corpus and regression thresholds
