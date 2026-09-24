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
make portable
make test-portable
```

The default build uses `-O3 -march=native -flto` and pthread block parallelism. It targets the CPU used to build it. `make portable` creates `basecompresser-portable` without `-march=native` for distribution to older CPUs of the same operating-system and CPU family; `make test-portable` runs the round-trip and malformed-input suite against that binary.

## Use

```bash
./basecompresser encode huge-file.bin
./basecompresser decode huge-file.bin.base9
```

Useful controls:

```bash
./basecompresser encode input.bin --block-mib 64 --min-region-kib 256 --threads 16
./basecompresser encode input.bin --level fast
```

If `--threads` is omitted, BASE9 chooses a worker count from CPU availability and available RAM. Memory scales with block size × active workers, not total file size.

Compression levels control how much time BASE9 spends trying candidate methods. `best` is the default and tries every method. `balanced` includes context-rANS but skips token-base. `fast` skips both context-rANS and token-base while keeping RAW, adaptive-base, rANS, delta-rANS, and adaptive splitting. Faster levels can produce larger files.

## Benchmarking

Run the built-in, deterministic corpus benchmark with:

```bash
make bench
python3 bench/benchmark.py --size-mib 16 --repeat 3 --threads 8 --json bench-results.json
python3 bench/benchmark.py --size-mib 16 --repeat 3 --threads 8 --level fast
```

The generated cases cover repetitive text, structured records, token-heavy records, a byte counter, random data, mixed data, and already-compressed data. Every run also decodes the output and verifies its SHA-256 against the input. Results include ratio, encode/decode throughput, region-method counts, the source revision, and host details. The default benchmark is CPU-only for repeatability; use `--gpu auto` or `--gpu force` to measure GPU policies.

To benchmark files from a representative local corpus, pass their paths instead of generating examples:

```bash
python3 bench/benchmark.py --repeat 3 --threads 8 --gpu off /path/to/text.dat /path/to/archive.zip
```

Benchmark numbers depend on the CPU, storage, block size, thread count, and GPU driver. Compare runs on the same host with the same settings.

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
./basecompresser gpu-bench
```

`gpu-bench` measures CPU-only plus histogram, radix, and combined GPU modes
with both one and two OpenCL lanes. Three rounds rotate the benchmark order so
CPU/GPU results are not biased by one configuration always running first, and
the median sample is used. All GPU variants must produce byte-for-byte identical
BASE9 output before a policy is accepted. The fastest stage/lane combination is
cached for the exact OpenCL vendor, device, and driver version. AUTO only
switches away from CPU when the measured winner is at least 5% faster, which
avoids persisting benchmark noise. A driver/device change invalidates the cache.
Until a valid calibration exists, AUTO stays on CPU rather than guessing.

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
BASECOMPRESSER_GPU_ASYNC=off ./basecompresser encode input.bin
BASECOMPRESSER_GPU_ASYNC=on ./basecompresser encode input.bin
```

`sharded` is the normal histogram implementation. `direct` keeps the original
single global-atomic histogram for comparison, while `tiled` remains an
experimental local-memory implementation. Large discrete GPUs have their
default shard count bounded so histogram scratch memory cannot grow without
limit; advanced testing can override it with `BASECOMPRESSER_GPU_GROUPS`.

One GPU lane uses the lower-overhead blocking OpenCL path. Two lanes use
independent command queues, non-blocking transfers, and completion events so
work from different blocks can overlap. In calibrated AUTO mode, if every GPU
lane is busy, the requesting CPU worker immediately uses the exact CPU fallback
instead of waiting for a GPU lane. Explicit GPU overrides still wait for the
requested GPU path.

For the same Skylake/Gen9 hosts supported by 265Encode's legacy Intel path,
BaseCompresser can prepare a private OpenCL compute runtime without installing
packages into `/usr` or `/etc`:

```bash
./tools/setup-intel-legacy-opencl.sh
./basecompresser gpu-info
```

The Intel HD P530 path was measured directly. On the validated machine the
calibrator chose CPU for both pair histograms and radix packing. NVIDIA, AMD,
modern Intel, and legacy Intel therefore use the same rule now: measured
end-to-end speed decides AUTO when a valid calibration exists. Explicit
environment overrides always take precedence over the cached policy.

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
- continue profiling CPU token discovery; the current exact pair counter is branch-free and four-way unrolled after AVX2 gather/scatter variants benchmarked slower on Skylake
- wider token dictionaries and longer super-symbols when they prove profitable
- SIMD histogram and transform kernels (AVX2 first)
- denser rANS model serialization
- integrate context-cost estimates directly into the adaptive split planner
- denser/higher-resolution context probability models
- higher-order and mixed context models
- content-defined region boundaries rather than boundaries limited to the minimum-region grid
- benchmark corpus and regression thresholds

### Rolling block pipeline

Encoding uses persistent worker threads instead of stopping at batch boundaries.
A small look-ahead queue lets CPU analysis of later blocks overlap OpenCL work
and output handling from earlier blocks while preserving exact block order.
The default is up to four extra queued slots, automatically reduced when the
RAM budget cannot safely support them.

For benchmarking, look-ahead can be controlled explicitly:

```bash
BASECOMPRESSER_PIPELINE_EXTRA=0 ./basecompresser encode input.bin
BASECOMPRESSER_PIPELINE_EXTRA=4 ./basecompresser encode input.bin
```

Changing pipeline depth changes scheduling only; regression tests require the
resulting `.base9` bytes to remain identical.
