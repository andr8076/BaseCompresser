#!/usr/bin/env python3
"""Reproducible end-to-end BASE9 benchmark for built-in or user-supplied files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = ROOT / "basecompresser"
MIB = 1024 * 1024


def write_repeated(path: Path, pattern: bytes, size: int) -> None:
    with path.open("wb") as stream:
        left = size
        chunk = pattern * max(1, min(1 << 16, size) // len(pattern))
        if not chunk:
            chunk = pattern
        while left:
            part = chunk[: min(left, len(chunk))]
            stream.write(part)
            left -= len(part)


def create_corpus(directory: Path, size: int) -> list[Path]:
    """Generate small, deterministic examples spanning common data shapes."""
    text_unit = (
        b"2026-09-24T12:34:56Z INFO worker=07 job=1842 status=complete "
        b"bytes=65536 elapsed_ms=23\n"
    )
    write_repeated(directory / "repetitive-text.bin", text_unit, size)

    record_unit = bytearray()
    for i in range(512):
        record_unit.extend(b"RECORD:")
        record_unit.extend(i.to_bytes(4, "little"))
        record_unit.extend(b":VALUE:")
        record_unit.extend(((i * 2654435761) & 0xFFFFFFFF).to_bytes(4, "little"))
        record_unit.extend(b"\n")
    write_repeated(directory / "structured-records.bin", bytes(record_unit), size)

    token_path = directory / "token-stream.bin"
    with token_path.open("wb") as stream:
        left = size
        value = 0
        while left:
            chunk = bytearray()
            while len(chunk) < min(left, 64 * 1024):
                chunk.extend(b"RECORD:")
                chunk.extend((value & 0xFFFFFFFF).to_bytes(4, "little"))
                chunk.extend(b":VALUE:")
                chunk.extend(((value * 2654435761) & 0xFFFFFFFF).to_bytes(4, "little"))
                chunk.extend(b"\n")
                value += 1
            part = bytes(chunk[:left])
            stream.write(part)
            left -= len(part)

    write_repeated(directory / "byte-counter.bin", bytes(range(256)), size)

    rng = random.Random(8076)
    noise_path = directory / "random-data.bin"
    with noise_path.open("wb") as stream:
        left = size
        while left:
            part = rng.randbytes(min(left, MIB))
            stream.write(part)
            left -= len(part)

    mixed_path = directory / "mixed-data.bin"
    patterns = (text_unit, bytes(range(256)))
    with mixed_path.open("wb") as stream:
        left = size
        part_index = 0
        while left:
            take = min(left, max(1, size // 3))
            if part_index % 3 == 1:
                stream.write(rng.randbytes(take))
            else:
                pattern = patterns[0 if part_index % 3 == 0 else 1]
                block = (pattern * ((take + len(pattern) - 1) // len(pattern)))[:take]
                stream.write(block)
            left -= take
            part_index += 1

    compressed_path = directory / "already-compressed.bin"
    compressed_tmp = directory / "noise.zlib"
    compressor = zlib.compressobj(level=6)
    with noise_path.open("rb") as source, compressed_tmp.open("wb") as output:
        for chunk in iter(lambda: source.read(MIB), b""):
            output.write(compressor.compress(chunk))
        output.write(compressor.flush())
    compressed_size = compressed_tmp.stat().st_size
    if not compressed_size:
        raise RuntimeError("generated zlib corpus unexpectedly has no data")
    with compressed_tmp.open("rb") as source, compressed_path.open("wb") as output:
        left = size
        while left:
            source.seek(0)
            while left and (chunk := source.read(min(MIB, left))):
                output.write(chunk)
                left -= len(chunk)
    compressed_tmp.unlink()

    return [
        directory / "repetitive-text.bin",
        directory / "structured-records.bin",
        token_path,
        directory / "byte-counter.bin",
        noise_path,
        mixed_path,
        compressed_path,
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_timed(command: list[str], env: dict[str, str]) -> tuple[float, str]:
    # BASE9 reports progress once per block; spool it so very large custom
    # inputs do not make the benchmark retain an unbounded progress log.
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        start = time.perf_counter()
        result = subprocess.run(
            command, env=env, stdout=stdout_file, stderr=stderr_file
        )
        elapsed = time.perf_counter() - start

        def read_tail(stream) -> str:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(0, end - 65536))
            return stream.read().decode("utf-8", errors="replace")

        stdout = read_tail(stdout_file)
        stderr = read_tail(stderr_file)

    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout (tail):\n{stdout}\nstderr (tail):\n{stderr}"
        )
    return elapsed, stdout


def git_revision(directory: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty(directory: Path) -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def run_case(
    source: Path,
    temporary: Path,
    index: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)

    encoded = temporary / f"{index}.base9"
    decoded = temporary / f"{index}.decoded"
    input_bytes = source.stat().st_size
    source_hash = sha256(source)
    encode_times: list[float] = []
    decode_times: list[float] = []
    output_bytes = 0
    method_counts: dict[str, int] = {}

    for repeat in range(args.repeat):
        if encoded.exists():
            encoded.unlink()
        if decoded.exists():
            decoded.unlink()
        encode_command = [
            str(args.binary), "encode", str(source), "-o", str(encoded),
            "--block-mib", str(args.block_mib), "--threads", str(args.threads),
        ]
        if args.level != "best":
            encode_command.extend(("--level", args.level))
        encode_elapsed, encode_output = run_timed(encode_command, env)
        encode_times.append(encode_elapsed)
        match = re.search(
            r"Regions:\s+\d+\s+\(raw=(\d+), base=(\d+), rans=(\d+), "
            r"delta-rans=(\d+), context-rans=(\d+), token-base=(\d+)\)",
            encode_output,
        )
        if match:
            method_counts = dict(zip(
                ("raw", "base", "rans", "delta_rans", "context_rans", "token_base"),
                map(int, match.groups()),
            ))
        output_bytes = encoded.stat().st_size
        decode_elapsed, _ = run_timed(
            [str(args.binary), "decode", str(encoded), "-o", str(decoded),
             "--threads", str(args.threads)],
            env,
        )
        decode_times.append(decode_elapsed)
        if decoded.stat().st_size != input_bytes or sha256(decoded) != source_hash:
            raise RuntimeError(f"Round-trip mismatch for {source}")

    encode_seconds = statistics.median(encode_times)
    decode_seconds = statistics.median(decode_times)
    return {
        "file": source.name,
        "input_sha256": source_hash,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "ratio": output_bytes / input_bytes if input_bytes else 1.0,
        "encode_seconds": encode_seconds,
        "encode_mib_s": input_bytes / MIB / encode_seconds if encode_seconds else 0.0,
        "decode_seconds": decode_seconds,
        "decode_mib_s": input_bytes / MIB / decode_seconds if decode_seconds else 0.0,
        "round_trip": "pass",
        "repeat": args.repeat,
        "methods": method_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="Files to benchmark; built-in corpus runs when omitted")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN,
                        help="BASE9 executable to benchmark (default: repository build)")
    parser.add_argument("--size-mib", type=int, default=2,
                        help="Size of each generated corpus file (default: 2 MiB)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Encode/decode repetitions; median is reported (default: 1)")
    parser.add_argument("--block-mib", type=int, default=4,
                        help="BASE9 block size (default: 4 MiB)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Worker count; 0 uses BASE9 automatic selection")
    parser.add_argument("--level", choices=("fast", "balanced", "best"), default="best",
                        help="Compression search level (default: best)")
    parser.add_argument("--gpu", choices=("off", "auto", "force"), default="off",
                        help="GPU policy for the run (default: off for repeatability)")
    parser.add_argument("--json", type=Path,
                        help="Write full machine-readable results to this path")
    args = parser.parse_args()

    args.binary = args.binary.resolve()
    if not args.binary.is_file():
        parser.error(f"{args.binary} is missing; run `make` first or pass --binary")
    if args.size_mib < 1 or args.repeat < 1 or args.block_mib < 1 or args.threads < 0:
        parser.error("size, repeat, and block size must be positive; threads must be >= 0")

    env = os.environ.copy()
    env["BASECOMPRESSER_GPU"] = args.gpu
    with tempfile.TemporaryDirectory(prefix="base9-bench-") as temp_name:
        temporary = Path(temp_name)
        if args.inputs:
            sources = [path.resolve() for path in args.inputs]
        else:
            corpus = temporary / "corpus"
            corpus.mkdir()
            sources = create_corpus(corpus, args.size_mib * MIB)

        results = [run_case(path, temporary, i, args, env)
                   for i, path in enumerate(sources)]

    report = {
        "revision": git_revision(),
        "binary_revision": git_revision(args.binary.parent),
        "binary_worktree_dirty": git_worktree_dirty(args.binary.parent),
        "platform": platform.platform(),
        "processor": cpu_name(),
        "python": sys.version.split()[0],
        "settings": {
            "block_mib": args.block_mib,
            "threads": args.threads,
            "level": args.level,
            "gpu": args.gpu,
            "repeat": args.repeat,
        },
        "results": results,
    }

    print(f"BASE9 benchmark | binary revision={report['binary_revision']} "
          f"| dirty={report['binary_worktree_dirty']} | GPU={args.gpu} "
          f"| threads={args.threads} | level={args.level}")
    print(f"{'input':34} {'MiB':>8} {'ratio':>8} {'enc MiB/s':>11} {'dec MiB/s':>11} {'check':>8}")
    for result in results:
        name = Path(str(result["file"])).name[:34]
        print(f"{name:34} {int(result['input_bytes']) / MIB:8.2f} "
              f"{float(result['ratio']):8.4f} {float(result['encode_mib_s']):11.2f} "
              f"{float(result['decode_mib_s']):11.2f} {result['round_trip']:>8}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON results: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
