#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/basecompresser"

if ! "$BIN" gpu-info >/dev/null 2>&1; then
    echo "BASE9 GPU equivalence test SKIP (no OpenCL GPU)"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 - <<'PY' "$TMP/input"
import sys
p=sys.argv[1]
with open(p,'wb') as f:
    for i in range(180_000):
        f.write(b'RECORD:')
        f.write((i & 0xffffffff).to_bytes(4,'little'))
        f.write(b':VALUE:')
        f.write(((i*2654435761)&0xffffffff).to_bytes(4,'little'))
        f.write(b'\n')
PY

BASECOMPRESSER_GPU=off "$BIN" encode "$TMP/input" -o "$TMP/cpu.base9" \
    --block-mib 8 --min-region-kib 8192 --threads 1 >/dev/null

BASECOMPRESSER_GPU=force \
BASECOMPRESSER_GPU_HIST=sharded \
BASECOMPRESSER_GPU_PACK=force \
BASECOMPRESSER_GPU_ZERO_COPY=off \
"$BIN" encode "$TMP/input" -o "$TMP/gpu.base9" \
    --block-mib 8 --min-region-kib 8192 --threads 1 >"$TMP/gpu.log"

cmp "$TMP/cpu.base9" "$TMP/gpu.base9"
grep -Eq 'GPU scans:[[:space:]]+[1-9][0-9]* pair histograms' "$TMP/gpu.log"
grep -Eq 'GPU radix:[[:space:]]+[1-9][0-9]* higher-base packs' "$TMP/gpu.log"

BASECOMPRESSER_GPU=off "$BIN" decode "$TMP/gpu.base9" -o "$TMP/out" \
    --threads 1 >/dev/null
cmp "$TMP/input" "$TMP/out"

echo "BASE9 CPU/GPU equivalence PASS"
