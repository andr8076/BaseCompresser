#!/usr/bin/env bash
set -euo pipefail
: "${BASECOMPRESSER_GPU:=off}"
export BASECOMPRESSER_GPU
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BASECOMPRESSER_BIN:-$ROOT/basecompresser}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

printf 'banana banana banana banana\n' > "$TMP/text"
: > "$TMP/empty"
python3 - <<'PY' "$TMP"
import os, sys
p=sys.argv[1]
with open(p+'/skewed','wb') as f:
    f.write(b'A'*900000 + bytes(range(256))*390)
with open(p+'/delta','wb') as f:
    f.write(bytes((i & 255) for i in range(2_000_000)))
with open(p+'/random','wb') as f:
    f.write(os.urandom(2_000_000))
with open(p+'/mixed','wb') as f:
    f.write(b'X'*(1024*1024))
    f.write(os.urandom(1024*1024))
    f.write(bytes((i & 255) for i in range(1024*1024)))
with open(p+'/context','wb') as f:
    x=0
    out=bytearray(4_000_000)
    for i in range(len(out)):
        out[i]=x
        x=(5*x+1)&255
    f.write(out)
with open(p+'/tokens','wb') as f:
    for i in range(100_000):
        f.write(b'RECORD:')
        f.write((i & 0xffffffff).to_bytes(4,'little'))
        f.write(b':VALUE:')
        f.write(((i*2654435761)&0xffffffff).to_bytes(4,'little'))
        f.write(b'\n')
PY

for f in text skewed delta random mixed context tokens; do
    "$BIN" encode "$TMP/$f" -o "$TMP/$f.base9" --block-mib 1 --min-region-kib 64 --threads 2 >/dev/null
    "$BIN" decode "$TMP/$f.base9" -o "$TMP/$f.out" --threads 2 >/dev/null
    cmp "$TMP/$f" "$TMP/$f.out"
done

"$BIN" encode "$TMP/empty" -o "$TMP/empty.base9" --threads 1 >/dev/null
"$BIN" decode "$TMP/empty.base9" -o "$TMP/empty.out" --threads 1 >/dev/null
cmp "$TMP/empty" "$TMP/empty.out"

for level in fast balanced best; do
    "$BIN" encode "$TMP/mixed" -o "$TMP/mixed.$level.base9" \
        --block-mib 1 --min-region-kib 64 --threads 2 --level "$level" >/dev/null
    "$BIN" decode "$TMP/mixed.$level.base9" -o "$TMP/mixed.$level.out" \
        --threads 2 >/dev/null
    cmp "$TMP/mixed" "$TMP/mixed.$level.out"
done

# Decoder rejects malformed lengths before allocating from archive-controlled
# sizes, and never truncates a source when input and output are the same file.
"$BIN" encode "$TMP/text" -o "$TMP/guard.base9" --block-mib 1 --threads 1 >/dev/null
cp "$TMP/guard.base9" "$TMP/guard.original.base9"
cp "$TMP/guard.base9" "$TMP/bad-block-size.base9"
cp "$TMP/guard.base9" "$TMP/bad-min-region.base9"
cp "$TMP/guard.base9" "$TMP/bad-body-size.base9"
cp "$TMP/guard.base9" "$TMP/bad-region-size.base9"
cp "$TMP/guard.base9" "$TMP/bad-file-reserved.base9"
cp "$TMP/guard.base9" "$TMP/bad-block-reserved.base9"
cp "$TMP/guard.base9" "$TMP/bad-region-flags.base9"
cp "$TMP/guard.base9" "$TMP/bad-region-crc.base9"
cp "$TMP/guard.base9" "$TMP/bad-block-crc.base9"
cp "$TMP/guard.base9" "$TMP/trailing-data.base9"
python3 - <<'PY' "$TMP"
import os, sys
p=sys.argv[1]
with open(p+'/bad-block-size.base9','r+b') as f:
    f.seek(24)
    f.write(bytes(4))
with open(p+'/bad-min-region.base9','r+b') as f:
    f.seek(28)
    f.write(bytes(4))
with open(p+'/bad-body-size.base9','r+b') as f:
    f.seek(48)
    f.write((1 << 63).to_bytes(8,'little'))
with open(p+'/bad-region-size.base9','r+b') as f:
    f.seek(68)
    f.write((0xffffffff).to_bytes(4,'little'))
with open(p+'/bad-file-reserved.base9','r+b') as f:
    f.seek(32)
    f.write((1).to_bytes(4,'little'))
with open(p+'/bad-block-reserved.base9','r+b') as f:
    f.seek(60)
    f.write((1).to_bytes(4,'little'))
with open(p+'/bad-region-flags.base9','r+b') as f:
    f.seek(65)
    f.write(b'\x01')
with open(p+'/bad-region-crc.base9','r+b') as f:
    f.seek(76)
    f.write((1).to_bytes(4,'little'))
with open(p+'/bad-block-crc.base9','r+b') as f:
    f.seek(56)
    f.write((1).to_bytes(4,'little'))
with open(p+'/trailing-data.base9','ab') as f:
    f.write(b'X')
PY
for bad in bad-block-size bad-min-region bad-body-size bad-region-size bad-file-reserved \
           bad-block-reserved bad-region-flags bad-region-crc bad-block-crc trailing-data; do
    if "$BIN" decode "$TMP/$bad.base9" -o "$TMP/$bad.out" --threads 1 >/dev/null 2>&1; then
        echo "Expected malformed archive to be rejected: $bad" >&2
        exit 1
    fi
done
if "$BIN" encode "$TMP/text" -o "$TMP/text" --threads 1 >/dev/null 2>&1; then
    echo "Expected encode to reject an identical input and output path" >&2
    exit 1
fi
cmp "$TMP/text" <(printf 'banana banana banana banana\n')
if "$BIN" decode "$TMP/guard.base9" -o "$TMP/guard.base9" --threads 1 >/dev/null 2>&1; then
    echo "Expected decode to reject an identical input and output path" >&2
    exit 1
fi
cmp "$TMP/guard.base9" "$TMP/guard.original.base9"
if "$BIN" encode "$TMP/text" -o "$TMP/invalid.base9" --threads nope >/dev/null 2>&1; then
    echo "Expected malformed numeric options to be rejected" >&2
    exit 1
fi
if "$BIN" encode "$TMP/text" -o "$TMP/invalid.base9" --threads -1 >/dev/null 2>&1; then
    echo "Expected negative numeric options to be rejected" >&2
    exit 1
fi
if "$BIN" encode "$TMP/text" -o "$TMP/invalid.base9" --level maximum >/dev/null 2>&1; then
    echo "Expected invalid compression levels to be rejected" >&2
    exit 1
fi

ctx_log="$TMP/context.log"
"$BIN" encode "$TMP/context" -o "$TMP/context.check.base9" --block-mib 4 --min-region-kib 256 --threads 1 >"$ctx_log"
grep -Eq 'context-rans=[1-9][0-9]*' "$ctx_log"

token_log="$TMP/tokens.log"
"$BIN" encode "$TMP/tokens" -o "$TMP/tokens.check.base9" --block-mib 4 --min-region-kib 256 --threads 1 >"$token_log"
grep -Eq 'token-base=[1-9][0-9]*' "$token_log"

# The rolling worker pipeline must never change deterministic output.
BASECOMPRESSER_PIPELINE_EXTRA=0 "$BIN" encode "$TMP/tokens" -o "$TMP/tokens.pipeline0.base9" --block-mib 1 --min-region-kib 256 --threads 2 >/dev/null
BASECOMPRESSER_PIPELINE_EXTRA=4 "$BIN" encode "$TMP/tokens" -o "$TMP/tokens.pipeline4.base9" --block-mib 1 --min-region-kib 256 --threads 2 >/dev/null
cmp "$TMP/tokens.pipeline0.base9" "$TMP/tokens.pipeline4.base9"

echo "BASE9 round-trip tests PASS"
