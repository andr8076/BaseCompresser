#!/usr/bin/env bash
set -euo pipefail
: "${BASECOMPRESSER_GPU:=off}"
export BASECOMPRESSER_GPU
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/basecompresser"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

printf 'banana banana banana banana\n' > "$TMP/text"
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
