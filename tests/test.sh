#!/usr/bin/env bash
set -euo pipefail
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
PY

for f in text skewed delta random mixed; do
    "$BIN" encode "$TMP/$f" -o "$TMP/$f.base9" --block-mib 1 --min-region-kib 64 --threads 2 >/dev/null
    "$BIN" decode "$TMP/$f.base9" -o "$TMP/$f.out" --threads 2 >/dev/null
    cmp "$TMP/$f" "$TMP/$f.out"
done

echo "BASE9 round-trip tests PASS"
