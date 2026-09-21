#!/usr/bin/env bash
set -euo pipefail

# Private OpenCL compute runtime for the same Skylake/Gen9 systems handled by
# 265Encode's legacy Intel path. Nothing is installed into /usr or /etc.

case "$(uname -s):$(uname -m)" in
    Linux:x86_64) ;;
    *) echo "Intel legacy GPU setup supports Linux x86_64 only." >&2; exit 1 ;;
esac

legacy_device=""
for device in /sys/class/drm/renderD*/device; do
    [[ -r "$device/vendor" && -r "$device/device" ]] || continue
    read -r vendor < "$device/vendor" || continue
    read -r device_id < "$device/device" || continue
    [[ "${vendor,,}" == 0x8086 ]] || continue
    driver="$(basename "$(readlink -f "$device/driver" 2>/dev/null || true)")"
    [[ "$driver" == i915 ]] || continue
    case "${device_id,,}" in
        0x1902|0x1906|0x190a|0x190b|0x190e|0x1912|0x1913|0x1915|0x1916|0x1917|0x191a|0x191b|0x191d|0x191e|0x1921|0x1923|0x1926|0x1927|0x192a|0x192b|0x192d|0x1932|0x193a|0x193b|0x193d)
            legacy_device="${device_id,,}"; break ;;
    esac
done

[[ -n "$legacy_device" ]] || {
    echo "No 265Encode-compatible Intel Skylake/Gen9 i915 GPU was detected." >&2
    exit 1
}

command -v apt-get >/dev/null 2>&1 || { echo "apt-get is required." >&2; exit 1; }
command -v dpkg-deb >/dev/null 2>&1 || { echo "dpkg-deb is required." >&2; exit 1; }

cache="${XDG_CACHE_HOME:-$HOME/.cache}/BaseCompresser/intel-legacy-opencl"
runtime="$cache/runtime"
manifest="$runtime/runtime-manifest.txt"
if [[ -r "$manifest" ]] &&
   grep -Fxq 'runtime_kind=intel-opencl-gen9-private' "$manifest" &&
   [[ -r "$runtime/etc/OpenCL/vendors/intel.icd" &&
      -r "$runtime/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so" ]]; then
    echo "BaseCompresser Intel legacy OpenCL runtime is already ready."
    echo "Device: $legacy_device"
    exit 0
fi

mkdir -p "$cache"
tmp="$(mktemp -d "$cache/setup.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/pkgs" "$tmp/runtime"
cd "$tmp/pkgs"

packages=(
    intel-opencl-icd
    libigc1
    libigdfcl1
    libopencl-clang14
    libllvmspirvlib14
    libllvm14t64
    libigdgmm12
)

if apt-cache show libclang-cpp14t64 >/dev/null 2>&1; then
    packages+=(libclang-cpp14t64)
else
    packages+=(libclang-cpp14)
fi

echo "BaseCompresser: fetching private Intel Gen9 OpenCL runtime..."
for pkg in "${packages[@]}"; do
    apt-get download "$pkg"
done

for deb in ./*.deb; do
    dpkg-deb -x "$deb" "$tmp/runtime"
done

libdir="$tmp/runtime/usr/lib/x86_64-linux-gnu"
icddir="$tmp/runtime/etc/OpenCL/vendors"
driver="$libdir/intel-opencl/libigdrcl.so"
[[ -r "$driver" ]] || { echo "Intel OpenCL driver missing after extraction." >&2; exit 1; }
[[ -r "$libdir/libigc.so.1" && -r "$libdir/libigdfcl.so.1" ]] || {
    echo "Intel compiler libraries missing after extraction." >&2; exit 1;
}

rm -rf "$runtime.new"
mv "$tmp/runtime" "$runtime.new"
rm -rf "$runtime"
mv "$runtime.new" "$runtime"

libdir="$runtime/usr/lib/x86_64-linux-gnu"
icddir="$runtime/etc/OpenCL/vendors"
driver="$libdir/intel-opencl/libigdrcl.so"
printf '%s\n' "$driver" > "$icddir/intel.icd"

{
    echo 'runtime_kind=intel-opencl-gen9-private'
    echo "device_id=$legacy_device"
    echo 'source=distribution-packages-private-extraction'
    for pkg in "${packages[@]}"; do
        version="$(apt-cache policy "$pkg" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
        echo "package_${pkg}=${version:-unknown}"
    done
} > "$runtime/runtime-manifest.txt"

echo "BaseCompresser Intel legacy OpenCL runtime is ready."
echo "Device: $legacy_device"
echo "Cache:  $runtime"
echo "Run:    ./basecompresser gpu-info"
