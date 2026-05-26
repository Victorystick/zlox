#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
ZIG_VERSION="0.16.0"
ZIG_ARCH="x86_64-linux"
ZIG_TARGET="zig-${ZIG_ARCH}-${ZIG_VERSION}"
ZIG_TARBALL="${ZIG_TARGET}.tar.xz"
ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/${ZIG_TARBALL}"

# Create a local bin directory for isolation
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

echo "=== Step 1: Downloading and Installing Zig ${ZIG_VERSION} ==="
if [ ! -d "$HOME/.local/zig" ]; then
    echo "Fetching Zig from ${ZIG_URL}..."
    curl -L "$ZIG_URL" -o "/tmp/${ZIG_TARBALL}"

    echo "Extracting Zig..."
    tar -xJ -f "/tmp/${ZIG_TARBALL}" -C "/tmp"

    mv "/tmp/${ZIG_TARGET}" "$HOME/.local/zig"
    rm "/tmp/${ZIG_TARBALL}"
else
    echo "Zig directory already exists in cache, skipping download."
fi

# Add Zig to the local path for this script's execution context
export PATH="$HOME/.local/zig:$PATH"

echo "Verifying Zig installation:"
zig version

echo "=== Step 2: Build ==="
pnpm build

echo "=== Build Process Completed Successfully ==="
