#!/usr/bin/env bash
set -euo pipefail

TARGET="aarch64-unknown-linux-gnu"

if ! command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
  echo "Missing aarch64-linux-gnu-gcc in PATH"
  echo "Install a GNU cross toolchain before building."
  exit 1
fi

rustup target add "${TARGET}" >/dev/null

export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="aarch64-linux-gnu-gcc"
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+crt-static"

cargo build --release --target "${TARGET}"

echo "Built: target/${TARGET}/release/kapsl"
