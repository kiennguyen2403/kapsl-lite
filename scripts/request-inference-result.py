#!/usr/bin/env python3
"""
Send one manual trigger to kapsl-lite and wait for a JSONL inference result.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

DEFAULT_PIPE = r"\\.\pipe\kapsl-lite-ingress-json"
DEFAULT_TARGET = "mistral-llm"
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_CONNECT_TIMEOUT_S = 120.0

ERROR_FILE_NOT_FOUND = 2
ERROR_SEM_TIMEOUT = 121
ERROR_PIPE_BUSY = 231
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
TOP_CLASS_ID_RE = re.compile(r"\btop_class_id=(\d+)\b")


def parse_args() -> argparse.Namespace:
    default_aimod = (
        Path(__file__).resolve().parents[2]
        / "kapsl-runtime"
        / "models"
        / "multi"
        / "mistral.aimod"
    )
    parser = argparse.ArgumentParser(
        description="Send manual trigger and wait for inference result JSONL record."
    )
    parser.add_argument(
        "--pipe",
        default=os.environ.get("KAPSL_LITE_INGRESS_PIPE", DEFAULT_PIPE),
        help="Windows JSON ingress pipe path.",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target package name.")
    parser.add_argument("--prompt", required=True, help="Manual prompt to send.")
    parser.add_argument(
        "--results-file",
        default=os.environ.get(
            "KAPSL_LITE_INFERENCE_RESULTS_PATH",
            str(
                Path(__file__).resolve().parents[1]
                / "target"
                / "inference-results.jsonl"
            ),
        ),
        help="Path to inference results JSONL file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Seconds to wait for matching inference result.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_S,
        help="Seconds to wait for ingress pipe to become available.",
    )
    parser.add_argument(
        "--aimod",
        default=os.environ.get("KAPSL_LITE_TOKENIZER_AIMOD", str(default_aimod)),
        help="Path to .aimod package containing tokenizer.json for token decode.",
    )
    parser.add_argument(
        "--no-token-decode",
        action="store_true",
        help="Disable top_class_id -> token decode enrichment.",
    )
    return parser.parse_args()


def send_manual_trigger(
    pipe_path: str, target: str, prompt: str, connect_timeout_s: float
) -> None:
    payload = {
        "type": "manual",
        "target_package": target,
        "custom_prompt": prompt,
    }
    encoded = json.dumps(payload, separators=(",", ":")) + "\n"
    write_line_to_named_pipe(pipe_path, encoded.encode("utf-8"), connect_timeout_s)


def write_line_to_named_pipe(pipe_path: str, data: bytes, timeout_s: float) -> None:
    if os.name != "nt":
        raise OSError("named pipe write is only supported on Windows")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        remaining_ms = max(1, int((deadline - time.time()) * 1000))
        wait_ms = min(1000, remaining_ms)

        if not kernel32.WaitNamedPipeW(
            ctypes.c_wchar_p(pipe_path), ctypes.c_uint(wait_ms)
        ):
            err = ctypes.get_last_error()
            if err in (ERROR_FILE_NOT_FOUND, ERROR_PIPE_BUSY, ERROR_SEM_TIMEOUT):
                time.sleep(0.05)
                continue
            raise OSError(err, f"WaitNamedPipeW failed for {pipe_path}")

        handle = kernel32.CreateFileW(
            ctypes.c_wchar_p(pipe_path),
            ctypes.c_uint(GENERIC_WRITE),
            ctypes.c_uint(0),
            ctypes.c_void_p(),
            ctypes.c_uint(OPEN_EXISTING),
            ctypes.c_uint(FILE_ATTRIBUTE_NORMAL),
            ctypes.c_void_p(),
        )
        if handle == INVALID_HANDLE_VALUE:
            err = ctypes.get_last_error()
            if err in (ERROR_FILE_NOT_FOUND, ERROR_PIPE_BUSY):
                time.sleep(0.05)
                continue
            raise OSError(err, f"CreateFileW failed for {pipe_path}")

        try:
            written = ctypes.c_uint(0)
            ok = kernel32.WriteFile(
                ctypes.c_void_p(handle),
                ctypes.c_char_p(data),
                ctypes.c_uint(len(data)),
                ctypes.byref(written),
                ctypes.c_void_p(),
            )
            if not ok:
                err = ctypes.get_last_error()
                raise OSError(err, f"WriteFile failed for {pipe_path}")
            if written.value != len(data):
                raise OSError(
                    f"partial pipe write: expected {len(data)} bytes, wrote {written.value}"
                )
            return
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))

    raise OSError(f"Timed out waiting for named pipe {pipe_path}")


def wait_for_result(
    results_path: Path, target: str, min_timestamp_ms: int, timeout_s: float
) -> dict[str, Any] | None:
    deadline = time.time() + timeout_s
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.touch(exist_ok=True)

    with results_path.open("r", encoding="utf-8") as reader:
        reader.seek(0, os.SEEK_SET)

        while time.time() < deadline:
            line = reader.readline()
            if not line:
                time.sleep(0.10)
                continue
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("package_name") != target:
                continue
            if int(row.get("timestamp_ms", 0)) < min_timestamp_ms:
                continue
            return row
    return None


def parse_top_class_id(summary: str) -> int | None:
    match = TOP_CLASS_ID_RE.search(summary)
    if not match:
        return None
    return int(match.group(1))


def load_id_to_token_map(aimod_path: Path) -> dict[int, str]:
    if not aimod_path.exists():
        raise FileNotFoundError(f"aimod not found: {aimod_path}")

    tokenizer: dict[str, Any] | None = None
    with tarfile.open(aimod_path, "r:gz") as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if name != "tokenizer.json":
                continue
            handle = tar.extractfile(member)
            if handle is None:
                break
            tokenizer = json.load(handle)
            break

    if tokenizer is None:
        raise FileNotFoundError(f"tokenizer.json not found in {aimod_path}")

    id_to_token: dict[int, str] = {}

    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        vocab = tokenizer.get("vocab")
    if isinstance(vocab, dict):
        for token, idx in vocab.items():
            if isinstance(token, str) and isinstance(idx, int) and idx >= 0:
                id_to_token.setdefault(idx, token)

    added = tokenizer.get("added_tokens")
    if isinstance(added, list):
        for item in added:
            if not isinstance(item, dict):
                continue
            idx = item.get("id")
            token = item.get("content")
            if isinstance(idx, int) and idx >= 0 and isinstance(token, str):
                id_to_token.setdefault(idx, token)

    return id_to_token


def prettify_token(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    if os.name != "nt":
        print(
            "This script currently supports Windows named-pipe ingress only.",
            file=sys.stderr,
        )
        return 2

    results_path = Path(args.results_file)
    now_ms = int(time.time() * 1000)

    try:
        send_manual_trigger(args.pipe, args.target, args.prompt, args.connect_timeout)
    except OSError as exc:
        print(f"Failed to send trigger to pipe '{args.pipe}': {exc}", file=sys.stderr)
        print(
            "Make sure kapsl runtime is running and ingress is enabled "
            "(KAPSL_LITE_INGRESS_ENABLED=1).",
            file=sys.stderr,
        )
        return 1

    result = wait_for_result(results_path, args.target, now_ms, args.timeout)
    if result is None:
        print(
            f"Timed out after {args.timeout:.1f}s waiting for result in '{results_path}'.",
            file=sys.stderr,
        )
        print(
            "Ensure runtime was started with KAPSL_LITE_INFERENCE_RESULTS_PATH set.",
            file=sys.stderr,
        )
        return 3

    if not args.no_token_decode:
        top_class_id = parse_top_class_id(str(result.get("output_summary", "")))
        if top_class_id is not None:
            try:
                token_map = load_id_to_token_map(Path(args.aimod))
                token = token_map.get(top_class_id)
                if token is not None:
                    result["top_token"] = token
                    result["top_token_pretty"] = prettify_token(token)
            except Exception as exc:
                result["token_decode_error"] = str(exc)

    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
