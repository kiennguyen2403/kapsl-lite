#!/usr/bin/env python3
"""
Capture webcam frames and publish them to kapsl-runtime-lite JSON ingress pipe.

Default target package is `mistral-llm`, configurable with --target.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from typing import TextIO

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Missing dependency: opencv-python. Install with: pip install opencv-python"
    ) from exc


DEFAULT_PIPE = r"\\.\pipe\kapsl-lite-ingress-json"
ERROR_FILE_NOT_FOUND = 2
ERROR_SEM_TIMEOUT = 121
ERROR_PIPE_BUSY = 231
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send webcam frames to kapsl-lite via Windows named pipe JSON ingress."
    )
    parser.add_argument("--pipe", default=DEFAULT_PIPE, help="Named pipe path.")
    parser.add_argument(
        "--target",
        default="mistral-llm",
        help="Runtime package name to receive frames.",
    )
    parser.add_argument(
        "--source-id", default="camera/front", help="Ingress source_id."
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index."
    )
    parser.add_argument(
        "--width", type=int, default=64, help="Resize width before sending (pixels)."
    )
    parser.add_argument(
        "--height", type=int, default=64, help="Resize height before sending (pixels)."
    )
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Send frame rate (events per second)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run forever).",
    )
    return parser.parse_args()


def write_line_to_named_pipe(
    pipe_path: str, data: bytes, timeout_s: float = 5.0
) -> None:
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


def build_event(
    frame_gray_u8,
    target: str,
    source_id: str,
    width: int,
    height: int,
) -> dict:
    flat = frame_gray_u8.reshape(-1).tolist()
    return {
        "type": "input",
        "target_package": target,
        "source_id": source_id,
        "timestamp_ms": int(time.time() * 1000),
        "payload": {
            "shape": [height, width],
            "data": flat,
        },
        "metadata": {
            "modality": "image",
            "encoding": "grayscale_u8",
            "width": width,
            "height": height,
        },
    }


def main() -> int:
    args = parse_args()
    if os.name != "nt":
        print("This script is for Windows named pipes.", file=sys.stderr)
        return 2
    if args.width <= 0 or args.height <= 0:
        print("--width and --height must be > 0", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("--fps must be > 0", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Unable to open webcam index {args.camera_index}", file=sys.stderr)
        return 1

    print(
        f"[webcam->kapsl] pipe={args.pipe} target={args.target} "
        f"camera={args.camera_index} size={args.width}x{args.height} fps={args.fps}"
    )

    frame_interval = 1.0 / args.fps
    sent = 0

    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            frame_small = cv2.resize(
                frame, (args.width, args.height), interpolation=cv2.INTER_AREA
            )
            frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            event = build_event(
                frame_gray, args.target, args.source_id, args.width, args.height
            )
            line = json.dumps(event, separators=(",", ":")) + "\n"
            write_line_to_named_pipe(args.pipe, line.encode("utf-8"))

            sent += 1
            if sent % 10 == 0:
                print(f"[webcam->kapsl] sent={sent}")

            if args.max_frames > 0 and sent >= args.max_frames:
                break

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    print(f"[webcam->kapsl] done sent={sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
