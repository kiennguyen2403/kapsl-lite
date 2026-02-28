#!/usr/bin/env python3
"""
Send one manual prompt to Voxtral and wait for the inference result.

This is a thin wrapper around request-inference-result.py with Voxtral defaults.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_TARGET = "voxtral-mini-3b-2507"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_aimod = (
        script_dir.parents[1] / "kapsl-runtime" / "models" / "multi" / "voxtral.aimod"
    )

    parser = argparse.ArgumentParser(
        description="Send manual trigger to Voxtral and wait for one result."
    )
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="Runtime target package."
    )
    parser.add_argument("--pipe", default=None, help="Ingress named pipe path.")
    parser.add_argument("--results-file", default=None, help="Inference JSONL path.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds to wait for inference result.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=None,
        help="Seconds to wait for ingress connection.",
    )
    parser.add_argument(
        "--aimod",
        default=str(default_aimod),
        help="Path to Voxtral .aimod for token decode.",
    )
    parser.add_argument(
        "--no-token-decode",
        action="store_true",
        help="Disable top_class_id token decode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    base_script = script_dir / "request-inference-result.py"

    if not base_script.exists():
        print(f"Missing script: {base_script}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        str(base_script),
        "--target",
        args.target,
        "--prompt",
        args.prompt,
    ]

    if args.pipe:
        cmd.extend(["--pipe", args.pipe])
    if args.results_file:
        cmd.extend(["--results-file", args.results_file])
    if args.timeout is not None:
        cmd.extend(["--timeout", str(args.timeout)])
    if args.connect_timeout is not None:
        cmd.extend(["--connect-timeout", str(args.connect_timeout)])

    if args.no_token_decode:
        cmd.append("--no-token-decode")
    else:
        cmd.extend(["--aimod", args.aimod])

    completed = subprocess.run(cmd)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
