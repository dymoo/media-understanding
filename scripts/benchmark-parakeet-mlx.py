#!/usr/bin/env python3
"""Benchmark Parakeet MLX variants from a subprocess.

Outputs a single JSON object to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn


def load_model(model_id: str):
    try:
        from parakeet_mlx import from_pretrained  # type: ignore
    except ImportError:
        from parakeet import from_pretrained  # type: ignore

    try:
        return from_pretrained(model_id, dtype=mx.bfloat16)
    except TypeError:
        return from_pretrained(model_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    parser.add_argument(
        "--model",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        help="Hugging Face model id to load before local encoder quantization",
    )
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--quantize-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    started = time.perf_counter()
    try:
        model = load_model(args.model)
        if args.quantize_bits > 0:
            nn.quantize(model.encoder, group_size=args.group_size, bits=args.quantize_bits)
        result = model.transcribe(args.audio_file)
        mx.synchronize()
    except Exception as exc:  # pragma: no cover - benchmark script
        print(json.dumps({"error": str(exc), "variant": args.model, "device": args.device}))
        return 1

    wall_ms = (time.perf_counter() - started) * 1000.0
    text = getattr(result, "text", "") or ""

    payload = {
        "runtime": "parakeet-mlx",
        "variant": f"{args.model} (encoder-int{args.quantize_bits})",
        "device": args.device,
        "totalMs": wall_ms,
        "transcriptChars": len(text),
        "text": text,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
