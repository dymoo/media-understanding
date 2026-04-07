# ASR Benchmarking

This repo includes benchmark-only scripts for comparing four local ASR paths on the same audio file:

- `Parakeet MLX INT4 GPU`
- `Parakeet MLX INT4 CPU`
- `Parakeet ONNX INT8 CPU`
- `Whisper.cpp q5_1`

These scripts are intentionally benchmark-only. They do not affect shipped runtime behavior in `src/`.

The MLX INT4 legs load `mlx-community/parakeet-tdt-0.6b-v3` and quantize the
encoder to 4-bit locally at runtime. The original plan to consume a pre-quantized
external INT4 checkpoint was dropped because the available public checkpoint was
not compatible with the published `parakeet-mlx` runtime.

## Prerequisites

- `pnpm build`
- `ffprobe` available on `PATH`
- local `whisper.cpp` binary (default: `/opt/homebrew/bin/whisper-cli`)
- local `ggml-base.en-q5_1.bin` model (default: `~/.cache/media-understanding/models/ggml-base.en-q5_1.bin`)
- Python virtualenv with `parakeet-mlx`

Recommended local MLX setup:

```bash
mkdir -p tmp/benchmarks
python3 -m venv tmp/benchmarks/.venv
tmp/benchmarks/.venv/bin/python -m pip install -U pip
tmp/benchmarks/.venv/bin/python -m pip install parakeet-mlx
```

The matrix runner defaults `PARAKEET_MLX_PYTHON` to `tmp/benchmarks/.venv/bin/python`.

## Commands

Current Parakeet CPU path:

```bash
pnpm build
node scripts/benchmark-ep.mjs --json /path/to/audio.wav
```

MLX INT4 directly:

```bash
tmp/benchmarks/.venv/bin/python scripts/benchmark-parakeet-mlx.py \
  --device gpu \
  --model mlx-community/parakeet-tdt-0.6b-v3 \
  /path/to/audio.wav
```

Whisper.cpp directly:

```bash
node scripts/benchmark-whisper-cpp.mjs \
  --bin /opt/homebrew/bin/whisper-cli \
  --model ~/.cache/media-understanding/models/ggml-base.en-q5_1.bin \
  /path/to/audio.wav
```

Full matrix:

```bash
pnpm build
node scripts/benchmark-asr-matrix.mjs --runs 3 /path/to/audio.wav
```

## Environment Overrides

- `PARAKEET_MLX_PYTHON` — Python executable with `parakeet-mlx` installed
- `PARAKEET_MLX_MODEL` — defaults to `mlx-community/parakeet-tdt-0.6b-v3`
- `WHISPER_CPP_BIN` — defaults to `/opt/homebrew/bin/whisper-cli`
- `WHISPER_CPP_MODEL` — defaults to `~/.cache/media-understanding/models/ggml-base.en-q5_1.bin`

## Output Artifacts

Matrix outputs go to `tmp/benchmarks/<timestamp>/`:

- `matrix.json`
- one `.json` summary per variant
- one `.txt` transcript per variant

Wall time includes startup and model load overheads by design.
