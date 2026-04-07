# media-understanding

## Project

MCP server that converts audio/video/image files into text + images — the two modalities LLMs accept. Audio becomes transcript text. Video becomes transcript text + keyframe grid images. Images pass through as base64 + metadata. A lightweight CLI stub provides metadata pre-digestion for OpenClaw-style agent integrations.

Package: `@dymoo/media-understanding`

## Commands

```bash
pnpm install       # install deps
pnpm build         # tsc → dist/
pnpm check-all     # format:check + lint + typecheck
pnpm format        # prettier write
pnpm test          # node:test on dist/tests/*.test.js (runs tsc first)
pnpm bench:asr:matrix -- /path/to/audio.wav  # local benchmark matrix (build first)
```

## Entry Points

| Binary                    | File          | Purpose                                                         |
| ------------------------- | ------------- | --------------------------------------------------------------- |
| `media-understanding`     | `dist/cli.js` | OpenClaw media model stub (probe-only, ~50ms, no transcription) |
| `media-understanding-mcp` | `dist/mcp.js` | MCP server over stdio (JSON-RPC 2.0)                            |

## Architecture

```
src/
  types.ts          — Shared interfaces: MediaInfo (includes optional `acceleration?: AccelInfo`),
                       Segment, GridOptions, MediaError, FILE_TOO_LARGE error code
  accel.ts          — Hardware acceleration adapter layer: AccelAdapter interface + AccelInfo,
                       SoftwareAdapter, VideoToolboxAdapter, CudaAdapter, VaapiAdapter, QsvAdapter;
                       getAdapter() factory (process-level singleton), resetAdapter() for tests,
                       getSoftwareAdapter() for explicit SW fallback
  asr-accel.ts      — ASR acceleration: ONNX Runtime execution provider detection
                       (CUDA/DirectML/CPU), split session options (encoder vs light models),
                       provider reporting. macOS uses CPU-only (CoreML is 2.7× slower for INT8).
  asr-tokenizer.ts  — SentencePiece vocab loader + token ID → text decoder; hasWordPrefix() for
                       word-boundary detection via raw ▁ prefix (decode() strips leading spaces)
  asr-decoder.ts    — TDT greedy decode loop: frame-by-frame encoder→decoder+joiner→token
  asr-pipeline.ts   — ASR orchestrator: lazy ONNX session singleton (preprocessor+encoder+decoder),
                       PCM → nemo128.onnx mel → encode → decode → Segment[].
                       Split EP config: encoder uses best EP, preprocessor+decoder use CPU-only.
                       Optional per-stage timing via MEDIA_UNDERSTANDING_DEBUG=1.
  asr-audio.ts      — PCM extraction: decode + resample to 16 kHz mono Float32 via node-av.
                       Uses convertSync() buffer API (NOT convertFrame — broken in node-av 5.2.2)
  asr-chunking.ts   — Long-form windowing: ~30s chunks with 2s overlap, segment stitching
  model-manager.ts  — Parakeet ONNX model download manager: HuggingFace download, SHA-256,
                       atomic writes, retry/backoff, concurrent downloads, cache management
  media.ts          — Core processing: probeMedia, transcribeAudio, extractFrameGrid, extractFrame,
                       extractFramesBatch (native batch frame extraction), extractFrameImages,
                       understandMedia; process-level Semaphore for heavy-op concurrency control
  index.ts          — Public API re-exports (entry point for "." export in package.json)
  mcp-types.ts      — MCP result types (McpErrorResult, McpContentItem, McpSuccessResult),
                        tool arg types, mcpError() helper
  mcp-budget.ts     — Payload budget enforcement: getTotalCharBudget, serializedContentLength,
                        estimateVisionTokens, richBudgetError, assertFitsBudget, appendIfFits
  mcp-preflight.ts  — Preflight safety checks: PREFLIGHT_* constants, formatDuration,
                        preflightFileSize, preflightDuration
  mcp-format.ts     — Formatting helpers: expandPaths, formatGridWindow, formatFrameLabel,
                        formatMediaInfo (includes acceleration backend), buildOpts, transcript
                        formatters (text/SRT/JSON), filterSegmentsByWindow
  mcp-handlers.ts   — 5 handle* functions + buildUnderstandMediaContent, overlapMs,
                        assignSegmentsToGrids; re-exports from mcp-types/budget/preflight/format
                        for backward compatibility
  youtube.ts        — yt-dlp integration layer: URL detection (isUrl), URL hashing (urlHash),
                       video info fetching (getVideoInfo), subtitle download + parsing
                       (downloadSubtitles, parseSubtitlesToSegments via @plussub/srt-vtt-parser),
                       media download (downloadVideo, downloadAudio, downloadThumbnail),
                       URL→local-path resolution (resolveUrlToLocalPath, resolveUrlToAudioPath),
                       yt-dlp detection (hasYtDlp, ensureYtDlp). Downloads cached at
                       $TMPDIR/media-understanding-ytdlp/<sha256-16-chars>/
  mcp.ts            — MCP server entrypoint + tool schemas
  cli.ts            — CLI stub: runs probeMedia() and outputs metadata + MCP tool guidance
                       Audio files: runs transcription inline and prints full transcript
                       Video/Image: type-aware directive MCP guidance
  tests/
    media.test.ts   — Integration tests for media.ts
    mcp.test.ts     — Integration tests for MCP tool surface (preflight, budget, probe, interleaving,
                       transcript formatting, time windowing)
    accel.test.ts   — Unit tests for accel.ts adapter factory + sharp helpers
    youtube.test.ts — Unit tests for isUrl, urlHash, parseSubtitlesToSegments + integration tests
                       for yt-dlp with YouTube, Instagram, Vimeo, TikTok, X/Twitter. Integration
                       tests skip when yt-dlp not installed; platform tests (TikTok, Vimeo, X)
                       use try/catch with graceful skip on YT_DLP_FAILED for auth/geo issues
docs/assets/      — README visuals: example grid and frame images generated from real video
testdata/         — Small committed fixtures for integration tests
  tiny.png        — 16×16 red PNG (committed)
  tiny.wav        — 1-second sine wave (gitignored, generated in CI)
  tiny.mp3        — same as wav, MP3 (gitignored, generated in CI)
  tiny.mp4        — 1-second 64×64 test video (gitignored, generated in CI)
```

## Key Packages

- `node-av` — FFmpeg v8 N-API bindings; provides `Demuxer`, `Decoder`, `Frame`, `SoftwareScaleContext`, `SoftwareResampleContext`, `ffmpegPath()`. Frame extraction uses native Demuxer/Decoder/SWS pipeline (no CLI spawning). Audio resampling to 16 kHz mono for ASR uses `SoftwareResampleContext`.
- `onnxruntime-node` — ONNX Runtime for Node.js; runs Parakeet TDT encoder + decoder. GPU acceleration on Linux (CUDA) and Windows (DirectML). macOS Apple Silicon uses CPU (benchmarks show CPU is 2.7× faster than CoreML for INT8 Parakeet — CoreML only supports 44% of graph nodes, causing heavy partitioning). Override with `MEDIA_UNDERSTANDING_EP=coreml` if needed.
- `@plussub/srt-vtt-parser` — SRT/VTT subtitle parsing; used in `youtube.ts` for `parseSubtitlesToSegments`. Requires SRT period-separator normalization (`.` → `,`) before parsing; does not strip HTML tags (post-parse `.replace(/<[^>]*>/g, "")` needed).
- `@modelcontextprotocol/sdk` — MCP server (v2 API: `registerTool` + `z.object()`)
- `zod` — Runtime schema validation for MCP tool inputs

## Conventions

- ESM only (`"type": "module"`)
- Node >= 22 required (uses `await using` / Explicit Resource Management, `fs.promises.glob()`)
- All `node-av` resources use `await using` / `using` for automatic cleanup
- `ffmpegPath()` from `node-av/ffmpeg` — never hardcode FFmpeg binary paths
- Parakeet TDT 0.6B v3 ONNX model auto-downloads from HuggingFace on first use (~670 MB); cached at `~/.cache/media-understanding/models/parakeet-tdt-0.6b-v3-int8/`
- Transcript cache: bounded LRU (`TranscriptCache` class in `media.ts`) with max 32 entries; transcripts >500K chars are not cached. Keyed by SHA-256 content fingerprint (`size + first 64 KB + last 64 KB`), lives for process lifetime
- Fail fast: throw typed `MediaError` with a `code` field for FFmpeg not found, unsupported format, etc.
- All MCP tool errors returned as `{ isError: true, content: [{ type: "text", text: "..." }] }`
- `understand_media` is single-file only; use `probe_media` for lightweight batch metadata scanning
- `probe_media` accepts a single `paths` param (string or string array) — each entry can be a literal file path or a glob pattern. No separate `file_path`/`file_paths`/`glob` params.
- Heavy operations (transcription, grid extraction, frame extraction) are single-file per MCP call; multi-file heavy work should use subagent orchestration at the agent layer
- Preflight checks in all heavy MCP handlers reject files >10 GB, videos >2h (understand_media) or >4h (get_transcript), with natural-language recovery guidance
- Heavy operations (transcription, frame extraction) are guarded by a process-level counting semaphore (`HEAVY_OP_CONCURRENCY = 2`, FIFO queue) in `media.ts` via `withHeavyOp()`
- Video frames and grid tiles carry exact timestamp metadata and visible overlays
- Payload budgets are enforced at the MCP handler layer using exact serialized char counts
- Budget errors include overage ratio, estimated LLM vision tokens, and resolution-aware suggestions
- `understand_media` interleaves transcript chunks with their corresponding grid images in chronological order
- `get_transcript` supports three output formats: `text` (default), `srt`, `json` — plus optional `start_sec`/`end_sec` time windowing
- `.prettierignore` excludes docs image assets and generated media so `pnpm format` remains safe repo-wide
- Hardware acceleration uses an adapter pattern (`src/accel.ts`). `media.ts` must NOT import `HardwareContext`, `FilterAPI`, `Encoder`, or HW codec constants directly — all live in `accel.ts`. `getAdapter()` returns the process-level singleton (synchronous, not async).
- Frame scale/encode pipeline outputs `yuvj420p` (JPEG-tagged full-range YUV) via filter, and sets `frame.colorRange = AVCOL_RANGE_JPEG` before MJPEG encoding. This avoids `Invalid argument` from the MJPEG encoder, which requires full-range input.
- MJPEG one-shot encoding requires `threadCount: 1` — frame-threaded MJPEG init fails with `ff_frame_thread_encoder_init`.
- All `FF_ENCODER_MJPEG*` constants share the same TypeScript type `FFVideoEncoder`; use `FFVideoEncoder` as the parameter type instead of duplicating the union.
- When a HW adapter fails, the SW retry opens a fresh Demuxer+SW Decoder (no HW context), logs to stderr, and proceeds. VideoToolbox frames in `videotoolbox_vld` pixel format cannot be downloaded via `hwdownload` to system memory — this is expected on macOS and the SW retry is the correct path.
- `MEDIA_UNDERSTANDING_DISABLE_HW=1` forces `SoftwareAdapter` regardless of detected hardware.
- All MCP handlers that accept `file_path` also accept URLs when yt-dlp is installed. The handler calls `isUrl()` then `resolveUrlToLocalPath()` (or `resolveUrlToAudioPath()` for `get_transcript`) to download to a cached temp file before processing. Core `media.ts` functions never see URLs — only local paths.
- `get_transcript` with a URL tries `downloadSubtitles()` first (instant, no media download); falls back to `resolveUrlToAudioPath()` → full ASR only when no subtitles are available. This makes it the fastest path for online video content.
- yt-dlp format selection: `downloadVideo()` uses `-f worst[ext=mp4]/worst` (prefers smallest MP4/H.264 for speed); `downloadAudio()` uses `-x --audio-format m4a`. All downloads pass `--ffmpeg-location` pointing to node-av's bundled FFmpeg.
- URL downloads are cached at `$TMPDIR/media-understanding-ytdlp/<sha256-16-chars>/` with in-memory `resolvedPaths` Map for cross-call dedup.
- Codec safety: the pipeline is fully codec-agnostic. `SoftwareAdapter` delegates to FFmpeg's auto-detected software decoder for any codec yt-dlp might download (H.264, VP9, AV1, Opus, AAC, etc.). HW→SW fallback in `extractFramesBatch()` handles cases where VideoToolbox/CUDA doesn't support the codec.

## Podcast Analysis Workflow

For analyzing podcasts or long audio:

1. **Overview pass** — `get_transcript(file, { format: "srt" })` to get a timestamped overview of the full episode. Scan the SRT to identify interesting segments, topic transitions, or key moments.
2. **Detail pass** — `get_transcript(file, { format: "json", start_sec: 120, end_sec: 300 })` to get precise segment data for a specific time range. Use for quoting, summarizing, or extracting specific portions.

This two-step pattern avoids dumping the entire precise transcript into context at once. The SRT overview is compact and scannable; the JSON detail pass gives you exact data for the segments that matter.

## LLM Vision Token Costs

Reference for understanding the cost of sending images to LLMs. These are separate from the MCP server's `max_total_chars` transport budget — that controls serialized response size, not LLM token consumption.

| Provider | Formula                                         | Example (1920x1080) |
| -------- | ----------------------------------------------- | ------------------- |
| Claude   | `width * height / 750`                          | ~2,765 tokens       |
| OpenAI   | `85 + 170 * ceil(width/512) * ceil(height/512)` | ~1,445 tokens       |

The MCP server's `estimateVisionTokens()` helper uses the Claude formula (`pixels / 750`) as a cross-provider ballpark. It appears in budget error messages to help agents understand the vision cost of requested images and make informed decisions about `thumb_width`, `cols`, `rows`, and `max_grids`.

**Practical guidance for agents:**

- Portrait video auto-defaults to `cols=3, rows=3, thumb_width=120` when grid-shape params are omitted — no manual adjustment needed
- For landscape video, defaults are `cols=4, rows=4, thumb_width=480`
- Explicit `cols`/`rows`/`thumb_width` values always override automatic portrait defaults
- Smaller `thumb_width` = fewer vision tokens per grid (e.g., 120px vs 480px default)
- Fewer `cols`/`rows` = fewer tiles per grid = cheaper per grid
- `aspect_mode: "contain"` on portrait video adds letterboxing (black bars cost tokens too)
- `get_video_grids` is the overview tool (grid contact sheets); `get_frames` is the detail tool (exact moments)
- `get_frames` budget errors suggest fewer timestamps or splitting requests, not grid-specific knobs

## ASR Model

Uses NVIDIA Parakeet TDT 0.6B v3 (INT8 quantized ONNX, ~670 MB total). Supports 25 languages. No model selection — single model for all use cases. Auto-downloads from HuggingFace on first transcription call.

Model files cached at `~/.cache/media-understanding/models/parakeet-tdt-0.6b-v3-int8/`:

- `encoder-model.int8.onnx` (652 MB) — FastConformer encoder
- `decoder_joint-model.int8.onnx` (18 MB) — TDT decoder + joiner
- `nemo128.onnx` (140 KB) — NeMo mel spectrogram preprocessor (PCM → 128-bin log-mel features)
- `vocab.txt` (94 KB) — SentencePiece vocabulary (8192 tokens)

## Environment Variables

| Variable                         | Default | Description                                             |
| -------------------------------- | ------- | ------------------------------------------------------- |
| `MEDIA_UNDERSTANDING_MAX_CHARS`  | `32000` | Max transcript characters                               |
| `MEDIA_UNDERSTANDING_MAX_GRIDS`  | `6`     | Max grid images per video call                          |
| `MEDIA_UNDERSTANDING_DISABLE_HW` | (unset) | Set to `1` to force software decode/encode              |
| `MEDIA_UNDERSTANDING_EP`         | (unset) | Override ASR execution provider (e.g. `coreml`, `cuda`) |
| `MEDIA_UNDERSTANDING_DEBUG`      | (unset) | Set to `1` for per-stage ASR timing logs                |

## Benchmarking

- Benchmark scripts live under `scripts/` and are benchmark-only; they must not change shipped runtime behavior in `src/`
- `scripts/benchmark-asr-matrix.mjs` compares `Parakeet MLX INT4 GPU`, `Parakeet MLX INT4 CPU`, `Parakeet ONNX INT8 CPU`, and `Whisper.cpp q5_1`
- `scripts/benchmark-parakeet-mlx.py` expects `parakeet-mlx` in a dedicated Python environment; recommended path is `tmp/benchmarks/.venv/bin/python`. It loads `mlx-community/parakeet-tdt-0.6b-v3` and locally quantizes the encoder to 4-bit for the benchmark.
- `scripts/benchmark-whisper-cpp.mjs` defaults to `/opt/homebrew/bin/whisper-cli` and `~/.cache/media-understanding/models/ggml-base.en-q5_1.bin`
- Benchmark artifacts are written to `tmp/benchmarks/<timestamp>/` and are gitignored
- Use audio files (`wav`, `mp3`, `flac`, `ogg`) for apples-to-apples comparison because `whisper-cli` does not accept arbitrary video containers

## GitHub Actions

- `ci.yml` — Runs on push/PR: `check-all` + `test` on Node 22 and 24; uses `SKIP_MODEL_DOWNLOAD=1` on install
- `ci.yml` also includes a macOS Node 24 packaged-consumer smoke test: `build` → `npm pack` → fresh temp project → `npm install` tarball with `SKIP_MODEL_DOWNLOAD=1`
- `release.yml` — Runs on GitHub release published: verifies `check-all` + `test` + `build`, then publishes to npm via OIDC trusted publishing

## Known Limitations / Sharp Edges

- Exact payload budgets are per MCP response, not across multiple calls
- Short video fixtures can fail if timestamps are too close to EOF; `extractFrame()` clamps near-end timestamps defensively
- Subtitle burn-in (libass) deferred to v2. We currently add timestamp overlays after extraction/compositing.
- `node-av` prebuilt binaries require glibc >= 2.31 on Linux (Ubuntu 20.04+)
- Windows: temp paths must use forward slashes in FFmpeg filter arguments (handled in `media.ts`)
- Very large files (>2 hours): transcript truncation is active by default at `maxChars`
- `node-av` native binding ships as a platform-specific optional package (`@seydx/node-av-<platform>-<arch>`). All 8 platform variants are declared as `optionalDependencies` in `package.json`. Their `postinstall` extracts the `.node` binary from a zip.
- Our `postinstall` script (`scripts/install-node-av.mjs`) resolves the current platform package and `node-av` by module resolution, walks upward to each package root, runs the platform package's `install.js` first, then runs `node-av`'s `dist/ffmpeg/install.js`. Do not assume either package lives under this package's nested `node_modules` — npm consumers may get hoisted layouts.
- `SKIP_TRANSCRIPTION=1` env var skips ASR model download in tests — used in CI where model download would be too slow.
- Tests use `import * as assert from "node:assert/strict"` (not default import) due to `verbatimModuleSyntax`.
- ESLint rule `no-floating-promises` is disabled for `src/tests/**` because `describe()`/`it()` from `node:test` return `Promise<void>` but are intentionally not awaited.
- `fs.promises.glob()` exists in Node 22 but is NOT a named export — must use `import * as fsPromises from "node:fs/promises"` then `fsPromises.glob()`. `@types/node` is `^22.0.0`.
- `pnpm install` postinstall can fail with `ERR_PACKAGE_PATH_NOT_EXPORTED` from `node-av` — use `pnpm install --ignore-scripts` to work around. Tests still pass since `node-av` is already installed.
- `tiny.wav` is a sine wave with no speech — `transcribeAudio()` returns an empty array for it. Tests for transcript formatting use synthetic `Segment` objects.
- `node-av` `Frame.alloc()` must be called before setting `.width`/`.height`/`.format` — without it, property setters silently fail and `allocBuffer()` returns EINVAL
- `node-av` `Demuxer.packets()` async generator is NOT reusable after `demuxer.seek()` — a new `packets()` iterator returns nothing. Open a fresh Demuxer+Decoder per target timestamp instead.
- B-frame codecs (H.264): `decoder.receive()` may return nothing during normal packet feeding. Must flush with `decoder.decode(null)` then drain `decoder.receive()` to get buffered frames.
- `node-av` `SoftwareResampleContext.convertFrame()` is broken in node-av 5.2.2 — returns AVERROR for all frames despite successful init. Use `convertSync([outBuf], maxOut, [inBuf], inSamples)` instead.
- Instagram Reels work with bare yt-dlp (no auth) for public reels. Private/restricted content may require cookies via `--cookies-from-browser` (not currently supported).
- TikTok, X/Twitter, and Vimeo may require auth or geo-based access. Integration tests for these platforms use try/catch with graceful skip on `YT_DLP_FAILED` rather than hard fail.
