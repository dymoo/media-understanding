# media-understanding

## Project

`@dymoo/media-understanding` is an MCP server and small programmatic library for turning media into the two modalities LLMs actually accept: text and images.

- Audio -> Whisper transcript text
- Video -> Whisper transcript text + timestamped keyframe grid images / exact frames
- Image -> compressed JPEG base64 + metadata
- URLs -> optional yt-dlp download/caching layer when `yt-dlp` is on `PATH`

The native MCP server transport is **stdio**. The official Docker image wraps the same server with `supergateway` so HTTP MCP clients can connect at `/mcp`.

## Commands

```bash
pnpm install       # install deps + node-av + default Whisper model (unless SKIP_MODEL_DOWNLOAD=1)
pnpm build         # tsc -> dist/
pnpm check-all     # format:check + lint + typecheck
pnpm format        # prettier write
pnpm test          # node:test on dist/tests/*.test.js (runs tsc first)

docker build -t media-understanding:dev .
docker run --rm -p 8000:8000 media-understanding:dev
node scripts/smoke-http-mcp.mjs  # expects container on localhost:8000
```

## Entry Points

| Binary / Runtime          | File                                | Purpose                                            |
| ------------------------- | ----------------------------------- | -------------------------------------------------- |
| `media-understanding`     | `dist/cli.js`                       | Lightweight CLI stub for probe-style pre-digestion |
| `media-understanding-mcp` | `dist/mcp.js`                       | MCP server over stdio                              |
| Docker image              | `ghcr.io/dymoo/media-understanding` | Streamable HTTP MCP at `/mcp` via `supergateway`   |

## Architecture

```text
src/
  accel.ts          - Hardware acceleration adapter layer for video decode/encode fallback
  cli.ts            - CLI stub that probes media and prints MCP guidance
  index.ts          - Public library exports
  mcp-budget.ts     - Exact serialized-char budget enforcement helpers
  mcp-format.ts     - MCP formatting helpers for transcript/grids/frames
  mcp-handlers.ts   - MCP tool handlers and URL-to-local resolution entrypoints
  mcp-preflight.ts  - File size / duration preflight checks for heavy tools
  mcp-types.ts      - MCP result types and helpers
  mcp.ts            - MCP server entrypoint + zod tool schemas (stdio only)
  media.ts          - Core media processing: probe, transcribe, grids, frames, understandMedia
  types.ts          - Shared types: MediaInfo, Segment, ProcessOptions, MediaError
  youtube.ts        - yt-dlp integration: URL detection, metadata, subtitles, downloads, cache
  tests/
    accel.test.ts   - Acceleration adapter/unit tests
    media.test.ts   - Integration tests for media.ts
    mcp.test.ts     - Integration tests for MCP handlers and formatting/budget behavior
    youtube.test.ts - yt-dlp unit + integration tests

scripts/
  download-whisper-model.mjs - Postinstall default Whisper model prewarm
  install-node-av.mjs        - node-av native binding + bundled ffmpeg setup
  smoke-http-mcp.mjs         - HTTP MCP smoke test for Docker/CI

.github/workflows/
  ci.yml           - check-all + test + npm smoke + Docker HTTP smoke
  release.yml      - verify -> publish npm -> publish GHCR image
```

## Key Packages

- `node-av` - Native FFmpeg bindings plus Whisper transcription runtime and bundled FFmpeg
- `sharp` - Image probing, resizing, and JPEG compression
- `@plussub/srt-vtt-parser` - Subtitle parsing in `youtube.ts`
- `@modelcontextprotocol/sdk` - MCP server SDK and transport helpers
- `zod` - Tool input validation
- `supergateway` - Docker-only stdio -> Streamable HTTP bridge (installed in the image, not package.json)

## Runtime Model / Caches

- Default transcription model: `base.en-q5_1`
- Runtime uses Whisper through `node-av` (`WhisperDownloader`, `WhisperTranscriber`)
- Model cache dir:
  - `$XDG_CACHE_HOME/media-understanding/models/`, or
  - `~/.cache/media-understanding/models/`
- The Docker build prewarms the default Whisper model so first transcription is immediate

## Conventions

- ESM only (`"type": "module"`)
- Node `>=22`
- Use `await using` / `using` with `node-av` resources
- `ffmpegPath()` from `node-av/ffmpeg` is the source of truth for FFmpeg paths
- `probe_media` accepts one `paths` field (string or string array); each entry may be a literal path, glob, or URL
- Heavy operations are single-file only per MCP call
- Heavy operations are gated by a process-level semaphore in `media.ts` (`HEAVY_OP_CONCURRENCY = 2`)
- Transcript cache is process-local, bounded LRU: max 32 entries, no caching for transcripts over 500K chars
- Portrait video auto-defaults to `cols=3`, `rows=3`, `thumb_width=120` when grid shape is omitted
- Payload budgets are enforced at the handler layer using serialized char counts
- Budget errors are descriptive and suggest how to shrink the request
- `understand_media` interleaves transcript chunks with grid images chronologically
- `get_transcript` supports `text`, `srt`, and `json`, plus `start_sec` / `end_sec` filtering

## URL / yt-dlp Behavior

- The npm package does **not** bundle yt-dlp; local stdio usage needs a system `yt-dlp`
- The official Docker image **does** include yt-dlp, so URL support is enabled out of the box there
- URL-capable handlers call `isUrl()` then resolve to a cached local file before entering `media.ts`
- `get_transcript` on a URL tries subtitle download first, then falls back to audio download + Whisper ASR
- `downloadVideo()` uses `-f worst[ext=mp4]/worst` for speed
- `downloadAudio()` uses `-x --audio-format m4a`
- URL downloads are cached under `$TMPDIR/media-understanding-ytdlp/<sha256-16>/`
- Codec handling is intentionally codec-agnostic: FFmpeg software decode is the fallback for anything yt-dlp downloads

## Docker / GHCR

- `Dockerfile` is multi-stage on `node:22-slim`
- The image includes:
  - built app from local source
  - `supergateway@3.4.3`
  - pinned `yt-dlp` release binary (checksum-verified)
  - default Whisper model cache
- Container defaults:
  - port `8000`
  - MCP endpoint `/mcp`
  - health endpoint `/healthz`
  - stateful Streamable HTTP transport
- Official registry: `ghcr.io/dymoo/media-understanding`
- Release workflow publishes npm + GHCR image from the GitHub Release event

## Test Fixtures

`testdata/` fixtures used in CI/tests:

- `tiny.png` - committed 16x16 red PNG
- `tiny.wav` - generated in CI, 1-second sine wave
- `tiny.mp3` - generated in CI, MP3 version of `tiny.wav`
- `tiny.mp4` - generated in CI, 5-second 320x240 color-bars test video with sine audio

## GitHub Actions

- `ci.yml`
  - Ubuntu Node 22 + 24: install, generate fixtures, `check-all`, `test`
  - macOS Node 24: `npm pack` consumer smoke test
  - Ubuntu Docker smoke: build image, run container, hit `/healthz`, run `scripts/smoke-http-mcp.mjs`
- `release.yml`
  - verify release
  - publish npm via OIDC
  - publish GHCR image (`1.1.0`, `1.1`, `latest` style tags)

## Known Sharp Edges

- `node-av` is native and platform-sensitive; Debian/glibc is the safe Docker base, not Alpine
- `pnpm install` may warn about ignored dependency build scripts, but this repo's own `postinstall` runs `install-node-av.mjs` explicitly
- `node-av` `SoftwareResampleContext.convertFrame()` is broken in 5.2.2; use `convertSync()`
- `Demuxer.packets()` iterators are not reusable after seek; reopen demuxer/decoder per target timestamp
- H.264 B-frame decode may require flush/drain before frames appear
- VideoToolbox hardware download failures on macOS are expected; `SoftwareAdapter` retry is the intended path
- Public Instagram Reels work without auth in current tests; TikTok, Vimeo, and X/Twitter may still fail due to auth or geo restrictions and should degrade gracefully
- Shipping yt-dlp in the Docker image does not change end-user responsibility for lawful use or site terms
