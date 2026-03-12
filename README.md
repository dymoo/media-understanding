# @dymoo/media-understanding

Turn audio, video, and images into the two modalities LLMs actually accept: **text** and **images**.

| Input                       | Output                                             |
| --------------------------- | -------------------------------------------------- |
| Audio (mp3, wav, m4a, ...)  | Timestamped transcript text                        |
| Video (mp4, mkv, mov, ...)  | Transcript text + timestamped keyframe grid images |
| Image (png, jpg, webp, ...) | Compressed JPEG (base64) + metadata                |

## Agent Setup

<details>
<summary>OpenCode</summary>

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "media-understanding": {
      "type": "local",
      "command": "npx",
      "args": ["-y", "@dymoo/media-understanding/mcp"],
      "env": {
        "MEDIA_UNDERSTANDING_MODEL": "base.en-q5_1"
      }
    }
  }
}
```

</details>

<details>
<summary>Claude Desktop</summary>

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "media-understanding": {
      "command": "npx",
      "args": ["-y", "@dymoo/media-understanding/mcp"],
      "env": {
        "MEDIA_UNDERSTANDING_MODEL": "base.en-q5_1"
      }
    }
  }
}
```

</details>

<details>
<summary>Cursor</summary>

Add to `.cursor/mcp.json` in your project or `~/.cursor/mcp.json` globally:

```json
{
  "mcpServers": {
    "media-understanding": {
      "command": "npx",
      "args": ["-y", "@dymoo/media-understanding/mcp"],
      "env": {
        "MEDIA_UNDERSTANDING_MODEL": "base.en-q5_1"
      }
    }
  }
}
```

</details>

<details>
<summary>Windsurf</summary>

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "media-understanding": {
      "command": "npx",
      "args": ["-y", "@dymoo/media-understanding/mcp"],
      "env": {
        "MEDIA_UNDERSTANDING_MODEL": "base.en-q5_1"
      }
    }
  }
}
```

</details>

<details>
<summary>Cline / Roo Code</summary>

Add a new MCP server:

- command: `npx`
- args: `-y @dymoo/media-understanding/mcp`
- env: `MEDIA_UNDERSTANDING_MODEL=base.en-q5_1`

</details>

## Quick Start

```bash
npm install @dymoo/media-understanding
```

Requirements:

- Node >= 20
- FFmpeg is handled automatically via `node-av`
- The default Whisper model (`base.en-q5_1`, ~57 MB) downloads on first use; set `SKIP_MODEL_DOWNLOAD=1` to defer

## How It Works

The server exposes five tools organized around a three-step workflow: **discover, analyze, iterate**.

```
Step 1: DISCOVER (cheap, batch-safe)
  probe_media  →  metadata for 1-200 files (~5-50ms each, header reads only)

Step 2: ANALYZE (expensive, one file at a time)
  understand_media  →  full analysis: metadata + transcript + keyframe grids
  get_transcript    →  timestamped speech text
  get_video_grids   →  visual keyframe contact sheets
  get_frames        →  exact frames at specific timestamps

Step 3: ITERATE (use output from one tool to target another)
  transcript timestamps → get_video_grids with start_sec/end_sec
  grid timestamps → get_frames with exact seconds
```

The cost boundary between cheap and expensive operations is encoded in the tool names. `probe_media` is safe to call on dozens of files. The analysis tools process one file per call and enforce payload budgets so an LLM does not accidentally blow up its context.

## MCP Tools

### `probe_media` — discover and triage

Scan files for metadata before committing to heavy analysis. Returns type, duration, resolution, codecs, file size. No decoding, no transcription, no images.

Accepts exactly one of `file_path`, `file_paths`, or `glob`. Default limit: 50 files (max 200).

```json
{ "file_path": "/path/to/video.mp4" }
{ "file_paths": ["/path/to/a.mp4", "/path/to/b.mp3"] }
{ "glob": "media/**/*.{mp4,mp3,wav}" }
{ "glob": "recordings/*.mp4", "max_files": 100 }
```

### `understand_media` — full single-file analysis

Returns metadata + transcript + keyframe grids for one file. Best for images, short audio, and short-to-medium video. For files over 2 hours, use the focused tools below instead.

```json
{ "file_path": "/path/to/video.mp4" }
{ "file_path": "/path/to/podcast.mp3", "model": "base.en" }
{ "file_path": "/path/to/clip.mp4", "start_sec": 60, "end_sec": 120, "max_grids": 3 }
```

Key options: `model`, `max_chars`, `max_total_chars`, `max_grids`, `start_sec`, `end_sec`, `sampling_strategy`, `seconds_per_frame`, `seconds_per_grid`, `cols`, `rows`, `thumb_width`, `aspect_mode`.

### `get_transcript` — speech content only

Timestamped transcript text. Supports files up to 4 hours. Each line includes `[start-end]` timestamps.

```json
{ "file_path": "/path/to/podcast.mp3" }
{ "file_path": "/path/to/meeting.mp4", "model": "base.en-q5_1", "max_chars": 16000 }
```

### `get_video_grids` — visual keyframe sampling

JPEG contact sheets of thumbnails. Every tile has an exact timestamp overlay. Budget-aware: omit `max_grids` and the server auto-fits as many grids as possible within `max_total_chars`.

```json
{ "file_path": "/path/to/video.mp4" }
{ "file_path": "/path/to/movie.mkv", "start_sec": 300, "end_sec": 600, "max_grids": 2, "seconds_per_frame": 8 }
{ "file_path": "/path/to/lecture.mp4", "sampling_strategy": "scene", "frame_interval": 150 }
```

### `get_frames` — exact moments

One JPEG per requested timestamp. Each frame includes a timestamp overlay.

```json
{ "file_path": "/path/to/video.mp4", "timestamps": [0, 30, 60] }
{ "file_path": "/path/to/clip.mp4", "timestamps": [83.5] }
```

## Recommended LLM Workflow

### Per-file type guidance

| Media type         | Recommended first tool                                              |
| ------------------ | ------------------------------------------------------------------- |
| Image              | `understand_media`                                                  |
| Short audio (<30m) | `understand_media`                                                  |
| Long audio         | `get_transcript`                                                    |
| Short video (<10m) | `understand_media`                                                  |
| Long video         | `get_transcript` (speech-first) or `get_video_grids` (visual-first) |

### Iteration pattern

1. Start with `understand_media` or `get_transcript` for a first-pass understanding
2. Use transcript timestamps to target `get_video_grids` on a narrow window
3. Use grid timestamps to target `get_frames` for exact moments

Narrow the window — don't widen it. Each follow-up call should be more targeted than the last.

## Processing Multiple Files

For a single file, call `understand_media` directly. For multiple files, follow the three-tier workflow.

### The three-tier approach

```
1. probe_media({ "glob": "media/**/*.mp4" })
   → metadata for all files (cheap, fast)

2. Pick files that matter based on probe results

3. Call analysis tools one file at a time:
   understand_media({ "file_path": "..." })
   get_transcript({ "file_path": "..." })
```

### Subagent orchestration (recommended for many files)

When processing many files, launch **subagents** — context-isolated workers each with MCP access — instead of accumulating raw media analysis in a single context.

**Why:** Each `understand_media` call returns transcript text and base64 images. Accumulating these for 10+ files floods the context window, causing compression and lost details. Subagents prevent this.

**How:**

1. Probe everything first: `probe_media` with a glob (cheap triage)
2. Launch one subagent per file (or per small group)
3. Pass each subagent a clear **intention**: _"Analyze this podcast episode. Extract the main topics discussed and any action items mentioned."_
4. Each subagent calls `understand_media` / `get_transcript` / etc., then returns a **distilled summary**
5. The orchestrator synthesizes summaries without ever accumulating raw media context

Subagents are **sacrificial** — their full analysis context is discarded after they return their distilled result. This is the key insight: the orchestrator works with summaries, not raw transcripts and images.

**When NOT to use subagents:**

- Single file — just call `understand_media` directly
- Quick metadata checks — `probe_media` is sufficient
- 2-3 small files — serial `understand_media` calls are fine

This is a **guideline**, not enforcement. The MCP server does not prevent serial `understand_media` calls in one context. But analysis quality degrades as context fills up with raw media data.

## Safety and Budgets

The server is conservative by default:

- **Payload budgets:** Every response is capped at `max_total_chars` (default 32,000). If a response would exceed the budget, the server returns a natural-language error explaining how to adjust (narrow the window, request fewer images, etc.).
- **Preflight checks:** Heavy tools reject obviously problematic requests before doing expensive work:
  - `understand_media`: files over 2 hours (transcription bottleneck)
  - `get_transcript`: files over 4 hours
  - All heavy tools: files over 10 GB
- **Bounded transcript cache:** In-memory LRU cache (max 32 entries, skips transcripts over 500K chars). Prevents unbounded memory growth.
- **Concurrency limits:** Frame extraction is capped at 4 concurrent FFmpeg processes to control memory peaks.

When the server rejects a request, it explains _why_ and suggests _what to do instead_ — it teaches the calling model to recover.

## CLI

```bash
media-understanding <file> [options]

Options:
  -m, --model <name>      Whisper model (default: base.en-q5_1)
  --max-chars <n>         Max transcript characters (default: 32000)
  -h, --help              Show help
```

## Programmatic API

```ts
import {
  extractFrameGridImages,
  extractFrameImage,
  probeMedia,
  transcribeAudio,
  understandMedia,
} from "@dymoo/media-understanding";

const info = await probeMedia("/path/to/video.mp4");
const segments = await transcribeAudio("/path/to/audio.mp3");
const frame = await extractFrameImage("/path/to/video.mp4", 30);
const grids = await extractFrameGridImages("/path/to/video.mp4", { maxGrids: 1 });
const result = await understandMedia("/path/to/video.mp4");
```

`understandMedia()` returns `{ info, segments, transcript, grids, gridImages }`.

## Environment Variables

| Variable                        | Default        | Description                    |
| ------------------------------- | -------------- | ------------------------------ |
| `MEDIA_UNDERSTANDING_MODEL`     | `base.en-q5_1` | Whisper model name             |
| `MEDIA_UNDERSTANDING_MAX_CHARS` | `32000`        | Max transcript characters      |
| `MEDIA_UNDERSTANDING_MAX_GRIDS` | `6`            | Max grid images per video call |

## Whisper Models

Default: `base.en-q5_1` (~57 MB, quantized). Models are cached at `~/.cache/media-understanding/models`.

Standard: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`

Quantized: `tiny.en-q5_1`, `base.en-q5_1`, `small.en-q5_1`, `large-v3-turbo-q5_0`

## Credits

Thanks to [Simon Willison](https://simonwillison.net/) for the inspiration around feeding timestamped video frames to LLMs. This library borrows that spirit — multiple separate images per turn, uniform time-distributed sampling, visible timestamps on every frame — and packages it into a conservative MCP workflow.

## License

MIT
