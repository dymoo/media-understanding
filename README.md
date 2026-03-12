# @dymoo/media-understanding

MCP server and CLI that converts audio, video, and image files into the two modalities LLMs accept: **text** and **images**.

| Input                     | Output                                 |
| ------------------------- | -------------------------------------- |
| Audio (mp3, wav, m4a, …)  | Transcript text                        |
| Video (mp4, mkv, mov, …)  | Transcript text + keyframe grid images |
| Image (png, jpg, webp, …) | Compressed JPEG (base64) + metadata    |

## Installation

```bash
npm install @dymoo/media-understanding
```

> **Requirements:** Node ≥ 20. FFmpeg and Whisper models are auto-downloaded on first use.

## MCP Server

Add to your MCP client config (e.g. Claude Desktop, OpenCode):

```json
{
  "mcpServers": {
    "media-understanding": {
      "command": "npx",
      "args": ["-y", "@dymoo/media-understanding/mcp"]
    }
  }
}
```

Or if installed globally:

```json
{
  "mcpServers": {
    "media-understanding": {
      "command": "media-understanding-mcp"
    }
  }
}
```

### Tools

#### `understand_media`

Full analysis in one call: metadata + transcript + keyframe grids.

```json
{ "file_path": "/path/to/video.mp4" }
{ "file_path": "/path/to/podcast.mp3", "model": "base.en" }
{ "file_path": "/path/to/clip.mp4", "start_sec": 60, "end_sec": 120 }
```

#### `get_video_grids`

Extract keyframe grid images without transcription.

```json
{ "file_path": "/path/to/movie.mkv", "max_grids": 4 }
{ "file_path": "/path/to/lecture.mp4", "scene_threshold": 0.5 }
```

#### `get_frames`

Extract individual frames at specific timestamps (seconds).

```json
{ "file_path": "/path/to/video.mp4", "timestamps": [0, 30, 60] }
{ "file_path": "/path/to/clip.mp4", "timestamps": [83.5] }
```

#### `get_transcript`

Transcribe audio/video, returns timestamped segments.

```json
{ "file_path": "/path/to/meeting.mp4" }
{ "file_path": "/path/to/podcast.mp3", "model": "base.en", "max_chars": 16000 }
```

## CLI

```bash
media-understanding <file> [options]

Options:
  -m, --model <name>      Whisper model (default: tiny.en)
  --max-chars <n>         Max transcript characters (default: 32000)
  -h, --help              Show help
```

## Environment Variables

| Variable                        | Default   | Description                    |
| ------------------------------- | --------- | ------------------------------ |
| `MEDIA_UNDERSTANDING_MODEL`     | `tiny.en` | Whisper model name             |
| `MEDIA_UNDERSTANDING_MAX_CHARS` | `32000`   | Max transcript characters      |
| `MEDIA_UNDERSTANDING_MAX_GRIDS` | `6`       | Max grid images per video call |

## Whisper Models

Default: `tiny.en` (~75 MB). Models are cached at `~/.cache/media-understanding/models`.

Available: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`

## Programmatic API

```typescript
import {
  probeMedia,
  transcribeAudio,
  extractFrameGrid,
  extractFrame,
  understandMedia,
} from "@dymoo/media-understanding";

// Probe metadata
const info = await probeMedia("/path/to/video.mp4");
// { type: "video", duration: 120.5, width: 1920, height: 1080, fps: 29.97, ... }

// Transcribe
const segments = await transcribeAudio("/path/to/audio.mp3");
// [{ start: 0, end: 3200, text: "Hello world." }, ...]

// Extract keyframe grids (returns JPEG Buffers)
const grids = await extractFrameGrid("/path/to/video.mp4", { maxGrids: 3 });

// Extract a single frame at 30 seconds
const frame = await extractFrame("/path/to/video.mp4", 30);

// Full analysis
const result = await understandMedia("/path/to/video.mp4");
// { info, segments, transcript, grids }
```

## License

MIT
