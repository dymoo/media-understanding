#!/usr/bin/env node
/**
 * MCP server (stdio transport) exposing 5 tools for LLM media understanding.
 *
 * Three-tier workflow:
 *   1. Discover — probe_media (cheap, batch-safe metadata)
 *   2. Analyze  — understand_media / get_transcript / get_video_grids / get_frames (heavy, single-file)
 *   3. Iterate  — use timestamps from one tool to target another
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import {
  handleGetFrames,
  handleGetTranscript,
  handleGetVideoGrids,
  handleProbeMedia,
  handleUnderstandMedia,
} from "./mcp-handlers.js";

const server = new McpServer({
  name: "media-understanding",
  version: "0.1.0",
});

server.registerTool(
  "probe_media",
  {
    description: `Discover and triage media files before committing to heavy analysis.

Returns metadata only: type, duration, resolution, codecs, file size. No decoding,
no transcription, no images. Reads file headers only (~5-50ms per file), so it is
safe to call on dozens of files at once.

Use this FIRST to scan a directory or batch of files, then pick individual files
for heavy analysis with understand_media, get_transcript, or get_video_grids.

Accepts a single \`paths\` parameter: a string or array of strings. Each string
can be a literal file path or a glob pattern. Default limit is 50 files
(absolute max 200).

Examples:
  { "paths": "/path/to/video.mp4" }
  { "paths": ["/path/to/a.mp4", "/path/to/b.mp3"] }
  { "paths": "media/**/*.{mp4,mp3,wav}" }
  { "paths": ["recordings/*.mp4", "specific/file.mp3"], "max_files": 100 }`,
    inputSchema: z.object({
      paths: z
        .union([z.string(), z.array(z.string()).min(1)])
        .describe(
          "One or more file paths or glob patterns. Each string can be a literal path or a glob pattern (e.g. `media/**/*.mp4`). Pass a single string or an array.",
        ),
      max_files: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Hard cap on matched files (default 50, absolute max 200). Probe is lightweight so generous limits are safe.",
        ),
    }),
  },
  handleProbeMedia,
);

server.registerTool(
  "understand_media",
  {
    description: `Fully analyze a media file (audio, video, or image).

Returns:
- Metadata (type, duration, resolution, codecs)
- Transcript text for audio/video (Whisper, auto-downloaded on first use)
- Keyframe grid images for video (JPEG base64, max 6 grids)
- Image data for image files

Use this as your first call for any media file. For long videos (>10 min) or
when you only need one modality, prefer the focused tools instead.

For multiple files, use probe_media first to discover and triage, then call this
tool once per file that needs full analysis.

Examples:
  { "file_path": "/path/to/video.mp4" }
  { "file_path": "/path/to/podcast.mp3", "model": "base.en" }
  { "file_path": "/path/to/clip.mp4", "start_sec": 60, "end_sec": 120, "max_grids": 3 }`,
    inputSchema: z.object({
      file_path: z.string().describe("Absolute or relative path to a single media file."),
      model: z
        .string()
        .optional()
        .describe(
          'Whisper model name. Default: "base.en-q5_1". Options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, tiny.en-q5_1, base.en-q5_1, small.en-q5_1, large-v3-turbo-q5_0.',
        ),
      max_chars: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Max transcript characters within the total budget (default 32000)."),
      max_total_chars: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Hard cap for the entire MCP response, including text and base64 images (default 48000).",
        ),
      max_grids: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Max grid images to return for video. If omitted, the server auto-fits as many as possible within budget.",
        ),
      start_sec: z
        .number()
        .nonnegative()
        .optional()
        .describe("Start offset in seconds for grid extraction (default 0)."),
      end_sec: z
        .number()
        .positive()
        .optional()
        .describe("End offset in seconds for grid extraction (default: end of file)."),
      sampling_strategy: z
        .enum(["uniform", "scene"])
        .optional()
        .describe(
          "Sampling strategy. `uniform` is the default and covers the whole window evenly.",
        ),
      seconds_per_frame: z
        .number()
        .positive()
        .optional()
        .describe("Spacing between frames within a grid, in seconds."),
      seconds_per_grid: z
        .number()
        .positive()
        .optional()
        .describe("Spacing between composite overview grids, in seconds."),
      aspect_mode: z
        .enum(["contain", "cover"])
        .optional()
        .describe("How frames fit in each grid tile. `contain` keeps the full frame visible."),
      cols: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .describe("Grid columns per tile (default 4; portrait video defaults to 3 when omitted)."),
      rows: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .describe("Grid rows per tile (default 4; portrait video defaults to 3 when omitted)."),
      thumb_width: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Thumbnail width in pixels per cell (default 480; portrait video defaults to 120 when omitted).",
        ),
    }),
  },
  handleUnderstandMedia,
);

server.registerTool(
  "get_video_grids",
  {
    description: `Extract keyframe grid images from a video file.

Each grid is a JPEG contact sheet of thumbnails arranged in a cols×rows tile.
Every tile has an exact timestamp overlay, and the accompanying text lists the
exact timestamps in row-major order.

Use this for visual inspection without transcription. It is budget-aware: if
you omit max_grids, the server returns as many grids as fit under max_total_chars.

Examples:
  { "file_path": "/path/to/video.mp4" }
  { "file_path": "/path/to/movie.mkv", "start_sec": 300, "end_sec": 600, "max_grids": 2, "seconds_per_frame": 8 }
  { "file_path": "/path/to/lecture.mp4", "sampling_strategy": "scene", "frame_interval": 150 }`,
    inputSchema: z.object({
      file_path: z.string().describe("Path to a video file."),
      max_total_chars: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Hard cap for the entire MCP response, including text and base64 images (default 48000).",
        ),
      max_grids: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Max grid images to return. If omitted, the server auto-fits as many as possible within budget.",
        ),
      start_sec: z
        .number()
        .nonnegative()
        .optional()
        .describe("Start offset in seconds (default 0)."),
      end_sec: z
        .number()
        .positive()
        .optional()
        .describe("End offset in seconds (default: end of file)."),
      sampling_strategy: z
        .enum(["uniform", "scene"])
        .optional()
        .describe(
          "Sampling strategy. `uniform` is the default and covers the whole window evenly.",
        ),
      scene_threshold: z
        .number()
        .min(0)
        .max(1)
        .optional()
        .describe(
          'Scene-change threshold 0–1. Higher = fewer keyframes. Used when sampling_strategy="scene".',
        ),
      frame_interval: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          'Fallback: include a frame every N frames even without scene change (default 300 ≈ 10s at 30fps). Used when sampling_strategy="scene".',
        ),
      seconds_per_frame: z
        .number()
        .positive()
        .optional()
        .describe("Spacing between frames within a grid, in seconds."),
      seconds_per_grid: z
        .number()
        .positive()
        .optional()
        .describe("Spacing between composite overview grids, in seconds."),
      cols: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .describe("Grid columns (default 4; portrait video defaults to 3 when omitted)."),
      rows: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .describe("Grid rows (default 4; portrait video defaults to 3 when omitted)."),
      aspect_mode: z
        .enum(["contain", "cover"])
        .optional()
        .describe("How frames fit in each grid tile. `contain` keeps the full frame visible."),
      thumb_width: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Thumbnail width per cell in pixels (default 480; portrait video defaults to 120 when omitted).",
        ),
    }),
  },
  handleGetVideoGrids,
);

server.registerTool(
  "get_frames",
  {
    description: `Extract individual video frames at specific timestamps.

Returns one JPEG image per requested timestamp. Useful for inspecting exact
moments identified from a transcript or grid (e.g. "show me the frame at 1:23").

Timestamps are in seconds (fractional values allowed).
Each returned frame image includes an exact timestamp overlay.

Examples:
  { "file_path": "/path/to/video.mp4", "timestamps": [0, 30, 60] }
  { "file_path": "/path/to/clip.mp4", "timestamps": [83.5] }`,
    inputSchema: z.object({
      file_path: z.string().describe("Path to a video file."),
      max_total_chars: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Hard cap for the entire MCP response, including text and base64 images (default 48000).",
        ),
      timestamps: z
        .array(z.number().nonnegative())
        .min(1)
        .max(20)
        .describe("Timestamps in seconds at which to extract frames. Max 20 per call."),
    }),
  },
  handleGetFrames,
);

server.registerTool(
  "get_transcript",
  {
    description: `Transcribe an audio or video file and return the text.

Uses OpenAI's Whisper (via whisper.cpp). The model auto-downloads on first use
(~57 MB for base.en-q5_1). Transcript is cached per file for the process lifetime.

Three output formats:
- "text" (default): timestamped lines — "[start–end] text" per segment
- "srt": standard SRT subtitle format (1-based index, HH:MM:SS,mmm timestamps)
- "json": machine-readable JSON with millisecond-precision segment timestamps

Optional time windowing with start_sec/end_sec filters the output segments
(transcription still processes the full file, but results are cached).

Two-step workflow for long media:
1. Overview: get_transcript(file, { format: "srt" }) — scan for topics/transitions
2. Detail: get_transcript(file, { format: "json", start_sec: 120, end_sec: 300 }) — precise data for a specific range

Examples:
  { "file_path": "/path/to/podcast.mp3" }
  { "file_path": "/path/to/meeting.mp4", "format": "srt" }
  { "file_path": "/path/to/episode.mp3", "format": "json", "start_sec": 60, "end_sec": 120 }
  { "file_path": "/path/to/meeting.mp4", "model": "base.en-q5_1", "max_chars": 16000 }`,
    inputSchema: z.object({
      file_path: z.string().describe("Path to an audio or video file."),
      model: z
        .string()
        .optional()
        .describe(
          'Whisper model. Default: "base.en-q5_1". Larger models are slower but more accurate.',
        ),
      max_chars: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Max characters to return (default 32000). Keeps first 60% + last 40%."),
      format: z
        .enum(["text", "srt", "json"])
        .optional()
        .describe(
          'Output format. "text" (default): timestamped lines. "srt": SRT subtitles. "json": machine-readable with ms timestamps.',
        ),
      start_sec: z
        .number()
        .nonnegative()
        .optional()
        .describe(
          "Filter output to segments starting at or after this time. Transcription still covers the full file (cached).",
        ),
      end_sec: z
        .number()
        .positive()
        .optional()
        .describe(
          "Filter output to segments ending at or before this time. Transcription still covers the full file (cached).",
        ),
    }),
  },
  handleGetTranscript,
);

const transport = new StdioServerTransport();
await server.connect(transport);
