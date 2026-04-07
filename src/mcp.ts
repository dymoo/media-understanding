#!/usr/bin/env node
/**
 * MCP server (stdio transport) for LLM media understanding.
 *
 * Three-tier workflow:
 *   1. Discover — probe_media (cheap, batch-safe metadata)
 *   2. Analyze  — understand_media / get_transcript / get_video_grids / get_frames (heavy, single-file)
 *   3. Iterate  — use timestamps from one tool to target another
 *
 * When yt-dlp is installed on the system, all tools additionally accept URLs
 * (YouTube, Vimeo, Loom, and 1800+ other platforms) and a dedicated
 * `fetch_ytdlp` tool is registered. Without yt-dlp, only local file paths
 * are accepted. The npm package does not bundle yt-dlp; the official Docker
 * image installs it for you.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import {
  handleFetchYtdlp,
  handleGetFrames,
  handleGetTranscript,
  handleGetVideoGrids,
  handleProbeMedia,
  handleUnderstandMedia,
} from "./mcp-handlers.js";
import { hasYtDlp } from "./youtube.js";

// ---------------------------------------------------------------------------
// Detect yt-dlp at startup (sync, cached)
// ---------------------------------------------------------------------------

const ytDlpAvailable = hasYtDlp();
if (ytDlpAvailable) {
  process.stderr.write("media-understanding: yt-dlp detected — URL support enabled\n");
} else {
  process.stderr.write("media-understanding: yt-dlp not found — file-only mode\n");
}

// ---------------------------------------------------------------------------
// URL blurb appended to descriptions when yt-dlp is available
// ---------------------------------------------------------------------------

const URL_PLATFORMS =
  "YouTube, Instagram (Reels & posts), Vimeo, Loom, Twitch, Dailymotion, TikTok, " +
  "X/Twitter, Facebook, LinkedIn, Reddit, SoundCloud, Dropbox, Google Drive, BBC, " +
  "CNN, and 1800+ other sites supported by yt-dlp";

const URL_BLURB = `

Also accepts URLs. When a URL is passed instead of a file path, the media is
automatically downloaded (lowest quality for speed) and cached in a temp
directory — first call may take 10-60s, subsequent calls are instant.

Supported platforms: ${URL_PLATFORMS}.`;

const URL_TRANSCRIPT_BLURB = `

Also accepts URLs. For URLs, subtitles are fetched first (instant, any language —
LLMs understand all languages). If no subtitles are available, falls back to
downloading audio and running ASR. This makes it the fastest way to get the
content of an online video.

Supported platforms: ${URL_PLATFORMS}.`;

// ---------------------------------------------------------------------------
// Shared input schemas (independent of yt-dlp availability)
// ---------------------------------------------------------------------------

const probeSchema = z.object({
  paths: z
    .union([z.string(), z.array(z.string()).min(1)])
    .describe(
      "One or more file paths, glob patterns, or URLs. Each string can be a literal path, a glob (e.g. `media/**/*.mp4`), or a URL. Pass a single string or an array.",
    ),
  max_files: z
    .number()
    .int()
    .positive()
    .optional()
    .describe(
      "Hard cap on matched files (default 50, absolute max 200). Probe is lightweight so generous limits are safe.",
    ),
});

const understandSchema = z.object({
  file_path: z
    .string()
    .describe("Absolute or relative path to a single media file (or URL if yt-dlp is available)."),
  model: z
    .string()
    .optional()
    .describe(
      'Whisper model name. Default: "base.en-q5_1" (English, ~57 MB, cached on first use).',
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
      "Sampling strategy. `uniform` (default) covers the window evenly — best for film/sports. `scene` captures frames at detected scene changes — best for screen recordings, slideshows, and lecture capture where uniform wastes budget on static high-fps content.",
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
      "Thumbnail width in pixels per cell (default 480; portrait video defaults to 120 when omitted). Use 320+ for screen recordings where UI text must be legible. Use 120 for a cheap overview where text is not needed.",
    ),
});

const videoGridsSchema = z.object({
  file_path: z.string().describe("Path to a video file (or URL if yt-dlp is available)."),
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
  start_sec: z.number().nonnegative().optional().describe("Start offset in seconds (default 0)."),
  end_sec: z
    .number()
    .positive()
    .optional()
    .describe("End offset in seconds (default: end of file)."),
  sampling_strategy: z
    .enum(["uniform", "scene"])
    .optional()
    .describe(
      "Sampling strategy. `uniform` (default) covers the window evenly — best for film/sports. `scene` captures frames at detected scene changes — best for screen recordings, slideshows, and lecture capture where uniform wastes budget on static high-fps content.",
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
      "Thumbnail width per cell in pixels (default 480; portrait video defaults to 120 when omitted). Use 320+ for screen recordings where UI text must be legible. Use 120 for a cheap overview where text is not needed.",
    ),
});

const framesSchema = z.object({
  file_path: z.string().describe("Path to a video file (or URL if yt-dlp is available)."),
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
});

const transcriptSchema = z.object({
  file_path: z.string().describe("Path to an audio or video file (or URL if yt-dlp is available)."),
  model: z
    .string()
    .optional()
    .describe(
      'Whisper model name. Default: "base.en-q5_1" (English, ~57 MB, cached on first use).',
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
});

// ---------------------------------------------------------------------------
// Tool descriptions — base (file-only) versions
// ---------------------------------------------------------------------------

const PROBE_DESC_BASE = `Discover and triage media files before committing to heavy analysis.

Returns metadata only: type, duration, resolution, codecs, file size. No decoding,
no transcription, no images. Reads file headers only (~5-50ms per file), so it is
safe to call on dozens of files at once.

Use this FIRST to scan a directory or batch of files, then pick individual files
for heavy analysis with understand_media, get_transcript, or get_video_grids.

Accepts a single \`paths\` parameter: a string or array of strings. Each string
can be a literal file path or a glob pattern.

Examples:
  { "paths": "/path/to/video.mp4" }
  { "paths": ["/path/to/a.mp4", "/path/to/b.mp3"] }
  { "paths": "media/**/*.{mp4,mp3,wav}" }
  { "paths": ["recordings/*.mp4", "specific/file.mp3"], "max_files": 100 }`;

const UNDERSTAND_DESC_BASE = `Fully analyze a media file (audio, video, or image).

Returns:
- Metadata (type, duration, resolution, codecs)
- Transcript text for audio/video (Whisper via node-av, default model auto-downloaded on first use)
- Keyframe grid images for video (JPEG base64, max 6 grids)
- Image data for image files

Use this as your first call for any media file. For long videos (>10 min) or
when you only need one modality, prefer the focused tools instead.

For multiple files, use probe_media first to discover and triage, then call this
tool once per file that needs full analysis.

**Content-type guidance (set params before calling):**
- Podcast / interview / lecture: transcript is the primary signal. Grids add little.
  Use get_transcript for audio-only. For video, omit grid params or use max_grids=1.
- Screen recording / UI walkthrough / tutorial: use sampling_strategy="scene" so
  grids capture actual transitions rather than redundant uniform frames. Use
  thumb_width=320 or higher so UI text is legible. Transcript is still primary.
- Film / B-roll / sports: uniform sampling (default) is correct. Default thumb_width
  (480) is fine. Grids are the primary signal.
- Portrait video (phone, TikTok): server auto-defaults to cols=3, rows=3,
  thumb_width=120 — no manual adjustment needed.

**Two-step workflow for long videos (>10 min):**
1. get_transcript(file, { format: "srt" }) — fast overview, identify key moments
2. get_frames(file, { timestamps: [...] }) — pull exact frames at those moments

Examples:
  { "file_path": "/path/to/video.mp4" }
  { "file_path": "/path/to/podcast.mp3" }
  { "file_path": "/path/to/clip.mp4", "start_sec": 60, "end_sec": 120, "max_grids": 3 }
  { "file_path": "/path/to/screen-recording.mp4", "sampling_strategy": "scene", "thumb_width": 320 }`;

const GRIDS_DESC_BASE = `Extract keyframe grid images from a video file.

Each grid is a JPEG contact sheet of thumbnails arranged in a cols×rows tile.
Every tile has an exact timestamp overlay, and the accompanying text lists the
exact timestamps in row-major order.

Use this for visual inspection without transcription. It is budget-aware: if
you omit max_grids, the server returns as many grids as fit under max_total_chars.

**Sampling strategy:**
- uniform (default): evenly-spaced frames across the window. Best for film,
  B-roll, sports, or any content where action is continuous.
- scene: only captures frames at detected scene changes. Best for screen
  recordings, slideshows, lecture capture, and UI walkthroughs — uniform wastes
  budget on static content at high fps. Use frame_interval as a fallback cap.

**thumb_width guidance:**
- 480 (default): good for film/video where you need visual detail.
- 320: readable for screen recordings with large UI elements.
- 120: overview-only; UI text will not be legible.
- Portrait video: server auto-defaults to 120 — no adjustment needed.

Examples:
  { "file_path": "/path/to/video.mp4" }
  { "file_path": "/path/to/movie.mkv", "start_sec": 300, "end_sec": 600, "max_grids": 2, "seconds_per_frame": 8 }
  { "file_path": "/path/to/screen-recording.mp4", "sampling_strategy": "scene", "thumb_width": 320 }
  { "file_path": "/path/to/lecture.mp4", "sampling_strategy": "scene", "frame_interval": 150 }`;

const FRAMES_DESC_BASE = `Extract individual video frames at specific timestamps.

Returns one JPEG image per requested timestamp. Useful for inspecting exact
moments identified from a transcript or grid (e.g. "show me the frame at 1:23").

Timestamps are in seconds (fractional values allowed).
Each returned frame image includes an exact timestamp overlay.

Examples:
  { "file_path": "/path/to/video.mp4", "timestamps": [0, 30, 60] }
  { "file_path": "/path/to/clip.mp4", "timestamps": [83.5] }`;

const TRANSCRIPT_DESC_BASE = `Transcribe an audio or video file and return the text.

Uses Whisper via node-av. Default model: "base.en-q5_1" (English, ~57 MB,
cached on first use). Transcript is cached per file for the process lifetime.

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
  { "file_path": "/path/to/meeting.mp4", "max_chars": 16000 }`;

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "media-understanding",
  version: "1.1.0",
});

// ---------------------------------------------------------------------------
// Register core tools (always available)
// Descriptions are extended with URL blurb when yt-dlp is detected
// ---------------------------------------------------------------------------

server.registerTool(
  "probe_media",
  {
    description: PROBE_DESC_BASE + (ytDlpAvailable ? URL_BLURB : ""),
    inputSchema: probeSchema,
  },
  handleProbeMedia,
);

server.registerTool(
  "understand_media",
  {
    description: UNDERSTAND_DESC_BASE + (ytDlpAvailable ? URL_BLURB : ""),
    inputSchema: understandSchema,
  },
  handleUnderstandMedia,
);

server.registerTool(
  "get_video_grids",
  {
    description: GRIDS_DESC_BASE + (ytDlpAvailable ? URL_BLURB : ""),
    inputSchema: videoGridsSchema,
  },
  handleGetVideoGrids,
);

server.registerTool(
  "get_frames",
  {
    description: FRAMES_DESC_BASE + (ytDlpAvailable ? URL_BLURB : ""),
    inputSchema: framesSchema,
  },
  handleGetFrames,
);

server.registerTool(
  "get_transcript",
  {
    description: TRANSCRIPT_DESC_BASE + (ytDlpAvailable ? URL_TRANSCRIPT_BLURB : ""),
    inputSchema: transcriptSchema,
  },
  handleGetTranscript,
);

// ---------------------------------------------------------------------------
// URL-only tools (registered only when yt-dlp is available)
// ---------------------------------------------------------------------------

if (ytDlpAvailable) {
  server.registerTool(
    "fetch_ytdlp",
    {
      description: `Fetch content from any yt-dlp-supported platform (not just YouTube).

Supported platforms: ${URL_PLATFORMS}.

Returns file paths to downloaded content in a temp directory, plus video metadata.
Subtitles are fetched and inlined by default (fastest way to understand a video).
Downloaded files are cached — repeated calls for the same URL are instant.

Use this tool when you want fine-grained control over what to download. For simpler
workflows, you can also pass URLs directly to probe_media, understand_media,
get_transcript, get_video_grids, or get_frames — they accept URLs transparently.

**Default behavior (subtitles only):** ~1-5 seconds. Fetches available subtitles
(manual or auto-generated, any language) and returns the transcript inline.
If no subtitles exist, suggests using get_transcript for ASR fallback.

**Optional downloads:**
- include_thumbnail: download video thumbnail image
- include_video: download lowest quality video for frame analysis with
  get_video_grids or get_frames (~10-60s depending on duration)
- include_audio: download audio track for ASR transcription with get_transcript

All files persist in temp directory across calls, so you can fetch subtitles first,
then request video later if frame analysis is needed — without re-downloading.

Examples:
  { "url": "https://youtube.com/watch?v=dQw4w9WgXcQ" }
  { "url": "https://www.loom.com/share/abc123" }
  { "url": "https://vimeo.com/123456", "include_video": true }
  { "url": "https://twitter.com/user/status/123", "include_audio": true }`,
      inputSchema: z.object({
        url: z.string().describe(`URL of the video to fetch. Works with ${URL_PLATFORMS}.`),
        include_subtitles: z
          .boolean()
          .optional()
          .describe(
            "Download and inline subtitles/captions (default true). Grabs first available language — LLMs understand any language.",
          ),
        include_video: z
          .boolean()
          .optional()
          .describe(
            "Download lowest quality video for frame analysis with get_video_grids / get_frames (default false). Slower but enables visual analysis.",
          ),
        include_audio: z
          .boolean()
          .optional()
          .describe(
            "Download audio for ASR transcription via get_transcript (default false). Use when no subtitles are available.",
          ),
        include_thumbnail: z
          .boolean()
          .optional()
          .describe("Download video thumbnail image (default false)."),
      }),
    },
    handleFetchYtdlp,
  );
}

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

const transport = new StdioServerTransport();
await server.connect(transport);
