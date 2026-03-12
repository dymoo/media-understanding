#!/usr/bin/env node
/**
 * MCP server (stdio transport) exposing 4 tools for LLM media understanding.
 *
 * Tools:
 *  - understand_media  → full analysis: info + transcript + grids in one call
 *  - get_video_grids   → keyframe grids only (for re-inspection without re-transcribing)
 *  - get_frames        → extract individual frames at specific timestamps
 *  - get_transcript    → transcript only (text, no images)
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import { readFile } from "node:fs/promises";

import {
  compressForLLM,
  extractFrame,
  extractFrameGrid,
  transcribeAudio,
  understandMedia,
} from "./media.js";
import { MediaError } from "./types.js";

// ---------------------------------------------------------------------------
// MCP error helper
// ---------------------------------------------------------------------------

type McpErrorResult = {
  isError: true;
  content: [{ type: "text"; text: string }];
};

function mcpError(err: unknown): McpErrorResult {
  let message: string;
  if (err instanceof MediaError) {
    message = `[${err.code}] ${err.message}`;
  } else if (err instanceof Error) {
    message = err.message;
  } else {
    message = String(err);
  }
  return { isError: true, content: [{ type: "text", text: message }] };
}

/**
 * Strip keys with undefined values and remove `undefined` from value types.
 * This satisfies `exactOptionalPropertyTypes` at call sites where Zod args
 * produce `T | undefined` for optional fields.
 */
type ExactPartial<T> = { [K in keyof T]?: Exclude<T[K], undefined> };

function buildOpts<T extends Record<string, unknown>>(raw: T): ExactPartial<T> {
  const out: ExactPartial<T> = {};
  for (const key of Object.keys(raw) as (keyof T)[]) {
    if (raw[key] !== undefined) {
      (out as Record<keyof T, unknown>)[key] = raw[key];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "media-understanding",
  version: "0.1.0",
});

// ---------------------------------------------------------------------------
// Tool: understand_media
// ---------------------------------------------------------------------------

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

Examples:
  { "file_path": "/path/to/video.mp4" }
  { "file_path": "/path/to/podcast.mp3", "model": "base.en" }
  { "file_path": "/path/to/clip.mp4", "start_sec": 60, "end_sec": 120, "max_grids": 3 }`,
    inputSchema: z.object({
      file_path: z.string().describe("Absolute or relative path to the media file."),
      model: z
        .string()
        .optional()
        .describe(
          'Whisper model name. Default: "tiny.en". Options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3.',
        ),
      max_chars: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Max transcript characters (default 32000). Truncates middle when exceeded."),
      max_grids: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Max grid images to return for video (default 6)."),
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
      cols: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .describe("Grid columns per tile (default 4)."),
      rows: z.number().int().min(1).max(8).optional().describe("Grid rows per tile (default 4)."),
      thumb_width: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Thumbnail width in pixels per cell (default 480)."),
    }),
  },
  async (args) => {
    try {
      const result = await understandMedia(
        args.file_path,
        buildOpts({
          model: args.model,
          maxChars: args.max_chars,
          maxGrids: args.max_grids,
          startSec: args.start_sec,
          endSec: args.end_sec,
          cols: args.cols,
          rows: args.rows,
          thumbWidth: args.thumb_width,
        }),
      );

      const content: Array<
        { type: "text"; text: string } | { type: "image"; data: string; mimeType: string }
      > = [];

      // Media info summary
      const { info } = result;
      const lines: string[] = [
        `File: ${info.path}`,
        `Type: ${info.type}`,
        `Duration: ${info.duration.toFixed(1)}s`,
      ];
      if (info.width) lines.push(`Resolution: ${info.width}x${info.height}`);
      if (info.fps) lines.push(`FPS: ${info.fps.toFixed(2)}`);
      if (info.videoCodec) lines.push(`Video codec: ${info.videoCodec}`);
      if (info.audioCodec) lines.push(`Audio codec: ${info.audioCodec}`);
      if (info.sampleRate) lines.push(`Sample rate: ${info.sampleRate}Hz`);
      if (info.channels) lines.push(`Channels: ${info.channels}`);

      content.push({ type: "text", text: lines.join("\n") });

      // Transcript
      if (result.transcript) {
        content.push({
          type: "text",
          text: `\n--- TRANSCRIPT ---\n${result.transcript}`,
        });
      }

      // Image (for image files)
      if (info.type === "image") {
        const raw = await readFile(info.path);
        const compressed = await compressForLLM(raw);
        content.push({
          type: "image",
          data: compressed.toString("base64"),
          mimeType: "image/jpeg",
        });
      }

      // Frame grids
      for (let i = 0; i < result.grids.length; i++) {
        const grid = result.grids[i];
        if (grid) {
          content.push({
            type: "text",
            text: `\n--- FRAME GRID ${i + 1}/${result.grids.length} ---`,
          });
          content.push({
            type: "image",
            data: grid.toString("base64"),
            mimeType: "image/jpeg",
          });
        }
      }

      return { content };
    } catch (err) {
      return mcpError(err);
    }
  },
);

// ---------------------------------------------------------------------------
// Tool: get_video_grids
// ---------------------------------------------------------------------------

server.registerTool(
  "get_video_grids",
  {
    description: `Extract keyframe grid images from a video file.

Each grid is a JPEG contact sheet of thumbnails arranged in a cols×rows tile.
Use for visual inspection without triggering transcription.
Ideal for re-scanning a specific time range or adjusting grid density.

Examples:
  { "file_path": "/path/to/video.mp4" }
  { "file_path": "/path/to/movie.mkv", "start_sec": 300, "end_sec": 600, "max_grids": 4 }
  { "file_path": "/path/to/lecture.mp4", "scene_threshold": 0.5, "frame_interval": 150 }`,
    inputSchema: z.object({
      file_path: z.string().describe("Path to a video file."),
      max_grids: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Max grid images to return (default 6)."),
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
      scene_threshold: z
        .number()
        .min(0)
        .max(1)
        .optional()
        .describe("Scene-change threshold 0–1. Higher = fewer keyframes. Default 0.3."),
      frame_interval: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "Fallback: include a frame every N frames even without scene change (default 300 ≈ 10s at 30fps).",
        ),
      cols: z.number().int().min(1).max(8).optional().describe("Grid columns (default 4)."),
      rows: z.number().int().min(1).max(8).optional().describe("Grid rows (default 4)."),
      thumb_width: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Thumbnail width per cell in pixels (default 480)."),
    }),
  },
  async (args) => {
    try {
      const grids = await extractFrameGrid(
        args.file_path,
        buildOpts({
          maxGrids: args.max_grids,
          startSec: args.start_sec,
          endSec: args.end_sec,
          sceneThreshold: args.scene_threshold,
          frameInterval: args.frame_interval,
          cols: args.cols,
          rows: args.rows,
          thumbWidth: args.thumb_width,
        }),
      );

      if (grids.length === 0) {
        return {
          content: [
            {
              type: "text" as const,
              text: "No keyframes found. Try lowering scene_threshold or frame_interval.",
            },
          ],
        };
      }

      const content: Array<
        { type: "text"; text: string } | { type: "image"; data: string; mimeType: string }
      > = [{ type: "text", text: `Extracted ${grids.length} grid(s).` }];

      for (let i = 0; i < grids.length; i++) {
        const grid = grids[i];
        if (grid) {
          content.push({
            type: "text",
            text: `Grid ${i + 1}/${grids.length}`,
          });
          content.push({
            type: "image",
            data: grid.toString("base64"),
            mimeType: "image/jpeg",
          });
        }
      }

      return { content };
    } catch (err) {
      return mcpError(err);
    }
  },
);

// ---------------------------------------------------------------------------
// Tool: get_frames
// ---------------------------------------------------------------------------

server.registerTool(
  "get_frames",
  {
    description: `Extract individual video frames at specific timestamps.

Returns one JPEG image per requested timestamp. Useful for inspecting exact
moments identified from a transcript or grid (e.g. "show me the frame at 1:23").

Timestamps are in seconds (fractional values allowed).

Examples:
  { "file_path": "/path/to/video.mp4", "timestamps": [0, 30, 60] }
  { "file_path": "/path/to/clip.mp4", "timestamps": [83.5] }`,
    inputSchema: z.object({
      file_path: z.string().describe("Path to a video file."),
      timestamps: z
        .array(z.number().nonnegative())
        .min(1)
        .max(20)
        .describe("Timestamps in seconds at which to extract frames. Max 20 per call."),
    }),
  },
  async (args) => {
    try {
      const content: Array<
        { type: "text"; text: string } | { type: "image"; data: string; mimeType: string }
      > = [];

      for (const ts of args.timestamps) {
        content.push({ type: "text", text: `Frame at ${ts.toFixed(2)}s` });
        const buf = await extractFrame(args.file_path, ts);
        content.push({
          type: "image",
          data: buf.toString("base64"),
          mimeType: "image/jpeg",
        });
      }

      return { content };
    } catch (err) {
      return mcpError(err);
    }
  },
);

// ---------------------------------------------------------------------------
// Tool: get_transcript
// ---------------------------------------------------------------------------

server.registerTool(
  "get_transcript",
  {
    description: `Transcribe an audio or video file and return the text.

Uses OpenAI's Whisper (via whisper.cpp). The model auto-downloads on first use
(~75 MB for tiny.en). Transcript is cached per file for the process lifetime.

Returns plain text with segment timestamps on each line: "[start–end] text"

Examples:
  { "file_path": "/path/to/podcast.mp3" }
  { "file_path": "/path/to/meeting.mp4", "model": "base.en", "max_chars": 16000 }`,
    inputSchema: z.object({
      file_path: z.string().describe("Path to an audio or video file."),
      model: z
        .string()
        .optional()
        .describe('Whisper model. Default: "tiny.en". Larger models are slower but more accurate.'),
      max_chars: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Max characters to return (default 32000). Keeps first 60% + last 40%."),
    }),
  },
  async (args) => {
    try {
      const maxChars =
        args.max_chars ?? parseInt(process.env["MEDIA_UNDERSTANDING_MAX_CHARS"] ?? "32000", 10);

      const segments = await transcribeAudio(args.file_path, buildOpts({ model: args.model }));

      if (segments.length === 0) {
        return {
          content: [{ type: "text" as const, text: "No speech detected." }],
        };
      }

      // Format: "[0.0–3.2] Hello world."
      const formatted = segments
        .map(
          (s) => `[${(s.start / 1000).toFixed(1)}–${(s.end / 1000).toFixed(1)}] ${s.text.trim()}`,
        )
        .join("\n");

      // Apply truncation
      const raw = segments
        .map((s) => s.text)
        .join(" ")
        .trim();
      const truncated =
        raw.length > maxChars
          ? formatted.slice(0, Math.floor(maxChars * 0.6)) +
            "\n\n[…transcript truncated…]\n\n" +
            formatted.slice(-Math.floor(maxChars * 0.4))
          : formatted;

      return {
        content: [{ type: "text" as const, text: truncated }],
      };
    } catch (err) {
      return mcpError(err);
    }
  },
);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

const transport = new StdioServerTransport();
await server.connect(transport);
