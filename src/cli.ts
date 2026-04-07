#!/usr/bin/env node
/**
 * CLI stub — type-aware media output for OpenClaw media model integration.
 *
 * Audio files:  runs transcription inline and outputs the full transcript.
 * Video files:  outputs metadata + directive MCP tool guidance.
 * Image files:  outputs metadata + "ready for direct analysis" message.
 *
 * Usage:
 *   media-understanding <file>
 *   media-understanding --help
 *   media-understanding --version
 *
 * Exit codes: 0 = success, 1 = error
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { probeMedia, transcribeAudio, truncateTranscript } from "./media.js";
import { MediaError } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const { version } = JSON.parse(readFileSync(join(__dirname, "../package.json"), "utf8")) as {
  version: string;
};

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    help: { type: "boolean", short: "h" },
    version: { type: "boolean", short: "V" },
  },
  allowPositionals: true,
});

if (values.version) {
  process.stdout.write(`${version}\n`);
  process.exit(0);
}

if (values.help || positionals.length === 0) {
  process.stdout.write(
    `Usage: media-understanding <file>

Probes a media file and outputs type-aware results:
  Audio: runs transcription and prints the full transcript.
  Video: outputs metadata and MCP tool guidance.
  Image: outputs metadata; ready for direct vision model analysis.

Options:
  -h, --help     Show this help
  -V, --version  Print version number
`,
  );
  process.exit(0);
}

const filePath = positionals[0];
if (!filePath) {
  process.stderr.write("Error: file path is required\n");
  process.exit(1);
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h${m > 0 ? ` ${m}m` : ""}`;
  if (m > 0) return `${m}m${s > 0 ? `${s}s` : ""}`;
  return `${s}s`;
}

try {
  const info = await probeMedia(filePath);

  const typeBadge = `[${info.type.charAt(0).toUpperCase() + info.type.slice(1)}]`;
  const parts: string[] = [typeBadge];

  if (info.type !== "image") parts.push(formatDuration(info.duration));
  if (info.width !== undefined) parts.push(`${info.width}x${info.height}`);
  if (info.videoCodec && info.audioCodec) parts.push(`${info.videoCodec}/${info.audioCodec}`);
  else if (info.audioCodec) parts.push(info.audioCodec);
  if (info.fileSizeBytes !== undefined) parts.push(formatSize(info.fileSizeBytes));

  const header = `File: ${info.path}\n${parts.join(" ")}\n`;

  if (info.type === "audio") {
    // ── Audio: run transcription inline ──
    process.stdout.write(header + "\n");

    const maxChars = parseInt(process.env["MEDIA_UNDERSTANDING_MAX_CHARS"] ?? "32000", 10);
    const segments = await transcribeAudio(info.path);
    const fullText = segments.map((s) => s.text.trim()).join(" ");
    const transcript = truncateTranscript(fullText, maxChars);

    process.stdout.write(`--- Transcript ---\n${transcript}\n--- End Transcript ---\n`);

    if (segments.length > 0) {
      process.stdout.write(
        `\nFor more options (SRT timestamps, JSON segments, time windowing), use MCP tools:
  - get_transcript(file_path: "${info.path}", format: "srt")
  - understand_media(file_path: "${info.path}")
`,
      );
    }
  } else if (info.type === "video") {
    // ── Video: directive MCP guidance ──
    process.stdout.write(
      `${header}
IMPORTANT: This video must be analyzed for both visual and audio content.
Run understand_media with the file path above for a complete analysis.

Available MCP tools for this file:
  - understand_media(file_path: "${info.path}") — transcript + keyframe grids
  - get_transcript(file_path: "${info.path}") — speech transcription only
  - get_video_grids(file_path: "${info.path}") — visual keyframe grids
  - get_frames(file_path: "${info.path}", timestamps: [...]) — exact frames
  - probe_media(paths: "${info.path}") — lightweight metadata only
`,
    );
  } else if (info.type === "image") {
    // ── Image: ready for direct analysis ──
    process.stdout.write(
      `${header}
This image is ready for direct analysis — no MCP tools needed.
Pass it to your vision model directly, or use probe_media for metadata.
`,
    );
  } else {
    // ── Unknown type: generic guidance ──
    process.stdout.write(
      `${header}
Use your media-understanding MCP tools for analysis:
  - probe_media(paths: "${info.path}")
  - understand_media(file_path: "${info.path}")
`,
    );
  }
} catch (err) {
  const message =
    err instanceof MediaError
      ? `[${err.code}] ${err.message}`
      : err instanceof Error
        ? err.message
        : String(err);
  process.stderr.write(`Error: ${message}\n`);
  process.exit(1);
}
