#!/usr/bin/env node
/**
 * CLI stub — lightweight probe-only output for OpenClaw media model integration.
 *
 * Runs probeMedia() (~5-50ms, header read only) and outputs metadata plus
 * guidance to use the MCP tools for full analysis. No transcription, no grid
 * extraction, no heavy work.
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

import { probeMedia } from "./media.js";
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

Runs a quick metadata probe (~50ms) and outputs file info plus MCP tool guidance.
For full analysis (transcription, keyframe grids, etc.), use the MCP server tools.

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

  process.stdout.write(
    `${parts.join(" ")}

This file has been detected by the media-understanding CLI.
Use your media-understanding MCP tools for full analysis:
  - probe_media: batch metadata scanning
  - understand_media: full analysis (transcript + keyframe grids)
  - get_transcript: speech content with format options (text/srt/json)
  - get_video_grids: visual keyframe inspection
  - get_frames: exact frame extraction at specific timestamps
`,
  );
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
