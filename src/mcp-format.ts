/**
 * Formatting helpers for MCP handler responses.
 *
 * Path expansion, grid/frame labels, media info display,
 * transcript formatting (text/SRT/JSON), and time windowing.
 */

import * as fsPromises from "node:fs/promises";
import { resolve } from "node:path";

import type { transcribeAudio } from "./media.js";
import { truncateTranscript } from "./media.js";
import type { MediaInfo, Segment, VideoFrameImage, VideoGridImage } from "./types.js";
import { MediaError } from "./types.js";

// ---------------------------------------------------------------------------
// Grid / frame label helpers
// ---------------------------------------------------------------------------

export function formatGridWindow(grid: VideoGridImage, index: number, total: number): string {
  const tileSummary = grid.tiles.map((tile) => tile.timestampLabel).join(", ");
  return `Grid ${index}/${total} covers ${grid.startSec.toFixed(3)}s-${grid.endSec.toFixed(3)}s. Tile timestamps: ${tileSummary}`;
}

export function formatFrameLabel(frame: VideoFrameImage): string {
  return `Frame at ${frame.timestampLabel} (${frame.timestampSec.toFixed(3)}s)`;
}

// ---------------------------------------------------------------------------
// Path expansion
// ---------------------------------------------------------------------------

function buildInvalidBatchError(message: string): MediaError {
  return new MediaError("INVALID_SAMPLING", message);
}

/** Returns true if the string contains glob metacharacters. */
function isGlobPattern(s: string): boolean {
  return /[*?[\]{]/.test(s);
}

/**
 * Expand a list of path strings into resolved file paths.
 * Each string can be a literal file path or a glob pattern — `fs.promises.glob()` handles both.
 * Literal paths that don't exist are preserved so probeMedia() reports the error per-file.
 */
export async function expandPaths(paths: string | string[]): Promise<string[]> {
  const entries = Array.isArray(paths) ? paths : [paths];
  if (entries.length === 0 || entries.every((e) => e.trim().length === 0)) {
    throw buildInvalidBatchError(
      "`paths` must contain at least one non-empty path or glob pattern.",
    );
  }

  const seen = new Set<string>();
  const result: string[] = [];

  for (const entry of entries) {
    const trimmed = entry.trim();
    if (trimmed.length === 0) continue;

    let matched = false;
    for await (const match of fsPromises.glob(trimmed)) {
      const resolved = resolve(match);
      if (!seen.has(resolved)) {
        seen.add(resolved);
        result.push(resolved);
        matched = true;
      }
    }

    // If a literal path (not a glob) matched nothing, keep it so probeMedia()
    // produces a per-file error rather than silently dropping it.
    if (!matched && !isGlobPattern(trimmed)) {
      const resolved = resolve(trimmed);
      if (!seen.has(resolved)) {
        seen.add(resolved);
        result.push(resolved);
      }
    }
  }

  if (result.length === 0) {
    throw buildInvalidBatchError(
      "The provided path(s) did not match any files. Check your paths or glob patterns.",
    );
  }

  return result;
}

// ---------------------------------------------------------------------------
// Transcript formatting
// ---------------------------------------------------------------------------

export function formatTranscriptSegments(
  segments: Awaited<ReturnType<typeof transcribeAudio>>,
  maxChars: number,
): string {
  const formatted = segments
    .map((s) => `[${(s.start / 1000).toFixed(1)}–${(s.end / 1000).toFixed(1)}] ${s.text.trim()}`)
    .join("\n");

  return truncateTranscript(formatted, maxChars);
}

/**
 * Format milliseconds as SRT timestamp: HH:MM:SS,mmm
 */
export function formatSrtTimestamp(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const hours = Math.floor(totalSec / 3600);
  const minutes = Math.floor((totalSec % 3600) / 60);
  const seconds = totalSec % 60;
  const millis = Math.round(ms % 1000);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")},${String(millis).padStart(3, "0")}`;
}

/**
 * Format segments as SRT subtitle text.
 * SRT indices are 1-based.
 */
export function formatTranscriptAsSRT(segments: Segment[], maxChars: number): string {
  const formatted = segments
    .map((s, i) => {
      const idx = i + 1;
      return `${idx}\n${formatSrtTimestamp(s.start)} --> ${formatSrtTimestamp(s.end)}\n${s.text.trim()}`;
    })
    .join("\n\n");

  return truncateTranscript(formatted, maxChars);
}

/**
 * Format segments as JSON with millisecond timestamps.
 */
export function formatTranscriptAsJSON(segments: Segment[], maxChars: number): string {
  const data = {
    segments: segments.map((s) => ({
      start: Math.round(s.start),
      end: Math.round(s.end),
      text: s.text.trim(),
    })),
  };
  const formatted = JSON.stringify(data, null, 2);
  return truncateTranscript(formatted, maxChars);
}

/**
 * Filter segments to those overlapping a time window [startSec, endSec).
 * A segment overlaps if segment.start < endMs AND segment.end > startMs.
 * Segments partially outside the window are included (not clipped).
 */
export function filterSegmentsByWindow(
  segments: Segment[],
  startSec?: number,
  endSec?: number,
): Segment[] {
  if (startSec === undefined && endSec === undefined) return segments;

  const startMs = (startSec ?? 0) * 1000;
  const endMs = endSec !== undefined ? endSec * 1000 : Infinity;

  return segments.filter((s) => s.start < endMs && s.end > startMs);
}

// ---------------------------------------------------------------------------
// Options / media info helpers
// ---------------------------------------------------------------------------

type ExactPartial<T> = { [K in keyof T]?: Exclude<T[K], undefined> };

export function buildOpts<T extends Record<string, unknown>>(raw: T): ExactPartial<T> {
  const out: ExactPartial<T> = {};
  for (const key of Object.keys(raw) as (keyof T)[]) {
    if (raw[key] !== undefined) {
      (out as Record<keyof T, unknown>)[key] = raw[key];
    }
  }
  return out;
}

export function formatMediaInfo(info: MediaInfo): string {
  const lines: string[] = [
    `File: ${info.path}`,
    `Type: ${info.type}`,
    `Duration: ${info.duration.toFixed(1)}s`,
  ];
  if (info.width !== undefined) lines.push(`Resolution: ${info.width}x${info.height}`);
  if (info.fps !== undefined) lines.push(`FPS: ${info.fps.toFixed(2)}`);
  if (info.videoCodec) lines.push(`Video codec: ${info.videoCodec}`);
  if (info.audioCodec) lines.push(`Audio codec: ${info.audioCodec}`);
  if (info.sampleRate !== undefined) lines.push(`Sample rate: ${info.sampleRate}Hz`);
  if (info.channels !== undefined) lines.push(`Channels: ${info.channels}`);
  if (info.fileSizeBytes !== undefined) lines.push(`Size: ${info.fileSizeBytes} bytes`);
  if (info.acceleration !== undefined) {
    lines.push(
      `Acceleration: ${info.acceleration.backend}${info.acceleration.hardware ? " (hardware)" : " (software)"}`,
    );
  }
  return lines.join("\n");
}
