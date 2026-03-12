import * as fsPromises from "node:fs/promises";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import {
  compressForLLM,
  extractFrameGridImages,
  extractFrameImage,
  probeMedia,
  transcribeAudio,
  truncateTranscript,
  understandMedia,
} from "./media.js";
import type { MediaInfo, Segment, VideoFrameImage, VideoGridImage } from "./types.js";
import { MediaError } from "./types.js";

export type McpErrorResult = {
  isError: true;
  content: [{ type: "text"; text: string }];
};

export type McpContentItem =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string };

export type McpSuccessResult = {
  content: McpContentItem[];
};

const DEFAULT_PROBE_MAX_FILES = 50;
const ABSOLUTE_MAX_PROBE_FILES = 200;

const DEFAULT_TOTAL_CHARS = 48_000;

// ---------------------------------------------------------------------------
// Preflight safety thresholds for heavy operations
// ---------------------------------------------------------------------------

/** Max file duration (seconds) for understand_media (transcribes full file). */
export const PREFLIGHT_MAX_DURATION_FULL = 7200; // 2 hours

/** Max file duration (seconds) for get_transcript. */
export const PREFLIGHT_MAX_DURATION_TRANSCRIPT = 14_400; // 4 hours

/** Absolute max file size (bytes) for any heavy operation. */
export const PREFLIGHT_MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024; // 10 GB

export type UnderstandMediaArgs = {
  file_path: string;
  model?: string | undefined;
  max_chars?: number | undefined;
  max_total_chars?: number | undefined;
  max_grids?: number | undefined;
  start_sec?: number | undefined;
  end_sec?: number | undefined;
  sampling_strategy?: "uniform" | "scene" | undefined;
  seconds_per_frame?: number | undefined;
  seconds_per_grid?: number | undefined;
  aspect_mode?: "contain" | "cover" | undefined;
  cols?: number | undefined;
  rows?: number | undefined;
  thumb_width?: number | undefined;
};

export type GetVideoGridsArgs = {
  file_path: string;
  max_total_chars?: number | undefined;
  max_grids?: number | undefined;
  start_sec?: number | undefined;
  end_sec?: number | undefined;
  sampling_strategy?: "uniform" | "scene" | undefined;
  scene_threshold?: number | undefined;
  frame_interval?: number | undefined;
  seconds_per_frame?: number | undefined;
  seconds_per_grid?: number | undefined;
  cols?: number | undefined;
  rows?: number | undefined;
  aspect_mode?: "contain" | "cover" | undefined;
  thumb_width?: number | undefined;
};

export type GetFramesArgs = {
  file_path: string;
  max_total_chars?: number | undefined;
  timestamps: number[];
};

export type GetTranscriptArgs = {
  file_path: string;
  model?: string | undefined;
  max_chars?: number | undefined;
  format?: "text" | "srt" | "json" | undefined;
  start_sec?: number | undefined;
  end_sec?: number | undefined;
};

export type ProbeMediaArgs = {
  paths: string | string[];
  max_files?: number | undefined;
};

export function mcpError(err: unknown): McpErrorResult {
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

// ---------------------------------------------------------------------------
// Preflight checks — fail fast before expensive work
// ---------------------------------------------------------------------------

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0 && m > 0) return `${h}h ${m}m`;
  if (h > 0) return `${h}h`;
  return `${m}m`;
}

export function preflightFileSize(info: MediaInfo, toolName: string): void {
  if (info.fileSizeBytes !== undefined && info.fileSizeBytes > PREFLIGHT_MAX_FILE_SIZE) {
    const sizeGB = (info.fileSizeBytes / (1024 * 1024 * 1024)).toFixed(1);
    throw new MediaError(
      "FILE_TOO_LARGE",
      `This file is ${sizeGB} GB — too large for ${toolName}. ` +
        `Maximum supported file size is 10 GB. ` +
        `Try a smaller file or re-encode at a lower bitrate.`,
    );
  }
}

export function preflightDuration(info: MediaInfo, maxDuration: number, toolName: string): void {
  if (info.type !== "audio" && info.type !== "video") return;
  if (info.duration <= maxDuration) return;

  const durationStr = formatDuration(info.duration);
  const maxStr = formatDuration(maxDuration);

  let guidance: string;
  if (toolName === "understand_media") {
    guidance =
      "Note: understand_media transcribes the entire audio track regardless of " +
      "start_sec/end_sec (those only affect visual grid extraction).\n\n" +
      "For long media, use the focused tools instead:\n" +
      "1. get_video_grids with start_sec/end_sec — sample visual content from specific sections (fast)\n" +
      "2. get_transcript — transcribe speech content (supports files up to " +
      formatDuration(PREFLIGHT_MAX_DURATION_TRANSCRIPT) +
      ")\n" +
      "3. get_frames — extract exact moments identified from transcripts or grids";
  } else {
    guidance =
      "For very long audio/video:\n" +
      "- Use probe_media first to check duration and plan your approach\n" +
      "- Use get_video_grids with start_sec/end_sec for visual sampling of specific sections\n" +
      "- Consider whether a summary of an initial portion is sufficient";
  }

  throw new MediaError(
    "FILE_TOO_LARGE",
    `This file is ${durationStr} long. ${toolName} is not recommended for files over ${maxStr}.\n\n${guidance}`,
  );
}

function getTotalCharBudget(maxTotalChars?: number): number {
  return maxTotalChars ?? DEFAULT_TOTAL_CHARS;
}

function serializedContentLength(content: McpContentItem[]): number {
  return JSON.stringify({ content }).length;
}

function summarizePayload(content: McpContentItem[]): string {
  const imageChars = content
    .filter((item): item is Extract<McpContentItem, { type: "image" }> => item.type === "image")
    .reduce((sum, item) => sum + item.data.length, 0);
  return `Payload: ${serializedContentLength(content)} chars total, ${imageChars} base64 image chars.`;
}

function buildBudgetExceededError(message: string): MediaError {
  return new MediaError("BUDGET_EXCEEDED", message);
}

// ---------------------------------------------------------------------------
// LLM Vision Token Estimation
// ---------------------------------------------------------------------------

/**
 * Estimate LLM vision tokens for an image at the given pixel dimensions.
 * Uses the Claude formula (pixels / 750) as a cross-provider ballpark.
 * Not exact for any provider — just informative for budget error messages.
 */
export function estimateVisionTokens(widthPx: number, heightPx: number): number {
  return Math.ceil((widthPx * heightPx) / 750);
}

/** Format an overage ratio like "~3.0x over budget". */
function formatOverageRatio(actual: number, budget: number): string {
  const ratio = actual / budget;
  return `~${ratio.toFixed(1)}x over budget`;
}

/** Detect portrait aspect ratio (height > width). */
function isPortrait(info: MediaInfo): boolean {
  return info.width !== undefined && info.height !== undefined && info.height > info.width;
}

/** Build resolution-aware suggestion lines for budget errors involving grid images. */
function buildGridBudgetHints(info: MediaInfo, thumbWidth?: number): string[] {
  const hints: string[] = [];
  if (isPortrait(info)) {
    hints.push(
      `This video is portrait (${info.width}x${info.height}) — each grid tile is tall, producing large images.`,
    );
  }
  hints.push("Try:");
  if (thumbWidth !== undefined && thumbWidth > 120) {
    hints.push(`- thumb_width=120 (smaller tiles, proportionally smaller payload)`);
  } else if (thumbWidth === undefined) {
    hints.push(`- thumb_width=120 (smaller tiles, proportionally smaller payload)`);
  }
  hints.push("- Fewer tiles: cols=3, rows=3 instead of 4x4");
  hints.push("- Shorter window: start_sec/end_sec to cover less time per grid");
  hints.push("- Increase max_total_chars if your client supports it");
  return hints;
}

/** Build suggestion lines for budget errors involving individual frame extraction. */
function buildFrameBudgetHints(info: MediaInfo): string[] {
  const hints: string[] = [];
  if (isPortrait(info)) {
    hints.push(
      `This video is portrait (${info.width}x${info.height}) — each frame image is large.`,
    );
  }
  hints.push("Try:");
  hints.push("- Fewer timestamps per call (split across multiple get_frames requests)");
  hints.push("- Increase max_total_chars if your client supports it");
  return hints;
}

/**
 * Build a rich budget error with actual vs budget sizes, overage ratio,
 * and optional vision token estimates.
 */
function richBudgetError(opts: {
  context: string;
  actualChars: number;
  budgetChars: number;
  imageWidthPx?: number;
  imageHeightPx?: number;
  mediaInfo?: MediaInfo;
  thumbWidth?: number;
  /** Controls which recovery hints to show. "grid" (default) suggests cols/rows/thumb_width; "frame" suggests fewer timestamps. */
  hintMode?: "grid" | "frame";
}): MediaError {
  const { context, actualChars, budgetChars } = opts;
  const lines: string[] = [];

  lines.push(
    `${context} (~${actualChars.toLocaleString()} chars, ${formatOverageRatio(actualChars, budgetChars)} at max_total_chars=${budgetChars.toLocaleString()}).`,
  );

  if (opts.imageWidthPx !== undefined && opts.imageHeightPx !== undefined) {
    const tokens = estimateVisionTokens(opts.imageWidthPx, opts.imageHeightPx);
    lines.push(
      `Estimated LLM cost: ~${tokens.toLocaleString()} vision tokens (${opts.imageWidthPx}x${opts.imageHeightPx} px).`,
    );
  }

  if (opts.mediaInfo) {
    lines.push("");
    const mode = opts.hintMode ?? "grid";
    if (mode === "frame") {
      lines.push(...buildFrameBudgetHints(opts.mediaInfo));
    } else {
      lines.push(...buildGridBudgetHints(opts.mediaInfo, opts.thumbWidth));
    }
  }

  return buildBudgetExceededError(lines.join("\n"));
}

function assertFitsBudget(content: McpContentItem[], maxTotalChars: number, detail: string): void {
  const totalChars = serializedContentLength(content);
  if (totalChars > maxTotalChars) {
    throw buildBudgetExceededError(
      `${detail} (~${totalChars.toLocaleString()} chars, ${formatOverageRatio(totalChars, maxTotalChars)} at max_total_chars=${maxTotalChars.toLocaleString()}). Try increasing seconds_per_frame, increasing seconds_per_grid, shortening the time window, requesting fewer images, or lowering max_chars.`,
    );
  }
}

function appendIfFits(
  content: McpContentItem[],
  item: McpContentItem,
  maxTotalChars: number,
): boolean {
  const next = [...content, item];
  if (serializedContentLength(next) > maxTotalChars) {
    return false;
  }
  content.push(item);
  return true;
}

function formatGridWindow(grid: VideoGridImage, index: number, total: number): string {
  const tileSummary = grid.tiles.map((tile) => tile.timestampLabel).join(", ");
  return `Grid ${index}/${total} covers ${grid.startSec.toFixed(3)}s-${grid.endSec.toFixed(3)}s. Tile timestamps: ${tileSummary}`;
}

function formatFrameLabel(frame: VideoFrameImage): string {
  return `Frame at ${frame.timestampLabel} (${frame.timestampSec.toFixed(3)}s)`;
}

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
async function expandPaths(paths: string | string[]): Promise<string[]> {
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

function formatTranscriptSegments(
  segments: Awaited<ReturnType<typeof transcribeAudio>>,
  maxChars: number,
): string {
  const formatted = segments
    .map((s) => `[${(s.start / 1000).toFixed(1)}–${(s.end / 1000).toFixed(1)}] ${s.text.trim()}`)
    .join("\n");

  return truncateTranscript(formatted, maxChars);
}

// ---------------------------------------------------------------------------
// Transcript format helpers (Phase 9)
// ---------------------------------------------------------------------------

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

function formatMediaInfo(info: MediaInfo): string {
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
  return lines.join("\n");
}

export async function handleProbeMedia(
  args: ProbeMediaArgs,
): Promise<McpSuccessResult | McpErrorResult> {
  try {
    const paths = await expandPaths(args.paths);

    const requestedMax = args.max_files ?? DEFAULT_PROBE_MAX_FILES;
    const maxFiles = Math.min(requestedMax, ABSOLUTE_MAX_PROBE_FILES);

    if (paths.length > maxFiles) {
      throw buildInvalidBatchError(
        `This request expands to ${paths.length} files, but the current limit is ${maxFiles} (absolute max ${ABSOLUTE_MAX_PROBE_FILES}). Narrow the glob, pass fewer file_paths, or set max_files up to ${ABSOLUTE_MAX_PROBE_FILES}.`,
      );
    }

    const results = await Promise.allSettled(paths.map((p) => probeMedia(p)));

    const lines: string[] = [];
    let succeeded = 0;
    let failed = 0;

    for (let i = 0; i < results.length; i++) {
      const result = results[i]!;
      if (result.status === "fulfilled") {
        succeeded++;
        lines.push(formatMediaInfo(result.value));
      } else {
        failed++;
        const reason: unknown = result.reason;
        const msg =
          reason instanceof MediaError
            ? `[${reason.code}] ${reason.message}`
            : reason instanceof Error
              ? reason.message
              : String(reason);
        lines.push(`File: ${paths[i]}\nError: ${msg}`);
      }
      lines.push(""); // blank separator
    }

    const summary = `Probed ${paths.length} file(s): ${succeeded} succeeded, ${failed} failed.`;
    const text = `${summary}\n\n${lines.join("\n").trim()}`;

    return { content: [{ type: "text", text }] };
  } catch (err) {
    return mcpError(err);
  }
}

export async function handleUnderstandMedia(
  args: UnderstandMediaArgs,
): Promise<McpSuccessResult | McpErrorResult> {
  try {
    const info = await probeMedia(args.file_path);
    preflightFileSize(info, "understand_media");
    preflightDuration(info, PREFLIGHT_MAX_DURATION_FULL, "understand_media");

    const maxTotalChars = getTotalCharBudget(args.max_total_chars);

    const result = await understandMedia(
      args.file_path,
      buildOpts({
        model: args.model,
        maxChars: args.max_chars,
        maxGrids: args.max_grids,
        startSec: args.start_sec,
        endSec: args.end_sec,
        samplingStrategy: args.sampling_strategy,
        secondsPerFrame: args.seconds_per_frame,
        secondsPerGrid: args.seconds_per_grid,
        aspectMode: args.aspect_mode,
        cols: args.cols,
        rows: args.rows,
        thumbWidth: args.thumb_width,
      }),
    );

    const content = await buildUnderstandMediaContent(result, args.file_path, maxTotalChars);

    content.push({ type: "text", text: summarizePayload(content) });
    assertFitsBudget(
      content,
      maxTotalChars,
      "The final understand_media payload exceeds the current budget.",
    );

    return { content };
  } catch (err) {
    return mcpError(err);
  }
}

// ---------------------------------------------------------------------------
// Segment ↔ grid alignment helpers (exported for testing)
// ---------------------------------------------------------------------------

/**
 * Compute how many milliseconds of a segment [segStart, segEnd) overlap
 * with a window [winStartMs, winEndMs).
 */
export function overlapMs(
  segStart: number,
  segEnd: number,
  winStartMs: number,
  winEndMs: number,
): number {
  const start = Math.max(segStart, winStartMs);
  const end = Math.min(segEnd, winEndMs);
  return Math.max(0, end - start);
}

/**
 * Assign each segment to the grid whose time window overlaps it most.
 * Segments that don't overlap any grid go to the `unassigned` bucket.
 *
 * Returns an array parallel to `grids` (each entry is the indices into
 * `segments` assigned to that grid), plus an `unassigned` array.
 */
export function assignSegmentsToGrids(
  segments: Segment[],
  grids: VideoGridImage[],
): { perGrid: number[][]; unassigned: number[] } {
  const perGrid: number[][] = grids.map(() => []);
  const unassigned: number[] = [];

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si]!;
    let bestGrid = -1;
    let bestOverlap = 0;

    for (let gi = 0; gi < grids.length; gi++) {
      const g = grids[gi]!;
      const winStartMs = g.startSec * 1000;
      const winEndMs = g.endSec * 1000;
      const overlap = overlapMs(seg.start, seg.end, winStartMs, winEndMs);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestGrid = gi;
      }
    }

    if (bestGrid >= 0) {
      perGrid[bestGrid]!.push(si);
    } else {
      unassigned.push(si);
    }
  }

  return { perGrid, unassigned };
}

/** Format a list of segments as timestamped lines: `[start–end] text`. */
function formatSegmentLines(segments: Segment[], indices: number[]): string {
  return indices
    .map((i) => {
      const s = segments[i]!;
      return `[${(s.start / 1000).toFixed(1)}–${(s.end / 1000).toFixed(1)}] ${s.text.trim()}`;
    })
    .join("\n");
}

async function buildUnderstandMediaContent(
  result: Awaited<ReturnType<typeof understandMedia>>,
  requestedPath: string,
  maxTotalChars: number,
): Promise<McpContentItem[]> {
  const content: McpContentItem[] = [];

  const { info } = result;

  // ---- Metadata block (same for all media types) ----
  const metaLines: string[] = [
    `File: ${info.path}`,
    `Type: ${info.type}`,
    `Duration: ${info.duration.toFixed(1)}s`,
  ];
  if (info.width) metaLines.push(`Resolution: ${info.width}x${info.height}`);
  if (info.fps) metaLines.push(`FPS: ${info.fps.toFixed(2)}`);
  if (info.videoCodec) metaLines.push(`Video codec: ${info.videoCodec}`);
  if (info.audioCodec) metaLines.push(`Audio codec: ${info.audioCodec}`);
  if (info.sampleRate) metaLines.push(`Sample rate: ${info.sampleRate}Hz`);
  if (info.channels) metaLines.push(`Channels: ${info.channels}`);
  if (info.fileSizeBytes !== undefined) metaLines.push(`Size: ${info.fileSizeBytes} bytes`);

  const grids = result.gridImages;
  const hasGrids = info.type === "video" && grids.length > 0;

  if (hasGrids) {
    metaLines.push(
      `Video sampling: ${grids.length} grid image(s), each tile has an exact timestamp overlay.`,
    );
    metaLines.push(
      `Output format: transcript segments are interleaved with their corresponding visual grids by time alignment.`,
    );
  }

  content.push({ type: "text", text: metaLines.join("\n") });
  assertFitsBudget(content, maxTotalChars, "Metadata alone exceeds the current budget.");

  // ---- Image file: unchanged ----
  if (info.type === "image") {
    const raw = await readFile(info.path);
    const compressed = await compressForLLM(raw);
    const imageItem: McpContentItem = {
      type: "image",
      data: compressed.toString("base64"),
      mimeType: "image/jpeg",
    };
    if (!appendIfFits(content, imageItem, maxTotalChars)) {
      const itemChars = JSON.stringify(imageItem).length;
      throw richBudgetError({
        context: `The image file alone is ~${itemChars.toLocaleString()} chars`,
        actualChars: serializedContentLength(content) + itemChars,
        budgetChars: maxTotalChars,
        mediaInfo: info,
      });
    }
    // Images have no transcript — return early
    return content;
  }

  // ---- Audio-only: single transcript block ----
  if (!hasGrids && result.segments.length > 0) {
    const transcriptHeader = "\n--- TRANSCRIPT ---\n";
    const baseLength = serializedContentLength(content);
    const availableForTranscript = Math.max(0, maxTotalChars - baseLength - 256);
    if (availableForTranscript > transcriptHeader.length + 32) {
      const allIndices = result.segments.map((_, i) => i);
      const transcriptBody = truncateTranscript(
        formatSegmentLines(result.segments, allIndices),
        Math.max(32, availableForTranscript - transcriptHeader.length),
      );
      const transcriptItem: McpContentItem = {
        type: "text",
        text: `${transcriptHeader}${transcriptBody}`,
      };
      if (!appendIfFits(content, transcriptItem, maxTotalChars)) {
        const itemChars = JSON.stringify(transcriptItem).length;
        throw richBudgetError({
          context: `Transcript plus existing content would be ~${(serializedContentLength(content) + itemChars).toLocaleString()} chars`,
          actualChars: serializedContentLength(content) + itemChars,
          budgetChars: maxTotalChars,
        });
      }
    }
    return content;
  }

  // ---- Video with grids: interleaved transcript + images ----
  if (hasGrids) {
    // Sort grids by start time (should already be sorted, but be safe)
    const sortedGrids = [...grids].sort((a, b) => a.startSec - b.startSec);

    const { perGrid, unassigned } = assignSegmentsToGrids(result.segments, sortedGrids);

    let addedGrids = 0;

    for (let gi = 0; gi < sortedGrids.length; gi++) {
      const grid = sortedGrids[gi]!;
      const segmentIndices = perGrid[gi]!;

      // Emit transcript chunk for this window (if there are segments)
      if (segmentIndices.length > 0) {
        const chunkText = formatSegmentLines(result.segments, segmentIndices);
        const header = `\n--- TRANSCRIPT ${grid.startSec.toFixed(1)}s–${grid.endSec.toFixed(1)}s ---\n`;
        const transcriptItem: McpContentItem = {
          type: "text",
          text: `${header}${truncateTranscript(chunkText, Math.max(32, maxTotalChars - serializedContentLength(content) - 1024))}`,
        };
        // Best-effort: skip transcript chunk if it doesn't fit (grid is more valuable)
        appendIfFits(content, transcriptItem, maxTotalChars);
      }

      // Emit grid label + image
      const labelItem: McpContentItem = {
        type: "text",
        text: `\n--- FRAME GRID ${gi + 1}/${sortedGrids.length} ---\n${formatGridWindow(grid, gi + 1, sortedGrids.length)}`,
      };
      const imageItem: McpContentItem = {
        type: "image",
        data: grid.image.toString("base64"),
        mimeType: "image/jpeg",
      };

      const fitsLabel = appendIfFits(content, labelItem, maxTotalChars);
      const fitsImage = fitsLabel && appendIfFits(content, imageItem, maxTotalChars);

      if (!fitsLabel || !fitsImage) {
        if (!fitsImage && fitsLabel) content.pop(); // remove label we added
        const imageChars = JSON.stringify(imageItem).length;
        throw richBudgetError({
          context: `Grid image ${gi + 1}/${sortedGrids.length} is ~${imageChars.toLocaleString()} chars`,
          actualChars: serializedContentLength(content) + imageChars,
          budgetChars: maxTotalChars,
          mediaInfo: info,
        });
      }

      addedGrids += 1;
    }

    // Emit any transcript segments not covered by any grid window
    if (unassigned.length > 0) {
      const remainingText = formatSegmentLines(result.segments, unassigned);
      const header = `\n--- TRANSCRIPT (remaining) ---\n`;
      const remainingItem: McpContentItem = {
        type: "text",
        text: `${header}${truncateTranscript(remainingText, Math.max(32, maxTotalChars - serializedContentLength(content) - 256))}`,
      };
      appendIfFits(content, remainingItem, maxTotalChars);
    }

    if (addedGrids === 0) {
      throw richBudgetError({
        context: `Even 1 timestamped grid image for ${requestedPath} does not fit`,
        actualChars: serializedContentLength(content),
        budgetChars: maxTotalChars,
        mediaInfo: info,
      });
    }
  }

  return content;
}

export async function handleGetVideoGrids(
  args: GetVideoGridsArgs,
): Promise<McpSuccessResult | McpErrorResult> {
  try {
    const info = await probeMedia(args.file_path);
    preflightFileSize(info, "get_video_grids");

    const grids = await extractFrameGridImages(
      args.file_path,
      buildOpts({
        maxGrids: args.max_grids,
        startSec: args.start_sec,
        endSec: args.end_sec,
        samplingStrategy: args.sampling_strategy,
        sceneThreshold: args.scene_threshold,
        frameInterval: args.frame_interval,
        secondsPerFrame: args.seconds_per_frame,
        secondsPerGrid: args.seconds_per_grid,
        cols: args.cols,
        rows: args.rows,
        aspectMode: args.aspect_mode,
        thumbWidth: args.thumb_width,
      }),
    );

    if (grids.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "No grid frames found. Try shortening the window, lowering seconds_per_frame, lowering seconds_per_grid, or using sampling_strategy=scene.",
          },
        ],
      };
    }

    const maxTotalChars = getTotalCharBudget(args.max_total_chars);
    const content: McpContentItem[] = [
      {
        type: "text",
        text: `Extracted ${grids.length} timestamped grid image(s). Omit max_grids to let the server auto-fit more grids within the current budget.`,
      },
    ];

    assertFitsBudget(content, maxTotalChars, "The intro text already exceeds the current budget.");

    for (let i = 0; i < grids.length; i++) {
      const grid = grids[i];
      if (!grid) continue;

      const labelItem: McpContentItem = {
        type: "text",
        text: formatGridWindow(grid, i + 1, grids.length),
      };
      const imageItem: McpContentItem = {
        type: "image",
        data: grid.image.toString("base64"),
        mimeType: "image/jpeg",
      };

      const fitsLabel = appendIfFits(content, labelItem, maxTotalChars);
      const fitsImage = fitsLabel && appendIfFits(content, imageItem, maxTotalChars);

      if (!fitsLabel || !fitsImage) {
        if (args.max_grids !== undefined) {
          const imageChars = JSON.stringify(imageItem).length;
          throw richBudgetError({
            context: `Requested ${args.max_grids} grid image(s), but grid ${i + 1} alone is ~${imageChars.toLocaleString()} chars`,
            actualChars: serializedContentLength(content) + imageChars,
            budgetChars: maxTotalChars,
            mediaInfo: info,
            ...(args.thumb_width !== undefined ? { thumbWidth: args.thumb_width } : {}),
          });
        }
        if (!fitsImage && fitsLabel) content.pop();
        break;
      }
    }

    content.push({ type: "text", text: summarizePayload(content) });
    assertFitsBudget(
      content,
      maxTotalChars,
      "The final get_video_grids payload exceeds the current budget.",
    );

    return { content };
  } catch (err) {
    return mcpError(err);
  }
}

export async function handleGetFrames(
  args: GetFramesArgs,
): Promise<McpSuccessResult | McpErrorResult> {
  try {
    const info = await probeMedia(args.file_path);
    preflightFileSize(info, "get_frames");

    const maxTotalChars = getTotalCharBudget(args.max_total_chars);
    const content: McpContentItem[] = [];

    for (const ts of args.timestamps) {
      const frame = await extractFrameImage(args.file_path, ts);
      const labelItem: McpContentItem = { type: "text", text: formatFrameLabel(frame) };
      const imageItem: McpContentItem = {
        type: "image",
        data: frame.image.toString("base64"),
        mimeType: "image/jpeg",
      };

      const fitsLabel = appendIfFits(content, labelItem, maxTotalChars);
      const fitsImage = fitsLabel && appendIfFits(content, imageItem, maxTotalChars);

      if (!fitsLabel || !fitsImage) {
        if (!fitsImage && fitsLabel) content.pop();
        const imageChars = JSON.stringify(imageItem).length;
        throw richBudgetError({
          context: `Requested ${args.timestamps.length} frame(s), but each frame is ~${imageChars.toLocaleString()} chars`,
          actualChars: serializedContentLength(content) + imageChars,
          budgetChars: maxTotalChars,
          mediaInfo: info,
          hintMode: "frame",
        });
      }
    }

    content.push({ type: "text", text: summarizePayload(content) });
    assertFitsBudget(
      content,
      maxTotalChars,
      "The final get_frames payload exceeds the current budget.",
    );

    return { content };
  } catch (err) {
    return mcpError(err);
  }
}

export async function handleGetTranscript(
  args: GetTranscriptArgs,
): Promise<McpSuccessResult | McpErrorResult> {
  try {
    const info = await probeMedia(args.file_path);
    preflightFileSize(info, "get_transcript");
    preflightDuration(info, PREFLIGHT_MAX_DURATION_TRANSCRIPT, "get_transcript");

    const maxChars =
      args.max_chars ?? parseInt(process.env["MEDIA_UNDERSTANDING_MAX_CHARS"] ?? "32000", 10);

    const allSegments = await transcribeAudio(args.file_path, buildOpts({ model: args.model }));

    // Apply time-window filter (if specified)
    const segments = filterSegmentsByWindow(allSegments, args.start_sec, args.end_sec);

    if (segments.length === 0) {
      const windowNote =
        args.start_sec !== undefined || args.end_sec !== undefined
          ? ` in the requested window (${args.start_sec ?? 0}s–${args.end_sec ?? "end"})`
          : "";
      return {
        content: [{ type: "text", text: `No speech detected${windowNote}.` }],
      };
    }

    const format = args.format ?? "text";
    let text: string;
    switch (format) {
      case "srt":
        text = formatTranscriptAsSRT(segments, maxChars);
        break;
      case "json":
        text = formatTranscriptAsJSON(segments, maxChars);
        break;
      default:
        text = formatTranscriptSegments(segments, maxChars);
        break;
    }

    return {
      content: [{ type: "text", text }],
    };
  } catch (err) {
    return mcpError(err);
  }
}
