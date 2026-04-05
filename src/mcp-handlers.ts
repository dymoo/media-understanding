/**
 * MCP tool handler functions.
 *
 * Each exported `handle*` function corresponds to one MCP tool.
 * Shared concerns are split into focused modules:
 *   - mcp-types.ts    — MCP result types, arg types, mcpError()
 *   - mcp-budget.ts   — payload budget enforcement + vision token estimation
 *   - mcp-preflight.ts — file size / duration safety checks
 *   - mcp-format.ts   — path expansion, grid/frame labels, transcript formatting
 */

import { readFile } from "node:fs/promises";

import {
  compressForLLM,
  extractFrameGridImages,
  extractFrameImages,
  probeMedia,
  transcribeAudio,
  truncateTranscript,
  understandMedia,
} from "./media.js";
import {
  appendIfFits,
  assertFitsBudget,
  getTotalCharBudget,
  richBudgetError,
  serializedContentLength,
  summarizePayload,
} from "./mcp-budget.js";
import {
  buildOpts,
  expandPaths,
  filterSegmentsByWindow,
  formatFrameLabel,
  formatGridWindow,
  formatMediaInfo,
  formatTranscriptAsJSON,
  formatTranscriptAsSRT,
  formatTranscriptSegments,
} from "./mcp-format.js";
import {
  PREFLIGHT_MAX_DURATION_FULL,
  PREFLIGHT_MAX_DURATION_TRANSCRIPT,
  preflightDuration,
  preflightFileSize,
} from "./mcp-preflight.js";
import type {
  FetchYoutubeArgs,
  GetFramesArgs,
  GetTranscriptArgs,
  GetVideoGridsArgs,
  McpContentItem,
  McpErrorResult,
  McpSuccessResult,
  ProbeMediaArgs,
  UnderstandMediaArgs,
} from "./mcp-types.js";
import { mcpError } from "./mcp-types.js";
import type { Segment, VideoGridImage } from "./types.js";
import { MediaError } from "./types.js";
import {
  downloadAudio,
  downloadSubtitles,
  downloadThumbnail,
  downloadVideo,
  getVideoInfo,
  isUrl,
  parseSubtitlesToSegments,
  resolveUrlToAudioPath,
  resolveUrlToLocalPath,
} from "./youtube.js";

// ---------------------------------------------------------------------------
// Re-exports for backward compatibility
// ---------------------------------------------------------------------------

export type { McpContentItem, McpErrorResult, McpSuccessResult } from "./mcp-types.js";
export type {
  FetchYoutubeArgs,
  GetFramesArgs,
  GetTranscriptArgs,
  GetVideoGridsArgs,
  ProbeMediaArgs,
  UnderstandMediaArgs,
} from "./mcp-types.js";
export { mcpError } from "./mcp-types.js";

export {
  PREFLIGHT_MAX_DURATION_FULL,
  PREFLIGHT_MAX_DURATION_TRANSCRIPT,
  PREFLIGHT_MAX_FILE_SIZE,
  formatDuration,
  preflightDuration,
  preflightFileSize,
} from "./mcp-preflight.js";

export { estimateVisionTokens } from "./mcp-budget.js";

export {
  filterSegmentsByWindow,
  formatSrtTimestamp,
  formatTranscriptAsJSON,
  formatTranscriptAsSRT,
} from "./mcp-format.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_PROBE_MAX_FILES = 50;
const ABSOLUTE_MAX_PROBE_FILES = 200;

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

// ---------------------------------------------------------------------------
// Tool handlers
// ---------------------------------------------------------------------------

export async function handleProbeMedia(
  args: ProbeMediaArgs,
): Promise<McpSuccessResult | McpErrorResult> {
  try {
    // Resolve any URLs to local cached files before path expansion
    const rawPaths = Array.isArray(args.paths) ? args.paths : [args.paths];
    const resolvedInputs = await Promise.all(
      rawPaths.map((p) => (isUrl(p.trim()) ? resolveUrlToLocalPath(p.trim()) : Promise.resolve(p))),
    );
    const paths = await expandPaths(resolvedInputs);

    const requestedMax = args.max_files ?? DEFAULT_PROBE_MAX_FILES;
    const maxFiles = Math.min(requestedMax, ABSOLUTE_MAX_PROBE_FILES);

    if (paths.length > maxFiles) {
      throw new MediaError(
        "INVALID_SAMPLING",
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
    // Resolve URL to local cached file if needed
    const filePath = isUrl(args.file_path)
      ? await resolveUrlToLocalPath(args.file_path)
      : args.file_path;
    const info = await probeMedia(filePath);
    preflightFileSize(info, "understand_media");
    preflightDuration(info, PREFLIGHT_MAX_DURATION_FULL, "understand_media");

    const maxTotalChars = getTotalCharBudget(args.max_total_chars);

    const result = await understandMedia(
      filePath,
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

    const content = await buildUnderstandMediaContent(result, filePath, maxTotalChars);

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
    // Resolve URL to local cached file if needed
    const filePath = isUrl(args.file_path)
      ? await resolveUrlToLocalPath(args.file_path)
      : args.file_path;
    const info = await probeMedia(filePath);
    preflightFileSize(info, "get_video_grids");

    const grids = await extractFrameGridImages(
      filePath,
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
    // Resolve URL to local cached file if needed
    const filePath = isUrl(args.file_path)
      ? await resolveUrlToLocalPath(args.file_path)
      : args.file_path;
    const info = await probeMedia(filePath);
    preflightFileSize(info, "get_frames");

    const maxTotalChars = getTotalCharBudget(args.max_total_chars);
    const content: McpContentItem[] = [];

    // Batch-extract all frames in a single Demuxer session.
    const frames = await extractFrameImages(filePath, args.timestamps);

    for (const frame of frames) {
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
    const maxChars =
      args.max_chars ?? parseInt(process.env["MEDIA_UNDERSTANDING_MAX_CHARS"] ?? "32000", 10);

    let allSegments: Segment[];

    if (isUrl(args.file_path)) {
      // Fast path: try yt-dlp subtitles first (no video download needed)
      const subsPath = await downloadSubtitles(args.file_path);
      if (subsPath) {
        allSegments = parseSubtitlesToSegments(subsPath);
      } else {
        // Slow path: no subtitles available — download audio and Whisper it
        const audioPath = await resolveUrlToAudioPath(args.file_path);
        const info = await probeMedia(audioPath);
        preflightFileSize(info, "get_transcript");
        preflightDuration(info, PREFLIGHT_MAX_DURATION_TRANSCRIPT, "get_transcript");
        allSegments = await transcribeAudio(audioPath, buildOpts({ model: args.model }));
      }
    } else {
      const info = await probeMedia(args.file_path);
      preflightFileSize(info, "get_transcript");
      preflightDuration(info, PREFLIGHT_MAX_DURATION_TRANSCRIPT, "get_transcript");
      allSegments = await transcribeAudio(args.file_path, buildOpts({ model: args.model }));
    }

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

// ---------------------------------------------------------------------------
// fetch_youtube handler
// ---------------------------------------------------------------------------

export async function handleFetchYoutube(
  args: FetchYoutubeArgs,
): Promise<McpSuccessResult | McpErrorResult> {
  try {
    const includeSubtitles = args.include_subtitles !== false; // default true
    const includeVideo = args.include_video === true;
    const includeAudio = args.include_audio === true;
    const includeThumbnail = args.include_thumbnail === true;

    // Always fetch info first
    const info = await getVideoInfo(args.url);

    const lines: string[] = [
      `Title: ${info.title}`,
      `Duration: ${info.duration}s`,
      `Uploader: ${info.uploader}`,
      `URL: ${info.url}`,
      "",
    ];

    // Subtitles (default: on)
    let subtitleSegments: Segment[] | null = null;
    if (includeSubtitles) {
      const subsPath = await downloadSubtitles(args.url);
      if (subsPath) {
        subtitleSegments = parseSubtitlesToSegments(subsPath);
        lines.push(`Subtitles: ${subsPath}`);
        lines.push(`Subtitle languages available: ${info.subtitleLanguages.join(", ") || "auto-generated"}`);
      } else {
        lines.push(
          "Subtitles: none available — use get_transcript on the audio/video path below for Whisper transcription fallback.",
        );
      }
    }

    // Thumbnail
    if (includeThumbnail) {
      const thumbPath = await downloadThumbnail(args.url);
      lines.push(`Thumbnail: ${thumbPath}`);
    }

    // Video (lowest quality for frame analysis)
    if (includeVideo) {
      const videoPath = await downloadVideo(args.url);
      lines.push(`Video: ${videoPath}`);
    }

    // Audio
    if (includeAudio) {
      const audioPath = await downloadAudio(args.url);
      lines.push(`Audio: ${audioPath}`);
    }

    lines.push("");
    lines.push(
      "Use these paths with probe_media, understand_media, get_video_grids, get_frames, or get_transcript for further analysis.",
    );

    const content: McpContentItem[] = [{ type: "text", text: lines.join("\n") }];

    // Inline subtitle transcript text when available (saves an extra tool call)
    if (subtitleSegments && subtitleSegments.length > 0) {
      const maxChars = parseInt(process.env["MEDIA_UNDERSTANDING_MAX_CHARS"] ?? "32000", 10);
      const transcriptText = formatTranscriptSegments(subtitleSegments, maxChars);
      content.push({
        type: "text",
        text: `\n--- TRANSCRIPT (from subtitles) ---\n${transcriptText}`,
      });
    }

    return { content };
  } catch (err) {
    return mcpError(err);
  }
}
