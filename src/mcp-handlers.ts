import { readdir, readFile } from "node:fs/promises";
import { matchesGlob, resolve } from "node:path";

import {
  compressForLLM,
  extractFrameGridImages,
  extractFrameImage,
  probeMedia,
  transcribeAudio,
  truncateTranscript,
  understandMedia,
} from "./media.js";
import type { MediaInfo, VideoFrameImage, VideoGridImage } from "./types.js";
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

const DEFAULT_TOTAL_CHARS = 32_000;

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
};

export type ProbeMediaArgs = {
  file_path?: string | undefined;
  file_paths?: string[] | undefined;
  glob?: string | undefined;
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

function assertFitsBudget(content: McpContentItem[], maxTotalChars: number, detail: string): void {
  const totalChars = serializedContentLength(content);
  if (totalChars > maxTotalChars) {
    throw buildBudgetExceededError(
      `${detail} This response would be ${totalChars} chars but max_total_chars=${maxTotalChars}. Try increasing seconds_per_frame, increasing seconds_per_grid, shortening the time window, requesting fewer images, or lowering max_chars.`,
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

async function expandGlobPaths(pattern: string): Promise<string[]> {
  const normalizedPattern = pattern.replaceAll("\\", "/");
  const entries = await readdir(process.cwd(), { recursive: true, withFileTypes: true });

  return entries
    .filter((entry) => entry.isFile())
    .map((entry) => resolve(process.cwd(), entry.parentPath, entry.name))
    .filter((filePath) => matchesGlob(filePath.replaceAll("\\", "/"), normalizedPattern))
    .sort();
}

async function expandProbePaths(args: ProbeMediaArgs): Promise<string[]> {
  const sources = [
    args.file_path !== undefined,
    args.file_paths !== undefined,
    args.glob !== undefined,
  ].filter(Boolean).length;

  if (sources !== 1) {
    throw buildInvalidBatchError(
      'Provide exactly one of `file_path`, `file_paths`, or `glob`. Example: {"file_path":"/path/to/video.mp4"} or {"file_paths":["a.mp4","b.mp3"]} or {"glob":"media/**/*.mp4"}.',
    );
  }

  if (args.file_path !== undefined) {
    return [args.file_path];
  }

  if (args.file_paths !== undefined) {
    const deduped = Array.from(new Set(args.file_paths.filter((value) => value.trim().length > 0)));
    if (deduped.length === 0) {
      throw buildInvalidBatchError("`file_paths` must contain at least one non-empty path.");
    }
    return deduped;
  }

  const matches = await expandGlobPaths(args.glob as string);
  if (matches.length === 0) {
    throw buildInvalidBatchError(
      `The glob \`${args.glob}\` did not match any files. Narrow the pattern or pass explicit file_paths instead.`,
    );
  }
  return matches;
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
    const paths = await expandProbePaths(args);

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

async function buildUnderstandMediaContent(
  result: Awaited<ReturnType<typeof understandMedia>>,
  requestedPath: string,
  maxTotalChars: number,
): Promise<McpContentItem[]> {
  const content: McpContentItem[] = [];

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
  if (info.fileSizeBytes !== undefined) lines.push(`Size: ${info.fileSizeBytes} bytes`);
  if (info.type === "video" && result.gridImages.length > 0) {
    lines.push(
      `Video sampling: ${result.gridImages.length} grid image(s), each tile has an exact timestamp overlay.`,
    );
  }

  content.push({ type: "text", text: lines.join("\n") });
  assertFitsBudget(content, maxTotalChars, "Metadata alone exceeds the current budget.");

  if (result.transcript) {
    const transcriptHeader = "\n--- TRANSCRIPT ---\n";
    const baseLength = serializedContentLength(content);
    const availableForTranscript = Math.max(0, maxTotalChars - baseLength - 256);
    if (availableForTranscript > transcriptHeader.length + 32) {
      const transcriptBody = truncateTranscript(
        result.transcript,
        Math.max(32, availableForTranscript - transcriptHeader.length),
      );
      const transcriptItem: McpContentItem = {
        type: "text",
        text: `${transcriptHeader}${transcriptBody}`,
      };
      if (!appendIfFits(content, transcriptItem, maxTotalChars)) {
        throw buildBudgetExceededError(
          `Transcript plus existing content exceeds max_total_chars=${maxTotalChars}. Lower max_chars, shorten the window, or request fewer images.`,
        );
      }
    }
  }

  if (info.type === "image") {
    const raw = await readFile(info.path);
    const compressed = await compressForLLM(raw);
    const imageItem: McpContentItem = {
      type: "image",
      data: compressed.toString("base64"),
      mimeType: "image/jpeg",
    };
    if (!appendIfFits(content, imageItem, maxTotalChars)) {
      throw buildBudgetExceededError(
        `The image file alone exceeds max_total_chars=${maxTotalChars}. Increase max_total_chars or use a smaller source image.`,
      );
    }
  }

  let addedGrids = 0;
  for (let i = 0; i < result.gridImages.length; i++) {
    const grid = result.gridImages[i];
    if (!grid) continue;

    const labelItem: McpContentItem = {
      type: "text",
      text: `\n--- FRAME GRID ${i + 1}/${result.gridImages.length} ---\n${formatGridWindow(grid, i + 1, result.gridImages.length)}`,
    };
    const imageItem: McpContentItem = {
      type: "image",
      data: grid.image.toString("base64"),
      mimeType: "image/jpeg",
    };

    const fitsLabel = appendIfFits(content, labelItem, maxTotalChars);
    const fitsImage = fitsLabel && appendIfFits(content, imageItem, maxTotalChars);

    if (!fitsLabel || !fitsImage) {
      throw buildBudgetExceededError(
        `Requested visual output for ${requestedPath} cannot fit within max_total_chars=${maxTotalChars}. Increase seconds_per_frame, increase seconds_per_grid, or shorten the window.`,
      );
    }

    addedGrids += 1;
  }

  if (info.type === "video" && result.gridImages.length > 0 && addedGrids === 0) {
    throw buildBudgetExceededError(
      `Even 1 timestamped grid image for ${requestedPath} does not fit within max_total_chars=${maxTotalChars}. Increase seconds_per_frame, increase seconds_per_grid, shorten the time window, or raise max_total_chars.`,
    );
  }

  return content;
}

export async function handleGetVideoGrids(
  args: GetVideoGridsArgs,
): Promise<McpSuccessResult | McpErrorResult> {
  try {
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
          throw buildBudgetExceededError(
            `Requested ${args.max_grids} grid image(s), but they would exceed max_total_chars=${maxTotalChars}. Increase seconds_per_frame, increase seconds_per_grid, shorten the window, or request fewer grids.`,
          );
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
        throw buildBudgetExceededError(
          `Requested ${args.timestamps.length} frame(s), but they would exceed max_total_chars=${maxTotalChars}. Increase max_total_chars or request fewer timestamps. For broader coverage, use get_video_grids with larger seconds_per_frame or seconds_per_grid.`,
        );
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

    const segments = await transcribeAudio(args.file_path, buildOpts({ model: args.model }));

    if (segments.length === 0) {
      return {
        content: [{ type: "text", text: "No speech detected." }],
      };
    }

    return {
      content: [{ type: "text", text: formatTranscriptSegments(segments, maxChars) }],
    };
  } catch (err) {
    return mcpError(err);
  }
}
