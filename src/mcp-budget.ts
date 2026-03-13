/**
 * Budget enforcement for MCP responses.
 *
 * Controls payload size limits, serialized content measurement,
 * vision token estimation, and rich budget error messages.
 */

import type { McpContentItem } from "./mcp-types.js";
import type { MediaInfo } from "./types.js";
import { MediaError } from "./types.js";

const DEFAULT_TOTAL_CHARS = 48_000;

export function getTotalCharBudget(maxTotalChars?: number): number {
  return maxTotalChars ?? DEFAULT_TOTAL_CHARS;
}

export function serializedContentLength(content: McpContentItem[]): number {
  return JSON.stringify({ content }).length;
}

export function summarizePayload(content: McpContentItem[]): string {
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
export function richBudgetError(opts: {
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

export function assertFitsBudget(
  content: McpContentItem[],
  maxTotalChars: number,
  detail: string,
): void {
  const totalChars = serializedContentLength(content);
  if (totalChars > maxTotalChars) {
    throw buildBudgetExceededError(
      `${detail} (~${totalChars.toLocaleString()} chars, ${formatOverageRatio(totalChars, maxTotalChars)} at max_total_chars=${maxTotalChars.toLocaleString()}). Try increasing seconds_per_frame, increasing seconds_per_grid, shortening the time window, requesting fewer images, or lowering max_chars.`,
    );
  }
}

export function appendIfFits(
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
