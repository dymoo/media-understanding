/**
 * Preflight safety checks for heavy MCP operations.
 *
 * Fail fast before expensive work if the file is too large or too long.
 */

import type { MediaInfo } from "./types.js";
import { MediaError } from "./types.js";

// ---------------------------------------------------------------------------
// Preflight safety thresholds
// ---------------------------------------------------------------------------

/** Max file duration (seconds) for understand_media (transcribes full file). */
export const PREFLIGHT_MAX_DURATION_FULL = 7200; // 2 hours

/** Max file duration (seconds) for get_transcript. */
export const PREFLIGHT_MAX_DURATION_TRANSCRIPT = 14_400; // 4 hours

/** Absolute max file size (bytes) for any heavy operation. */
export const PREFLIGHT_MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024; // 10 GB

// ---------------------------------------------------------------------------
// Preflight checks
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
