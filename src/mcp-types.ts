/**
 * Shared types and helpers used across MCP handler modules.
 */

import { MediaError } from "./types.js";

// ---------------------------------------------------------------------------
// MCP result types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MCP tool argument types
// ---------------------------------------------------------------------------

export type UnderstandMediaArgs = {
  file_path: string;
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
  max_chars?: number | undefined;
  format?: "text" | "srt" | "json" | undefined;
  start_sec?: number | undefined;
  end_sec?: number | undefined;
};

export type ProbeMediaArgs = {
  paths: string | string[];
  max_files?: number | undefined;
};

export type FetchYtdlpArgs = {
  url: string;
  include_video?: boolean | undefined;
  include_audio?: boolean | undefined;
  include_thumbnail?: boolean | undefined;
  include_subtitles?: boolean | undefined;
};

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

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
