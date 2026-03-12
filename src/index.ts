/**
 * Public API for @dymoo/media-understanding.
 *
 * Re-exports the core processing functions and all shared types.
 * The MCP server and CLI are separate entry points (mcp.ts / cli.ts).
 */

export {
  compressForLLM,
  extractFrame,
  extractFrameGrid,
  probeMedia,
  resolveModelDir,
  transcribeAudio,
  truncateTranscript,
  understandMedia,
} from "./media.js";

export type {
  GridOptions,
  MediaInfo,
  ProcessOptions,
  Segment,
  TranscribeOptions,
  UnderstandResult,
} from "./types.js";

export { MediaError } from "./types.js";
export type { MediaErrorCode } from "./types.js";
