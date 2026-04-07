/**
 * Public API for @dymoo/media-understanding.
 *
 * Re-exports the core processing functions and all shared types.
 * The MCP server and CLI stub are separate entry points (mcp.ts / cli.ts).
 */

export {
  compressForLLM,
  extractFrame,
  extractFrameGridImages,
  extractFrameImage,
  extractFrameImages,
  extractFrameGrid,
  probeMedia,
  transcribeAudio,
  truncateTranscript,
  understandMedia,
} from "./media.js";

export { resolveModelDir } from "./model-manager.js";

export {
  handleGetFrames,
  handleGetTranscript,
  handleGetVideoGrids,
  handleProbeMedia,
  handleUnderstandMedia,
} from "./mcp-handlers.js";

export type {
  GridOptions,
  MediaInfo,
  ProcessOptions,
  Segment,
  TranscribeOptions,
  UnderstandResult,
  VideoFrameImage,
  VideoGridImage,
  VideoGridTile,
} from "./types.js";

export { MediaError } from "./types.js";
export type { MediaErrorCode } from "./types.js";
