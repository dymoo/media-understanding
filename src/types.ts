/**
 * Shared types for media-understanding.
 */

/** A Whisper transcription segment with millisecond timestamps. */
export interface Segment {
  /** Start time in milliseconds. */
  start: number;
  /** End time in milliseconds. */
  end: number;
  /** Transcribed text. */
  text: string;
  /** Speaker turn (when VAD is enabled). */
  turn?: boolean;
}

/** Metadata extracted from a media file. */
export interface MediaInfo {
  /** Absolute path to the file. */
  path: string;
  /** Detected media type. */
  type: "video" | "audio" | "image" | "unknown";
  /** Duration in seconds (0 for images). */
  duration: number;
  /** Video width in pixels (undefined for audio). */
  width?: number;
  /** Video height in pixels (undefined for audio). */
  height?: number;
  /** Video codec name e.g. "h264" (undefined for audio). */
  videoCodec?: string;
  /** Approximate video frame rate (undefined for audio). */
  fps?: number;
  /** Audio codec name e.g. "aac" (undefined for images). */
  audioCodec?: string;
  /** Audio sample rate in Hz (undefined for images). */
  sampleRate?: number;
  /** Audio channel count (undefined for images). */
  channels?: number;
  /** File size in bytes. */
  fileSizeBytes?: number;
}

/** Options controlling keyframe grid extraction. */
export interface GridOptions {
  /**
   * Maximum number of grid images to return.
   * @default 6
   */
  maxGrids?: number;
  /**
   * Start offset in seconds (inclusive).
   * @default 0
   */
  startSec?: number;
  /**
   * End offset in seconds (exclusive). Defaults to end of file.
   */
  endSec?: number;
  /**
   * Sampling strategy for video frames.
   * `uniform` spreads samples across the requested window.
   * `scene` uses scene-change detection plus periodic fallback sampling.
   * @default "uniform"
   */
  samplingStrategy?: "uniform" | "scene";
  /**
   * Scene-change detection threshold (0–1). Higher = fewer keyframes.
   * @default 0.3
   */
  sceneThreshold?: number;
  /**
   * Fallback sampling: select a frame every N frames even without scene change.
   * @default 300  (10 s at 30 fps)
   */
  frameInterval?: number;
  /**
   * Desired spacing between sampled frames within a grid, in seconds.
   * If omitted, the spacing is derived from the requested time window.
   */
  secondsPerFrame?: number;
  /**
   * Desired spacing between composite grid windows, in seconds.
   * If omitted, the spacing is derived from the requested time window.
   */
  secondsPerGrid?: number;
  /**
   * Columns in each grid tile.
   * @default 4
   */
  cols?: number;
  /**
   * Rows in each grid tile.
   * @default 4
   */
  rows?: number;
  /**
   * Thumbnail width in pixels (height auto-scaled).
   * @default 480
   */
  thumbWidth?: number;
  /**
   * How frames should fit inside each tile.
   * `contain` keeps the whole frame visible with letterboxing/pillarboxing.
   * `cover` fills the tile and crops as needed.
   * @default "contain"
   */
  aspectMode?: "contain" | "cover";
}

/** Options controlling transcription. */
export interface TranscribeOptions {
  /**
   * Whisper model name.
   * @default "base.en-q5_1"
   */
  model?: string;
  /**
   * Maximum characters of transcript to return (truncates middle).
   * @default 32000
   */
  maxChars?: number;
}

/** Options for understand_media — superset of grid + transcribe options. */
export interface ProcessOptions extends GridOptions, TranscribeOptions {}

/** One exact extracted video frame. */
export interface VideoFrameImage {
  /** JPEG image buffer ready for MCP serialization. */
  image: Buffer;
  /** Exact extraction timestamp in seconds. */
  timestampSec: number;
  /** Human-readable timestamp label, e.g. 00:01:23.456. */
  timestampLabel: string;
}

/** One timestamped tile inside a composite grid image. */
export interface VideoGridTile {
  /** Exact extraction timestamp in seconds. */
  timestampSec: number;
  /** Human-readable timestamp label, e.g. 00:01:23.456. */
  timestampLabel: string;
}

/** One composite grid image plus exact tile timestamps. */
export interface VideoGridImage {
  /** JPEG grid image buffer ready for MCP serialization. */
  image: Buffer;
  /** Inclusive start of the covered window in seconds. */
  startSec: number;
  /** Exclusive end of the covered window in seconds. */
  endSec: number;
  /** Exact timestamps for each tile in row-major order. */
  tiles: VideoGridTile[];
}

/** The combined result returned by understandMedia(). */
export interface UnderstandResult {
  info: MediaInfo;
  /** Raw transcript segments (before truncation). */
  segments: Segment[];
  /** Transcript string (possibly truncated). */
  transcript: string;
  /** Keyframe grid images as JPEG buffers (video only). */
  grids: Buffer[];
  /** Rich grid metadata, when available (video only). */
  gridImages: VideoGridImage[];
}

/** Error codes for MediaError. */
export type MediaErrorCode =
  | "FFMPEG_NOT_FOUND"
  | "FILE_NOT_FOUND"
  | "UNSUPPORTED_FORMAT"
  | "NO_AUDIO_STREAM"
  | "NO_VIDEO_STREAM"
  | "TRANSCRIBE_FAILED"
  | "GRID_FAILED"
  | "FRAME_FAILED"
  | "INVALID_SAMPLING"
  | "BUDGET_EXCEEDED"
  | "UNKNOWN";

/** Structured error thrown by media processing functions. */
export class MediaError extends Error {
  constructor(
    public readonly code: MediaErrorCode,
    message: string,
    // `override` required: ES2022 Error base class exposes a `cause` property
    public override readonly cause?: unknown,
  ) {
    super(message);
    this.name = "MediaError";
  }
}
