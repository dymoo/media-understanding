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
}

/** Options controlling transcription. */
export interface TranscribeOptions {
  /**
   * Whisper model name.
   * @default "tiny.en"
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

/** The combined result returned by understandMedia(). */
export interface UnderstandResult {
  info: MediaInfo;
  /** Raw transcript segments (before truncation). */
  segments: Segment[];
  /** Transcript string (possibly truncated). */
  transcript: string;
  /** Keyframe grid images as JPEG buffers (video only). */
  grids: Buffer[];
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
