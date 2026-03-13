/**
 * Core media processing: probe, transcribe, extract grids/frames.
 */

import { createHash } from "node:crypto";
import { open, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

import { fileTypeFromFile } from "file-type";
import { Decoder, Demuxer, WhisperDownloader, WhisperTranscriber } from "node-av/api";
import { AVSEEK_FLAG_BACKWARD, AV_PIX_FMT_RGB24, SWS_BILINEAR } from "node-av/constants";
import { isFfmpegAvailable } from "node-av/ffmpeg";
import { FFmpegError as NativeFFmpegError, Frame, SoftwareScaleContext } from "node-av/lib";
import { avGetCodecName } from "node-av/lib";
import sharp from "sharp";

import type {
  GridOptions,
  MediaInfo,
  ProcessOptions,
  Segment,
  TranscribeOptions,
  VideoFrameImage,
  VideoGridImage,
  VideoGridTile,
  UnderstandResult,
} from "./types.js";
import { MediaError } from "./types.js";

// ---------------------------------------------------------------------------
// Process-level counting semaphore for heavy operations (frame extraction,
// transcription). Limits concurrent native-resource usage to avoid memory
// spikes and GPU contention. Excess callers queue as Promises (FIFO).
// ---------------------------------------------------------------------------

const HEAVY_OP_CONCURRENCY = 2;

class Semaphore {
  private permits: number;
  private queue: Array<() => void> = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }
    return new Promise<void>((resolve) => {
      this.queue.push(resolve);
    });
  }

  release(): void {
    const next = this.queue.shift();
    if (next) {
      next();
    } else {
      this.permits++;
    }
  }
}

const heavySemaphore = new Semaphore(HEAVY_OP_CONCURRENCY);

/** Run `fn` under the shared heavy-operation semaphore. */
async function withHeavyOp<T>(fn: () => Promise<T>): Promise<T> {
  await heavySemaphore.acquire();
  try {
    return await fn();
  } finally {
    heavySemaphore.release();
  }
}

// ---------------------------------------------------------------------------
// Bounded LRU transcript cache (keyed by SHA-256 content fingerprint).
// Max 32 entries; transcripts exceeding 500K chars of segment text are not
// cached to avoid unbounded memory growth on very long files.
// ---------------------------------------------------------------------------

const TRANSCRIPT_CACHE_MAX_ENTRIES = 32;
const TRANSCRIPT_CACHE_MAX_TEXT_CHARS = 500_000;

class TranscriptCache {
  private cache = new Map<string, Segment[]>();

  get(key: string): Segment[] | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Promote to most-recently-used (Map preserves insertion order)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: string, segments: Segment[]): void {
    const totalChars = segments.reduce((sum, s) => sum + s.text.length, 0);
    if (totalChars > TRANSCRIPT_CACHE_MAX_TEXT_CHARS) return;

    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= TRANSCRIPT_CACHE_MAX_ENTRIES) {
      // Evict least-recently-used (first entry in Map iteration order)
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
    }
    this.cache.set(key, segments);
  }
}

const transcriptCache = new TranscriptCache();

// ---------------------------------------------------------------------------
// Extension fallback for formats file-type cannot detect via magic bytes
// (e.g. SVG is XML with no magic header, RAW camera formats, etc.)
// ---------------------------------------------------------------------------

const IMAGE_EXTENSIONS_FALLBACK = new Set([
  ".svg",
  ".svgz",
  ".cr2",
  ".cr3",
  ".nef",
  ".arw",
  ".dng",
  ".orf",
  ".rw2",
  ".psd",
]);

const DEFAULT_SCENE_THRESHOLD = 0.3;
const DEFAULT_FRAME_INTERVAL = 300;
const DEFAULT_COLS = 4;
const DEFAULT_ROWS = 4;
const DEFAULT_THUMB_WIDTH = 480;
const DEFAULT_ASPECT_MODE = "contain" as const;

// Portrait video (height > width) produces tall tiles that blow up grid payloads.
// When the caller omits grid-shape params, use these smaller overview-oriented defaults.
const PORTRAIT_DEFAULT_COLS = 3;
const PORTRAIT_DEFAULT_ROWS = 3;
const PORTRAIT_DEFAULT_THUMB_WIDTH = 120;
const DEFAULT_SAMPLING_STRATEGY = "uniform" as const;
const OVERLAY_BANNER_HEIGHT = 34;
const OVERLAY_FONT_SIZE = 20;
const OVERLAY_PADDING = 8;

interface NormalizedGridOptions {
  maxGrids: number;
  startSec: number;
  endSec: number;
  samplingStrategy: "uniform" | "scene";
  sceneThreshold: number;
  frameInterval: number;
  secondsPerFrame: number | undefined;
  secondsPerGrid: number | undefined;
  cols: number;
  rows: number;
  thumbWidth: number;
  aspectMode: "contain" | "cover";
}

interface PlannedGridWindow {
  startSec: number;
  endSec: number;
  timestampsSec: number[];
}

/**
 * Classify a file by content (magic bytes first, extension fallback).
 * Returns "image" | "av" | "unknown".
 *   image  → probe with sharp
 *   av     → probe with Demuxer (handles both audio and video)
 *   unknown → try Demuxer, let it fail with its own error if unsupported
 */
async function classifyFile(abs: string): Promise<"image" | "av" | "unknown"> {
  const ft = await fileTypeFromFile(abs);

  if (ft) {
    if (ft.mime.startsWith("image/")) return "image";
    if (ft.mime.startsWith("audio/") || ft.mime.startsWith("video/")) return "av";
    // application/ogg, application/mp4, etc. — let Demuxer handle them
    return "av";
  }

  // file-type returned undefined (no magic bytes matched)
  // Check extension as last resort for known image-only formats
  const dot = abs.lastIndexOf(".");
  const ext = dot >= 0 ? abs.slice(dot).toLowerCase() : "";
  if (IMAGE_EXTENSIONS_FALLBACK.has(ext)) return "image";

  // Unknown — attempt Demuxer; it will throw UNSUPPORTED_FORMAT if it can't open
  return "unknown";
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Return the directory where Whisper GGML models are cached. */
export function resolveModelDir(): string {
  const xdg = process.env["XDG_CACHE_HOME"];
  if (xdg) return join(xdg, "media-understanding", "models");

  if (process.platform === "win32") {
    const localAppData = process.env["LOCALAPPDATA"];
    if (localAppData) return join(localAppData, "media-understanding", "models");
  }

  return join(homedir(), ".cache", "media-understanding", "models");
}

/** Resize + JPEG-encode a buffer for LLM consumption (max 1280px wide, q=75). */
export async function compressForLLM(input: Buffer, maxWidth = 1280): Promise<Buffer> {
  return sharp(input)
    .resize({ width: maxWidth, withoutEnlargement: true })
    .jpeg({ quality: 75 })
    .toBuffer();
}

/** Throw MediaError.FFMPEG_NOT_FOUND if ffmpeg binary is missing. */
function assertFfmpeg(): void {
  if (!isFfmpegAvailable()) {
    throw new MediaError(
      "FFMPEG_NOT_FOUND",
      "FFmpeg binary not found. Re-install node-av: `npm install node-av`",
    );
  }
}

/** Resolve absolute path, assert the file exists, and return path + size. */
async function assertFile(filePath: string): Promise<{ abs: string; size: number }> {
  const abs = resolve(filePath);
  try {
    const s = await stat(abs);
    return { abs, size: s.size };
  } catch {
    throw new MediaError("FILE_NOT_FOUND", `File not found: ${abs}`);
  }
}

const HASH_SAMPLE_BYTES = 64 * 1024; // 64 KB

/**
 * Compute a fast content fingerprint for a file.
 * Hashes: file size (as string) + first 64 KB + last 64 KB.
 * Constant-time I/O regardless of file size; collision-resistant in practice.
 */
async function fileFingerprint(filePath: string): Promise<string> {
  const s = await stat(filePath);
  const size = s.size;
  const hash = createHash("sha256");
  hash.update(String(size));
  const fh = await open(filePath, "r");
  try {
    const headBuf = Buffer.allocUnsafe(Math.min(HASH_SAMPLE_BYTES, size));
    await fh.read(headBuf, 0, headBuf.length, 0);
    hash.update(headBuf);
    if (size > HASH_SAMPLE_BYTES * 2) {
      const tailBuf = Buffer.allocUnsafe(HASH_SAMPLE_BYTES);
      await fh.read(tailBuf, 0, HASH_SAMPLE_BYTES, size - HASH_SAMPLE_BYTES);
      hash.update(tailBuf);
    }
  } finally {
    await fh.close();
  }
  return hash.digest("hex");
}

/**
 * Truncate a transcript string to at most `maxChars`.
 * Keeps the first 60% and last 40% with a […] separator.
 */
export function truncateTranscript(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const keep1 = Math.floor(maxChars * 0.6);
  const keep2 = maxChars - keep1;
  return text.slice(0, keep1) + "\n\n[…transcript truncated…]\n\n" + text.slice(-keep2);
}

function formatTimestamp(timestampSec: number): string {
  const totalMs = Math.max(0, Math.round(timestampSec * 1000));
  const hours = Math.floor(totalMs / 3_600_000);
  const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
  const seconds = Math.floor((totalMs % 60_000) / 1000);
  const millis = totalMs % 1000;

  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}.${millis
    .toString()
    .padStart(3, "0")}`;
}

function escapeSvgText(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

async function addTimestampOverlay(
  input: Buffer,
  label: string,
  widthHint?: number,
): Promise<Buffer> {
  const meta = await sharp(input).metadata();
  const width = meta.width ?? widthHint ?? DEFAULT_THUMB_WIDTH;
  const height = meta.height ?? Math.round((width * 9) / 16);
  const safeLabel = escapeSvgText(label);

  const svg = Buffer.from(`
    <svg width="${width}" height="${OVERLAY_BANNER_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="${width}" height="${OVERLAY_BANNER_HEIGHT}" fill="rgba(0,0,0,0.82)" />
      <text
        x="${width - OVERLAY_PADDING}"
        y="${Math.round(OVERLAY_BANNER_HEIGHT / 2) + Math.round(OVERLAY_FONT_SIZE / 3)}"
        text-anchor="end"
        font-family="Menlo, Monaco, Consolas, monospace"
        font-size="${OVERLAY_FONT_SIZE}"
        fill="#ffffff"
      >${safeLabel}</text>
    </svg>
  `);

  return sharp({
    create: {
      width,
      height: height + OVERLAY_BANNER_HEIGHT,
      channels: 3,
      background: { r: 0, g: 0, b: 0 },
    },
  })
    .composite([
      { input, left: 0, top: 0 },
      { input: svg, left: 0, top: height },
    ])
    .jpeg({ quality: 85 })
    .toBuffer();
}

function normalizeGridOptions(opts: GridOptions, durationSec: number): NormalizedGridOptions {
  const maxGrids =
    opts.maxGrids ?? parseInt(process.env["MEDIA_UNDERSTANDING_MAX_GRIDS"] ?? "6", 10);
  const startSec = opts.startSec ?? 0;
  const endSec = Math.min(opts.endSec ?? durationSec, durationSec);
  const samplingStrategy = opts.samplingStrategy ?? DEFAULT_SAMPLING_STRATEGY;
  const sceneThreshold = opts.sceneThreshold ?? DEFAULT_SCENE_THRESHOLD;
  const frameInterval = opts.frameInterval ?? DEFAULT_FRAME_INTERVAL;
  const cols = opts.cols ?? DEFAULT_COLS;
  const rows = opts.rows ?? DEFAULT_ROWS;
  const thumbWidth = opts.thumbWidth ?? DEFAULT_THUMB_WIDTH;
  const aspectMode = opts.aspectMode ?? DEFAULT_ASPECT_MODE;

  if (!Number.isFinite(durationSec) || durationSec <= 0) {
    throw new MediaError("INVALID_SAMPLING", "Video duration must be greater than 0.");
  }

  if (maxGrids <= 0) {
    throw new MediaError("INVALID_SAMPLING", "maxGrids must be greater than 0.");
  }

  if (startSec < 0) {
    throw new MediaError("INVALID_SAMPLING", `startSec must be >= 0, got ${startSec}.`);
  }

  if (endSec <= startSec) {
    throw new MediaError(
      "INVALID_SAMPLING",
      `endSec must be greater than startSec. Got startSec=${startSec}, endSec=${endSec}.`,
    );
  }

  if (opts.secondsPerFrame !== undefined && opts.secondsPerFrame <= 0) {
    throw new MediaError(
      "INVALID_SAMPLING",
      `secondsPerFrame must be > 0, got ${opts.secondsPerFrame}.`,
    );
  }

  if (opts.secondsPerGrid !== undefined && opts.secondsPerGrid <= 0) {
    throw new MediaError(
      "INVALID_SAMPLING",
      `secondsPerGrid must be > 0, got ${opts.secondsPerGrid}.`,
    );
  }

  return {
    maxGrids,
    startSec,
    endSec,
    samplingStrategy,
    sceneThreshold,
    frameInterval,
    secondsPerFrame: opts.secondsPerFrame,
    secondsPerGrid: opts.secondsPerGrid,
    cols,
    rows,
    thumbWidth,
    aspectMode,
  };
}

function buildUniformSamplingPlan(opts: NormalizedGridOptions): PlannedGridWindow[] {
  const framesPerGrid = opts.cols * opts.rows;
  const totalWindow = opts.endSec - opts.startSec;
  const gridCount = opts.maxGrids;
  const gridSpan = opts.secondsPerGrid ?? totalWindow / gridCount;
  const frameSpan = opts.secondsPerFrame ?? gridSpan / framesPerGrid;

  if (frameSpan <= 0 || gridSpan <= 0) {
    throw new MediaError(
      "INVALID_SAMPLING",
      "The requested sampling settings produce zero-width frame or grid windows.",
    );
  }

  const requiredWindow = (gridCount - 1) * gridSpan + framesPerGrid * frameSpan;
  if (requiredWindow - totalWindow > 1e-6) {
    throw new MediaError(
      "INVALID_SAMPLING",
      `Requested window ${opts.startSec.toFixed(3)}s-${opts.endSec.toFixed(3)}s is too short for ${gridCount} grid(s) at seconds_per_grid=${gridSpan.toFixed(3)} and seconds_per_frame=${frameSpan.toFixed(3)}. Increase seconds_per_grid, increase seconds_per_frame, reduce max_grids, or shorten the number of frames per grid.`,
    );
  }

  const slack = Math.max(0, totalWindow - requiredWindow);
  const offset = slack / 2;
  const windows: PlannedGridWindow[] = [];

  for (let gridIndex = 0; gridIndex < gridCount; gridIndex += 1) {
    const gridStart = opts.startSec + offset + gridIndex * gridSpan;
    const gridEnd = Math.min(opts.endSec, gridStart + framesPerGrid * frameSpan);
    const timestampsSec: number[] = [];

    for (let frameIndex = 0; frameIndex < framesPerGrid; frameIndex += 1) {
      const binStart = gridStart + frameIndex * frameSpan;
      const binCenter = binStart + frameSpan / 2;
      const safeEnd = Math.max(opts.startSec, opts.endSec - 0.001);
      timestampsSec.push(Math.min(Math.max(binCenter, opts.startSec), safeEnd));
    }

    windows.push({ startSec: gridStart, endSec: gridEnd, timestampsSec });
  }

  return windows;
}

function buildSceneSamplingPlan(
  opts: NormalizedGridOptions,
  durationSec: number,
): PlannedGridWindow[] {
  const fps = 30;
  const frameIntervalSec = Math.max(1 / fps, opts.frameInterval / fps);
  const clone: NormalizedGridOptions = {
    ...opts,
    secondsPerFrame: opts.secondsPerFrame ?? frameIntervalSec,
    secondsPerGrid: opts.secondsPerGrid,
    endSec: Math.min(opts.endSec, durationSec),
  };

  return buildUniformSamplingPlan(clone);
}

function planVideoSampling(opts: NormalizedGridOptions, durationSec: number): PlannedGridWindow[] {
  if (opts.samplingStrategy === "scene") {
    return buildSceneSamplingPlan(opts, durationSec);
  }

  return buildUniformSamplingPlan(opts);
}

/**
 * Extract a single video frame and return both the image and its exact timestamp.
 */
export async function extractFrameImage(
  filePath: string,
  timestampSec: number,
): Promise<VideoFrameImage> {
  assertFfmpeg();
  const { abs } = await assertFile(filePath);
  const batch = await extractFramesBatch(abs, [timestampSec], abs);
  const result = batch[0];
  if (!result) {
    throw new MediaError("FRAME_FAILED", `No frame data returned at ${timestampSec}s in: ${abs}`);
  }
  return {
    image: result.buffer,
    timestampSec: result.timestampSec,
    timestampLabel: formatTimestamp(result.timestampSec),
  };
}

/**
 * Extract multiple video frames in one batch call.
 * Returns VideoFrameImage[] in the same order as the input timestamps.
 * Opens a single Demuxer/Decoder session for all frames — much faster
 * than calling extractFrameImage() in a loop.
 */
export async function extractFrameImages(
  filePath: string,
  timestampsSec: number[],
): Promise<VideoFrameImage[]> {
  assertFfmpeg();
  const { abs } = await assertFile(filePath);
  const batch = await extractFramesBatch(abs, timestampsSec, abs);
  return batch.map((r) => ({
    image: r.buffer,
    timestampSec: r.timestampSec,
    timestampLabel: formatTimestamp(r.timestampSec),
  }));
}

// ---------------------------------------------------------------------------
// extractFramesBatch — native batch frame extraction via node-av
// ---------------------------------------------------------------------------

/**
 * Extract multiple video frames in a single semaphore-guarded session.
 * Opens a fresh Demuxer+Decoder per target timestamp because the node-av
 * `packets()` async generator cannot be restarted after a seek. Each
 * Demuxer open is lightweight (~1ms) vs. the old FFmpeg CLI spawn (~50ms).
 *
 * After feeding all packets from the seek point forward, the decoder is
 * flushed (send null packet) to emit B-frame-buffered frames — critical
 * for codecs with reordering delay (e.g. H.264 with B-frames).
 *
 * The SoftwareScaleContext (YUV→RGB24) is lazily initialised and reused
 * across timestamps as long as source dimensions/format remain constant.
 *
 * Guarded by the process-level heavy-op semaphore.
 *
 * @param filePath Absolute path to a video file.
 * @param timestampsSec Timestamps in seconds to extract (need not be sorted).
 * @param overlayPrefix Optional prefix for the timestamp overlay label.
 * @returns Array of { timestampSec, buffer } in the SAME order as `timestampsSec`.
 */
async function extractFramesBatch(
  filePath: string,
  timestampsSec: number[],
  overlayPrefix?: string,
): Promise<Array<{ timestampSec: number; buffer: Buffer }>> {
  if (timestampsSec.length === 0) return [];

  // Build sorted work list, preserving original indices for output ordering.
  const work = timestampsSec.map((ts, i) => ({ ts, originalIndex: i }));
  work.sort((a, b) => a.ts - b.ts);

  const results = new Array<{ timestampSec: number; buffer: Buffer }>(timestampsSec.length);

  await withHeavyOp(async () => {
    // Shared scaler state — lazy-init'd on first frame, reused across timestamps.
    let scaler: SoftwareScaleContext | null = null;
    let dstFrame: Frame | null = null;
    let scalerSrcW = 0;
    let scalerSrcH = 0;
    let scalerSrcFmt = -1;

    try {
      for (const item of work) {
        // Open a fresh Demuxer+Decoder per timestamp (packets() is not reusable after seek).
        await using demuxer = await Demuxer.open(filePath);
        const videoStream = demuxer.video();
        if (!videoStream) {
          throw new MediaError("NO_VIDEO_STREAM", `No video stream found in: ${filePath}`);
        }

        const streamIndex = videoStream.index;
        const duration = demuxer.duration;
        const targetSec = Math.min(item.ts, Math.max(0, duration - 0.2));
        const tb = videoStream.timeBase;
        const targetPts = BigInt(Math.round((targetSec * tb.den) / tb.num));

        using decoder = await Decoder.create(videoStream);

        // Seek to the nearest keyframe before the target timestamp.
        if (targetSec > 0) {
          await demuxer.seek(targetSec, streamIndex, AVSEEK_FLAG_BACKWARD);
        }

        // Decode forward, collecting the best frame (closest to or past targetPts).
        let bestFrame: Frame | null = null;
        let bestPts: bigint = BigInt(-1);
        let foundExact = false;

        // Phase 1: Feed packets and drain frames.
        for await (const packet of demuxer.packets(streamIndex)) {
          if (packet === null) break;

          await decoder.decode(packet);

          let frame: Frame | null | undefined;
          while (
            (frame = (await decoder.receive()) as Frame | null) !== null &&
            frame !== undefined
          ) {
            const framePts = frame.bestEffortTimestamp;

            if (framePts >= targetPts) {
              // At or past target — keep it and stop.
              if (bestFrame) bestFrame.free();
              bestFrame = frame;
              bestPts = framePts;
              foundExact = true;
              break;
            }

            // Before target — keep closest so far.
            if (bestFrame) bestFrame.free();
            bestFrame = frame;
            bestPts = framePts;
          }

          if (foundExact) break;
        }

        // Phase 2: Flush decoder to emit B-frame-buffered frames.
        if (!foundExact) {
          try {
            await decoder.decode(null);
          } catch {
            // Some decoders don't accept null flush — ignore.
          }

          let frame: Frame | null | undefined;
          while (
            (frame = (await decoder.receive()) as Frame | null) !== null &&
            frame !== undefined
          ) {
            const framePts = frame.bestEffortTimestamp;

            if (bestFrame === null || (bestPts < targetPts && framePts >= targetPts)) {
              if (bestFrame) bestFrame.free();
              bestFrame = frame;
              bestPts = framePts;
              if (framePts >= targetPts) break;
            } else if (framePts > bestPts && framePts < targetPts) {
              // Closer to target but still before it.
              if (bestFrame) bestFrame.free();
              bestFrame = frame;
              bestPts = framePts;
            } else {
              frame.free();
            }
          }
        }

        if (!bestFrame) {
          throw new MediaError("FRAME_FAILED", `No frame decoded at ${item.ts}s in: ${filePath}`);
        }

        try {
          // Lazy-init or reinit scaler if source dimensions/format changed.
          const srcW = bestFrame.width;
          const srcH = bestFrame.height;
          const srcFmt = bestFrame.format as number;

          if (!scaler || srcW !== scalerSrcW || srcH !== scalerSrcH || srcFmt !== scalerSrcFmt) {
            if (scaler) scaler.freeContext();
            if (dstFrame) dstFrame.free();

            scaler = new SoftwareScaleContext();
            scaler.getContext(
              srcW,
              srcH,
              bestFrame.format as typeof AV_PIX_FMT_RGB24,
              srcW,
              srcH,
              AV_PIX_FMT_RGB24,
              SWS_BILINEAR,
            );
            const initRet = scaler.initContext();
            NativeFFmpegError.throwIfError(initRet, "initContext");

            dstFrame = new Frame();
            dstFrame.alloc();
            dstFrame.width = srcW;
            dstFrame.height = srcH;
            dstFrame.format = AV_PIX_FMT_RGB24;
            const allocRet = dstFrame.allocBuffer();
            NativeFFmpegError.throwIfError(allocRet, "allocBuffer");

            scalerSrcW = srcW;
            scalerSrcH = srcH;
            scalerSrcFmt = srcFmt;
          }

          // Scale YUV → RGB24
          const scaleRet = await scaler.scaleFrame(dstFrame!, bestFrame);
          NativeFFmpegError.throwIfError(scaleRet, "scaleFrame");

          // Get RGB buffer and encode to JPEG via sharp.
          const rgbBuffer = dstFrame!.toBuffer();
          const compressed = await sharp(rgbBuffer, {
            raw: { width: srcW, height: srcH, channels: 3 },
          })
            .resize({ width: Math.min(srcW, 1280), withoutEnlargement: true })
            .jpeg({ quality: 75 })
            .toBuffer();

          // Add timestamp overlay.
          const safeTs = Math.min(item.ts, Math.max(0, duration - 0.2));
          const overlayLabel =
            `${overlayPrefix ? `${overlayPrefix.split("/").at(-1) ?? overlayPrefix} ` : ""}${formatTimestamp(safeTs)}`.trim();
          const withOverlay = await addTimestampOverlay(compressed, overlayLabel);

          results[item.originalIndex] = {
            timestampSec: item.ts,
            buffer: withOverlay,
          };
        } finally {
          bestFrame.free();
        }
      }
    } finally {
      if (dstFrame) dstFrame.free();
      if (scaler) scaler.freeContext();
    }
  });

  return results;
}

// ---------------------------------------------------------------------------
// probeMedia
// ---------------------------------------------------------------------------

/**
 * Return metadata about a media file without decoding any frames.
 * File type is detected by magic bytes (via file-type) with an extension
 * fallback for formats that have no magic header (e.g. SVG).
 * Images are probed via sharp; audio/video via Demuxer.
 */
export async function probeMedia(filePath: string): Promise<MediaInfo> {
  const { abs, size: fileSizeBytes } = await assertFile(filePath);
  const kind = await classifyFile(abs);

  if (kind === "image") {
    try {
      const meta = await sharp(abs).metadata();
      return {
        path: abs,
        type: "image",
        duration: 0,
        width: meta.width,
        height: meta.height,
        fileSizeBytes,
      };
    } catch (err) {
      throw new MediaError("UNSUPPORTED_FORMAT", `Cannot read image: ${abs}`, err);
    }
  }

  // "av" or "unknown" — try Demuxer
  assertFfmpeg();

  try {
    await using input = await Demuxer.open(abs);

    const videoStream = input.video();
    const audioStream = input.audio();

    const hasVideo = videoStream !== undefined;
    const hasAudio = audioStream !== undefined;

    let type: MediaInfo["type"] = "unknown";
    if (hasVideo) type = "video";
    else if (hasAudio) type = "audio";

    const info: MediaInfo = {
      path: abs,
      type,
      duration: input.duration,
      fileSizeBytes,
    };

    if (videoStream) {
      const cp = videoStream.codecpar;
      info.width = cp.width;
      info.height = cp.height;
      const vc = avGetCodecName(cp.codecId);
      if (vc !== null) info.videoCodec = vc;
      const fr = videoStream.avgFrameRate;
      if (fr.den !== 0) info.fps = fr.num / fr.den;
    }

    if (audioStream) {
      const cp = audioStream.codecpar;
      const ac = avGetCodecName(cp.codecId);
      if (ac !== null) info.audioCodec = ac;
      info.sampleRate = cp.sampleRate;
      info.channels = cp.channels;
    }

    return info;
  } catch (err) {
    if (err instanceof MediaError) throw err;
    throw new MediaError("UNSUPPORTED_FORMAT", `Cannot probe media: ${abs}`, err);
  }
}

// ---------------------------------------------------------------------------
// transcribeAudio
// ---------------------------------------------------------------------------

/**
 * Transcribe the audio track of a media file.
 * Results are cached in-memory keyed by SHA-256 content fingerprint.
 */
export async function transcribeAudio(
  filePath: string,
  opts: TranscribeOptions = {},
): Promise<Segment[]> {
  assertFfmpeg();
  const { abs } = await assertFile(filePath);

  const modelName = opts.model ?? process.env["MEDIA_UNDERSTANDING_MODEL"] ?? "base.en-q5_1";

  if (!WhisperDownloader.isValidModel(modelName)) {
    throw new MediaError(
      "TRANSCRIBE_FAILED",
      `Invalid Whisper model name: "${modelName}". ` +
        `Valid models: tiny, tiny.en, base, base.en, small, small.en, ` +
        `medium, medium.en, large-v1, large-v2, large-v3, ` +
        `tiny.en-q5_1, base.en-q5_1, small.en-q5_1, large-v3-turbo-q5_0, etc.`,
    );
  }

  const cacheKey = await fileFingerprint(abs);
  const cached = transcriptCache.get(cacheKey);
  if (cached) return cached;

  const modelDir = resolveModelDir();

  const segments: Segment[] = await withHeavyOp(async () => {
    const segs: Segment[] = [];

    try {
      await using demuxer = await Demuxer.open(abs);

      const audioStream = demuxer.audio();
      if (!audioStream) {
        throw new MediaError("NO_AUDIO_STREAM", `No audio stream found in: ${abs}`);
      }

      using decoder = await Decoder.create(audioStream);
      using transcriber = await WhisperTranscriber.create({
        model: modelName,
        modelDir,
        language: "auto",
      });

      for await (const seg of transcriber.transcribe(
        decoder.frames(demuxer.packets(audioStream.index)),
      )) {
        segs.push({
          start: seg.start,
          end: seg.end,
          text: seg.text,
          ...(seg.turn !== undefined && { turn: seg.turn }),
        });
      }
    } catch (err) {
      if (err instanceof MediaError) throw err;
      throw new MediaError("TRANSCRIBE_FAILED", `Transcription failed: ${abs}`, err);
    }

    return segs;
  });

  transcriptCache.set(cacheKey, segments);
  return segments;
}

// ---------------------------------------------------------------------------
// extractFrameGrid
// ---------------------------------------------------------------------------

/**
 * Extract a set of keyframe grid images from a video file.
 * Returns an array of JPEG Buffers — one per grid tile sheet.
 */
export async function extractFrameGrid(
  filePath: string,
  opts: GridOptions = {},
): Promise<Buffer[]> {
  const grids = await extractFrameGridImages(filePath, opts);
  return grids.map((grid) => grid.image);
}

/**
 * Extract timestamped composite grid images from a video file.
 */
export async function extractFrameGridImages(
  filePath: string,
  opts: GridOptions = {},
): Promise<VideoGridImage[]> {
  assertFfmpeg();
  const { abs } = await assertFile(filePath);

  const info = await probeMedia(abs);
  if (info.type !== "video") {
    throw new MediaError("NO_VIDEO_STREAM", `No video stream found in: ${abs}`);
  }

  // Apply portrait-aware defaults when the caller omits grid-shape params.
  const isPortrait =
    info.width !== undefined && info.height !== undefined && info.height > info.width;
  const effectiveOpts: GridOptions = isPortrait
    ? {
        ...opts,
        cols: opts.cols ?? PORTRAIT_DEFAULT_COLS,
        rows: opts.rows ?? PORTRAIT_DEFAULT_ROWS,
        thumbWidth: opts.thumbWidth ?? PORTRAIT_DEFAULT_THUMB_WIDTH,
      }
    : opts;

  const normalized = normalizeGridOptions(effectiveOpts, info.duration);
  const plans = planVideoSampling(normalized, info.duration);

  try {
    const grids: VideoGridImage[] = [];
    for (const plan of plans) {
      // Batch-extract all frames for this grid in a single Demuxer session.
      const batchResults = await extractFramesBatch(abs, plan.timestampsSec, abs);

      const frames: VideoFrameImage[] = batchResults.map((r) => ({
        image: r.buffer,
        timestampSec: r.timestampSec,
        timestampLabel: formatTimestamp(r.timestampSec),
      }));

      if (frames.length === 0) continue;

      const grid = await composeGrid(
        frames,
        normalized.cols,
        normalized.thumbWidth,
        normalized.aspectMode,
      );
      grids.push({
        image: await compressForLLM(grid),
        startSec: plan.startSec,
        endSec: plan.endSec,
        tiles: frames.map<VideoGridTile>((frame) => ({
          timestampSec: frame.timestampSec,
          timestampLabel: frame.timestampLabel,
        })),
      });
    }

    return grids;
  } catch (err) {
    if (err instanceof MediaError) throw err;
    throw new MediaError("GRID_FAILED", `Frame grid extraction failed: ${abs}`, err);
  }
}

/**
 * Compose a list of JPEG file paths into a single grid image.
 * Arranges tiles left-to-right, top-to-bottom.
 */
async function composeGrid(
  frames: VideoFrameImage[],
  cols: number,
  thumbWidth: number,
  aspectMode: "contain" | "cover",
): Promise<Buffer> {
  if (frames.length === 0) {
    throw new MediaError("GRID_FAILED", "No frames to compose into grid");
  }

  const [firstFrame] = frames;
  if (!firstFrame) {
    throw new MediaError("GRID_FAILED", "No frames to compose into grid");
  }

  const firstMeta = await sharp(firstFrame.image).metadata();
  const sourceW = firstMeta.width ?? thumbWidth;
  const sourceH = firstMeta.height ?? Math.round((thumbWidth * 3) / 4);
  const tileW = thumbWidth;
  const tileH = Math.max(1, Math.round((thumbWidth * sourceH) / sourceW));

  const thumbs = await Promise.all(
    frames.map(async (frame) => {
      if (aspectMode === "cover") {
        return sharp(frame.image)
          .resize(tileW, tileH, { fit: "cover", position: "centre" })
          .jpeg({ quality: 82 })
          .toBuffer();
      }

      return sharp({
        create: {
          width: tileW,
          height: tileH,
          channels: 3,
          background: { r: 0, g: 0, b: 0 },
        },
      })
        .composite([
          {
            input: await sharp(frame.image)
              .resize(tileW, tileH, { fit: "contain", background: { r: 0, g: 0, b: 0 } })
              .jpeg({ quality: 82 })
              .toBuffer(),
            left: 0,
            top: 0,
          },
        ])
        .jpeg({ quality: 82 })
        .toBuffer();
    }),
  );

  const rows = Math.ceil(frames.length / cols);
  const gridW = cols * tileW;
  const gridH = rows * tileH;

  const compositeInputs = thumbs.map((buf, i) => ({
    input: buf,
    left: (i % cols) * tileW,
    top: Math.floor(i / cols) * tileH,
  }));

  return sharp({
    create: {
      width: gridW,
      height: gridH,
      channels: 3,
      background: { r: 0, g: 0, b: 0 },
    },
  })
    .composite(compositeInputs)
    .jpeg({ quality: 85 })
    .toBuffer();
}

// ---------------------------------------------------------------------------
// extractFrame
// ---------------------------------------------------------------------------

/**
 * Extract a single video frame at the given timestamp (seconds).
 * Returns a compressed JPEG Buffer with timestamp overlay.
 * Delegates to extractFramesBatch() for the actual work.
 */
export async function extractFrame(
  filePath: string,
  timestampSec: number,
  overlayPrefix?: string,
): Promise<Buffer> {
  assertFfmpeg();
  const { abs } = await assertFile(filePath);

  if (timestampSec < 0) {
    throw new MediaError("FRAME_FAILED", `Timestamp must be >= 0, got: ${timestampSec}`);
  }

  try {
    const info = await probeMedia(abs);
    if (info.type !== "video") {
      throw new MediaError("NO_VIDEO_STREAM", `No video stream found in: ${abs}`);
    }

    const batch = await extractFramesBatch(abs, [timestampSec], overlayPrefix);
    const result = batch[0];
    if (!result) {
      throw new MediaError("FRAME_FAILED", `No frame data returned at ${timestampSec}s in: ${abs}`);
    }
    return result.buffer;
  } catch (err) {
    if (err instanceof MediaError) throw err;
    throw new MediaError(
      "FRAME_FAILED",
      `Frame extraction failed at ${timestampSec}s in: ${abs}`,
      err,
    );
  }
}

// ---------------------------------------------------------------------------
// understandMedia — high-level orchestrator
// ---------------------------------------------------------------------------

/**
 * Fully process a media file: probe + optional transcription + optional frame grids.
 * This is the main entry point used by the `understand_media` MCP tool.
 */
export async function understandMedia(
  filePath: string,
  opts: ProcessOptions = {},
): Promise<UnderstandResult> {
  const { abs } = await assertFile(filePath);
  const info = await probeMedia(abs);

  const maxChars =
    opts.maxChars ?? parseInt(process.env["MEDIA_UNDERSTANDING_MAX_CHARS"] ?? "32000", 10);

  let segments: Segment[] = [];
  let transcript = "";

  if (info.type === "audio" || info.type === "video") {
    segments = await transcribeAudio(abs, opts);
    const raw = segments
      .map((s) => s.text)
      .join(" ")
      .trim();
    transcript = truncateTranscript(raw, maxChars);
  }

  let grids: Buffer[] = [];
  let gridImages: VideoGridImage[] = [];
  if (info.type === "video") {
    gridImages = await extractFrameGridImages(abs, opts);
    grids = gridImages.map((grid) => grid.image);
  }

  return { info, segments, transcript, grids, gridImages };
}
