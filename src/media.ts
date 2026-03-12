/**
 * Core media processing: probe, transcribe, extract grids/frames.
 */

import { execFile as execFileCb } from "node:child_process";
import { createHash } from "node:crypto";
import { open, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

import { fileTypeFromFile } from "file-type";
import { Decoder, Demuxer, WhisperDownloader, WhisperTranscriber } from "node-av/api";
import { avGetCodecName } from "node-av/lib";
import { ffmpegPath, isFfmpegAvailable } from "node-av/ffmpeg";
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

const execFile = promisify(execFileCb);

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
const DEFAULT_SAMPLING_STRATEGY = "uniform" as const;
const FRAME_EXTRACTION_CONCURRENCY = 4;
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

/**
 * Map over items with bounded concurrency.
 * At most `limit` items are processed simultaneously.
 * Safe in single-threaded JS: index increment is synchronous between await points.
 */
async function mapWithLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (nextIndex < items.length) {
      const i = nextIndex++;
      results[i] = await fn(items[i]!);
    }
  }

  const workerCount = Math.min(limit, items.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
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

async function buildVideoFrameImage(
  filePath: string,
  timestampSec: number,
  overlayPrefix?: string,
): Promise<VideoFrameImage> {
  const image = await extractFrame(filePath, timestampSec, overlayPrefix);
  return {
    image,
    timestampSec,
    timestampLabel: formatTimestamp(timestampSec),
  };
}

/**
 * Extract a single video frame and return both the image and its exact timestamp.
 */
export async function extractFrameImage(
  filePath: string,
  timestampSec: number,
): Promise<VideoFrameImage> {
  const { abs } = await assertFile(filePath);
  return buildVideoFrameImage(abs, timestampSec, abs);
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

  const segments: Segment[] = [];

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
      segments.push({
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

  const normalized = normalizeGridOptions(opts, info.duration);
  const plans = planVideoSampling(normalized, info.duration);

  try {
    const grids: VideoGridImage[] = [];
    for (const plan of plans) {
      const frames = await mapWithLimit(
        plan.timestampsSec,
        FRAME_EXTRACTION_CONCURRENCY,
        (timestampSec) => buildVideoFrameImage(abs, timestampSec, abs),
      );

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
 * Returns a compressed JPEG Buffer.
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

    const safeTimestamp = Math.min(timestampSec, Math.max(0, info.duration - 0.2));
    const ffmpeg = ffmpegPath();
    const args = [
      "-i",
      abs,
      "-ss",
      String(safeTimestamp),
      "-vframes",
      "1",
      "-f",
      "image2pipe",
      "-vcodec",
      "mjpeg",
      "pipe:1",
    ];

    const { stdout } = await execFile(ffmpeg, args, {
      encoding: "buffer",
      maxBuffer: 32 * 1024 * 1024,
    });

    if (!stdout || stdout.length === 0) {
      throw new MediaError(
        "FRAME_FAILED",
        `No frame data returned at ${safeTimestamp}s in: ${abs}`,
      );
    }

    const compressed = await compressForLLM(stdout as unknown as Buffer);
    const overlayLabel =
      `${overlayPrefix ? `${overlayPrefix.split("/").at(-1) ?? overlayPrefix} ` : ""}${formatTimestamp(safeTimestamp)}`.trim();
    return addTimestampOverlay(compressed, overlayLabel);
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
