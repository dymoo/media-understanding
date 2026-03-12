/**
 * Core media processing: probe, transcribe, extract grids/frames.
 */

import { execFile as execFileCb } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, open, readdir, rm, stat } from "node:fs/promises";
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
  UnderstandResult,
} from "./types.js";
import { MediaError } from "./types.js";

const execFile = promisify(execFileCb);

// ---------------------------------------------------------------------------
// In-memory transcript cache (keyed by SHA-256 content fingerprint: size + first 64 KB + last 64 KB)
// ---------------------------------------------------------------------------

const transcriptCache = new Map<string, Segment[]>();

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

/** Resolve absolute path and assert the file exists. */
async function assertFile(filePath: string): Promise<string> {
  const abs = resolve(filePath);
  try {
    await stat(abs);
  } catch {
    throw new MediaError("FILE_NOT_FOUND", `File not found: ${abs}`);
  }
  return abs;
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
  const abs = await assertFile(filePath);
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
  const abs = await assertFile(filePath);

  const cacheKey = await fileFingerprint(abs);
  const cached = transcriptCache.get(cacheKey);
  if (cached) return cached;

  const modelName = opts.model ?? process.env["MEDIA_UNDERSTANDING_MODEL"] ?? "tiny.en";

  if (!WhisperDownloader.isValidModel(modelName)) {
    throw new MediaError(
      "TRANSCRIBE_FAILED",
      `Invalid Whisper model name: "${modelName}". ` +
        `Valid models: tiny, tiny.en, base, base.en, small, small.en, ` +
        `medium, medium.en, large-v1, large-v2, large-v3, etc.`,
    );
  }

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
  assertFfmpeg();
  const abs = await assertFile(filePath);

  const maxGrids =
    opts.maxGrids ?? parseInt(process.env["MEDIA_UNDERSTANDING_MAX_GRIDS"] ?? "6", 10);
  const startSec = opts.startSec ?? 0;
  const endSec = opts.endSec; // undefined = end of file
  const sceneThreshold = opts.sceneThreshold ?? 0.3;
  const frameInterval = opts.frameInterval ?? 300;
  const cols = opts.cols ?? 4;
  const rows = opts.rows ?? 4;
  const thumbWidth = opts.thumbWidth ?? 480;

  const framesPerGrid = cols * rows;

  // Build the FFmpeg filter for frame selection.
  // select filter syntax for FFmpeg filter args (no shell, so only FFmpeg-level escaping needed).
  const selectExpr = `gt(scene\\,${sceneThreshold})+not(mod(n\\,${frameInterval}))`;

  const vf = [
    startSec > 0
      ? `trim=start=${startSec}${endSec !== undefined ? `:end=${endSec}` : ""},setpts=PTS-STARTPTS`
      : endSec !== undefined
        ? `trim=end=${endSec},setpts=PTS-STARTPTS`
        : null,
    `select='${selectExpr}'`,
    `scale=${thumbWidth}:-2`,
    `fps=fps=1/1`, // slow down to 1fps so we can use -vsync 0
  ]
    .filter(Boolean)
    .join(",");

  const tmpDir = await mkdtemp(join(homedir(), ".cache", "media-understanding-tmp-"));

  try {
    const ffmpeg = ffmpegPath();

    // Extract selected frames to temp dir as JPEG files
    const args = [
      "-i",
      abs,
      "-vf",
      vf,
      "-vsync",
      "0",
      "-frame_pts",
      "1",
      "-q:v",
      "3",
      join(tmpDir, "frame_%06d.jpg"),
    ];

    await execFile(ffmpeg, args, { maxBuffer: 64 * 1024 * 1024 });

    // Read extracted frames
    const files = (await readdir(tmpDir)).filter((f) => f.endsWith(".jpg")).sort();

    if (files.length === 0) {
      // No frames selected — return empty array rather than error
      return [];
    }

    // Group frames into batches of framesPerGrid, capped at maxGrids
    const batches: string[][] = [];
    for (let i = 0; i < files.length && batches.length < maxGrids; i += framesPerGrid) {
      batches.push(files.slice(i, i + framesPerGrid).map((f) => join(tmpDir, f)));
    }

    // Compose each batch into a grid using sharp
    const grids: Buffer[] = [];
    for (const batch of batches) {
      const grid = await composeGrid(batch, cols, thumbWidth);
      grids.push(await compressForLLM(grid));
    }

    return grids;
  } catch (err) {
    if (err instanceof MediaError) throw err;
    throw new MediaError("GRID_FAILED", `Frame grid extraction failed: ${abs}`, err);
  } finally {
    await rm(tmpDir, { recursive: true, force: true });
  }
}

/**
 * Compose a list of JPEG file paths into a single grid image.
 * Arranges tiles left-to-right, top-to-bottom.
 */
async function composeGrid(
  framePaths: string[],
  cols: number,
  thumbWidth: number,
): Promise<Buffer> {
  if (framePaths.length === 0) {
    throw new MediaError("GRID_FAILED", "No frames to compose into grid");
  }

  // Read and normalize all frames to the same dimensions
  const thumbs = await Promise.all(
    framePaths.map((p) =>
      sharp(p)
        .resize({ width: thumbWidth, withoutEnlargement: true })
        .jpeg({ quality: 80 })
        .toBuffer(),
    ),
  );

  // Get dimensions from first frame
  const firstMeta = await sharp(thumbs[0]).metadata();
  const tileW = firstMeta.width ?? thumbWidth;
  const tileH = firstMeta.height ?? Math.round((thumbWidth * 9) / 16);

  const rows = Math.ceil(framePaths.length / cols);
  const gridW = cols * tileW;
  const gridH = rows * tileH;

  // Build composite instructions
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
export async function extractFrame(filePath: string, timestampSec: number): Promise<Buffer> {
  assertFfmpeg();
  const abs = await assertFile(filePath);

  if (timestampSec < 0) {
    throw new MediaError("FRAME_FAILED", `Timestamp must be >= 0, got: ${timestampSec}`);
  }

  try {
    const ffmpeg = ffmpegPath();
    const args = [
      "-ss",
      String(timestampSec),
      "-i",
      abs,
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
      throw new MediaError("FRAME_FAILED", `No frame data returned at ${timestampSec}s in: ${abs}`);
    }

    return compressForLLM(stdout as unknown as Buffer);
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
  const abs = await assertFile(filePath);
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
  if (info.type === "video") {
    grids = await extractFrameGrid(abs, opts);
  }

  return { info, segments, transcript, grids };
}
