/**
 * YouTube / yt-dlp integration layer.
 *
 * Uses ytdlp-nodejs as a wrapper around a **system-installed** yt-dlp binary.
 * Downloads are cached in a temp directory keyed by URL hash so repeated calls
 * are instant.  Subtitle files are parsed into Segment[] for compatibility with
 * the existing transcript pipeline.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { execSync } from "node:child_process";
import { YtDlp } from "ytdlp-nodejs";
import type { VideoInfo, VideoThumbnail, DownloadResult } from "ytdlp-nodejs";

import { ffmpegPath } from "node-av/ffmpeg";

import type { Segment } from "./types.js";
import { MediaError } from "./types.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CACHE_ROOT = join(tmpdir(), "media-understanding-ytdlp");

// ---------------------------------------------------------------------------
// URL detection
// ---------------------------------------------------------------------------

/** Returns true when the input looks like an HTTP(S) URL rather than a path. */
export function isUrl(input: string): boolean {
  return /^https?:\/\//i.test(input.trim());
}

// ---------------------------------------------------------------------------
// yt-dlp binary resolution
// ---------------------------------------------------------------------------

let resolvedBinaryPath: string | undefined;

/** Locate the system yt-dlp binary.  Throws YT_DLP_NOT_FOUND on failure. */
export function ensureYtDlp(): string {
  if (resolvedBinaryPath) return resolvedBinaryPath;

  try {
    const result = execSync("which yt-dlp", { encoding: "utf8" }).trim();
    if (result) {
      resolvedBinaryPath = result;
      return result;
    }
  } catch {
    // fall through
  }

  throw new MediaError(
    "YT_DLP_NOT_FOUND",
    "yt-dlp is not installed on this system. Install it (https://github.com/yt-dlp/yt-dlp#installation) and ensure it is on $PATH.",
  );
}

// ---------------------------------------------------------------------------
// Singleton YtDlp instance (lazy)
// ---------------------------------------------------------------------------

let instance: YtDlp | undefined;

function getYtDlp(): YtDlp {
  if (instance) return instance;

  const binaryPath = ensureYtDlp();

  // Try to hand the node-av bundled ffmpeg to yt-dlp so it can mux formats.
  let ffmpeg: string | undefined;
  try {
    ffmpeg = ffmpegPath();
  } catch {
    // node-av ffmpeg not available — yt-dlp will use system ffmpeg or skip muxing
  }

  instance = new YtDlp({
    binaryPath,
    ...(ffmpeg ? { ffmpegPath: ffmpeg } : {}),
  });

  return instance;
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

function urlHash(url: string): string {
  return createHash("sha256").update(url.trim()).digest("hex").slice(0, 16);
}

function cacheDir(url: string): string {
  const dir = join(CACHE_ROOT, urlHash(url));
  mkdirSync(dir, { recursive: true });
  return dir;
}

/**
 * Find any file in the cache dir matching a glob-like prefix.
 * yt-dlp names files unpredictably, so we scan the directory.
 */
function findFileInDir(dir: string, prefix: string, extensions?: string[]): string | null {
  try {
    const entries = readdirSync(dir);
    for (const entry of entries) {
      if (!entry.startsWith(prefix)) continue;
      if (extensions) {
        const ext = entry.slice(entry.lastIndexOf(".")).toLowerCase();
        if (extensions.some((e) => ext === e || ext === `.${e}`)) return join(dir, entry);
      } else {
        return join(dir, entry);
      }
    }
  } catch {
    // directory doesn't exist
  }
  return null;
}

// ---------------------------------------------------------------------------
// Public API — Video info
// ---------------------------------------------------------------------------

export interface YtDlpVideoInfo {
  id: string;
  title: string;
  duration: number;
  description: string;
  uploader: string;
  viewCount: number;
  uploadDate: string;
  thumbnailUrl: string;
  hasSubtitles: boolean;
  subtitleLanguages: string[];
  url: string;
}

export async function getVideoInfo(url: string): Promise<YtDlpVideoInfo> {
  const ytdlp = getYtDlp();
  try {
    const info = (await ytdlp.getInfoAsync(url)) as VideoInfo;
    const subtitles = await ytdlp.getSubtitles(url).catch(() => [] as { language: string }[]);
    return {
      id: info.id,
      title: info.title,
      duration: info.duration ?? 0,
      description: info.description ?? "",
      uploader: info.uploader ?? "",
      viewCount: info.view_count ?? 0,
      uploadDate: info.upload_date ?? "",
      thumbnailUrl: info.thumbnail ?? "",
      hasSubtitles: subtitles.length > 0,
      subtitleLanguages: subtitles.map((s) => s.language),
      url,
    };
  } catch (err) {
    throw new MediaError(
      "YT_DLP_FAILED",
      `Failed to fetch video info for ${url}: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  }
}

// ---------------------------------------------------------------------------
// Public API — Subtitle download + parsing
// ---------------------------------------------------------------------------

/**
 * Download subtitles for a URL.  Tries manual captions first, then auto-generated.
 * Returns the path to the downloaded subtitle file, or null if none available.
 */
export async function downloadSubtitles(url: string): Promise<string | null> {
  const dir = cacheDir(url);

  // Check cache
  const cached = findFileInDir(dir, "subs", [".srt", ".vtt", ".ass", ".json3"]);
  if (cached) return cached;

  const ytdlp = getYtDlp();

  try {
    // Use execAsync with subtitle flags — the fluent builder doesn't expose all subtitle options easily
    await ytdlp.execAsync(url, {
      writeSubs: true,
      writeAutoSubs: true,
      subLangs: ["all"],
      subFormat: "srt/vtt/best",
      skipDownload: true,
      output: join(dir, "subs.%(ext)s"),
    });
  } catch {
    // yt-dlp may exit non-zero when no subs exist
  }

  // Find whatever subtitle file was written
  return findFileInDir(dir, "subs", [".srt", ".vtt", ".ass", ".json3"]);
}

/**
 * Parse an SRT or VTT subtitle file into our Segment[] format so it plugs
 * directly into the existing transcript pipeline.
 */
export function parseSubtitlesToSegments(subtitlePath: string): Segment[] {
  const content = readFileSync(subtitlePath, "utf8");
  const ext = subtitlePath.slice(subtitlePath.lastIndexOf(".")).toLowerCase();

  if (ext === ".srt") return parseSrt(content);
  if (ext === ".vtt") return parseVtt(content);

  // For other formats, return the full text as a single segment
  return [{ start: 0, end: 0, text: content }];
}

function parseSrt(content: string): Segment[] {
  const segments: Segment[] = [];
  // SRT blocks are separated by blank lines
  const blocks = content.split(/\n\s*\n/).filter((b) => b.trim());

  for (const block of blocks) {
    const lines = block.trim().split("\n");
    // Find the timestamp line (contains -->)
    const tsLineIdx = lines.findIndex((l) => l.includes("-->"));
    if (tsLineIdx < 0) continue;

    const tsLine = lines[tsLineIdx]!;
    const match = tsLine.match(
      /(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})/,
    );
    if (!match) continue;

    const start = timestampToMs(match[1]!, match[2]!, match[3]!, match[4]!);
    const end = timestampToMs(match[5]!, match[6]!, match[7]!, match[8]!);
    const text = lines
      .slice(tsLineIdx + 1)
      .join(" ")
      .replace(/<[^>]*>/g, "") // strip HTML tags
      .trim();

    if (text) segments.push({ start, end, text });
  }

  return segments;
}

function parseVtt(content: string): Segment[] {
  // VTT is similar to SRT but with WEBVTT header and slightly different format
  // Strip the WEBVTT header and any style blocks
  const stripped = content
    .replace(/^WEBVTT.*?\n\n/s, "")
    .replace(/^STYLE\n[\s\S]*?\n\n/gm, "")
    .replace(/^NOTE\n[\s\S]*?\n\n/gm, "");

  return parseSrt(stripped);
}

function timestampToMs(h: string, m: string, s: string, ms: string): number {
  return parseInt(h) * 3600000 + parseInt(m) * 60000 + parseInt(s) * 1000 + parseInt(ms);
}

// ---------------------------------------------------------------------------
// Public API — Thumbnail download
// ---------------------------------------------------------------------------

/**
 * Download the video thumbnail. Returns the local file path.
 */
export async function downloadThumbnail(url: string): Promise<string> {
  const dir = cacheDir(url);

  // Check cache
  const cached = findFileInDir(dir, "thumb", [".jpg", ".png", ".webp"]);
  if (cached) return cached;

  const ytdlp = getYtDlp();

  try {
    await ytdlp.execAsync(url, {
      writeThumbnail: true,
      skipDownload: true,
      output: join(dir, "thumb.%(ext)s"),
    });
  } catch (err) {
    throw new MediaError(
      "YT_DLP_FAILED",
      `Failed to download thumbnail for ${url}: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  }

  const result = findFileInDir(dir, "thumb", [".jpg", ".png", ".webp"]);
  if (!result) {
    throw new MediaError("YT_DLP_FAILED", `No thumbnail was downloaded for ${url}`);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API — Video download (lowest quality for frame analysis)
// ---------------------------------------------------------------------------

/**
 * Download the video at lowest quality (for frame analysis speed).
 * Returns the local file path.
 */
export async function downloadVideo(url: string): Promise<string> {
  const dir = cacheDir(url);

  // Check cache
  const cached = findFileInDir(dir, "video", [".mp4", ".webm", ".mkv"]);
  if (cached) return cached;

  const ytdlp = getYtDlp();

  try {
    const result: DownloadResult = await ytdlp.downloadAsync(url, {
      format: { filter: "mergevideo", quality: "lowest" } as never,
      output: join(dir, "video.%(ext)s"),
    });

    const filePath = result.filePaths?.[0];
    if (filePath && existsSync(filePath)) return filePath;
  } catch {
    // Fallback: try with worst format string directly
  }

  // Fallback: use execAsync with explicit worst format
  try {
    await ytdlp.execAsync(url, {
      format: "worst[ext=mp4]/worst",
      output: join(dir, "video.%(ext)s"),
    });
  } catch (err) {
    throw new MediaError(
      "YT_DLP_FAILED",
      `Failed to download video for ${url}: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  }

  const result = findFileInDir(dir, "video", [".mp4", ".webm", ".mkv"]);
  if (!result) {
    throw new MediaError("YT_DLP_FAILED", `No video file was downloaded for ${url}`);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API — Audio download (for Whisper fallback)
// ---------------------------------------------------------------------------

/**
 * Download audio only.  Returns the local file path.
 */
export async function downloadAudio(url: string): Promise<string> {
  const dir = cacheDir(url);

  // Check cache
  const cached = findFileInDir(dir, "audio", [".m4a", ".mp3", ".wav", ".opus", ".ogg"]);
  if (cached) return cached;

  const ytdlp = getYtDlp();

  try {
    const result: DownloadResult = await ytdlp.downloadAudio(url, "m4a");

    const filePath = result.filePaths?.[0];
    if (filePath && existsSync(filePath)) return filePath;
  } catch {
    // fallback below
  }

  // Fallback with explicit options
  try {
    await ytdlp.execAsync(url, {
      extractAudio: true,
      audioFormat: "m4a",
      output: join(dir, "audio.%(ext)s"),
    });
  } catch (err) {
    throw new MediaError(
      "YT_DLP_FAILED",
      `Failed to download audio for ${url}: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  }

  const result = findFileInDir(dir, "audio", [".m4a", ".mp3", ".wav", ".opus", ".ogg"]);
  if (!result) {
    throw new MediaError("YT_DLP_FAILED", `No audio file was downloaded for ${url}`);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API — Resolve URL to local video path (for transparent integration)
// ---------------------------------------------------------------------------

// In-memory map to avoid repeated fs.stat on cache hits within the same process.
const resolvedPaths = new Map<string, string>();

/**
 * Resolve a URL to a local file path, downloading if necessary.
 * Downloads lowest quality video+audio for versatility (frame analysis + Whisper).
 * Cached across calls.
 */
export async function resolveUrlToLocalPath(url: string): Promise<string> {
  const cached = resolvedPaths.get(url);
  if (cached && existsSync(cached)) return cached;

  const localPath = await downloadVideo(url);
  resolvedPaths.set(url, localPath);
  return localPath;
}

/**
 * Resolve a URL to a local audio-only path for transcript use.
 * Falls back to full video download if audio-only fails.
 */
export async function resolveUrlToAudioPath(url: string): Promise<string> {
  try {
    return await downloadAudio(url);
  } catch {
    // If audio extraction fails, fall back to full video
    return await resolveUrlToLocalPath(url);
  }
}

// ---------------------------------------------------------------------------
// Public API — Available subtitles listing
// ---------------------------------------------------------------------------

export async function listSubtitles(
  url: string,
): Promise<{ language: string; languages: string[]; ext: string; autoCaption: boolean }[]> {
  const ytdlp = getYtDlp();
  try {
    return await ytdlp.getSubtitles(url);
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Public API — Get thumbnails info
// ---------------------------------------------------------------------------

export async function getThumbnails(url: string): Promise<VideoThumbnail[]> {
  const ytdlp = getYtDlp();
  try {
    return await ytdlp.getThumbnailsAsync(url);
  } catch {
    return [];
  }
}
