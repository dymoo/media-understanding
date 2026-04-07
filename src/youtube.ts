/**
 * YouTube / yt-dlp integration layer.
 *
 * Calls a **system-installed** yt-dlp binary directly via child_process.
 * We intentionally do NOT bundle or download yt-dlp (copyright concerns).
 * If yt-dlp is not on $PATH, all URL features are gracefully unavailable.
 *
 * Downloads are cached in a temp directory keyed by URL hash so repeated
 * calls are instant. Subtitle files are parsed into Segment[] for
 * compatibility with the existing transcript pipeline.
 */

import { execFile as execFileCb, execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { promisify } from "node:util";

import { ffmpegPath } from "node-av/ffmpeg";
import { parse as parseSubtitles } from "@plussub/srt-vtt-parser";

import type { Segment } from "./types.js";
import { MediaError } from "./types.js";

const execFile = promisify(execFileCb);

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
let ytDlpChecked = false;

/** Locate the system yt-dlp binary. Throws YT_DLP_NOT_FOUND on failure. */
export function ensureYtDlp(): string {
  if (resolvedBinaryPath) return resolvedBinaryPath;

  try {
    const result = execSync("which yt-dlp", { encoding: "utf8" }).trim();
    if (result) {
      resolvedBinaryPath = result;
      ytDlpChecked = true;
      return result;
    }
  } catch {
    // fall through
  }

  ytDlpChecked = true;
  throw new MediaError(
    "YT_DLP_NOT_FOUND",
    "yt-dlp is not installed on this system. Install it (https://github.com/yt-dlp/yt-dlp#installation) and ensure it is on $PATH.",
  );
}

/**
 * Synchronous check whether yt-dlp is available on the system.
 * Result is cached at module level after first call.
 */
export function hasYtDlp(): boolean {
  if (ytDlpChecked) return resolvedBinaryPath !== undefined;
  try {
    ensureYtDlp();
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// yt-dlp invocation helpers
// ---------------------------------------------------------------------------

/** Common args prepended to every yt-dlp call. */
function baseArgs(): string[] {
  const args: string[] = ["--no-check-certificates"];

  // Hand node-av's bundled ffmpeg to yt-dlp for muxing
  try {
    args.push("--ffmpeg-location", ffmpegPath());
  } catch {
    // node-av ffmpeg not available — yt-dlp will use system ffmpeg
  }

  return args;
}

/**
 * Run yt-dlp with the given arguments. Returns { stdout, stderr }.
 * Throws MediaError on non-zero exit.
 */
async function runYtDlp(
  args: string[],
  context: string,
): Promise<{ stdout: string; stderr: string }> {
  const bin = ensureYtDlp();
  try {
    return await execFile(bin, [...baseArgs(), ...args], {
      maxBuffer: 50 * 1024 * 1024, // 50 MB — --dump-json can be large
      timeout: 5 * 60 * 1000, // 5 min
    });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new MediaError("YT_DLP_FAILED", `${context}: ${msg}`, err);
  }
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

export function urlHash(url: string): string {
  return createHash("sha256").update(url.trim()).digest("hex").slice(0, 16);
}

function cacheDir(url: string): string {
  const dir = join(CACHE_ROOT, urlHash(url));
  mkdirSync(dir, { recursive: true });
  return dir;
}

/**
 * Find any file in the cache dir matching a prefix + extension set.
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
  const { stdout } = await runYtDlp(
    ["--dump-json", "--no-download", url],
    `Failed to fetch video info for ${url}`,
  );

  const info = JSON.parse(stdout) as Record<string, unknown>;

  /** Safely coerce an unknown value to string. */
  const str = (v: unknown): string =>
    typeof v === "string" ? v : v == null ? "" : JSON.stringify(v);
  const num = (v: unknown): number => (typeof v === "number" ? v : Number(v) || 0);

  // Extract subtitle languages from the subtitles/automatic_captions objects
  const subtitleLangs = new Set<string>();
  for (const key of ["subtitles", "automatic_captions"] as const) {
    const obj = info[key];
    if (obj && typeof obj === "object") {
      for (const lang of Object.keys(obj as Record<string, unknown>)) {
        subtitleLangs.add(lang);
      }
    }
  }

  return {
    id: str(info["id"]),
    title: str(info["title"]),
    duration: num(info["duration"]),
    description: str(info["description"]),
    uploader: str(info["uploader"]) || str(info["channel"]),
    viewCount: num(info["view_count"]),
    uploadDate: str(info["upload_date"]),
    thumbnailUrl: str(info["thumbnail"]),
    hasSubtitles: subtitleLangs.size > 0,
    subtitleLanguages: [...subtitleLangs],
    url,
  };
}

// ---------------------------------------------------------------------------
// Public API — Subtitle download + parsing
// ---------------------------------------------------------------------------

/**
 * Download subtitles for a URL. Tries manual captions first, then auto-generated.
 * Returns the path to the downloaded subtitle file, or null if none available.
 */
export async function downloadSubtitles(url: string): Promise<string | null> {
  const dir = cacheDir(url);

  // Check cache
  const cached = findFileInDir(dir, "subs", [".srt", ".vtt", ".ass", ".json3"]);
  if (cached) return cached;

  try {
    await runYtDlp(
      [
        "--write-subs",
        "--write-auto-subs",
        "--sub-lang",
        "all",
        "--sub-format",
        "srt/vtt/best",
        "--skip-download",
        "-o",
        join(dir, "subs.%(ext)s"),
        url,
      ],
      `Subtitle download for ${url}`,
    );
  } catch {
    // yt-dlp may exit non-zero when no subs exist — that's fine
  }

  return findFileInDir(dir, "subs", [".srt", ".vtt", ".ass", ".json3"]);
}

/**
 * Parse an SRT or VTT subtitle file into our Segment[] format so it plugs
 * directly into the existing transcript pipeline.
 *
 * Uses @plussub/srt-vtt-parser for robust SRT/VTT handling (STYLE/NOTE
 * block stripping, timestamp parsing, multiline cues). HTML tags are
 * stripped post-parse and newlines are joined with spaces.
 */
export function parseSubtitlesToSegments(subtitlePath: string): Segment[] {
  const content = readFileSync(subtitlePath, "utf8");
  const ext = subtitlePath.slice(subtitlePath.lastIndexOf(".")).toLowerCase();

  // For formats the library doesn't handle, return full text as single segment
  if (ext !== ".srt" && ext !== ".vtt") {
    return [{ start: 0, end: 0, text: content }];
  }

  // Normalize SRT period-separator timestamps to commas: some tools (including
  // certain yt-dlp outputs) write "00:00:01.000 --> 00:00:03.500" instead of
  // the standard "00:00:01,000 --> 00:00:03,500". The library expects commas.
  const normalized =
    ext === ".srt" ? content.replace(/(\d{2}:\d{2}:\d{2})\.(\d{3})/g, "$1,$2") : content;

  const parsed = parseSubtitles(normalized) as {
    entries: Array<{ from: number; to: number; text: string }>;
  };
  const segments: Segment[] = [];
  for (const entry of parsed.entries) {
    const text = entry.text
      .replace(/<[^>]*>/g, "")
      .replace(/\n/g, " ")
      .trim();
    if (text) segments.push({ start: entry.from, end: entry.to, text });
  }
  return segments;
}

// ---------------------------------------------------------------------------
// Public API — Thumbnail download
// ---------------------------------------------------------------------------

/** Download the video thumbnail. Returns the local file path. */
export async function downloadThumbnail(url: string): Promise<string> {
  const dir = cacheDir(url);

  const cached = findFileInDir(dir, "thumb", [".jpg", ".png", ".webp"]);
  if (cached) return cached;

  await runYtDlp(
    ["--write-thumbnail", "--skip-download", "-o", join(dir, "thumb.%(ext)s"), url],
    `Failed to download thumbnail for ${url}`,
  );

  const result = findFileInDir(dir, "thumb", [".jpg", ".png", ".webp"]);
  if (!result) {
    throw new MediaError("YT_DLP_FAILED", `No thumbnail was downloaded for ${url}`);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API — Video download (lowest quality for frame analysis)
// ---------------------------------------------------------------------------

/** Download video at lowest quality (for frame analysis speed). Returns local path. */
export async function downloadVideo(url: string): Promise<string> {
  const dir = cacheDir(url);

  const cached = findFileInDir(dir, "video", [".mp4", ".webm", ".mkv"]);
  if (cached) return cached;

  await runYtDlp(
    ["-f", "worst[ext=mp4]/worst", "-o", join(dir, "video.%(ext)s"), url],
    `Failed to download video for ${url}`,
  );

  const result = findFileInDir(dir, "video", [".mp4", ".webm", ".mkv"]);
  if (!result) {
    throw new MediaError("YT_DLP_FAILED", `No video file was downloaded for ${url}`);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API — Audio download (for ASR fallback)
// ---------------------------------------------------------------------------

/** Download audio only. Returns the local file path. */
export async function downloadAudio(url: string): Promise<string> {
  const dir = cacheDir(url);

  const cached = findFileInDir(dir, "audio", [".m4a", ".mp3", ".wav", ".opus", ".ogg"]);
  if (cached) return cached;

  await runYtDlp(
    ["-x", "--audio-format", "m4a", "-o", join(dir, "audio.%(ext)s"), url],
    `Failed to download audio for ${url}`,
  );

  const result = findFileInDir(dir, "audio", [".m4a", ".mp3", ".wav", ".opus", ".ogg"]);
  if (!result) {
    throw new MediaError("YT_DLP_FAILED", `No audio file was downloaded for ${url}`);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API — Resolve URL to local path (transparent integration)
// ---------------------------------------------------------------------------

const resolvedPaths = new Map<string, string>();

/**
 * Resolve a URL to a local file path, downloading if necessary.
 * Downloads lowest quality video+audio for versatility.
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
    return await resolveUrlToLocalPath(url);
  }
}
