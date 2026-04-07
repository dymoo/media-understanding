/**
 * Parakeet ONNX model download manager.
 *
 * Downloads INT8-quantized Parakeet TDT 0.6B v3 ONNX models from HuggingFace.
 * Features:
 *   - Progress tracking logged to stderr
 *   - Atomic writes (download to .tmp, rename on completion)
 *   - SHA-256 verification via HF LFS oid
 *   - Retry with exponential backoff (3 attempts)
 *   - Concurrent downloads (encoder + decoder + vocab in parallel)
 *   - Lazy download on first use
 */

import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, rename, stat, readFile, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx";
const HF_REVISION = "main";

/** Model files to download with their expected SHA-256 (from HF LFS oid). */
const MODEL_FILES: readonly ModelFileSpec[] = [
  {
    name: "encoder-model.int8.onnx",
    size: 652_183_999,
    sha256: "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
  },
  {
    name: "decoder_joint-model.int8.onnx",
    size: 18_202_004,
    sha256: "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
  },
  {
    name: "nemo128.onnx",
    size: 139_764,
    sha256: "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f",
  },
  {
    name: "vocab.txt",
    size: 93_939,
    sha256: undefined, // small file, no LFS
  },
] as const;

const MAX_RETRIES = 3;
const INITIAL_BACKOFF_MS = 1000;
const PROGRESS_INTERVAL_MS = 5000;

interface ModelFileSpec {
  readonly name: string;
  readonly size: number;
  readonly sha256: string | undefined;
}

// ---------------------------------------------------------------------------
// Cache directory
// ---------------------------------------------------------------------------

/** Return the directory where Parakeet ONNX models are cached. */
export function resolveModelDir(): string {
  const xdg = process.env["XDG_CACHE_HOME"];
  if (xdg) return join(xdg, "media-understanding", "models", "parakeet-tdt-0.6b-v3-int8");

  if (process.platform === "win32") {
    const localAppData = process.env["LOCALAPPDATA"];
    if (localAppData)
      return join(localAppData, "media-understanding", "models", "parakeet-tdt-0.6b-v3-int8");
  }

  return join(homedir(), ".cache", "media-understanding", "models", "parakeet-tdt-0.6b-v3-int8");
}

/** Return the full path to a specific model file. */
export function modelFilePath(fileName: string): string {
  return join(resolveModelDir(), fileName);
}

// ---------------------------------------------------------------------------
// Download helpers
// ---------------------------------------------------------------------------

function hfUrl(fileName: string): string {
  const encodedRepo = HF_REPO.split("/").map(encodeURIComponent).join("/");
  const encodedFile = encodeURIComponent(fileName);
  return `https://huggingface.co/${encodedRepo}/resolve/${HF_REVISION}/${encodedFile}`;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(1)} KB`;
  return `${bytes} B`;
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Verify SHA-256 of a file. Returns true if hash matches or no hash is specified.
 */
async function verifySha256(filePath: string, expected: string | undefined): Promise<boolean> {
  if (!expected) return true;
  const data = await readFile(filePath);
  const actual = createHash("sha256").update(data).digest("hex");
  return actual === expected;
}

/**
 * Download a single file with progress tracking, retries, and atomic writes.
 */
async function downloadFile(spec: ModelFileSpec, destDir: string): Promise<void> {
  const destPath = join(destDir, spec.name);
  const tmpPath = destPath + ".tmp";
  const url = hfUrl(spec.name);

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = await fetch(url, { redirect: "follow" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }
      if (!response.body) {
        throw new Error("Response has no body");
      }

      const contentLength = parseInt(response.headers.get("content-length") ?? "0", 10);
      const total = contentLength || spec.size;

      let downloaded = 0;
      let lastProgressTime = Date.now();

      // Create a transform that tracks progress
      const progressStream = new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, controller) {
          downloaded += chunk.byteLength;
          controller.enqueue(chunk);

          const now = Date.now();
          if (now - lastProgressTime >= PROGRESS_INTERVAL_MS || downloaded === total) {
            const pct = total > 0 ? ((downloaded / total) * 100).toFixed(1) : "?";
            process.stderr.write(
              `[download] ${spec.name}: ${formatBytes(downloaded)}/${formatBytes(total)} (${pct}%)\n`,
            );
            lastProgressTime = now;
          }
        },
      });

      const tracked = response.body.pipeThrough(progressStream);

      // Convert web ReadableStream to Node.js Readable for fs pipeline
      const nodeStream = Readable.fromWeb(tracked);
      const fileStream = createWriteStream(tmpPath);
      await pipeline(nodeStream, fileStream);

      // Verify SHA-256 if available
      if (spec.sha256) {
        const valid = await verifySha256(tmpPath, spec.sha256);
        if (!valid) {
          await unlink(tmpPath).catch(() => {});
          throw new Error(`SHA-256 mismatch for ${spec.name}`);
        }
      }

      // Atomic rename
      await rename(tmpPath, destPath);
      process.stderr.write(`[download] ${spec.name}: complete\n`);
      return;
    } catch (err) {
      // Clean up tmp file on failure
      await unlink(tmpPath).catch(() => {});

      if (attempt < MAX_RETRIES) {
        const backoff = INITIAL_BACKOFF_MS * 2 ** (attempt - 1);
        process.stderr.write(
          `[download] ${spec.name}: attempt ${attempt}/${MAX_RETRIES} failed (${(err as Error).message}), retrying in ${backoff}ms\n`,
        );
        await new Promise((resolve) => setTimeout(resolve, backoff));
      } else {
        throw new Error(
          `Failed to download ${spec.name} after ${MAX_RETRIES} attempts: ${(err as Error).message}`,
          { cause: err },
        );
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Check if all model files are present in the cache. */
export async function isModelCached(): Promise<boolean> {
  const dir = resolveModelDir();
  for (const spec of MODEL_FILES) {
    if (!(await fileExists(join(dir, spec.name)))) return false;
  }
  return true;
}

/**
 * Ensure all model files are downloaded and cached.
 * Downloads missing files in parallel. No-op if all files are present.
 *
 * @throws If any download fails after retries.
 */
export async function ensureModel(): Promise<void> {
  const dir = resolveModelDir();
  await mkdir(dir, { recursive: true });

  // Check which files need downloading
  const needed: ModelFileSpec[] = [];
  for (const spec of MODEL_FILES) {
    if (!(await fileExists(join(dir, spec.name)))) {
      needed.push(spec);
    }
  }

  if (needed.length === 0) return;

  const totalSize = needed.reduce((sum, s) => sum + s.size, 0);
  process.stderr.write(
    `[media-understanding] Downloading Parakeet ASR model (~${formatBytes(totalSize)})...\n`,
  );

  // Download all needed files in parallel
  await Promise.all(needed.map((spec) => downloadFile(spec, dir)));

  process.stderr.write(`[media-understanding] Model ready at ${dir}\n`);
}

/** Return paths to all model files. Call after ensureModel(). */
export function getModelPaths(): {
  encoder: string;
  decoder: string;
  preprocessor: string;
  vocab: string;
} {
  const dir = resolveModelDir();
  return {
    encoder: join(dir, "encoder-model.int8.onnx"),
    decoder: join(dir, "decoder_joint-model.int8.onnx"),
    preprocessor: join(dir, "nemo128.onnx"),
    vocab: join(dir, "vocab.txt"),
  };
}
