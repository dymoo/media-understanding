#!/usr/bin/env node
/**
 * Postinstall: pre-warm the default Whisper model so first transcription
 * is instant. Non-fatal — if the download fails the model will be fetched
 * on first use instead.
 *
 * Skip with: SKIP_MODEL_DOWNLOAD=1 (set this in CI to avoid large downloads)
 */

import { homedir } from "node:os";
import { join } from "node:path";
import { WhisperDownloader } from "node-av/api";

if (process.env["SKIP_MODEL_DOWNLOAD"] === "1") {
  process.exit(0);
}

const DEFAULT_MODEL = "base.en-q5_1";
const model = process.env["MEDIA_UNDERSTANDING_MODEL"] ?? DEFAULT_MODEL;

function resolveModelDir() {
  const xdg = process.env["XDG_CACHE_HOME"];
  if (xdg) return join(xdg, "media-understanding", "models");

  if (process.platform === "win32") {
    const localAppData = process.env["LOCALAPPDATA"];
    if (localAppData) return join(localAppData, "media-understanding", "models");
  }

  return join(homedir(), ".cache", "media-understanding", "models");
}

if (!WhisperDownloader.isValidModel(model)) {
  console.warn(`[media-understanding] Unknown model "${model}", skipping pre-warm.`);
  process.exit(0);
}

const modelDir = resolveModelDir();

if (WhisperDownloader.modelExists(model, modelDir)) {
  console.log(`[media-understanding] Whisper model "${model}" already cached.`);
  process.exit(0);
}

console.log(`[media-understanding] Downloading Whisper model "${model}" (~57 MB)…`);

try {
  await WhisperDownloader.downloadModel({ model, outputPath: modelDir });
  console.log(`[media-understanding] Model "${model}" ready at ${modelDir}`);
} catch (err) {
  console.warn(
    `[media-understanding] Model pre-warm failed (non-fatal): ${err.message}\n` +
      `  The model will be downloaded automatically on first transcription.`,
  );
}
