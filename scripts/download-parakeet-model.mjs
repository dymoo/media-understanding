#!/usr/bin/env node
/**
 * Optional prewarm: download the Parakeet TDT 0.6B v3 ONNX model (~670 MB)
 * so that first transcription is instant.
 *
 * Not run by default postinstall (670 MB is too large). Instead:
 *   - MEDIA_UNDERSTANDING_PREWARM=1 in postinstall triggers it
 *   - Or run manually: node scripts/download-parakeet-model.mjs
 *
 * Skip with: SKIP_MODEL_DOWNLOAD=1 (set in CI to avoid large downloads)
 */

if (process.env["SKIP_MODEL_DOWNLOAD"] === "1") {
  process.exit(0);
}

// When called from postinstall (not as a standalone prewarm), skip unless opted in.
// The --prewarm flag or MEDIA_UNDERSTANDING_PREWARM=1 env var enables it.
const isPrewarm =
  process.env["MEDIA_UNDERSTANDING_PREWARM"] === "1" || process.argv.includes("--prewarm");

if (!isPrewarm) {
  // Silent exit in postinstall — model downloads lazily on first transcription
  process.exit(0);
}

try {
  const { ensureModel, resolveModelDir } = await import("../dist/model-manager.js");
  const dir = resolveModelDir();

  console.log(`[media-understanding] Downloading Parakeet TDT 0.6B v3 model (~670 MB)...`);
  await ensureModel();
  console.log(`[media-understanding] Model ready at ${dir}`);
} catch (err) {
  console.warn(
    `[media-understanding] Model pre-warm failed (non-fatal): ${err.message}\n` +
      `  The model will be downloaded automatically on first transcription.`,
  );
}
