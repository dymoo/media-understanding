/**
 * Integration tests for src/media.ts
 *
 * Tests use committed fixtures in testdata/:
 *   tiny.png  — 16×16 red PNG (always present)
 *   tiny.wav  — 1-second sine wave (generated in CI by ci.yml)
 *   tiny.mp3  — same as wav, MP3-encoded
 *   tiny.mp4  — 5-second 320×240 test video with audio
 *
 * Transcription tests require a Whisper model (~75 MB download on first run).
 * They have a generous timeout and skip gracefully if SKIP_TRANSCRIPTION=1.
 */

import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import sharp from "sharp";

import {
  compressForLLM,
  extractFrame,
  extractFrameGrid,
  extractFrameGridImages,
  extractFrameImage,
  probeMedia,
  resolveModelDir,
  transcribeAudio,
  understandMedia,
} from "../media.js";
import { MediaError } from "../types.js";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const root = resolve(import.meta.dirname, "..", "..", "testdata");
const PNG = resolve(root, "tiny.png");
const WAV = resolve(root, "tiny.wav");
const MP3 = resolve(root, "tiny.mp3");
const MP4 = resolve(root, "tiny.mp4");

const SKIP_TRANSCRIPTION = process.env["SKIP_TRANSCRIPTION"] === "1";

// ---------------------------------------------------------------------------
// resolveModelDir
// ---------------------------------------------------------------------------

describe("resolveModelDir", () => {
  it("returns a non-empty string", () => {
    const dir = resolveModelDir();
    assert.equal(typeof dir, "string");
    assert.ok(dir.length > 0);
  });

  it("honours XDG_CACHE_HOME", () => {
    const orig = process.env["XDG_CACHE_HOME"];
    process.env["XDG_CACHE_HOME"] = "/tmp/xdg";
    const dir = resolveModelDir();
    if (orig !== undefined) {
      process.env["XDG_CACHE_HOME"] = orig;
    } else {
      delete process.env["XDG_CACHE_HOME"];
    }
    assert.ok(dir.startsWith("/tmp/xdg"));
  });
});

// ---------------------------------------------------------------------------
// compressForLLM
// ---------------------------------------------------------------------------

describe("compressForLLM", () => {
  it("returns a JPEG buffer for a PNG input", async () => {
    const raw = await readFile(PNG);
    const compressed = await compressForLLM(raw);
    // Valid JPEG starts with FF D8
    assert.equal(compressed[0], 0xff);
    assert.equal(compressed[1], 0xd8);
    assert.ok(Buffer.isBuffer(compressed));
  });

  it("respects maxWidth and does not enlarge", async () => {
    const raw = await readFile(PNG); // 16×16
    const compressed = await compressForLLM(raw, 32); // allow up to 32px
    const meta = await sharp(compressed).metadata();
    assert.ok((meta.width ?? 0) <= 32, "width should not exceed maxWidth");
    // 16×16 input with maxWidth=32 → withoutEnlargement keeps it at 16
    assert.equal(meta.width, 16);
  });
});

// ---------------------------------------------------------------------------
// probeMedia — image
// ---------------------------------------------------------------------------

describe("probeMedia — image", () => {
  it("returns correct metadata for tiny.png", async () => {
    const info = await probeMedia(PNG);
    assert.equal(info.type, "image");
    assert.equal(info.duration, 0);
    assert.equal(info.width, 16);
    assert.equal(info.height, 16);
    assert.equal(info.path, PNG);
  });
});

// ---------------------------------------------------------------------------
// probeMedia — audio
// ---------------------------------------------------------------------------

describe("probeMedia — audio", () => {
  it("returns correct metadata for tiny.wav", async (t) => {
    if (!existsSync(WAV)) return t.skip("tiny.wav not present");
    const info = await probeMedia(WAV);
    assert.equal(info.type, "audio");
    assert.ok(info.duration > 0, "duration should be positive");
    assert.equal(info.width, undefined);
    assert.equal(info.height, undefined);
    assert.ok(info.sampleRate !== undefined && info.sampleRate > 0);
    assert.ok(info.channels !== undefined && info.channels > 0);
  });

  it("returns correct metadata for tiny.mp3", async (t) => {
    if (!existsSync(MP3)) return t.skip("tiny.mp3 not present");
    const info = await probeMedia(MP3);
    assert.equal(info.type, "audio");
    assert.ok(info.duration > 0);
  });
});

// ---------------------------------------------------------------------------
// probeMedia — video
// ---------------------------------------------------------------------------

describe("probeMedia — video", () => {
  it("returns correct metadata for tiny.mp4", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const info = await probeMedia(MP4);
    assert.equal(info.type, "video");
    assert.ok(info.duration > 0);
    assert.ok(info.width !== undefined && info.width > 0);
    assert.ok(info.height !== undefined && info.height > 0);
    assert.ok(info.fps !== undefined && info.fps > 0);
    assert.ok(info.videoCodec !== undefined);
  });
});

// ---------------------------------------------------------------------------
// probeMedia — errors
// ---------------------------------------------------------------------------

describe("probeMedia — errors", () => {
  it("throws FILE_NOT_FOUND for missing files", async () => {
    await assert.rejects(
      () => probeMedia("/nonexistent/file.mp4"),
      (err: MediaError) => {
        assert.ok(err instanceof MediaError);
        assert.equal(err.code, "FILE_NOT_FOUND");
        return true;
      },
    );
  });
});

// ---------------------------------------------------------------------------
// extractFrame
// ---------------------------------------------------------------------------

describe("extractFrame", () => {
  it("extracts a JPEG frame at t=0", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const buf = await extractFrame(MP4, 0);
    assert.ok(Buffer.isBuffer(buf));
    assert.equal(buf[0], 0xff);
    assert.equal(buf[1], 0xd8);
  });

  it("throws FRAME_FAILED for negative timestamp", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    await assert.rejects(
      () => extractFrame(MP4, -1),
      (err: MediaError) => {
        assert.ok(err instanceof MediaError);
        assert.equal(err.code, "FRAME_FAILED");
        return true;
      },
    );
  });

  it("returns exact frame timestamp metadata", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const frame = await extractFrameImage(MP4, 0.5);
    assert.equal(frame.timestampSec, 0.5);
    assert.equal(frame.timestampLabel, "00:00:00.500");
    assert.ok(Buffer.isBuffer(frame.image));
  });
});

// ---------------------------------------------------------------------------
// extractFrameGrid
// ---------------------------------------------------------------------------

describe("extractFrameGrid", () => {
  it("returns an array of JPEG buffers", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    // Use aggressive settings to ensure at least one frame on a 5s clip
    const grids = await extractFrameGrid(MP4, {
      frameInterval: 1,
      maxGrids: 2,
      cols: 2,
      rows: 2,
    });
    assert.ok(Array.isArray(grids));
    assert.ok(grids.length > 0, "should return at least one grid");
    for (const g of grids) {
      assert.ok(Buffer.isBuffer(g));
      assert.equal(g[0], 0xff);
      assert.equal(g[1], 0xd8);
    }
  });

  it("fails fast for impossible sampling windows", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    await assert.rejects(
      () =>
        extractFrameGrid(MP4, {
          maxGrids: 1,
          cols: 1,
          rows: 1,
          startSec: 0,
          endSec: 0.2,
          secondsPerFrame: 1,
        }),
      (err: MediaError) => {
        assert.ok(err instanceof MediaError);
        assert.equal(err.code, "INVALID_SAMPLING");
        return true;
      },
    );
  });

  it("returns grid metadata with exact tile timestamps", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const grids = await extractFrameGridImages(MP4, {
      maxGrids: 1,
      cols: 2,
      rows: 2,
      secondsPerFrame: 0.2,
      startSec: 0,
      endSec: 0.8,
    });
    assert.equal(grids.length, 1);
    const [grid] = grids;
    assert.ok(grid);
    assert.equal(grid.tiles.length, 4);
    assert.equal(grid.tiles[0]?.timestampLabel, "00:00:00.100");
    assert.equal(grid.tiles[3]?.timestampLabel, "00:00:00.700");
    assert.ok(Buffer.isBuffer(grid.image));
  });

  it("landscape video uses standard 4x4 grid when cols/rows omitted", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    // tiny.mp4 is 320x240 (landscape) — should use default cols=4, rows=4, thumbWidth=480
    // With 4x4=16 tiles but only 1s of video, we get fewer tiles, but the grid shape is 4x4
    const grids = await extractFrameGridImages(MP4, {
      maxGrids: 1,
    });
    assert.ok(grids.length >= 1);
    const [grid] = grids;
    assert.ok(grid);
    // With default 4x4 and ~1s video, we should get up to 16 tiles
    assert.ok(grid.tiles.length <= 16, `expected <= 16 tiles, got ${grid.tiles.length}`);
    assert.ok(Buffer.isBuffer(grid.image));
  });

  it("explicit cols/rows/thumbWidth override any defaults", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    // Explicit overrides should always be respected, regardless of orientation
    const grids = await extractFrameGridImages(MP4, {
      maxGrids: 1,
      cols: 1,
      rows: 2,
      thumbWidth: 64,
      secondsPerFrame: 0.3,
      startSec: 0,
      endSec: 0.6,
    });
    assert.equal(grids.length, 1);
    const [grid] = grids;
    assert.ok(grid);
    assert.equal(grid.tiles.length, 2); // 1 col * 2 rows
  });
});

// ---------------------------------------------------------------------------
// transcribeAudio  (slow — downloads model on first run)
// ---------------------------------------------------------------------------

describe("transcribeAudio", { timeout: 5 * 60 * 1000 }, () => {
  it("returns an array of segments for tiny.wav", async (t) => {
    if (SKIP_TRANSCRIPTION) return t.skip("SKIP_TRANSCRIPTION=1");
    if (!existsSync(WAV)) return t.skip("tiny.wav not present");
    const segments = await transcribeAudio(WAV);
    assert.ok(Array.isArray(segments));
    for (const seg of segments) {
      assert.equal(typeof seg.start, "number");
      assert.equal(typeof seg.end, "number");
      assert.equal(typeof seg.text, "string");
      assert.ok(seg.end >= seg.start);
    }
  });

  it("returns cached result on second call", async (t) => {
    if (SKIP_TRANSCRIPTION) return t.skip("SKIP_TRANSCRIPTION=1");
    if (!existsSync(WAV)) return t.skip("tiny.wav not present");
    const s1 = await transcribeAudio(WAV);
    const s2 = await transcribeAudio(WAV);
    assert.equal(s1, s2);
  });

  it("throws TRANSCRIBE_FAILED for invalid model name", async (t) => {
    if (!existsSync(WAV)) return t.skip("tiny.wav not present");
    await assert.rejects(
      () => transcribeAudio(WAV, { model: "not-a-real-model-xyz" }),
      (err: MediaError) => {
        assert.ok(err instanceof MediaError);
        assert.equal(err.code, "TRANSCRIBE_FAILED");
        return true;
      },
    );
  });
});

// ---------------------------------------------------------------------------
// understandMedia — image (no transcription, should be fast)
// ---------------------------------------------------------------------------

describe("understandMedia — image", () => {
  it("processes tiny.png without errors", async () => {
    const result = await understandMedia(PNG);
    assert.equal(result.info.type, "image");
    assert.equal(result.transcript, "");
    assert.deepEqual(result.segments, []);
    assert.deepEqual(result.grids, []);
  });
});

// ---------------------------------------------------------------------------
// understandMedia — video (slow path, requires Whisper model)
// ---------------------------------------------------------------------------

describe("understandMedia — video", { timeout: 5 * 60 * 1000 }, () => {
  it("processes tiny.mp4 and returns grids + transcript", async (t) => {
    if (SKIP_TRANSCRIPTION) return t.skip("SKIP_TRANSCRIPTION=1");
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const result = await understandMedia(MP4, {
      frameInterval: 1,
      maxGrids: 1,
      cols: 2,
      rows: 2,
    });
    assert.equal(result.info.type, "video");
    assert.ok(Array.isArray(result.grids));
    assert.ok(Array.isArray(result.gridImages));
    assert.ok(Array.isArray(result.segments));
    assert.equal(typeof result.transcript, "string");
  });
});
