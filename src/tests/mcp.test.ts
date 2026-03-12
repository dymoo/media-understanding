/**
 * Tests for MCP server tool behavior.
 *
 * Since the MCP tools delegate directly to media.ts functions, these tests
 * verify the integration layer: argument mapping, error surface (MediaError),
 * and content-type shaping. We exercise the functions called by each tool
 * rather than spinning up a full MCP stdio transport.
 */

import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { probeMedia, extractFrameGrid, extractFrame, compressForLLM } from "../media.js";
import { MediaError } from "../types.js";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const root = resolve(import.meta.dirname, "..", "..", "testdata");
const PNG = resolve(root, "tiny.png");
const MP4 = resolve(root, "tiny.mp4");

// ---------------------------------------------------------------------------
// MediaError shape (used by mcpError in mcp.ts)
// ---------------------------------------------------------------------------

describe("MediaError", () => {
  it("has the correct code and message", () => {
    const err = new MediaError("FILE_NOT_FOUND", "test message");
    assert.equal(err.code, "FILE_NOT_FOUND");
    assert.equal(err.message, "test message");
    assert.equal(err.name, "MediaError");
    assert.ok(err instanceof Error);
    assert.ok(err instanceof MediaError);
  });

  it("stores cause when provided", () => {
    const cause = new Error("root cause");
    const err = new MediaError("UNKNOWN", "wrapper", cause);
    assert.equal(err.cause, cause);
  });

  it("is identifiable by instanceof after throw/catch", () => {
    let caught: unknown;
    try {
      throw new MediaError("FFMPEG_NOT_FOUND", "ffmpeg missing");
    } catch (e) {
      caught = e;
    }
    assert.ok(caught instanceof MediaError);
    assert.equal((caught as MediaError).code, "FFMPEG_NOT_FOUND");
  });
});

// ---------------------------------------------------------------------------
// understand_media tool surface — image path (no transcription)
// ---------------------------------------------------------------------------

describe("understand_media tool surface — image", () => {
  it("probeMedia returns image type for tiny.png", async () => {
    const info = await probeMedia(PNG);
    assert.equal(info.type, "image");
  });

  it("produces a JPEG from probeMedia + compressForLLM", async () => {
    const { readFile } = await import("node:fs/promises");
    const info = await probeMedia(PNG);
    const raw = await readFile(info.path);
    const compressed = await compressForLLM(raw);
    assert.equal(compressed[0], 0xff);
    assert.equal(compressed[1], 0xd8);
    const b64 = compressed.toString("base64");
    assert.ok(b64.length > 0);
  });
});

// ---------------------------------------------------------------------------
// get_video_grids tool surface
// ---------------------------------------------------------------------------

describe("get_video_grids tool surface", () => {
  it("extractFrameGrid returns JPEG buffers for tiny.mp4", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const grids = await extractFrameGrid(MP4, {
      frameInterval: 1,
      maxGrids: 1,
      cols: 2,
      rows: 2,
    });
    assert.ok(Array.isArray(grids));
    for (const g of grids) {
      assert.ok(Buffer.isBuffer(g));
    }
  });

  it("extractFrameGrid propagates FILE_NOT_FOUND", async () => {
    await assert.rejects(
      () => extractFrameGrid("/nonexistent/video.mp4"),
      (err: MediaError) => {
        assert.ok(err instanceof MediaError);
        assert.equal(err.code, "FILE_NOT_FOUND");
        return true;
      },
    );
  });
});

// ---------------------------------------------------------------------------
// get_frames tool surface
// ---------------------------------------------------------------------------

describe("get_frames tool surface", () => {
  it("extractFrame returns a JPEG for t=0", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const buf = await extractFrame(MP4, 0);
    assert.ok(Buffer.isBuffer(buf));
    assert.equal(buf[0], 0xff);
    assert.equal(buf[1], 0xd8);
  });

  it("extractFrame propagates FRAME_FAILED for negative timestamp", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    await assert.rejects(
      () => extractFrame(MP4, -5),
      (err: MediaError) => {
        assert.ok(err instanceof MediaError);
        assert.equal(err.code, "FRAME_FAILED");
        return true;
      },
    );
  });
});

// ---------------------------------------------------------------------------
// get_transcript tool surface — validates argument mapping only (no download)
// ---------------------------------------------------------------------------

describe("get_transcript tool surface — validation", () => {
  it("transcribeAudio throws TRANSCRIBE_FAILED for invalid model name", async (t) => {
    if (!existsSync(PNG)) return t.skip("tiny.png not present");
    const { transcribeAudio } = await import("../media.js");
    await assert.rejects(
      () => transcribeAudio(PNG, { model: "definitely-not-a-valid-model-xyz" }),
      (err: MediaError) => {
        assert.ok(err instanceof MediaError);
        assert.equal(err.code, "TRANSCRIBE_FAILED");
        return true;
      },
    );
  });

  it("transcribeAudio throws an appropriate error for image files (no audio)", async (t) => {
    if (!existsSync(PNG)) return t.skip("tiny.png not present");
    const { transcribeAudio } = await import("../media.js");
    // PNG has no audio stream; after model validation we expect a meaningful error
    await assert.rejects(
      () => transcribeAudio(PNG),
      (err: MediaError) => {
        assert.ok(err instanceof MediaError);
        assert.ok(
          err.code === "NO_AUDIO_STREAM" ||
            err.code === "TRANSCRIBE_FAILED" ||
            err.code === "UNSUPPORTED_FORMAT",
          `unexpected code: ${err.code}`,
        );
        return true;
      },
    );
  });
});
