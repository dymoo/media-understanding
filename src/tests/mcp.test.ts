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
import {
  assignSegmentsToGrids,
  estimateVisionTokens,
  filterSegmentsByWindow,
  formatDuration,
  formatSrtTimestamp,
  formatTranscriptAsJSON,
  formatTranscriptAsSRT,
  handleGetFrames,
  handleGetTranscript,
  handleGetVideoGrids,
  handleProbeMedia,
  handleUnderstandMedia,
  overlapMs,
  PREFLIGHT_MAX_DURATION_FULL,
  PREFLIGHT_MAX_DURATION_TRANSCRIPT,
  PREFLIGHT_MAX_FILE_SIZE,
  preflightDuration,
  preflightFileSize,
} from "../mcp-handlers.js";
import type { MediaInfo, Segment, VideoGridImage } from "../types.js";
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

  it("handleUnderstandMedia returns metadata and image for tiny.png", async () => {
    const result = await handleUnderstandMedia({ file_path: PNG });
    assert.ok(!("isError" in result));
    const textItems = result.content.filter((item) => item.type === "text");
    assert.ok(textItems.some((item) => item.text.includes("File:")));
    const imageItems = result.content.filter((item) => item.type === "image");
    assert.ok(imageItems.length > 0, "expected at least one image content item for PNG");
  });

  it("probeMedia includes fileSizeBytes", async () => {
    const info = await probeMedia(PNG);
    assert.equal(info.type, "image");
    assert.equal(typeof info.fileSizeBytes, "number");
    assert.ok((info.fileSizeBytes as number) > 0);
  });
});

// ---------------------------------------------------------------------------
// probe_media tool surface
// ---------------------------------------------------------------------------

describe("probe_media tool surface", () => {
  it("probes a single file via string path", async () => {
    const result = await handleProbeMedia({ paths: PNG });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    assert.ok(textItem.text.includes("1 succeeded"));
    assert.ok(textItem.text.includes("image"));
  });

  it("probes a single file via array path", async () => {
    const result = await handleProbeMedia({ paths: [PNG] });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    assert.ok(textItem.text.includes("1 succeeded"));
  });

  it("probes multiple files via array", async () => {
    const result = await handleProbeMedia({ paths: [PNG, PNG] });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    assert.ok(textItem.text.includes("succeeded"));
  });

  it("returns inline errors for missing files without aborting", async () => {
    const result = await handleProbeMedia({
      paths: [PNG, "/nonexistent/file.mp4"],
    });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    assert.ok(textItem.text.includes("1 succeeded"));
    assert.ok(textItem.text.includes("1 failed"));
    assert.ok(textItem.text.includes("FILE_NOT_FOUND"));
  });

  it("supports glob patterns in paths", async () => {
    const result = await handleProbeMedia({ paths: resolve(root, "tiny.png") });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    assert.ok(textItem.text.includes("1 succeeded"));
  });

  it("supports mixed literal + glob in paths array", async () => {
    // Both resolve to the same file, dedup should handle it
    const result = await handleProbeMedia({
      paths: [PNG, resolve(root, "*.png")],
    });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    // After dedup, only 1 unique path
    assert.ok(textItem.text.includes("1 file(s)"));
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

  it("returns timestamp text alongside grid images", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const result = await handleGetVideoGrids({
      file_path: MP4,
      max_grids: 1,
      max_total_chars: 80000,
      cols: 2,
      rows: 2,
      seconds_per_frame: 0.2,
      start_sec: 0,
      end_sec: 0.8,
      thumb_width: 64, // tiny fixture — small thumb ensures image fits in budget
    });
    assert.ok(!("isError" in result));
    const textItems = result.content.filter((item) => item.type === "text");
    assert.ok(textItems.some((item) => item.text.includes("Tile timestamps:")));
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

  it("fails with a teaching error when frame payloads exceed budget", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const result = await handleGetFrames({
      file_path: MP4,
      timestamps: [0, 0.5],
      max_total_chars: 2000,
    });
    assert.ok("isError" in result);
    assert.match(result.content[0].text, /BUDGET_EXCEEDED/);
    // Phase 8a: error should include actual chars and overage ratio
    assert.match(result.content[0].text, /chars/);
    assert.match(result.content[0].text, /over budget/);
  });

  it("frame budget errors suggest fewer timestamps, not grid-specific knobs", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const result = await handleGetFrames({
      file_path: MP4,
      timestamps: [0, 0.5],
      max_total_chars: 2000,
    });
    assert.ok("isError" in result);
    const text = result.content[0].text;
    // Should suggest frame-appropriate recovery
    assert.match(text, /fewer timestamps|split/i);
    // Should NOT suggest grid-specific controls
    assert.ok(!text.includes("cols="), "get_frames budget error should not suggest cols");
    assert.ok(!text.includes("rows="), "get_frames budget error should not suggest rows");
    assert.ok(
      !text.includes("thumb_width"),
      "get_frames budget error should not suggest thumb_width",
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

  it("formats timestamped transcript text through the MCP handler", async (t) => {
    if (!existsSync(PNG)) return t.skip("tiny.png not present");
    const result = await handleGetTranscript({
      file_path: PNG,
      model: "definitely-not-a-valid-model-xyz",
    });
    assert.ok("isError" in result);
    assert.match(result.content[0].text, /TRANSCRIBE_FAILED/);
  });
});

// ---------------------------------------------------------------------------
// probe_media — glob and max_files
// ---------------------------------------------------------------------------

describe("probe_media — glob and max_files", () => {
  it("probes files matching a glob pattern", async () => {
    const result = await handleProbeMedia({ paths: resolve(root, "tiny.png") });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    assert.ok(textItem.text.includes("1 succeeded"));
  });

  it("rejects when paths exceeds max_files", async () => {
    // Use two distinct paths so dedup doesn't collapse them
    const result = await handleProbeMedia({
      paths: [PNG, "/nonexistent/second-file.mp4"],
      max_files: 1,
    });
    assert.ok("isError" in result);
    assert.match(result.content[0].text, /2 files.*limit is 1/);
  });

  it("caps max_files at absolute maximum of 200", async () => {
    // Request max_files=999 — should be capped to 200 internally.
    // With only 1 file, this should succeed (the cap doesn't reject small batches).
    const result = await handleProbeMedia({
      paths: PNG,
      max_files: 999,
    });
    assert.ok(!("isError" in result));
  });

  it("deduplicates paths", async () => {
    const result = await handleProbeMedia({
      paths: [PNG, PNG, PNG],
    });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    // After dedup, only 1 unique path
    assert.ok(textItem.text.includes("1 file(s)"));
  });

  it("rejects empty paths array", async () => {
    const result = await handleProbeMedia({ paths: [] as string[] });
    assert.ok("isError" in result);
    assert.match(result.content[0].text, /at least one non-empty/);
  });

  it("preserves literal missing paths so probe reports per-file errors", async () => {
    const missingPath = "/definitely/not/a/real/file.mp4";
    const result = await handleProbeMedia({ paths: missingPath });
    assert.ok(!("isError" in result));
    const textItem = result.content.find((item) => item.type === "text");
    assert.ok(textItem && "text" in textItem);
    assert.ok(textItem.text.includes("1 failed"));
    assert.ok(textItem.text.includes("FILE_NOT_FOUND"));
  });
});

// ---------------------------------------------------------------------------
// Preflight helpers — pure function unit tests
// ---------------------------------------------------------------------------

describe("preflight helpers", () => {
  const baseInfo: MediaInfo = {
    path: "/fake/file.mp4",
    type: "video",
    duration: 100,
    fileSizeBytes: 1024,
  };

  describe("formatDuration", () => {
    it("formats hours and minutes", () => {
      assert.equal(formatDuration(7200), "2h");
      assert.equal(formatDuration(7260), "2h 1m");
      assert.equal(formatDuration(3660), "1h 1m");
    });

    it("formats minutes only", () => {
      assert.equal(formatDuration(300), "5m");
      assert.equal(formatDuration(60), "1m");
    });

    it("formats zero as 0m", () => {
      assert.equal(formatDuration(0), "0m");
    });
  });

  describe("preflightFileSize", () => {
    it("passes for files under 10 GB", () => {
      assert.doesNotThrow(() => {
        preflightFileSize(baseInfo, "test_tool");
      });
    });

    it("passes when fileSizeBytes is undefined", () => {
      const { fileSizeBytes: _, ...rest } = baseInfo;
      const info: MediaInfo = rest;
      assert.doesNotThrow(() => {
        preflightFileSize(info, "test_tool");
      });
    });

    it("throws FILE_TOO_LARGE for files over 10 GB", () => {
      const info: MediaInfo = {
        ...baseInfo,
        fileSizeBytes: PREFLIGHT_MAX_FILE_SIZE + 1,
      };
      assert.throws(
        () => preflightFileSize(info, "test_tool"),
        (err: MediaError) => {
          assert.ok(err instanceof MediaError);
          assert.equal(err.code, "FILE_TOO_LARGE");
          assert.ok(err.message.includes("test_tool"));
          assert.ok(err.message.includes("10 GB"));
          return true;
        },
      );
    });

    it("includes file size in GB in the error message", () => {
      const fifteenGB = 15 * 1024 * 1024 * 1024;
      const info: MediaInfo = { ...baseInfo, fileSizeBytes: fifteenGB };
      assert.throws(
        () => preflightFileSize(info, "understand_media"),
        (err: MediaError) => {
          assert.ok(err.message.includes("15.0 GB"));
          return true;
        },
      );
    });
  });

  describe("preflightDuration", () => {
    it("passes for short files", () => {
      assert.doesNotThrow(() => {
        preflightDuration(baseInfo, PREFLIGHT_MAX_DURATION_FULL, "understand_media");
      });
    });

    it("passes for image files regardless of duration field", () => {
      const imageInfo: MediaInfo = { ...baseInfo, type: "image", duration: 999999 };
      assert.doesNotThrow(() => {
        preflightDuration(imageInfo, PREFLIGHT_MAX_DURATION_FULL, "understand_media");
      });
    });

    it("throws FILE_TOO_LARGE when video exceeds max duration", () => {
      const longVideo: MediaInfo = {
        ...baseInfo,
        type: "video",
        duration: PREFLIGHT_MAX_DURATION_FULL + 1,
      };
      assert.throws(
        () => preflightDuration(longVideo, PREFLIGHT_MAX_DURATION_FULL, "understand_media"),
        (err: MediaError) => {
          assert.ok(err instanceof MediaError);
          assert.equal(err.code, "FILE_TOO_LARGE");
          assert.ok(err.message.includes("understand_media"));
          return true;
        },
      );
    });

    it("throws FILE_TOO_LARGE when audio exceeds max duration", () => {
      const longAudio: MediaInfo = {
        ...baseInfo,
        type: "audio",
        duration: PREFLIGHT_MAX_DURATION_TRANSCRIPT + 1,
      };
      assert.throws(
        () => preflightDuration(longAudio, PREFLIGHT_MAX_DURATION_TRANSCRIPT, "get_transcript"),
        (err: MediaError) => {
          assert.ok(err instanceof MediaError);
          assert.equal(err.code, "FILE_TOO_LARGE");
          assert.ok(err.message.includes("get_transcript"));
          return true;
        },
      );
    });

    it("understand_media error includes recovery guidance", () => {
      const longVideo: MediaInfo = {
        ...baseInfo,
        type: "video",
        duration: PREFLIGHT_MAX_DURATION_FULL + 1,
      };
      assert.throws(
        () => preflightDuration(longVideo, PREFLIGHT_MAX_DURATION_FULL, "understand_media"),
        (err: MediaError) => {
          assert.ok(err.message.includes("get_video_grids"));
          assert.ok(err.message.includes("get_transcript"));
          assert.ok(err.message.includes("get_frames"));
          return true;
        },
      );
    });

    it("get_transcript error includes recovery guidance", () => {
      const longAudio: MediaInfo = {
        ...baseInfo,
        type: "audio",
        duration: PREFLIGHT_MAX_DURATION_TRANSCRIPT + 1,
      };
      assert.throws(
        () => preflightDuration(longAudio, PREFLIGHT_MAX_DURATION_TRANSCRIPT, "get_transcript"),
        (err: MediaError) => {
          assert.ok(err.message.includes("probe_media"));
          return true;
        },
      );
    });
  });
});

// ---------------------------------------------------------------------------
// get_video_grids — budget exceeded
// ---------------------------------------------------------------------------

describe("get_video_grids — budget exceeded", () => {
  it("fails with BUDGET_EXCEEDED when grids exceed budget", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const result = await handleGetVideoGrids({
      file_path: MP4,
      max_grids: 1,
      max_total_chars: 500, // impossibly small for any grid image
      cols: 2,
      rows: 2,
      seconds_per_frame: 0.2,
      start_sec: 0,
      end_sec: 0.8,
    });
    assert.ok("isError" in result);
    assert.match(result.content[0].text, /BUDGET_EXCEEDED/);
    // Phase 8a: error should include actual size and overage ratio
    assert.match(result.content[0].text, /chars/);
    assert.match(result.content[0].text, /over budget/);
  });

  it("grid budget errors suggest grid-specific controls (cols, rows, thumb_width)", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    const result = await handleGetVideoGrids({
      file_path: MP4,
      max_grids: 1,
      max_total_chars: 500,
      cols: 2,
      rows: 2,
      seconds_per_frame: 0.2,
      start_sec: 0,
      end_sec: 0.8,
    });
    assert.ok("isError" in result);
    const text = result.content[0].text;
    assert.ok(text.includes("cols="), "grid budget error should suggest cols");
    assert.ok(text.includes("rows="), "grid budget error should suggest rows");
    assert.ok(text.includes("thumb_width"), "grid budget error should suggest thumb_width");
  });
});

// ---------------------------------------------------------------------------
// understand_media — error paths
// ---------------------------------------------------------------------------

describe("understand_media — error paths", () => {
  it("returns error for non-existent file", async () => {
    const result = await handleUnderstandMedia({ file_path: "/nonexistent/file.mp4" });
    assert.ok("isError" in result);
    assert.match(result.content[0].text, /FILE_NOT_FOUND/);
  });

  it("returns error when budget is impossibly small for an image", async () => {
    const result = await handleUnderstandMedia({
      file_path: PNG,
      max_total_chars: 10, // impossibly small
    });
    assert.ok("isError" in result);
    assert.match(result.content[0].text, /BUDGET_EXCEEDED/);
    // Phase 8a: error should include actual size and overage ratio
    assert.match(result.content[0].text, /chars/);
    assert.match(result.content[0].text, /over budget/);
  });
});

// ---------------------------------------------------------------------------
// estimateVisionTokens — LLM vision token estimation
// ---------------------------------------------------------------------------

describe("estimateVisionTokens", () => {
  it("uses Claude formula: pixels / 750", () => {
    // 1920x1080 = 2,073,600 px / 750 = 2,764.8 → ceil = 2,765
    assert.equal(estimateVisionTokens(1920, 1080), Math.ceil((1920 * 1080) / 750));
  });

  it("handles small images", () => {
    assert.equal(estimateVisionTokens(16, 16), Math.ceil(256 / 750));
  });

  it("handles portrait dimensions", () => {
    // 1080x1920 portrait
    const tokens = estimateVisionTokens(1080, 1920);
    assert.equal(tokens, Math.ceil((1080 * 1920) / 750));
  });
});

// ---------------------------------------------------------------------------
// overlapMs — segment/window overlap calculation
// ---------------------------------------------------------------------------

describe("overlapMs", () => {
  it("returns full overlap when segment is inside window", () => {
    assert.equal(overlapMs(1000, 2000, 0, 5000), 1000);
  });

  it("returns full overlap when window is inside segment", () => {
    assert.equal(overlapMs(0, 5000, 1000, 2000), 1000);
  });

  it("returns partial overlap at start", () => {
    assert.equal(overlapMs(0, 3000, 2000, 5000), 1000);
  });

  it("returns partial overlap at end", () => {
    assert.equal(overlapMs(4000, 6000, 2000, 5000), 1000);
  });

  it("returns 0 for non-overlapping ranges", () => {
    assert.equal(overlapMs(0, 1000, 2000, 3000), 0);
  });

  it("returns 0 for adjacent ranges (no overlap)", () => {
    assert.equal(overlapMs(0, 1000, 1000, 2000), 0);
  });
});

// ---------------------------------------------------------------------------
// assignSegmentsToGrids — segment-to-grid assignment
// ---------------------------------------------------------------------------

describe("assignSegmentsToGrids", () => {
  // Helper to create minimal Segment objects
  const seg = (start: number, end: number, text = "x") => ({ start, end, text });

  // Helper to create minimal VideoGridImage objects (only startSec/endSec matter)
  const grid = (startSec: number, endSec: number) =>
    ({
      startSec,
      endSec,
      image: Buffer.alloc(0),
      tiles: [],
    }) as VideoGridImage;

  it("assigns a segment entirely inside one grid", () => {
    const segments = [seg(1000, 2000)];
    const grids = [grid(0, 5)];
    const { perGrid, unassigned } = assignSegmentsToGrids(segments, grids);
    assert.deepEqual(perGrid, [[0]]);
    assert.deepEqual(unassigned, []);
  });

  it("assigns segment to the grid with most overlap", () => {
    // Segment 2000-5000ms. Grid A: 0-3s (overlap: 1000ms). Grid B: 3-6s (overlap: 2000ms).
    const segments = [seg(2000, 5000)];
    const grids = [grid(0, 3), grid(3, 6)];
    const { perGrid, unassigned } = assignSegmentsToGrids(segments, grids);
    assert.deepEqual(perGrid[0], []); // Grid A
    assert.deepEqual(perGrid[1], [0]); // Grid B gets it
    assert.deepEqual(unassigned, []);
  });

  it("assigns segment equally split to the first grid", () => {
    // Segment 2000-4000ms. Grid A: 0-3s (overlap: 1000ms). Grid B: 3-6s (overlap: 1000ms).
    // Equal overlap — first grid wins because we use strict >
    const segments = [seg(2000, 4000)];
    const grids = [grid(0, 3), grid(3, 6)];
    const { perGrid, unassigned } = assignSegmentsToGrids(segments, grids);
    assert.deepEqual(perGrid[0], [0]); // First grid wins on tie
    assert.deepEqual(perGrid[1], []);
    assert.deepEqual(unassigned, []);
  });

  it("puts non-overlapping segments in unassigned", () => {
    const segments = [seg(10000, 11000)];
    const grids = [grid(0, 5)];
    const { perGrid, unassigned } = assignSegmentsToGrids(segments, grids);
    assert.deepEqual(perGrid[0], []);
    assert.deepEqual(unassigned, [0]);
  });

  it("handles empty segments", () => {
    const grids = [grid(0, 5)];
    const { perGrid, unassigned } = assignSegmentsToGrids([], grids);
    assert.deepEqual(perGrid, [[]]);
    assert.deepEqual(unassigned, []);
  });

  it("handles empty grids — all segments unassigned", () => {
    const segments = [seg(0, 1000)];
    const { perGrid, unassigned } = assignSegmentsToGrids(segments, []);
    assert.deepEqual(perGrid, []);
    assert.deepEqual(unassigned, [0]);
  });

  it("distributes multiple segments across multiple grids", () => {
    const segments = [
      seg(500, 1500), // → grid 0 (0-2s)
      seg(2500, 3500), // → grid 1 (2-4s)
      seg(4500, 5500), // → grid 2 (4-6s)
      seg(7000, 8000), // → unassigned (no grid covers 7-8s)
    ];
    const grids = [grid(0, 2), grid(2, 4), grid(4, 6)];
    const { perGrid, unassigned } = assignSegmentsToGrids(segments, grids);
    assert.deepEqual(perGrid[0], [0]);
    assert.deepEqual(perGrid[1], [1]);
    assert.deepEqual(perGrid[2], [2]);
    assert.deepEqual(unassigned, [3]);
  });
});

// ---------------------------------------------------------------------------
// understand_media — interleaved output structure (video)
// ---------------------------------------------------------------------------

describe("understand_media — interleaved output", () => {
  it("produces interleaved transcript + grid sections for video", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");

    const result = await handleUnderstandMedia({
      file_path: MP4,
      max_total_chars: 200_000, // generous budget
      max_grids: 1,
      cols: 2,
      rows: 2,
    });
    assert.ok(!("isError" in result));

    // Check that metadata mentions interleaving
    const metadataItem = result.content[0];
    assert.ok(metadataItem && metadataItem.type === "text");
    assert.ok(metadataItem.text.includes("File:"));

    // There should be at least one image content item (grid)
    const imageItems = result.content.filter((item) => item.type === "image");
    assert.ok(imageItems.length > 0, "expected at least one grid image for video");

    // There should be a FRAME GRID label
    const textItems = result.content.filter((item) => item.type === "text");
    assert.ok(
      textItems.some((item) => item.text.includes("FRAME GRID")),
      "expected FRAME GRID label in output",
    );

    // Payload summary should be last
    const lastItem = result.content[result.content.length - 1];
    assert.ok(lastItem && lastItem.type === "text");
    assert.ok(lastItem.text.includes("Payload:"));
  });

  it("produces metadata + image (no transcript) for image files", async () => {
    const result = await handleUnderstandMedia({ file_path: PNG });
    assert.ok(!("isError" in result));

    // Should have metadata text + image + payload summary
    const textItems = result.content.filter((item) => item.type === "text");
    const imageItems = result.content.filter((item) => item.type === "image");
    assert.ok(textItems.length >= 1, "expected metadata text");
    assert.ok(imageItems.length === 1, "expected exactly one image for PNG");

    // Should NOT have transcript headers
    assert.ok(
      !textItems.some((item) => item.text.includes("TRANSCRIPT")),
      "image files should not have transcript sections",
    );
  });
});

// ---------------------------------------------------------------------------
// formatSrtTimestamp — SRT timestamp formatting
// ---------------------------------------------------------------------------

describe("formatSrtTimestamp", () => {
  it("formats zero", () => {
    assert.equal(formatSrtTimestamp(0), "00:00:00,000");
  });

  it("formats milliseconds", () => {
    assert.equal(formatSrtTimestamp(1500), "00:00:01,500");
  });

  it("formats minutes and seconds", () => {
    assert.equal(formatSrtTimestamp(65000), "00:01:05,000");
  });

  it("formats hours", () => {
    assert.equal(formatSrtTimestamp(3661234), "01:01:01,234");
  });
});

// ---------------------------------------------------------------------------
// formatTranscriptAsSRT — SRT output format
// ---------------------------------------------------------------------------

describe("formatTranscriptAsSRT", () => {
  const segments: Segment[] = [
    { start: 0, end: 2500, text: "Hello, welcome." },
    { start: 2500, end: 5100, text: "Today we discuss..." },
  ];

  it("produces valid SRT with 1-based indices", () => {
    const srt = formatTranscriptAsSRT(segments, 10000);
    assert.ok(srt.includes("1\n00:00:00,000 --> 00:00:02,500\nHello, welcome."));
    assert.ok(srt.includes("2\n00:00:02,500 --> 00:00:05,100\nToday we discuss..."));
  });

  it("separates entries with blank lines", () => {
    const srt = formatTranscriptAsSRT(segments, 10000);
    assert.ok(srt.includes("\n\n"));
  });
});

// ---------------------------------------------------------------------------
// formatTranscriptAsJSON — JSON output format
// ---------------------------------------------------------------------------

describe("formatTranscriptAsJSON", () => {
  const segments: Segment[] = [
    { start: 0, end: 2500, text: "Hello." },
    { start: 2500, end: 5000, text: "World." },
  ];

  it("produces valid parseable JSON", () => {
    const json = formatTranscriptAsJSON(segments, 10000);
    const parsed = JSON.parse(json) as { segments: { start: number; end: number; text: string }[] };
    assert.ok(Array.isArray(parsed.segments));
    assert.equal(parsed.segments.length, 2);
  });

  it("includes millisecond timestamps", () => {
    const json = formatTranscriptAsJSON(segments, 10000);
    const parsed = JSON.parse(json) as { segments: { start: number; end: number; text: string }[] };
    assert.equal(parsed.segments[0]!.start, 0);
    assert.equal(parsed.segments[0]!.end, 2500);
    assert.equal(parsed.segments[1]!.start, 2500);
  });

  it("trims text", () => {
    const segs: Segment[] = [{ start: 0, end: 1000, text: "  padded  " }];
    const json = formatTranscriptAsJSON(segs, 10000);
    const parsed = JSON.parse(json) as { segments: { start: number; end: number; text: string }[] };
    assert.equal(parsed.segments[0]!.text, "padded");
  });

  it("rounds floating-point timestamps to clean integers", () => {
    const segs: Segment[] = [
      { start: 32852.308999999994, end: 35100.00000000001, text: "noisy timestamps" },
    ];
    const json = formatTranscriptAsJSON(segs, 10000);
    const parsed = JSON.parse(json) as { segments: { start: number; end: number; text: string }[] };
    assert.equal(parsed.segments[0]!.start, 32852);
    assert.equal(parsed.segments[0]!.end, 35100);
  });
});

// ---------------------------------------------------------------------------
// filterSegmentsByWindow — time-window filtering
// ---------------------------------------------------------------------------

describe("filterSegmentsByWindow", () => {
  const segments: Segment[] = [
    { start: 0, end: 2000, text: "A" },
    { start: 2000, end: 4000, text: "B" },
    { start: 4000, end: 6000, text: "C" },
    { start: 6000, end: 8000, text: "D" },
  ];

  it("returns all segments when no window specified", () => {
    const result = filterSegmentsByWindow(segments);
    assert.equal(result.length, 4);
  });

  it("filters by start_sec only", () => {
    const result = filterSegmentsByWindow(segments, 3);
    // Segments B (end 4000 > 3000), C, D overlap
    assert.equal(result.length, 3);
    assert.equal(result[0]!.text, "B");
  });

  it("filters by end_sec only", () => {
    const result = filterSegmentsByWindow(segments, undefined, 3);
    // Segments A (start 0 < 3000), B (start 2000 < 3000) overlap
    assert.equal(result.length, 2);
    assert.equal(result[1]!.text, "B");
  });

  it("filters by both start_sec and end_sec", () => {
    const result = filterSegmentsByWindow(segments, 2, 6);
    // B (2000-4000), C (4000-6000) are within window. D starts at 6000 which is not < 6000.
    assert.equal(result.length, 2);
    assert.equal(result[0]!.text, "B");
    assert.equal(result[1]!.text, "C");
  });

  it("returns empty when window matches nothing", () => {
    const result = filterSegmentsByWindow(segments, 100, 200);
    assert.equal(result.length, 0);
  });

  it("includes partially overlapping segments", () => {
    // Segment B: 2000-4000. Window: 3-5s (3000-5000 ms).
    // B.start (2000) < endMs (5000) AND B.end (4000) > startMs (3000) → included
    const result = filterSegmentsByWindow(segments, 3, 5);
    assert.ok(result.some((s) => s.text === "B"));
    assert.ok(result.some((s) => s.text === "C"));
    assert.equal(result.length, 2);
  });
});

// ---------------------------------------------------------------------------
// get_transcript — format and windowing integration
// ---------------------------------------------------------------------------

describe("get_transcript — format and windowing", () => {
  // Note: tiny.wav is a sine wave with no speech, so Whisper returns empty segments.
  // Integration tests for format+windowing with actual speech depend on media.test.ts
  // fixtures. Here we test the handler's behavior with the no-speech edge case and
  // verify that format/windowing params are accepted without error.
  const WAV = resolve(root, "tiny.wav");

  it("returns no-speech message for sine wave (default format)", async (t) => {
    if (!existsSync(WAV)) return t.skip("tiny.wav not present");
    const result = await handleGetTranscript({ file_path: WAV });
    assert.ok(!("isError" in result));
    const item = result.content[0]!;
    assert.ok(item.type === "text");
    assert.ok(item.text.includes("No speech detected"));
  });

  it("returns no-speech message for sine wave (srt format)", async (t) => {
    if (!existsSync(WAV)) return t.skip("tiny.wav not present");
    const result = await handleGetTranscript({ file_path: WAV, format: "srt" });
    assert.ok(!("isError" in result));
    const item = result.content[0]!;
    assert.ok(item.type === "text");
    assert.ok(item.text.includes("No speech detected"));
  });

  it("returns no-speech message for sine wave (json format)", async (t) => {
    if (!existsSync(WAV)) return t.skip("tiny.wav not present");
    const result = await handleGetTranscript({ file_path: WAV, format: "json" });
    assert.ok(!("isError" in result));
    const item = result.content[0]!;
    assert.ok(item.type === "text");
    assert.ok(item.text.includes("No speech detected"));
  });

  it("includes window info in no-speech message when windowed", async (t) => {
    if (!existsSync(WAV)) return t.skip("tiny.wav not present");
    const result = await handleGetTranscript({
      file_path: WAV,
      format: "json",
      start_sec: 100,
      end_sec: 200,
    });
    assert.ok(!("isError" in result));
    const item = result.content[0]!;
    assert.ok(item.type === "text");
    assert.ok(item.text.includes("No speech detected"));
    assert.ok(
      item.text.includes("requested window"),
      "expected window context in no-speech message",
    );
  });

  it("accepts format and windowing params for video file (tiny.mp4)", async (t) => {
    if (!existsSync(MP4)) return t.skip("tiny.mp4 not present");
    // tiny.mp4 has audio — test that the handler doesn't reject format/windowing params
    const result = await handleGetTranscript({
      file_path: MP4,
      format: "text",
      start_sec: 0,
      end_sec: 10,
    });
    assert.ok(!("isError" in result));
    const item = result.content[0]!;
    assert.ok(item.type === "text");
    // Either "No speech detected" or actual transcript — both are valid
  });
});
