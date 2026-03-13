/**
 * Unit tests for src/accel.ts
 *
 * These tests exercise the AccelAdapter interface via the SoftwareAdapter
 * (always available; no GPU required) and the factory helpers getAdapter /
 * resetAdapter.
 *
 * `scaleAndEncode` and `createDecoder` require an open video stream and a
 * live FFmpeg pipeline; they are covered by extractFramesBatch in
 * media.test.ts.  Here we focus on the pure-sharp helpers and factory logic
 * that can run without media fixtures.
 */

import { describe, it, after } from "node:test";
import * as assert from "node:assert/strict";
import { resolve } from "node:path";

import { getAdapter, resetAdapter } from "../accel.js";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const root = resolve(import.meta.dirname, "..", "..", "testdata");
const PNG = resolve(root, "tiny.png");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

import sharp from "sharp";

/** Build a minimal valid JPEG buffer using the PNG fixture via sharp. */
async function makeJpegBuffer(): Promise<Buffer> {
  return sharp(PNG).jpeg({ quality: 80 }).toBuffer();
}

// ---------------------------------------------------------------------------
// Factory — getAdapter / resetAdapter
// ---------------------------------------------------------------------------

describe("getAdapter", () => {
  after(() => {
    // Restore original env and singleton after each sub-test group.
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });

  it("returns an AccelAdapter with a non-empty backend name", () => {
    resetAdapter();
    const adapter = getAdapter();
    assert.ok(typeof adapter.info.backend === "string");
    assert.ok(adapter.info.backend.length > 0);
  });

  it("returns the same singleton on repeated calls", () => {
    resetAdapter();
    const a = getAdapter();
    const b = getAdapter();
    assert.equal(a, b);
  });

  it("returns SoftwareAdapter when MEDIA_UNDERSTANDING_DISABLE_HW=1", () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const adapter = getAdapter();
    assert.equal(adapter.info.backend, "software");
    assert.equal(adapter.info.hardware, false);
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });

  it("resetAdapter clears the singleton so next call re-initialises", () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const first = getAdapter();
    resetAdapter();
    const second = getAdapter();
    // Both should be SW adapters (env is still set), but they are different instances.
    assert.notEqual(first, second);
    assert.equal(second.info.backend, "software");
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });
});

// ---------------------------------------------------------------------------
// SoftwareAdapter — probeImageDimensions
// ---------------------------------------------------------------------------

describe("AccelAdapter.probeImageDimensions", () => {
  it("returns correct dimensions for tiny.png (16×16)", async () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const adapter = getAdapter();
    const dims = await adapter.probeImageDimensions(PNG);
    assert.equal(dims.width, 16);
    assert.equal(dims.height, 16);
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });

  it("throws when the file does not exist", async () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const adapter = getAdapter();
    await assert.rejects(() => adapter.probeImageDimensions("/nonexistent/image.png"));
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });
});

// ---------------------------------------------------------------------------
// SoftwareAdapter — probeBufferDimensions
// ---------------------------------------------------------------------------

describe("AccelAdapter.probeBufferDimensions", () => {
  it("returns correct dimensions for a JPEG buffer", async () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const adapter = getAdapter();
    const jpeg = await makeJpegBuffer();
    const dims = await adapter.probeBufferDimensions(jpeg);
    assert.equal(dims.width, 16);
    assert.equal(dims.height, 16);
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });

  it("throws for an invalid buffer", async () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const adapter = getAdapter();
    await assert.rejects(() => adapter.probeBufferDimensions(Buffer.from("not an image")));
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });
});

// ---------------------------------------------------------------------------
// SoftwareAdapter — resizeJpeg
// ---------------------------------------------------------------------------

describe("AccelAdapter.resizeJpeg", () => {
  it("cover: resizes JPEG to target dimensions (32×32 from 16×16)", async () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const adapter = getAdapter();
    const jpeg = await makeJpegBuffer();
    const result = await adapter.resizeJpeg(jpeg, 32, 32, "q82", "cover");
    // Result must start with JPEG magic bytes.
    assert.equal(result[0], 0xff);
    assert.equal(result[1], 0xd8);
    // Verify output dimensions via probeBufferDimensions.
    const dims = await adapter.probeBufferDimensions(result);
    assert.equal(dims.width, 32);
    assert.equal(dims.height, 32);
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });

  it("contain: pads to 32×32 with black letterbox", async () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const adapter = getAdapter();
    const jpeg = await makeJpegBuffer();
    const result = await adapter.resizeJpeg(jpeg, 32, 32, "q82", "contain");
    assert.equal(result[0], 0xff);
    assert.equal(result[1], 0xd8);
    const dims = await adapter.probeBufferDimensions(result);
    assert.equal(dims.width, 32);
    assert.equal(dims.height, 32);
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });

  it("returns a JPEG buffer smaller than 1 MB for a 480px thumbnail", async () => {
    resetAdapter();
    process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] = "1";
    const adapter = getAdapter();
    const jpeg = await makeJpegBuffer();
    const result = await adapter.resizeJpeg(jpeg, 480, 270, "q75", "cover");
    // 480×270 JPEG should never exceed 1 MB for a test fixture.
    assert.ok(result.byteLength < 1024 * 1024);
    delete process.env["MEDIA_UNDERSTANDING_DISABLE_HW"];
    resetAdapter();
  });
});
