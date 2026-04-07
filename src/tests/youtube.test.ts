/**
 * Tests for youtube.ts — yt-dlp integration layer.
 *
 * Unit tests (pure functions) always run. Integration tests that require
 * yt-dlp skip automatically when yt-dlp is not installed, following the
 * same pattern as SKIP_TRANSCRIPTION for ASR tests.
 */

import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { isUrl, urlHash, parseSubtitlesToSegments, hasYtDlp, getVideoInfo } from "../youtube.js";
import { handleFetchYtdlp, handleGetTranscript } from "../mcp-handlers.js";
import { MediaError } from "../types.js";

// ---------------------------------------------------------------------------
// Unit tests — pure functions, always run
// ---------------------------------------------------------------------------

describe("isUrl", () => {
  it("returns true for https URLs", () => {
    assert.equal(isUrl("https://youtube.com/watch?v=abc"), true);
  });

  it("returns true for http URLs", () => {
    assert.equal(isUrl("http://example.com"), true);
  });

  it("trims whitespace before checking", () => {
    assert.equal(isUrl("  https://youtube.com  "), true);
  });

  it("is case-insensitive", () => {
    assert.equal(isUrl("HTTPS://EXAMPLE.COM"), true);
    assert.equal(isUrl("Http://Example.Com"), true);
  });

  it("returns false for file paths", () => {
    assert.equal(isUrl("/path/to/file.mp4"), false);
    assert.equal(isUrl("relative/path.mp4"), false);
    assert.equal(isUrl("file.mp4"), false);
  });

  it("returns false for empty string", () => {
    assert.equal(isUrl(""), false);
  });

  it("returns false for ftp URLs", () => {
    assert.equal(isUrl("ftp://example.com"), false);
  });

  it("returns true for Instagram Reel URL", () => {
    assert.equal(isUrl("https://www.instagram.com/reel/DWzLXpixFF4/"), true);
  });

  it("returns true for various social media URLs", () => {
    assert.equal(isUrl("https://www.tiktok.com/@user/video/123"), true);
    assert.equal(isUrl("https://x.com/user/status/123"), true);
    assert.equal(isUrl("https://vimeo.com/123456"), true);
  });
});

describe("urlHash", () => {
  it("returns a 16-character hex string", () => {
    const hash = urlHash("https://example.com");
    assert.equal(hash.length, 16);
    assert.match(hash, /^[0-9a-f]{16}$/);
  });

  it("is deterministic", () => {
    const url = "https://youtube.com/watch?v=abc";
    assert.equal(urlHash(url), urlHash(url));
  });

  it("produces different hashes for different URLs", () => {
    assert.notEqual(
      urlHash("https://youtube.com/watch?v=abc"),
      urlHash("https://youtube.com/watch?v=xyz"),
    );
  });

  it("trims whitespace before hashing", () => {
    assert.equal(urlHash("  https://example.com  "), urlHash("https://example.com"));
  });
});

describe("parseSubtitlesToSegments", () => {
  let tmpDir: string;

  // Create a fresh temp dir for each describe block
  const writeTempFile = (name: string, content: string): string => {
    if (!tmpDir) tmpDir = mkdtempSync(join(tmpdir(), "mu-test-"));
    const path = join(tmpDir, name);
    writeFileSync(path, content, "utf8");
    return path;
  };

  // Cleanup temp dir after all tests in this describe
  it("cleanup", { skip: false }, () => {
    if (tmpDir) rmSync(tmpDir, { recursive: true, force: true });
  });

  it("parses a standard SRT file into segments", () => {
    const srt = `1
00:00:01,000 --> 00:00:03,500
Hello world

2
00:00:04,000 --> 00:00:06,000
Second line
`;
    const path = writeTempFile("test.srt", srt);
    const segments = parseSubtitlesToSegments(path);

    assert.equal(segments.length, 2);
    assert.equal(segments[0]!.start, 1000);
    assert.equal(segments[0]!.end, 3500);
    assert.equal(segments[0]!.text, "Hello world");
    assert.equal(segments[1]!.start, 4000);
    assert.equal(segments[1]!.end, 6000);
    assert.equal(segments[1]!.text, "Second line");
  });

  it("handles SRT with period separators", () => {
    const srt = `1
00:00:01.000 --> 00:00:03.500
Period separated
`;
    const path = writeTempFile("period.srt", srt);
    const segments = parseSubtitlesToSegments(path);

    assert.equal(segments.length, 1);
    assert.equal(segments[0]!.start, 1000);
    assert.equal(segments[0]!.end, 3500);
  });

  it("strips HTML tags from subtitle text", () => {
    const srt = `1
00:00:01,000 --> 00:00:03,000
<i>Italicized text</i>

2
00:00:04,000 --> 00:00:06,000
<b>Bold</b> and <font color="red">colored</font>
`;
    const path = writeTempFile("html.srt", srt);
    const segments = parseSubtitlesToSegments(path);

    assert.equal(segments[0]!.text, "Italicized text");
    assert.equal(segments[1]!.text, "Bold and colored");
  });

  it("joins multi-line cue text with spaces", () => {
    const srt = `1
00:00:01,000 --> 00:00:05,000
First line
second line
third line
`;
    const path = writeTempFile("multiline.srt", srt);
    const segments = parseSubtitlesToSegments(path);

    assert.equal(segments.length, 1);
    assert.ok(segments[0]!.text.includes("First line") && segments[0]!.text.includes("third line"));
  });

  it("parses a standard VTT file", () => {
    const vtt = `WEBVTT

00:00:01.000 --> 00:00:03.500
Hello from VTT

00:00:04.000 --> 00:00:06.000
Second cue
`;
    const path = writeTempFile("test.vtt", vtt);
    const segments = parseSubtitlesToSegments(path);

    assert.equal(segments.length, 2);
    assert.equal(segments[0]!.start, 1000);
    assert.equal(segments[0]!.end, 3500);
    assert.equal(segments[0]!.text, "Hello from VTT");
  });

  it("handles VTT with STYLE and NOTE blocks", () => {
    const vtt = `WEBVTT

STYLE
::cue {
  color: white;
}

NOTE This is a comment

00:00:01.000 --> 00:00:03.000
Actual content
`;
    const path = writeTempFile("styled.vtt", vtt);
    const segments = parseSubtitlesToSegments(path);

    assert.equal(segments.length, 1);
    assert.equal(segments[0]!.text, "Actual content");
  });

  it("returns empty array for SRT with no cues", () => {
    const path = writeTempFile("empty.srt", "");
    const segments = parseSubtitlesToSegments(path);
    assert.equal(segments.length, 0);
  });

  it("returns single segment for unsupported formats (.ass)", () => {
    const content =
      "[Script Info]\nTitle: Test\n\n[Events]\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello";
    const path = writeTempFile("test.ass", content);
    const segments = parseSubtitlesToSegments(path);

    assert.equal(segments.length, 1);
    assert.equal(segments[0]!.start, 0);
    assert.equal(segments[0]!.end, 0);
    assert.ok(segments[0]!.text.includes("[Script Info]"));
  });

  it("handles hour-long timestamps correctly", () => {
    const srt = `1
01:30:45,123 --> 01:30:50,456
Long video segment
`;
    const path = writeTempFile("long.srt", srt);
    const segments = parseSubtitlesToSegments(path);

    assert.equal(segments.length, 1);
    // 1h 30m 45s 123ms = 5445123ms
    assert.equal(segments[0]!.start, 5445123);
    assert.equal(segments[0]!.end, 5450456);
  });
});

// ---------------------------------------------------------------------------
// Integration tests — require yt-dlp, skip in CI
// ---------------------------------------------------------------------------

const SKIP_YTDLP = !hasYtDlp();

describe("yt-dlp integration", { skip: SKIP_YTDLP ? "yt-dlp not installed" : false }, () => {
  it("hasYtDlp returns true", () => {
    assert.equal(hasYtDlp(), true);
  });

  it("getVideoInfo fetches metadata for a YouTube video", { timeout: 30_000 }, async () => {
    // "Me at the zoo" — first YouTube video ever, short and stable
    const info = await getVideoInfo("https://www.youtube.com/watch?v=jNQXAC9IVRw");
    assert.ok(info.title.length > 0);
    assert.ok(info.duration > 0);
    assert.equal(typeof info.id, "string");
    assert.ok(info.id.length > 0);
  });

  it("handleFetchYtdlp returns metadata for a YouTube video", { timeout: 60_000 }, async () => {
    const result = await handleFetchYtdlp({
      url: "https://www.youtube.com/watch?v=jNQXAC9IVRw",
    });
    assert.ok(!("isError" in result));
    assert.ok(result.content.length > 0);
    const text = result.content[0];
    assert.ok(text && text.type === "text" && text.text.includes("Title:"));
  });

  it("handleGetTranscript works with an Instagram Reel URL", { timeout: 120_000 }, async () => {
    const result = await handleGetTranscript({
      file_path: "https://www.instagram.com/reel/DWzLXpixFF4/",
    });
    // Instagram reels may or may not have subtitles — either a transcript
    // or "No speech detected" is a valid success response
    assert.ok(!("isError" in result), "Should not return an MCP error");
    assert.ok(result.content.length > 0, "Should return content");
  });

  it("handleFetchYtdlp works with an Instagram Reel URL", { timeout: 60_000 }, async () => {
    const result = await handleFetchYtdlp({
      url: "https://www.instagram.com/reel/DWzLXpixFF4/",
    });
    assert.ok(!("isError" in result), "Should not return an MCP error");
    assert.ok(result.content.length > 0, "Should return content");
    const text = result.content[0];
    assert.ok(text && text.type === "text" && text.text.includes("Title:"));
  });

  // -------------------------------------------------------------------------
  // Platform matrix: TikTok, Vimeo, X/Twitter
  // These platforms may require auth or geo-restrict content, so we catch
  // YT_DLP_FAILED errors and skip gracefully rather than hard-failing.
  // -------------------------------------------------------------------------

  it("getVideoInfo fetches metadata for a Vimeo video", { timeout: 30_000 }, async () => {
    try {
      // Vimeo staff pick — stable, public, short
      const info = await getVideoInfo("https://vimeo.com/148751763");
      assert.ok(info.title.length > 0, "Vimeo title should be non-empty");
      assert.ok(info.duration > 0, "Vimeo duration should be > 0");
    } catch (err) {
      if (err instanceof MediaError && err.code === "YT_DLP_FAILED") {
        console.log("  ⏭ Vimeo getVideoInfo skipped: yt-dlp could not fetch (auth/geo?)");
        return;
      }
      throw err;
    }
  });

  it("handleFetchYtdlp works with a Vimeo URL", { timeout: 60_000 }, async () => {
    const result = await handleFetchYtdlp({
      url: "https://vimeo.com/148751763",
    });
    if ("isError" in result) {
      // Graceful skip — Vimeo may require auth or geo-restrict
      console.log("  ⏭ Vimeo handleFetchYtdlp skipped: MCP error returned");
      return;
    }
    assert.ok(result.content.length > 0, "Should return content");
    const text = result.content[0];
    assert.ok(text && text.type === "text" && text.text.includes("Title:"));
  });

  it("getVideoInfo fetches metadata for a TikTok video", { timeout: 30_000 }, async () => {
    try {
      // Public TikTok — Khaby Lame (large account, stable content)
      const info = await getVideoInfo(
        "https://www.tiktok.com/@khaboringlife/video/7487775559832805633",
      );
      assert.ok(info.title.length > 0 || info.id.length > 0, "TikTok should return metadata");
    } catch (err) {
      if (err instanceof MediaError && err.code === "YT_DLP_FAILED") {
        console.log("  ⏭ TikTok getVideoInfo skipped: yt-dlp could not fetch (auth/geo?)");
        return;
      }
      throw err;
    }
  });

  it("handleFetchYtdlp works with a TikTok URL", { timeout: 60_000 }, async () => {
    try {
      const result = await handleFetchYtdlp({
        url: "https://www.tiktok.com/@khaboringlife/video/7487775559832805633",
      });
      if ("isError" in result) {
        console.log("  ⏭ TikTok handleFetchYtdlp skipped: MCP error returned");
        return;
      }
      assert.ok(result.content.length > 0, "Should return content");
    } catch (err) {
      if (err instanceof MediaError && err.code === "YT_DLP_FAILED") {
        console.log("  ⏭ TikTok handleFetchYtdlp skipped: yt-dlp failed (auth/geo?)");
        return;
      }
      throw err;
    }
  });

  it("getVideoInfo fetches metadata for an X/Twitter video", { timeout: 30_000 }, async () => {
    try {
      // NASA tweet with video — public, stable
      const info = await getVideoInfo("https://x.com/NASA/status/1893710191948685363");
      assert.ok(info.title.length > 0 || info.id.length > 0, "X/Twitter should return metadata");
    } catch (err) {
      if (err instanceof MediaError && err.code === "YT_DLP_FAILED") {
        console.log("  ⏭ X/Twitter getVideoInfo skipped: yt-dlp could not fetch (auth/geo?)");
        return;
      }
      throw err;
    }
  });

  it("handleFetchYtdlp works with an X/Twitter URL", { timeout: 60_000 }, async () => {
    try {
      const result = await handleFetchYtdlp({
        url: "https://x.com/NASA/status/1893710191948685363",
      });
      if ("isError" in result) {
        console.log("  ⏭ X/Twitter handleFetchYtdlp skipped: MCP error returned");
        return;
      }
      assert.ok(result.content.length > 0, "Should return content");
    } catch (err) {
      if (err instanceof MediaError && err.code === "YT_DLP_FAILED") {
        console.log("  ⏭ X/Twitter handleFetchYtdlp skipped: yt-dlp failed (auth/geo?)");
        return;
      }
      throw err;
    }
  });
});
