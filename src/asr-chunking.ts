/**
 * Long-form audio chunking for Parakeet ASR.
 *
 * Parakeet's FastConformer encoder expects fixed-size input. For audio longer
 * than ~30 seconds, we split into overlapping windows, transcribe each, and
 * stitch the resulting segments together with overlap deduplication.
 *
 * Strategy:
 *   - Window size: 30 seconds
 *   - Overlap: 2 seconds (reduces boundary artifacts)
 *   - Stitching: for overlapping regions, prefer the segment from the window
 *     where it's farther from the edge (less likely to be cut off)
 *
 * The 30s window is a sweet spot — Parakeet handles it well, and the
 * encoder context is large enough for good accuracy. Shorter windows
 * increase boundary artifacts; longer ones hit memory limits.
 */

import type { Segment } from "./types.js";
import { transcribePcm } from "./asr-pipeline.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Parakeet expects 16 kHz mono input. */
const SAMPLE_RATE = 16_000;

/** Window size in seconds. */
const WINDOW_SEC = 30;

/** Overlap between adjacent windows in seconds. */
const OVERLAP_SEC = 2;

/** Window size in samples. */
const WINDOW_SAMPLES = WINDOW_SEC * SAMPLE_RATE;

/** Overlap in samples. */
const OVERLAP_SAMPLES = OVERLAP_SEC * SAMPLE_RATE;

/** Step size in samples (window minus overlap). */
const STEP_SAMPLES = WINDOW_SAMPLES - OVERLAP_SAMPLES;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Transcribe PCM audio of any length, chunking if necessary.
 *
 * For audio ≤ WINDOW_SEC this is a simple pass-through to `transcribePcm()`.
 * For longer audio, it splits into overlapping windows, transcribes each,
 * and stitches the results.
 *
 * @param pcm Mono Float32 PCM at 16 kHz.
 * @returns Stitched `Segment[]` with millisecond timestamps.
 */
export async function transcribeLongForm(pcm: Float32Array): Promise<Segment[]> {
  if (pcm.length <= WINDOW_SAMPLES) {
    // Short audio — single pass
    return transcribePcm(pcm, 0);
  }

  // Split into windows
  const windows = computeWindows(pcm.length);
  const allSegments: Segment[][] = [];

  for (const win of windows) {
    const chunk = pcm.subarray(win.startSample, win.endSample);
    const timeOffset = win.startSample / SAMPLE_RATE;
    const segments = await transcribePcm(chunk, timeOffset);
    allSegments.push(segments);
  }

  return stitchSegments(allSegments, windows);
}

// ---------------------------------------------------------------------------
// Window computation
// ---------------------------------------------------------------------------

interface Window {
  /** Start sample index (inclusive). */
  startSample: number;
  /** End sample index (exclusive). */
  endSample: number;
  /** Start time in milliseconds. */
  startMs: number;
  /** End time in milliseconds. */
  endMs: number;
}

function computeWindows(totalSamples: number): Window[] {
  const windows: Window[] = [];
  let start = 0;

  while (start < totalSamples) {
    const end = Math.min(start + WINDOW_SAMPLES, totalSamples);
    windows.push({
      startSample: start,
      endSample: end,
      startMs: Math.round((start / SAMPLE_RATE) * 1000),
      endMs: Math.round((end / SAMPLE_RATE) * 1000),
    });

    if (end >= totalSamples) break;
    start += STEP_SAMPLES;
  }

  return windows;
}

// ---------------------------------------------------------------------------
// Segment stitching
// ---------------------------------------------------------------------------

/**
 * Stitch overlapping segment arrays into a single timeline.
 *
 * For each pair of adjacent windows, there's a 2-second overlap region.
 * Within that region:
 *   - Segments from the left window that end before the midpoint of the
 *     overlap are kept.
 *   - Segments from the right window that start at or after the midpoint
 *     are kept.
 *   - Segments that span the midpoint are assigned to whichever window
 *     has more of the segment.
 *
 * This avoids duplicating words that appear in both windows' transcripts.
 */
function stitchSegments(allSegments: Segment[][], windows: Window[]): Segment[] {
  if (allSegments.length === 0) return [];
  if (allSegments.length === 1) return allSegments[0]!;

  const result: Segment[] = [];

  for (let i = 0; i < allSegments.length; i++) {
    const segments = allSegments[i]!;
    const win = windows[i]!;

    // Determine the boundaries for this window's contribution
    // Left boundary: midpoint of overlap with previous window
    const leftBoundaryMs = i === 0 ? 0 : Math.round((win.startMs + windows[i - 1]!.endMs) / 2);

    // Right boundary: midpoint of overlap with next window
    const rightBoundaryMs =
      i === allSegments.length - 1
        ? Infinity
        : Math.round((win.endMs + windows[i + 1]!.startMs) / 2);

    for (const seg of segments) {
      // Segment midpoint determines which window "owns" it
      const segMid = (seg.start + seg.end) / 2;

      if (segMid >= leftBoundaryMs && segMid < rightBoundaryMs) {
        result.push(seg);
      }
    }
  }

  return result;
}
