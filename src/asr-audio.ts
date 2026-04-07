/**
 * PCM audio extraction from media files via node-av.
 *
 * Decodes the audio stream of any media file and resamples to
 * 16 kHz mono Float32 PCM — the format required by Parakeet mel spectrogram.
 *
 * Uses `SoftwareResampleContext` (ffmpeg's libswresample) for high-quality
 * sample rate conversion + channel downmixing + format conversion in one pass.
 *
 * NOTE: We use the buffer-based `convertSync()` API rather than the frame-based
 * `convertFrame()` because the latter is broken in node-av 5.2.2 (returns
 * AVERROR for all frames despite correct init). `convertSync` works correctly.
 */

import { Decoder, Demuxer } from "node-av/api";
import { AV_SAMPLE_FMT_FLT, AV_CHANNEL_ORDER_NATIVE } from "node-av/constants";
import type { AVSampleFormat } from "node-av/constants";
import { SoftwareResampleContext } from "node-av/lib";
import type { ChannelLayout } from "node-av/lib";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Parakeet expects 16 kHz mono input. */
const TARGET_SAMPLE_RATE = 16_000;

/** Mono channel layout: native order, 1 channel, front-center (mask=0x4). */
const MONO_LAYOUT: ChannelLayout = {
  nbChannels: 1,
  order: AV_CHANNEL_ORDER_NATIVE,
  mask: 4n, // AV_CH_FRONT_CENTER
};

/**
 * Extra output samples to allocate beyond the ratio-based estimate.
 * Accounts for resampler filter delay + rounding.
 */
const OUTPUT_PADDING = 32;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Extract Float32 samples from a Buffer into an owned Float32Array.
 *
 * Creates a zero-copy view over the Buffer's underlying ArrayBuffer, then
 * `.slice()` to produce an owned copy (native memcpy — much faster than
 * per-element `readFloatLE`). The slice is necessary because the Buffer may
 * be reused or GC'd.
 */
function bufferToFloat32(buf: Buffer, count: number): Float32Array {
  return new Float32Array(buf.buffer, buf.byteOffset, count).slice();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Extract 16 kHz mono Float32 PCM from a media file.
 *
 * Opens the file, finds the first audio stream, decodes it, and resamples
 * to the target format in one pass. Returns the complete PCM buffer.
 *
 * @param filePath Absolute path to audio/video file.
 * @returns Float32Array of mono 16 kHz PCM samples.
 * @throws If no audio stream is found.
 */
export async function extractPcm(filePath: string): Promise<Float32Array> {
  await using demuxer = await Demuxer.open(filePath);
  const audioStream = demuxer.audio();
  if (!audioStream) {
    throw new Error(`No audio stream found in: ${filePath}`);
  }

  using decoder = await Decoder.create(audioStream);

  // Resampler will be lazily initialized on first decoded frame
  // (to pick up actual input format from the decoder).
  let resampler: SoftwareResampleContext | null = null;
  let inputRate = 0;
  const chunks: Float32Array[] = [];
  let totalSamples = 0;

  for await (const frame of decoder.frames(demuxer.packets(audioStream.index))) {
    if (frame === null) break; // EOF

    // Lazy-init resampler on first decoded frame
    if (!resampler) {
      resampler = new SoftwareResampleContext();
      inputRate = frame.sampleRate;

      const inLayout: ChannelLayout = frame.channelLayout ?? {
        nbChannels: frame.channels ?? 1,
        order: AV_CHANNEL_ORDER_NATIVE,
        mask: (frame.channels ?? 1) >= 2 ? 3n : 4n,
      };
      const initRet = resampler.allocSetOpts2(
        MONO_LAYOUT,
        AV_SAMPLE_FMT_FLT,
        TARGET_SAMPLE_RATE,
        inLayout,
        frame.format as AVSampleFormat,
        frame.sampleRate,
      );
      if (initRet < 0) throw new Error(`Resampler allocSetOpts2 failed: ${initRet}`);
      const initRet2 = resampler.init();
      if (initRet2 < 0) throw new Error(`Resampler init failed: ${initRet2}`);
    }

    // Extract input buffer from the first data plane
    const inBuf = frame.data?.[0];
    if (!inBuf) continue;

    // Convert via buffer-based API: input plane → output mono FLT buffer
    const maxOut = Math.ceil((frame.nbSamples * TARGET_SAMPLE_RATE) / inputRate) + OUTPUT_PADDING;
    const outBuf = Buffer.alloc(maxOut * 4); // Float32 = 4 bytes per sample
    const n = resampler.convertSync([outBuf], maxOut, [inBuf], frame.nbSamples);

    if (n > 0) {
      chunks.push(bufferToFloat32(outBuf, n));
      totalSamples += n;
    }
  }

  // Flush resampler (drain buffered samples from the filter delay line)
  if (resampler) {
    const flushBuf = Buffer.alloc(1024 * 4);
    const n = resampler.convertSync([flushBuf], 1024, null, 0);
    if (n > 0) {
      chunks.push(bufferToFloat32(flushBuf, n));
      totalSamples += n;
    }
    resampler[Symbol.dispose]();
  }

  // Concatenate all chunks into a single Float32Array
  if (chunks.length === 0) return new Float32Array(0);
  if (chunks.length === 1) return chunks[0]!;

  const result = new Float32Array(totalSamples);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}
