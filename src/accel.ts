/**
 * Hardware acceleration adapter layer.
 *
 * Provides an `AccelAdapter` interface with concrete implementations for:
 *   - SoftwareAdapter  — pure CPU path (FFmpeg filter + MJPEG encoder)
 *   - VideoToolboxAdapter — macOS GPU (scale + MJPEG via VideoToolbox)
 *   - CudaAdapter      — NVIDIA GPU (scale_cuda + SW MJPEG; no HW MJPEG on NVENC)
 *   - VaapiAdapter     — Linux VA-API (scale_vaapi + HW MJPEG)
 *   - QsvAdapter       — Intel QSV (scale_qsv + HW MJPEG)
 *
 * Heavy operations (frame scaling/encoding, decoding) use HW paths.
 * "Light" operations (resizeJpeg for grid tiles, probeImageDimensions) always
 * delegate to sharp — the GPU overhead isn't worth it for small thumbnails.
 *
 * Factory: `getAdapter()` returns a process-level singleton. It calls
 * `HardwareContext.auto()` once; result is cached. Set
 * `MEDIA_UNDERSTANDING_DISABLE_HW=1` to force the software path.
 *
 * Design invariant: media.ts must NOT import HardwareContext, FilterAPI,
 * Encoder, or HW codec constants directly. All such details live here.
 */

import sharp from "sharp";
import { Decoder, Encoder, FilterAPI, HardwareContext } from "node-av/api";
import {
  AVCOL_RANGE_JPEG,
  AV_CODEC_FLAG_QSCALE,
  FF_ENCODER_MJPEG,
  FF_ENCODER_MJPEG_VAAPI,
  FF_ENCODER_MJPEG_VIDEOTOOLBOX,
} from "node-av/constants";
import type { FFVideoEncoder } from "node-av/constants";
import type { Frame, Stream } from "node-av/lib";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** JPEG quality level mapped to an MJPEG qscale value (lower = better). */
export type JpegQuality = "q75" | "q82" | "q85";

/** Metadata about the active acceleration backend. */
export interface AccelInfo {
  /** Human-readable backend name, e.g. "software", "videotoolbox", "cuda". */
  readonly backend: string;
  /** Whether hardware acceleration is active. */
  readonly hardware: boolean;
}

/**
 * Acceleration adapter interface.
 *
 * Each method is called per-frame or per-operation; implementations should
 * avoid holding long-lived state where possible (Encoder/Filter objects that
 * need re-creation per call are fine for our single-frame-at-a-time use case).
 */
export interface AccelAdapter {
  readonly info: AccelInfo;

  /**
   * Create a Decoder for the given video stream.
   * HW adapters pass their HardwareContext; SW adapter uses default.
   */
  createDecoder(stream: Stream): Promise<Decoder>;

  /**
   * Scale a raw video Frame to `targetWidth` and encode it as JPEG.
   * The output buffer is a self-contained JPEG (not raw RGB).
   * Caller is responsible for calling `frame.free()` after this returns.
   */
  scaleAndEncode(frame: Frame, targetWidth: number, quality: JpegQuality): Promise<Buffer>;

  /**
   * Resize + re-encode a JPEG Buffer to the given dimensions.
   * - "cover": crop-to-fill (no black bars)
   * - "contain": letterbox with black padding
   * Delegates to sharp on all adapters (not worth GPU for small tiles).
   */
  resizeJpeg(
    input: Buffer,
    width: number,
    height: number,
    quality: JpegQuality,
    fit: "cover" | "contain",
  ): Promise<Buffer>;

  /**
   * Return the pixel dimensions of an image file on disk.
   * Delegates to sharp on all adapters.
   */
  probeImageDimensions(filePath: string): Promise<{ width: number; height: number }>;

  /**
   * Return the pixel dimensions of a JPEG buffer in memory.
   * Delegates to sharp on all adapters.
   */
  probeBufferDimensions(input: Buffer): Promise<{ width: number; height: number }>;
}

// ---------------------------------------------------------------------------
// Quality mapping
// ---------------------------------------------------------------------------

/** Maps JpegQuality to MJPEG qscale (1-31, lower = better quality). */
const QSCALE: Record<JpegQuality, number> = {
  q85: 2,
  q82: 3,
  q75: 5,
};

/**
 * FF_QP2LAMBDA — the scale factor for global_quality in FFmpeg.
 * Not exported by node-av/constants, but its value is the well-known 118.
 */
const FF_QP2LAMBDA = 118;

// ---------------------------------------------------------------------------
// Shared sharp helpers (used by all adapters)
// ---------------------------------------------------------------------------

/** Resize + JPEG-encode a buffer using sharp. "cover" crops; "contain" letterboxes. */
async function sharpResizeJpeg(
  input: Buffer,
  width: number,
  height: number,
  quality: number,
  fit: "cover" | "contain",
): Promise<Buffer> {
  if (fit === "cover") {
    return sharp(input)
      .resize(width, height, { fit: "cover", position: "centre" })
      .jpeg({ quality })
      .toBuffer();
  }

  // "contain" — letterbox with black background
  return sharp({
    create: {
      width,
      height,
      channels: 3,
      background: { r: 0, g: 0, b: 0 },
    },
  })
    .composite([
      {
        input: await sharp(input)
          .resize(width, height, { fit: "contain", background: { r: 0, g: 0, b: 0 } })
          .jpeg({ quality })
          .toBuffer(),
        left: 0,
        top: 0,
      },
    ])
    .jpeg({ quality })
    .toBuffer();
}

/** Quality integer (0-100) from JpegQuality tag for sharp. */
function qualityInt(q: JpegQuality): number {
  return parseInt(q.slice(1), 10); // "q75" → 75
}

// ---------------------------------------------------------------------------
// Shared probe helpers (used by all adapters)
// ---------------------------------------------------------------------------

async function sharpProbeFile(filePath: string): Promise<{ width: number; height: number }> {
  const meta = await sharp(filePath).metadata();
  if (!meta.width || !meta.height) {
    throw new Error(`Cannot determine dimensions for image: ${filePath}`);
  }
  return { width: meta.width, height: meta.height };
}

async function sharpProbeBuffer(input: Buffer): Promise<{ width: number; height: number }> {
  const meta = await sharp(input).metadata();
  if (!meta.width || !meta.height) {
    throw new Error("Cannot determine dimensions for image buffer");
  }
  return { width: meta.width, height: meta.height };
}

// ---------------------------------------------------------------------------
// Shared MJPEG encode helper
// ---------------------------------------------------------------------------

/**
 * Encode a Frame to JPEG using the given FFmpeg encoder codec name
 * and quality level. Returns the raw JPEG bytes.
 *
 * This is a one-shot encode: create encoder → encodeAll(frame) → encodeAll(null) flush.
 * Per-frame encoder creation is intentional — we process one frame at a time.
 */
async function encodeFrameAsJpeg(
  frame: Frame,
  encoderCodec: FFVideoEncoder,
  quality: JpegQuality,
): Promise<Buffer> {
  // MJPEG requires full-range YUV (AVCOL_RANGE_JPEG).  Frames decoded from
  // real video files carry AVCOL_RANGE_MPEG (limited range); setting the
  // color range to JPEG before encoding makes the frame match the encoder's
  // expectation.  The filter already outputs yuvj420p (JPEG-tagged pixel
  // format), so this is semantically correct — we're producing a JPEG still.
  //
  // threadCount: 1 — MJPEG one-shot still encoding doesn't benefit from
  // frame-threaded encoding and it causes ff_frame_thread_encoder_init to fail.
  frame.colorRange = AVCOL_RANGE_JPEG;

  using encoder = await Encoder.create(encoderCodec, {
    threadCount: 1,
  });

  // Enable constant-quality mode (equivalent to -qscale:v N in FFmpeg CLI).
  // Must be set before first frame is sent (lazy init fires on encodeAll).
  encoder.setCodecFlags(AV_CODEC_FLAG_QSCALE);
  const ctx = encoder.getCodecContext();
  if (ctx) {
    ctx.globalQuality = QSCALE[quality] * FF_QP2LAMBDA;
  }

  // encodeAll(frame): send + drain. MJPEG produces one packet per frame.
  const framePackets = await encoder.encodeAll(frame);
  // encodeAll(null): flush any buffered packets.
  const flushPackets = await encoder.encodeAll(null);

  const allPackets = [...framePackets, ...flushPackets];
  for (const pkt of allPackets) {
    const data = pkt.data;
    pkt.free();
    if (data) return Buffer.from(data); // copy before packet is freed
  }

  throw new Error("MJPEG encoder produced no packet");
}

// ---------------------------------------------------------------------------
// Shared scale helper — uses FilterAPI.processAll (push+drain in one call)
// ---------------------------------------------------------------------------

/**
 * Scale a Frame to `outW x outH` + convert to yuv420p via FilterAPI,
 * then encode the first result frame with MJPEG and free it.
 * Caller must free the input `frame` separately.
 *
 * When `hwCtx` is provided the filter graph runs in HW context.
 * When the input frame is HW-resident (hwFramesCtx != null) and no `hwCtx`
 * is provided (SW fallback path), `hwdownload` is prepended to the graph so
 * FFmpeg can transfer the frame to system memory before scaling.
 */
async function scaleAndEncodeViaFilter(
  frame: Frame,
  outW: number,
  outH: number,
  encoderCodec: FFVideoEncoder,
  quality: JpegQuality,
  hwCtx?: HardwareContext,
): Promise<Buffer> {
  // yuvj420p is the "JPEG-tagged" variant of yuv420p — it carries full-range
  // color semantics (AVCOL_RANGE_JPEG) implicitly so MJPEG accepts it without
  // needing strict compliance relaxation.
  const scaleFilter = `scale=${outW}:${outH}:flags=bilinear,format=yuvj420p`;
  // If the frame lives in GPU memory and we have no HW context (SW fallback),
  // prepend hwdownload so FFmpeg transfers it to system memory first.
  const needsDownload = !hwCtx && !!frame.hwFramesCtx;
  const filterStr = needsDownload ? `hwdownload,format=yuv420p,${scaleFilter}` : scaleFilter;

  using filter = FilterAPI.create(filterStr, {
    hardware: hwCtx ?? null,
  });

  const scaled = await filter.processAll(frame);
  if (scaled.length === 0) {
    throw new Error("FilterAPI.processAll returned no frames during scale");
  }

  const scaledFrame = scaled[0]!;
  try {
    return await encodeFrameAsJpeg(scaledFrame, encoderCodec, quality);
  } finally {
    for (const f of scaled) f.free();
  }
}

// ---------------------------------------------------------------------------
// SoftwareAdapter
// ---------------------------------------------------------------------------

class SoftwareAdapter implements AccelAdapter {
  readonly info: AccelInfo = { backend: "software", hardware: false };

  async createDecoder(stream: Stream): Promise<Decoder> {
    return Decoder.create(stream);
  }

  async scaleAndEncode(frame: Frame, targetWidth: number, quality: JpegQuality): Promise<Buffer> {
    const srcW = frame.width;
    const srcH = frame.height;
    const outW = Math.min(srcW, targetWidth);
    const outH = Math.max(1, Math.round((outW * srcH) / srcW));
    return scaleAndEncodeViaFilter(frame, outW, outH, FF_ENCODER_MJPEG, quality);
  }

  async resizeJpeg(
    input: Buffer,
    width: number,
    height: number,
    quality: JpegQuality,
    fit: "cover" | "contain",
  ): Promise<Buffer> {
    return sharpResizeJpeg(input, width, height, qualityInt(quality), fit);
  }

  async probeImageDimensions(filePath: string): Promise<{ width: number; height: number }> {
    return sharpProbeFile(filePath);
  }

  async probeBufferDimensions(input: Buffer): Promise<{ width: number; height: number }> {
    return sharpProbeBuffer(input);
  }
}

// ---------------------------------------------------------------------------
// VideoToolboxAdapter  (macOS)
// ---------------------------------------------------------------------------

class VideoToolboxAdapter implements AccelAdapter {
  readonly info: AccelInfo = { backend: "videotoolbox", hardware: true };
  private hw: HardwareContext;

  constructor(hw: HardwareContext) {
    this.hw = hw;
  }

  async createDecoder(stream: Stream): Promise<Decoder> {
    return Decoder.create(stream, { hardware: this.hw });
  }

  async scaleAndEncode(frame: Frame, targetWidth: number, quality: JpegQuality): Promise<Buffer> {
    const srcW = frame.width;
    const srcH = frame.height;
    const outW = Math.min(srcW, targetWidth);
    const outH = Math.max(1, Math.round((outW * srcH) / srcW));
    try {
      // VideoToolbox HW MJPEG encoder can accept yuv420p directly.
      return await scaleAndEncodeViaFilter(
        frame,
        outW,
        outH,
        FF_ENCODER_MJPEG_VIDEOTOOLBOX,
        quality,
        this.hw,
      );
    } catch {
      // VideoToolbox MJPEG may not be available on all macOS versions / frame
      // formats.  Use pure-SW path; scaleAndEncodeViaFilter will prepend
      // hwdownload if the frame is HW-resident.
      return await scaleAndEncodeViaFilter(frame, outW, outH, FF_ENCODER_MJPEG, quality);
    }
  }

  async resizeJpeg(
    input: Buffer,
    width: number,
    height: number,
    quality: JpegQuality,
    fit: "cover" | "contain",
  ): Promise<Buffer> {
    return sharpResizeJpeg(input, width, height, qualityInt(quality), fit);
  }

  async probeImageDimensions(filePath: string): Promise<{ width: number; height: number }> {
    return sharpProbeFile(filePath);
  }

  async probeBufferDimensions(input: Buffer): Promise<{ width: number; height: number }> {
    return sharpProbeBuffer(input);
  }
}

// ---------------------------------------------------------------------------
// CudaAdapter  (NVIDIA)
// ---------------------------------------------------------------------------

class CudaAdapter implements AccelAdapter {
  readonly info: AccelInfo = { backend: "cuda", hardware: true };
  private hw: HardwareContext;

  constructor(hw: HardwareContext) {
    this.hw = hw;
  }

  async createDecoder(stream: Stream): Promise<Decoder> {
    return Decoder.create(stream, { hardware: this.hw });
  }

  async scaleAndEncode(frame: Frame, targetWidth: number, quality: JpegQuality): Promise<Buffer> {
    const srcW = frame.width;
    const srcH = frame.height;
    const outW = Math.min(srcW, targetWidth);
    const outH = Math.max(1, Math.round((outW * srcH) / srcW));
    // CUDA: scale on GPU, format to yuv420p, then SW MJPEG (NVENC has no MJPEG encoder).
    // If the HW filter graph fails, fall back to pure SW path; scaleAndEncodeViaFilter
    // will prepend hwdownload automatically if the frame is HW-resident.
    try {
      return await scaleAndEncodeViaFilter(frame, outW, outH, FF_ENCODER_MJPEG, quality, this.hw);
    } catch {
      return await scaleAndEncodeViaFilter(frame, outW, outH, FF_ENCODER_MJPEG, quality);
    }
  }

  async resizeJpeg(
    input: Buffer,
    width: number,
    height: number,
    quality: JpegQuality,
    fit: "cover" | "contain",
  ): Promise<Buffer> {
    return sharpResizeJpeg(input, width, height, qualityInt(quality), fit);
  }

  async probeImageDimensions(filePath: string): Promise<{ width: number; height: number }> {
    return sharpProbeFile(filePath);
  }

  async probeBufferDimensions(input: Buffer): Promise<{ width: number; height: number }> {
    return sharpProbeBuffer(input);
  }
}

// ---------------------------------------------------------------------------
// VaapiAdapter  (Linux VA-API)
// ---------------------------------------------------------------------------

class VaapiAdapter implements AccelAdapter {
  readonly info: AccelInfo = { backend: "vaapi", hardware: true };
  private hw: HardwareContext;

  constructor(hw: HardwareContext) {
    this.hw = hw;
  }

  async createDecoder(stream: Stream): Promise<Decoder> {
    return Decoder.create(stream, { hardware: this.hw });
  }

  async scaleAndEncode(frame: Frame, targetWidth: number, quality: JpegQuality): Promise<Buffer> {
    const srcW = frame.width;
    const srcH = frame.height;
    const outW = Math.min(srcW, targetWidth);
    const outH = Math.max(1, Math.round((outW * srcH) / srcW));
    // VAAPI has a native MJPEG encoder; fall back to SW if unavailable.
    try {
      return await scaleAndEncodeViaFilter(
        frame,
        outW,
        outH,
        FF_ENCODER_MJPEG_VAAPI,
        quality,
        this.hw,
      );
    } catch {
      // Download HW frame to SW via hwdownload filter if needed, then SW MJPEG.
      return await scaleAndEncodeViaFilter(frame, outW, outH, FF_ENCODER_MJPEG, quality);
    }
  }

  async resizeJpeg(
    input: Buffer,
    width: number,
    height: number,
    quality: JpegQuality,
    fit: "cover" | "contain",
  ): Promise<Buffer> {
    return sharpResizeJpeg(input, width, height, qualityInt(quality), fit);
  }

  async probeImageDimensions(filePath: string): Promise<{ width: number; height: number }> {
    return sharpProbeFile(filePath);
  }

  async probeBufferDimensions(input: Buffer): Promise<{ width: number; height: number }> {
    return sharpProbeBuffer(input);
  }
}

// ---------------------------------------------------------------------------
// QsvAdapter  (Intel Quick Sync)
// ---------------------------------------------------------------------------

class QsvAdapter implements AccelAdapter {
  readonly info: AccelInfo = { backend: "qsv", hardware: true };
  private hw: HardwareContext;

  constructor(hw: HardwareContext) {
    this.hw = hw;
  }

  async createDecoder(stream: Stream): Promise<Decoder> {
    return Decoder.create(stream, { hardware: this.hw });
  }

  async scaleAndEncode(frame: Frame, targetWidth: number, quality: JpegQuality): Promise<Buffer> {
    const srcW = frame.width;
    const srcH = frame.height;
    const outW = Math.min(srcW, targetWidth);
    const outH = Math.max(1, Math.round((outW * srcH) / srcW));
    // QSV has no standard MJPEG encoder; use GPU scale + SW MJPEG encode.
    return scaleAndEncodeViaFilter(frame, outW, outH, FF_ENCODER_MJPEG, quality, this.hw);
  }

  async resizeJpeg(
    input: Buffer,
    width: number,
    height: number,
    quality: JpegQuality,
    fit: "cover" | "contain",
  ): Promise<Buffer> {
    return sharpResizeJpeg(input, width, height, qualityInt(quality), fit);
  }

  async probeImageDimensions(filePath: string): Promise<{ width: number; height: number }> {
    return sharpProbeFile(filePath);
  }

  async probeBufferDimensions(input: Buffer): Promise<{ width: number; height: number }> {
    return sharpProbeBuffer(input);
  }
}

// ---------------------------------------------------------------------------
// Factory — process-level singleton
// ---------------------------------------------------------------------------

/** Map from HardwareContext.deviceTypeName to adapter constructor. */
type AdapterFactory = (hw: HardwareContext) => AccelAdapter;

const adapterMap: Record<string, AdapterFactory> = {
  videotoolbox: (hw) => new VideoToolboxAdapter(hw),
  cuda: (hw) => new CudaAdapter(hw),
  vaapi: (hw) => new VaapiAdapter(hw),
  qsv: (hw) => new QsvAdapter(hw),
};

let _singleton: AccelAdapter | undefined;

/**
 * Return the process-level AccelAdapter singleton.
 *
 * On first call:
 *  1. If `MEDIA_UNDERSTANDING_DISABLE_HW=1`, returns SoftwareAdapter.
 *  2. Otherwise, calls `HardwareContext.auto()` and picks the matching
 *     HW adapter; falls back to SoftwareAdapter if none match.
 */
export function getAdapter(): AccelAdapter {
  if (_singleton) return _singleton;

  if (process.env["MEDIA_UNDERSTANDING_DISABLE_HW"] === "1") {
    _singleton = new SoftwareAdapter();
    return _singleton;
  }

  const hw = HardwareContext.auto();
  if (hw) {
    const name = hw.deviceTypeName as string;
    const factory = adapterMap[name];
    if (factory) {
      _singleton = factory(hw);
      return _singleton;
    }
    // Unknown HW type — dispose and fall through to SW.
    hw.dispose();
  }

  _singleton = new SoftwareAdapter();
  return _singleton;
}

/**
 * Reset the singleton (for test isolation).
 * Call this between tests that need to exercise different adapter paths.
 */
export function resetAdapter(): void {
  _singleton = undefined;
}

/**
 * Return a fresh SoftwareAdapter — always CPU, never cached.
 *
 * Use this for an explicit SW fallback when the HW path has already failed.
 * Does NOT touch the process-level singleton returned by getAdapter().
 */
export function getSoftwareAdapter(): AccelAdapter {
  return new SoftwareAdapter();
}

// Re-export types used by media.ts
export type { AccelAdapter as default };
