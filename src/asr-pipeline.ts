/**
 * ASR pipeline orchestrator — lazy session singleton, audio → Segment[].
 *
 * Ties together:
 *   - model-manager.ts  — download + cache Parakeet ONNX models
 *   - asr-accel.ts      — execution provider selection (CoreML/CUDA/DML/CPU)
 *   - nemo128.onnx      — mel spectrogram preprocessing (replaces asr-mel.ts)
 *   - asr-tokenizer.ts  — vocab loading + token decode
 *   - asr-decoder.ts    — TDT greedy decode loop
 *
 * The ONNX sessions are created lazily on first use and reused for the process
 * lifetime (they are expensive to create). The model is downloaded lazily on
 * first transcription call.
 *
 * Mel spectrogram preprocessing is handled by nemo128.onnx — the canonical
 * NeMo preprocessor from the same HuggingFace repo as the encoder/decoder.
 * This replaces the previous hand-rolled pure-JS FFT pipeline.
 *
 * Session EP split: the encoder uses the platform's best EP (CoreML/CUDA/DML/CPU),
 * while the preprocessor and decoder always use CPU. The decoder runs ~4,000
 * inference calls per file in the TDT greedy loop — GPU dispatch overhead
 * would dominate the actual computation for a model this small (18 MB).
 */

import { performance } from "node:perf_hooks";
import type { InferenceSession } from "onnxruntime-node";
import * as ort from "onnxruntime-node";

import type { Segment } from "./types.js";
import { ensureModel, getModelPaths } from "./model-manager.js";
import {
  getEncoderSessionOptions,
  getLightSessionOptions,
  detectActiveProvider,
  type AsrAccelInfo,
} from "./asr-accel.js";
import { loadTokenizer, type Tokenizer } from "./asr-tokenizer.js";
import {
  tdtGreedyDecode,
  transposeEncoder,
  FRAME_TIME_STRIDE,
  type DecodedToken,
} from "./asr-decoder.js";

// ---------------------------------------------------------------------------
// Debug timing
// ---------------------------------------------------------------------------

/** Enable per-stage timing logs to stderr. Set MEDIA_UNDERSTANDING_DEBUG=1. */
const DEBUG_TIMING = process.env["MEDIA_UNDERSTANDING_DEBUG"] === "1";

function logTiming(label: string, ms: number): void {
  if (DEBUG_TIMING) {
    process.stderr.write(`[asr-timing] ${label}: ${ms.toFixed(1)}ms\n`);
  }
}

// ---------------------------------------------------------------------------
// Lazy singleton
// ---------------------------------------------------------------------------

interface AsrSession {
  preprocessor: InferenceSession;
  encoder: InferenceSession;
  decoder: InferenceSession;
  tokenizer: Tokenizer;
  accel: AsrAccelInfo;
}

let sessionPromise: Promise<AsrSession> | undefined;

/**
 * Get (or create) the process-level ASR session singleton.
 * First call triggers model download if not cached.
 */
async function getSession(): Promise<AsrSession> {
  if (sessionPromise) return sessionPromise;

  sessionPromise = (async (): Promise<AsrSession> => {
    const t0 = performance.now();

    // 1. Ensure model files are downloaded
    await ensureModel();
    const paths = getModelPaths();

    // 2. Create ONNX sessions with split EP configs:
    //    - Encoder: best available EP (CoreML/CUDA/DML/CPU)
    //    - Preprocessor + Decoder: CPU only (too small for GPU dispatch overhead)
    const encoderOpts = getEncoderSessionOptions();
    const lightOpts = getLightSessionOptions();

    const [preprocessor, encoder, decoder, tokenizer] = await Promise.all([
      ort.InferenceSession.create(paths.preprocessor, lightOpts),
      ort.InferenceSession.create(paths.encoder, encoderOpts),
      ort.InferenceSession.create(paths.decoder, lightOpts),
      loadTokenizer(paths.vocab),
    ]);

    const accel = detectActiveProvider(
      (encoderOpts.executionProviders as InferenceSession.ExecutionProviderConfig[] | undefined) ??
        [],
    );

    const initMs = performance.now() - t0;
    process.stderr.write(
      `[media-understanding] ASR ready: Parakeet TDT 0.6B v3 — encoder: ${accel.provider}${accel.hardware ? " GPU" : ""}, preprocessor+decoder: cpu (${initMs.toFixed(0)}ms init)\n`,
    );

    return { preprocessor, encoder, decoder, tokenizer, accel };
  })();

  // If initialization fails, reset so next attempt can retry
  sessionPromise.catch(() => {
    sessionPromise = undefined;
  });

  return sessionPromise;
}

/** Reset the singleton (for tests). */
export function resetAsrSession(): void {
  sessionPromise = undefined;
}

// ---------------------------------------------------------------------------
// Mel spectrogram via nemo128.onnx
// ---------------------------------------------------------------------------

/**
 * Run the NeMo mel spectrogram preprocessor.
 *
 * nemo128.onnx inputs:
 *   - waveforms:      float32[1, samples]
 *   - waveforms_lens: int64[1]
 *
 * nemo128.onnx outputs:
 *   - features:      float32[1, 128, T]
 *   - features_lens: int64[1]
 *
 * @returns { features: Float32Array, T: number } where features is [128 × T]
 *          row-major (128 mel bins × T frames), ready for the encoder.
 */
async function runPreprocessor(
  session: InferenceSession,
  pcm: Float32Array,
): Promise<{ features: Float32Array; T: number }> {
  const wavTensor = new ort.Tensor("float32", pcm, [1, pcm.length]);
  const lenTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(pcm.length)]), [1]);

  let out: ort.InferenceSession.OnnxValueMapType;
  try {
    out = await session.run({ waveforms: wavTensor, waveforms_lens: lenTensor });
  } finally {
    if (typeof wavTensor.dispose === "function") wavTensor.dispose();
    if (typeof lenTensor.dispose === "function") lenTensor.dispose();
  }

  const featTensor = out["features"];
  if (!featTensor) throw new Error("nemo128.onnx returned no 'features' tensor");

  // dims: [1, 128, T] — strip the batch dimension
  const T = featTensor.dims[2] as number;
  const features = featTensor.data as Float32Array;
  if (typeof featTensor.dispose === "function") featTensor.dispose();

  return { features, T };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Report which ASR acceleration backend is active.
 * Returns `undefined` if the session hasn't been created yet.
 */
export async function getAsrAccelInfo(): Promise<AsrAccelInfo> {
  const session = await getSession();
  return session.accel;
}

/**
 * Transcribe 16 kHz mono PCM audio into timestamped segments.
 *
 * This is the primary inference entry point. For the full pipeline including
 * PCM extraction from media files and long-form chunking, see `asr-audio.ts`
 * and `asr-chunking.ts`.
 *
 * @param pcm Mono Float32 PCM at 16 kHz.
 * @param timeOffset Offset in seconds added to all timestamps (for chunked audio).
 * @returns Array of `Segment` objects with millisecond timestamps.
 */
export async function transcribePcm(pcm: Float32Array, timeOffset = 0): Promise<Segment[]> {
  const session = await getSession();

  // 1. Mel spectrogram via nemo128.onnx
  const t0 = performance.now();
  const { features, T } = await runPreprocessor(session.preprocessor, pcm);
  logTiming("preprocessor", performance.now() - t0);
  if (T === 0) return [];

  // 2. Run encoder
  const N_MELS = 128;
  const input = new ort.Tensor("float32", features, [1, N_MELS, T]);
  const lenTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(T)]), [1]);

  const t1 = performance.now();
  let encOutput: ort.Tensor;
  try {
    const encOut = await session.encoder.run({ audio_signal: input, length: lenTensor });
    const outputTensor = encOut["outputs"] ?? Object.values(encOut)[0];
    if (!outputTensor) throw new Error("Encoder returned no output tensors");
    encOutput = outputTensor;
  } finally {
    if (typeof input.dispose === "function") input.dispose();
    if (typeof lenTensor.dispose === "function") lenTensor.dispose();
  }
  logTiming("encoder", performance.now() - t1);

  // 3. Transpose [1, D, Tenc] → [Tenc, D]
  const t2 = performance.now();
  const { transposed, Tenc, D } = transposeEncoder(encOutput.data as Float32Array, encOutput.dims);
  if (typeof encOutput.dispose === "function") encOutput.dispose();
  logTiming("transpose", performance.now() - t2);

  if (Tenc === 0) return [];

  // 4. TDT greedy decode
  const t3 = performance.now();
  const decodedTokens = await tdtGreedyDecode(
    transposed,
    Tenc,
    D,
    session.decoder,
    session.tokenizer,
    timeOffset,
  );
  logTiming("decoder", performance.now() - t3);

  // 5. Convert decoded tokens → Segment[]
  return tokensToSegments(decodedTokens, session.tokenizer);
}

/**
 * Convert an array of decoded tokens into `Segment[]` with word-level grouping.
 *
 * SentencePiece `▁` marks word boundaries. Tokens within the same word are
 * merged into a single segment spanning from the first token's start to the
 * last token's end. We use `tokenizer.hasWordPrefix(id)` to detect the `▁`
 * prefix directly from the raw vocabulary, avoiding the fragile approach of
 * decoding and checking for leading spaces (which `decode()` strips).
 */
function tokensToSegments(tokens: DecodedToken[], tokenizer: Tokenizer): Segment[] {
  if (tokens.length === 0) return [];

  const segments: Segment[] = [];
  let currentText = "";
  let segStart = 0;
  let segEnd = 0;

  for (const tok of tokens) {
    const decoded = tokenizer.decode([tok.id]);
    if (decoded.length === 0) continue;

    // Detect word boundary via the raw vocab's ▁ prefix
    const isWordStart = tokenizer.hasWordPrefix(tok.id) || currentText === "";

    if (isWordStart && currentText !== "") {
      // Flush previous word segment
      segments.push({
        start: Math.round(segStart * 1000),
        end: Math.round(segEnd * 1000),
        text: currentText,
      });
      currentText = decoded;
      segStart = tok.startSec;
      segEnd = tok.endSec;
    } else {
      if (currentText === "") {
        segStart = tok.startSec;
      }
      currentText += decoded;
      segEnd = tok.endSec;
    }
  }

  // Flush last segment
  if (currentText) {
    segments.push({
      start: Math.round(segStart * 1000),
      end: Math.round(segEnd * 1000),
      text: currentText,
    });
  }

  return segments;
}

// Re-export for consumers
export { FRAME_TIME_STRIDE };
