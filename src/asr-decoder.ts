/**
 * TDT (Token-and-Duration Transducer) greedy decoder for Parakeet models.
 *
 * Ported from parakeet.js (MIT) `src/parakeet.js` — adapted to TypeScript
 * with `onnxruntime-node`.
 *
 * The decoder runs frame-by-frame over encoder output:
 *   1. Extract one encoder frame [1, D, 1]
 *   2. Feed it + previous token + LSTM state into decoder_joint model
 *   3. Joiner output = [vocab_size + duration_bins] logits
 *   4. Argmax over vocab portion → token (or blank)
 *   5. Argmax over duration portion → frame advance step
 *   6. Non-blank tokens update the LSTM state; blanks keep it unchanged
 *
 * Reference: onnx-asr (Python) `src/onnx_asr/asr.py` line ~200.
 */

import type { InferenceSession, Tensor } from "onnxruntime-node";
import * as ort from "onnxruntime-node";

import type { Tokenizer } from "./asr-tokenizer.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Encoder subsampling factor (FastConformer 8× temporal downsampling). */
const SUBSAMPLING = 8;

/** Feature frame stride in seconds (10 ms). */
const WINDOW_STRIDE = 0.01;

/** Time stride per encoder frame = subsampling × window_stride = 80 ms. */
const FRAME_TIME_STRIDE = SUBSAMPLING * WINDOW_STRIDE;

/** Maximum tokens to emit from a single encoder frame before force-advancing. */
const MAX_TOKENS_PER_STEP = 10;

/** Number of LSTM layers in the prediction network. */
const PRED_LAYERS = 2;

/** Hidden dimension of the LSTM prediction network. */
const PRED_HIDDEN = 640;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single decoded token with its timestamp. */
export interface DecodedToken {
  /** Token ID from vocabulary. */
  id: number;
  /** Encoder frame index that emitted this token. */
  frame: number;
  /** Start time in seconds. */
  startSec: number;
  /** End time in seconds. */
  endSec: number;
}

export { FRAME_TIME_STRIDE };

// ---------------------------------------------------------------------------
// LSTM state helpers
// ---------------------------------------------------------------------------

interface LstmState {
  state1: Tensor;
  state2: Tensor;
}

function createZeroState(): LstmState {
  const size = PRED_LAYERS * 1 * PRED_HIDDEN;
  const dims: readonly number[] = [PRED_LAYERS, 1, PRED_HIDDEN];
  return {
    state1: new ort.Tensor("float32", new Float32Array(size), dims),
    state2: new ort.Tensor("float32", new Float32Array(size), dims),
  };
}

// ---------------------------------------------------------------------------
// Core decode
// ---------------------------------------------------------------------------

/**
 * Greedily decode encoder output into tokens with timestamps.
 *
 * @param encoderOutput Transposed encoder output `[Tenc, D]` as Float32Array (row-major).
 * @param Tenc Number of encoder time frames.
 * @param D Encoder feature dimension.
 * @param joinerSession The decoder_joint ONNX session.
 * @param tokenizer Tokenizer with `blankId` and `vocabSize`.
 * @param timeOffset Offset added to all timestamps (for chunked transcription).
 * @returns Array of decoded tokens with frame-level timestamps.
 */
export async function tdtGreedyDecode(
  encoderOutput: Float32Array,
  Tenc: number,
  D: number,
  joinerSession: InferenceSession,
  tokenizer: Tokenizer,
  timeOffset = 0,
): Promise<DecodedToken[]> {
  const tokens: DecodedToken[] = [];
  const blankId = tokenizer.blankId;

  // Pre-allocate reusable tensors
  const targetIdArr = new Int32Array(1);
  const targetTensor = new ort.Tensor("int32", targetIdArr, [1, 1]);
  const targetLenArr = new Int32Array([1]);
  const targetLenTensor = new ort.Tensor("int32", targetLenArr, [1]);

  // Encoder frame buffer — reused each iteration
  const encFrameBuf = new Float32Array(D);
  const encFrameTensor = new ort.Tensor("float32", encFrameBuf, [1, D, 1]);

  // LSTM state
  let state = createZeroState();
  let emittedThisFrame = 0;

  let t = 0;
  while (t < Tenc) {
    // Copy encoder frame into reusable buffer
    const frameStart = t * D;
    encFrameBuf.set(encoderOutput.subarray(frameStart, frameStart + D));

    // Set previous token (blank at start)
    const prevTok = tokens.length > 0 ? tokens[tokens.length - 1]!.id : blankId;
    targetIdArr[0] = prevTok;

    const feeds: Record<string, Tensor> = {
      encoder_outputs: encFrameTensor,
      targets: targetTensor,
      target_length: targetLenTensor,
      input_states_1: state.state1,
      input_states_2: state.state2,
    };

    const out = await joinerSession.run(feeds);
    const logits = out["outputs"];
    if (!logits) throw new Error("Decoder output missing 'outputs' tensor");

    const data = logits.data as Float32Array;
    const vocabSize = tokenizer.vocabSize;

    // Argmax over token logits [0..vocabSize)
    let maxLogit = -Infinity;
    let maxId = 0;
    for (let i = 0; i < vocabSize; i++) {
      const v = data[i]!;
      if (v > maxLogit) {
        maxLogit = v;
        maxId = i;
      }
    }

    // Argmax over duration logits [vocabSize..totalDim)
    const totalDim = data.length;
    let step = 0;
    if (totalDim > vocabSize) {
      let maxDur = -Infinity;
      for (let i = vocabSize; i < totalDim; i++) {
        const v = data[i]!;
        if (v > maxDur) {
          maxDur = v;
          step = i - vocabSize;
        }
      }
    }

    // Extract new LSTM states (may fall back to current if not present)
    const newState1 = out["output_states_1"] ?? state.state1;
    const newState2 = out["output_states_2"] ?? state.state2;

    if (maxId !== blankId) {
      // Non-blank: emit token, update LSTM state
      const durFrames = step > 0 ? step : 1;
      const endFrame = Math.min(Tenc, t + Math.max(1, durFrames));
      tokens.push({
        id: maxId,
        frame: t,
        startSec: timeOffset + t * FRAME_TIME_STRIDE,
        endSec: timeOffset + endFrame * FRAME_TIME_STRIDE,
      });

      // Dispose old LSTM state tensors before replacing (prevent ~20 MB leak per file).
      // Only dispose when the joiner returned new state tensors (i.e. they differ).
      if (newState1 !== state.state1 && typeof state.state1.dispose === "function") {
        state.state1.dispose();
      }
      if (newState2 !== state.state2 && typeof state.state2.dispose === "function") {
        state.state2.dispose();
      }

      // Only update state on non-blank (matches Python reference)
      state = { state1: newState1, state2: newState2 };
      emittedThisFrame++;
    }

    // Dispose joiner output logits (data already consumed)
    if (typeof logits.dispose === "function") logits.dispose();

    // Frame advancement:
    // 1. TDT step > 0 → advance by step
    // 2. Blank or max tokens per step → advance by 1
    // 3. Otherwise stay on frame for next token
    if (step > 0) {
      t += step;
      emittedThisFrame = 0;
    } else if (maxId === blankId || emittedThisFrame >= MAX_TOKENS_PER_STEP) {
      t += 1;
      emittedThisFrame = 0;
    }
    // else: stay on same frame, emit more tokens
  }

  // Dispose final LSTM state tensors
  if (typeof state.state1.dispose === "function") state.state1.dispose();
  if (typeof state.state2.dispose === "function") state.state2.dispose();

  return tokens;
}

/**
 * Transpose encoder output from ONNX shape [1, D, Tenc] to [Tenc, D].
 *
 * The encoder ONNX model outputs `[batch=1, D, Tenc]` (channels-first).
 * The decoder loop needs `[Tenc, D]` (row-major, one row per frame).
 *
 * @returns `{ transposed, Tenc, D }`.
 */
export function transposeEncoder(
  encData: Float32Array,
  dims: readonly number[],
): { transposed: Float32Array; Tenc: number; D: number } {
  // Expected: [1, D, Tenc]
  const D = dims[1]!;
  const Tenc = dims[2]!;
  const transposed = new Float32Array(Tenc * D);

  // 8-wide unrolled transpose (matching parakeet.js benchmark-driven path)
  for (let t = 0; t < Tenc; t++) {
    const tOffset = t * D;
    let d = 0;
    for (; d <= D - 8; d += 8) {
      const srcOffset = d * Tenc + t;
      transposed[tOffset + d] = encData[srcOffset]!;
      transposed[tOffset + d + 1] = encData[srcOffset + Tenc]!;
      transposed[tOffset + d + 2] = encData[srcOffset + 2 * Tenc]!;
      transposed[tOffset + d + 3] = encData[srcOffset + 3 * Tenc]!;
      transposed[tOffset + d + 4] = encData[srcOffset + 4 * Tenc]!;
      transposed[tOffset + d + 5] = encData[srcOffset + 5 * Tenc]!;
      transposed[tOffset + d + 6] = encData[srcOffset + 6 * Tenc]!;
      transposed[tOffset + d + 7] = encData[srcOffset + 7 * Tenc]!;
    }
    for (; d < D; d++) {
      transposed[tOffset + d] = encData[d * Tenc + t]!;
    }
  }

  return { transposed, Tenc, D };
}
