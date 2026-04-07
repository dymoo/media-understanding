/**
 * ASR acceleration adapter — detects available ONNX Runtime execution providers.
 *
 * Provides platform-aware execution provider selection with separate configs
 * for heavy models (encoder) and light models (preprocessor, decoder):
 *
 *   Heavy (encoder):
 *     - macOS Apple Silicon: CPU (see note below)
 *     - Linux x64:           CUDA (NVIDIA GPUs), CPU fallback
 *     - Windows x64:         DirectML (any DX12 GPU), CPU fallback
 *     - All platforms:       CPU fallback (always available)
 *
 *   Light (preprocessor, decoder):
 *     - All platforms: CPU only. These models are too small for GPU dispatch
 *       overhead to pay off — the per-call overhead exceeds the computation,
 *       especially for the decoder which runs thousands of inference calls
 *       per file in the TDT greedy loop.
 *
 * macOS CoreML note: Benchmarks show CPU is 2.7× faster than CoreML for the
 * INT8 Parakeet encoder on Apple Silicon. CoreML only supports 44% of graph
 * nodes (1,428/3,249), causing heavy partitioning with costly memory copies.
 * Session init with CoreML takes ~4.6s (model compilation) vs ~0.7s for CPU.
 * CoreML also causes CoreAnalytics context leaks. Use MEDIA_UNDERSTANDING_EP=coreml
 * to force CoreML if a future ORT version improves support.
 *
 * Design: separate from src/accel.ts which handles video frame HW acceleration
 * via node-av. This module handles ASR inference acceleration via onnxruntime-node.
 */

import type { InferenceSession } from "onnxruntime-node";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Metadata about the active ASR acceleration backend. */
export interface AsrAccelInfo {
  /** Execution provider name for the encoder, e.g. "coreml", "cuda", "dml", "cpu". */
  readonly provider: string;
  /** Whether GPU/hardware acceleration is active for the encoder. */
  readonly hardware: boolean;
}

// ---------------------------------------------------------------------------
// Provider detection
// ---------------------------------------------------------------------------

/**
 * Return the ordered list of execution providers for the encoder (heavy model).
 * ONNX Runtime will use the first available one and fall back automatically.
 *
 * On macOS Apple Silicon, defaults to CPU because CoreML is 2.7× slower for
 * the INT8 Parakeet encoder (only 44% of nodes supported, heavy partitioning,
 * 4.6s model compilation overhead). Set `MEDIA_UNDERSTANDING_EP=coreml` to
 * force CoreML for testing/future ORT versions.
 */
function getEncoderProviders(): InferenceSession.ExecutionProviderConfig[] {
  const platform = process.platform;
  const arch = process.arch;

  // Allow explicit EP override for benchmarking / future CoreML improvements
  const epOverride = process.env["MEDIA_UNDERSTANDING_EP"];
  if (epOverride) {
    return [epOverride, "cpu"];
  }

  if (platform === "darwin" && arch === "arm64") {
    // CPU is 2.7× faster than CoreML for INT8 Parakeet on Apple Silicon.
    // CoreML model compilation adds ~4s to session init with no inference benefit.
    return ["cpu"];
  }
  if (platform === "linux" && arch === "x64") {
    return ["cuda", "cpu"];
  }
  if (platform === "win32" && arch === "x64") {
    return ["dml", "cpu"];
  }

  // Fallback for any other platform (macOS Intel, Linux ARM, etc.)
  return ["cpu"];
}

/**
 * Return CPU-only execution providers for light models (preprocessor, decoder).
 */
function getLightProviders(): InferenceSession.ExecutionProviderConfig[] {
  return ["cpu"];
}

/** Common session options shared by all models. */
const BASE_OPTS: Omit<InferenceSession.SessionOptions, "executionProviders"> = {
  graphOptimizationLevel: "all",
  enableCpuMemArena: true,
  enableMemPattern: true,
};

/**
 * Session options for the encoder (heavy model — may use GPU on Linux/Windows).
 */
export function getEncoderSessionOptions(): InferenceSession.SessionOptions {
  return { ...BASE_OPTS, executionProviders: getEncoderProviders() };
}

/**
 * Session options for light models (preprocessor + decoder — CPU only).
 */
export function getLightSessionOptions(): InferenceSession.SessionOptions {
  return { ...BASE_OPTS, executionProviders: getLightProviders() };
}

/**
 * Build session options with the best available execution provider.
 * @deprecated Use `getEncoderSessionOptions()` or `getLightSessionOptions()` instead.
 */
export function getSessionOptions(): InferenceSession.SessionOptions {
  return getEncoderSessionOptions();
}

/**
 * Determine which execution provider a session is actually using.
 * Call after creating a session to report what was selected.
 */
export function detectActiveProvider(
  requestedProviders: InferenceSession.ExecutionProviderConfig[],
): AsrAccelInfo {
  // onnxruntime-node doesn't expose which EP was selected after session creation.
  // We infer from the requested list: the first non-cpu provider is the "attempt",
  // and we optimistically report it. If it failed, ORT logged a warning to stderr
  // and fell back to CPU — but we can't detect that programmatically.
  //
  // For accurate reporting we'd need to test session creation + catch errors,
  // but that's expensive. This is good enough for user-facing info.
  const first = requestedProviders[0];
  const name = typeof first === "string" ? first : (first?.name ?? "cpu");
  const hardware = name !== "cpu";
  return { provider: name, hardware };
}
