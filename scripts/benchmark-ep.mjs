#!/usr/bin/env node
/**
 * Benchmark the current Parakeet ONNX path (typically CPU on macOS).
 *
 * Uses the low-level pipeline directly (extractPcm + transcribeLongForm) to
 * bypass the transcript cache in media.ts, ensuring each run does real work.
 *
 * Usage:
 *   node scripts/benchmark-ep.mjs <audio-file>
 *   node scripts/benchmark-ep.mjs --runs 3 --json <audio-file>
 */

import { performance } from "node:perf_hooks";
import { extractPcm } from "../dist/asr-audio.js";
import { transcribeLongForm } from "../dist/asr-chunking.js";
import { resetAsrSession } from "../dist/asr-pipeline.js";

function parseArgs(argv) {
  let json = false;
  let runs = 1;
  let file;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--json") {
      json = true;
    } else if (arg === "--runs") {
      runs = parseInt(argv[++i] ?? "1", 10);
    } else if (!arg.startsWith("-")) {
      file = arg;
    }
  }

  if (!file) {
    throw new Error("Usage: node scripts/benchmark-ep.mjs [--json] [--runs N] <audio-file>");
  }

  return { file, json, runs };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

async function benchRun(file, label, jsonMode) {
  resetAsrSession();

  const t0 = performance.now();
  const pcmStart = performance.now();
  const pcm = await extractPcm(file);
  const pcmMs = performance.now() - pcmStart;

  const asrStart = performance.now();
  const segments = await transcribeLongForm(pcm);
  const asrMs = performance.now() - asrStart;

  const totalMs = performance.now() - t0;
  const text = segments
    .map((segment) => segment.text)
    .join(" ")
    .trim();
  const transcriptChars = text.length;
  const durationSec = pcm.length / 16_000;
  const rtf = durationSec > 0 ? asrMs / 1000 / durationSec : 0;

  const result = {
    label,
    runtime: "onnxruntime-node",
    variant: "parakeet-int8-current",
    ep: process.env["MEDIA_UNDERSTANDING_EP"] ?? "(platform default)",
    totalMs,
    pcmMs,
    asrMs,
    durationSec,
    rtf,
    segments: segments.length,
    transcriptChars,
    text,
  };

  if (!jsonMode) {
    console.log(
      `[${label}] total=${totalMs.toFixed(0)}ms, pcm=${pcmMs.toFixed(0)}ms, asr=${asrMs.toFixed(0)}ms, ` +
        `segs=${segments.length}, chars=${transcriptChars}, audio=${durationSec.toFixed(1)}s, ` +
        `RTF=${rtf.toFixed(3)}x`,
    );
  }

  return result;
}

async function main() {
  const { file, json, runs } = parseArgs(process.argv.slice(2));
  const results = [];

  if (!json) {
    const ep = process.env["MEDIA_UNDERSTANDING_EP"] ?? "(platform default)";
    console.log(`\nBenchmark: ${file}`);
    console.log(`EP: ${ep}`);
    console.log(`Runs: ${runs}\n`);
  }

  for (let i = 0; i < runs; i++) {
    results.push(await benchRun(file, `run ${i + 1}`, json));
  }

  const summary = {
    runtime: "onnxruntime-node",
    variant: "parakeet-int8-current",
    ep: process.env["MEDIA_UNDERSTANDING_EP"] ?? "(platform default)",
    runs,
    metrics: {
      totalMs: median(results.map((result) => result.totalMs)),
      pcmMs: median(results.map((result) => result.pcmMs)),
      asrMs: median(results.map((result) => result.asrMs)),
      durationSec: results[0]?.durationSec ?? 0,
      rtf: median(results.map((result) => result.rtf)),
      transcriptChars: results[0]?.transcriptChars ?? 0,
      segments: results[0]?.segments ?? 0,
    },
    text: results[0]?.text ?? "",
    perRun: results.map(({ text, ...result }) => result),
  };

  if (json) {
    console.log(JSON.stringify(summary));
    return;
  }

  console.log(`\n=== Results (${summary.ep}) ===`);
  console.log(`ASR median:   ${summary.metrics.asrMs.toFixed(0)}ms`);
  console.log(`Total median: ${summary.metrics.totalMs.toFixed(0)}ms`);
  console.log(`RTF:          ${summary.metrics.rtf.toFixed(3)}x`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
