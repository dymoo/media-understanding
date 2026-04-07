#!/usr/bin/env node
/**
 * Benchmark local whisper.cpp via whisper-cli.
 * Emits a single JSON object to stdout.
 */

import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { promisify } from "node:util";
import { execFile as execFileCb } from "node:child_process";

const execFile = promisify(execFileCb);

function parseArgs(argv) {
  let file;
  let bin = "/opt/homebrew/bin/whisper-cli";
  let model = path.join(os.homedir(), ".cache/media-understanding/models/ggml-base.en-q5_1.bin");
  let noGpu = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--bin") {
      bin = argv[++i] ?? bin;
    } else if (arg === "--model") {
      model = argv[++i] ?? model;
    } else if (arg === "--no-gpu") {
      noGpu = true;
    } else if (!arg.startsWith("-")) {
      file = arg;
    }
  }

  if (!file) {
    throw new Error(
      "Usage: node scripts/benchmark-whisper-cpp.mjs [--bin path] [--model path] [--no-gpu] <audio-file>",
    );
  }

  return { file, bin, model, noGpu };
}

async function main() {
  const { file, bin, model, noGpu } = parseArgs(process.argv.slice(2));
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "whisper-bench-"));
  const outputBase = path.join(tempDir, "result");

  const args = [
    "--model",
    model,
    "--file",
    file,
    "--output-txt",
    "--output-file",
    outputBase,
    "--no-prints",
  ];
  if (noGpu) args.push("--no-gpu");

  const started = performance.now();
  try {
    await execFile(bin, args, { maxBuffer: 10 * 1024 * 1024 });
    const text = await readFile(`${outputBase}.txt`, "utf8");
    const totalMs = performance.now() - started;
    console.log(
      JSON.stringify({
        runtime: "whisper.cpp",
        variant: path.basename(model),
        device: noGpu ? "cpu" : "gpu",
        totalMs,
        transcriptChars: text.length,
        text,
      }),
    );
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
