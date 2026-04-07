#!/usr/bin/env node
/**
 * Run a benchmark matrix across Parakeet MLX INT4 (GPU/CPU),
 * Parakeet ONNX INT8 current path, and whisper.cpp q5_1.
 */

import { mkdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { execFile as execFileCb } from "node:child_process";

const execFile = promisify(execFileCb);

function parseArgs(argv) {
  let file;
  let runs = 3;
  let outDir = path.join(
    process.cwd(),
    "tmp",
    "benchmarks",
    new Date().toISOString().replaceAll(":", "-"),
  );

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--runs") {
      runs = parseInt(argv[++i] ?? "3", 10);
    } else if (arg === "--out-dir") {
      outDir = argv[++i] ?? outDir;
    } else if (!arg.startsWith("-")) {
      file = arg;
    }
  }

  if (!file) {
    throw new Error(
      "Usage: node scripts/benchmark-asr-matrix.mjs [--runs N] [--out-dir dir] <audio-file>",
    );
  }

  return { file, runs, outDir };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

async function getDurationSec(file) {
  const { stdout } = await execFile("ffprobe", [
    "-v",
    "quiet",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    file,
  ]);
  return parseFloat(stdout.trim());
}

async function runJson(command, args, env = {}) {
  const { stdout, stderr } = await execFile(command, args, {
    cwd: process.cwd(),
    env: { ...process.env, ...env },
    maxBuffer: 20 * 1024 * 1024,
  });

  const lines = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const last = lines.at(-1);
  if (!last)
    throw new Error(
      `No JSON output from ${command} ${args.join(" ")}${stderr ? `\n${stderr}` : ""}`,
    );
  return JSON.parse(last);
}

async function runVariant(variant, file, runs, durationSec, outDir) {
  const results = [];
  for (let i = 0; i < runs; i++) {
    let result;
    if (variant.kind === "mlx") {
      result = await runJson(variant.python, [
        path.join(process.cwd(), "scripts", "benchmark-parakeet-mlx.py"),
        "--device",
        variant.device,
        "--model",
        variant.model,
        file,
      ]);
    } else if (variant.kind === "onnx") {
      const raw = await runJson(
        "node",
        [path.join(process.cwd(), "scripts", "benchmark-ep.mjs"), "--json", file],
        variant.env,
      );
      result = {
        runtime: raw.runtime,
        variant: raw.variant,
        device: raw.ep === "(platform default)" ? "cpu" : raw.ep,
        totalMs: raw.metrics.totalMs,
        transcriptChars: raw.metrics.transcriptChars,
        text: raw.text,
      };
    } else {
      result = await runJson("node", [
        path.join(process.cwd(), "scripts", "benchmark-whisper-cpp.mjs"),
        "--bin",
        variant.bin,
        "--model",
        variant.model,
        ...(variant.noGpu ? ["--no-gpu"] : []),
        file,
      ]);
    }

    results.push({
      ...result,
      durationSec,
      rtf: durationSec > 0 ? result.totalMs / 1000 / durationSec : 0,
    });
  }

  const summary = {
    name: variant.name,
    runtime: results[0]?.runtime ?? variant.kind,
    variant: results[0]?.variant ?? variant.name,
    device: results[0]?.device ?? variant.device ?? "unknown",
    runs,
    totalMs: median(results.map((result) => result.totalMs)),
    transcriptChars: results[0]?.transcriptChars ?? 0,
    durationSec,
    rtf: median(results.map((result) => result.rtf)),
    text: results[0]?.text ?? "",
    perRun: results.map(({ text, ...result }) => result),
  };

  const slug = variant.name.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-");
  await writeFile(path.join(outDir, `${slug}.txt`), summary.text, "utf8");
  await writeFile(path.join(outDir, `${slug}.json`), JSON.stringify(summary, null, 2), "utf8");
  return summary;
}

function printTable(results) {
  console.log("\n| Variant | Device | Total Median | RTF | Chars |");
  console.log("| --- | --- | ---: | ---: | ---: |");
  for (const result of results) {
    console.log(
      `| ${result.name} | ${result.device} | ${result.totalMs.toFixed(0)} ms | ${result.rtf.toFixed(3)}x | ${result.transcriptChars} |`,
    );
  }
}

async function main() {
  const { file, runs, outDir } = parseArgs(process.argv.slice(2));
  await mkdir(outDir, { recursive: true });
  const durationSec = await getDurationSec(file);

  const variants = [
    {
      kind: "mlx",
      name: "Parakeet MLX INT4 GPU",
      device: "gpu",
      model: process.env["PARAKEET_MLX_MODEL"] ?? "mlx-community/parakeet-tdt-0.6b-v3",
      python:
        process.env["PARAKEET_MLX_PYTHON"] ??
        path.join(process.cwd(), "tmp", "benchmarks", ".venv", "bin", "python"),
    },
    {
      kind: "mlx",
      name: "Parakeet MLX INT4 CPU",
      device: "cpu",
      model: process.env["PARAKEET_MLX_MODEL"] ?? "mlx-community/parakeet-tdt-0.6b-v3",
      python:
        process.env["PARAKEET_MLX_PYTHON"] ??
        path.join(process.cwd(), "tmp", "benchmarks", ".venv", "bin", "python"),
    },
    {
      kind: "onnx",
      name: "Parakeet ONNX INT8 CPU",
      env: {},
    },
    {
      kind: "whisper",
      name: "Whisper.cpp q5_1",
      device: "gpu",
      bin: process.env["WHISPER_CPP_BIN"] ?? "/opt/homebrew/bin/whisper-cli",
      model:
        process.env["WHISPER_CPP_MODEL"] ??
        path.join(os.homedir(), ".cache", "media-understanding", "models", "ggml-base.en-q5_1.bin"),
    },
  ];

  const results = [];
  for (const variant of variants) {
    console.log(`Running ${variant.name}...`);
    results.push(await runVariant(variant, file, runs, durationSec, outDir));
  }

  printTable(results);
  await writeFile(path.join(outDir, "matrix.json"), JSON.stringify(results, null, 2), "utf8");
  console.log(`\nArtifacts written to ${outDir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
