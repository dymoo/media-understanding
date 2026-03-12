#!/usr/bin/env node
/**
 * CLI shim — text-only output suitable for use as an OpenClaw media model.
 *
 * Usage:
 *   media-understanding <file> [--model <whisper-model>] [--max-chars <n>]
 *
 * Outputs:
 *   - Metadata block
 *   - Transcript (audio/video)
 *   - Frame timestamps are NOT output in CLI mode (images not printable)
 *
 * Exit codes: 0 = success, 1 = error
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { probeMedia, transcribeAudio, truncateTranscript } from "./media.js";
import { MediaError } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const { version } = JSON.parse(readFileSync(join(__dirname, "../package.json"), "utf8")) as {
  version: string;
};

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: "string", short: "m" },
    "max-chars": { type: "string" },
    help: { type: "boolean", short: "h" },
    version: { type: "boolean", short: "V" },
  },
  allowPositionals: true,
});

if (values.version) {
  process.stdout.write(`${version}\n`);
  process.exit(0);
}

if (values.help || positionals.length === 0) {
  process.stdout.write(
    `Usage: media-understanding <file> [options]

Options:
  -m, --model <name>      Whisper model (default: base.en-q5_1)
  --max-chars <n>         Max transcript characters (default: 32000)
  -h, --help              Show this help
  -V, --version           Print version number

Environment:
  MEDIA_UNDERSTANDING_MODEL      Override default Whisper model
  MEDIA_UNDERSTANDING_MAX_CHARS  Override max transcript characters
`,
  );
  process.exit(0);
}

const filePath = positionals[0];
if (!filePath) {
  process.stderr.write("Error: file path is required\n");
  process.exit(1);
}

const model = values.model;
const maxCharsRaw = values["max-chars"];
const maxCharsEnv = parseInt(process.env["MEDIA_UNDERSTANDING_MAX_CHARS"] ?? "32000", 10);
const maxChars = maxCharsRaw !== undefined ? parseInt(maxCharsRaw, 10) : maxCharsEnv;

if (isNaN(maxChars) || maxChars <= 0) {
  process.stderr.write(
    `Error: --max-chars must be a positive integer, got: "${maxCharsRaw ?? ""}"\n`,
  );
  process.exit(1);
}

try {
  const info = await probeMedia(filePath);

  const lines: string[] = [
    `file: ${info.path}`,
    `type: ${info.type}`,
    `duration: ${info.duration.toFixed(1)}s`,
  ];
  if (info.width) lines.push(`resolution: ${info.width}x${info.height}`);
  if (info.fps) lines.push(`fps: ${info.fps.toFixed(2)}`);
  if (info.videoCodec) lines.push(`video_codec: ${info.videoCodec}`);
  if (info.audioCodec) lines.push(`audio_codec: ${info.audioCodec}`);
  if (info.sampleRate) lines.push(`sample_rate: ${info.sampleRate}Hz`);
  if (info.channels) lines.push(`channels: ${info.channels}`);

  process.stdout.write(lines.join("\n") + "\n");

  if (info.type === "audio" || info.type === "video") {
    const segments = await transcribeAudio(filePath, model !== undefined ? { model } : {});

    if (segments.length === 0) {
      process.stdout.write("\n[no speech detected]\n");
    } else {
      const raw = segments
        .map((s) => s.text)
        .join(" ")
        .trim();
      process.stdout.write("\n--- TRANSCRIPT ---\n" + truncateTranscript(raw, maxChars) + "\n");
    }
  }
} catch (err) {
  const message =
    err instanceof MediaError
      ? `[${err.code}] ${err.message}`
      : err instanceof Error
        ? err.message
        : String(err);
  process.stderr.write(`Error: ${message}\n`);
  process.exit(1);
}
