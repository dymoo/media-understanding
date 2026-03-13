#!/usr/bin/env node

import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { type as osType } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { execFile as execFileCb } from "node:child_process";

const execFile = promisify(execFileCb);
const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, "..");

function resolvePlatformPackageName() {
  if (process.platform === "win32") {
    const runtime = osType() !== "Windows_NT" ? "mingw" : "msvc";
    return `@seydx/node-av-${process.platform}-${process.arch}-${runtime}`;
  }

  return `@seydx/node-av-${process.platform}-${process.arch}`;
}

async function runNode(scriptPath) {
  await execFile(process.execPath, [scriptPath], {
    cwd: rootDir,
    stdio: "inherit",
  });
}

async function installPlatformBinding() {
  const packageName = resolvePlatformPackageName();

  let packageJsonPath;
  try {
    packageJsonPath = require.resolve(`${packageName}/package.json`);
  } catch (error) {
    throw new Error(
      `Current platform package not installed: ${packageName}. ` +
        `Ensure optionalDependencies are enabled for pnpm install.`,
      { cause: error },
    );
  }

  const packageDir = dirname(packageJsonPath);
  const binaryPath = join(packageDir, "node-av.node");
  const installScriptPath = join(packageDir, "install.js");
  const zipPath = join(packageDir, "node-av.node.zip");

  if (existsSync(binaryPath)) return;
  if (!existsSync(installScriptPath)) {
    throw new Error(`Missing platform install script: ${installScriptPath}`);
  }
  if (!existsSync(zipPath)) {
    throw new Error(
      `Missing platform archive for ${packageName}: ${zipPath}. ` +
        `Reinstall dependencies from a clean package cache before continuing.`,
    );
  }

  await runNode(installScriptPath);

  if (!existsSync(binaryPath)) {
    throw new Error(`Platform native binding was not extracted for ${packageName}: ${binaryPath}`);
  }
}

async function installFfmpeg() {
  // Use a direct filesystem path instead of require.resolve() to avoid
  // ERR_PACKAGE_PATH_NOT_EXPORTED on Node 24, which enforces the exports map
  // strictly for subpath imports that node-av doesn't expose.
  const ffmpegInstallPath = join(rootDir, "node_modules/node-av/dist/ffmpeg/install.js");
  if (!existsSync(ffmpegInstallPath)) {
    throw new Error(`node-av FFmpeg install script not found: ${ffmpegInstallPath}`);
  }
  await runNode(ffmpegInstallPath);
}

await installPlatformBinding();
await installFfmpeg();
