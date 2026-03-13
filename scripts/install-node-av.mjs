#!/usr/bin/env node

import { existsSync, readFileSync } from "node:fs";
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

function readPackageJsonName(packageJsonPath) {
  const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
  return typeof packageJson.name === "string" ? packageJson.name : undefined;
}

function findInstalledPackageRoot(packageName) {
  const packagePathParts = packageName.split("/");
  let currentDir = rootDir;

  while (true) {
    const packageJsonPath = join(currentDir, "node_modules", ...packagePathParts, "package.json");
    if (existsSync(packageJsonPath) && readPackageJsonName(packageJsonPath) === packageName) {
      return dirname(packageJsonPath);
    }

    const parentDir = dirname(currentDir);
    if (parentDir === currentDir) {
      return undefined;
    }

    currentDir = parentDir;
  }
}

function findPackageRootFromResolvedPath(resolvedPath, packageName) {
  let currentDir = dirname(resolvedPath);

  while (true) {
    const packageJsonPath = join(currentDir, "package.json");
    if (existsSync(packageJsonPath) && readPackageJsonName(packageJsonPath) === packageName) {
      return currentDir;
    }

    const parentDir = dirname(currentDir);
    if (parentDir === currentDir) {
      throw new Error(
        `Resolved ${packageName} to ${resolvedPath} but could not locate its package root.`,
      );
    }

    currentDir = parentDir;
  }
}

function resolvePackageRoot(packageName, missingMessage) {
  const installedRoot = findInstalledPackageRoot(packageName);
  if (installedRoot) {
    return installedRoot;
  }

  try {
    return findPackageRootFromResolvedPath(require.resolve(packageName), packageName);
  } catch (error) {
    throw new Error(missingMessage, { cause: error });
  }
}

async function installPlatformBinding() {
  const packageName = resolvePlatformPackageName();

  const packageDir = resolvePackageRoot(
    packageName,
    `Current platform package not installed: ${packageName}. ` +
      `Ensure optionalDependencies are enabled and your package manager did not skip them.`,
  );
  const binaryPath = join(packageDir, "node-av.node");
  const installScriptPath = join(packageDir, "install.js");
  const zipPath = join(packageDir, "node-av.node.zip");

  if (existsSync(binaryPath)) return;
  if (!existsSync(installScriptPath)) {
    throw new Error(`Missing platform install script for ${packageName}: ${installScriptPath}`);
  }
  if (!existsSync(zipPath)) {
    throw new Error(
      `Missing platform archive for ${packageName} at ${zipPath}. ` +
        `Reinstall dependencies from a clean package cache before continuing.`,
    );
  }

  await runNode(installScriptPath);

  if (!existsSync(binaryPath)) {
    throw new Error(`Platform native binding was not extracted for ${packageName}: ${binaryPath}`);
  }
}

async function installFfmpeg() {
  const nodeAvRoot = resolvePackageRoot(
    "node-av",
    "Required dependency `node-av` is not installed. Reinstall the package and try again.",
  );
  const ffmpegInstallPath = join(nodeAvRoot, "dist/ffmpeg/install.js");
  if (!existsSync(ffmpegInstallPath)) {
    throw new Error(
      `node-av FFmpeg install script not found in resolved package root ${nodeAvRoot}: ${ffmpegInstallPath}`,
    );
  }
  await runNode(ffmpegInstallPath);
}

await installPlatformBinding();
await installFfmpeg();
