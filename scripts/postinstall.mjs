#!/usr/bin/env node
/**
 * npm postinstall hook for platodoom.
 * Sets up a lightweight isolated Python virtual environment at ~/.platodoom/venv
 * containing only the minimal local CLI dependencies (selenium, PyYAML, pydantic, httpx).
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

if (process.env.PLATODOOM_SKIP_POSTINSTALL === "1" || process.env.CI === "true") {
  process.exit(0);
}

const isWin = process.platform === "win32";
const pkgRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const reqFile = join(pkgRoot, "requirements.txt");

if (!existsSync(reqFile)) {
  process.exit(0);
}

function findSystemPython() {
  const candidates = isWin ? ["python", "python3", "py"] : ["python3", "python"];
  for (const bin of candidates) {
    const res = spawnSync(bin, ["-c", "import sys; assert sys.version_info >= (3, 10)"], {
      stdio: "ignore",
    });
    if (res.status === 0) return bin;
  }
  return null;
}

const sysPython = findSystemPython();
if (!sysPython) {
  console.warn(
    "[platodoom] Warning: Python 3.10+ was not found on PATH. Install Python 3.10+ and run `platodoom` again.",
  );
  process.exit(0);
}

const venvDir = join(homedir(), ".platodoom", "venv");
const venvPython = isWin
  ? join(venvDir, "Scripts", "python.exe")
  : join(venvDir, "bin", "python3");

function hasCoreDeps(pyBin) {
  if (!existsSync(pyBin)) return false;
  const check = spawnSync(pyBin, ["-c", "import selenium, yaml, pydantic, httpx"], {
    stdio: "ignore",
  });
  return check.status === 0;
}

if (hasCoreDeps(venvPython)) {
  process.exit(0);
}

try {
  mkdirSync(dirname(venvDir), { recursive: true, mode: 0o700 });
  if (!existsSync(venvPython)) {
    const mk = spawnSync(sysPython, ["-m", "venv", venvDir], { stdio: "inherit" });
    if (mk.status !== 0) {
      console.warn(
        `[platodoom] Could not create ~/.platodoom/venv automatically. You can install dependencies manually with:\n  ${sysPython} -m pip install -r ${reqFile}`,
      );
      process.exit(0);
    }
  }
  const pip = spawnSync(
    venvPython,
    ["-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", reqFile],
    { stdio: "inherit" },
  );
  if (pip.status !== 0) {
    console.warn(
      `[platodoom] Python dependency install did not complete (offline or missing pip). Run manually when online:\n  ${venvPython} -m pip install -r ${reqFile}`,
    );
  }
} catch (err) {
  console.warn(`[platodoom] Optional postinstall setup skipped: ${err?.message || err}`);
}
