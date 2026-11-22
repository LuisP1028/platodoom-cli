#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, copyFileSync, existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const isWin = process.platform === "win32";
const rawScriptPath = fileURLToPath(import.meta.url);
const realScriptPath = existsSync(rawScriptPath) ? realpathSync(rawScriptPath) : rawScriptPath;
const pkgRoot = resolve(dirname(realScriptPath), "..");

const argv = process.argv.slice(2);

const RUNTIME_BUNDLE_MANIFEST = [
  "bin/platodoom.mjs",
  "requirements.txt",
  "scripts/postinstall.mjs",
  "platodoom/__init__.py",
  "platodoom/bundle.py",
  "platodoom/cache_model.py",
  "platodoom/catalog.py",
  "platodoom/cli.py",
  "platodoom/config.py",
  "platodoom/discovery.py",
  "platodoom/mcp_bridge.py",
  "platodoom/onboarding.py",
  "platodoom/repl.py",
  "platodoom/response_cache.py",
  "platodoom/security.py",
  "platodoom/session.py",
  "platodoom/standby.py",
  "platodoom/swarm.py",
  "platodoom/tools.py",
  "platodoom/assets/platodoom_utf8_blocks_truecolor.ansi",
  "curation/__init__.py",
  "curation/api_client.py",
  "curation/colab_server.py",
  "notebooks/serve_sandcastle_vllm.ipynb",
  "scripts/__init__.py",
  "scripts/omp_qwen_adapter.py",
  "scripts/overnight_pipeline_watchdog.py",
  "scripts/provision_colab_serving.py",
  "scripts/qwen_worktree_agent.py",
];

const SYNTHETIC_INITS = {
  "curation/__init__.py": '"""Minimal runtime curation package for platodoom."""\n',
  "scripts/__init__.py": '"""Minimal runtime scripts package for platodoom."""\n',
};

function resolveWorkspaceDir(cwd) {
  let ws = resolve(cwd);
  if (ws.endsWith("/.platodoom/runtime")) {
    return resolve(ws, "..", "..");
  }
  if (ws.endsWith("/.platodoom")) {
    return resolve(ws, "..");
  }
  return ws;
}

function computeSourceDigest(srcRoot) {
  const hash = createHash("sha256");
  for (const rel of RUNTIME_BUNDLE_MANIFEST) {
    const fp = join(srcRoot, rel);
    let buf;
    if (existsSync(fp)) {
      buf = readFileSync(fp);
    } else if (rel in SYNTHETIC_INITS) {
      buf = Buffer.from(SYNTHETIC_INITS[rel], "utf8");
    } else {
      return null;
    }
    hash.update(`${rel}:${buf.length}\0`);
    hash.update(buf);
  }
  return hash.digest("hex").slice(0, 16);
}

function bootstrapLocalPlatodoomRuntime(cwd, sourceRoot) {
  const wsDir = resolveWorkspaceDir(cwd);
  const platoDir = join(wsDir, ".platodoom");
  const runtimeDir = join(platoDir, "runtime");

  for (const sub of [
    platoDir,
    join(platoDir, "sessions"),
    join(platoDir, "cache"),
    join(platoDir, "cache", "responses"),
    join(platoDir, "standby"),
  ]) {
    mkdirSync(sub, { recursive: true, mode: 0o700 });
    try {
      chmodSync(sub, 0o700);
    } catch {}
  }

  const digest = computeSourceDigest(sourceRoot);
  const versionFile = join(runtimeDir, "VERSION");
  const currentVersion = existsSync(versionFile) ? readFileSync(versionFile, "utf8").trim() : "";
  const actualRuntimeDigest = existsSync(join(runtimeDir, "platodoom", "cli.py"))
    ? computeSourceDigest(runtimeDir)
    : null;

  // Always verify actual file contents of runtimeDir (never trust a repo-committed VERSION file alone).
  if (
    resolve(sourceRoot) !== resolve(runtimeDir) &&
    digest &&
    (currentVersion !== digest || actualRuntimeDigest !== digest)
  ) {
    for (const rel of RUNTIME_BUNDLE_MANIFEST) {
      const srcFp = join(sourceRoot, rel);
      const dstFp = join(runtimeDir, rel);
      mkdirSync(dirname(dstFp), { recursive: true });
      if (existsSync(srcFp)) {
        copyFileSync(srcFp, dstFp);
      } else if (rel in SYNTHETIC_INITS) {
        writeFileSync(dstFp, SYNTHETIC_INITS[rel], "utf8");
      }
    }
    for (const execRel of ["bin/platodoom.mjs", "platodoom/cli.py"]) {
      const execFp = join(runtimeDir, execRel);
      if (existsSync(execFp)) {
        try {
          chmodSync(execFp, 0o755);
        } catch {}
      }
    }
    writeFileSync(versionFile, `${digest}\n`, "utf8");
  }

  const configPath = join(platoDir, "config.json");
  if (!existsSync(configPath)) {
    const defaultCfg = {
      version: 1,
      default_model: "qwen3.8-27b",
      default_gpu: "auto",
      default_stack: null,
      default_access: "workspace-write",
      default_backend: "colab",
      colab: {
        max_concurrent_sessions: 3,
        subscription_tier: "pro_plus",
        keepalive_interval_sec: 20,
      },
      storage: {
        cache_backend: "hf_hub",
        gcs_bucket: null,
        gcs_project_id: null,
        gcs_models_prefix: "models/",
        gdrive_models_dir: "/content/drive/MyDrive/platodoom_models",
      },
      response_cache: {
        enabled: true,
        ttl_seconds: 604800,
        cache_nondeterministic: false,
      },
      auth: {
        hf_token_env: "PLATODOOM_HF_TOKEN",
        hf_token: null,
        ngrok_authtoken: null,
        tunnel_provider: "cloudflare",
        google_token_path: "~/.mcp-colab/token.json",
        google_credentials_path: "~/.mcp-colab/credentials.json",
      },
      onboarding: {
        completed: false,
        skipped: false,
      },
      models: {
        custom: {},
      },
      security: {
        extra_denylist_globs: [],
        redact_env_secrets: true,
      },
    };
    writeFileSync(configPath, `${JSON.stringify(defaultCfg, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
    try {
      chmodSync(configPath, 0o600);
    } catch {}
  }

  const gitExclude = join(wsDir, ".git", "info", "exclude");
  if (existsSync(join(wsDir, ".git"))) {
    try {
      mkdirSync(dirname(gitExclude), { recursive: true });
      const existing = existsSync(gitExclude) ? readFileSync(gitExclude, "utf8") : "";
      const lines = new Set(existing.split("\n").map((l) => l.trim()));
      if (!lines.has(".platodoom/") && !lines.has(".platodoom")) {
        const prefix = !existing || existing.endsWith("\n") ? "" : "\n";
        writeFileSync(gitExclude, `${existing}${prefix}.platodoom/\n`, "utf8");
      }
    } catch {}
  }

  return { platoDir, runtimeDir };
}

const { platoDir, runtimeDir } = bootstrapLocalPlatodoomRuntime(process.cwd(), pkgRoot);
const verifiedPkgDigest = computeSourceDigest(pkgRoot);
const verifiedRuntimeDigest = existsSync(join(runtimeDir, "platodoom", "cli.py"))
  ? computeSourceDigest(runtimeDir)
  : null;
const effectiveRuntimeRoot =
  verifiedPkgDigest && verifiedRuntimeDigest === verifiedPkgDigest ? runtimeDir : pkgRoot;

if (argv.length === 1 && (argv[0] === "-v" || argv[0] === "--version")) {
  console.log("platodoom v1.0.0");
  process.exit(0);
}

const managedVenvPython = isWin
  ? join(homedir(), ".platodoom", "venv", "Scripts", "python.exe")
  : join(homedir(), ".platodoom", "venv", "bin", "python3");

function pythonHasCoreDeps(pyCmd) {
  const probe = spawnSync(pyCmd, ["-c", "import yaml, pydantic"], { stdio: "ignore" });
  return probe.status === 0;
}

// Never execute untrusted workspace-local binaries (.platodoom/.venv or .sandcastle/bin).
const candidates = isWin
  ? [managedVenvPython, "python", "python3"]
  : [
      managedVenvPython,
      "/opt/anaconda3/bin/python",
      "python3",
      "python",
    ];

let pythonBin =
  candidates.find((c) => (c === "python3" || c === "python" || existsSync(c)) && pythonHasCoreDeps(c)) ||
  null;

if (!pythonBin) {
  const postinstallScript = join(pkgRoot, "scripts", "postinstall.mjs");
  if (existsSync(postinstallScript)) {
    spawnSync(process.execPath, [postinstallScript], { stdio: "inherit" });
  }
  if (existsSync(managedVenvPython)) {
    pythonBin = managedVenvPython;
  } else {
    pythonBin =
      candidates.find((c) => c === "python3" || c === "python" || existsSync(c)) ||
      (isWin ? "python" : "python3");
  }
}

const childEnv = {
  ...process.env,
  PLATODOOM_RUNTIME_ROOT: effectiveRuntimeRoot,
  PLATODOOM_HOME: process.env.PLATODOOM_CONFIG_DIR || process.env.PLATODOOM_HOME || platoDir,
  PYTHONPATH: process.env.PYTHONPATH
    ? `${effectiveRuntimeRoot}${delimiter}${process.env.PYTHONPATH}`
    : effectiveRuntimeRoot,
};

process.on("SIGINT", () => {
  // Let the foreground Python CLI handle SIGINT and flush "PEACE,YO!" before exiting.
});

const result = spawnSync(pythonBin, ["-m", "platodoom.cli", ...argv], {
  stdio: "inherit",
  env: childEnv,
});

if (result.signal === "SIGINT") {
  process.exit(130);
}
process.exit(result.status ?? 0);
