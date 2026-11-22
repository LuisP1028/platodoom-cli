import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const PLATODOOM_VERSION = "1.0.0";

export const HARDWARE_TIER_SPECS = {
  G4: { accelerator_type: "GPU", memory_type: "VRAM", nominal_vram_gb: 96, vram_gb: 95.0, max_num_seqs: 256 },
  H100: { accelerator_type: "GPU", memory_type: "VRAM", nominal_vram_gb: 80, vram_gb: 79.2, max_num_seqs: 160 },
  "A100-80GB": { accelerator_type: "GPU", memory_type: "VRAM", nominal_vram_gb: 80, vram_gb: 79.2, max_num_seqs: 128 },
  A100: { accelerator_type: "GPU", memory_type: "VRAM", nominal_vram_gb: 40, vram_gb: 40.0, max_num_seqs: 96 },
  "v6e-1": { accelerator_type: "TPU", memory_type: "HBM", nominal_vram_gb: 32, vram_gb: 32.0, max_num_seqs: 8 },
  L4: { accelerator_type: "GPU", memory_type: "VRAM", nominal_vram_gb: 24, vram_gb: 23.5, max_num_seqs: 64 },
  T4: { accelerator_type: "GPU", memory_type: "VRAM", nominal_vram_gb: 16, vram_gb: 15.8, max_num_seqs: 32 },
  "v5e-1": { accelerator_type: "TPU", memory_type: "HBM", nominal_vram_gb: 16, vram_gb: 16.0, max_num_seqs: 4 },
};

export const BUILTIN_MODEL_CATALOG = {
  "qwen3.8-27b": {
    alias: "qwen3.8-27b",
    hf_repo_id: "Qwen/Qwen3.8-27B",
    quantized_hf_repo_id: "cyankiwi/Qwen3.8-27B-AWQ-INT4",
    preferred_gpu: "A100",
    min_vram_gb: 19.0,
    template_family: "qwen3_chatml",
  },
  "qwen-9b": {
    alias: "qwen-9b",
    hf_repo_id: "Qwen/Qwen3.5-9B",
    quantized_hf_repo_id: "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    preferred_gpu: "T4",
    min_vram_gb: 9.0,
    template_family: "qwen3_chatml",
  },
  "r1-14b": {
    alias: "r1-14b",
    hf_repo_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    quantized_hf_repo_id: "casperhansen/deepseek-r1-distill-qwen-14b-awq",
    preferred_gpu: "L4",
    min_vram_gb: 12.2,
    template_family: "deepseek_r1_distill",
  },
  "qwen-27b": {
    alias: "qwen-27b",
    hf_repo_id: "Qwen/Qwen3.5-27B",
    quantized_hf_repo_id: "cyankiwi/Qwen3.5-27B-AWQ-4bit",
    preferred_gpu: "A100",
    min_vram_gb: 19.0,
    template_family: "qwen3_chatml",
  },
  "qwen-35b-moe": {
    alias: "qwen-35b-moe",
    hf_repo_id: "Qwen/Qwen3.5-35B-A3B",
    quantized_hf_repo_id: "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    preferred_gpu: "A100",
    min_vram_gb: 23.0,
    template_family: "qwen3_chatml",
  },
  "r1-32b": {
    alias: "r1-32b",
    hf_repo_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    quantized_hf_repo_id: "casperhansen/deepseek-r1-distill-qwen-32b-awq",
    preferred_gpu: "A100",
    min_vram_gb: 30.0,
    template_family: "deepseek_r1_distill",
  },
  "qwen-30b-moe": {
    alias: "qwen-30b-moe",
    hf_repo_id: "Qwen/Qwen3-30B-A3B",
    quantized_hf_repo_id: "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
    preferred_gpu: "A100",
    min_vram_gb: 20.8,
    template_family: "qwen3_chatml",
  },
  "qwen-coder-30b-moe": {
    alias: "qwen-coder-30b-moe",
    hf_repo_id: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    quantized_hf_repo_id: "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
    preferred_gpu: "A100",
    min_vram_gb: 20.8,
    template_family: "qwen3_chatml",
  },
  "qwen-14b": {
    alias: "qwen-14b",
    hf_repo_id: "Qwen/Qwen3-14B",
    quantized_hf_repo_id: "Qwen/Qwen3-14B-AWQ",
    preferred_gpu: "L4",
    min_vram_gb: 12.2,
    template_family: "qwen3_chatml",
  },
  "qwen-8b": {
    alias: "qwen-8b",
    hf_repo_id: "Qwen/Qwen3-8B",
    quantized_hf_repo_id: "Qwen/Qwen3-8B-AWQ",
    preferred_gpu: "T4",
    min_vram_gb: 8.7,
    template_family: "qwen3_chatml",
  },
  "qwen-4b": {
    alias: "qwen-4b",
    hf_repo_id: "Qwen/Qwen3-4B",
    quantized_hf_repo_id: "Qwen/Qwen3-4B-AWQ",
    preferred_gpu: "T4",
    min_vram_gb: 6.4,
    template_family: "qwen3_chatml",
  },
  "qwen-122b-moe": {
    alias: "qwen-122b-moe",
    hf_repo_id: "Qwen/Qwen3.5-122B-A10B",
    quantized_hf_repo_id: "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    preferred_gpu: "H100",
    min_vram_gb: 70.0,
    template_family: "qwen3_chatml",
  },
  "r1-7b": {
    alias: "r1-7b",
    hf_repo_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    quantized_hf_repo_id: "casperhansen/deepseek-r1-distill-qwen-7b-awq",
    preferred_gpu: "T4",
    min_vram_gb: 8.4,
    template_family: "deepseek_r1_distill",
  },
  "deepseek-v2-lite": {
    alias: "deepseek-v2-lite",
    hf_repo_id: "deepseek-ai/DeepSeek-V2-Lite-Chat",
    quantized_hf_repo_id: "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
    preferred_gpu: "L4",
    min_vram_gb: 12.8,
    template_family: "deepseek_v2_chat",
  },
};

export const DEFAULT_PLATO_CATALOG = BUILTIN_MODEL_CATALOG;
export const DEFAULT_MODEL_CATALOG = BUILTIN_MODEL_CATALOG;

export function resolveUserPlatoConfigPath(env = process.env) {
  const customHome = env.PLATODOOM_CONFIG_DIR || env.PLATODOOM_HOME;
  const baseDir = customHome && customHome.trim() ? resolve(customHome.trim()) : join(homedir(), ".platodoom");
  return join(baseDir, "config.json");
}

export function generatePlatoBearerToken() {
  return `sk-plato-${randomBytes(32).toString("hex")}`;
}

export function createInteractiveSessionSlug(sessionId) {
  const suffix = sessionId && sessionId.trim() ? sessionId.trim() : randomBytes(4).toString("hex");
  return suffix.startsWith("platodoom-chat-") ? suffix : `platodoom-chat-${suffix}`;
}

export function resolvePlatoConfig(cliOverrides = {}, cwd = process.cwd(), env = process.env) {
  const base = {
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
    models: { custom: {} },
    security: { extra_denylist_globs: [], redact_env_secrets: true },
    response_cache: { enabled: true, ttl_seconds: 604800, cache_nondeterministic: false },
  };

  const userCfgPath = resolveUserPlatoConfigPath(env);
  if (existsSync(userCfgPath)) {
    try {
      const parsed = JSON.parse(readFileSync(userCfgPath, "utf8"));
      if (parsed && typeof parsed === "object") {
        Object.assign(base, parsed);
      }
    } catch {}
  }

  const projCfgPath = join(resolve(cwd), ".platodoom.json");
  if (existsSync(projCfgPath)) {
    try {
      const proj = JSON.parse(readFileSync(projCfgPath, "utf8"));
      if (proj && typeof proj === "object") {
        if (typeof proj.model === "string") base.default_model = proj.model;
        if (typeof proj.gpu === "string") base.default_gpu = proj.gpu;
        if (typeof proj.stack === "string" || proj.stack === null) base.default_stack = proj.stack;
        if (typeof proj.access === "string") base.default_access = proj.access;
      }
    } catch {}
  }

  if (env.PLATODOOM_MODEL) base.default_model = env.PLATODOOM_MODEL;
  if (env.PLATODOOM_GPU) base.default_gpu = env.PLATODOOM_GPU;
  if (env.PLATODOOM_STACK) base.default_stack = env.PLATODOOM_STACK;
  if (env.PLATODOOM_ACCESS) base.default_access = env.PLATODOOM_ACCESS;

  if (typeof cliOverrides.model === "string") base.default_model = cliOverrides.model;
  if (typeof cliOverrides.default_model === "string") base.default_model = cliOverrides.default_model;
  if (typeof cliOverrides.gpu === "string") base.default_gpu = cliOverrides.gpu;
  if (typeof cliOverrides.default_gpu === "string") base.default_gpu = cliOverrides.default_gpu;
  return base;
}

export function savePlatoConfig(config, env = process.env) {
  const cfgPath = resolveUserPlatoConfigPath(env);
  mkdirSync(dirname(cfgPath), { recursive: true, mode: 0o700 });
  const merged = { ...resolvePlatoConfig({}, process.cwd(), env), ...config };
  writeFileSync(cfgPath, `${JSON.stringify(merged, null, 2)}
`, { encoding: "utf8", mode: 0o600 });
  try {
    chmodSync(cfgPath, 0o600);
  } catch {}
  return cfgPath;
}

export function runPlatoCli(argv = process.argv.slice(2), env = process.env) {
  const pkgRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const launcher = join(pkgRoot, "bin", "platodoom.mjs");
  return new Promise((resolvePromise) => {
    const child = spawn(process.execPath, [launcher, ...argv], {
      stdio: "inherit",
      env: {
        ...env,
        PYTHONPATH: env.PYTHONPATH ? `${pkgRoot}${delimiter}${env.PYTHONPATH}` : pkgRoot,
      },
    });
    child.on("error", () => resolvePromise(1));
    child.on("close", (code, sig) => resolvePromise(sig === "SIGINT" && code === null ? 130 : (code ?? 0)));
  });
}

export const launchPlatoCli = runPlatoCli;
export const loadPlatoConfig = resolvePlatoConfig;
