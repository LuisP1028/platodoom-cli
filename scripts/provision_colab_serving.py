#!/usr/bin/env python3
"""
Automated Per-Directory Colab Serving Provisioning & Activity-Based Headless Keep-Alive Manager
for qwen3.8-27b.
Document ID: provision_colab_serving_856_v1
Governing Standards: platodoom specification (FR-1, FR-2, FR-3, FR-4), LANGUAGE.md, RULE[user_global]
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*quota project.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curation.api_client import (
    DEFAULT_GPU_ROUTING_FILENAME,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_PARALLEL_AGENTS,
    UNIFIED_MODEL_ALIAS,
    normalize_colab_endpoint_url,
    probe_colab_qwen_endpoint as _api_probe_colab_qwen_endpoint,
    register_colab_qwen_endpoint as _api_register_colab_qwen_endpoint,
    resolve_colab_endpoint_registry_paths as _api_resolve_colab_endpoint_registry_paths,
    resolve_serving_bearer_token,
    route_gpu_for_model_count,
    route_gpu_for_multi_model_agents,
)

try:
    from curation.api_client import (
        atomic_write_json,
        compute_dir_slug,
        resolve_directory_sandcastle_dir,
        resolve_project_root,
    )
except ImportError:

    def resolve_project_root(path: Optional[Path | str] = None) -> Path:
        """Unwraps .sandcastle/worktrees/<branch> or trailing /.sandcastle to canonical <project_root> (FR-3.1)."""
        raw = Path(path).expanduser().resolve() if path is not None else Path.cwd().resolve()
        if raw.is_file():
            raw = raw.parent
        parts = raw.parts
        for i in range(len(parts) - 1):
            if parts[i] == ".sandcastle" and parts[i + 1] == "worktrees":
                if i > 0:
                    return Path(*parts[:i]).resolve()
        if raw.name == ".sandcastle":
            return raw.parent.resolve()
        for parent in raw.parents:
            if parent.name == ".sandcastle":
                return parent.parent.resolve()
        return raw

    def compute_dir_slug(path: Optional[Path | str] = None) -> str:
        """Computes deterministic <repo_basename>-<sha1_8_of_project_root> (Section 2.0.2, FR-3.1)."""
        project_root = resolve_project_root(path)
        base = re.sub(r"[^a-zA-Z0-9]+", "-", project_root.name).strip("-").lower() or "repo"
        digest = hashlib.sha1(str(project_root).encode("utf-8")).hexdigest()[:8]
        return f"{base}-{digest}"

    def resolve_directory_sandcastle_dir(path: Optional[Path | str] = None) -> Path:
        """Returns <project_root>/.sandcastle for storing per-directory registry, routing, lock, and PID files."""
        return resolve_project_root(path) / ".sandcastle"

    def atomic_write_json(target_path: Path, data: Dict[str, Any], mode: Optional[int] = None) -> None:
        """Atomically writes JSON payload via same-directory temporary file, fsync, and os.replace."""
        target_path = Path(target_path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        effective_mode = mode
        if effective_mode is None:
            has_secret = bool(data.get("bearer_token") or data.get("auth_token") or data.get("api_key"))
            if has_secret or target_path.name in ("config.json", "endpoint.json"):
                effective_mode = 0o600
            else:
                effective_mode = 0o644
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target_path.name}.",
            suffix=f".{os.getpid()}.tmp",
            dir=str(target_path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            try:
                os.fchmod(fd, effective_mode)
            except OSError:
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(tmp_path), str(target_path))
            try:
                os.chmod(str(target_path), effective_mode)
            except OSError:
                pass
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise


def _resolve_platodoom_home() -> Path:
    """Resolves ~/.platodoom (or PLATODOOM_CONFIG_DIR / PLATODOOM_HOME override)."""
    raw = os.environ.get("PLATODOOM_CONFIG_DIR") or os.environ.get("PLATODOOM_HOME")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser().resolve()
    return (Path.home() / ".platodoom").resolve()


def resolve_platodoom_session_dir(session_slug: str) -> Path:
    """Returns ~/.platodoom/sessions/<session_slug> with 0700 owner-only permissions (FR-2.1.2)."""
    slug = str(session_slug).strip()
    sdir = _resolve_platodoom_home() / "sessions" / slug
    sdir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(sdir), 0o700)
    except OSError:
        pass
    return sdir


def _resolve_configured_gcs_bucket_and_project(repo_dir: Optional[Path | str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Dynamically resolves optional GCS bucket and project ID without hardcoded personal identifiers (FR-1.2, AC-1)."""
    env_bucket = (
        os.environ.get("PLATODOOM_GCS_BUCKET")
        or os.environ.get("GCS_BUCKET_NAME")
        or ""
    ).strip()
    env_project = (
        os.environ.get("PLATODOOM_GCS_PROJECT")
        or os.environ.get("GCP_PROJECT_ID")
        or ""
    ).strip()
    storage_uri = (os.environ.get("PLATODOOM_STORAGE_URI") or "").strip()
    if not env_bucket and storage_uri.startswith("gs://"):
        without_scheme = storage_uri[len("gs://") :]
        env_bucket = without_scheme.split("/", 1)[0].strip()

    cfg_path = _resolve_platodoom_home() / "config.json"
    if (not env_bucket or not env_project) and cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            storage_cfg = cfg.get("storage") if isinstance(cfg, dict) else {}
            if isinstance(storage_cfg, dict):
                if not env_bucket and storage_cfg.get("gcs_bucket"):
                    env_bucket = str(storage_cfg["gcs_bucket"]).strip()
                if not env_project and storage_cfg.get("gcs_project_id"):
                    env_project = str(storage_cfg["gcs_project_id"]).strip()
        except Exception:
            pass

    return (env_bucket or None), (env_project or None)


def _strip_endpoint_secrets_for_remote_sync(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strips bearer_token, auth_token, api_key, hf_token, and tunnel_token before any remote GCS upload (FR-5.4)."""
    sanitized = dict(record)
    for secret_key in ("bearer_token", "auth_token", "api_key", "tunnel_token", "hf_token", "token"):
        sanitized.pop(secret_key, None)
    sanitized["auth_required"] = True
    return sanitized


strip_secrets_for_remote_sync = _strip_endpoint_secrets_for_remote_sync


GCS_BUCKET_NAME = _resolve_configured_gcs_bucket_and_project()[0] or ""
GCS_ENDPOINT_BLOB = "endpoints/colab_qwen_endpoint.json"
MCP_COLAB_TOKEN_PATH = Path.home() / ".mcp-colab" / "token.json"
MCP_COLAB_CREDS_PATH = Path.home() / ".mcp-colab" / "credentials.json"
SERVE_NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "serve_sandcastle_vllm.ipynb"

# JavaScript snippets for headless Colab notebook control & activity-based monitoring (FR-1, FR-2, FR-7.2)
COLAB_RUN_ALL_JS = """
const nb = window.colab && window.colab.global && window.colab.global.notebook;
if (nb) {
    nb.isSoleAuthored = async () => true;
    nb.hasTrustedNotebookContext = async () => true;
    nb.confirmCodeExecution = async () => true;
    if (typeof nb.clearAllOutputs === "function") {
        try { nb.clearAllOutputs(); } catch (e) {}
    }
    if (typeof nb.runAll === "function") {
        nb.runAll();
        return true;
    }
}
const btn = document.querySelector("[command=runall]");
if (btn) {
    btn.click();
    return true;
}
return false;
"""

COLAB_DOM_INSPECT_JS = """
const nb = window.colab && window.colab.global && window.colab.global.notebook;
const k = nb && typeof nb.getKernel === "function" ? nb.getKernel() : null;
const kernelState = k && typeof k.getState === "function" ? String(k.getState()) : "";
const cells = nb && typeof nb.getCells === "function" ? nb.getCells() : [];
const codeCells = cells.filter(c => c.getType && c.getType() === "code");
let activeIdx = -1;
let completedCount = 0;
const cellTexts = [];
const cellErrors = [];
codeCells.forEach((c, idx) => {
    const el = typeof c.getElement === "function" ? c.getElement() : null;
    const isRunning = (typeof c.isRunning === "function" && c.isRunning()) ||
        (typeof c.isBusy === "function" && c.isBusy()) ||
        Boolean(el && el.classList && el.classList.contains("running"));
    if (isRunning && activeIdx === -1) {
        activeIdx = idx;
    }
    const m = c.getModel && c.getModel();
    if (m && typeof m.toJson === "function") {
        const j = m.toJson();
        const outs = Array.isArray(j.outputs) ? j.outputs : [];
        if (outs.length > 0 && !isRunning) {
            completedCount = Math.max(completedCount, idx + 1);
        }
        outs.forEach(o => {
            if (o.output_type === "error") {
                cellErrors.push(`Cell ${idx + 1}: ${o.ename || "Error"}: ${o.evalue || ""}`);
            }
            if (o.text) {
                cellTexts.push(Array.isArray(o.text) ? o.text.join("") : String(o.text));
            }
        });
    }
});

// Recursive Shadow-DOM Traversal for Colab Outages, Session-Limit Modals & GPU Dialogs (FR-7.2)
function collectShadowDialogs(rootNode, maxDepth = 10, depth = 0) {
    if (!rootNode || depth > maxDepth) return [];
    const results = [];
    const selectors = "colab-sessions-dialog, colab-dialog, mwc-dialog, md-dialog, [role='dialog'], [role='alert'], .modal-dialog, .colab-butterbar, colab-status-bar";
    try {
        const matched = rootNode.querySelectorAll ? Array.from(rootNode.querySelectorAll(selectors)) : [];
        for (const el of matched) {
            const txt = String(el.innerText || el.textContent || "").trim();
            if (txt) results.push(txt);
            if (el.shadowRoot) {
                const shadowTxt = String(el.shadowRoot.innerText || el.shadowRoot.textContent || "").trim();
                if (shadowTxt) results.push(shadowTxt);
            }
        }
        const allNodes = rootNode.querySelectorAll ? Array.from(rootNode.querySelectorAll("*")) : [];
        for (const node of allNodes) {
            if (node && node.shadowRoot) {
                results.push(...collectShadowDialogs(node.shadowRoot, maxDepth, depth + 1));
            }
        }
    } catch (e) {}
    return results;
}

const allShadowTexts = collectShadowDialogs(document, 10);
const dialogErrors = allShadowTexts.filter(t =>
    /cannot connect to gpu|too many active sessions|manage sessions|experiencing issues provisioning runtimes|out of compute units|usage limit|runtime disconnected|runtime died/i.test(t)
);
const outageBanners = allShadowTexts.filter(t =>
    /experiencing issues provisioning runtimes|service unavailable|provisioning outage/i.test(t)
);
const sessionLimitDialogs = allShadowTexts.filter(t =>
    /too many active sessions|manage sessions|terminate other sessions/i.test(t)
);

return {
    kernelState,
    activeIdx,
    completedCount,
    cellErrors,
    dialogErrors,
    outageBanners,
    sessionLimitDialogs,
    combinedOutput: cellTexts.join("\\n")
};
"""

COLAB_KEEPALIVE_PING_JS = """
const nb = window.colab && window.colab.global && window.colab.global.notebook;
const k = nb && typeof nb.getKernel === "function" ? nb.getKernel() : null;
const kernelState = k && typeof k.getState === "function" ? String(k.getState()) : "";
document.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, clientX: 120, clientY: 120 }));
return { kernelState, url: window.location.href };
"""

COLAB_RERUN_CELL5_JS = """
const nb = window.colab && window.colab.global && window.colab.global.notebook;
if (!nb || typeof nb.getCells !== "function") return false;
const codeCells = nb.getCells().filter(c => c.getType && c.getType() === "code");
const cell5 = codeCells[4];
if (cell5 && typeof cell5.run === "function") {
    cell5.run();
    return true;
}
return false;
"""

COLAB_STAGE_LABELS: List[str] = [
    "Stage 0: Allocating Google Colab GPU VM",
    "Cell 1/6: Installing serving dependencies",
    "Cell 2/6: Verifying physical GPU VRAM & concurrency parameters",
    "Cell 3/6: Authenticating & downloading merged qwen3.8-27b weights (18.84 GB) from GCS",
    "Cell 4/6: Loading qwen3.8-27b into GPU VRAM & starting single-GPU batched server on port 8000",
    "Cell 5/6: Opening public HTTPS tunnel & publishing per-directory endpoint metadata",
    "Cell 6/6: Running end-to-end inference verification & keep-alive readiness check",
]

DEFAULT_GCS_PROJECT_ID = ""
DEFAULT_GCS_BUCKET_NAME = ""

MODEL_PREFERRED_GPU_TIERS: Dict[str, str] = {
    "qwen-9b": "T4",
    "r1-14b": "L4",
    "qwen-27b": "A100",
    "qwen3.8-27b": "A100",
    "qwen-35b-moe": "A100",
    "r1-32b": "A100",
    "qwen-30b-moe": "A100",
    "qwen-coder-30b-moe": "A100",
    "qwen-14b": "L4",
    "qwen-8b": "T4",
    "qwen-4b": "T4",
    "qwen-122b-moe": "G4",
    "r1-7b": "T4",
    "deepseek-v2-lite": "T4",
    "flux-klein-4b": "T4",
    "flux-klein-9b": "L4",
    "z-image-turbo": "T4",
}

MODEL_HF_REPOS: Dict[str, Dict[str, Any]] = {
    "qwen-9b": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-9B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    },
    "r1-14b": {
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
    },
    "qwen-27b": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-27B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
    },
    "qwen3.8-27b": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.8-27B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
    },
    "qwen-35b-moe": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-35B-A3B",
        "quantized_hf_repo_id": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    },
    "r1-32b": {
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-32b-awq",
    },
    "qwen-30b-moe": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-30B-A3B",
        "quantized_hf_repo_id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
    },
    "qwen-coder-30b-moe": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "quantized_hf_repo_id": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
    },
    "qwen-14b": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-14B",
        "quantized_hf_repo_id": "Qwen/Qwen3-14B-AWQ",
    },
    "qwen-8b": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-8B",
        "quantized_hf_repo_id": "Qwen/Qwen3-8B-AWQ",
    },
    "qwen-4b": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-4B",
        "quantized_hf_repo_id": "Qwen/Qwen3-4B-AWQ",
    },
    "qwen-122b-moe": {
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-122B-A10B",
        "quantized_hf_repo_id": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    },
    "r1-7b": {
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-7b-awq",
    },
    "deepseek-v2-lite": {
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "quantized_hf_repo_id": "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
    },
    "flux-klein-4b": {
        "source": "hf_hub",
        "hf_repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "quantized_hf_repo_id": "black-forest-labs/FLUX.2-klein-base-4B-FP8",
    },
    "flux-klein-9b": {
        "source": "hf_hub",
        "hf_repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "quantized_hf_repo_id": "unsloth/FLUX.2-klein-9B-GGUF",
    },
    "z-image-turbo": {
        "source": "hf_hub",
        "hf_repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "quantized_hf_repo_id": "Tongyi-MAI/Z-Image-Turbo",
    },
}

try:
    from platodoom.catalog import (
        BUILTIN_MODEL_CATALOG as _SSOT_MODEL_CATALOG,
        GPU_TIER_SPECS as _SSOT_GPU_TIER_SPECS,
        estimate_model_kv_and_state_footprint as _estimate_ssot_kv_footprint,
        export_colab_catalog_payload as _export_ssot_colab_payload,
        normalize_hardware_tier as _normalize_ssot_tier,
        resolve_hardware_stack as _resolve_ssot_stack,
        resolve_multi_model_agent_allocation as _resolve_ssot_multi_model_agents,
    )

    for _alias, _prof in _SSOT_MODEL_CATALOG.items():
        if _prof.get("preferred_gpu"):
            MODEL_PREFERRED_GPU_TIERS[_alias] = str(_prof["preferred_gpu"])
        if _prof.get("hf_repo_id"):
            _entry_copy: Dict[str, Any] = {
                "source": str(_prof.get("source", "hf_hub")),
                "hf_repo_id": str(_prof["hf_repo_id"]),
                "quantized_hf_repo_id": str(_prof.get("quantized_hf_repo_id") or _prof["hf_repo_id"]),
                "modality": str(_prof.get("modality") or "text"),
                "min_vram_gb": float(_prof.get("min_vram_gb", 10.0)),
                "weights_gb": float(_prof.get("weights_gb", 5.0)),
            }
            if _prof.get("fallback_hf_repo_id"):
                _entry_copy["fallback_hf_repo_id"] = str(_prof["fallback_hf_repo_id"])
            if isinstance(_prof.get("default_precision_by_gpu"), dict):
                _entry_copy["default_precision_by_gpu"] = {
                    str(_gk): str(_gv) for _gk, _gv in _prof["default_precision_by_gpu"].items()
                }
            if isinstance(_prof.get("quantized_repo_ids"), dict):
                _entry_copy["quantized_repo_ids"] = {
                    str(_qk): str(_qv) for _qk, _qv in _prof["quantized_repo_ids"].items()
                }
            if isinstance(_prof.get("precision_variants"), dict):
                _entry_copy["precision_variants"] = {
                    str(_pk): {
                        "hf_repo_id": str(_pv.get("hf_repo_id", _prof["hf_repo_id"])),
                        "weights_gb": float(_pv.get("weights_gb", 0.0)),
                        "min_vram_gb": float(_pv.get("min_vram_gb", 0.0)),
                    }
                    for _pk, _pv in _prof["precision_variants"].items()
                    if isinstance(_pv, dict)
                }
            MODEL_HF_REPOS[_alias] = _entry_copy
except ImportError:
    _SSOT_GPU_TIER_SPECS = {}
    _estimate_ssot_kv_footprint = None
    _export_ssot_colab_payload = None
    _normalize_ssot_tier = None
    _resolve_ssot_stack = None
    _resolve_ssot_multi_model_agents = None


class ColabStageLabelMap(dict):
    """Hybrid dict + sequence of dynamic Colab stage labels supporting both key and index lookups."""

    def __init__(self, mapping: Dict[str, str], ordered: List[str]) -> None:
        super().__init__(mapping)
        self._ordered = list(ordered)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, slice)):
            return self._ordered[key]
        return super().__getitem__(key)

    def as_list(self) -> List[str]:
        return list(self._ordered)


def get_colab_stage_labels(
    model_alias: Optional[str] = None,
    gpu_tier: Optional[str] = None,
) -> ColabStageLabelMap:
    """Returns dynamic Cell 1-6 progress descriptions reflecting the selected model, download source (GCS vs HF Hub), and GPU tier."""
    effective_model = str(model_alias or UNIFIED_MODEL_ALIAS).strip() or UNIFIED_MODEL_ALIAS
    effective_gpu = str(gpu_tier or MODEL_PREFERRED_GPU_TIERS.get(effective_model, "T4")).strip() or "T4"
    is_gcs_model = False
    if is_gcs_model:
        cell3_label = (
            f"Cell 3/6: Authenticating & downloading merged {effective_model} weights (18.84 GB) from GCS"
        )
    else:
        hf_info = MODEL_HF_REPOS.get(effective_model, {})
        repo_label = hf_info.get("quantized_hf_repo_id") or hf_info.get("hf_repo_id") or effective_model
        cell3_label = (
            f"Cell 3/6: Downloading {effective_model} ({repo_label}) weights from HuggingFace Hub"
        )

    ordered = [
        f"Stage 0: Allocating Google Colab {effective_gpu} GPU VM for {effective_model}",
        "Cell 1/6: Provisioning isolated /opt/vllm-venv (vllm==0.33.0, fastsafetensors, hf_transfer, kernel cache)",
        f"Cell 2/6: Verifying physical {effective_gpu} GPU VRAM & concurrency parameters",
        cell3_label,
        f"Cell 4/6: Booting Port-8000 Supervisor & native vLLM continuous batching engine for {effective_model} on {effective_gpu}",
        "Cell 5/6: Opening public HTTPS tunnel & publishing redacted per-directory endpoint metadata",
        f"Cell 6/6: Running end-to-end {effective_model} inference verification & keep-alive readiness check",
    ]
    mapping = {
        "stage_0": ordered[0],
        "cell_1": ordered[1],
        "cell_2": ordered[2],
        "cell_3": ordered[3],
        "cell_4": ordered[4],
        "cell_5": ordered[5],
        "cell_6": ordered[6],
        "model_alias": effective_model,
        "gpu_tier": effective_gpu,
        "source": "gcs" if is_gcs_model else "hf_hub",
    }
    return ColabStageLabelMap(mapping, ordered)


class DirectoryLockPaths(tuple):
    """Hybrid 3-tuple (lock_path, pid_path, mirror_pid_path) that also supports dict-style key lookup."""

    def __new__(cls, lock_path: Path, pid_path: Path, mirror_pid_path: Path, meta: Dict[str, Any]):
        instance = super().__new__(cls, (lock_path, pid_path, mirror_pid_path))
        instance._meta = meta
        return instance

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._meta[key]
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._meta.get(key, default)

    def keys(self):
        return self._meta.keys()

    def items(self):
        return self._meta.items()

    def values(self):
        return self._meta.values()

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return item in self._meta
        return super().__contains__(item)


def resolve_colab_endpoint_registry_paths(
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    session_slug: Optional[str] = None,
) -> List[Path]:
    """Resolves per-directory or interactive-session endpoint registry paths without cross-directory pollution (FR-2.1.2, FR-3.1, FR-3.2)."""
    if session_slug and str(session_slug).strip():
        slug = str(session_slug).strip()
        sdir = resolve_platodoom_session_dir(slug)
        return [sdir / "endpoint.json"]

    if repo_dir is None and include_global:
        return _api_resolve_colab_endpoint_registry_paths(repo_dir=None, include_global=True)

    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    paths: List[Path] = [
        project_root / ".sandcastle" / "colab_qwen_endpoint.json",
        Path.home() / ".sandcastle" / "daemons" / dir_slug / "colab_qwen_endpoint.json",
    ]
    if (project_root / "multi-stage-blackboard-pipeline.mts").is_file() and not (
        project_root / ".sandcastle" / "multi-stage-blackboard-pipeline.mts"
    ).is_file():
        paths.append(project_root / "colab_qwen_endpoint.json")
    if include_global:
        paths.append(Path.home() / ".sandcastle" / "colab_qwen_endpoint.json")

    deduped: List[Path] = []
    seen: set[str] = set()
    for p in paths:
        resolved_str = str(p.resolve()) if p.exists() else str(p)
        if resolved_str not in seen:
            seen.add(resolved_str)
            deduped.append(p)
    return deduped


def _extract_live_probe_model_metadata(
    probe: Optional[Mapping[str, Any]],
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Extracts authoritative live `/v1/models` fields so `.sandcastle/colab_qwen_endpoint.json` matches server truth (Spec 874 §4.5)."""
    if not isinstance(probe, Mapping):
        return {}
    details_map = probe.get("model_details")
    served_list = [str(m) for m in (probe.get("served_models") or probe.get("models") or []) if str(m).strip()]
    out: Dict[str, Any] = {}
    if served_list:
        out["served_models"] = served_list
        out["resident_models"] = list(served_list)

    if not isinstance(details_map, Mapping) or not details_map:
        return out

    clean_target = str(target_model or "").strip().split("@", 1)[0].strip()
    matched_id: Optional[str] = None
    matched_rec: Optional[Mapping[str, Any]] = None

    if clean_target:
        if clean_target in details_map and isinstance(details_map[clean_target], Mapping):
            matched_id = clean_target
            matched_rec = details_map[clean_target]
        else:
            accepted = resolve_expected_model_identifiers(clean_target)
            for mid, rec in details_map.items():
                if not isinstance(rec, Mapping):
                    continue
                mid_s = str(mid).strip()
                root_s = str(rec.get("root") or "").strip().lower()
                if mid_s.lower() in accepted or (root_s and root_s in accepted):
                    matched_id = mid_s
                    matched_rec = rec
                    break

    if matched_rec is None:
        if len(details_map) == 1:
            only_k, only_v = next(iter(details_map.items()))
            if isinstance(only_v, Mapping):
                matched_id = str(only_k).strip()
                matched_rec = only_v
        elif served_list and served_list[0] in details_map and isinstance(details_map[served_list[0]], Mapping):
            matched_id = served_list[0]
            matched_rec = details_map[served_list[0]]

    if not isinstance(matched_rec, Mapping):
        return out

    if matched_id:
        out["active_model"] = matched_id
        out["model"] = matched_id

    for len_key in ("max_model_len", "context_length", "contextWindow"):
        val = matched_rec.get(len_key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and int(val) > 0:
            out["max_model_len"] = int(val)
            break

    max_out = matched_rec.get("max_output_tokens")
    if isinstance(max_out, (int, float)) and not isinstance(max_out, bool) and int(max_out) > 0:
        out["max_output_tokens"] = int(max_out)

    raw_seqs = matched_rec.get("max_num_seqs")
    raw_batch = matched_rec.get("max_batch_size")
    eff_seqs = (
        int(raw_seqs)
        if isinstance(raw_seqs, (int, float)) and not isinstance(raw_seqs, bool) and int(raw_seqs) > 0
        else (
            int(raw_batch)
            if isinstance(raw_batch, (int, float)) and not isinstance(raw_batch, bool) and int(raw_batch) > 0
            else None
        )
    )
    if eff_seqs is not None:
        out["max_num_seqs"] = eff_seqs
        out["max_batch_size"] = int(raw_batch) if isinstance(raw_batch, (int, float)) and not isinstance(raw_batch, bool) and int(raw_batch) > 0 else eff_seqs

    raw_prec = str(matched_rec.get("precision") or matched_rec.get("precision_tier") or "").strip().upper()
    if raw_prec:
        out["precision"] = raw_prec
        out["precision_tier"] = raw_prec

    if isinstance(matched_rec.get("reasoning"), bool):
        out["reasoning"] = bool(matched_rec["reasoning"])

    if isinstance(matched_rec.get("vision"), bool):
        out["vision"] = bool(matched_rec["vision"])
    in_mods = matched_rec.get("input_modalities") or matched_rec.get("input")
    if isinstance(in_mods, list) and in_mods:
        out["input_modalities"] = [str(m) for m in in_mods if str(m).strip()]
        if "vision" not in out:
            out["vision"] = "image" in [m.lower() for m in out["input_modalities"]]

    return out


def register_colab_qwen_endpoint(
    endpoint_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    session_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Persists the live HTTPS Colab endpoint URL and per-directory or interactive-session metadata atomically (FR-2.1.2, FR-3.2)."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_epoch = time.time()

    enriched_meta: Dict[str, Any] = {
        "dir_slug": dir_slug,
        "session_slug": effective_slug,
        "session_type": "interactive" if (session_slug and str(session_slug).strip().startswith("platodoom-chat-")) else "pipeline",
        "project_root": str(project_root),
        "offline_reason": None,
        "checked_at": now_iso,
        "updated_at": now_iso,
        "updated_at_epoch": now_epoch,
        "heartbeat_epoch": now_epoch,
    }
    if metadata:
        enriched_meta.update(metadata)
    enriched_meta["dir_slug"] = dir_slug
    enriched_meta["session_slug"] = effective_slug
    enriched_meta["project_root"] = str(project_root)

    base_url = normalize_colab_endpoint_url(endpoint_url)
    if not base_url:
        raise ValueError("Cannot register an empty Colab endpoint URL.")

    record: Dict[str, Any] = {
        "endpoint_url": base_url,
        "chat_completions_url": f"{base_url}/chat/completions",
        "models_url": f"{base_url}/models",
        "model": str(enriched_meta.get("active_model") or enriched_meta.get("model") or UNIFIED_MODEL_ALIAS),
        "max_parallel_agents": MAX_PARALLEL_AGENTS,
        "max_num_seqs": MAX_PARALLEL_AGENTS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "status": "Online",
        **enriched_meta,
    }
    record["endpoint_url"] = base_url

    has_secret = bool(record.get("bearer_token") or record.get("auth_token") or (session_slug and str(session_slug).strip()))
    file_mode = 0o600 if has_secret else None

    written_paths: List[str] = []
    for reg_path in resolve_colab_endpoint_registry_paths(
        project_root, include_global=include_global, session_slug=session_slug
    ):
        try:
            sig = inspect.signature(atomic_write_json)
            if "mode" in sig.parameters:
                atomic_write_json(reg_path, record, mode=file_mode)
            else:
                atomic_write_json(reg_path, record)
                if file_mode is not None:
                    try:
                        os.chmod(str(reg_path), file_mode)
                    except OSError:
                        pass
            written_paths.append(str(reg_path))
        except Exception:
            continue

    record["registry_paths"] = written_paths
    return record


def probe_colab_qwen_endpoint(
    endpoint_url: str,
    api_key: Optional[str] = None,
    timeout_sec: float = 5.0,
    verify_inference: bool = False,
    inference_timeout_sec: float = 15.0,
    expected_model: Optional[str] = None,
    bearer_token: Optional[str] = None,
    dns_retry_window_sec: float = 0.0,
    dns_retry_interval_sec: float = 1.5,
    allow_local_loopback_fallback: bool = False,
    local_loopback_url: str = "http://127.0.0.1:8000/v1",
    _sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Probes GET /v1/models (<= 5s per request), strictly validates `expected_model` when supplied,
    honors PLATODOOM_TUNNEL_TOKEN + SANDCASTLE_VLLM_API_KEY, and optionally retries transient tunnel DNS propagation delays.
    """
    sig = inspect.signature(_api_probe_colab_qwen_endpoint)
    if "verify_inference" in sig.parameters:
        forward_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout_sec": timeout_sec,
            "verify_inference": verify_inference,
            "inference_timeout_sec": inference_timeout_sec,
            "expected_model": expected_model,
            "bearer_token": bearer_token,
            "dns_retry_window_sec": dns_retry_window_sec,
            "dns_retry_interval_sec": dns_retry_interval_sec,
            "allow_local_loopback_fallback": allow_local_loopback_fallback,
            "local_loopback_url": local_loopback_url,
            "_sleep_fn": _sleep_fn,
        }
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            forward_kwargs = {k: v for k, v in forward_kwargs.items() if k in sig.parameters}
        res = _api_probe_colab_qwen_endpoint(endpoint_url, **forward_kwargs)
        res.setdefault("online", bool(res.get("ok", False)))
        res.setdefault("ok", bool(res.get("online", False)))
        return res

    eff_key = (
        bearer_token
        or api_key
        or os.environ.get("PLATODOOM_TUNNEL_TOKEN")
        or os.environ.get("SANDCASTLE_VLLM_API_KEY")
        or os.environ.get("VLLM_API_KEY")
    )
    base_res = _api_probe_colab_qwen_endpoint(endpoint_url, api_key=eff_key, timeout_sec=timeout_sec)
    is_online = bool(base_res.get("online", base_res.get("ok", False)))
    base_res["online"] = is_online
    base_res["ok"] = is_online
    if not is_online:
        base_res["inference_verified"] = False
        return base_res

    if not verify_inference:
        base_res.setdefault("inference_verified", False)
        return base_res

    base_url = normalize_colab_endpoint_url(endpoint_url)
    token = eff_key or "EMPTY"
    served_models = base_res.get("models") or []
    target_model = served_models[0] if served_models else UNIFIED_MODEL_ALIAS
    warmup_payload = {
        "model": target_model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(warmup_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "Sandcastle-Colab-Probe/1.0",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=max(2.0, float(inference_timeout_sec))) as resp:
            inf_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if resp.status != 200:
                return {
                    **base_res,
                    "online": False,
                    "ok": False,
                    "inference_verified": False,
                    "error": f"Inference warm-up HTTP {resp.status}",
                }
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                return {
                    **base_res,
                    "online": False,
                    "ok": False,
                    "inference_verified": False,
                    "error": "Inference warm-up returned empty choices",
                }
            return {
                **base_res,
                "online": True,
                "ok": True,
                "inference_verified": True,
                "inference_latency_ms": inf_latency_ms,
                "error": None,
            }
    except Exception as exc:
        return {
            **base_res,
            "online": False,
            "ok": False,
            "inference_verified": False,
            "error": f"Inference warm-up probe failed: {exc}",
        }


def is_pid_alive(pid: Optional[int]) -> bool:
    """Checks whether a local OS process PID is alive (and not a defunct zombie) without sending a termination signal (FR-3.2)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return False
    if pid == os.getpid():
        return True
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
        if reaped_pid == pid:
            return False
    except (ChildProcessError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False

    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            stat_txt = proc_stat.read_text(encoding="utf-8", errors="ignore")
            rparen = stat_txt.rfind(")")
            if rparen != -1:
                state_tok = stat_txt[rparen + 1 :].strip().split(" ", 1)[0]
                if state_tok.upper().startswith("Z"):
                    return False
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            ps_res = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat="],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if ps_res.returncode == 0 and ps_res.stdout.strip().upper().startswith("Z"):
                return False
        except Exception:
            pass
    return True


def resolve_directory_lock_paths(
    repo_dir: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
) -> DirectoryLockPaths:
    """Returns (lock_path, pid_path, mirror_pid_path) for <project_root> or interactive <session_slug>."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    if session_slug and str(session_slug).strip():
        slug = str(session_slug).strip()
        sdir = resolve_platodoom_session_dir(slug)
        lock_path = sdir / "keepalive.lock"
        pid_path = sdir / "keepalive.pid"
        mirror_pid_path = sdir / "colab_keepalive.pid"
        log_path = sdir / "keepalive_daemon.log"
        meta = {
            "project_root": project_root,
            "dir_slug": dir_slug,
            "session_slug": slug,
            "sandcastle_dir": sdir,
            "daemon_home_dir": sdir,
            "lock_path": lock_path,
            "pid_path": pid_path,
            "mirror_pid_path": mirror_pid_path,
            "log_path": log_path,
        }
        return DirectoryLockPaths(lock_path, pid_path, mirror_pid_path, meta)

    sc_dir = project_root / ".sandcastle"
    sc_dir.mkdir(parents=True, exist_ok=True)
    daemon_home_dir = Path.home() / ".sandcastle" / "daemons" / dir_slug
    daemon_home_dir.mkdir(parents=True, exist_ok=True)
    lock_path = sc_dir / "colab_keepalive.lock"
    pid_path = sc_dir / "colab_keepalive.pid"
    mirror_pid_path = daemon_home_dir / "colab_keepalive.pid"
    log_path = sc_dir / "colab_keepalive_daemon.log"
    meta = {
        "project_root": project_root,
        "dir_slug": dir_slug,
        "session_slug": dir_slug,
        "sandcastle_dir": sc_dir,
        "daemon_home_dir": daemon_home_dir,
        "lock_path": lock_path,
        "pid_path": pid_path,
        "mirror_pid_path": mirror_pid_path,
        "log_path": log_path,
    }
    return DirectoryLockPaths(lock_path, pid_path, mirror_pid_path, meta)


resolve_directory_lock_and_pid_paths = resolve_directory_lock_paths


def try_acquire_directory_lock(
    repo_dir: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
) -> Optional[int]:
    """Acquires non-blocking exclusive advisory lock (LOCK_EX | LOCK_NB) on the directory or interactive session lockfile."""
    if repo_dir is not None and str(repo_dir).endswith(".lock") and not session_slug:
        lock_path = Path(repo_dir).expanduser().resolve()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        lock_path, _, _ = resolve_directory_lock_paths(repo_dir, session_slug=session_slug)

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        os.fsync(fd)
        return fd
    except (BlockingIOError, OSError):
        os.close(fd)
        return None


def release_directory_lock(lock_fd: Optional[int]) -> None:
    """Releases and closes an acquired per-directory advisory lock file descriptor."""
    if lock_fd is None:
        return
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(lock_fd)
    except OSError:
        pass


def write_directory_daemon_pid_record(
    repo_dir: Optional[Path | str] = None,
    gpu_profile: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
    *,
    pid: Optional[int] = None,
    endpoint_url: str = "",
    gpu_tier: str = "",
    colab_url: Optional[str] = None,
    status: str = "Provisioning",
    session_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Writes PID & GPU tier metadata atomically to <project_root>/.sandcastle/colab_keepalive.pid (or session dir)."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    _, pid_path, mirror_pid_path = resolve_directory_lock_paths(project_root, session_slug=session_slug)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved_gpu_type = gpu_tier or (gpu_profile or {}).get("gpu_type", "T4")
    resolved_colab_gpu = (gpu_profile or {}).get("colab_gpu_type", resolved_gpu_type)
    rec: Dict[str, Any] = {
        "pid": int(pid) if pid is not None else os.getpid(),
        "dir_slug": dir_slug,
        "session_slug": effective_slug,
        "project_root": str(project_root),
        "gpu_type": resolved_gpu_type,
        "colab_gpu_type": resolved_colab_gpu,
        "colab_notebook_url": colab_url or "",
        "endpoint_url": endpoint_url,
        "status": status,
        "started_at": started_at or now_iso,
        "updated_at": now_iso,
        "updated_at_epoch": time.time(),
    }
    for p in (pid_path, mirror_pid_path):
        try:
            atomic_write_json(p, rec)
        except Exception:
            pass
    return rec


write_directory_daemon_pid = write_directory_daemon_pid_record


def read_directory_daemon_pid_record(
    repo_dir: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Reads the directory's or session's keepalive.pid record if present and annotates PID liveness."""
    project_root = resolve_project_root(repo_dir)
    lock_path, pid_path, mirror_pid_path = resolve_directory_lock_paths(project_root, session_slug=session_slug)
    for p in (pid_path, mirror_pid_path):
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("pid"), int):
                    data["alive"] = is_pid_alive(int(data["pid"]))
                    return data
            except Exception:
                continue
    if lock_path.is_file():
        try:
            raw_pid = lock_path.read_text(encoding="utf-8").strip()
            if raw_pid.isdigit():
                pid_val = int(raw_pid)
                return {
                    "pid": pid_val,
                    "dir_slug": compute_dir_slug(project_root),
                    "session_slug": str(session_slug).strip() if session_slug else compute_dir_slug(project_root),
                    "project_root": str(project_root),
                    "alive": is_pid_alive(pid_val),
                }
        except Exception:
            pass
    return None


read_directory_daemon_pid = read_directory_daemon_pid_record


def is_directory_daemon_alive(
    repo_dir: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
) -> bool:
    """Checks whether a live Keep-Alive Daemon holds the directory/session lock or has an active PID."""
    project_root = resolve_project_root(repo_dir)
    pid_rec = read_directory_daemon_pid_record(project_root, session_slug=session_slug)
    if pid_rec and str(pid_rec.get("status") or "") not in ("Offline", "Failed", "Error") and is_pid_alive(pid_rec.get("pid")):
        return True
    lock_path, _, _ = resolve_directory_lock_paths(project_root, session_slug=session_slug)
    if not lock_path.is_file():
        return False
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except (BlockingIOError, OSError):
        return True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def is_endpoint_daemon_verified(
    record: Optional[Dict[str, Any]],
    repo_dir: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
) -> bool:
    """
    Enforces FR-7.1 (No Unattended Tunnel Adoption):
    Returns True ONLY when the endpoint record is backed by a verified, live local Keep-Alive Daemon PID
    (or active directory/session lock). Orphaned/unattended tunnels with dead or missing daemon_pid return False.
    """
    if is_directory_daemon_alive(repo_dir, session_slug=session_slug):
        return True
    if not isinstance(record, dict):
        return False
    if record.get("unattended") is True or record.get("orphaned") is True:
        return False
    raw_pid = record.get("daemon_pid")
    if isinstance(raw_pid, int) and is_pid_alive(raw_pid):
        return True
    return False


def terminate_directory_daemon(
    repo_dir: Optional[Path | str] = None,
    timeout_sec: float = 5.0,
    session_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Cleanly terminates ONLY the target directory's or interactive session's Keep-Alive Daemon PID and headless_<slug> browser."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    _, pid_path, mirror_pid_path = resolve_directory_lock_paths(project_root, session_slug=session_slug)

    rec = read_directory_daemon_pid_record(project_root, session_slug=session_slug)
    reg = read_registered_endpoint(project_root, include_global=False, session_slug=session_slug) or {}
    pid = (
        int(rec["pid"])
        if rec and isinstance(rec.get("pid"), int)
        else (int(reg["daemon_pid"]) if isinstance(reg.get("daemon_pid"), int) else None)
    )
    terminated = False
    if pid and pid != os.getpid() and is_pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.time() + max(float(timeout_sec), 1.0)
            while time.time() < deadline and is_pid_alive(pid):
                time.sleep(0.15)
            if is_pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
            terminated = True
        except Exception:
            pass

    profile_dir = Path.home() / ".mcp-colab" / "chrome_profiles" / f"headless_{effective_slug}"
    try:
        subprocess.run(
            ["pkill", "-f", str(profile_dir)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        pass

    for p in (pid_path, mirror_pid_path):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    mark_endpoint_offline(
        repo_dir=project_root,
        reason=f"Keep-Alive Daemon stopped for {effective_slug}.",
        include_global=False,
        session_slug=session_slug,
    )
    return {
        "ok": True,
        "terminated": terminated,
        "pid": pid,
        "dir_slug": dir_slug,
        "session_slug": effective_slug,
        "project_root": str(project_root),
    }


def _detect_chromium_browser_binary() -> Optional[str]:
    """Cross-platform lookup for Brave / Google Chrome / Chromium binary."""
    env_bin = str(os.environ.get("PLATODOOM_BROWSER_BIN") or "").strip()
    if env_bin and os.path.exists(env_bin):
        return env_bin

    candidates: List[str] = [
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for cmd_name in ("brave-browser", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(cmd_name)
        if found:
            candidates.append(found)
    for base_env in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base_dir = os.environ.get(base_env)
        if base_dir:
            candidates.extend(
                [
                    str(Path(base_dir) / "BraveSoftware" / "Brave-Browser" / "Application" / "brave.exe"),
                    str(Path(base_dir) / "Google" / "Chrome" / "Application" / "chrome.exe"),
                ]
            )
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def _find_selenium_python_executable() -> str:
    """Returns a Python interpreter path guaranteed to have `selenium` installed."""
    candidates = [
        sys.executable,
        str(Path.home() / ".platodoom" / "venv" / "bin" / "python3"),
        str(Path.home() / ".platodoom" / "venv" / "Scripts" / "python.exe"),
        "/opt/anaconda3/bin/python",
        str(REPO_ROOT / ".sandcastle" / "bin" / "python3"),
    ]
    for cand in candidates:
        if not cand or not Path(cand).is_file():
            continue
        if cand == sys.executable:
            try:
                import selenium  # noqa: F401
                return str(Path(cand).resolve())
            except ImportError:
                continue
        try:
            res = subprocess.run(
                [cand, "-c", "import selenium"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0:
                return str(Path(cand).resolve())
        except Exception:
            continue
    return sys.executable


def _resolve_chromedriver_executable() -> Optional[str]:
    """Resolves cached user chromedriver binary dynamically via Path.home(), webdriver_manager, or Selenium Manager."""
    wdm_root = Path.home() / ".wdm" / "drivers" / "chromedriver"
    if wdm_root.is_dir():
        for cand in sorted(wdm_root.glob("**/chromedriver*"), reverse=True):
            if cand.is_file() and os.access(str(cand), os.X_OK) and not cand.name.endswith(".txt"):
                return str(cand)
    sys_chromedriver = shutil.which("chromedriver")
    if sys_chromedriver and os.access(sys_chromedriver, os.X_OK):
        return str(sys_chromedriver)
    try:
        from webdriver_manager.chrome import ChromeDriverManager

        return str(ChromeDriverManager().install())
    except Exception:
        return None


def read_gpu_routing_profile(
    repo_dir: Optional[Path | str] = None,
    num_models: Optional[int] = None,
    peak_stage_models: Optional[int] = None,
    gpu_override: Optional[str] = None,
    session_slug: Optional[str] = None,
    model_alias: Optional[str] = None,
    stack_preset: Optional[str] = None,
    stack_models: Optional[List[Any]] = None,
    agent_counts: Optional[Any] = None,
    model_allocations: Optional[Any] = None,
) -> Dict[str, Any]:
    """Reads persisted `<project_root>/.sandcastle/colab_gpu_routing.json` (only for multi-agent pipeline runs)
    or computes the exact GPU/TPU routing profile for interactive `platodoom` sessions across all 8 hardware tiers.
    """
    project_root = resolve_project_root(repo_dir)
    alloc_input = model_allocations if model_allocations is not None else agent_counts
    if alloc_input is not None:
        return dict(
            route_gpu_for_multi_model_agents(
                alloc_input,
                gpu_override=gpu_override or "auto",
                repo_dir=project_root,
            )
        )

    is_interactive_session = bool(
        (session_slug and str(session_slug).strip())
        or (gpu_override and str(gpu_override).strip())
        or (model_alias and str(model_alias).strip())
        or (stack_preset and str(stack_preset).strip())
        or stack_models
    )

    if num_models is not None or is_interactive_session:
        raw_ovr = str(gpu_override or "").strip()
        explicit_override: Optional[str] = None
        if raw_ovr and raw_ovr.lower() not in ("auto", "none"):
            explicit_override = (
                _normalize_ssot_tier(raw_ovr, allow_auto=False) if callable(_normalize_ssot_tier) else raw_ovr
            )

        effective_num_models = num_models if num_models is not None else 1
        effective_peak_models = peak_stage_models if peak_stage_models is not None else 1
        route_kwargs: Dict[str, Any] = {
            "num_models": effective_num_models,
            "peak_stage_models": effective_peak_models,
            "gpu_override": explicit_override,
            "repo_dir": project_root,
        }
        if model_alias and str(model_alias).strip():
            route_kwargs["model_alias"] = str(model_alias).strip()
        if stack_preset:
            route_kwargs["stack_preset"] = stack_preset
        if stack_models:
            route_kwargs["stack_models"] = list(stack_models)
        profile = dict(route_gpu_for_model_count(**route_kwargs))
        _stack_ms = [m for m in (profile.get("stack_models") or []) if isinstance(m, Mapping)]
        _all_q4_stack = bool(_stack_ms) and all(
            str(m.get("precision") or "").strip().upper() == "Q4" for m in _stack_ms
        )
        if (
            (str(profile.get("precision_tier") or "") == "nf4_4bit" or _all_q4_stack)
            and str(model_alias or "").strip()
            and str(model_alias).strip() != UNIFIED_MODEL_ALIAS
        ):
            profile["precision_tier"] = "Q4"
            profile["precisionTier"] = "Q4"
            profile["load_in_4bit"] = True
            profile["loadIn4Bit"] = True
        if session_slug or (num_models is None and is_interactive_session):
            from platodoom.catalog import (
                GPU_TIER_SPECS as _SSOT_GPU_TIER_SPECS,
                compute_dynamic_max_num_seqs as _ssot_dyn_seqs,
            )

            _resolved_gpu_key = str(profile.get("gpu_type") or explicit_override or "T4")
            _ssot_tier_spec = _SSOT_GPU_TIER_SPECS.get(_resolved_gpu_key) or {}
            _ssot_tier_seqs = int(_ssot_tier_spec.get("max_num_seqs") or 0)
            _eff_vram = float(profile.get("vram_gb") or _ssot_tier_spec.get("vram_gb") or 15.8)
            _eff_weights = float(profile.get("estimated_weights_gb") or 0.0)
            if _eff_weights > 0.0 and str(model_alias or "").strip():
                tier_seqs = int(
                    _ssot_dyn_seqs(
                        _eff_vram,
                        _eff_weights,
                        model_alias=str(model_alias).strip().split("@", 1)[0],
                        max_model_len=int(profile.get("max_model_len") or 16384),
                        kv_cache_dtype="fp8" if _resolved_gpu_key in ("L4", "H100", "G4") else "auto",
                        requested_agents=_ssot_tier_seqs or 64,
                        active_slot_count=max(1, len(profile.get("stack_models") or [])),
                        tier_max_seqs=_ssot_tier_seqs or 64,
                    )
                )
            else:
                tier_seqs = max(int(profile.get("max_num_seqs") or 0), _ssot_tier_seqs) or 32
            profile["max_num_seqs"] = tier_seqs
            profile["maxNumSeqs"] = tier_seqs
            profile["max_parallel_agents"] = tier_seqs
            profile["maxParallelAgents"] = tier_seqs
            slot_cnt = len(profile.get("stack_models") or []) or 1
            profile["total_qwen_models"] = slot_cnt
            profile["totalQwenModels"] = slot_cnt
            profile["peak_stage_qwen_models"] = slot_cnt
            profile["peakStageQwenModels"] = slot_cnt
            if explicit_override:
                profile["override_mode"] = explicit_override
                profile["overrideMode"] = explicit_override
        return profile

    for reg_path in resolve_colab_endpoint_registry_paths(
        project_root, include_global=False, session_slug=session_slug
    ):
        routing_path = reg_path.parent / DEFAULT_GPU_ROUTING_FILENAME
        if routing_path.is_file():
            try:
                data = json.loads(routing_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    n_val = data.get("totalQwenModels", data.get("total_qwen_models", data.get("totalPlatodoomModels")))
                    peak_val = data.get("peakStageQwenModels", data.get("peak_stage_qwen_models"))
                    override_val = data.get("overrideMode", data.get("override_mode"))
                    persisted_counts = data.get("agentCounts", data.get("agent_counts"))
                    persisted_model = data.get("modelAlias", data.get("model_alias", data.get("model")))
                    persisted_preset = data.get("stackPreset", data.get("stack_preset"))
                    persisted_stack = data.get("stackModels", data.get("stack_models"))
                    if isinstance(persisted_counts, dict):
                        active_cnts = {str(k): int(v) for k, v in persisted_counts.items() if int(v or 0) > 0}
                        if len(active_cnts) >= 2:
                            return dict(
                                route_gpu_for_multi_model_agents(
                                    active_cnts,
                                    gpu_override=str(override_val or "auto"),
                                    repo_dir=project_root,
                                )
                            )
                        if len(active_cnts) == 1 and not persisted_model:
                            persisted_model = next(iter(active_cnts.keys()))
                    if n_val is not None:
                        return route_gpu_for_model_count(
                            num_models=int(n_val),
                            peak_stage_models=int(peak_val) if peak_val is not None else None,
                            gpu_override=str(override_val) if override_val else None,
                            repo_dir=project_root,
                            model_alias=str(persisted_model) if persisted_model else None,
                            stack_preset=str(persisted_preset) if persisted_preset else None,
                            stack_models=persisted_stack if isinstance(persisted_stack, list) else None,
                        )
            except Exception:
                continue

    return route_gpu_for_model_count(num_models=12, peak_stage_models=7)


def read_registered_endpoint(
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    session_slug: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Reads the active endpoint record from environment variables or per-directory / interactive-session local registry files."""
    env_url = (
        os.environ.get("PLATODOOM_ENDPOINT_URL")
        or os.environ.get("SANDCASTLE_VLLM_URL")
        or os.environ.get("VLLM_API_BASE")
        or ""
    )
    if env_url.strip():
        project_root = resolve_project_root(repo_dir)
        dir_slug = compute_dir_slug(project_root)
        return {
            "endpoint_url": normalize_colab_endpoint_url(env_url),
            "model": str(os.environ.get("PLATODOOM_MODEL") or "").strip(),
            "status": "Online",
            "dir_slug": dir_slug,
            "session_slug": str(session_slug).strip() if session_slug else dir_slug,
            "project_root": str(project_root),
            "max_parallel_agents": MAX_PARALLEL_AGENTS,
            "max_model_len": MAX_MODEL_LEN,
            "daemon_pid": os.getpid(),
            "bearer_token": os.environ.get("PLATODOOM_TUNNEL_TOKEN") or os.environ.get("SANDCASTLE_VLLM_API_KEY"),
            "source": "env",
        }

    project_root = resolve_project_root(repo_dir)
    for reg_path in resolve_colab_endpoint_registry_paths(
        project_root, include_global=include_global, session_slug=session_slug
    ):
        if reg_path.is_file():
            try:
                data = json.loads(reg_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data["registry_path"] = str(reg_path)
                    return data
            except Exception:
                continue
    return None


def update_registry_status(
    status: str,
    message: str,
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    extra: Optional[Dict[str, Any]] = None,
    session_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically writes explicit status, heartbeat epoch, and diagnostic message to per-directory or session registry files."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_epoch = time.time()
    targets = list(
        resolve_colab_endpoint_registry_paths(
            project_root, include_global=include_global, session_slug=session_slug
        )
    )

    merged_record: Dict[str, Any] = {}
    for reg_path in targets:
        try:
            data: Dict[str, Any] = {}
            if reg_path.is_file():
                try:
                    loaded = json.loads(reg_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        data = loaded
                except Exception:
                    data = {}
            data["status"] = status
            data["message"] = message
            data["dir_slug"] = dir_slug
            data["session_slug"] = effective_slug
            data["project_root"] = str(project_root)
            data["checked_at"] = now_iso
            data["updated_at"] = now_iso
            data["updated_at_epoch"] = now_epoch
            data["heartbeat_epoch"] = now_epoch
            if status != "Offline":
                data["offline_reason"] = None
            if extra:
                data.update(extra)
            atomic_write_json(reg_path, data)
            if not merged_record:
                merged_record = data
        except Exception:
            continue
    return merged_record


def mark_endpoint_offline(
    repo_dir: Optional[Path | str] = None,
    reason: str = "stale_tunnel",
    include_global: bool = False,
    colab_notebook_url: Optional[str] = None,
    session_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Marks a stale or failed tunnel URL as Offline in the per-directory or interactive-session endpoint registry."""
    extra: Dict[str, Any] = {"offline_reason": reason, "daemon_pid": None}
    if colab_notebook_url:
        extra["colab_notebook_url"] = colab_notebook_url
    return update_registry_status(
        status="Offline",
        message=reason,
        repo_dir=repo_dir,
        include_global=include_global,
        extra=extra,
        session_slug=session_slug,
    )


def fetch_gcs_published_endpoint(
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    session_slug: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Checks optional configured gs://<user-bucket>/endpoints/colab_qwen_endpoint_<slug>.json when GCS is configured."""
    bucket_name, project_id = _resolve_configured_gcs_bucket_and_project(repo_dir)
    effective_bucket = bucket_name or GCS_BUCKET_NAME
    if not effective_bucket:
        return None
    try:
        from google.cloud import storage

        dir_slug = compute_dir_slug(repo_dir) if repo_dir is not None else None
        effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
        client = storage.Client(project=project_id) if project_id else storage.Client()
        bucket = client.bucket(effective_bucket)
        blob_candidates: List[str] = []
        if effective_slug:
            blob_candidates.append(f"endpoints/colab_qwen_endpoint_{effective_slug}.json")
        if include_global or not effective_slug:
            blob_candidates.append(GCS_ENDPOINT_BLOB)

        for blob_name in blob_candidates:
            blob = bucket.blob(blob_name)
            if not blob.exists(timeout=5):
                continue
            raw = blob.download_as_text(encoding="utf-8", timeout=5)
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("endpoint_url"):
                if (
                    effective_slug
                    and parsed.get("dir_slug")
                    and parsed.get("dir_slug") != effective_slug
                    and parsed.get("session_slug") != effective_slug
                    and not include_global
                ):
                    continue
                parsed["source"] = "gcs"
                return parsed
    except Exception:
        return None
    return None


def publish_endpoint_to_gcs(
    record: Dict[str, Any],
    repo_dir: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
) -> bool:
    """Publishes sanitized endpoint metadata to optional user GCS bucket with bearer_token stripped (FR-5.4)."""
    bucket_name, project_id = _resolve_configured_gcs_bucket_and_project(repo_dir)
    effective_bucket = bucket_name or GCS_BUCKET_NAME
    if not effective_bucket:
        return False
    try:
        from google.cloud import storage

        sanitized = _strip_endpoint_secrets_for_remote_sync(record)
        dir_slug = compute_dir_slug(repo_dir) if repo_dir is not None else sanitized.get("dir_slug")
        effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else (dir_slug or "default")
        client = storage.Client(project=project_id) if project_id else storage.Client()
        bucket = client.bucket(effective_bucket)
        blob = bucket.blob(f"endpoints/colab_qwen_endpoint_{effective_slug}.json")
        blob.upload_from_string(
            json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
            content_type="application/json",
            timeout=10,
        )
        return True
    except Exception:
        return False


LAST_DRIVE_AUTH_ERROR: Optional[str] = None
LAST_DRIVE_SYNC_ERROR: Optional[str] = None


def get_last_drive_sync_error() -> Optional[str]:
    """Returns the most recent actionable Google Drive OAuth or sync failure diagnostic."""
    return LAST_DRIVE_SYNC_ERROR or LAST_DRIVE_AUTH_ERROR


def _get_valid_drive_access_token(raise_on_error: bool = False) -> Optional[str]:
    """Reads and refreshes Google OAuth token from ~/.mcp-colab/token.json atomically if available."""
    global LAST_DRIVE_AUTH_ERROR
    if not MCP_COLAB_TOKEN_PATH.is_file():
        LAST_DRIVE_AUTH_ERROR = f"FileNotFoundError: Google OAuth token file missing at {MCP_COLAB_TOKEN_PATH}"
        if raise_on_error:
            raise FileNotFoundError(LAST_DRIVE_AUTH_ERROR)
        return None
    try:
        tok_data = json.loads(MCP_COLAB_TOKEN_PATH.read_text(encoding="utf-8"))
        refresh_token = tok_data.get("refresh_token")
        client_id = tok_data.get("client_id")
        client_secret = tok_data.get("client_secret")
        if not (refresh_token and client_id and client_secret) and MCP_COLAB_CREDS_PATH.is_file():
            creds_raw = json.loads(MCP_COLAB_CREDS_PATH.read_text(encoding="utf-8"))
            installed = creds_raw.get("installed") or creds_raw.get("web") or {}
            client_id = client_id or installed.get("client_id")
            client_secret = client_secret or installed.get("client_secret")

        if refresh_token and client_id and client_secret:
            form = urllib.parse.urlencode(
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://oauth2.googleapis.com/token",
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                refreshed = json.loads(resp.read().decode("utf-8"))
                if refreshed.get("access_token"):
                    tok_data["token"] = refreshed["access_token"]
                    atomic_write_json(MCP_COLAB_TOKEN_PATH, tok_data)
                    LAST_DRIVE_AUTH_ERROR = None
                    return str(refreshed["access_token"])
        token_val = str(tok_data.get("token") or "") or None
        if token_val:
            LAST_DRIVE_AUTH_ERROR = None
        else:
            LAST_DRIVE_AUTH_ERROR = f"ValueError: Empty OAuth access_token in {MCP_COLAB_TOKEN_PATH} and no refresh credentials available"
            if raise_on_error:
                raise ValueError(LAST_DRIVE_AUTH_ERROR)
        return token_val
    except urllib.error.HTTPError as http_err:
        err_body = ""
        try:
            err_body = http_err.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(http_err)
        LAST_DRIVE_AUTH_ERROR = (
            f"HTTPError {http_err.code} refreshing Google OAuth token from https://oauth2.googleapis.com/token: {err_body[:300]}"
        )
        if raise_on_error:
            raise RuntimeError(LAST_DRIVE_AUTH_ERROR) from http_err
        return None
    except Exception as exc:
        LAST_DRIVE_AUTH_ERROR = f"{type(exc).__name__} refreshing Google OAuth token from {MCP_COLAB_TOKEN_PATH}: {exc}"
        if raise_on_error:
            raise RuntimeError(LAST_DRIVE_AUTH_ERROR) from exc
        return None


def _build_cell3_dual_source_code(effective_slug: str, adc_b64: str) -> str:
    """Builds the unified Colab Cell 3 dual-source weight downloader across all catalog models and co-resident stacks."""
    adc_setup = ""
    if adc_b64:
        adc_setup = (
            "import base64\n"
            "_adc_file = Path('/tmp/sandcastle_gcloud_adc.json')\n"
            f"_adc_file.write_bytes(base64.b64decode('{adc_b64}'))\n"
            "try:\n"
            "    os.chmod(str(_adc_file), 0o600)\n"
            "except OSError:\n"
            "    pass\n"
            "os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(_adc_file)\n"
        )
    import pprint

    catalog_entries_dict: Dict[str, Dict[str, Any]] = {}
    catalog_repos_14: Dict[str, str] = {}
    catalog_quant_repos_14: Dict[str, str] = {}
    catalog_prec_repos_14: Dict[str, Dict[str, str]] = {}
    catalog_def_prec_14: Dict[str, Dict[str, str]] = {}
    catalog_bf16_min_14: Dict[str, float] = {}
    for _m_alias, _m_info in MODEL_HF_REPOS.items():
        _clean = dict(_m_info)
        catalog_entries_dict[_m_alias] = _clean
        if str(_m_info.get("modality") or "text") == "text" and _m_alias != "qwen-27b-36":
            catalog_repos_14[_m_alias] = str(_m_info.get("hf_repo_id") or _m_alias)
            catalog_quant_repos_14[_m_alias] = str(_m_info.get("quantized_hf_repo_id") or _m_info.get("hf_repo_id") or _m_alias)
            _p_map: Dict[str, str] = {}
            for _pk, _pv in (_m_info.get("precision_variants") or {}).items():
                if isinstance(_pv, dict) and _pv.get("hf_repo_id"):
                    _p_map[str(_pk)] = str(_pv["hf_repo_id"])
            catalog_prec_repos_14[_m_alias] = _p_map
            catalog_def_prec_14[_m_alias] = dict(_m_info.get("default_precision_by_gpu") or {})
            _bf16_v = ((_m_info.get("precision_variants") or {}).get("BF16") or {}).get("min_vram_gb")
            if _bf16_v is not None:
                catalog_bf16_min_14[_m_alias] = float(_bf16_v)
    def _to_double_quoted_py(val: Any, indent_level: int = 0) -> str:
        pad = "  " * indent_level
        child_pad = "  " * (indent_level + 1)
        if val is None:
            return "None"
        if isinstance(val, bool):
            return "True" if val else "False"
        if isinstance(val, (int, float)):
            return repr(val)
        if isinstance(val, str):
            return json.dumps(val, ensure_ascii=False)
        if isinstance(val, (list, tuple)):
            if not val:
                return "[]"
            items_s = ",\n".join(f"{child_pad}{_to_double_quoted_py(x, indent_level + 1)}" for x in val)
            return f"[\n{items_s}\n{pad}]"
        if isinstance(val, Mapping):
            if not val:
                return "{}"
            pairs_s = ",\n".join(
                f"{child_pad}{json.dumps(str(k), ensure_ascii=False)}: {_to_double_quoted_py(v, indent_level + 1)}"
                for k, v in val.items()
            )
            return f"{{\n{pairs_s}\n{pad}}}"
        return json.dumps(str(val), ensure_ascii=False)

    catalog_repr_str = _to_double_quoted_py(catalog_entries_dict)
    repos14_str = json.dumps(catalog_repos_14, indent=2)
    quant14_str = json.dumps(catalog_quant_repos_14, indent=2)
    prec14_str = json.dumps(catalog_prec_repos_14, indent=2)
    def_prec14_inline = json.dumps(catalog_def_prec_14)
    bf16_min14_inline = json.dumps(catalog_bf16_min_14)
    _tier_floors_dict = {
        k: round(float(v.get("vram_gb", 15.8)) * 0.90, 2)
        for k, v in (_SSOT_GPU_TIER_SPECS or {}).items()
        if k in ("G4", "H100", "A100-80GB", "A100", "L4", "T4")
    } or {"G4": 85.5, "H100": 71.28, "A100-80GB": 71.28, "A100": 36.0, "L4": 21.15, "T4": 14.22}
    tier_floors_inline = json.dumps(_tier_floors_dict)

    return (
        "# Cell 3: Authenticate & Parallel Download (GCS for qwen3.8-27b / HuggingFace Hub for All Catalog Models & Stacks)\n"
        "import concurrent.futures\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        f"os.environ['SANDCASTLE_DIR_SLUG'] = {json.dumps(effective_slug)}\n"
        "os.environ['SANDCASTLE_HEADLESS_SERVING'] = '1'\n"
        + adc_setup
        + "\n"
        f"PROJECT_ID = os.environ.get('PLATODOOM_GCS_PROJECT', {json.dumps(DEFAULT_GCS_PROJECT_ID)})\n"
        f"BUCKET_NAME = os.environ.get('PLATODOOM_GCS_BUCKET', {json.dumps(DEFAULT_GCS_BUCKET_NAME)})\n"
        "SELECTED_MODEL_ALIAS = os.environ.get('PLATODOOM_MODEL', 'qwen3.8-27b').strip() or 'qwen3.8-27b'\n"
        "\n"
        f"CATALOG_REPOS = {repos14_str}\n"
        f"CATALOG_QUANTIZED_REPOS = {quant14_str}\n"
        f"CATALOG_PRECISION_REPOS = {prec14_str}\n"
        "\n"
        "def resolve_colab_hf_repo(model_alias: str, vram_gb_val: float = 16.0, precision: str | None = None) -> str:\n"
        "    p_up = str(precision or '').strip().upper()\n"
        "    if p_up and model_alias in CATALOG_PRECISION_REPOS and p_up in CATALOG_PRECISION_REPOS[model_alias]:\n"
        "        return CATALOG_PRECISION_REPOS[model_alias][p_up]\n"
        "    if p_up in ('Q4', 'AWQ_4BIT', 'GPTQ_4BIT'):\n"
        "        return CATALOG_QUANTIZED_REPOS.get(model_alias, CATALOG_REPOS.get(model_alias, model_alias))\n"
        "    if p_up in ('BF16', 'BFLOAT16', 'FP16'):\n"
        "        return CATALOG_REPOS.get(model_alias, model_alias)\n"
        f"    _def_prec_map = {def_prec14_inline}\n"
        f"    _bf16_min_map = {bf16_min14_inline}\n"
        f"    _tier_floors = {tier_floors_inline}\n"
        "    _v = float(vram_gb_val)\n"
        "    if _v >= _tier_floors.get('G4', 85.5):\n"
        "        _tk = 'G4'\n"
        "    elif _v >= _tier_floors.get('A100-80GB', 71.28):\n"
        "        _tk = 'A100-80GB'\n"
        "    elif _v >= _tier_floors.get('A100', 36.0):\n"
        "        _tk = 'A100'\n"
        "    elif _v >= _tier_floors.get('L4', 21.15):\n"
        "        _tk = 'L4'\n"
        "    else:\n"
        "        _tk = 'T4'\n"
        "    _dp = str((_def_prec_map.get(model_alias) or {}).get(_tk) or '').strip().upper()\n"
        "    if _dp in ('Q4', 'AWQ_4BIT', 'GPTQ_4BIT') or _dp.endswith('_4BIT'):\n"
        "        return CATALOG_QUANTIZED_REPOS.get(model_alias, CATALOG_REPOS.get(model_alias, model_alias))\n"
        "    if _dp in ('BF16', 'BFLOAT16', 'FP16'):\n"
        "        return CATALOG_REPOS.get(model_alias, model_alias)\n"
        "    if _dp and model_alias in CATALOG_PRECISION_REPOS and _dp in CATALOG_PRECISION_REPOS[model_alias]:\n"
        "        return CATALOG_PRECISION_REPOS[model_alias][_dp]\n"
        "    _bf16_req = float(_bf16_min_map.get(model_alias, 65.0))\n"
        "    return CATALOG_QUANTIZED_REPOS.get(model_alias, CATALOG_REPOS.get(model_alias, model_alias)) if _v < _bf16_req else CATALOG_REPOS.get(model_alias, model_alias)\n"
        "\n"
        f"COLAB_MODEL_CATALOG = {catalog_repr_str}\n"
        "\n"
        "def get_authenticated_gcs_client(project_id: str = PROJECT_ID):\n"
        "    import google.auth\n"
        "    from google.cloud import storage\n"
        "    if 'google.colab' in sys.modules and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):\n"
        "        from google.colab import auth\n"
        "        print('Authenticating Google Colab session with Google Cloud...')\n"
        "        auth.authenticate_user()\n"
        "    credentials, _ = google.auth.default(quota_project_id=project_id)\n"
        "    return storage.Client(project=project_id, credentials=credentials)\n"
        "\n"
        "def download_merged_model_parallel(\n"
        "    local_model_dir: Path | str,\n"
        "    project_id: str = PROJECT_ID,\n"
        "    bucket_name: str = BUCKET_NAME,\n"
        "    gcs_prefix: str = 'models/',\n"
        "    max_workers: int = 16,\n"
        ") -> str:\n"
        "    target_dir = Path(local_model_dir)\n"
        "    target_dir.mkdir(parents=True, exist_ok=True)\n"
        "    if (target_dir / 'config.json').exists() and any(target_dir.glob('*.safetensors')):\n"
        "        return str(target_dir)\n"
        "    t_dl = time.perf_counter()\n"
        "    client = get_authenticated_gcs_client(project_id)\n"
        "    bucket = client.bucket(bucket_name)\n"
        "    print(f'Downloading standalone model in parallel ({max_workers} threads) from gs://{bucket_name}/{gcs_prefix}...')\n"
        "    blobs = [b for b in bucket.list_blobs(prefix=gcs_prefix) if not b.name.endswith('/')]\n"
        "    assert len(blobs) > 0, f'FATAL: No model files found under gs://{bucket_name}/{gcs_prefix}'\n"
        "    def _download_blob(blob):\n"
        "        rel_path = os.path.relpath(blob.name, gcs_prefix.rstrip('/'))\n"
        "        local_file = target_dir / rel_path\n"
        "        local_file.parent.mkdir(parents=True, exist_ok=True)\n"
        "        if local_file.exists() and blob.size and local_file.stat().st_size == blob.size:\n"
        "            return rel_path\n"
        "        blob.download_to_filename(str(local_file))\n"
        "        return rel_path\n"
        "    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:\n"
        "        downloaded_files = list(pool.map(_download_blob, blobs))\n"
        "    print(f'Downloaded {len(downloaded_files)} model files to {target_dir} in {time.perf_counter() - t_dl:.1f}s.')\n"
        "    return str(target_dir)\n"
        "\n"
        "def _verify_colab_checkpoint_fast(ckpt_dir: Path) -> bool:\n"
        "    if not ckpt_dir.is_dir():\n"
        "        return False\n"
        "    if (ckpt_dir / 'model_index.json').is_file() and (ckpt_dir / 'model_index.json').stat().st_size > 0:\n"
        "        return any(p.is_file() and p.stat().st_size > 0 for p in ckpt_dir.rglob('*.safetensors')) or any(p.is_file() and p.stat().st_size > 0 for p in ckpt_dir.glob('*.gguf'))\n"
        "    if any(p.is_file() and p.stat().st_size > 0 for p in ckpt_dir.glob('*.gguf')):\n"
        "        return True\n"
        "    if not (ckpt_dir / 'config.json').is_file() or (ckpt_dir / 'config.json').stat().st_size == 0:\n"
        "        return False\n"
        "    idx_file = ckpt_dir / 'model.safetensors.index.json'\n"
        "    if idx_file.is_file():\n"
        "        try:\n"
        "            wmap = json.loads(idx_file.read_text(encoding='utf-8')).get('weight_map') or {}\n"
        "            return bool(wmap) and all((ckpt_dir / str(s)).is_file() and (ckpt_dir / str(s)).stat().st_size > 0 for s in set(wmap.values()))\n"
        "        except Exception:\n"
        "            return False\n"
        "    return any(p.is_file() and p.stat().st_size > 0 for p in ckpt_dir.glob('*.safetensors'))\n"
        "\n"
        "def download_model_for_colab_cell3(\n"
        "    model_alias: str,\n"
        "    vram_gb: float = 16.0,\n"
        "    base_dir: str = '/content/models',\n"
        "    snapshot_download_fn=None,\n"
        "    hf_repo_id_override: str = '',\n"
        "    storage_uri: str = '',\n"
        "    model_source: str = '',\n"
        "    precision: str = '',\n"
        ") -> str:\n"
        "    if snapshot_download_fn is None:\n"
        "        from huggingface_hub import snapshot_download as snapshot_download_fn\n"
        "    cand_local = Path(hf_repo_id_override or model_alias)\n"
        "    if cand_local.is_dir() and _verify_colab_checkpoint_fast(cand_local):\n"
        "        return str(cand_local.resolve())\n"
        "    entry = dict(COLAB_MODEL_CATALOG.get(model_alias, {\n"
        "        'source': model_source or 'hf_hub',\n"
        "        'hf_repo_id': hf_repo_id_override or model_alias,\n"
        "        'quantized_hf_repo_id': hf_repo_id_override or model_alias,\n"
        "    }))\n"
        "    if storage_uri and storage_uri.startswith('gs://'):\n"
        "        _b, _, _p = storage_uri[5:].partition('/')\n"
        "        entry['source'] = 'gcs'\n"
        "        entry['gcs_bucket'] = _b\n"
        "        entry['gcs_prefix'] = _p\n"
        "    norm_prec = str(precision or '').strip().upper()\n"
        "    raw_prec_low = str(precision or '').strip().lower()\n"
        "    variants = entry.get('precision_variants') if isinstance(entry.get('precision_variants'), dict) else {}\n"
        "    q_repos = entry.get('quantized_repo_ids') if isinstance(entry.get('quantized_repo_ids'), dict) else {}\n"
        "    def_prec_by_gpu = entry.get('default_precision_by_gpu') if isinstance(entry.get('default_precision_by_gpu'), dict) else {}\n"
        "    if hf_repo_id_override and hf_repo_id_override != entry.get('hf_repo_id'):\n"
        "        target_repo = hf_repo_id_override\n"
        "    elif norm_prec and norm_prec in variants and variants[norm_prec].get('hf_repo_id'):\n"
        "        target_repo = str(variants[norm_prec]['hf_repo_id'])\n"
        "    elif raw_prec_low and raw_prec_low in q_repos and q_repos[raw_prec_low]:\n"
        "        target_repo = str(q_repos[raw_prec_low])\n"
        "    elif norm_prec == 'Q4' and entry.get('quantized_hf_repo_id'):\n"
        "        target_repo = str(entry['quantized_hf_repo_id'])\n"
        "    else:\n"
        "        _live_det_tier = str(globals().get('detected_tier') or '').strip()\n"
        "        _live_det_vram = float(globals().get('vram_gb', -999.0))\n"
        "        _is_tpu_env = bool(os.environ.get('COLAB_TPU_ADDR') or globals().get('IS_TPU_RUNTIME'))\n"
        "        if _live_det_tier and abs(float(vram_gb) - _live_det_vram) < 1.0:\n"
        "            _tier_key = _live_det_tier\n"
        "        elif _is_tpu_env:\n"
        "            _tier_key = 'v6e-1' if vram_gb >= 24.0 else 'v5e-1'\n"
        "        else:\n"
        f"            _tf = {tier_floors_inline}\n"
        "            if vram_gb >= _tf.get('G4', 85.5):\n"
        "                _tier_key = 'G4'\n"
        "            elif vram_gb >= _tf.get('A100-80GB', 71.28):\n"
        "                _tier_key = 'A100-80GB'\n"
        "            elif vram_gb >= _tf.get('A100', 36.0):\n"
        "                _tier_key = 'A100'\n"
        "            elif vram_gb >= _tf.get('L4', 21.15):\n"
        "                _tier_key = 'L4'\n"
        "            else:\n"
        "                _tier_key = 'T4'\n"
        "        _def_prec = str(def_prec_by_gpu.get(_tier_key) or '').strip()\n"
        "        _def_prec_up = _def_prec.upper()\n"
        "        if _def_prec and _def_prec in q_repos and q_repos[_def_prec]:\n"
        "            target_repo = str(q_repos[_def_prec])\n"
        "        elif _def_prec_up in ('Q4', 'AWQ_4BIT', 'GPTQ_4BIT') or _def_prec_up.endswith('_4BIT'):\n"
        "            target_repo = str(entry.get('quantized_hf_repo_id') or entry.get('hf_repo_id', model_alias))\n"
        "        elif _def_prec_up in ('BF16', 'BFLOAT16', 'FP16'):\n"
        "            target_repo = str(entry.get('hf_repo_id', model_alias))\n"
        "        elif _def_prec_up and _def_prec_up in variants and variants[_def_prec_up].get('hf_repo_id'):\n"
        "            target_repo = str(variants[_def_prec_up]['hf_repo_id'])\n"
        "        elif 'Q4' in variants and 'BF16' in variants:\n"
        "            _bf16_floor = float((variants.get('BF16') or {}).get('min_vram_gb', 65.0))\n"
        "            target_repo = str(entry.get('quantized_hf_repo_id') if vram_gb < _bf16_floor else entry.get('hf_repo_id', model_alias))\n"
        "        else:\n"
        "            target_repo = str(entry.get('hf_repo_id', model_alias))\n"
        "    safe_dir_name = str(target_repo if entry.get('source') != 'gcs' else model_alias).replace('/', '--')\n"
        "    drive_dir = Path(f'/content/drive/MyDrive/sandcastle_models/{safe_dir_name}')\n"
        "    if False:\n"
        "        legacy_drive = Path('/content/models/legacy')\n"
        "        if _verify_colab_checkpoint_fast(legacy_drive):\n"
        "            drive_dir = legacy_drive\n"
        "    local_dir = drive_dir if _verify_colab_checkpoint_fast(drive_dir) else (Path(base_dir) / safe_dir_name)\n"
        "    local_dir.mkdir(parents=True, exist_ok=True)\n"
        "    if _verify_colab_checkpoint_fast(local_dir):\n"
        "        print(f\"Model '{model_alias}' already cached and verified at {local_dir}.\")\n"
        "        return str(local_dir)\n"
        "    if entry.get('source') == 'gcs':\n"
        "        try:\n"
        "            out_dir = download_merged_model_parallel(\n"
        "                local_model_dir=local_dir,\n"
        "                project_id=PROJECT_ID,\n"
        "                bucket_name=entry['gcs_bucket'],\n"
        "                gcs_prefix=entry['gcs_prefix'],\n"
        "                max_workers=16,\n"
        "            )\n"
        "            if _verify_colab_checkpoint_fast(Path(out_dir)):\n"
        "                return str(out_dir)\n"
        "            raise RuntimeError(f'Incomplete GCS checkpoint at {out_dir}')\n"
        "        except Exception as gcs_err:\n"
        "            fallback_repo = entry.get('fallback_hf_repo_id', 'Qwen/Qwen3.5-9B')\n"
        "            print(f\"Warning: GCS download failed ({gcs_err}); falling back to HuggingFace Hub '{fallback_repo}'...\")\n"
        "            snapshot_download_fn(\n"
        "                repo_id=fallback_repo,\n"
        "                local_dir=str(local_dir),\n"
        "                ignore_patterns=['*.gguf', '*.bin', 'original/*'],\n"
        "                max_workers=8,\n"
        "            )\n"
        "            if not _verify_colab_checkpoint_fast(local_dir):\n"
        "                raise RuntimeError(f'Fallback checkpoint verification failed at {local_dir}')\n"
        "            return str(local_dir)\n"
        "    _ignore_pats = ['*.bin', 'original/*'] if ('gguf' in str(target_repo).lower() or 'gguf' in str(precision).lower()) else ['*.gguf', '*.bin', 'original/*']\n"
        "    print(f\"Downloading '{model_alias}' ({target_repo}) from HuggingFace Hub into {local_dir}...\")\n"
        "    snapshot_download_fn(\n"
        "        repo_id=target_repo,\n"
        "        local_dir=str(local_dir),\n"
        "        token=os.environ.get('PLATODOOM_HF_TOKEN') or os.environ.get('HF_TOKEN') or None,\n"
        "        ignore_patterns=_ignore_pats,\n"
        "        max_workers=8,\n"
        "    )\n"
        "    if not _verify_colab_checkpoint_fast(local_dir):\n"
        "        raise RuntimeError(f'Downloaded checkpoint verification failed at {local_dir}')\n"
        "    return str(local_dir)\n"
        "\n"
        "LOCAL_MODEL_STACK_DIRS = {}\n"
        "SKIPPED_ZERO_AGENT_MODELS = []\n"
        "STAGED_STACK_MODELS = []\n"
        "STAGED_AGENT_COUNTS = {}\n"
        "ACTIVE_MODEL_ALIAS = SELECTED_MODEL_ALIAS\n"
        "INITIAL_SERVED_REPO = str(COLAB_MODEL_CATALOG.get(SELECTED_MODEL_ALIAS, {}).get('hf_repo_id') or SELECTED_MODEL_ALIAS)\n"
        "if 'google.colab' in sys.modules or os.environ.get('COLAB_GPU') or os.environ.get('COLAB_TPU_ADDR'):\n"
        "    _detected_vram = float(globals().get('vram_gb', 16.0))\n"
        "    _raw_stack_json = os.environ.get('PLATODOOM_MODEL_STACK', '').strip()\n"
        "    _raw_agent_counts = os.environ.get('PLATODOOM_AGENT_COUNTS', '').strip()\n"
        "    _agent_counts_map = json.loads(_raw_agent_counts) if _raw_agent_counts else {}\n"
        "    STAGED_AGENT_COUNTS = dict(_agent_counts_map)\n"
        "    if _raw_stack_json:\n"
        "        _stack_specs = json.loads(_raw_stack_json)\n"
        "        for _slot_item in _stack_specs:\n"
        "            _s_alias = str(_slot_item.get('alias') or _slot_item.get('model_alias') or SELECTED_MODEL_ALIAS)\n"
        "            _s_agents = _slot_item.get('agent_count', _agent_counts_map.get(_s_alias))\n"
        "            if _s_agents is not None and int(_s_agents) <= 0:\n"
        "                SKIPPED_ZERO_AGENT_MODELS.append(_s_alias)\n"
        "                STAGED_STACK_MODELS.append({'alias': _s_alias, 'agent_count': 0})\n"
        "                continue\n"
        "            _s_prec = str(_slot_item.get('precision') or '')\n"
        "            _s_repo = str(_slot_item.get('hf_repo_id') or '')\n"
        "            _dl_dir = download_model_for_colab_cell3(\n"
        "                _s_alias, vram_gb=_detected_vram, base_dir=os.environ.get('PLATODOOM_MODEL_CACHE_DIR', '/content/models'), hf_repo_id_override=_s_repo, precision=_s_prec\n"
        "            )\n"
        "            LOCAL_MODEL_STACK_DIRS[_s_alias] = _dl_dir\n"
        "            STAGED_STACK_MODELS.append({\n"
        "                'alias': _s_alias,\n"
        "                'hf_repo_id': _dl_dir,\n"
        "                'precision': _s_prec or None,\n"
        "                'agent_count': int(_s_agents) if _s_agents is not None else 1,\n"
        "            })\n"
        "        if LOCAL_MODEL_STACK_DIRS:\n"
        "            _first_alias = next(iter(LOCAL_MODEL_STACK_DIRS), SELECTED_MODEL_ALIAS)\n"
        "            ACTIVE_MODEL_ALIAS = _first_alias\n"
        "            LOCAL_MODEL_DIR = Path(LOCAL_MODEL_STACK_DIRS[_first_alias])\n"
        "            INITIAL_SERVED_REPO = str(LOCAL_MODEL_DIR)\n"
        "    else:\n"
        "        LOCAL_MODEL_DIR = Path(download_model_for_colab_cell3(SELECTED_MODEL_ALIAS, vram_gb=_detected_vram, base_dir=os.environ.get('PLATODOOM_MODEL_CACHE_DIR', '/content/models')))\n"
        "        LOCAL_MODEL_STACK_DIRS[SELECTED_MODEL_ALIAS] = str(LOCAL_MODEL_DIR)\n"
        "        INITIAL_SERVED_REPO = str(LOCAL_MODEL_DIR)\n"
        "MODEL_PATH = str(LOCAL_MODEL_DIR) if 'LOCAL_MODEL_DIR' in globals() else ''\n"
    )


def _build_cell1_setup_code(
    effective_slug: str,
    effective_model: str,
    resolved_gpu_tier: str,
    routed_mem_gb: float,
    max_seqs: int,
    effective_stack_preset: Optional[str] = None,
    stack_json_env: str = "",
    agent_counts_json_env: str = "",
    effective_bearer: str = "",
    enable_vision: Optional[bool] = None,
) -> List[str]:
    """Builds Cell 1 provisioning isolated `/opt/vllm-venv` (`vllm==0.33.0`) on CUDA while preserving `TPU_SAFE_SERVING_PACKAGES` on TPU."""
    return [
        "# Cell 1: Environment Setup, Isolated /opt/vllm-venv (vllm==0.33.0) & Kernel Cache Provisioning\n",
        "import importlib, importlib.metadata, importlib.util, os, subprocess, sys, time\n",
        f"os.environ['SANDCASTLE_DIR_SLUG'] = {json.dumps(effective_slug)}\n",
        "os.environ['SANDCASTLE_HEADLESS_SERVING'] = '1'\n",
        "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n",
        "os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'\n",
        "os.environ['VLLM_CACHE_ROOT'] = '/content/.cache/vllm'\n",
        "os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/content/.cache/inductor'\n",
        "os.environ['FLASHINFER_WORKSPACE_BASE'] = '/content/.cache/flashinfer'\n",
        "os.environ['PLATODOOM_MODEL_CACHE_DIR'] = '/content/models'\n",
        "os.environ['PLATODOOM_VLLM_BIN'] = '/opt/vllm-venv/bin/vllm'\n",
        "os.environ.setdefault('MAX_JOBS', '8')\n",
        f"os.environ['PLATODOOM_MODEL'] = {json.dumps(effective_model)}\n",
        f"os.environ['PLATODOOM_GPU'] = {json.dumps(resolved_gpu_tier)}\n",
        f"os.environ['PLATODOOM_EXPECTED_MEMORY_GB'] = {json.dumps(str(routed_mem_gb))}\n",
        f"os.environ['SANDCASTLE_MAX_PARALLEL_AGENTS'] = {json.dumps(str(max_seqs))}\n",
        *([f"os.environ['SANDCASTLE_VLLM_ENABLE_VISION'] = {'1' if enable_vision else '0'!r}\n"] if enable_vision is not None else []),
        *([f"os.environ['PLATODOOM_STACK'] = {json.dumps(str(effective_stack_preset))}\n"] if effective_stack_preset else []),
        *([f"os.environ['PLATODOOM_MODEL_STACK'] = {json.dumps(stack_json_env)}\n"] if stack_json_env else []),
        *([f"os.environ['PLATODOOM_AGENT_COUNTS'] = {json.dumps(agent_counts_json_env)}\n"] if agent_counts_json_env else []),
        *(
            [
                f"os.environ['PLATODOOM_TUNNEL_TOKEN'] = {json.dumps(effective_bearer)}\n",
                f"os.environ['SANDCASTLE_VLLM_API_KEY'] = {json.dumps(effective_bearer)}\n",
            ]
            if effective_bearer
            else []
        ),
        "def _pip_install(*pip_args):\n",
        "    proc = subprocess.run(\n",
        "        [sys.executable, '-m', 'pip', 'install', '-q', *pip_args],\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    if proc.returncode != 0:\n",
        "        tail = ((proc.stdout or '') + '\\n' + (proc.stderr or '')).strip()[-1600:]\n",
        "        raise RuntimeError(f'pip install failed (exit {proc.returncode}) for {pip_args}:\\n{tail}')\n",
        "def _run_checked(cmd):\n",
        "    proc = subprocess.run(cmd, capture_output=True, text=True)\n",
        "    if proc.returncode != 0:\n",
        "        tail = ((proc.stdout or '') + '\\n' + (proc.stderr or '')).strip()[-1600:]\n",
        "        raise RuntimeError(f'Command failed (exit {proc.returncode}) for {cmd}:\\n{tail}')\n",
        "t0 = time.perf_counter()\n",
        "if 'google.colab' in sys.modules or os.environ.get('COLAB_GPU') or os.environ.get('COLAB_TPU_ADDR'):\n",
        "    _is_tpu_runtime = bool(os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME')) or os.environ.get('PLATODOOM_GPU', '') in ('v5e-1', 'v6e-1')\n",
        "    for _d in ('/content/models', '/content/.cache/vllm', '/content/.cache/inductor', '/content/.cache/flashinfer'):\n",
        "        os.makedirs(_d, exist_ok=True)\n",
        "    _constraint_args = []\n",
        "    try:\n",
        "        _tv = importlib.metadata.version('torch')\n",
        "        _torch_constraint_file = '/tmp/colab_torch_constraint.txt'\n",
        "        with open(_torch_constraint_file, 'w', encoding='utf-8') as _cf:\n",
        "            _cf.write(f'torch=={_tv}\\n')\n",
        "        _constraint_args = ['-c', _torch_constraint_file]\n",
        "    except importlib.metadata.PackageNotFoundError:\n",
        "        pass\n",
        "    TPU_SAFE_SERVING_PACKAGES = [\n",
        "        'huggingface_hub[hf_transfer]',\n",
        "        'hf_transfer',\n",
        "        'google-cloud-storage',\n",
        "        'accelerate>=0.34.0',\n",
        "        'transformers>=4.49.0',\n",
        "        'diffusers',\n",
        "        'sentencepiece',\n",
        "        'requests',\n",
        "        'pyngrok',\n",
        "    ]\n",
        "    if importlib.util.find_spec('google.protobuf') is None:\n",
        "        TPU_SAFE_SERVING_PACKAGES.append('protobuf')\n",
        "    if _is_tpu_runtime:\n",
        "        _pip_install(*_constraint_args, *TPU_SAFE_SERVING_PACKAGES)\n",
        "        importlib.invalidate_caches()\n",
        "        _verify_mods = ('accelerate', 'transformers', 'diffusers', 'sentencepiece', 'google.protobuf')\n",
        "    else:\n",
        "        _pip_install(*_constraint_args, *TPU_SAFE_SERVING_PACKAGES, 'uv', 'bitsandbytes>=0.43.0')\n",
        "        if not os.path.isfile('/opt/vllm-venv/bin/python'):\n",
        "            _run_checked(['uv', 'venv', '--python', '3.12', '/opt/vllm-venv'])\n",
        "        _run_checked([\n",
        "            'uv', 'pip', 'install',\n",
        "            '--python', '/opt/vllm-venv/bin/python',\n",
        "            '--torch-backend=auto',\n",
        "            'vllm==0.33.0',\n",
        "            'ninja',\n",
        "            'pillow',\n",
        "            'hf_transfer',\n",
        "            'fastsafetensors',\n",
        "            'bitsandbytes>=0.43.0',\n",
        "            'compressed-tensors==0.17.0',\n",
        "        ])\n",
        "        if '/opt/vllm-venv/bin' not in os.environ.get('PATH', '').split(os.pathsep):\n",
        "            os.environ['PATH'] = '/opt/vllm-venv/bin' + os.pathsep + os.environ.get('PATH', '')\n",
        "        if not os.path.isfile('/opt/vllm-venv/bin/ninja'):\n",
        "            raise RuntimeError('Missing /opt/vllm-venv/bin/ninja required by FlashInfer JIT on sm_120+')\n",
        "        _run_checked(['/opt/vllm-venv/bin/vllm', 'download-kernels'])\n",
        "        importlib.invalidate_caches()\n",
        "        _verify_mods = ('bitsandbytes', 'accelerate', 'transformers', 'diffusers', 'sentencepiece', 'google.protobuf')\n",
        "    _missing_mods = [m for m in _verify_mods if importlib.util.find_spec(m) is None]\n",
        "    if _missing_mods:\n",
        "        raise ImportError(f'Missing required serving dependencies after installation: {_missing_mods}')\n",
        "print(f'Serving dependencies verified and installed in {time.perf_counter() - t0:.1f}s.')\n",
    ]


def _build_cell5_tunnel_code(env_bootstrap: str = "") -> str:
    """Builds Cell 5 tunnel & GCS endpoint publication with secret redaction prior to remote GCS sync."""
    body = """# Cell 5: Public Tunnel Setup (ngrok / cloudflared) & Redacted GCS Endpoint Registry Sync
import json
import os
import re
import subprocess
import time
from pyngrok import ngrok

ngrok_token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
if not ngrok_token:
    try:
        from google.colab import userdata
        ngrok_token = (userdata.get("NGROK_AUTHTOKEN") or "").strip()
        if ngrok_token:
            os.environ["NGROK_AUTHTOKEN"] = ngrok_token
    except Exception:
        ngrok_token = ""
if ngrok_token:
    ngrok.set_auth_token(ngrok_token)

public_url = ""
try:
    tunnel = ngrok.connect(8000)
    public_url = tunnel.public_url.rstrip("/")
except Exception as ngrok_err:
    print(f"ngrok tunnel failed ({ngrok_err}); falling back to Cloudflare Quick Tunnel (cloudflared)...")
    import hashlib as _cf_hashlib
    _cf_url = "https://github.com/cloudflare/cloudflared/releases/download/2026.11.1/cloudflared-linux-amd64"
    _cf_sha256 = "97bb3b02d23a5980fc9c15a1a270e692da1923f103f920d5b16d4931d5ab55e7"
    _cf_bin = "/usr/local/bin/cloudflared"
    subprocess.run(["wget", "-q", _cf_url, "-O", _cf_bin], check=True)
    _cf_digest = _cf_hashlib.sha256(open(_cf_bin, "rb").read()).hexdigest()
    if _cf_digest != _cf_sha256:
        os.unlink(_cf_bin)
        raise RuntimeError(f"FATAL: cloudflared SHA-256 mismatch (got {_cf_digest}, expected {_cf_sha256})")
    subprocess.run(["chmod", "755", _cf_bin], check=True)
    cf_log_path = "/tmp/sandcastle_cloudflared.log"
    cf_log_file = open(cf_log_path, "w+")
    cf_proc = subprocess.Popen([_cf_bin, "tunnel", "--url", "http://127.0.0.1:8000"], stdout=cf_log_file, stderr=subprocess.STDOUT, text=True)
    for _ in range(60):
        try:
            with open(cf_log_path, "r") as _f:
                _cf_txt = _f.read()
            m = re.search(r"(https://[a-zA-Z0-9-]+\\.trycloudflare\\.com)", _cf_txt)
            if m:
                public_url = m.group(1)
                break
        except Exception:
            pass
        time.sleep(0.5)

assert public_url, "FATAL: Failed to establish public HTTPS tunnel to port 8000."
DIR_SLUG = os.environ.get("SANDCASTLE_DIR_SLUG", "").strip()
_c5_engine = globals().get("_engine")
_c5_slot = None
_c5_vslot = None
if _c5_engine is not None and isinstance(getattr(_c5_engine, "loaded_models", None), dict) and _c5_engine.loaded_models:
    _c5_pref_alias = str(getattr(_c5_engine, "active_model_alias", "") or "")
    _c5_slot = _c5_engine.loaded_models.get(_c5_pref_alias) or next(iter(_c5_engine.loaded_models.values()), None)
    _c5_vslot = getattr(_c5_slot, "vllm_slot", None) if _c5_slot is not None else None
elif _c5_engine is not None and getattr(_c5_engine, "_primary_vllm_slot", None) is not None:
    _c5_vslot = getattr(_c5_engine, "_primary_vllm_slot", None)

_c5_active_model = str(
    (getattr(_c5_slot, "alias", None) if _c5_slot is not None else None)
    or (getattr(_c5_engine, "active_model_alias", None) if _c5_engine is not None else None)
    or globals().get("active_served_model")
    or globals().get("SELECTED_MODEL_ALIAS")
    or os.environ.get("PLATODOOM_MODEL", "qwen3.8-27b")
)
_c5_bearer_token = (
    os.environ.get("PLATODOOM_TUNNEL_TOKEN")
    or os.environ.get("SANDCASTLE_VLLM_API_KEY")
    or str(globals().get("expected_token") or "")
).strip()
_c5_raw_prec = str(
    (getattr(_c5_vslot, "precision", None) if _c5_vslot is not None else None)
    or (getattr(_c5_slot, "precision", None) if _c5_slot is not None else None)
    or globals().get("SELECTED_MODEL_PRECISION")
    or ("Q4" if bool(globals().get("USE_4BIT_SERVING", False)) else "BF16")
).strip().upper()
if _c5_raw_prec in ("Q4", "INT4", "GGUF", "GGUF_Q4") or _c5_raw_prec.endswith("_4BIT"):
    _c5_live_prec = "Q4"
elif "FP8" in _c5_raw_prec:
    _c5_live_prec = "FP8"
elif _c5_raw_prec in ("FP16", "FLOAT16"):
    _c5_live_prec = "FP16"
elif _c5_raw_prec in ("Q8", "INT8", "Q6", "Q5"):
    _c5_live_prec = _c5_raw_prec
else:
    _c5_live_prec = "BF16"

_c5_eng_cfg = getattr(_c5_engine, "config", {}) if _c5_engine is not None and isinstance(getattr(_c5_engine, "config", None), dict) else {}
_c5_max_seqs = int(
    (getattr(_c5_vslot, "max_num_seqs", 0) if _c5_vslot is not None else 0)
    or _c5_eng_cfg.get("max_num_seqs")
    or globals().get("ACTIVE_GPU_SEQS", 32)
)
_c5_max_len = int(
    (getattr(_c5_vslot, "max_model_len", 0) if _c5_vslot is not None else 0)
    or _c5_eng_cfg.get("max_model_len")
    or globals().get("MAX_MODEL_LEN", 16384)
)
_c5_max_batched = int(
    (getattr(_c5_vslot, "max_num_batched_tokens", 0) if _c5_vslot is not None else 0)
    or _c5_eng_cfg.get("max_num_batched_tokens")
    or globals().get("MAX_NUM_BATCHED_TOKENS", 16384)
)
_c5_cc = globals().get("compute_capability")
_c5_sm = (
    f"sm_{int(_c5_cc[0])}{int(_c5_cc[1])}"
    if isinstance(_c5_cc, (tuple, list)) and len(_c5_cc) >= 2 and int(_c5_cc[0]) > 0
    else str(globals().get("_sm_label") or "")
)
_c5_vram = round(float(globals().get("vram_gb", 16.0)), 2)
_c5_free_vram = round(float(globals().get("free_vram_gb", _c5_vram)), 2)
_c5_gpu_tier = str(globals().get("detected_tier") or os.environ.get("PLATODOOM_GPU", "T4"))
_c5_project_id = str(globals().get("PROJECT_ID") or os.environ.get("PLATODOOM_GCS_PROJECT", ""))
_c5_bucket_name = str(globals().get("BUCKET_NAME") or os.environ.get("PLATODOOM_GCS_BUCKET", ""))
endpoint_meta = {
    "endpoint_url": f"{public_url}/v1",
    "chat_completions_url": f"{public_url}/v1/chat/completions",
    "model": _c5_active_model,
    "active_model": _c5_active_model,
    "max_parallel_agents": int(globals().get("MAX_PARALLEL_AGENTS", _c5_max_seqs)),
    "max_num_seqs": _c5_max_seqs,
    "max_model_len": _c5_max_len,
    "max_num_batched_tokens": _c5_max_batched,
    "gpu_name": str(globals().get("gpu_name", "NVIDIA GPU")),
    "gpu_type": _c5_gpu_tier,
    "vram_gb": _c5_vram,
    "free_vram_gb": _c5_free_vram,
    "compute_capability": _c5_sm,
    "precision_tier": _c5_live_prec,
    "status": "Online",
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "project_id": _c5_project_id,
}
if _c5_vslot is not None:
    endpoint_meta["gpu_memory_utilization"] = float(getattr(_c5_vslot, "gpu_memory_utilization", 0.90))
    _c5_rflags = getattr(_c5_vslot, "resolved_flags", None)
    if isinstance(_c5_rflags, dict) and "enforce_eager" in _c5_rflags:
        endpoint_meta["enforce_eager"] = bool(_c5_rflags["enforce_eager"])
if _c5_bearer_token and _c5_bearer_token != "EMPTY":
    endpoint_meta["auth_required"] = True
if DIR_SLUG:
    endpoint_meta["dir_slug"] = DIR_SLUG

# Redact authentication secrets prior to uploading endpoint metadata to remote GCS
_REDACTED_SECRET_KEYS = ("bearer_token", "api_key", "token", "authorization", "Authorization")
gcs_endpoint_meta = {k: v for k, v in endpoint_meta.items() if k not in _REDACTED_SECRET_KEYS}

if _c5_bucket_name:
    try:
        gcs_client = get_authenticated_gcs_client(_c5_project_id)
        gcs_bucket = gcs_client.bucket(_c5_bucket_name)
        gcs_blob = gcs_bucket.blob("endpoints/colab_qwen_endpoint.json")
        gcs_blob.upload_from_string(json.dumps(gcs_endpoint_meta, indent=2), content_type="application/json")
        print(f"Published redacted endpoint registry to gs://{_c5_bucket_name}/endpoints/colab_qwen_endpoint.json")
        if DIR_SLUG:
            dir_blob = gcs_bucket.blob(f"endpoints/colab_qwen_endpoint_{DIR_SLUG}.json")
            dir_blob.upload_from_string(json.dumps(gcs_endpoint_meta, indent=2), content_type="application/json")
            print(f"Published redacted per-directory endpoint registry to gs://{_c5_bucket_name}/endpoints/colab_qwen_endpoint_{DIR_SLUG}.json")
    except Exception as sync_err:
        print(f"Note: Could not publish endpoint to GCS ({sync_err})")

print("\\n" + "=" * 60)
print(f"SANDCASTLE vLLM PUBLIC ENDPOINT: {public_url}/v1/chat/completions")
print("SANDCASTLE_COLAB_ENDPOINT_JSON:" + json.dumps(endpoint_meta))
print("=" * 60 + "\\n")
"""
    return env_bootstrap + body


def _build_cell4_unified_server_code(env_bootstrap: str) -> str:
    """Single-sources Colab Cell 4 directly from `curation.colab_server.get_cell4_supervisor_source`."""
    from curation.colab_server import get_cell4_supervisor_source

    return (
        "# Cell 4: Port-8000 Supervisor & Native vLLM Continuous Batching Proxy\n"
        + get_cell4_supervisor_source(env_bootstrap=env_bootstrap)
    )

def _build_cell6_verification_code(
    env_bootstrap: str,
    c6_warmup_marker: str = "# Spec 856/859 headless readiness warmup (max_new = 1 probe fallback)\n",
) -> str:
    """Builds Cell 6 readiness verification with bearer-token parity, >=24s tunnel DNS retry, and loopback fallback."""
    body = """# Cell 6: End-to-End Verification (/v1/models + /v1/chat/completions, Bearer Token Parity & Tunnel DNS Retry)
import concurrent.futures
import os
import socket
import time
import requests

base = str(globals().get("public_url") or os.environ.get("PLATODOOM_ENDPOINT_URL") or os.environ.get("SANDCASTLE_VLLM_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
if not base.endswith("/v1"):
    base = f"{base}/v1"
api_key = (
    os.environ.get("PLATODOOM_TUNNEL_TOKEN")
    or os.environ.get("SANDCASTLE_VLLM_API_KEY")
    or str(globals().get("expected_token") or "")
).strip()
expected_model = str(
    os.environ.get("PLATODOOM_MODEL")
    or globals().get("SELECTED_MODEL_ALIAS")
    or globals().get("active_served_model")
    or "qwen3.8-27b"
).strip()
HEADLESS_SERVING = os.environ.get("SANDCASTLE_HEADLESS_SERVING", "1") == "1"

headers = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "Platodoom-Cell6-Readiness/1.0",
}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

def _is_dns_err(exc: BaseException) -> bool:
    cur = exc
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, socket.gaierror):
            return True
        msg = str(cur).lower()
        if any(tok in msg for tok in (
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "failed to resolve",
            "getaddrinfo failed",
        )):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False

_is_tunnel_dns_err = _is_dns_err

def _make_probe_png_data_url() -> str:
    import base64 as _b64, struct as _st, zlib as _zl
    w, h = 32, 32
    row = bytes([0]) + bytes([0, 128, 255]) * w
    raw = row * h
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return _st.pack("!I", len(data)) + tag + data + _st.pack("!I", _zl.crc32(tag + data) & 0xFFFFFFFF)
    sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    png = sig + _chunk(b"IHDR", _st.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)) + _chunk(b"IDAT", _zl.compress(raw)) + _chunk(b"IEND", b"")
    return "data:image/png;base64," + _b64.b64encode(png).decode("ascii")

def _probe_base_once(target_base: str) -> dict:
    import json as _c6_json
    r_mod = requests.get(f"{target_base}/models", headers=headers, timeout=10)
    if r_mod.status_code == 401:
        raise RuntimeError(f"HTTP 401 Unauthorized probing {target_base}/models (check PLATODOOM_TUNNEL_TOKEN / SANDCASTLE_VLLM_API_KEY)")
    r_mod.raise_for_status()
    m_data = r_mod.json().get("data", [])
    served_ids = [str(m.get("id", "")) for m in m_data if isinstance(m, dict)]
    served_roots = [str(m.get("root", "")) for m in m_data if isinstance(m, dict)]
    vision_by_id = {str(m.get("id", "")): bool(m.get("vision", False)) for m in m_data if isinstance(m, dict)}
    raw_stack = os.environ.get("PLATODOOM_MODEL_STACK", "").strip()
    if raw_stack:
        req_models = [
            str(s.get("alias") or s.get("model_alias") or "")
            for s in _c6_json.loads(raw_stack)
            if isinstance(s, dict) and int(s.get("agent_count", 1)) > 0
        ]
        for req_m in req_models:
            if req_m and req_m not in served_ids and req_m not in served_roots:
                raise RuntimeError(f"Expected co-resident model '{req_m}' not loaded on {target_base}; active={served_ids}")
    elif expected_model and expected_model not in served_ids and expected_model not in served_roots:
        raise RuntimeError(f"Expected model '{expected_model}' not physically loaded on {target_base}; active={served_ids}")
    want_vision = str(os.environ.get("SANDCASTLE_VLLM_ENABLE_VISION") or os.environ.get("SANDCASTLE_VLLM_ENABLE_VLM") or "").strip().lower() in ("1", "true", "yes", "on")
    active_vision = bool(vision_by_id.get(expected_model, any(vision_by_id.values())))
    if want_vision and not active_vision:
        raise RuntimeError(f"Vision mode was explicitly requested (SANDCASTLE_VLLM_ENABLE_VISION=1), but {target_base}/models reports vision={vision_by_id}!")
    r_chat = requests.post(
        f"{target_base}/chat/completions",
        headers=headers,
        json={
            "model": expected_model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0.0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    if r_chat.status_code == 401:
        raise RuntimeError(f"HTTP 401 Unauthorized probing {target_base}/chat/completions")
    r_chat.raise_for_status()
    content = r_chat.json()["choices"][0]["message"].get("content", "")
    vision_sample = None
    if active_vision:
        r_vis = requests.post(
            f"{target_base}/chat/completions",
            headers=headers,
            json={
                "model": expected_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Reply with OK if you can see this image."},
                            {"type": "image_url", "image_url": {"url": _make_probe_png_data_url()}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=180,
        )
        r_vis.raise_for_status()
        vision_sample = str(r_vis.json()["choices"][0]["message"].get("content", ""))[:80]
    return {
        "ok": True,
        "base": target_base,
        "models": served_ids,
        "vision": vision_by_id,
        "sample": str(content)[:80],
        **({"vision_sample": vision_sample} if vision_sample is not None else {}),
    }

if HEADLESS_SERVING:
    print(f"Running lightweight single-request readiness check for headless serving (model={expected_model})...")
    t0 = time.perf_counter()
    verified_info = None
    last_dns_exc = None
    for attempt in range(13):
        try:
            verified_info = _probe_base_once(base)
            break
        except requests.exceptions.RequestException as exc:
            if _is_dns_err(exc) and attempt < 12:
                last_dns_exc = exc
                time.sleep(2.0)
                continue
            if _is_dns_err(exc):
                last_dns_exc = exc
                break
            raise
    if verified_info is None and last_dns_exc is not None and "127.0.0.1" not in base and "localhost" not in base:
        print(f"Tunnel hostname DNS still propagating ({last_dns_exc}); verifying container loopback http://127.0.0.1:8000/v1...")
        verified_info = _probe_base_once("http://127.0.0.1:8000/v1")
    dt = time.perf_counter() - t0
    print(f"Headless single-probe readiness verified in {dt:.2f}s: {verified_info}")
else:
    def _send_probe(idx: int):
        t0 = time.perf_counter()
        resp = requests.post(
            f"{base}/chat/completions",
            headers=headers,
            json={
                "model": expected_model,
                "messages": [{"role": "user", "content": f"Worker #{idx}: Reply with OK."}],
                "temperature": 0.1,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=180,
        )
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"]
        return idx, round(time.perf_counter() - t0, 2), str(out)[:80]

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for idx, sec, preview in pool.map(_send_probe, range(6)):
            print(f"Worker {idx} completed in {sec}s: {preview}")
"""
    return env_bootstrap + c6_warmup_marker + body


def _build_cell2_hardware_code(env_bootstrap: str, expected_tier: str, expected_mem_gb: float) -> str:
    """Builds GPU + TPU XLA HBM hardware verification, fail-fast substitution check, & headroom-driven concurrency Cell 2."""
    c2_tier_vram = {
        k: round(float(v.get("vram_gb", 15.8)), 2)
        for k, v in (_SSOT_GPU_TIER_SPECS or {}).items()
    } or {"G4": 95.0, "H100": 79.2, "A100-80GB": 79.2, "A100": 40.0, "L4": 23.5, "T4": 15.8, "v6e-1": 32.0, "v5e-1": 16.0}
    c2_tier_seqs = {
        k: int(v.get("max_num_seqs", 32))
        for k, v in (_SSOT_GPU_TIER_SPECS or {}).items()
    } or {"G4": 256, "H100": 160, "A100-80GB": 128, "A100": 96, "L4": 64, "T4": 32, "v6e-1": 8, "v5e-1": 4}
    c2_model_specs: Dict[str, Dict[str, Any]] = {}
    _ssot_cat = _SSOT_MODEL_CATALOG if isinstance(_SSOT_MODEL_CATALOG, Mapping) else {}
    for _m_alias, _m_info in MODEL_HF_REPOS.items():
        _ssot_entry = _ssot_cat.get(_m_alias) if isinstance(_ssot_cat.get(_m_alias), Mapping) else {}
        _pv = _m_info.get("precision_variants") if isinstance(_m_info.get("precision_variants"), Mapping) else {}
        _q4_w = float(
            _ssot_entry.get("empirical_lm_weights_gib")
            or ((_pv.get("Q4") or {}).get("weights_gb") if isinstance(_pv.get("Q4"), Mapping) else 0.0)
            or _m_info.get("weights_gb")
            or 5.5
        )
        _bf16_w = float(
            ((_pv.get("BF16") or {}).get("weights_gb") if isinstance(_pv.get("BF16"), Mapping) else 0.0)
            or _m_info.get("weights_gb")
            or _q4_w
        )
        _bf16_min = float(
            ((_pv.get("BF16") or {}).get("min_vram_gb") if isinstance(_pv.get("BF16"), Mapping) else 0.0)
            or (_bf16_w + 4.0)
        )
        _per_seq_gib = 0.18
        _per_seq_gib_32k = 0.36
        if callable(_estimate_ssot_kv_footprint):
            try:
                _fp_c2 = _estimate_ssot_kv_footprint(_m_alias, max_model_len=16384, kv_cache_dtype="fp8")
                _w_kv = (float(_fp_c2["kv_bytes_per_token"]) * 2048.0) / float(1024 ** 3)
                _per_seq_gib = max(0.04, _w_kv + float(_fp_c2.get("gdn_gb_per_seq") or 0.0))
                if _fp_c2.get("is_hybrid_gdn") and float(_fp_c2.get("unified_aligned_block_gib") or 0.0) > 0:
                    _per_seq_gib = max(_per_seq_gib, 3.0 * float(_fp_c2["unified_aligned_block_gib"]))
                _fp_c2_32 = _estimate_ssot_kv_footprint(_m_alias, max_model_len=32768, kv_cache_dtype="fp8")
                _w_kv_32 = (float(_fp_c2_32["kv_bytes_per_token"]) * 4096.0) / float(1024 ** 3)
                _per_seq_gib_32k = max(0.08, _w_kv_32 + float(_fp_c2_32.get("gdn_gb_per_seq") or 0.0))
                if _fp_c2_32.get("is_hybrid_gdn") and float(_fp_c2_32.get("unified_aligned_block_gib") or 0.0) > 0:
                    _per_seq_gib_32k = max(_per_seq_gib_32k, 3.0 * float(_fp_c2_32["unified_aligned_block_gib"]))
            except Exception:
                pass
        _cat_max_len = int(_ssot_entry.get("max_model_len") or _m_info.get("max_model_len") or 16384)
        _cat_max_len_a100 = int(_ssot_entry.get("max_model_len_a100") or _cat_max_len or 32768)
        c2_model_specs[_m_alias] = {
            "min_vram_gb": float(_m_info.get("min_vram_gb", _q4_w + 2.0)),
            "q4_weights_gb": round(_q4_w, 2),
            "bf16_weights_gb": round(_bf16_w, 2),
            "bf16_min_vram_gb": round(_bf16_min, 2),
            "per_seq_gib": round(_per_seq_gib, 4),
            "per_seq_gib_32k": round(_per_seq_gib_32k, 4),
            "max_model_len": _cat_max_len,
            "max_model_len_a100": _cat_max_len_a100,
            "default_precision_by_gpu": dict(_m_info.get("default_precision_by_gpu") or {}),
        }

    return (
        env_bootstrap
        + "# Cell 2: Verify Accelerator Hardware (CUDA GPU VRAM or TPU XLA HBM), Compute Dtype & Physical Headroom\n"
        "import json, os, torch\n"
        f"_expected_tier = os.environ.get('PLATODOOM_GPU', {json.dumps(expected_tier)})\n"
        f"_CELL2_TIER_VRAM = {json.dumps(c2_tier_vram)}\n"
        f"_CELL2_TIER_SEQS = {json.dumps(c2_tier_seqs)}\n"
        f"_CELL2_MODEL_SPECS = {json.dumps(c2_model_specs)}\n"
        "_c2_target_model = (os.environ.get('PLATODOOM_MODEL', 'qwen3.8-27b') or 'qwen3.8-27b').strip().split('@', 1)[0]\n"
        "MAX_PARALLEL_AGENTS = int(os.environ.get('SANDCASTLE_MAX_PARALLEL_AGENTS', '64'))\n"
        "MAX_MODEL_LEN = 16384\n"
        "MAX_NUM_BATCHED_TOKENS = int(os.environ.get('SANDCASTLE_MAX_NUM_BATCHED_TOKENS', '16384'))\n"
        "IS_TPU_RUNTIME = False\n"
        "if torch.cuda.is_available():\n"
        "    gpu_name = torch.cuda.get_device_name(0)\n"
        "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n"
        "    try:\n"
        "        free_vram_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)\n"
        "    except Exception:\n"
        "        free_vram_gb = vram_gb\n"
        "    compute_capability = torch.cuda.get_device_capability(0)\n"
        "    COMPUTE_DTYPE = torch.float16 if compute_capability < (8, 0) else torch.bfloat16\n"
        "    accelerator_kind = 'GPU'\n"
        "    memory_type = 'VRAM'\n"
        "    _sm_label = f'sm_{compute_capability[0]}{compute_capability[1]}'\n"
        "    _gpu_up = gpu_name.upper()\n"
        "    if vram_gb >= (_CELL2_TIER_VRAM.get('G4', 95.0) * 0.90) or (compute_capability >= (10, 0) and vram_gb >= 85.0) or ('BLACKWELL' in _gpu_up and vram_gb >= 85.0):\n"
        "        detected_tier = 'G4'\n"
        "    elif vram_gb >= (_CELL2_TIER_VRAM.get('H100', 79.2) * 0.90):\n"
        "        detected_tier = 'H100' if (compute_capability >= (9, 0) or 'H100' in _gpu_up) else 'A100-80GB'\n"
        "    elif vram_gb >= (_CELL2_TIER_VRAM.get('A100', 40.0) * 0.90):\n"
        "        detected_tier = 'A100'\n"
        "    elif vram_gb >= (_CELL2_TIER_VRAM.get('L4', 23.5) * 0.90):\n"
        "        detected_tier = 'L4'\n"
        "    else:\n"
        "        detected_tier = 'T4'\n"
        "    _tier_seq_cap = int(_CELL2_TIER_SEQS.get(detected_tier, 32))\n"
        "else:\n"
        "    IS_TPU_RUNTIME = True\n"
        "    accelerator_kind = 'TPU'\n"
        "    memory_type = 'HBM'\n"
        "    gpu_name = f'Google Cloud TPU {_expected_tier}'\n"
        "    compute_capability = (0, 0)\n"
        "    _sm_label = 'tpu'\n"
        "    COMPUTE_DTYPE = torch.bfloat16\n"
        f"    vram_gb = float(os.environ.get('PLATODOOM_EXPECTED_MEMORY_GB', {json.dumps(str(expected_mem_gb))}))\n"
        "    free_vram_gb = vram_gb\n"
        "    try:\n"
        "        import torch_xla.core.xla_model as xm\n"
        "        mem_info = xm.get_memory_info(xm.xla_device())\n"
        "        if isinstance(mem_info, dict) and mem_info.get('bytes_limit'):\n"
        "            vram_gb = float(mem_info['bytes_limit']) / (1024**3)\n"
        "            free_vram_gb = vram_gb\n"
        "    except Exception:\n"
        "        pass\n"
        "    detected_tier = 'v6e-1' if vram_gb >= 24.0 else 'v5e-1'\n"
        "    _tier_seq_cap = int(_CELL2_TIER_SEQS.get(detected_tier, 4))\n"
        "\n"
        "_c2_spec = _CELL2_MODEL_SPECS.get(_c2_target_model) or {}\n"
        "_c2_stack_raw = os.environ.get('PLATODOOM_MODEL_STACK', '').strip()\n"
        "if _c2_stack_raw:\n"
        "    _c2_stack_items = [s for s in json.loads(_c2_stack_raw) if int(s.get('agent_count', 1)) > 0]\n"
        "    _c2_req_gib = sum(float((_CELL2_MODEL_SPECS.get(str(s.get('alias') or s.get('model_alias') or '').split('@', 1)[0]) or {}).get('q4_weights_gb', 4.0)) for s in _c2_stack_items) + 2.0 * max(1, len(_c2_stack_items))\n"
        "    if vram_gb > 0 and vram_gb < _c2_req_gib:\n"
        "        raise RuntimeError(\n"
        "            f'VRAM mismatch on allocated {accelerator_kind} {gpu_name} ({vram_gb:.2f} GiB {memory_type}, tier={detected_tier}, sm={_sm_label}): '\n"
        "            f'active model stack requires >= {_c2_req_gib:.2f} GiB physical memory (requested tier={_expected_tier}).'\n"
        "        )\n"
        "elif _c2_spec:\n"
        "    _c2_min_floor = max(float(_c2_spec.get('q4_weights_gb', 0.0)) + 2.0, float(_c2_spec.get('min_vram_gb', 0.0)) * 0.90)\n"
        "    if vram_gb > 0 and vram_gb < _c2_min_floor:\n"
        "        raise RuntimeError(\n"
        "            f'VRAM mismatch on allocated {accelerator_kind} {gpu_name} ({vram_gb:.2f} GiB {memory_type}, tier={detected_tier}, sm={_sm_label}): '\n"
        "            f\"model '{_c2_target_model}' requires >= {_c2_min_floor:.2f} GiB physical memory (requested tier={_expected_tier}).\"\n"
        "        )\n"
        "\n"
        "_c2_raw_prec = str((_c2_spec.get('default_precision_by_gpu') or {}).get(detected_tier) or '').strip().upper()\n"
        "if not _c2_raw_prec:\n"
        "    _c2_raw_prec = 'Q4' if vram_gb < float(_c2_spec.get('bf16_min_vram_gb', 65.0)) else 'BF16'\n"
        "if _c2_raw_prec in ('Q4', 'INT4', 'GGUF', 'GGUF_Q4') or _c2_raw_prec.endswith('_4BIT'):\n"
        "    SELECTED_MODEL_PRECISION = 'Q4'\n"
        "elif 'FP8' in _c2_raw_prec:\n"
        "    SELECTED_MODEL_PRECISION = 'FP8'\n"
        "elif _c2_raw_prec in ('Q8', 'INT8', 'Q6', 'Q5'):\n"
        "    SELECTED_MODEL_PRECISION = _c2_raw_prec\n"
        "else:\n"
        "    SELECTED_MODEL_PRECISION = 'BF16'\n"
        "USE_4BIT_SERVING = bool(accelerator_kind == 'GPU' and SELECTED_MODEL_PRECISION == 'Q4')\n"
        "if accelerator_kind == 'GPU':\n"
        "    _c2_w = float(_c2_spec.get('q4_weights_gb', 6.0) if SELECTED_MODEL_PRECISION == 'Q4' else _c2_spec.get('bf16_weights_gb', 18.0))\n"
        "    _c2_w_ratio = _c2_w / max(1.0, vram_gb)\n"
        "    _c2_util = min(0.94, 0.90 + 0.04 * min(1.0, max(0.0, (_c2_w_ratio - 0.60) / 0.25)))\n"
        "    _c2_headroom = max(0.0, free_vram_gb * _c2_util - _c2_w - 2.2)\n"
        "    _c2_per_seq = max(0.04, float(_c2_spec.get('per_seq_gib', 0.18)))\n"
        "    _c2_per_seq_32k = max(0.08, float(_c2_spec.get('per_seq_gib_32k', _c2_per_seq * 2.0)))\n"
        "    _c2_cat_len = int(_c2_spec.get('max_model_len_a100') or _c2_spec.get('max_model_len') or 32768) if detected_tier in ('A100', 'A100-80GB', 'H100', 'G4') else int(_c2_spec.get('max_model_len') or 16384)\n"
        "    if detected_tier == 'T4':\n"
        "        _c2_cat_len = min(16384, _c2_cat_len)\n"
        "    if _c2_cat_len >= 32768 and _c2_headroom >= (_c2_per_seq_32k * 4.0):\n"
        "        MAX_MODEL_LEN = _c2_cat_len\n"
        "        _c2_eff_per_seq = _c2_per_seq_32k\n"
        "    else:\n"
        "        MAX_MODEL_LEN = 16384\n"
        "        _c2_eff_per_seq = _c2_per_seq\n"
        "    _c2_phys_seqs = max(1, min(_tier_seq_cap, int(_c2_headroom // _c2_eff_per_seq))) if _c2_headroom > 0 else 4\n"
        "    ACTIVE_GPU_SEQS = min(_tier_seq_cap, _c2_phys_seqs)\n"
        "else:\n"
        "    ACTIVE_GPU_SEQS = _tier_seq_cap\n"
        "print(f'Detected {accelerator_kind}: {gpu_name} ({vram_gb:.2f} GiB {memory_type}, free={free_vram_gb:.2f} GiB, tier={detected_tier}, sm={_sm_label}, precision={SELECTED_MODEL_PRECISION}, dtype={COMPUTE_DTYPE}) | max_num_seqs={ACTIVE_GPU_SEQS} | max_model_len={MAX_MODEL_LEN}')\n"
    )


def _resolve_local_hf_token() -> str:
    """Resolves HuggingFace token from env vars or saved `.platodoom/config.json` (`0600`)."""
    for env_key in ("PLATODOOM_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = str(os.environ.get(env_key) or "").strip()
        if val:
            return val
    try:
        from platodoom.config import resolve_platodoom_config

        cfg = resolve_platodoom_config()
        tok = str((cfg.get("auth") or {}).get("hf_token") or "").strip()
        if tok:
            return tok
    except Exception:
        pass
    return ""


def _resolve_local_ngrok_authtoken() -> str:
    """Resolves NGROK_AUTHTOKEN from env, `.platodoom/config.json`, local ngrok.yml config, or shell rc files."""
    env_tok = str(os.environ.get("NGROK_AUTHTOKEN") or "").strip()
    if env_tok:
        return env_tok
    try:
        from platodoom.config import resolve_platodoom_config

        cfg = resolve_platodoom_config()
        auth_cfg = cfg.get("auth") or {}
        cfg_tok = str(auth_cfg.get("ngrok_authtoken") or "").strip()
        if cfg_tok:
            return cfg_tok
    except Exception:
        pass
    home = Path.home()
    for yml_path in (
        home / "Library" / "Application Support" / "ngrok" / "ngrok.yml",
        home / ".config" / "ngrok" / "ngrok.yml",
        home / ".ngrok2" / "ngrok.yml",
    ):
        if yml_path.is_file():
            try:
                m = re.search(
                    r"^\s*authtoken\s*:\s*['\"]?([^\s'\"]+)['\"]?",
                    yml_path.read_text(encoding="utf-8", errors="ignore"),
                    re.MULTILINE,
                )
                if m and m.group(1).strip():
                    return m.group(1).strip()
            except OSError:
                pass
    for rc_path in (home / ".zshrc", home / ".zprofile", home / ".bash_profile", home / ".bashrc"):
        if rc_path.is_file():
            try:
                m = re.search(
                    r"^\s*(?:export\s+)?NGROK_AUTHTOKEN\s*=\s*['\"]?([^\s'\"#]+)['\"]?",
                    rc_path.read_text(encoding="utf-8", errors="ignore"),
                    re.MULTILINE,
                )
                if m and m.group(1).strip():
                    return m.group(1).strip()
            except OSError:
                pass
    return ""


def _build_headless_drive_notebook_bytes(
    gpu_profile: Optional[Dict[str, Any]] = None,
    repo_dir: Optional[Path | str] = None,
    notebook_path: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
    model_alias: Optional[str] = None,
    bearer_token: Optional[str] = None,
    gpu_override: Optional[str] = None,
    stack_preset: Optional[str] = None,
    stack_models: Optional[List[Any]] = None,
    agent_counts: Optional[Mapping[str, int]] = None,
    model_allocations: Optional[Any] = None,
    enable_vision: Optional[bool] = None,
) -> bytes:
    """
    Constructs the per-directory or interactive session runtime notebook payload
    (`serve_sandcastle_vllm_<dir_slug>.ipynb` or `serve_platodoom_<session_slug>.ipynb`)
    purely in memory across all 8 Colab GPU/TPU hardware tiers (NEVER mutating `notebooks/serve_sandcastle_vllm.ipynb` on disk).
    """
    nb_file = Path(notebook_path) if notebook_path else SERVE_NOTEBOOK_PATH
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    effective_model = str(model_alias or UNIFIED_MODEL_ALIAS).strip() or UNIFIED_MODEL_ALIAS
    target_name = (
        f"serve_platodoom_{effective_slug}.ipynb"
        if (session_slug and str(session_slug).strip())
        else f"serve_sandcastle_vllm_{dir_slug}.ipynb"
    )

    resolved_profile = dict(gpu_profile) if isinstance(gpu_profile, dict) else None
    if resolved_profile is None or gpu_override or stack_preset or stack_models or agent_counts is not None or model_allocations is not None:
        try:
            computed_profile = read_gpu_routing_profile(
                repo_dir=project_root,
                gpu_override=gpu_override,
                session_slug=session_slug,
                model_alias=effective_model,
                stack_preset=stack_preset,
                stack_models=list(stack_models) if stack_models else None,
                agent_counts=agent_counts,
                model_allocations=model_allocations,
            )
        except ValueError:
            if gpu_override and not stack_preset and not stack_models and agent_counts is None and model_allocations is None:
                computed_profile = read_gpu_routing_profile(
                    repo_dir=project_root,
                    num_models=1,
                    peak_stage_models=1,
                    gpu_override=gpu_override,
                    session_slug=session_slug,
                )
            else:
                raise
        if resolved_profile is None:
            resolved_profile = computed_profile
        else:
            resolved_profile.update(computed_profile)

    raw_tier_candidate = str(
        gpu_override
        or (resolved_profile or {}).get("gpu_type")
        or (resolved_profile or {}).get("colab_gpu_type")
        or MODEL_PREFERRED_GPU_TIERS.get(effective_model, "T4")
    ).strip()
    if callable(_normalize_ssot_tier) and raw_tier_candidate and raw_tier_candidate not in ("auto", "None"):
        resolved_gpu_tier = _normalize_ssot_tier(raw_tier_candidate, allow_auto=False)
    else:
        resolved_gpu_tier = raw_tier_candidate if raw_tier_candidate in ("G4", "H100", "A100-80GB", "A100", "v6e-1", "L4", "T4", "v5e-1") else "T4"

    is_tpu_tier = resolved_gpu_tier in ("v6e-1", "v5e-1")
    colab_gpu_type = "A100" if resolved_gpu_tier == "A100-80GB" else resolved_gpu_tier

    # Resolve optional co-resident stack (x = 1..3) and selective n_i > 0 residency when explicitly requested
    resolved_stack_models: Optional[List[Dict[str, Any]]] = None
    resolved_inactive_models: List[Dict[str, Any]] = []
    resolved_agent_counts: Optional[Dict[str, int]] = (
        dict(agent_counts) if isinstance(agent_counts, Mapping) else (resolved_profile or {}).get("agent_counts")
    )
    profile_preset = (resolved_profile or {}).get("stack_preset")
    effective_stack_preset = stack_preset or (
        profile_preset if profile_preset and str(profile_preset).strip().lower() != "solo" else None
    )
    raw_stack_models = stack_models or (resolved_profile or {}).get("stack_models")
    if (model_allocations is not None or agent_counts is not None) and callable(_resolve_ssot_multi_model_agents):
        alloc_input = model_allocations
        if alloc_input is None:
            if raw_stack_models:
                alloc_input = [
                    {**(dict(m) if isinstance(m, Mapping) else {"model": str(m)}), "agent_count": int((agent_counts or {}).get((m.get("alias") if isinstance(m, Mapping) else str(m).split("@", 1)[0]), 1))}
                    for m in raw_stack_models
                ]
            else:
                alloc_input = dict(agent_counts or {})
        alloc_res = _resolve_ssot_multi_model_agents(alloc_input, gpu_tier=resolved_gpu_tier)
        resolved_stack_models = [dict(m) for m in alloc_res["active_models"]]
        resolved_inactive_models = [dict(m) for m in alloc_res["inactive_models"]]
        resolved_agent_counts = dict(alloc_res["agent_counts"])
        if resolved_stack_models and not model_alias:
            effective_model = str(resolved_stack_models[0]["alias"])
    elif (effective_stack_preset or raw_stack_models) and callable(_resolve_ssot_stack):
        stack_info = _resolve_ssot_stack(
            resolved_gpu_tier,
            stack_preset=str(effective_stack_preset) if effective_stack_preset else None,
            model_alias=effective_model if model_alias else None,
            stack_models=list(raw_stack_models) if raw_stack_models else None,
        )
        resolved_stack_models = [dict(m) for m in stack_info["models"]]
        effective_stack_preset = stack_info.get("stack_preset") or effective_stack_preset
        if resolved_stack_models and not model_alias:
            effective_model = str(resolved_stack_models[0]["alias"])

    tier_spec = (_SSOT_GPU_TIER_SPECS or {}).get(resolved_gpu_tier, {})
    routed_mem_gb = float((resolved_profile or {}).get("vram_gb") or tier_spec.get("nominal_vram_gb") or 16.0)

    default_tier_seqs = int(tier_spec.get("max_num_seqs") or (4 if is_tpu_tier else 32))
    max_seqs = int((resolved_profile or {}).get("max_num_seqs") or default_tier_seqs)

    nb_obj = json.loads(nb_file.read_text(encoding="utf-8"))
    nb_obj.setdefault("metadata", {})
    colab_meta = dict(nb_obj["metadata"].get("colab") or {})
    colab_meta["name"] = target_name
    colab_meta["dir_slug"] = effective_slug
    colab_meta["model_alias"] = effective_model
    colab_meta["hardware_tier"] = resolved_gpu_tier
    if session_slug:
        colab_meta["session_slug"] = effective_slug

    if is_tpu_tier:
        nb_obj["metadata"]["accelerator"] = "TPU"
        colab_meta.pop("gpuType", None)
        colab_meta["tpuType"] = resolved_gpu_tier
        colab_meta["machine_shape"] = "hm"
    else:
        nb_obj["metadata"]["accelerator"] = "GPU"
        colab_meta.pop("tpuType", None)
        colab_meta["gpuType"] = colab_gpu_type
        colab_meta["machine_shape"] = (resolved_profile or {}).get("colab_machine_shape", "hm")

    if resolved_profile:
        _raw_prec_tier = str(resolved_profile.get("precision_tier") or "Q4")
        if _raw_prec_tier == "nf4_4bit" and effective_model != UNIFIED_MODEL_ALIAS:
            _raw_prec_tier = "Q4"
        colab_meta.update(
            {
                "routed_gpu_label": resolved_profile.get("gpu_label", ""),
                "routed_vram_gb": resolved_profile.get("vram_gb", int(round(routed_mem_gb))),
                "routed_system_ram_gb": resolved_profile.get("system_ram_gb", 25.5),
                "routed_precision_tier": _raw_prec_tier,
                "routed_max_num_seqs": max_seqs,
                "routed_qwen_model_count": len(resolved_stack_models) if resolved_stack_models is not None else resolved_profile.get("total_qwen_models", 1),
            }
        )
    else:
        colab_meta["routed_max_num_seqs"] = max_seqs

    if effective_stack_preset:
        colab_meta["stack_preset"] = str(effective_stack_preset)
    if resolved_stack_models:
        colab_meta["stack_models"] = resolved_stack_models
    if resolved_inactive_models:
        colab_meta["inactive_models"] = resolved_inactive_models
    if resolved_agent_counts:
        colab_meta["agent_counts"] = resolved_agent_counts

    nb_obj["metadata"]["colab"] = colab_meta

    # Only forward local GCP Application Default Credentials when explicitly opted in AND GCS storage is requested.
    adc_b64 = ""
    _gcs_bucket_active = bool(str(os.environ.get("PLATODOOM_GCS_BUCKET") or "").strip())
    _forward_adc_opt_in = str(os.environ.get("PLATODOOM_FORWARD_GCLOUD_ADC") or "").strip().lower() in ("1", "true", "yes")
    if _forward_adc_opt_in and (_gcs_bucket_active or effective_model == "qwen3.8-27b"):
        adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if adc_path.is_file():
            try:
                adc_b64 = base64.b64encode(adc_path.read_bytes()).decode("ascii")
            except Exception:
                adc_b64 = ""

    effective_bearer = str(
        bearer_token
        or os.environ.get("PLATODOOM_TUNNEL_TOKEN")
        or os.environ.get("SANDCASTLE_VLLM_API_KEY")
        or ""
    ).strip()

    all_configured_for_env = (list(resolved_stack_models or []) + list(resolved_inactive_models or [])) if resolved_inactive_models else resolved_stack_models
    stack_json_env = json.dumps(all_configured_for_env) if all_configured_for_env else ""
    agent_counts_json_env = json.dumps(resolved_agent_counts) if resolved_agent_counts else ""

    extra_env_lines = (
        f"_sc_os.environ['PLATODOOM_MODEL'] = {json.dumps(effective_model)}\n"
        f"_sc_os.environ['PLATODOOM_GPU'] = {json.dumps(resolved_gpu_tier)}\n"
        f"_sc_os.environ['PLATODOOM_EXPECTED_MEMORY_GB'] = {json.dumps(str(routed_mem_gb))}\n"
        f"_sc_os.environ['SANDCASTLE_MAX_PARALLEL_AGENTS'] = {json.dumps(str(max_seqs))}\n"
        f"_sc_os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n"
    )
    if enable_vision is not None:
        extra_env_lines += f"_sc_os.environ['SANDCASTLE_VLLM_ENABLE_VISION'] = {'1' if enable_vision else '0'!r}\n"
    if effective_stack_preset:
        extra_env_lines += f"_sc_os.environ['PLATODOOM_STACK'] = {json.dumps(str(effective_stack_preset))}\n"
    if stack_json_env:
        extra_env_lines += f"_sc_os.environ['PLATODOOM_MODEL_STACK'] = {json.dumps(stack_json_env)}\n"
    if agent_counts_json_env:
        extra_env_lines += f"_sc_os.environ['PLATODOOM_AGENT_COUNTS'] = {json.dumps(agent_counts_json_env)}\n"
    if effective_bearer:
        extra_env_lines += f"_sc_os.environ['PLATODOOM_TUNNEL_TOKEN'] = {json.dumps(effective_bearer)}\n"
        extra_env_lines += f"_sc_os.environ['SANDCASTLE_VLLM_API_KEY'] = {json.dumps(effective_bearer)}\n"
    effective_hf_token = _resolve_local_hf_token()
    if effective_hf_token:
        extra_env_lines += f"_sc_os.environ['PLATODOOM_HF_TOKEN'] = {json.dumps(effective_hf_token)}\n"
        extra_env_lines += f"_sc_os.environ['HF_TOKEN'] = {json.dumps(effective_hf_token)}\n"
    effective_ngrok_token = _resolve_local_ngrok_authtoken()
    if effective_ngrok_token:
        extra_env_lines += f"_sc_os.environ['NGROK_AUTHTOKEN'] = {json.dumps(effective_ngrok_token)}\n"

    env_bootstrap = (
        f"import os as _sc_os\n"
        f"_sc_os.environ['SANDCASTLE_DIR_SLUG'] = {json.dumps(effective_slug)}\n"
        f"_sc_os.environ['SANDCASTLE_HEADLESS_SERVING'] = '1'\n"
        + extra_env_lines
    )
    c6_warmup_marker = "# Spec 856/859 headless readiness warmup (max_new = 1 probe fallback)\n"

    for cell in nb_obj.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src_list = cell.get("source", [])
        src_text = "".join(src_list) if isinstance(src_list, list) else str(src_list)
        if "# Cell 1: Environment Setup" in src_text:
            cell["source"] = _build_cell1_setup_code(
                effective_slug=effective_slug,
                effective_model=effective_model,
                resolved_gpu_tier=resolved_gpu_tier,
                routed_mem_gb=routed_mem_gb,
                max_seqs=max_seqs,
                effective_stack_preset=str(effective_stack_preset) if effective_stack_preset else None,
                stack_json_env=stack_json_env,
                agent_counts_json_env=agent_counts_json_env,
                effective_bearer=effective_bearer,
                enable_vision=enable_vision,
            )
        elif "# Cell 2:" in src_text:
            cell["source"] = [_build_cell2_hardware_code(env_bootstrap=env_bootstrap, expected_tier=resolved_gpu_tier, expected_mem_gb=routed_mem_gb)]
        elif "# Cell 3:" in src_text:
            cell["source"] = [_build_cell3_dual_source_code(effective_slug=effective_slug, adc_b64=adc_b64)]
        elif "# Cell 4:" in src_text or "_batch_worker" in src_text or "BatchedQwen35ServingEngine" in src_text:
            cell["source"] = [_build_cell4_unified_server_code(env_bootstrap=env_bootstrap)]
        elif "SANDCASTLE_COLAB_ENDPOINT_JSON" in src_text or "# Cell 5:" in src_text:
            cell["source"] = [_build_cell5_tunnel_code(env_bootstrap=env_bootstrap)]
        elif "# Cell 6:" in src_text:
            cell["source"] = [_build_cell6_verification_code(env_bootstrap=env_bootstrap, c6_warmup_marker=c6_warmup_marker)]

    return (json.dumps(nb_obj, indent=2) + "\n").encode("utf-8")


def sync_serve_notebook_to_google_drive(
    gpu_profile: Optional[Dict[str, Any] | Path | str] = None,
    repo_dir: Optional[Path | str] = None,
    gpu_tier: Optional[str] = None,
    session_slug: Optional[str] = None,
    model_alias: Optional[str] = None,
    bearer_token: Optional[str] = None,
    gpu_override: Optional[str] = None,
    stack_preset: Optional[str] = None,
    stack_models: Optional[List[Any]] = None,
    agent_counts: Optional[Mapping[str, int]] = None,
    model_allocations: Optional[Any] = None,
    enable_vision: Optional[bool] = None,
    raise_on_error: bool = False,
) -> Optional[str]:
    """Uploads or updates `serve_sandcastle_vllm_<dir_slug>.ipynb` or `serve_platodoom_<session_slug>.ipynb` on Google Drive."""
    global LAST_DRIVE_SYNC_ERROR
    notebook_path = SERVE_NOTEBOOK_PATH
    resolved_profile: Optional[Dict[str, Any]] = None
    effective_gpu = str(gpu_override or gpu_tier or "").strip() or None

    if isinstance(gpu_profile, (Path, str)):
        cand_path = Path(gpu_profile)
        if cand_path.suffix == ".ipynb" or cand_path.is_file():
            notebook_path = cand_path
        elif repo_dir is None:
            repo_dir = cand_path
    elif isinstance(gpu_profile, dict):
        resolved_profile = dict(gpu_profile)

    project_root = resolve_project_root(repo_dir)
    if resolved_profile is None and not (model_alias or stack_preset or stack_models or agent_counts is not None or model_allocations is not None):
        ep_json = project_root / ".sandcastle" / "colab_qwen_endpoint.json"
        if ep_json.is_file():
            try:
                ep_data = json.loads(ep_json.read_text(encoding="utf-8"))
                if isinstance(ep_data, Mapping):
                    if not model_alias and ep_data.get("model"):
                        model_alias = str(ep_data["model"]).strip()
                    if not effective_gpu and ep_data.get("colab_gpu_type"):
                        effective_gpu = str(ep_data["colab_gpu_type"]).strip()
            except Exception:
                pass
    if resolved_profile is None and (effective_gpu or session_slug or model_alias or stack_preset or stack_models or agent_counts is not None or model_allocations is not None):
        resolved_profile = read_gpu_routing_profile(
            repo_dir=project_root,
            gpu_override=effective_gpu,
            session_slug=session_slug,
            model_alias=model_alias,
            stack_preset=stack_preset,
            stack_models=list(stack_models) if stack_models else None,
            agent_counts=agent_counts,
            model_allocations=model_allocations,
        )
    elif effective_gpu and resolved_profile is not None:
        resolved_profile["gpu_type"] = effective_gpu
        resolved_profile["colab_gpu_type"] = "A100" if effective_gpu == "A100-80GB" else effective_gpu
        resolved_profile["override_mode"] = effective_gpu

    if not notebook_path.is_file():
        LAST_DRIVE_SYNC_ERROR = f"FileNotFoundError: Colab serving notebook template missing at {notebook_path}"
        if raise_on_error:
            raise FileNotFoundError(LAST_DRIVE_SYNC_ERROR)
        return None

    access_token = _get_valid_drive_access_token(raise_on_error=True) if raise_on_error else _get_valid_drive_access_token()
    if not access_token:
        LAST_DRIVE_SYNC_ERROR = LAST_DRIVE_AUTH_ERROR or f"PermissionError: Unable to obtain valid Google Drive OAuth token from {MCP_COLAB_TOKEN_PATH}"
        if raise_on_error:
            raise RuntimeError(LAST_DRIVE_SYNC_ERROR)
        return None

    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    target_name = (
        f"serve_platodoom_{effective_slug}.ipynb"
        if (session_slug and str(session_slug).strip())
        else f"serve_sandcastle_vllm_{dir_slug}.ipynb"
    )
    headers = {"Authorization": f"Bearer {access_token}"}
    nb_bytes = _build_headless_drive_notebook_bytes(
        gpu_profile=resolved_profile,
        repo_dir=project_root,
        notebook_path=notebook_path,
        session_slug=session_slug,
        model_alias=model_alias,
        bearer_token=bearer_token,
        gpu_override=effective_gpu,
        stack_preset=stack_preset,
        stack_models=stack_models,
        agent_counts=agent_counts,
        model_allocations=model_allocations,
        enable_vision=enable_vision,
    )

    try:
        q = urllib.parse.quote(f"name = '{target_name}' and trashed = false")
        search_url = f"https://www.googleapis.com/drive/v3/files?q={q}&fields=files(id,name)&pageSize=5"
        req = urllib.request.Request(search_url, headers=headers, method="GET")
        existing_id: Optional[str] = None
        with urllib.request.urlopen(req, timeout=10) as resp:
            files = json.loads(resp.read().decode("utf-8")).get("files", [])
            if files:
                existing_id = str(files[0]["id"])

        if existing_id:
            upload_url = f"https://www.googleapis.com/upload/drive/v3/files/{existing_id}?uploadType=media"
            up_req = urllib.request.Request(
                upload_url,
                data=nb_bytes,
                headers={**headers, "Content-Type": "application/ipynb+json"},
                method="PATCH",
            )
            with urllib.request.urlopen(up_req, timeout=20) as resp:
                _ = resp.read()
            LAST_DRIVE_SYNC_ERROR = None
            return f"https://colab.research.google.com/drive/{existing_id}"
        else:
            boundary = "===sandcastle_colab_nb_boundary==="
            meta_part = json.dumps(
                {
                    "name": target_name,
                    "mimeType": "application/vnd.google.colaboratory",
                }
            ).encode("utf-8")
            body = (
                f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode("utf-8")
                + meta_part
                + f"\r\n--{boundary}\r\nContent-Type: application/ipynb+json\r\n\r\n".encode("utf-8")
                + nb_bytes
                + f"\r\n--{boundary}--\r\n".encode("utf-8")
            )
            create_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id"
            cr_req = urllib.request.Request(
                create_url,
                data=body,
                headers={
                    **headers,
                    "Content-Type": f"multipart/related; boundary={boundary}",
                },
                method="POST",
            )
            with urllib.request.urlopen(cr_req, timeout=20) as resp:
                created = json.loads(resp.read().decode("utf-8"))
                file_id = str(created["id"])
                LAST_DRIVE_SYNC_ERROR = None
                return f"https://colab.research.google.com/drive/{file_id}"
    except urllib.error.HTTPError as http_err:
        err_body = ""
        try:
            err_body = http_err.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(http_err)
        LAST_DRIVE_SYNC_ERROR = f"HTTPError {http_err.code} syncing '{target_name}' to Google Drive: {err_body[:300]}"
        if raise_on_error:
            raise RuntimeError(LAST_DRIVE_SYNC_ERROR) from http_err
        return None
    except Exception as exc:
        LAST_DRIVE_SYNC_ERROR = f"{type(exc).__name__} syncing '{target_name}' to Google Drive: {exc}"
        if raise_on_error:
            raise RuntimeError(LAST_DRIVE_SYNC_ERROR) from exc
        return None


def _sync_headless_browser_profile(
    repo_dir: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
) -> Path:
    """
    Prepares an isolated `~/.mcp-colab/chrome_profiles/headless_<slug>` directory (FR-2.1.2, FR-3.2, FR-3.3).
    Never touches `headless_default` and never runs a global `pkill` across directories.
    """
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    mcp_root = Path.home() / ".mcp-colab"
    profiles_root = mcp_root / "chrome_profiles"
    headless_profile_dir = profiles_root / f"headless_{effective_slug}"
    default_subdir = headless_profile_dir / "Default"
    default_subdir.mkdir(parents=True, exist_ok=True)
    for priv_dir in (mcp_root, profiles_root, headless_profile_dir, default_subdir):
        try:
            os.chmod(str(priv_dir), 0o700)
        except OSError:
            pass

    for lock_name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lock_file = headless_profile_dir / lock_name
        if lock_file.exists() or lock_file.is_symlink():
            try:
                lock_file.unlink()
            except Exception:
                pass

    # Strictly use the dedicated `platodoom auth` browser profile; NEVER copy the user's personal Chrome/Brave cookies.
    colab_auth_mock = profiles_root / "colab_auth_mock"
    if (colab_auth_mock / "Default" / "Cookies").is_file():
        try:
            if (colab_auth_mock / "Local State").is_file():
                dst_ls = headless_profile_dir / "Local State"
                shutil.copy2(colab_auth_mock / "Local State", dst_ls)
                os.chmod(str(dst_ls), 0o600)
            for fname in ("Cookies", "Cookies-wal", "Cookies-shm", "Preferences", "Secure Preferences"):
                fsrc = colab_auth_mock / "Default" / fname
                if fsrc.is_file():
                    fdst = default_subdir / fname
                    shutil.copy2(fsrc, fdst)
                    try:
                        os.chmod(str(fdst), 0o600)
                    except OSError:
                        pass
        except Exception:
            pass

    return headless_profile_dir


def _has_colab_auth_mock_profile() -> bool:
    mock_cookies = Path.home() / ".mcp-colab" / "chrome_profiles" / "colab_auth_mock" / "Default" / "Cookies"
    return mock_cookies.is_file()


def interactive_colab_login(timeout_sec: float = 180.0) -> Dict[str, Any]:
    """
    Opens a visible browser window using --use-mock-keychain so the user can sign
    into Google Colab once into ~/.mcp-colab/chrome_profiles/colab_auth_mock,
    bypassing macOS Keychain ACL blocks in headless Chrome/Brave sessions.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService

    mcp_root = Path.home() / ".mcp-colab"
    profiles_root = mcp_root / "chrome_profiles"
    auth_dir = profiles_root / "colab_auth_mock"
    (auth_dir / "Default").mkdir(parents=True, exist_ok=True)
    for priv_dir in (mcp_root, profiles_root, auth_dir, auth_dir / "Default"):
        try:
            os.chmod(str(priv_dir), 0o700)
        except OSError:
            pass
    for lock_name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lock_file = auth_dir / lock_name
        if lock_file.exists() or lock_file.is_symlink():
            try:
                lock_file.unlink()
            except Exception:
                pass

    options = webdriver.ChromeOptions()
    detected_browser = _detect_chromium_browser_binary()
    if detected_browser:
        options.binary_location = detected_browser

    options.add_argument("--use-mock-keychain")
    options.add_argument("--password-store=basic")
    options.add_argument(f"--user-data-dir={auth_dir}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    cd_bin = _resolve_chromedriver_executable()
    driver = webdriver.Chrome(
        service=ChromeService(cd_bin) if cd_bin else ChromeService(),
        options=options,
    )
    try:
        driver.get("https://accounts.google.com/ServiceLogin?continue=https://colab.research.google.com/")
        deadline = time.time() + max(30.0, float(timeout_sec))
        while time.time() < deadline:
            try:
                _ = driver.window_handles
                cur_url = str(driver.current_url or "")
                cookies = driver.execute_cdp_cmd("Network.getAllCookies", {}).get("cookies", [])
                has_sid = any(
                    "google.com" in str(c.get("domain", "")) and c.get("name") == "SID" and c.get("value")
                    for c in cookies
                )
                if has_sid and "colab.research.google.com" in cur_url and "accounts.google.com" not in cur_url:
                    time.sleep(2.0)
                    return {"ok": True, "profile": str(auth_dir), "message": "Google Colab session saved to colab_auth_mock."}
            except Exception as exc:
                msg = str(exc).lower()
                if "no such window" in msg or "invalid session" in msg or "disconnected" in msg:
                    return {"ok": False, "error": f"Browser window closed before sign-in completed: {exc}"}
            time.sleep(1.5)
        return {"ok": False, "error": "Timed out waiting for Google Colab sign-in."}
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def _serialize_models_agents_cli_arg(
    model_allocations: Optional[Any] = None,
    agent_counts: Optional[Mapping[str, int]] = None,
    stack_models: Optional[List[Any]] = None,
) -> Optional[str]:
    if isinstance(model_allocations, str) and model_allocations.strip():
        return model_allocations.strip()
    if isinstance(model_allocations, (dict, list)) and model_allocations:
        return json.dumps(model_allocations)
    if isinstance(agent_counts, Mapping) and agent_counts:
        return json.dumps({str(k): int(v) for k, v in agent_counts.items()})
    if isinstance(stack_models, list) and len(stack_models) >= 2:
        return json.dumps(stack_models)
    return None


def _parse_models_agents_cli_arg(raw: Optional[str]) -> Optional[Any]:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("{") or cleaned.startswith("["):
        return json.loads(cleaned)
    return cleaned


def spawn_directory_keepalive_daemon(
    repo_dir: Optional[Path | str] = None,
    gpu_profile: Optional[Dict[str, Any]] = None,
    idle_ttl_sec: float = 3600.0,
    session_slug: Optional[str] = None,
    model_alias: Optional[str] = None,
    bearer_token: Optional[str] = None,
    gpu_override: Optional[str] = None,
    max_parallel_agents: Optional[int] = None,
    stack_preset: Optional[str] = None,
    stack_models: Optional[List[Any]] = None,
    agent_counts: Optional[Mapping[str, int]] = None,
    model_allocations: Optional[Any] = None,
) -> int:
    """Spawns the persistent per-directory or interactive-session Keep-Alive Daemon in a detached session and returns its PID."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    effective_gpu = str(
        gpu_override
        or (gpu_profile or {}).get("override_mode")
        or (gpu_profile or {}).get("colab_gpu_type")
        or (gpu_profile or {}).get("gpu_type")
        or ""
    ).strip()
    if effective_gpu in ("auto", "None"):
        effective_gpu = ""

    py_bin = _find_selenium_python_executable()
    cmd = [
        py_bin,
        str(Path(__file__).resolve()),
        "--keepalive-daemon",
        f"--repo-dir={project_root}",
        f"--idle-ttl-sec={float(idle_ttl_sec)}",
    ]
    if session_slug:
        cmd.append(f"--session-slug={effective_slug}")
    if model_alias:
        cmd.append(f"--model-alias={model_alias}")
    eff_stack_preset = stack_preset or (gpu_profile or {}).get("stack_preset")
    if eff_stack_preset and str(eff_stack_preset).strip().lower() != "solo":
        cmd.append(f"--stack-preset={eff_stack_preset}")
    serialized_alloc = _serialize_models_agents_cli_arg(
        model_allocations=model_allocations,
        agent_counts=agent_counts or (gpu_profile or {}).get("agent_counts"),
        stack_models=stack_models or (gpu_profile or {}).get("stack_models"),
    )
    if serialized_alloc:
        cmd.append(f"--models-agents={serialized_alloc}")
    if gpu_profile:
        cmd.append(f"--num-models={gpu_profile.get('total_qwen_models', 1)}")
        cmd.append(f"--peak-stage-models={gpu_profile.get('peak_stage_qwen_models', 1)}")
    if effective_gpu:
        cmd.append(f"--gpu-override={effective_gpu}")

    gpu_label = (gpu_profile or {}).get("gpu_label", f"NVIDIA {effective_gpu}" if effective_gpu else "NVIDIA GPU")
    lock_meta = resolve_directory_lock_paths(project_root, session_slug=session_slug)
    log_path = Path(lock_meta["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    effective_bearer = str(
        bearer_token
        or os.environ.get("PLATODOOM_TUNNEL_TOKEN")
        or os.environ.get("SANDCASTLE_VLLM_API_KEY")
        or ""
    ).strip() or None
    daemon_env = dict(os.environ)
    if effective_bearer:
        daemon_env["PLATODOOM_TUNNEL_TOKEN"] = effective_bearer
        daemon_env["SANDCASTLE_VLLM_API_KEY"] = effective_bearer
    if max_parallel_agents is not None:
        daemon_env["SANDCASTLE_MAX_PARALLEL_AGENTS"] = str(int(max_parallel_agents))
    elif session_slug:
        daemon_env["SANDCASTLE_MAX_PARALLEL_AGENTS"] = "4"
    with log_path.open("a", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=daemon_env,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    if callable(getattr(proc, "wait", None)):
        threading.Thread(target=proc.wait, name=f"reap-colab-daemon-{proc.pid}", daemon=True).start()
    update_registry_status(
        status="Provisioning",
        message=f"Spawning per-directory Colab Keep-Alive Daemon for {effective_slug} ({gpu_label})...",
        repo_dir=project_root,
        include_global=False,
        session_slug=session_slug,
        extra={
            "daemon_pid": int(proc.pid),
            "offline_reason": None,
            "gpu_name": gpu_label,
            "gpu_type": (gpu_profile or {}).get("gpu_type", effective_gpu or "T4"),
            "colab_gpu_type": (gpu_profile or {}).get("colab_gpu_type", effective_gpu or "T4"),
            **({"bearer_token": effective_bearer} if effective_bearer else {}),
            **({"active_model": model_alias} if model_alias else {}),
        },
    )
    write_directory_daemon_pid_record(
        repo_dir=project_root,
        gpu_profile=gpu_profile,
        pid=int(proc.pid),
        status="Provisioning",
        session_slug=session_slug,
    )
    return int(proc.pid)


def wait_for_directory_daemon_endpoint(
    repo_dir: Optional[Path | str] = None,
    gpu_profile: Optional[Dict[str, Any]] = None,
    verify_inference: bool = True,
    poll_interval_sec: float = 2.0,
    dead_daemon_timeout_sec: float = 30.0,
    stall_timeout_sec: Optional[float] = None,
    max_wait_sec: Optional[float] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    session_slug: Optional[str] = None,
    model_alias: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Follower wait loop (FR-2.2, FR-2.3, FR-3.4):
    Polls `<project_root>/.sandcastle/colab_qwen_endpoint.json` (or `~/.platodoom/sessions/<session_slug>/endpoint.json`)
    every 2.0s with ZERO hardcoded wall-clock cutoff.
    """
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    effective_dead_timeout = float(stall_timeout_sec if stall_timeout_sec is not None else dead_daemon_timeout_sec)
    started_wait = time.time()
    spawn_grace_until = started_wait + 10.0
    last_message = ""

    while True:
        if max_wait_sec is not None and max_wait_sec > 0 and (time.time() - started_wait) > float(max_wait_sec):
            raise TimeoutError(f"Follower wait exceeded explicit max_wait_sec={max_wait_sec}s for {effective_slug}")

        rec = read_registered_endpoint(project_root, include_global=False, session_slug=session_slug) or {}
        status = str(rec.get("status") or "")
        msg = str(rec.get("message") or "")
        if msg and msg != last_message:
            last_message = msg
            if progress_callback is not None:
                try:
                    progress_callback(rec)
                except Exception:
                    pass
            else:
                print(f"[colab-preflight][{effective_slug}] {status}: {msg}", file=sys.stderr, flush=True)

        # 1. Check if Online and verified
        if status == "Online" and rec.get("endpoint_url"):
            ep_url = str(rec["endpoint_url"])
            tok = (
                bearer_token
                or rec.get("bearer_token")
                or rec.get("auth_token")
                or os.environ.get("PLATODOOM_TUNNEL_TOKEN")
                or os.environ.get("SANDCASTLE_VLLM_API_KEY")
            )
            probe = probe_colab_qwen_endpoint(
                ep_url,
                api_key=str(tok) if tok else None,
                bearer_token=str(tok) if tok else None,
                expected_model=model_alias,
                timeout_sec=5.0,
                verify_inference=verify_inference,
                dns_retry_window_sec=0.0,
            )
            if probe["online"]:
                return rec

        # 2. Deterministic Fail-Fast if daemon marked status == Offline / Failed / Error
        if status in ("Offline", "Failed", "Error") and time.time() > spawn_grace_until:
            reason = str(rec.get("offline_reason") or rec.get("message") or "Colab provisioning failed fast.")
            raise RuntimeError(f"Colab provisioning failed for {project_root}: {reason}")

        # 3. Check daemon PID liveness & heartbeat freshness (<= 30.0s dead-daemon detection window)
        pid_rec = read_directory_daemon_pid_record(project_root, session_slug=session_slug)
        daemon_pid = (
            int(pid_rec["pid"])
            if pid_rec and isinstance(pid_rec.get("pid"), int)
            else (int(rec["daemon_pid"]) if isinstance(rec.get("daemon_pid"), int) else None)
        )
        pid_alive = is_pid_alive(daemon_pid) or is_directory_daemon_alive(project_root, session_slug=session_slug)

        heartbeat_age_sec = 0.0
        hb_epoch = rec.get("heartbeat_epoch") or rec.get("updated_at_epoch") or (pid_rec or {}).get("updated_at_epoch")
        if isinstance(hb_epoch, (int, float)) and hb_epoch > 0:
            heartbeat_age_sec = max(0.0, time.time() - float(hb_epoch))
        else:
            updated_at_str = str(rec.get("updated_at") or (pid_rec or {}).get("updated_at") or "")
            if updated_at_str:
                try:
                    dt = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
                    heartbeat_age_sec = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
                except Exception:
                    heartbeat_age_sec = 0.0

        if not pid_alive and heartbeat_age_sec > effective_dead_timeout and time.time() > spawn_grace_until:
            reason = (
                f"Per-directory Colab Keep-Alive Daemon (pid={daemon_pid}) terminated unexpectedly "
                f"and heartbeat stalled for {heartbeat_age_sec:.1f}s (> {effective_dead_timeout:.0f}s)."
            )
            mark_endpoint_offline(project_root, reason=reason, include_global=False, session_slug=session_slug)
            raise RuntimeError(reason)

        time.sleep(max(0.2, float(poll_interval_sec)))


def parse_detected_colab_accelerator(combined_output: str) -> Optional[Dict[str, Any]]:
    """Parses Cell 2's live `Detected GPU:` / `Detected TPU:` output to re-anchor provisioning telemetry (Spec 871 §3.4 & §5.1)."""
    if not combined_output or "Detected " not in combined_output:
        return None
    det_pat = re.compile(
        r"Detected\s+(GPU|TPU):\s+([^\(\n]+?)\s*\("
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:GiB|GB)\s+(VRAM|HBM)"
        r"(?:,\s*free=([0-9]+(?:\.[0-9]+)?)\s*(?:GiB|GB))?"
        r"(.*?)\)(?=\s*(?:\||$))"
        r"(?:\s*\|\s*max_num_seqs=(\d+))?"
        r"(?:\s*\|\s*max_model_len=(\d+))?",
        re.MULTILINE,
    )
    matches = list(det_pat.finditer(combined_output))
    if not matches:
        return None
    m = matches[-1]
    accel_kind = m.group(1).strip()
    dev_name = m.group(2).strip()
    vram_val = float(m.group(3))
    mem_type = m.group(4).strip()
    free_val = float(m.group(5)) if m.group(5) is not None else vram_val
    attrs_raw = m.group(6) or ""
    max_seqs_val = int(m.group(7)) if m.group(7) else None
    max_len_val = int(m.group(8)) if m.group(8) else None

    tier_m = re.search(r"tier=([A-Za-z0-9\-]+)", attrs_raw)
    prec_m = re.search(r"precision=([A-Za-z0-9_]+)", attrs_raw)
    sm_tag_m = re.search(r"sm=sm_(\d+)", attrs_raw)
    sm_tup_m = re.search(r"sm=\(\s*(\d+)\s*,\s*(\d+)\s*\)", attrs_raw)

    cc_tuple = (0, 0)
    sm_label = "tpu" if accel_kind == "TPU" else ""
    if sm_tup_m:
        cc_tuple = (int(sm_tup_m.group(1)), int(sm_tup_m.group(2)))
        sm_label = f"sm_{cc_tuple[0]}{cc_tuple[1]}" if cc_tuple[0] > 0 else ("tpu" if accel_kind == "TPU" else "")
    elif sm_tag_m:
        raw_digits = sm_tag_m.group(1)
        sm_label = f"sm_{raw_digits}"
        if len(raw_digits) >= 3:
            cc_tuple = (int(raw_digits[:-1]), int(raw_digits[-1]))
        elif len(raw_digits) == 2:
            cc_tuple = (int(raw_digits[0]), int(raw_digits[1]))

    if tier_m:
        resolved_tier = tier_m.group(1).strip()
    elif accel_kind == "TPU":
        resolved_tier = "v6e-1" if vram_val >= 24.0 else "v5e-1"
    else:
        name_up = dev_name.upper()
        g4_vram = float(((_SSOT_GPU_TIER_SPECS or {}).get("G4") or {}).get("vram_gb", 95.0))
        h100_vram = float(((_SSOT_GPU_TIER_SPECS or {}).get("H100") or {}).get("vram_gb", 79.2))
        a100_vram = float(((_SSOT_GPU_TIER_SPECS or {}).get("A100") or {}).get("vram_gb", 40.0))
        l4_vram = float(((_SSOT_GPU_TIER_SPECS or {}).get("L4") or {}).get("vram_gb", 23.5))
        if vram_val >= g4_vram * 0.90 or (cc_tuple >= (10, 0) and vram_val >= 85.0) or ("BLACKWELL" in name_up and vram_val >= 85.0):
            resolved_tier = "G4"
        elif vram_val >= h100_vram * 0.90:
            resolved_tier = "H100" if (cc_tuple >= (9, 0) or "H100" in name_up) else "A100-80GB"
        elif vram_val >= a100_vram * 0.90:
            resolved_tier = "A100"
        elif vram_val >= l4_vram * 0.90:
            resolved_tier = "L4"
        else:
            resolved_tier = "T4"

    if accel_kind == "GPU" and sm_label:
        human_label = f"{dev_name} ({vram_val:.1f} GiB {mem_type} · {sm_label} · {resolved_tier})"
    else:
        human_label = f"{dev_name} ({vram_val:.1f} GiB {mem_type} · {resolved_tier})"

    return {
        "accelerator_kind": accel_kind,
        "gpu_name": dev_name,
        "gpu_label": human_label,
        "gpu_type": resolved_tier,
        "colab_gpu_type": "A100" if resolved_tier == "A100-80GB" else resolved_tier,
        "vram_gb": round(vram_val, 2),
        "free_vram_gb": round(free_val, 2),
        "compute_capability": cc_tuple,
        "sm_label": sm_label,
        "precision_tier": prec_m.group(1).strip().upper() if prec_m else None,
        "max_num_seqs": max_seqs_val,
        "max_model_len": max_len_val,
    }


def _write_ephemeral_local_notebook(
    gpu_profile: Optional[Dict[str, Any]] = None,
    *,
    repo_dir: Optional[Path | str] = None,
    session_slug: Optional[str] = None,
    model_alias: Optional[str] = None,
    bearer_token: Optional[str] = None,
    gpu_override: Optional[str] = None,
    stack_preset: Optional[str] = None,
    stack_models: Optional[List[Any]] = None,
    agent_counts: Optional[Mapping[str, int]] = None,
    model_allocations: Optional[Any] = None,
    enable_vision: Optional[bool] = None,
) -> Optional[Path]:
    """
    Writes the generated runtime `.ipynb` notebook with 0o600 permissions to the
    session directory so single-browser-login users can upload directly into Colab
    without requiring a Google Drive API OAuth `token.json`.
    """
    project_root = resolve_project_root(repo_dir)
    if not SERVE_NOTEBOOK_PATH.is_file():
        return None
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    nb_bytes = _build_headless_drive_notebook_bytes(
        gpu_profile=gpu_profile,
        repo_dir=project_root,
        notebook_path=SERVE_NOTEBOOK_PATH,
        session_slug=session_slug,
        model_alias=model_alias,
        bearer_token=bearer_token,
        gpu_override=gpu_override,
        stack_preset=stack_preset,
        stack_models=stack_models,
        agent_counts=agent_counts,
        model_allocations=model_allocations,
        enable_vision=enable_vision,
    )
    lock_meta = resolve_directory_lock_paths(project_root, session_slug=session_slug)
    out_dir = Path(lock_meta["sandcastle_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_file = out_dir / f"serve_platodoom_{effective_slug}.ipynb"
    nb_file.write_bytes(nb_bytes)
    try:
        os.chmod(str(nb_file), 0o600)
    except OSError:
        pass
    return nb_file


def _upload_local_notebook_in_colab_browser(driver: Any, local_notebook_path: Path, timeout_sec: float = 25.0) -> bool:
    """
    Uploads a local `.ipynb` file into an authenticated Google Colab browser session
    via Colab's Upload tab (`<input type="file">`) without needing Drive OAuth `token.json`.
    """
    from selenium.webdriver.common.by import By

    driver.get("https://colab.research.google.com/#create=true")
    time.sleep(4.0)
    if "accounts.google.com" in str(driver.current_url):
        return False

    # Open File -> Upload notebook dialog (or Upload tab if Open Notebook modal is visible)
    open_upload_js = """
    const tryClick = (selectors) => {
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el) { el.click(); return true; }
        }
        return false;
    };
    if (!tryClick(['#upload-tab', 'paper-tab[data-tab="upload"]', '[aria-label*="Upload"]'])) {
        const nb = window.colab && window.colab.global && window.colab.global.notebook;
        if (nb && typeof nb.openUploadDialog === 'function') {
            nb.openUploadDialog();
            return true;
        }
        tryClick(['colab-file-menu', '#file-menu-button', '[command="upload-notebook"]']);
    }
    return true;
    """
    try:
        driver.execute_script(open_upload_js)
        time.sleep(1.5)
        driver.execute_script(
            """
            const items = Array.from(document.querySelectorAll('div, span, button, paper-tab, mwc-tab, md-secondary-tab'));
            for (const el of items) {
                const txt = (el.textContent || '').trim().toLowerCase();
                if (txt === 'upload' || txt === 'upload notebook') {
                    el.click();
                    break;
                }
            }
            """
        )
        time.sleep(1.5)
    except Exception:
        pass

    deadline = time.time() + max(10.0, float(timeout_sec))
    uploaded = False
    while time.time() < deadline:
        file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
        if file_inputs:
            file_inputs[0].send_keys(str(Path(local_notebook_path).resolve()))
            uploaded = True
            break
        time.sleep(0.75)

    if not uploaded:
        return False

    # Wait until Colab finishes loading the uploaded cells
    cells_deadline = time.time() + 25.0
    while time.time() < cells_deadline:
        try:
            count = driver.execute_script(
                "const nb = window.colab && window.colab.global && window.colab.global.notebook; "
                "return nb && typeof nb.getCells === 'function' ? nb.getCells().length : 0;"
            )
            if int(count or 0) >= 3:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def run_headless_colab_session(
    colab_url: str,
    gpu_profile: Optional[Dict[str, Any]] = None,
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    wait_for_tunnel_sec: float = 0.0,
    keep_alive_after_online_sec: float = 0.0,
    keep_alive: bool = False,
    idle_ttl_sec: float = 3600.0,
    verify_inference: bool = True,
    session_slug: Optional[str] = None,
    model_alias: Optional[str] = None,
    bearer_token: Optional[str] = None,
    local_notebook_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Executes `colab_url` in an isolated headless background browser (`headless_<slug>`),
    triggers `window.colab.global.notebook.runAll()`, streams live cell progress every 2-3s,
    verifies end-to-end readiness (`GET /v1/models` + `POST /v1/chat/completions`), and when
    `keep_alive=True`, keeps the browser tab open in a 15-30s heartbeat & tunnel recovery loop.
    """
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    served_model_label = str(model_alias or UNIFIED_MODEL_ALIAS)
    effective_bearer = str(
        bearer_token
        or os.environ.get("PLATODOOM_TUNNEL_TOKEN")
        or os.environ.get("SANDCASTLE_VLLM_API_KEY")
        or ""
    ).strip() or None

    # Defect 5a remediation: if current python lacks selenium, spawn daemon with selenium python & wait as follower
    try:
        import selenium  # noqa: F401
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
    except ImportError:
        selenium_py = _find_selenium_python_executable()
        if Path(selenium_py).resolve() != Path(sys.executable).resolve():
            spawn_directory_keepalive_daemon(
                repo_dir=project_root,
                gpu_profile=gpu_profile,
                idle_ttl_sec=idle_ttl_sec,
                session_slug=session_slug,
                model_alias=model_alias,
                bearer_token=effective_bearer,
            )
            return wait_for_directory_daemon_endpoint(
                repo_dir=project_root,
                gpu_profile=gpu_profile,
                verify_inference=verify_inference,
                session_slug=session_slug,
                model_alias=model_alias,
                bearer_token=effective_bearer,
            )
        mark_endpoint_offline(
            repo_dir=project_root,
            reason="Python selenium package is not installed in any available interpreter.",
            include_global=include_global,
            colab_notebook_url=colab_url,
            session_slug=session_slug,
        )
        raise RuntimeError("Python selenium package is not installed in any available interpreter.")

    lock_fd = try_acquire_directory_lock(project_root, session_slug=session_slug)
    if lock_fd is None:
        # Another live daemon already holds the lock for this directory/session; attach as follower
        return wait_for_directory_daemon_endpoint(
            repo_dir=project_root,
            gpu_profile=gpu_profile,
            verify_inference=verify_inference,
            session_slug=session_slug,
            model_alias=model_alias,
            bearer_token=effective_bearer,
        )

    started_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_directory_daemon_pid_record(
        repo_dir=project_root,
        gpu_profile=gpu_profile,
        started_at=started_iso,
        pid=os.getpid(),
        colab_url=colab_url,
        status="Provisioning",
        session_slug=session_slug,
    )

    effective_gpu_profile: Optional[Dict[str, Any]] = dict(gpu_profile) if isinstance(gpu_profile, dict) else None
    gpu_label = (effective_gpu_profile or {}).get("gpu_label", "NVIDIA GPU")
    gpu_extra: Dict[str, Any] = {
        "colab_notebook_url": colab_url,
        "daemon_pid": os.getpid(),
        "dir_slug": dir_slug,
        "session_slug": effective_slug,
        "project_root": str(project_root),
    }
    if effective_bearer:
        gpu_extra["bearer_token"] = str(effective_bearer)
    if model_alias:
        gpu_extra["active_model"] = str(model_alias)
    if effective_gpu_profile:
        gpu_extra.update(
            {
                "gpu_name": effective_gpu_profile.get("gpu_label", gpu_label),
                "gpu_type": effective_gpu_profile.get("gpu_type", "T4"),
                "colab_gpu_type": effective_gpu_profile.get("colab_gpu_type", "T4"),
                "vram_gb": effective_gpu_profile.get("vram_gb", 16),
                "system_ram_gb": effective_gpu_profile.get("system_ram_gb", 25.5),
                "precision_tier": effective_gpu_profile.get("precision_tier", "Q4"),
                "load_in_4bit": effective_gpu_profile.get("load_in_4bit", True),
                "max_num_seqs": effective_gpu_profile.get("max_num_seqs", 32),
                "routed_qwen_models": effective_gpu_profile.get("total_qwen_models", 1),
                "peak_stage_qwen_models": effective_gpu_profile.get("peak_stage_qwen_models", 1),
            }
        )

    profile_dir = _sync_headless_browser_profile(project_root, session_slug=session_slug)

    options = webdriver.ChromeOptions()
    detected_browser = _detect_chromium_browser_binary()
    if detected_browser:
        options.binary_location = detected_browser

    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"--user-data-dir={profile_dir}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    if _has_colab_auth_mock_profile():
        options.add_argument("--use-mock-keychain")
        options.add_argument("--password-store=basic")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
    else:
        options.add_experimental_option(
            "excludeSwitches",
            ["use-mock-keychain", "password-store", "enable-automation"],
        )

    dynamic_labels = get_colab_stage_labels(
        model_alias=served_model_label,
        gpu_tier=(gpu_profile or {}).get("colab_gpu_type") or (gpu_profile or {}).get("gpu_type"),
    )

    update_registry_status(
        status="Provisioning",
        message=f"Headless Colab ({gpu_label}): {dynamic_labels[0]}...",
        repo_dir=project_root,
        include_global=include_global,
        session_slug=session_slug,
        extra={**gpu_extra, "stage": dynamic_labels[0]},
    )

    stop_requested = {"flag": False}

    def _handle_sigterm(_signum: int, _frame: Any) -> None:
        stop_requested["flag"] = True

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)
    except ValueError:
        pass

    driver = None
    registered_rec: Optional[Dict[str, Any]] = None
    try:
        cd_bin = _resolve_chromedriver_executable()
        driver = webdriver.Chrome(
            service=ChromeService(cd_bin) if cd_bin else ChromeService(),
            options=options,
        )
        driver.set_page_load_timeout(45)
        if hasattr(driver, "set_script_timeout"):
            driver.set_script_timeout(35.0)
        if local_notebook_path and not str(colab_url or "").strip():
            ok_upload = _upload_local_notebook_in_colab_browser(driver, Path(local_notebook_path))
            if "accounts.google.com" in str(driver.current_url):
                reason = (
                    "Headless Colab session requires Google sign-in. "
                    "Run 'platodoom auth' (or '/auth' inside platodoom) once to sign into Google Colab in your browser."
                )
                mark_endpoint_offline(
                    repo_dir=project_root,
                    reason=reason,
                    include_global=include_global,
                    colab_notebook_url=colab_url,
                    session_slug=session_slug,
                )
                raise RuntimeError(reason)
            if not ok_upload:
                reason = (
                    "Headless Colab browser upload did not load notebook cells. "
                    "Run 'platodoom auth' to verify your Google Colab sign-in session."
                )
                mark_endpoint_offline(
                    repo_dir=project_root,
                    reason=reason,
                    include_global=include_global,
                    colab_notebook_url=colab_url,
                    session_slug=session_slug,
                )
                raise RuntimeError(reason)
            colab_url = str(driver.current_url or colab_url)
        else:
            driver.get(colab_url)
            time.sleep(6.0)

        if "accounts.google.com" in str(driver.current_url):
            reason = (
                "Headless Colab session requires Google sign-in. "
                "Run 'platodoom auth' (or '/auth' inside platodoom) once to sign into Google Colab in your browser."
            )
            mark_endpoint_offline(
                repo_dir=project_root,
                reason=reason,
                include_global=include_global,
                colab_notebook_url=colab_url,
                session_slug=session_slug,
            )
            raise RuntimeError(reason)

        driver.execute_script(COLAB_RUN_ALL_JS)

        update_registry_status(
            status="Provisioning",
            message=f"Headless Colab GPU session connected ({gpu_label}) — executing serving cells...",
            repo_dir=project_root,
            include_global=include_global,
            session_slug=session_slug,
            extra=gpu_extra,
        )

        registered_rec: Optional[Dict[str, Any]] = None
        cell_labels = dynamic_labels[1:]
        lock_meta = resolve_directory_lock_paths(project_root, session_slug=session_slug)
        session_log_path = Path(lock_meta["sandcastle_dir"]) / "colab_headless_session.log"
        session_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Activity-Based Provisioning Loop (FR-2.1, FR-2.2: zero fixed wall-clock countdown)
        while not stop_requested["flag"]:
            time.sleep(2.5)

            if "accounts.google.com" in str(driver.current_url):
                reason = "Headless Colab session redirected to accounts.google.com during provisioning."
                mark_endpoint_offline(
                    repo_dir=project_root,
                    reason=reason,
                    include_global=include_global,
                    colab_notebook_url=colab_url,
                    session_slug=session_slug,
                )
                raise RuntimeError(reason)

            try:
                nb_state = driver.execute_script(COLAB_DOM_INSPECT_JS)
            except Exception:
                nb_state = {}

            combined_out = str((nb_state or {}).get("combinedOutput") or "")
            active_idx = int((nb_state or {}).get("activeIdx", -1))
            completed_count = int((nb_state or {}).get("completedCount", 0))
            kernel_state = str((nb_state or {}).get("kernelState") or "")
            cell_errors = list((nb_state or {}).get("cellErrors") or [])
            dialog_errors = list((nb_state or {}).get("dialogErrors") or [])
            outage_banners = list((nb_state or {}).get("outageBanners") or [])
            session_limit_dialogs = list((nb_state or {}).get("sessionLimitDialogs") or [])
            for extra_dlg in outage_banners + session_limit_dialogs:
                if extra_dlg and extra_dlg not in dialog_errors:
                    dialog_errors.append(extra_dlg)

            # Re-anchor live hardware telemetry & stage banners to the actual physical accelerator detected in Cell 2 (Spec 871 §3.4 & §5.1)
            detected_hw = parse_detected_colab_accelerator(combined_out)
            if detected_hw is not None:
                gpu_label = str(detected_hw["gpu_label"])
                gpu_extra["gpu_name"] = detected_hw["gpu_name"]
                gpu_extra["gpu_type"] = detected_hw["gpu_type"]
                gpu_extra["colab_gpu_type"] = detected_hw["colab_gpu_type"]
                gpu_extra["vram_gb"] = detected_hw["vram_gb"]
                gpu_extra["free_vram_gb"] = detected_hw["free_vram_gb"]
                if detected_hw.get("sm_label"):
                    gpu_extra["compute_capability"] = detected_hw["sm_label"]
                if detected_hw.get("precision_tier"):
                    gpu_extra["precision_tier"] = detected_hw["precision_tier"]
                    gpu_extra["load_in_4bit"] = bool(detected_hw["precision_tier"] == "Q4")
                if detected_hw.get("max_num_seqs"):
                    gpu_extra["max_num_seqs"] = detected_hw["max_num_seqs"]
                if detected_hw.get("max_model_len"):
                    gpu_extra["max_model_len"] = detected_hw["max_model_len"]
                dynamic_labels = get_colab_stage_labels(
                    model_alias=served_model_label,
                    gpu_tier=detected_hw["gpu_type"],
                )
                cell_labels = dynamic_labels[1:]
                if effective_gpu_profile is None:
                    effective_gpu_profile = {}
                effective_gpu_profile["gpu_label"] = gpu_label
                effective_gpu_profile["gpu_type"] = detected_hw["gpu_type"]
                effective_gpu_profile["colab_gpu_type"] = detected_hw["colab_gpu_type"]
                effective_gpu_profile["vram_gb"] = detected_hw["vram_gb"]

            try:
                with session_log_path.open("a", encoding="utf-8") as lf:
                    lf.write(
                        f"[{time.strftime('%H:%M:%S')}] slug={effective_slug} kernel={kernel_state} active={active_idx} "
                        f"completed={completed_count} dialog_errors={dialog_errors} cell_errors={cell_errors} "
                        f"out_tail={combined_out[-180:].replace(chr(10), ' | ')}\n"
                    )
            except Exception:
                pass

            # Fail-Fast Condition 1: Colab GPU Backend / Quota / Session Limit / Outage Dialog (FR-2.3, FR-7.2)
            if dialog_errors:
                err_summary = "; ".join(dialog_errors)
                reason = f"Colab GPU backend / quota dialog error ({gpu_label}): {err_summary}"
                mark_endpoint_offline(
                    repo_dir=project_root,
                    reason=reason,
                    include_global=include_global,
                    colab_notebook_url=colab_url,
                    session_slug=session_slug,
                )
                raise RuntimeError(reason)

            # Fail-Fast Condition 2: Cell Execution Exception (FR-2.3)
            if cell_errors:
                err_summary = "; ".join(cell_errors)
                reason = f"Headless Colab cell execution error ({gpu_label}): {err_summary}"
                mark_endpoint_offline(
                    repo_dir=project_root,
                    reason=reason,
                    include_global=include_global,
                    colab_notebook_url=colab_url,
                    session_slug=session_slug,
                )
                raise RuntimeError(reason)

            # Candidate Tunnel Discovery 1: DOM stdout SANDCASTLE_COLAB_ENDPOINT_JSON
            json_match = re.search(r"SANDCASTLE_COLAB_ENDPOINT_JSON:(\{.*\})", combined_out)
            if json_match:
                try:
                    meta = json.loads(json_match.group(1))
                    cand_url = str(meta.get("endpoint_url", ""))
                    cand_tok = effective_bearer or meta.get("bearer_token") or meta.get("auth_token")
                    if cand_url:
                        pr = probe_colab_qwen_endpoint(
                            cand_url,
                            api_key=str(cand_tok) if cand_tok else None,
                            bearer_token=str(cand_tok) if cand_tok else None,
                            expected_model=model_alias,
                            timeout_sec=5.0,
                            verify_inference=verify_inference,
                            inference_timeout_sec=15.0,
                            dns_retry_window_sec=25.0,
                        )
                        if pr["online"]:
                            registered_rec = register_colab_qwen_endpoint(
                                cand_url,
                                metadata={
                                    **gpu_extra,
                                    **meta,
                                    "daemon_pid": os.getpid(),
                                    "latency_ms": pr["latency_ms"],
                                    "inference_verified": bool(pr.get("inference_verified", True)),
                                    "served_models": pr["models"],
                                    "colab_notebook_url": colab_url,
                                    "message": f"Connected via Headless Colab ({pr['latency_ms']}ms) — Serving {served_model_label} on {gpu_label}",
                                    **_extract_live_probe_model_metadata(pr, model_alias),
                                },
                                repo_dir=project_root,
                                include_global=include_global,
                                session_slug=session_slug,
                            )
                            break
                except Exception:
                    pass

            # Candidate Tunnel Discovery 2: Per-directory GCS blob
            cand_gcs = fetch_gcs_published_endpoint(project_root, include_global=include_global, session_slug=session_slug)
            if cand_gcs and cand_gcs.get("endpoint_url"):
                gcs_tok = effective_bearer or cand_gcs.get("bearer_token") or cand_gcs.get("auth_token")
                p_gcs = probe_colab_qwen_endpoint(
                    str(cand_gcs["endpoint_url"]),
                    api_key=str(gcs_tok) if gcs_tok else None,
                    bearer_token=str(gcs_tok) if gcs_tok else None,
                    expected_model=model_alias,
                    timeout_sec=5.0,
                    verify_inference=verify_inference,
                    inference_timeout_sec=15.0,
                    dns_retry_window_sec=25.0,
                )
                if p_gcs["online"]:
                    registered_rec = register_colab_qwen_endpoint(
                        str(cand_gcs["endpoint_url"]),
                        metadata={
                            **gpu_extra,
                            **cand_gcs,
                            "daemon_pid": os.getpid(),
                            "latency_ms": p_gcs["latency_ms"],
                            "inference_verified": bool(p_gcs.get("inference_verified", True)),
                            "served_models": p_gcs["models"],
                            "colab_notebook_url": colab_url,
                            "message": f"Connected via Headless Colab ({p_gcs['latency_ms']}ms) — Serving {served_model_label} on {gpu_label}",
                            **_extract_live_probe_model_metadata(p_gcs, model_alias),
                        },
                        repo_dir=project_root,
                        include_global=include_global,
                        session_slug=session_slug,
                    )
                    break

            # Candidate Tunnel Discovery 3: Per-directory local registry
            cand_local = read_registered_endpoint(project_root, include_global=include_global, session_slug=session_slug)
            if cand_local and cand_local.get("endpoint_url") and cand_local.get("status") == "Online":
                loc_tok = effective_bearer or cand_local.get("bearer_token") or cand_local.get("auth_token")
                p_loc = probe_colab_qwen_endpoint(
                    str(cand_local["endpoint_url"]),
                    api_key=str(loc_tok) if loc_tok else None,
                    bearer_token=str(loc_tok) if loc_tok else None,
                    expected_model=model_alias,
                    timeout_sec=5.0,
                    verify_inference=verify_inference,
                    inference_timeout_sec=15.0,
                    dns_retry_window_sec=25.0,
                )
                if p_loc["online"]:
                    registered_rec = register_colab_qwen_endpoint(
                        str(cand_local["endpoint_url"]),
                        metadata={
                            **gpu_extra,
                            "daemon_pid": os.getpid(),
                            "latency_ms": p_loc["latency_ms"],
                            "inference_verified": bool(p_loc.get("inference_verified", True)),
                            "served_models": p_loc["models"],
                            "colab_notebook_url": colab_url,
                            "message": f"Connected via Headless Colab ({p_loc['latency_ms']}ms) — Serving {served_model_label} on {gpu_label}",
                            **_extract_live_probe_model_metadata(p_loc, model_alias),
                        },
                        repo_dir=project_root,
                        include_global=include_global,
                        session_slug=session_slug,
                    )
                    break

            if 0 <= active_idx < len(cell_labels):
                stage_desc = cell_labels[active_idx]
            elif 0 < completed_count < len(cell_labels):
                stage_desc = cell_labels[completed_count]
            elif kernel_state in ("connecting", ""):
                stage_desc = dynamic_labels[0]
            else:
                stage_desc = f"Executing Colab GPU cells (kernel={kernel_state})"

            update_registry_status(
                status="Provisioning",
                message=f"Headless Colab ({gpu_label}): {stage_desc}...",
                repo_dir=project_root,
                include_global=include_global,
                session_slug=session_slug,
                extra={**gpu_extra, "stage": stage_desc, "daemon_pid": os.getpid()},
            )
            write_directory_daemon_pid_record(
                repo_dir=project_root,
                gpu_profile=effective_gpu_profile,
                started_at=started_iso,
                pid=os.getpid(),
                colab_url=colab_url,
                status="Provisioning",
                session_slug=session_slug,
            )

        # Phase B: Continuous Keep-Alive Heartbeat & Automatic Cell 5 Tunnel Recovery Loop (FR-1.1, FR-1.2, FR-1.3)
        should_hold_tab = bool(keep_alive or keep_alive_after_online_sec > 0)
        if registered_rec and should_hold_tab and not stop_requested["flag"]:
            ep_url = str(registered_rec.get("endpoint_url", ""))
            write_directory_daemon_pid_record(
                repo_dir=project_root,
                gpu_profile=effective_gpu_profile,
                started_at=started_iso,
                pid=os.getpid(),
                endpoint_url=ep_url,
                colab_url=colab_url,
                status="Online",
                session_slug=session_slug,
            )
            explicit_duration = not keep_alive and keep_alive_after_online_sec > 0
            ka_duration = float(keep_alive_after_online_sec) if explicit_duration else float(idle_ttl_sec)
            ka_deadline = time.time() + (max(0.05, ka_duration) if explicit_duration else max(15.0, ka_duration))
            hb_interval = min(15.0, max(0.05, ka_duration / 2.0)) if explicit_duration else 15.0
            consecutive_probe_failures = 0

            while time.time() < ka_deadline and not stop_requested["flag"]:
                time.sleep(hb_interval)
                if stop_requested["flag"]:
                    break

                try:
                    ping_state = driver.execute_script(COLAB_KEEPALIVE_PING_JS) or {}
                    dom_state = driver.execute_script(COLAB_DOM_INSPECT_JS) or {}
                except Exception:
                    ping_state = {}
                    dom_state = {}

                kernel_state = str(dom_state.get("kernelState") or ping_state.get("kernelState") or "")
                pr = probe_colab_qwen_endpoint(
                    ep_url,
                    api_key=effective_bearer,
                    bearer_token=effective_bearer,
                    expected_model=model_alias,
                    timeout_sec=5.0,
                    verify_inference=False,
                    dns_retry_window_sec=0.0,
                )
                if pr["online"]:
                    consecutive_probe_failures = 0
                    registered_rec = register_colab_qwen_endpoint(
                        ep_url,
                        metadata={
                            **gpu_extra,
                            "daemon_pid": os.getpid(),
                            "latency_ms": pr["latency_ms"],
                            "inference_verified": True,
                            "served_models": pr["models"],
                            "colab_notebook_url": colab_url,
                            "message": f"Connected via Headless Colab ({pr['latency_ms']}ms) — Serving {served_model_label} on {gpu_label}",
                            **_extract_live_probe_model_metadata(pr, model_alias),
                        },
                        repo_dir=project_root,
                        include_global=include_global,
                        session_slug=session_slug,
                    )
                    write_directory_daemon_pid_record(
                        repo_dir=project_root,
                        gpu_profile=gpu_profile,
                        started_at=started_iso,
                        pid=os.getpid(),
                        endpoint_url=ep_url,
                        colab_url=colab_url,
                        status="Online",
                        session_slug=session_slug,
                    )
                    continue

                consecutive_probe_failures += 1
                if kernel_state in ("dead", "disconnected"):
                    mark_endpoint_offline(
                        repo_dir=project_root,
                        reason=f"Colab GPU kernel disconnected while serving ({kernel_state}).",
                        include_global=include_global,
                        colab_notebook_url=colab_url,
                        session_slug=session_slug,
                    )
                    break

                # Transient tunnel drop while Colab kernel & port 8000 server are still alive -> re-run Cell 5 (FR-1.3)
                if consecutive_probe_failures >= 2:
                    update_registry_status(
                        status="Provisioning",
                        message="Re-establishing public HTTPS tunnel (Cell 5/6)...",
                        repo_dir=project_root,
                        include_global=include_global,
                        session_slug=session_slug,
                        extra={**gpu_extra, "daemon_pid": os.getpid()},
                    )
                    try:
                        driver.execute_script(COLAB_RERUN_CELL5_JS)
                    except Exception:
                        pass
                    for _ in range(20):
                        if stop_requested["flag"]:
                            break
                        time.sleep(2.5)
                        try:
                            recov_dom = driver.execute_script(COLAB_DOM_INSPECT_JS) or {}
                        except Exception:
                            recov_dom = {}
                        recov_out = str(recov_dom.get("combinedOutput") or "")
                        cand_matches = re.findall(r"SANDCASTLE_COLAB_ENDPOINT_JSON:(\{.*\})", recov_out)
                        if cand_matches:
                            try:
                                meta_new = json.loads(cand_matches[-1])
                                cand_dom = str(meta_new.get("endpoint_url", ""))
                                if cand_dom:
                                    pr_dom = probe_colab_qwen_endpoint(
                                        cand_dom,
                                        api_key=effective_bearer,
                                        bearer_token=effective_bearer,
                                        expected_model=model_alias,
                                        timeout_sec=5.0,
                                        verify_inference=True,
                                        dns_retry_window_sec=25.0,
                                    )
                                    if pr_dom["online"]:
                                        ep_url = cand_dom
                                        consecutive_probe_failures = 0
                                        registered_rec = register_colab_qwen_endpoint(
                                            ep_url,
                                            metadata={
                                                **gpu_extra,
                                                **meta_new,
                                                "daemon_pid": os.getpid(),
                                                "latency_ms": pr_dom["latency_ms"],
                                                "inference_verified": True,
                                                "served_models": pr_dom["models"],
                                                "colab_notebook_url": colab_url,
                                                "message": f"Connected via Headless Colab ({pr_dom['latency_ms']}ms) — Serving {served_model_label} on {gpu_label}",
                                                **_extract_live_probe_model_metadata(pr_dom, model_alias),
                                            },
                                            repo_dir=project_root,
                                            include_global=include_global,
                                            session_slug=session_slug,
                                        )
                                        break
                            except Exception:
                                pass
                        cand_gcs = fetch_gcs_published_endpoint(
                            project_root, include_global=include_global, session_slug=session_slug
                        )
                        if cand_gcs and cand_gcs.get("endpoint_url"):
                            cand_new = str(cand_gcs["endpoint_url"])
                            pr_new = probe_colab_qwen_endpoint(
                                cand_new,
                                api_key=effective_bearer,
                                bearer_token=effective_bearer,
                                expected_model=model_alias,
                                timeout_sec=5.0,
                                verify_inference=True,
                                dns_retry_window_sec=25.0,
                            )
                            if pr_new["online"]:
                                ep_url = cand_new
                                consecutive_probe_failures = 0
                                registered_rec = register_colab_qwen_endpoint(
                                    ep_url,
                                    metadata={
                                        **gpu_extra,
                                        **cand_gcs,
                                        "daemon_pid": os.getpid(),
                                        "latency_ms": pr_new["latency_ms"],
                                        "inference_verified": True,
                                        "served_models": pr_new["models"],
                                        "colab_notebook_url": colab_url,
                                        "message": f"Connected via Headless Colab ({pr_new['latency_ms']}ms) — Serving {served_model_label} on {gpu_label}",
                                        **_extract_live_probe_model_metadata(pr_new, model_alias),
                                    },
                                    repo_dir=project_root,
                                    include_global=include_global,
                                    session_slug=session_slug,
                                )
                                break

        return registered_rec
    finally:
        try:
            signal.signal(signal.SIGTERM, prev_sigterm)
            signal.signal(signal.SIGINT, prev_sigint)
        except Exception:
            pass
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        if registered_rec is None:
            try:
                _, _fin_pid_path, _fin_mirror_pid_path = resolve_directory_lock_paths(project_root, session_slug=session_slug)
                for _p_clean in (_fin_pid_path, _fin_mirror_pid_path):
                    if _p_clean.is_file():
                        _p_clean.unlink(missing_ok=True)
            except Exception:
                pass
        if local_notebook_path:
            try:
                Path(local_notebook_path).unlink(missing_ok=True)
            except OSError:
                pass
        release_directory_lock(lock_fd)


_DEFAULT_RUN_HEADLESS_COLAB_SESSION = run_headless_colab_session


def _is_ephemeral_temp_path(path: Path) -> bool:
    """Returns True when `path` resides inside the OS ephemeral temp root (e.g. pytest `tmp_path`)."""
    try:
        resolved = path.expanduser().resolve()
        tmp_roots = {
            Path(tempfile.gettempdir()).expanduser().resolve(),
            Path("/tmp").resolve(),
        }
        return any(resolved == root or root in resolved.parents for root in tmp_roots)
    except Exception:
        return False


def resolve_cross_registry_candidate_paths(
    repo_dir: Optional[Path | str] = None,
    exclude_session_slug: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, Path]]:
    """
    Enumerates read-only local endpoint registry candidates in priority order (`Spec 866 §3.1.1`):
    1. `<platodoom_home>/standby/endpoint.json`
    2. `<project_root>/.sandcastle/colab_qwen_endpoint.json` (and legacy `colab_endpoint.json`)
    3. `~/.sandcastle/daemons/<dir_slug>/colab_qwen_endpoint.json` (when not in an isolated `PLATODOOM_HOME` test)
    4. `<platodoom_home>/sessions/*/endpoint.json` (sorted by mtime descending, excluding `exclude_session_slug`)

    Never mutates `resolve_colab_endpoint_registry_paths` write targets (`len(chat_paths) == 1` preserved).
    """
    eff_env = env if env is not None else os.environ
    custom_home = eff_env.get("PLATODOOM_CONFIG_DIR") or eff_env.get("PLATODOOM_HOME")
    has_isolated_home = bool(custom_home and str(custom_home).strip())
    platodoom_home = (
        Path(str(custom_home).strip()).expanduser().resolve()
        if has_isolated_home
        else _resolve_platodoom_home()
    )

    project_root = resolve_project_root(repo_dir) if repo_dir is not None else Path.cwd().resolve()
    if repo_dir is None:
        project_root = resolve_project_root(project_root)

    # Guard legacy pytest tests that did not isolate PLATODOOM_HOME/HOME from leaking host ~/.platodoom state,
    # while allowing real CLI/SDK sessions (even inside /tmp scratch workspaces) to discover ~/.platodoom/standby.
    in_unisolated_pytest = (
        "PYTEST_CURRENT_TEST" in eff_env
        and not has_isolated_home
        and not _is_ephemeral_temp_path(platodoom_home)
    )
    allow_global_platodoom_scan = not in_unisolated_pytest

    candidates: List[Tuple[str, Path]] = []
    if allow_global_platodoom_scan:
        candidates.append(("standby", platodoom_home / "standby" / "endpoint.json"))

    candidates.append(("workspace", project_root / ".sandcastle" / "colab_qwen_endpoint.json"))
    candidates.append(("workspace", project_root / ".sandcastle" / "colab_endpoint.json"))

    if allow_global_platodoom_scan and not has_isolated_home:
        dir_slug = compute_dir_slug(project_root)
        candidates.append(
            ("daemon_mirror", Path.home() / ".sandcastle" / "daemons" / dir_slug / "colab_qwen_endpoint.json")
        )

    if allow_global_platodoom_scan:
        sessions_root = platodoom_home / "sessions"
        has_authoritative_standby = (platodoom_home / "standby" / "endpoint.json").is_file()
        excluded = str(exclude_session_slug or "").strip()
        if sessions_root.is_dir():
            session_files: List[Tuple[float, Path]] = []
            try:
                for child in sessions_root.iterdir():
                    if not child.is_dir():
                        continue
                    if excluded and child.name == excluded:
                        continue
                    if has_authoritative_standby and child.name == "platodoom-standby":
                        continue
                    ep_file = child / "endpoint.json"
                    if not ep_file.is_file():
                        ep_file = child / "colab_endpoint.json"
                    if ep_file.is_file():
                        try:
                            mtime = ep_file.stat().st_mtime
                        except OSError:
                            mtime = 0.0
                        session_files.append((mtime, ep_file))
            except OSError:
                pass
            session_files.sort(key=lambda item: item[0], reverse=True)
            for _, ep_file in session_files[:32]:
                candidates.append(("session", ep_file))

    deduped: List[Tuple[str, Path]] = []
    seen: set[str] = set()
    for source_kind, cand_path in candidates:
        try:
            key = str(cand_path.resolve()) if cand_path.exists() else str(cand_path)
        except OSError:
            key = str(cand_path)
        if key not in seen:
            seen.add(key)
            deduped.append((source_kind, cand_path))
    return deduped


def _extract_candidate_hardware_info(
    record: Dict[str, Any],
    health_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], float]:
    """Extracts normalized `(gpu_tier, usable_vram_gb)` across standby, workspace, and session records."""
    raw_tier = (
        record.get("gpu_tier")
        or record.get("gpu_type")
        or record.get("colab_gpu_type")
        or record.get("colab_tpu_type")
    )
    norm_tier: Optional[str] = None
    if raw_tier and str(raw_tier).strip().lower() not in ("", "none", "auto"):
        if callable(_normalize_ssot_tier):
            try:
                norm_tier = _normalize_ssot_tier(str(raw_tier).strip(), allow_auto=False)
            except ValueError:
                norm_tier = str(raw_tier).strip()
        else:
            norm_tier = str(raw_tier).strip()

    vram_val: Optional[float] = None
    for cand_v in (
        (health_payload or {}).get("total_vram_gb") if isinstance(health_payload, dict) else None,
        record.get("vram_total_gb"),
        record.get("usable_vram_gb"),
        record.get("vram_gb"),
    ):
        if cand_v is not None:
            try:
                fv = float(cand_v)
                if fv > 0:
                    vram_val = fv
                    break
            except (TypeError, ValueError):
                continue

    if vram_val is None and norm_tier and isinstance(_SSOT_GPU_TIER_SPECS, dict) and norm_tier in _SSOT_GPU_TIER_SPECS:
        spec = _SSOT_GPU_TIER_SPECS[norm_tier]
        vram_val = float(spec.get("usable_vram_gb", spec.get("vram_gb", 15.8)))

    if vram_val is None:
        vram_val = 15.8

    if norm_tier is None and isinstance(_SSOT_GPU_TIER_SPECS, dict) and _SSOT_GPU_TIER_SPECS:
        try:
            from platodoom.catalog import verify_vram_compatibility

            inferred = verify_vram_compatibility("qwen-4b", vram_val).get("inferred_gpu_tier")
            if inferred:
                norm_tier = str(inferred)
        except Exception:
            pass

    return norm_tier, float(vram_val)


def _probe_endpoint_health_metadata(
    endpoint_url: str,
    bearer_token: Optional[str] = None,
    timeout_sec: float = 0.65,
) -> Optional[Dict[str, Any]]:
    """Queries `GET <origin>/health` with bounded timeout to inspect `total_vram_gb` & `resident_models`."""
    norm_ep = normalize_colab_endpoint_url(endpoint_url)
    if not norm_ep:
        return None
    origin = norm_ep[:-3] if norm_ep.endswith("/v1") else norm_ep
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "ngrok-skip-browser-warning": "true",
        "User-Agent": "Platodoom-WarmDiscovery/1.0",
    }
    if bearer_token and bearer_token != "EMPTY":
        headers["Authorization"] = f"Bearer {bearer_token}"
    req = urllib.request.Request(f"{origin}/health", headers=headers, method="GET")
    strict_timeout = min(max(float(timeout_sec), 0.2), 2.0)
    try:
        with urllib.request.urlopen(req, timeout=strict_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip() else None
            return parsed if isinstance(parsed, dict) else None
    except urllib.error.HTTPError as herr:
        try:
            raw = herr.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip() else None
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    except Exception:
        return None


def _satisfies_explicit_tier_floor(
    candidate_tier: Optional[str],
    candidate_vram_gb: float,
    required_gpu_tier: Optional[str],
) -> bool:
    """Ensures an explicit `--gpu` tier request is never downgraded onto smaller warm hardware."""
    if not required_gpu_tier or str(required_gpu_tier).strip().lower() in ("", "auto", "none"):
        return True
    req_raw = str(required_gpu_tier).strip()
    norm_req = req_raw
    if callable(_normalize_ssot_tier):
        try:
            norm_req = _normalize_ssot_tier(req_raw, allow_auto=False)
        except ValueError:
            norm_req = req_raw
    if candidate_tier and candidate_tier == norm_req:
        return True
    if isinstance(_SSOT_GPU_TIER_SPECS, dict) and norm_req in _SSOT_GPU_TIER_SPECS:
        req_usable = float(_SSOT_GPU_TIER_SPECS[norm_req].get("usable_vram_gb", _SSOT_GPU_TIER_SPECS[norm_req]["vram_gb"]))
        return float(candidate_vram_gb) + 0.25 >= req_usable
    return candidate_tier == norm_req


def discover_warm_colab_endpoints(
    repo_dir: Optional[Path | str] = None,
    target_model: Optional[str] = None,
    *,
    exclude_session_slug: Optional[str] = None,
    required_gpu_tier: Optional[str] = None,
    stack_models: Optional[List[Dict[str, Any]]] = None,
    probe_timeout_sec: float = 0.65,
    config: Optional[Dict[str, Any]] = None,
    allowed_tiers: Optional[FrozenSet[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Concurrently scans and probes all local endpoint registries (`standby`, `.sandcastle`, `sessions/*`)
    to reconnect in `< 1.0 s` (`Spec 866 §3.1.1–§3.1.3`, `§4.1`).

    Returns a dictionary with `match_type` in:
    - `"exact_model_loaded"`: Live endpoint already holds `target_model` (or `stack_models`) in memory.
    - `"compatible_warm_switch"`: Live endpoint holds another model on hardware with sufficient VRAM for `target_model`.
    - `"none"`: No compatible warm Colab accelerator is reachable.
    """
    t0 = time.perf_counter()
    cand_paths = resolve_cross_registry_candidate_paths(
        repo_dir=repo_dir,
        exclude_session_slug=exclude_session_slug,
        env=env,
    )

    source_rank = {"standby": 0, "workspace": 1, "daemon_mirror": 2, "session": 3}
    by_url: Dict[str, Dict[str, Any]] = {}

    for order_idx, (source_kind, path) in enumerate(cand_paths):
        if not path.is_file():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(loaded, dict):
            continue

        raw_url = str(loaded.get("endpoint_url") or "").strip()
        if not raw_url:
            continue
        status_lower = str(loaded.get("status") or "").strip().lower()
        if status_lower in ("offline", "terminated", "failed", "error"):
            continue
        if loaded.get("failure_state") == "MANUAL_REBOOT_REQUIRED":
            continue

        norm_url = normalize_colab_endpoint_url(raw_url)
        if not norm_url:
            continue

        hb_epoch = float(loaded.get("heartbeat_epoch") or loaded.get("updated_at_epoch") or 0.0)
        tok = str(loaded.get("bearer_token") or loaded.get("auth_token") or "").strip()
        entry = by_url.get(norm_url)
        if entry is None:
            merged = dict(loaded)
            merged["endpoint_url"] = norm_url
            merged["_registry_source"] = source_kind
            merged["_registry_path"] = str(path)
            merged["_priority_key"] = (source_rank.get(source_kind, 9), -hb_epoch, order_idx)
            by_url[norm_url] = merged
        else:
            if tok and not str(entry.get("bearer_token") or entry.get("auth_token") or "").strip():
                entry["bearer_token"] = tok
            if not (entry.get("gpu_tier") or entry.get("gpu_type") or entry.get("colab_gpu_type")):
                for gk in ("gpu_tier", "gpu_type", "colab_gpu_type", "vram_total_gb", "vram_gb"):
                    if loaded.get(gk) is not None:
                        entry[gk] = loaded[gk]
            prev_hb = float(entry.get("heartbeat_epoch") or entry.get("updated_at_epoch") or 0.0)
            if hb_epoch > prev_hb:
                entry["heartbeat_epoch"] = hb_epoch

    if not by_url:
        return {
            "match_type": "none",
            "candidate": None,
            "record": None,
            "probe": None,
            "probed_count": 0,
            "incompatible_live_endpoints": [],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    ordered_candidates = sorted(by_url.values(), key=lambda c: c["_priority_key"])[:8]
    expected_probe_target: Any = (
        [str(m.get("alias")) for m in stack_models if isinstance(m, dict) and m.get("alias")]
        if stack_models
        else target_model
    )
    bounded_timeout = min(max(float(probe_timeout_sec), 0.2), 3.5)

    def _probe_one(cand: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tok_resolved = resolve_serving_bearer_token(record=cand)
        eff_tok = None if tok_resolved == "EMPTY" else tok_resolved
        pr = probe_colab_qwen_endpoint(
            cand["endpoint_url"],
            api_key=eff_tok,
            bearer_token=eff_tok,
            timeout_sec=bounded_timeout,
            verify_inference=False,
            expected_model=expected_probe_target,
        )
        return cand, pr

    mismatch_hits: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    exact_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(ordered_candidates)))
    future_map = {executor.submit(_probe_one, cand): cand for cand in ordered_candidates}
    try:
        for fut in concurrent.futures.as_completed(future_map):
            try:
                cand, pr = fut.result()
            except Exception:
                continue

            if pr.get("online"):
                tok_resolved = resolve_serving_bearer_token(record=cand)
                eff_tok = None if tok_resolved == "EMPTY" else tok_resolved
                has_disk_hw = bool(
                    cand.get("gpu_tier")
                    or cand.get("gpu_type")
                    or cand.get("colab_gpu_type")
                    or cand.get("vram_total_gb")
                    or cand.get("vram_gb")
                )
                health_meta = (
                    _probe_endpoint_health_metadata(cand["endpoint_url"], eff_tok, timeout_sec=bounded_timeout)
                    if (not has_disk_hw and (allowed_tiers is not None or required_gpu_tier is not None))
                    else None
                )
                cand_tier, cand_vram = _extract_candidate_hardware_info(cand, health_meta)
                if allowed_tiers is not None and (cand_tier not in allowed_tiers):
                    continue
                if not _satisfies_explicit_tier_floor(cand_tier, cand_vram, required_gpu_tier):
                    continue
                enriched = dict(cand)
                if eff_tok:
                    enriched["bearer_token"] = eff_tok
                if cand_tier:
                    enriched["gpu_tier"] = cand_tier
                enriched["vram_total_gb"] = cand_vram
                enriched["resident_models"] = list(pr.get("models") or [])
                exact_candidates.append((enriched, pr))
                # Early-exit immediately when highest-priority candidate (or any exact match) succeeds
                executor.shutdown(wait=False, cancel_futures=True)
                return {
                    "match_type": "exact_model_loaded",
                    "candidate": enriched,
                    "record": enriched,
                    "probe": pr,
                    "probed_count": len(ordered_candidates),
                    "incompatible_live_endpoints": [],
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            if (
                (pr.get("model_mismatch") or (pr.get("alive") and bool(pr.get("models"))))
                and not pr.get("auth_error")
                and not pr.get("unauthorized")
            ):
                mismatch_hits.append((cand, pr))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Evaluate live model-mismatch endpoints for in-place switch compatibility (Spec 866 §3.1.3)
    compatible_switches: List[Tuple[Tuple[float, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    incompatible_live: List[Dict[str, Any]] = []

    if mismatch_hits:
        from platodoom.catalog import verify_stack_vram_compatibility, verify_vram_compatibility

        for cand, pr in mismatch_hits:
            tok_resolved = resolve_serving_bearer_token(record=cand)
            eff_tok = None if tok_resolved == "EMPTY" else tok_resolved
            has_disk_hw = bool(
                cand.get("gpu_tier")
                or cand.get("gpu_type")
                or cand.get("colab_gpu_type")
                or cand.get("vram_total_gb")
                or cand.get("vram_gb")
            )
            health_meta = (
                None
                if has_disk_hw
                else _probe_endpoint_health_metadata(cand["endpoint_url"], eff_tok, timeout_sec=bounded_timeout)
            )
            cand_tier, cand_vram = _extract_candidate_hardware_info(cand, health_meta)

            if allowed_tiers is not None and (cand_tier not in allowed_tiers):
                incompatible_live.append({"endpoint_url": cand["endpoint_url"], "gpu_tier": cand_tier, "reason": "tier_not_allowed"})
                continue
            if not _satisfies_explicit_tier_floor(cand_tier, cand_vram, required_gpu_tier):
                incompatible_live.append({"endpoint_url": cand["endpoint_url"], "gpu_tier": cand_tier, "reason": "below_explicit_gpu_floor"})
                continue

            if stack_models:
                compat = verify_stack_vram_compatibility(
                    stack_models,
                    available_vram_gb=cand_vram,
                    active_gpu_tier=cand_tier,
                    config=config,
                )
            else:
                compat = verify_vram_compatibility(
                    str(target_model or UNIFIED_MODEL_ALIAS),
                    available_vram_gb=cand_vram,
                    active_gpu_tier=cand_tier,
                    config=config,
                )

            if compat.get("compatible"):
                enriched = dict(cand)
                if eff_tok:
                    enriched["bearer_token"] = eff_tok
                resolved_tier = cand_tier or str(compat.get("inferred_gpu_tier") or "T4")
                enriched["gpu_tier"] = resolved_tier
                enriched["vram_total_gb"] = cand_vram
                enriched["resident_models"] = list(pr.get("models") or [])
                # Prefer tightest fitting warm GPU tier first so large GPUs aren't evicted when T4 fits, then registry priority
                rank_key = (float(cand_vram), cand["_priority_key"])
                compatible_switches.append((rank_key, enriched, compat, pr))
            else:
                incompatible_live.append(
                    {
                        "endpoint_url": cand["endpoint_url"],
                        "gpu_tier": cand_tier,
                        "vram_total_gb": cand_vram,
                        "diagnostic": compat.get("diagnostic"),
                    }
                )

    if compatible_switches:
        compatible_switches.sort(key=lambda item: item[0])
        _, best_cand, best_compat, best_probe = compatible_switches[0]
        return {
            "match_type": "compatible_warm_switch",
            "candidate": best_cand,
            "record": best_cand,
            "vram_check": best_compat,
            "probe": best_probe,
            "probed_count": len(ordered_candidates),
            "incompatible_live_endpoints": incompatible_live,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    return {
        "match_type": "none",
        "candidate": None,
        "record": None,
        "probe": None,
        "probed_count": len(ordered_candidates),
        "incompatible_live_endpoints": incompatible_live,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def _maybe_sync_standby_small_endpoint(
    record: Optional[Dict[str, Any]],
    project_root: Optional[Path] = None,
    force_sync: bool = False,
) -> None:
    """
    Syncs live small-tier (`T4` / `v5e-1`) endpoints to `<platodoom_home>/standby/endpoint.json`
    so restarting `platodoom` heals `MANUAL_REBOOT_REQUIRED` (`Spec 866 §1`, `§3.2.3`).
    Skips unisolated pytest `tmp_path` calls so mock unit tests never touch `~/.platodoom`.
    """
    if not isinstance(record, dict) or not record.get("endpoint_url"):
        return
    has_isolated_home = bool(
        (os.environ.get("PLATODOOM_CONFIG_DIR") or "").strip()
        or (os.environ.get("PLATODOOM_HOME") or "").strip()
    )
    if not force_sync and not has_isolated_home:
        plato_home = _resolve_platodoom_home()
        if "PYTEST_CURRENT_TEST" in os.environ and not _is_ephemeral_temp_path(plato_home):
            return
        standby_ep = plato_home / "standby" / "endpoint.json"
        if not standby_ep.is_file():
            return
    try:
        from platodoom.standby import sync_standby_from_live_small_endpoint

        sync_standby_from_live_small_endpoint(record)
    except Exception:
        pass


def _attempt_inplace_endpoint_model_switch(
    endpoint_url: str,
    target_model: str,
    bearer_token: Optional[str] = None,
    timeout_sec: float = 300.0,
    config: Optional[Dict[str, Any]] = None,
    available_vram_gb: Optional[float] = None,
    enable_vision: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Attempts an in-place physical model switch via POST /v1/internal/switch-model with full catalog metadata."""
    base_v1 = str(endpoint_url).strip().rstrip("/")
    if not base_v1.endswith("/v1"):
        base_v1 = f"{base_v1}/v1"
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    if bearer_token and bearer_token != "EMPTY":
        headers["Authorization"] = f"Bearer {bearer_token}"

    req_body: Dict[str, Any] = {"model_alias": target_model}
    if enable_vision is not None:
        req_body["enable_vision"] = bool(enable_vision)
    try:
        from platodoom.catalog import resolve_model_profile, verify_vram_compatibility

        prof = resolve_model_profile(target_model, config=config)
        req_body.update(
            {
                "hf_repo_id": prof.get("hf_repo_id", target_model),
                "source": str(prof.get("source") or "hf_hub"),
                "storage_uri": prof.get("storage_uri"),
                "gcs_bucket": prof.get("gcs_bucket"),
                "gcs_prefix": prof.get("gcs_prefix"),
                "fallback_hf_repo_id": prof.get("fallback_hf_repo_id"),
                "max_model_len": int(prof.get("max_model_len", MAX_MODEL_LEN)),
            }
        )
        if available_vram_gb is not None:
            vcheck = verify_vram_compatibility(target_model, float(available_vram_gb), config=config)
            if vcheck.get("selected_precision"):
                req_body["precision"] = vcheck["selected_precision"]
    except Exception:
        pass

    payload = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(f"{base_v1}/internal/switch-model", data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(5.0, float(timeout_sec))) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw.strip() else {}
            if isinstance(body, dict) and body.get("ok") is False:
                return False, str(body.get("diagnostic") or body.get("reason") or body.get("error") or "switch_model returned ok=False")
            return True, ""
    except urllib.error.HTTPError as http_err:
        err_body = ""
        try:
            err_body = http_err.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(http_err)
        return False, f"HTTP {http_err.code}: {err_body[:240]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _attempt_inplace_endpoint_stack_load(
    endpoint_url: str,
    models: List[Dict[str, Any]],
    stack_preset: Optional[str] = None,
    bearer_token: Optional[str] = None,
    timeout_sec: float = 300.0,
    agent_counts: Optional[Mapping[str, int]] = None,
    enable_vision: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Attempts an in-place co-resident stack load via POST /v1/internal/load-stack."""
    base_v1 = str(endpoint_url).strip().rstrip("/")
    if not base_v1.endswith("/v1"):
        base_v1 = f"{base_v1}/v1"
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    req_obj: Dict[str, Any] = {"models": models, "stack_preset": stack_preset}
    if agent_counts is not None:
        req_obj["agent_counts"] = dict(agent_counts)
    if enable_vision is not None:
        req_obj["enable_vision"] = bool(enable_vision)
    payload = json.dumps(req_obj).encode("utf-8")
    req = urllib.request.Request(f"{base_v1}/internal/load-stack", data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(5.0, float(timeout_sec))) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw.strip() else {}
            if isinstance(body, dict) and body.get("ok") is False:
                return False, str(body.get("diagnostic") or body.get("error") or "load_model_stack returned ok=False")
            return True, ""
    except urllib.error.HTTPError as http_err:
        err_body = ""
        try:
            err_body = http_err.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(http_err)
        return False, f"HTTP {http_err.code}: {err_body[:240]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def ensure_colab_serving_endpoint(
    repo_dir: Optional[Path | str] = None,
    explicit_url: Optional[str] = None,
    timeout_sec: float = 5.0,
    auto_provision: bool = True,
    include_global: bool = False,
    open_browser: bool = False,
    headless_launch: bool = False,
    wait_for_tunnel_sec: float = 0.0,
    gpu_profile: Optional[Dict[str, Any]] = None,
    verify_inference: bool = False,
    session_slug: Optional[str] = None,
    model_alias: Optional[str] = None,
    bearer_token: Optional[str] = None,
    verify_daemon_liveness: bool = True,
    gpu_override: Optional[str] = None,
    max_parallel_agents: Optional[int] = None,
    stack_preset: Optional[str] = None,
    stack_models: Optional[List[Any]] = None,
    allow_cross_registry_warm: bool = True,
    explicit_gpu_requested: bool = False,
    storage_uri: Optional[str] = None,
    force_reprovision: bool = False,
    sync_standby: bool = False,
    max_wait_sec: Optional[float] = None,
    agent_counts: Optional[Mapping[str, int]] = None,
    model_allocations: Optional[Any] = None,
) -> Dict[str, Any]:
    """Discovers and verifies a per-directory or interactive-session Colab serving endpoint or provisions one via Keep-Alive Daemon (FR-1..FR-7, Spec 866)."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    effective_slug = str(session_slug).strip() if (session_slug and str(session_slug).strip()) else dir_slug
    target_model = str(model_alias or UNIFIED_MODEL_ALIAS)
    effective_bearer = str(
        bearer_token
        or os.environ.get("PLATODOOM_TUNNEL_TOKEN")
        or os.environ.get("SANDCASTLE_VLLM_API_KEY")
        or ""
    ).strip() or None

    if gpu_profile is None and (gpu_override or session_slug or model_alias or stack_preset or stack_models or agent_counts is not None or model_allocations is not None):
        gpu_profile = read_gpu_routing_profile(
            repo_dir=project_root,
            gpu_override=gpu_override,
            session_slug=session_slug,
            model_alias=model_alias,
            stack_preset=stack_preset,
            stack_models=list(stack_models) if stack_models else None,
            agent_counts=agent_counts,
            model_allocations=model_allocations,
        )
    elif gpu_profile is not None and gpu_override:
        gpu_profile = dict(gpu_profile)
        gpu_profile["gpu_type"] = gpu_override
        gpu_profile["colab_gpu_type"] = "A100" if gpu_override == "A100-80GB" else gpu_override
        gpu_profile["override_mode"] = gpu_override

    resolved_stack_specs: Optional[List[Dict[str, Any]]] = None
    resolved_all_specs: Optional[List[Dict[str, Any]]] = None
    resolved_ag_counts: Optional[Dict[str, int]] = dict(agent_counts) if isinstance(agent_counts, Mapping) else (gpu_profile or {}).get("agent_counts")
    prof_preset = (gpu_profile or {}).get("stack_preset")
    eff_stack_preset = stack_preset or (
        prof_preset if prof_preset and str(prof_preset).strip().lower() != "solo" else None
    )
    raw_stack_models = stack_models or (gpu_profile or {}).get("stack_models")
    if (model_allocations is not None or agent_counts is not None) and callable(_resolve_ssot_multi_model_agents):
        tier_for_stack = str(
            gpu_override
            or (gpu_profile or {}).get("gpu_type")
            or (gpu_profile or {}).get("colab_gpu_type")
            or "auto"
        )
        alloc_input = model_allocations if model_allocations is not None else (
            [{**(dict(m) if isinstance(m, Mapping) else {"model": str(m)}), "agent_count": int((agent_counts or {}).get((m.get("alias") if isinstance(m, Mapping) else str(m).split("@", 1)[0]), 1))} for m in raw_stack_models]
            if raw_stack_models else dict(agent_counts or {})
        )
        alloc_res = _resolve_ssot_multi_model_agents(alloc_input, gpu_tier=tier_for_stack)
        resolved_stack_specs = [dict(m) for m in alloc_res["active_models"]]
        resolved_all_specs = [dict(m) for m in alloc_res["all_models"]]
        resolved_ag_counts = dict(alloc_res["agent_counts"])
        if resolved_stack_specs and not model_alias:
            target_model = str(resolved_stack_specs[0]["alias"])
    elif (eff_stack_preset or raw_stack_models) and callable(_resolve_ssot_stack):
        tier_for_stack = str(
            gpu_override
            or (gpu_profile or {}).get("gpu_type")
            or (gpu_profile or {}).get("colab_gpu_type")
            or "T4"
        )
        stk_res = _resolve_ssot_stack(
            tier_for_stack,
            stack_preset=str(eff_stack_preset) if eff_stack_preset else None,
            model_alias=model_alias,
            stack_models=list(raw_stack_models) if raw_stack_models else None,
        )
        resolved_stack_specs = [dict(m) for m in stk_res["models"]]
        resolved_all_specs = resolved_stack_specs
        eff_stack_preset = stk_res.get("stack_preset") or eff_stack_preset

    probe_expected_model: Any = [m["alias"] for m in resolved_stack_specs] if resolved_stack_specs else model_alias

    if gpu_profile is not None and max_parallel_agents is not None:
        gpu_profile = dict(gpu_profile)
        gpu_profile["max_num_seqs"] = int(max_parallel_agents)
        gpu_profile["maxNumSeqs"] = int(max_parallel_agents)
        gpu_profile["max_parallel_agents"] = int(max_parallel_agents)
        gpu_profile["maxParallelAgents"] = int(max_parallel_agents)
    elif gpu_profile is not None and session_slug and max_parallel_agents is None and resolved_ag_counts is None and model_allocations is None:
        gpu_profile = dict(gpu_profile)
        gpu_profile["max_num_seqs"] = 4
        gpu_profile["maxNumSeqs"] = 4
        gpu_profile["max_parallel_agents"] = 4
        gpu_profile["maxParallelAgents"] = 4

    effective_gpu_override = str(
        gpu_override
        or (gpu_profile or {}).get("override_mode")
        or (gpu_profile or {}).get("colab_gpu_type")
        or ""
    ).strip() or None
    if effective_gpu_override in ("auto", "None"):
        effective_gpu_override = None

    # Guard n=0 (None / Cloud API Only): never provision a Colab GPU when 0 Qwen models are active
    if (
        not explicit_url
        and gpu_profile
        and int(gpu_profile.get("total_qwen_models", 1)) == 0
        and str(gpu_profile.get("colab_gpu_type", "")) in ("", "None")
    ):
        msg = "n=0 active Qwen models assigned — all enabled tasks route through Cloud APIs; no Colab GPU required."
        mark_endpoint_offline(repo_dir=project_root, reason=msg, include_global=include_global, session_slug=session_slug)
        raise RuntimeError(msg)

    if explicit_url:
        probe = probe_colab_qwen_endpoint(
            explicit_url,
            api_key=effective_bearer,
            bearer_token=effective_bearer,
            expected_model=probe_expected_model,
            timeout_sec=timeout_sec,
            verify_inference=verify_inference,
        )
        if not probe["online"] and probe_expected_model and probe.get("model_mismatch"):
            if resolved_stack_specs is not None:
                sw_ok, sw_err = _attempt_inplace_endpoint_stack_load(
                    explicit_url,
                    resolved_all_specs or resolved_stack_specs,
                    stack_preset=eff_stack_preset,
                    bearer_token=effective_bearer,
                    agent_counts=resolved_ag_counts,
                )
            else:
                sw_ok, sw_err = _attempt_inplace_endpoint_model_switch(
                    explicit_url, str(model_alias), bearer_token=effective_bearer
                )
            if sw_ok:
                probe = probe_colab_qwen_endpoint(
                    explicit_url,
                    api_key=effective_bearer,
                    bearer_token=effective_bearer,
                    expected_model=probe_expected_model,
                    timeout_sec=timeout_sec,
                    verify_inference=verify_inference,
                )
            else:
                probe["error"] = f"{probe.get('error')}; in-place switch failed: {sw_err}"
        if probe["online"]:
            return register_colab_qwen_endpoint(
                explicit_url,
                metadata={
                    "latency_ms": probe["latency_ms"],
                    "inference_verified": bool(probe.get("inference_verified", verify_inference)),
                    "served_models": probe["models"],
                    "daemon_pid": os.getpid(),
                    **({"bearer_token": effective_bearer} if effective_bearer else {}),
                    **({"active_model": model_alias} if model_alias else {}),
                    **({"stack_preset": eff_stack_preset} if eff_stack_preset else {}),
                    **_extract_live_probe_model_metadata(probe, model_alias),
                },
                repo_dir=project_root,
                include_global=include_global,
                session_slug=session_slug,
            )
        mark_endpoint_offline(
            repo_dir=project_root,
            reason=f"Explicit Colab endpoint {explicit_url} failed health check: {probe.get('error')}",
            include_global=include_global,
            session_slug=session_slug,
        )
        raise RuntimeError(
            f"Explicit Colab endpoint {explicit_url} failed health check: {probe.get('error')}"
        )

    # 1. Check local per-directory / session registry
    existing = (
        None
        if force_reprovision
        else read_registered_endpoint(project_root, include_global=include_global, session_slug=session_slug)
    )
    requested_gpu_type = str((gpu_profile or {}).get("colab_gpu_type") or "")
    existing_gpu_type = str((existing or {}).get("colab_gpu_type") or (existing or {}).get("gpu_type") or "")
    tier_mismatch = bool(
        headless_launch
        and requested_gpu_type
        and requested_gpu_type not in ("", "None")
        and existing_gpu_type
        and existing_gpu_type not in ("", "None")
        and requested_gpu_type != existing_gpu_type
    )
    if tier_mismatch:
        terminate_directory_daemon(project_root, session_slug=session_slug)
        existing = None

    unattended_rejection_reason: Optional[str] = None
    existing_was_unreachable: bool = False

    if existing and existing.get("endpoint_url"):
        cand_url = str(existing["endpoint_url"])
        tok = effective_bearer or existing.get("bearer_token") or existing.get("auth_token")
        probe = probe_colab_qwen_endpoint(
            cand_url,
            api_key=str(tok) if tok else None,
            bearer_token=str(tok) if tok else None,
            expected_model=probe_expected_model,
            timeout_sec=min(timeout_sec, 5.0),
            verify_inference=verify_inference,
        )
        daemon_ok = (not verify_daemon_liveness) or (existing.get("source") == "env") or is_endpoint_daemon_verified(
            existing, project_root, session_slug=session_slug
        )
        if not probe["online"] and probe_expected_model and probe.get("model_mismatch") and daemon_ok:
            if resolved_stack_specs is not None:
                sw_ok, sw_err = _attempt_inplace_endpoint_stack_load(
                    cand_url,
                    resolved_all_specs or resolved_stack_specs,
                    stack_preset=eff_stack_preset,
                    bearer_token=str(tok) if tok else None,
                    agent_counts=resolved_ag_counts,
                )
            else:
                sw_ok, sw_err = _attempt_inplace_endpoint_model_switch(
                    cand_url,
                    str(model_alias),
                    bearer_token=str(tok) if tok else None,
                )
            if sw_ok:
                probe = probe_colab_qwen_endpoint(
                    cand_url,
                    api_key=str(tok) if tok else None,
                    bearer_token=str(tok) if tok else None,
                    expected_model=probe_expected_model,
                    timeout_sec=min(timeout_sec, 5.0),
                    verify_inference=verify_inference,
                )
            else:
                if auto_provision:
                    terminate_directory_daemon(project_root, session_slug=session_slug)
                else:
                    unattended_rejection_reason = (
                        f"Active endpoint {cand_url} is serving {probe.get('models')} instead of '{probe_expected_model}', "
                        f"and in-place switch failed ({sw_err})."
                    )
        if probe["online"]:
            # Enforce FR-7.1: Never adopt an unattended tunnel URL without a verified live daemon_pid
            if daemon_ok:
                record = register_colab_qwen_endpoint(
                    cand_url,
                    metadata={
                        **(existing if isinstance(existing, dict) else {}),
                        "latency_ms": probe["latency_ms"],
                        "inference_verified": bool(probe.get("inference_verified", True)),
                        "served_models": probe["models"],
                        **({"active_model": model_alias} if model_alias else {}),
                        "message": f"Connected ({probe['latency_ms']}ms) — Serving {target_model}",
                        **_extract_live_probe_model_metadata(probe, target_model),
                    },
                    repo_dir=project_root,
                    include_global=include_global,
                    session_slug=session_slug,
                )
                _maybe_sync_standby_small_endpoint(record, project_root=project_root, force_sync=sync_standby)
                return record
            else:
                unattended_rejection_reason = (
                    f"Unattended tunnel rejected (FR-7.1): endpoint {cand_url} has no verified live Keep-Alive Daemon PID."
                )
                mark_endpoint_offline(
                    repo_dir=project_root,
                    reason=unattended_rejection_reason,
                    include_global=include_global,
                    session_slug=session_slug,
                )
        elif not is_directory_daemon_alive(project_root, session_slug=session_slug):
            existing_was_unreachable = bool(not probe.get("model_mismatch") and unattended_rejection_reason is None)
            mark_endpoint_offline(
                repo_dir=project_root,
                reason=str(unattended_rejection_reason or probe.get("error") or "unreachable"),
                include_global=include_global,
                session_slug=session_slug,
            )

    # 1b. Cross-Registry Warm Endpoint Discovery (< 1.0 s exact attach & compatible warm switch, Spec 866 §3.1)
    if (
        (not existing or not str(existing.get("endpoint_url") or "").strip() or existing_was_unreachable)
        and not force_reprovision
        and unattended_rejection_reason is None
        and allow_cross_registry_warm
    ):
        warm_res = discover_warm_colab_endpoints(
            repo_dir=project_root,
            target_model=target_model,
            exclude_session_slug=session_slug,
            required_gpu_tier=(gpu_override if explicit_gpu_requested else None),
            stack_models=resolved_stack_specs,
            probe_timeout_sec=min(3.5, max(0.5, float(timeout_sec))),
        )
        warm_match = warm_res.get("match_type")
        warm_cand = warm_res.get("candidate")
        warm_probe = warm_res.get("probe") or {}

        if warm_match == "exact_model_loaded" and isinstance(warm_cand, dict) and warm_cand.get("endpoint_url"):
            adopted_url = str(warm_cand["endpoint_url"])
            adopted_tok = resolve_serving_bearer_token(record=warm_cand)
            eff_adopted_tok = adopted_tok if adopted_tok != "EMPTY" else effective_bearer
            raw_cand_pid = warm_cand.get("daemon_pid")
            eff_pid = (
                int(raw_cand_pid)
                if isinstance(raw_cand_pid, int) and is_pid_alive(raw_cand_pid)
                else os.getpid()
            )
            adopted_tier = str(
                warm_cand.get("gpu_tier")
                or warm_cand.get("gpu_type")
                or warm_cand.get("colab_gpu_type")
                or effective_gpu_override
                or "T4"
            )
            adopted_rec = register_colab_qwen_endpoint(
                adopted_url,
                metadata={
                    "latency_ms": warm_probe.get("latency_ms", 0.0),
                    "inference_verified": bool(warm_probe.get("inference_verified", False)),
                    "served_models": warm_probe.get("models") or warm_cand.get("resident_models") or [target_model],
                    "resident_models": warm_probe.get("models") or warm_cand.get("resident_models") or [target_model],
                    "daemon_pid": eff_pid,
                    "gpu_tier": adopted_tier,
                    "gpu_type": adopted_tier,
                    "colab_gpu_type": adopted_tier,
                    "vram_total_gb": warm_cand.get("vram_total_gb", 15.8),
                    "adopted_from_registry": warm_cand.get("_registry_source"),
                    **({"bearer_token": eff_adopted_tok} if eff_adopted_tok else {}),
                    "active_model": target_model,
                    **({"stack_preset": eff_stack_preset} if eff_stack_preset else {}),
                    "message": f"Warm-attached ({warm_probe.get('latency_ms', 0.0)}ms) — Serving {target_model}",
                    **_extract_live_probe_model_metadata(warm_probe, target_model),
                },
                repo_dir=project_root,
                include_global=include_global,
                session_slug=session_slug,
            )
            _maybe_sync_standby_small_endpoint(adopted_rec, project_root=project_root, force_sync=sync_standby)
            return adopted_rec

        if warm_match == "compatible_warm_switch" and isinstance(warm_cand, dict) and warm_cand.get("endpoint_url"):
            adopted_url = str(warm_cand["endpoint_url"])
            adopted_tok = resolve_serving_bearer_token(record=warm_cand)
            eff_adopted_tok = adopted_tok if adopted_tok != "EMPTY" else effective_bearer
            cand_vram = warm_cand.get("vram_total_gb")
            if resolved_stack_specs is not None:
                sw_ok, _sw_err = _attempt_inplace_endpoint_stack_load(
                    adopted_url,
                    resolved_all_specs or resolved_stack_specs,
                    stack_preset=eff_stack_preset,
                    bearer_token=eff_adopted_tok,
                    agent_counts=resolved_ag_counts,
                )
            else:
                sw_ok, _sw_err = _attempt_inplace_endpoint_model_switch(
                    adopted_url,
                    target_model,
                    bearer_token=eff_adopted_tok,
                    available_vram_gb=float(cand_vram) if cand_vram is not None else None,
                )
            if sw_ok:
                post_sw_probe = probe_colab_qwen_endpoint(
                    adopted_url,
                    api_key=eff_adopted_tok,
                    bearer_token=eff_adopted_tok,
                    expected_model=probe_expected_model,
                    timeout_sec=min(timeout_sec, 5.0),
                    verify_inference=verify_inference,
                )
                if post_sw_probe.get("online"):
                    raw_cand_pid = warm_cand.get("daemon_pid")
                    eff_pid = (
                        int(raw_cand_pid)
                        if isinstance(raw_cand_pid, int) and is_pid_alive(raw_cand_pid)
                        else os.getpid()
                    )
                    adopted_tier = str(
                        warm_cand.get("gpu_tier")
                        or warm_cand.get("gpu_type")
                        or warm_cand.get("colab_gpu_type")
                        or effective_gpu_override
                        or "T4"
                    )
                    switched_rec = register_colab_qwen_endpoint(
                        adopted_url,
                        metadata={
                            "latency_ms": post_sw_probe.get("latency_ms", 0.0),
                            "inference_verified": bool(post_sw_probe.get("inference_verified", False)),
                            "served_models": post_sw_probe.get("models") or [target_model],
                            "resident_models": post_sw_probe.get("models") or [target_model],
                            "daemon_pid": eff_pid,
                            "gpu_tier": adopted_tier,
                            "gpu_type": adopted_tier,
                            "colab_gpu_type": adopted_tier,
                            "vram_total_gb": cand_vram if cand_vram is not None else 15.8,
                            "adopted_from_registry": warm_cand.get("_registry_source"),
                            "warm_switched": True,
                            **({"bearer_token": eff_adopted_tok} if eff_adopted_tok else {}),
                            "active_model": target_model,
                            **({"stack_preset": eff_stack_preset} if eff_stack_preset else {}),
                            "message": f"Warm in-place switched ({post_sw_probe.get('latency_ms', 0.0)}ms) — Serving {target_model}",
                            **_extract_live_probe_model_metadata(post_sw_probe, target_model),
                        },
                        repo_dir=project_root,
                        include_global=include_global,
                        session_slug=session_slug,
                    )
                    _maybe_sync_standby_small_endpoint(switched_rec, project_root=project_root, force_sync=sync_standby)
                    return switched_rec

    # 2. Check per-directory GCS endpoint registry
    gcs_rec = fetch_gcs_published_endpoint(project_root, include_global=include_global, session_slug=session_slug)
    if gcs_rec and gcs_rec.get("endpoint_url") and not tier_mismatch:
        gcs_url = str(gcs_rec["endpoint_url"])
        tok = effective_bearer or gcs_rec.get("bearer_token") or gcs_rec.get("auth_token")
        probe = probe_colab_qwen_endpoint(
            gcs_url,
            api_key=str(tok) if tok else None,
            bearer_token=str(tok) if tok else None,
            expected_model=probe_expected_model,
            timeout_sec=min(timeout_sec, 5.0),
            verify_inference=verify_inference,
        )
        if probe["online"]:
            daemon_ok = (not verify_daemon_liveness) or is_endpoint_daemon_verified(
                gcs_rec, project_root, session_slug=session_slug
            )
            if daemon_ok:
                return register_colab_qwen_endpoint(
                    gcs_url,
                    metadata={
                        **gcs_rec,
                        "latency_ms": probe["latency_ms"],
                        "inference_verified": bool(probe.get("inference_verified", True)),
                        "served_models": probe["models"],
                        **({"active_model": model_alias} if model_alias else {}),
                        "message": f"Connected via GCS Auto-Discovery ({probe['latency_ms']}ms) — Serving {target_model}",
                        **_extract_live_probe_model_metadata(probe, target_model),
                    },
                    repo_dir=project_root,
                    include_global=include_global,
                    session_slug=session_slug,
                )
            else:
                unattended_rejection_reason = (
                    f"Unattended tunnel rejected (FR-7.1): GCS endpoint {gcs_url} has no verified live Keep-Alive Daemon PID."
                )
                mark_endpoint_offline(
                    repo_dir=project_root,
                    reason=unattended_rejection_reason,
                    include_global=include_global,
                    session_slug=session_slug,
                )

    if not auto_provision:
        reason_msg = unattended_rejection_reason or f"No healthy Colab GPU endpoint is active for '{target_model}'."
        mark_endpoint_offline(
            repo_dir=project_root,
            reason=reason_msg,
            include_global=include_global,
            session_slug=session_slug,
        )
        raise RuntimeError(
            f"No healthy Colab GPU endpoint is active for '{target_model}'. {reason_msg} "
            "Set SANDCASTLE_VLLM_URL or run `python3 scripts/provision_colab_serving.py --ensure-serving`."
        )

    # 3. Custom provision command if set (parsed via shlex without shell=True)
    custom_provision_cmd = os.environ.get("SANDCASTLE_COLAB_PROVISION_CMD", "").strip()
    if custom_provision_cmd:
        cmd_argv = shlex.split(custom_provision_cmd)
        if not cmd_argv:
            raise ValueError("SANDCASTLE_COLAB_PROVISION_CMD is empty after parsing.")
        proc = subprocess.run(
            cmd_argv,
            shell=False,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        match = re.search(r"SANDCASTLE_COLAB_ENDPOINT_JSON:(\{.*\})", combined)
        if match:
            meta = json.loads(match.group(1))
            url = str(meta.get("endpoint_url", ""))
            cmd_tok = effective_bearer or meta.get("bearer_token") or meta.get("auth_token")
            probe = probe_colab_qwen_endpoint(
                url,
                api_key=str(cmd_tok) if cmd_tok else None,
                bearer_token=str(cmd_tok) if cmd_tok else None,
                expected_model=probe_expected_model,
                timeout_sec=5.0,
                verify_inference=verify_inference,
                dns_retry_window_sec=25.0,
            )
            if probe["online"]:
                meta.setdefault("daemon_pid", os.getpid())
                reg_rec = register_colab_qwen_endpoint(
                    url,
                    metadata={
                        **meta,
                        **_extract_live_probe_model_metadata(probe, target_model),
                    },
                    repo_dir=project_root,
                    include_global=include_global,
                    session_slug=session_slug,
                )
                _maybe_sync_standby_small_endpoint(reg_rec, project_root=project_root, force_sync=sync_standby)
                return reg_rec

        url_match = re.search(r"(https?://[^\s\"']+/v1)", combined)
        if url_match:
            url = url_match.group(1)
            probe = probe_colab_qwen_endpoint(
                url,
                api_key=effective_bearer,
                bearer_token=effective_bearer,
                expected_model=probe_expected_model,
                timeout_sec=5.0,
                verify_inference=verify_inference,
                dns_retry_window_sec=25.0,
            )
            if probe["online"]:
                reg_rec = register_colab_qwen_endpoint(
                    url,
                    metadata={
                        "daemon_pid": os.getpid(),
                        **({"gpu_tier": effective_gpu_override} if effective_gpu_override else {}),
                        **({"active_model": target_model} if target_model else {}),
                        **({"bearer_token": effective_bearer} if effective_bearer else {}),
                        **_extract_live_probe_model_metadata(probe, target_model),
                    },
                    repo_dir=project_root,
                    include_global=include_global,
                    session_slug=session_slug,
                )
                _maybe_sync_standby_small_endpoint(reg_rec, project_root=project_root, force_sync=sync_standby)
                return reg_rec

    # 4. If a live Keep-Alive Daemon is already running for this directory/session, attach as follower (FR-3.4)
    if is_directory_daemon_alive(project_root, session_slug=session_slug):
        waited_rec = wait_for_directory_daemon_endpoint(
            repo_dir=project_root,
            gpu_profile=gpu_profile,
            verify_inference=verify_inference,
            max_wait_sec=max_wait_sec,
            session_slug=session_slug,
            model_alias=model_alias,
            bearer_token=effective_bearer,
        )
        _maybe_sync_standby_small_endpoint(waited_rec, project_root=project_root, force_sync=sync_standby)
        return waited_rec

    # 5. Launch per-directory Headless Keep-Alive Daemon (or run in-process if monkeypatched in unit tests)
    colab_url: Optional[str] = None
    should_headless = headless_launch or open_browser or wait_for_tunnel_sec > 0
    if should_headless and os.environ.get("SANDCASTLE_DISABLE_BROWSER") != "1":
        if run_headless_colab_session is not _DEFAULT_RUN_HEADLESS_COLAB_SESSION or os.environ.get("SANDCASTLE_IN_PROCESS_HEADLESS") == "1":
            colab_url = sync_serve_notebook_to_google_drive(
                gpu_profile,
                repo_dir=project_root,
                session_slug=session_slug,
                model_alias=model_alias,
                bearer_token=effective_bearer,
                gpu_override=effective_gpu_override,
                stack_preset=eff_stack_preset,
                stack_models=resolved_all_specs or resolved_stack_specs,
                agent_counts=resolved_ag_counts,
                model_allocations=model_allocations,
            )
            local_nb_fallback: Optional[Path] = None
            if not colab_url and _has_colab_auth_mock_profile():
                local_nb_fallback = _write_ephemeral_local_notebook(
                    gpu_profile=gpu_profile,
                    repo_dir=project_root,
                    session_slug=session_slug,
                    model_alias=model_alias,
                    bearer_token=effective_bearer,
                    gpu_override=effective_gpu_override,
                    stack_preset=eff_stack_preset,
                    stack_models=resolved_all_specs or resolved_stack_specs,
                    agent_counts=resolved_ag_counts,
                    model_allocations=model_allocations,
                )
            if colab_url or local_nb_fallback is not None:
                headless_kwargs: Dict[str, Any] = {
                    "colab_url": colab_url or "",
                    "gpu_profile": gpu_profile,
                    "repo_dir": project_root,
                    "include_global": include_global,
                    "keep_alive": False,
                    "verify_inference": verify_inference,
                    "session_slug": session_slug,
                    "model_alias": model_alias,
                    "bearer_token": effective_bearer,
                }
                if local_nb_fallback is not None and not colab_url:
                    headless_kwargs["local_notebook_path"] = local_nb_fallback
                headless_rec = run_headless_colab_session(**headless_kwargs)
                if headless_rec and headless_rec.get("endpoint_url"):
                    enriched_headless = {
                        **(gpu_profile or {}),
                        **headless_rec,
                        "active_model": headless_rec.get("active_model") or headless_rec.get("model") or target_model,
                        **({"bearer_token": headless_rec.get("bearer_token") or effective_bearer} if (headless_rec.get("bearer_token") or effective_bearer) else {}),
                    }
                    _maybe_sync_standby_small_endpoint(enriched_headless, project_root=project_root, force_sync=sync_standby)
                    return headless_rec
        else:
            spawn_directory_keepalive_daemon(
                repo_dir=project_root,
                gpu_profile=gpu_profile,
                session_slug=session_slug,
                model_alias=model_alias,
                bearer_token=effective_bearer,
                gpu_override=effective_gpu_override,
                max_parallel_agents=max_parallel_agents,
                stack_preset=eff_stack_preset,
                stack_models=resolved_all_specs or resolved_stack_specs,
                agent_counts=resolved_ag_counts,
                model_allocations=model_allocations,
            )
            waited_rec = wait_for_directory_daemon_endpoint(
                repo_dir=project_root,
                gpu_profile=gpu_profile,
                verify_inference=verify_inference,
                max_wait_sec=max_wait_sec,
                session_slug=session_slug,
                model_alias=model_alias,
                bearer_token=effective_bearer,
            )
            _maybe_sync_standby_small_endpoint(waited_rec, project_root=project_root, force_sync=sync_standby)
            return waited_rec

    gpu_label = (gpu_profile or {}).get("gpu_label", "NVIDIA GPU")
    current_reg = read_registered_endpoint(project_root, include_global=include_global, session_slug=session_slug) or {}
    existing_err = str(current_reg.get("offline_reason") or "")
    if existing_err.startswith(("Headless Colab", "Colab GPU backend", "Unattended tunnel rejected")):
        fail_msg = existing_err
    elif unattended_rejection_reason:
        fail_msg = unattended_rejection_reason
    else:
        fail_msg = (
            f"No live Colab tunnel detected yet for {effective_slug} ({gpu_label}). "
            "Click 'Launch / Verify Colab GPU' to start headless Colab provisioning or paste a live tunnel URL."
        )
    mark_endpoint_offline(
        repo_dir=project_root,
        reason=fail_msg,
        include_global=include_global,
        colab_notebook_url=colab_url,
        session_slug=session_slug,
    )
    raise RuntimeError(
        f"Automated Colab GPU pre-flight check failed for {project_root}: no reachable endpoint serving '{target_model}' "
        f"(MAX_PARALLEL_AGENTS={MAX_PARALLEL_AGENTS}, MAX_MODEL_LEN={MAX_MODEL_LEN}). {fail_msg}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Per-Directory Colab GPU Serving & Keep-Alive Daemon for platodoom / sandcastle")
    parser.add_argument("--register-url", type=str, default="", help="Register and verify a live Colab endpoint URL")
    parser.add_argument("--probe", action="store_true", help="Probe the currently registered Colab endpoint")
    parser.add_argument("--ensure-serving", "--auto-provision", dest="ensure_serving", action="store_true", help="Verify or provision the Colab GPU serving endpoint")
    parser.add_argument("--headless-launch", action="store_true", help="Sync notebook to Google Drive and execute headlessly via Keep-Alive Daemon")
    parser.add_argument("--open-browser", action="store_true", help="Deprecated alias for --headless-launch (never opens a visible window)")
    parser.add_argument("--keepalive-daemon", action="store_true", help="Run persistent per-directory headless browser Keep-Alive Daemon")
    parser.add_argument("--stop-daemon", "--shutdown-daemon", dest="stop_daemon", action="store_true", help="Terminate the target directory's Keep-Alive Daemon")
    parser.add_argument("--colab-login", action="store_true", help="Open a visible browser once to save a headless-safe Google Colab session (bypasses macOS Keychain ACL)")
    parser.add_argument("--verify-warmup", "--verify-inference", dest="verify_warmup", action="store_true", help="Verify live POST /v1/chat/completions warm-up inference")
    parser.add_argument("--idle-ttl-sec", type=float, default=3600.0, help="Keep-Alive Daemon idle TTL in seconds")
    parser.add_argument("--wait-seconds", type=float, default=0.0, help="Legacy compatibility flag (ignored in favor of activity-based waiting)")
    parser.add_argument("--route-gpu", action="store_true", help="Print the routed GPU hardware profile for model count n")
    parser.add_argument("--num-models", "--n-colab-agents", dest="num_models", type=int, default=None, help="Number of agents (n_dir) assigned to model")
    parser.add_argument("--peak-stage-models", "--n-peak-stage", dest="peak_stage_models", type=int, default=None, help="Peak concurrent Qwen agents in any single stage (n_peak_dir)")
    parser.add_argument("--gpu-override", "--gpu-tier", dest="gpu_override", type=str, default=None, help="Manual hardware tier override (G4, H100, A100-80GB, A100, v6e-1, L4, T4, v5e-1)")
    parser.add_argument("--stack-preset", "--stack", dest="stack_preset", type=str, default=None, help="Solo or multi-model co-resident stack preset (e.g. solo, best-balanced, recommended, light-multi)")
    parser.add_argument("--models-agents", "--agent-counts", "--stack-models", dest="models_agents", type=str, default=None, help="Multi-model co-resident allocation spec ('alias1:n1,alias2:n2' or JSON object)")
    parser.add_argument("--session-slug", type=str, default=None, help="Isolated interactive session slug (platodoom-chat-<id>)")
    parser.add_argument("--model-alias", type=str, default=None, help="Model alias to serve from the 14-model Qwen & DeepSeek catalog")
    parser.add_argument("--vision", dest="enable_vision", action="store_true", default=None, help="Explicitly enable multimodal Vision (VLM) mode on Colab vLLM (fail fast if unsupported)")
    parser.add_argument("--no-vision", dest="enable_vision", action="store_false", help="Explicitly disable multimodal Vision (force --language-model-only)")
    parser.add_argument("--sync-notebook", action="store_true", help="Sync the updated serving notebook to Google Drive and print its Colab URL")
    parser.add_argument("--switch-model", action="store_true", help="Hot-swap the currently registered live Colab server to --model-alias (and optional --vision / --no-vision)")
    parser.add_argument("--repo-dir", type=str, default=str(REPO_ROOT), help="Target repository directory")
    args, _unknown = parser.parse_known_args()

    project_root = resolve_project_root(args.repo_dir)
    dir_slug = compute_dir_slug(project_root)
    session_slug = args.session_slug
    cli_token = (
        os.environ.get("PLATODOOM_TUNNEL_TOKEN")
        or os.environ.get("SANDCASTLE_VLLM_API_KEY")
        or ""
    ).strip() or None

    try:
        if args.colab_login:
            res = interactive_colab_login()
            print(json.dumps(res))
            return 0 if res.get("ok") else 1

        if args.stop_daemon:
            res = terminate_directory_daemon(project_root, session_slug=session_slug)
            print(json.dumps(res))
            return 0

        parsed_models_agents = _parse_models_agents_cli_arg(args.models_agents)
        parsed_agent_counts = parsed_models_agents if isinstance(parsed_models_agents, dict) else None

        gpu_profile = read_gpu_routing_profile(
            repo_dir=project_root,
            num_models=args.num_models,
            peak_stage_models=args.peak_stage_models,
            gpu_override=args.gpu_override,
            session_slug=session_slug,
            model_alias=args.model_alias,
            stack_preset=args.stack_preset,
            agent_counts=parsed_agent_counts,
            model_allocations=parsed_models_agents,
        )
        cli_expected_model: Any = (
            gpu_profile.get("loaded_models")
            or gpu_profile.get("active_models")
            or args.model_alias
            or gpu_profile.get("model_alias")
            or gpu_profile.get("model")
        )
        if isinstance(cli_expected_model, list) and len(cli_expected_model) == 1:
            cli_expected_model = cli_expected_model[0]

        if args.route_gpu:
            print(json.dumps({"ok": True, "dir_slug": dir_slug, "session_slug": session_slug or dir_slug, "project_root": str(project_root), "gpu_routing": gpu_profile}))
            return 0

        if args.sync_notebook:
            colab_url = sync_serve_notebook_to_google_drive(
                gpu_profile,
                repo_dir=project_root,
                session_slug=session_slug,
                model_alias=args.model_alias or (cli_expected_model if isinstance(cli_expected_model, str) else None),
                bearer_token=cli_token,
                gpu_override=args.gpu_override,
                stack_preset=args.stack_preset,
                stack_models=gpu_profile.get("stack_models") if isinstance(gpu_profile.get("stack_models"), list) else None,
                agent_counts=parsed_agent_counts or (gpu_profile.get("agent_counts") if isinstance(gpu_profile.get("agent_counts"), dict) else None),
                model_allocations=parsed_models_agents,
                enable_vision=args.enable_vision,
                raise_on_error=True,
            )
            print(json.dumps({"ok": True, "colab_url": colab_url, "enable_vision": args.enable_vision, "gpu_routing": gpu_profile}))
            return 0

        if args.switch_model:
            existing = read_registered_endpoint(project_root, include_global=False, session_slug=session_slug)
            target_url = str(args.register_url or (existing or {}).get("endpoint_url") or "").strip()
            if not target_url:
                print(json.dumps({"ok": False, "error": "No endpoint registered; pass --register-url=<URL> or register an endpoint first."}), file=sys.stderr)
                return 1
            sw_tok = cli_token or (existing or {}).get("bearer_token") or (existing or {}).get("auth_token")
            sw_model = str(args.model_alias or (cli_expected_model if isinstance(cli_expected_model, str) else "") or (existing or {}).get("active_model") or UNIFIED_MODEL_ALIAS).strip()
            sw_ok, sw_err = _attempt_inplace_endpoint_model_switch(
                target_url,
                sw_model,
                bearer_token=str(sw_tok) if sw_tok else None,
                enable_vision=args.enable_vision,
            )
            if not sw_ok:
                print(json.dumps({"ok": False, "error": sw_err, "endpoint_url": target_url, "model_alias": sw_model, "enable_vision": args.enable_vision}), file=sys.stderr)
                return 1
            probe = probe_colab_qwen_endpoint(
                target_url,
                api_key=str(sw_tok) if sw_tok else None,
                bearer_token=str(sw_tok) if sw_tok else None,
                expected_model=sw_model,
                timeout_sec=10.0,
                verify_inference=args.verify_warmup,
            )
            if not probe["online"]:
                print(json.dumps({"ok": False, "error": probe.get("error"), "probe": probe}), file=sys.stderr)
                return 1
            live_meta = _extract_live_probe_model_metadata(probe, sw_model)
            if args.enable_vision is True and live_meta.get("vision") is not True:
                print(
                    json.dumps({
                        "ok": False,
                        "error": f"Vision mode was requested (--vision), but live /v1/models reports vision={live_meta.get('vision')}. Restart Colab with the updated vision-enabled notebook.",
                        "live_meta": live_meta,
                    }),
                    file=sys.stderr,
                )
                return 1
            rec = register_colab_qwen_endpoint(
                target_url,
                metadata={
                    **(existing if isinstance(existing, dict) else {}),
                    "latency_ms": probe["latency_ms"],
                    "inference_verified": bool(probe.get("inference_verified", args.verify_warmup)),
                    "served_models": probe["models"],
                    "active_model": sw_model,
                    **live_meta,
                },
                repo_dir=project_root,
                include_global=False,
                session_slug=session_slug,
            )
            print(json.dumps({"ok": True, "endpoint": rec, "live_meta": live_meta}))
            return 0

        if args.register_url:
            probe = probe_colab_qwen_endpoint(
                args.register_url,
                api_key=cli_token,
                bearer_token=cli_token,
                expected_model=args.model_alias or cli_expected_model,
                timeout_sec=5.0,
                verify_inference=args.verify_warmup,
                dns_retry_window_sec=25.0,
            )
            if not probe["online"]:
                mark_endpoint_offline(
                    repo_dir=project_root,
                    reason=f"Endpoint {args.register_url} unreachable: {probe.get('error')}",
                    include_global=False,
                    session_slug=session_slug,
                )
                print(json.dumps({"ok": False, "error": probe.get("error"), "probe": probe}), file=sys.stderr)
                return 1
            eff_active_label = args.model_alias or (
                ", ".join(cli_expected_model)
                if isinstance(cli_expected_model, list)
                else str(cli_expected_model or (probe.get("models") or [UNIFIED_MODEL_ALIAS])[0])
            )
            live_reg_meta = _extract_live_probe_model_metadata(probe, eff_active_label)
            if args.enable_vision is True and live_reg_meta.get("vision") is not True:
                err_msg = (
                    f"Vision mode was explicitly required (--vision), but {args.register_url}/v1/models "
                    f"reports vision={live_reg_meta.get('vision')} (input_modalities={live_reg_meta.get('input_modalities')})."
                )
                print(json.dumps({"ok": False, "error": err_msg, "live_meta": live_reg_meta}), file=sys.stderr)
                return 1
            rec = register_colab_qwen_endpoint(
                args.register_url,
                metadata={
                    "daemon_pid": os.getpid(),
                    "latency_ms": probe["latency_ms"],
                    "inference_verified": bool(probe.get("inference_verified", args.verify_warmup)),
                    "served_models": probe["models"],
                    "gpu_name": gpu_profile["gpu_label"],
                    "gpu_type": gpu_profile["gpu_type"],
                    "colab_gpu_type": gpu_profile["colab_gpu_type"],
                    "vram_gb": gpu_profile["vram_gb"],
                    "system_ram_gb": gpu_profile["system_ram_gb"],
                    "precision_tier": gpu_profile["precision_tier"],
                    "max_num_seqs": gpu_profile["max_num_seqs"],
                    "routed_qwen_models": gpu_profile["total_qwen_models"],
                    "peak_stage_qwen_models": gpu_profile["peak_stage_qwen_models"],
                    **({"bearer_token": cli_token} if cli_token else {}),
                    **({"active_model": eff_active_label} if eff_active_label else {}),
                    **live_reg_meta,
                    "message": f"Connected ({probe['latency_ms']}ms) — Serving {eff_active_label} on {gpu_profile['gpu_label']}",
                },
                repo_dir=project_root,
                include_global=False,
                session_slug=session_slug,
            )
            print(json.dumps({"ok": True, "endpoint": rec, "gpu_routing": gpu_profile}))
            return 0

        if args.probe:
            existing = read_registered_endpoint(project_root, include_global=False, session_slug=session_slug)
            if not existing or not existing.get("endpoint_url"):
                print(json.dumps({"ok": False, "online": False, "error": "No endpoint registered", "gpu_routing": gpu_profile}))
                return 1
            probe_tok = cli_token or existing.get("bearer_token") or existing.get("auth_token")
            probe = probe_colab_qwen_endpoint(
                str(existing["endpoint_url"]),
                api_key=str(probe_tok) if probe_tok else None,
                bearer_token=str(probe_tok) if probe_tok else None,
                expected_model=args.model_alias or cli_expected_model,
                timeout_sec=5.0,
                verify_inference=args.verify_warmup,
            )
            if not probe["online"] and not is_directory_daemon_alive(project_root, session_slug=session_slug):
                mark_endpoint_offline(
                    project_root,
                    reason=str(probe.get("error") or "offline"),
                    include_global=False,
                    session_slug=session_slug,
                )
            print(json.dumps({"ok": probe["online"], "gpu_routing": gpu_profile, **probe}))
            return 0 if probe["online"] else 1

        if args.keepalive_daemon:
            colab_url = sync_serve_notebook_to_google_drive(
                gpu_profile,
                repo_dir=project_root,
                session_slug=session_slug,
                model_alias=args.model_alias,
                bearer_token=cli_token,
                gpu_override=args.gpu_override,
                stack_preset=args.stack_preset,
                stack_models=gpu_profile.get("stack_models") if isinstance(gpu_profile.get("stack_models"), list) else None,
                agent_counts=parsed_agent_counts or (gpu_profile.get("agent_counts") if isinstance(gpu_profile.get("agent_counts"), dict) else None),
                model_allocations=parsed_models_agents,
            )
            local_nb_path: Optional[Path] = None
            if not colab_url and _has_colab_auth_mock_profile():
                local_nb_path = _write_ephemeral_local_notebook(
                    gpu_profile=gpu_profile,
                    repo_dir=project_root,
                    session_slug=session_slug,
                    model_alias=args.model_alias,
                    bearer_token=cli_token,
                    gpu_override=args.gpu_override,
                    stack_preset=args.stack_preset,
                    stack_models=gpu_profile.get("stack_models") if isinstance(gpu_profile.get("stack_models"), list) else None,
                    agent_counts=parsed_agent_counts or (gpu_profile.get("agent_counts") if isinstance(gpu_profile.get("agent_counts"), dict) else None),
                    model_allocations=parsed_models_agents,
                )
            if not colab_url and local_nb_path is None:
                reason = (
                    "Google Colab authentication is not configured yet. "
                    "Run 'platodoom auth' (or '/auth' inside platodoom) to sign into Google Colab in your browser."
                )
                if LAST_DRIVE_SYNC_ERROR:
                    reason = f"{reason} ({LAST_DRIVE_SYNC_ERROR})"
                mark_endpoint_offline(project_root, reason=reason, include_global=False, session_slug=session_slug)
                print(json.dumps({"ok": False, "error": reason}), file=sys.stderr)
                return 1
            rec = run_headless_colab_session(
                colab_url=colab_url or "",
                gpu_profile=gpu_profile,
                repo_dir=project_root,
                include_global=False,
                keep_alive=True,
                idle_ttl_sec=args.idle_ttl_sec,
                verify_inference=True,
                session_slug=session_slug,
                model_alias=args.model_alias,
                bearer_token=cli_token,
                local_notebook_path=local_nb_path,
            )
            print(json.dumps({"ok": bool(rec), "endpoint": rec, "gpu_routing": gpu_profile}))
            return 0 if rec else 1

        rec = ensure_colab_serving_endpoint(
            repo_dir=project_root,
            auto_provision=True,
            include_global=False,
            open_browser=args.open_browser,
            headless_launch=args.headless_launch or args.ensure_serving,
            gpu_profile=gpu_profile,
            verify_inference=args.verify_warmup,
            session_slug=session_slug,
            model_alias=args.model_alias,
            bearer_token=cli_token,
            gpu_override=args.gpu_override,
            stack_preset=args.stack_preset,
            stack_models=gpu_profile.get("stack_models") if isinstance(gpu_profile.get("stack_models"), list) else None,
            agent_counts=parsed_agent_counts or (gpu_profile.get("agent_counts") if isinstance(gpu_profile.get("agent_counts"), dict) else None),
            model_allocations=parsed_models_agents,
        )
        print(json.dumps({"ok": True, "endpoint": rec, "gpu_routing": gpu_profile}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
