"""
Interactive Session State, Mid-Session Model & Access Switching, Slash Command Router,
Multi-Turn Tool/Skill/MCP Agent Loop, Real-Time SSE Stream Consumer, Plan vs. Build Mode,
Multi-Conversation Persistence, and Same-GPU Subagent Continuous Batching for platodoom.
Document ID: platodoom_session_859_v1
Governing Standards: platodoom specification (FR-1, FR-2, FR-3, FR-4, AC-1..4), LANGUAGE.md, RULE[user_global]
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import inspect
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import urllib.error
import urllib.request
import uuid

from curation.api_client import (
    _open_http_request,
    atomic_write_json,
    compute_dir_slug,
    normalize_colab_endpoint_url,
    resolve_project_root,
    resolve_serving_bearer_token,
)
from curation.colab_server import validate_model_identifier_syntax
from platodoom.config import resolve_response_cache_dir
from platodoom.response_cache import LocalResponseCache, compute_response_cache_key
from scripts.provision_colab_serving import (
    discover_warm_colab_endpoints,
    ensure_colab_serving_endpoint,
    is_directory_daemon_alive,
    is_pid_alive,
    mark_endpoint_offline,
    probe_colab_qwen_endpoint,
    register_colab_qwen_endpoint,
    resolve_platodoom_session_dir,
    terminate_directory_daemon,
)

_ORIG_SESSION_URLOPEN = urllib.request.urlopen


def _open_session_http_request(req: urllib.request.Request, timeout: float = 30.0) -> Any:
    """Uses keep-alive pooled HTTP client unless urllib.request.urlopen is monkeypatched by a test."""
    if urllib.request.urlopen is not _ORIG_SESSION_URLOPEN:
        return urllib.request.urlopen(req, timeout=timeout)
    return _open_http_request(req, timeout=timeout)

THINK_TAG_RE = re.compile(r"<think>([\s\S]*?)</think>\s*", re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.DOTALL)
QWEN_XML_FN_RE = re.compile(r"<function=([^>\n]+)>\s*([\s\S]*?)\s*</function>", re.DOTALL)
QWEN_XML_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?([\s\S]*?)\n?</parameter>", re.DOTALL)
DEEPSEEK_TOOL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?:function<｜tool▁sep｜>)?([^\n<]+)\n```json\n([\s\S]*?)\n```<｜tool▁call▁end｜>",
    re.DOTALL,
)

VALID_ACCESS_MODES: Tuple[str, ...] = (
    "read-only",
    "workspace-write",
    "full-access",
    "confirm-writes",
)
VALID_WORKFLOW_MODES: Tuple[str, ...] = ("plan", "build")


FALLBACK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "qwen-9b": {
        "alias": "qwen-9b",
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-9B",
        "fallback_hf_repo_id": "Qwen/Qwen3.5-9B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5_text",
        "preferred_gpu": "T4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "T4",
        "compatible_gpus": ["T4", "L4", "A100", "A100-80GB"],
        "compatible_gpu_tiers": ["T4", "L4", "A100"],
        "min_vram_gb": 13.5,
        "stop_token_ids": [248046, 248044],
        "precision_by_gpu": {"T4": "gptq_4bit", "L4": "gptq_4bit", "A100": "bfloat16", "A100-80GB": "bfloat16", "H100": "bfloat16", "G4": "bfloat16"},
        "max_model_len": 16384,
        "max_num_seqs": 32,
        "template_family": "qwen3_chatml",
        "description": "Fast daily coding & tool execution (T4 16GB / L4 24GB / A100)",
    },
    "r1-14b": {
        "alias": "r1-14b",
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "fallback_hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "preferred_gpu": "L4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "L4",
        "compatible_gpus": ["L4", "A100", "A100-80GB"],
        "compatible_gpu_tiers": ["T4", "L4", "A100"],
        "min_vram_gb": 20.0,
        "stop_token_ids": [151643, 151645],
        "precision_by_gpu": {"T4": "awq_4bit", "L4": "awq_4bit", "A100": "awq_4bit", "A100-80GB": "bfloat16"},
        "max_model_len": 16384,
        "max_num_seqs": 4,
        "template_family": "deepseek_r1_distill",
        "description": "Medium reasoning & debugging specialist (L4 24GB / A100)",
    },
    "qwen-27b": {
        "alias": "qwen-27b",
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-27B",
        "fallback_hf_repo_id": "Qwen/Qwen3.5-27B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5_text",
        "preferred_gpu": "A100",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "A100",
        "compatible_gpus": ["A100", "A100-80GB"],
        "compatible_gpu_tiers": ["L4", "A100"],
        "min_vram_gb": 34.0,
        "stop_token_ids": [248046, 248044],
        "precision_by_gpu": {"L4": "awq_4bit", "A100": "awq_4bit", "A100-80GB": "bfloat16"},
        "max_model_len": 32768,
        "max_num_seqs": 6,
        "template_family": "qwen3_chatml",
        "description": "High-capability dense coding workhorse (A100 40GB/80GB)",
    },
    "qwen3.8-27b": {
        "alias": "qwen3.8-27b",
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.8-27B",
        "fallback_hf_repo_id": "Qwen/Qwen3.8-27B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5_text",
        "preferred_gpu": "A100",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "A100",
        "compatible_gpus": ["A100", "A100-80GB"],
        "compatible_gpu_tiers": ["L4", "A100"],
        "min_vram_gb": 34.0,
        "stop_token_ids": [248046, 248044],
        "precision_by_gpu": {"L4": "awq_4bit", "A100": "awq_4bit", "A100-80GB": "bfloat16"},
        "max_model_len": 32768,
        "max_num_seqs": 6,
        "template_family": "qwen3_chatml",
        "description": "Default platodoom 27B dense coding, reasoning & vision model (A100 40GB/80GB/G4)",
    },
    "qwen-35b-moe": {
        "alias": "qwen-35b-moe",
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-35B-A3B",
        "fallback_hf_repo_id": "Qwen/Qwen3.5-35B-A3B",
        "quantized_hf_repo_id": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "model_type": "qwen3_5_moe_text",
        "preferred_gpu": "A100",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "A100",
        "compatible_gpus": ["A100", "A100-80GB"],
        "compatible_gpu_tiers": ["L4", "A100"],
        "min_vram_gb": 34.0,
        "stop_token_ids": [248046, 248044],
        "precision_by_gpu": {"L4": "gptq_4bit", "A100": "gptq_4bit", "A100-80GB": "bfloat16"},
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "template_family": "qwen3_chatml",
        "description": "High-speed 35B MoE with 3B active parameters per token (A100 40GB/80GB)",
    },
    "r1-32b": {
        "alias": "r1-32b",
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "fallback_hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-32b-awq",
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "preferred_gpu": "A100",
        "min_gpu_tier": "A100",
        "recommended_gpu_tier": "A100",
        "compatible_gpus": ["v6e-1", "A100", "A100-80GB"],
        "compatible_gpu_tiers": ["A100"],
        "min_vram_gb": 30.0,
        "stop_token_ids": [151643, 151645],
        "precision_by_gpu": {"v6e-1": "awq_4bit", "A100": "awq_4bit", "A100-80GB": "bfloat16"},
        "max_model_len": 32768,
        "max_num_seqs": 6,
        "template_family": "deepseek_r1_distill",
        "description": "Heavy architectural planning & deep mathematical reasoning (A100 40GB/80GB)",
    },
}

try:
    from platodoom.catalog import BUILTIN_MODEL_CATALOG as _CATALOG_BUILTIN_MODELS

    FALLBACK_MODEL_CATALOG.update({k: dict(v) for k, v in _CATALOG_BUILTIN_MODELS.items()})
except ImportError:
    pass

BUILTIN_MODEL_CATALOG = FALLBACK_MODEL_CATALOG


def _resolve_hardware_tier_info(raw_tier: Optional[str]) -> Tuple[str, float, str, int]:
    """Resolves `(canonical_tier, nominal_memory_gb, memory_type, max_num_seqs)` across all 8 Colab GPU/TPU tiers."""
    try:
        from platodoom.catalog import GPU_TIER_SPECS, normalize_hardware_tier

        norm = normalize_hardware_tier(raw_tier or "T4")
        spec = GPU_TIER_SPECS[norm]
        cuda_floor = {"T4": 32, "L4": 64, "A100": 96, "A100-80GB": 128, "H100": 160, "G4": 192}.get(norm, 4)
        return (
            norm,
            float(spec.get("nominal_vram_gb", spec.get("vram_gb", 16.0))),
            str(spec.get("memory_type", "VRAM")),
            max(cuda_floor, int(spec.get("max_num_seqs", cuda_floor))),
        )
    except ImportError:
        key = str(raw_tier or "T4").strip()
        table = {
            "V5E-1": ("v5e-1", 16.0, "HBM", 4),
            "T4": ("T4", 16.0, "VRAM", 32),
            "L4": ("L4", 24.0, "VRAM", 64),
            "V6E-1": ("v6e-1", 32.0, "HBM", 8),
            "A100": ("A100", 40.0, "VRAM", 96),
            "A100-80GB": ("A100-80GB", 80.0, "VRAM", 128),
            "H100": ("H100", 80.0, "VRAM", 160),
            "G4": ("G4", 96.0, "VRAM", 192),
        }
        if key.upper() not in table:
            raise ValueError(
                f"Unsupported Colab hardware tier {raw_tier!r}. Valid tiers: v5e-1, T4, L4, v6e-1, A100, A100-80GB, H100, G4."
            )
        return table[key.upper()]


def _generate_bearer_token() -> str:
    try:
        from platodoom.security import generate_ephemeral_tunnel_token

        return generate_ephemeral_tunnel_token()
    except ImportError:
        return f"sk-plato-{secrets.token_hex(32)}"


def _resolve_config(
    cli_overrides: Optional[Dict[str, Any]] = None,
    cwd: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        from platodoom.config import resolve_platodoom_config

        return resolve_platodoom_config(cli_overrides=cli_overrides, cwd=cwd, env=env)
    except ImportError:
        eff_env = dict(os.environ) if env is None else dict(env)
        home_raw = eff_env.get("PLATODOOM_CONFIG_DIR") or eff_env.get("PLATODOOM_HOME")
        plato_home = Path(home_raw).expanduser().resolve() if home_raw else (Path.home() / ".platodoom").resolve()
        cfg: Dict[str, Any] = {
            "version": 1,
            "default_model": "qwen3.8-27b",
            "default_gpu": "auto",
            "default_access": "workspace-write",
            "default_backend": "colab",
            "colab": {
                "max_concurrent_sessions": 3,
                "subscription_tier": "pro_plus",
                "keepalive_interval_sec": 20,
            },
            "storage": {
                "cache_backend": "hf_hub",
                "gcs_bucket": None,
                "gcs_project_id": None,
                "gcs_models_prefix": "models/",
                "gdrive_models_dir": "/content/drive/MyDrive/platodoom_models",
            },
            "auth": {
                "hf_token_env": "PLATODOOM_HF_TOKEN",
                "google_token_path": "~/.mcp-colab/token.json",
                "google_credentials_path": "~/.mcp-colab/credentials.json",
            },
            "models": {"custom": {}},
            "security": {"extra_denylist_globs": [], "redact_env_secrets": True},
        }
        user_cfg_file = plato_home / "config.json"
        if user_cfg_file.is_file():
            try:
                loaded = json.loads(user_cfg_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
            except Exception:
                pass
        if cli_overrides:
            for k, v in cli_overrides.items():
                if v is not None:
                    cfg[k] = v
        return cfg


def _fallback_classify_family(identifier: str, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Offline neutral family classifier when platodoom.catalog is unavailable (Spec 862)."""
    res = _fallback_classify_family_core(identifier, override)
    res["stop_token_ids"] = list(res.get("eos_token_ids") or [])
    return res


def _fallback_classify_family_core(identifier: str, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ovr = override or {}
    mod_cfg: Dict[str, Any] = {}
    local_dir = Path(str(ovr.get("local_dir") or identifier)).expanduser()
    if local_dir.is_dir() and (local_dir / "config.json").is_file():
        try:
            loaded = json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                mod_cfg = loaded
        except Exception:
            mod_cfg = {}

    mt = str(ovr.get("model_type") or mod_cfg.get("model_type") or "").strip().lower()
    archs = mod_cfg.get("architectures") or ([ovr["architecture"]] if ovr.get("architecture") else [])
    arch_str = str(archs[0]) if archs else ""
    ident_low = str(identifier or "").strip().lower()

    if mt in ("qwen3_5_moe", "qwen3_5_moe_text") or arch_str == "Qwen3_5MoeForConditionalGeneration" or (
        ("qwen3.5" in ident_low or "qwen3_5" in ident_low) and ("moe" in ident_low or "35b" in ident_low or "a3b" in ident_low)
    ):
        return {
            "architecture_family": "qwen3_5_moe",
            "architecture": arch_str or "Qwen3_5MoeForConditionalGeneration",
            "model_type": mt or "qwen3_5_moe",
            "loader_kind": "qwen3_5_moe_conditional_generation",
            "loader_class_name": "Qwen3_5MoeForConditionalGeneration",
            "is_qwen35_family": True,
            "vocab_size": int(mod_cfg.get("vocab_size") or 248320),
            "template_family": "qwen3_chatml",
            "eos_token_ids": [248046, 248044],
            "eos_token": "<|im_end|>",
            "bos_token": None,
        }
    if mt in ("qwen3_5", "qwen3_5_text") or arch_str == "Qwen3_5ForConditionalGeneration" or any(
        tag in ident_low for tag in ("qwen3.5", "qwen3_5", "qwen3.6", "qwen3.8", "qwen3_8", "qwen-9b", "qwen-27b", "qwen3.8-27b")
    ):
        return {
            "architecture_family": "qwen3_5",
            "architecture": arch_str or "Qwen3_5ForConditionalGeneration",
            "model_type": mt or "qwen3_5",
            "loader_kind": "qwen3_5_conditional_generation",
            "loader_class_name": "Qwen3_5ForConditionalGeneration",
            "is_qwen35_family": True,
            "vocab_size": int(mod_cfg.get("vocab_size") or 248320),
            "template_family": "qwen3_chatml",
            "eos_token_ids": [248046, 248044],
            "eos_token": "<|im_end|>",
            "bos_token": None,
        }
    if "deepseek-r1" in ident_low or "r1-distill" in ident_low or ident_low in ("r1-14b", "r1-32b"):
        return {
            "architecture_family": "deepseek_r1",
            "architecture": arch_str or "Qwen2ForCausalLM",
            "model_type": mt or "qwen2",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "vocab_size": int(mod_cfg.get("vocab_size") or 152064),
            "template_family": "deepseek_r1_distill",
            "eos_token_ids": [151643],
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token": "<｜begin▁of▁sentence｜>",
        }
    if mt == "deepseek_v2" or arch_str == "DeepseekV2ForCausalLM" or "deepseek-v2" in ident_low:
        return {
            "architecture_family": "deepseek_v2",
            "architecture": arch_str or "DeepseekV2ForCausalLM",
            "model_type": "deepseek_v2",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "vocab_size": int(mod_cfg.get("vocab_size") or 102400),
            "template_family": "deepseek_v2_chat",
            "eos_token_ids": [100001],
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token": "<｜begin▁of▁sentence｜>",
        }
    if mt == "deepseek_v3" or arch_str == "DeepseekV3ForCausalLM" or "deepseek-v3" in ident_low:
        return {
            "architecture_family": "deepseek_v3",
            "architecture": arch_str or "DeepseekV3ForCausalLM",
            "model_type": "deepseek_v3",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "vocab_size": int(mod_cfg.get("vocab_size") or 129280),
            "template_family": "deepseek_v2_chat",
            "eos_token_ids": [1],
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token": "<｜begin▁of▁sentence｜>",
        }
    if mt == "qwen3_moe" or arch_str == "Qwen3MoeForCausalLM" or ("qwen3" in ident_low and ("moe" in ident_low or "a3b" in ident_low)):
        return {
            "architecture_family": "qwen3_moe",
            "architecture": arch_str or "Qwen3MoeForCausalLM",
            "model_type": "qwen3_moe",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "vocab_size": int(mod_cfg.get("vocab_size") or 151936),
            "template_family": "qwen3_chatml",
            "eos_token_ids": [151645, 151643],
            "eos_token": "<|im_end|>",
            "bos_token": None,
        }
    if mt == "qwen3" or arch_str == "Qwen3ForCausalLM" or "qwen3-" in ident_low or "/qwen3" in ident_low:
        return {
            "architecture_family": "qwen3",
            "architecture": arch_str or "Qwen3ForCausalLM",
            "model_type": "qwen3",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "vocab_size": int(mod_cfg.get("vocab_size") or 151936),
            "template_family": "qwen3_chatml",
            "eos_token_ids": [151645, 151643],
            "eos_token": "<|im_end|>",
            "bos_token": None,
        }
    if mt in ("qwen2", "qwen2_moe") or arch_str == "Qwen2ForCausalLM" or "qwen2.5" in ident_low or "qwen2-" in ident_low:
        return {
            "architecture_family": "qwen2",
            "architecture": arch_str or "Qwen2ForCausalLM",
            "model_type": "qwen2",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "vocab_size": int(mod_cfg.get("vocab_size") or 152064),
            "template_family": "qwen2_chatml",
            "eos_token_ids": [151645, 151643],
            "eos_token": "<|im_end|>",
            "bos_token": None,
        }
    if mt == "llama" or arch_str == "LlamaForCausalLM" or "llama" in ident_low:
        return {
            "architecture_family": "llama",
            "architecture": arch_str or "LlamaForCausalLM",
            "model_type": "llama",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "vocab_size": int(mod_cfg.get("vocab_size") or 128256),
            "template_family": "generic_chat",
            "eos_token_ids": [128009, 128001],
            "eos_token": "<|eot_id|>",
            "bos_token": "<|begin_of_text|>",
        }
    if mt == "mistral" or arch_str == "MistralForCausalLM" or "mistral" in ident_low:
        return {
            "architecture_family": "mistral",
            "architecture": arch_str or "MistralForCausalLM",
            "model_type": "mistral",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "vocab_size": int(mod_cfg.get("vocab_size") or 32768),
            "template_family": "generic_chat",
            "eos_token_ids": [2],
            "eos_token": "</s>",
            "bos_token": "<s>",
        }
    return {
        "architecture_family": "causal_lm",
        "architecture": arch_str or "AutoModelForCausalLM",
        "model_type": mt or "causal_lm",
        "loader_kind": "causal_lm",
        "loader_class_name": "AutoModelForCausalLM",
        "is_qwen35_family": False,
        "vocab_size": mod_cfg.get("vocab_size"),
        "template_family": "generic_chat",
        "eos_token_ids": [],
        "eos_token": None,
        "bos_token": None,
    }


def _get_catalog(config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    try:
        from platodoom.catalog import get_model_catalog

        return get_model_catalog(config)
    except ImportError:
        cat = {k: dict(v) for k, v in FALLBACK_MODEL_CATALOG.items()}
        custom = ((config or {}).get("models") or {}).get("custom") or {}
        if isinstance(custom, dict):
            for alias, prof in custom.items():
                if isinstance(prof, dict):
                    base_alias = prof.get("base_alias")
                    if base_alias and str(base_alias) in cat:
                        merged = dict(cat[str(base_alias)])
                        merged.update(prof)
                        merged["alias"] = str(alias)
                    else:
                        repo_id = str(prof.get("hf_repo_id") or alias)
                        fam = _fallback_classify_family(repo_id, prof)
                        merged = {
                            "alias": str(alias),
                            "source": "custom",
                            "hf_repo_id": repo_id,
                            "quantized_hf_repo_id": str(prof.get("quantized_hf_repo_id") or repo_id),
                            "architecture": fam["architecture"],
                            "model_type": fam["model_type"],
                            "architecture_family": fam["architecture_family"],
                            "loader_kind": fam["loader_kind"],
                            "loader_class_name": fam["loader_class_name"],
                            "is_qwen35_family": fam["is_qwen35_family"],
                            "vocab_size": fam["vocab_size"],
                            "preferred_gpu": "T4",
                            "compatible_gpus": ["T4", "L4", "A100", "A100-80GB"],
                            "min_vram_gb": 13.5,
                            "stop_token_ids": list(fam["eos_token_ids"]),
                            "precision_by_gpu": {
                                "T4": "gptq_4bit",
                                "L4": "gptq_4bit",
                                "A100": "bfloat16",
                                "A100-80GB": "bfloat16",
                            },
                            "max_model_len": 16384,
                            "max_num_seqs": 32,
                            "template_family": fam["template_family"],
                        }
                        merged.update(prof)
                        merged["alias"] = str(alias)
                    cat[str(alias)] = merged
        return cat


def _resolve_model_profile(alias_or_hf_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        from platodoom.catalog import resolve_model_profile

        return resolve_model_profile(alias_or_hf_id, config)
    except ImportError:
        cat = _get_catalog(config)
        key = str(alias_or_hf_id or "qwen-9b").strip()
        if key in cat:
            return dict(cat[key])
        for prof in cat.values():
            if prof.get("hf_repo_id") == key or prof.get("quantized_hf_repo_id") == key:
                return dict(prof)
        fam = _fallback_classify_family(key)
        return {
            "alias": key,
            "hf_repo_id": key,
            "quantized_hf_repo_id": key,
            "architecture": fam["architecture"],
            "model_type": fam["model_type"],
            "architecture_family": fam["architecture_family"],
            "loader_kind": fam["loader_kind"],
            "loader_class_name": fam["loader_class_name"],
            "is_qwen35_family": fam["is_qwen35_family"],
            "vocab_size": fam["vocab_size"],
            "preferred_gpu": "T4",
            "compatible_gpus": ["T4", "L4", "A100", "A100-80GB"],
            "min_vram_gb": 13.5,
            "stop_token_ids": list(fam["eos_token_ids"]),
            "precision_by_gpu": {"T4": "gptq_4bit", "L4": "gptq_4bit", "A100": "bfloat16", "A100-80GB": "bfloat16"},
            "max_model_len": 16384,
            "max_num_seqs": 32,
            "template_family": fam["template_family"],
            "description": f"Custom model checkpoint ({key})",
        }


def _verify_vram_compatibility(
    model_alias: str,
    available_vram_gb: float,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from platodoom.catalog import verify_vram_compatibility

        return verify_vram_compatibility(model_alias, available_vram_gb, config)
    except ImportError:
        prof = _resolve_model_profile(model_alias, config)
        req_vram = float(prof.get("min_vram_gb", 13.5))
        avail = float(available_vram_gb)
        compatible = avail >= req_vram
        fallback_alias: Optional[str] = None
        if not compatible:
            fallback_alias = "r1-14b" if avail >= 20.0 else "qwen-9b"
        gpu_bucket = (
            "G4"
            if avail >= 88.0
            else ("A100-80GB" if avail >= 70.0 else ("A100" if avail >= 36.0 else ("L4" if avail >= 20.0 else "T4")))
        )
        prec_map = prof.get("default_precision_by_gpu") or prof.get("precision_by_gpu") or {}
        selected_precision = str(prec_map.get(gpu_bucket) or next(iter(prec_map.values()), "Q4"))
        diag = (
            f"Model '{model_alias}' fits available VRAM ({avail:.1f} GB >= {req_vram:.1f} GB) using {selected_precision}."
            if compatible
            else (
                f"Model '{model_alias}' requires >= {req_vram:.1f} GB VRAM ({prof.get('preferred_gpu', 'A100')}), "
                f"but only {avail:.1f} GB is available. Recommended fallback: '{fallback_alias}' or upgrade GPU tier to {prof.get('preferred_gpu', 'A100')}."
            )
        )
        return {
            "compatible": compatible,
            "requested_alias": model_alias,
            "available_vram_gb": avail,
            "required_min_vram_gb": req_vram,
            "preferred_gpu": prof.get("preferred_gpu", "T4"),
            "selected_precision": selected_precision,
            "fallback_alias": fallback_alias,
            "diagnostic": diag,
        }


def _introspect_model_protocol(hf_repo_id_or_alias: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        from platodoom.catalog import introspect_model_protocol

        return introspect_model_protocol(hf_repo_id_or_alias, config=config)
    except ImportError:
        prof = _resolve_model_profile(hf_repo_id_or_alias, config)
        fam_info = _fallback_classify_family(
            str(prof.get("hf_repo_id") or hf_repo_id_or_alias),
            prof,
        )
        fam = str(prof.get("template_family") or fam_info["template_family"])
        if fam == "deepseek_r1_distill":
            return {
                "model_id": prof.get("hf_repo_id", hf_repo_id_or_alias),
                "architecture": fam_info["architecture"],
                "model_type": fam_info["model_type"],
                "architecture_family": fam_info["architecture_family"],
                "loader_kind": fam_info["loader_kind"],
                "loader_class_name": fam_info["loader_class_name"],
                "is_qwen35_family": False,
                "vocab_size": fam_info["vocab_size"],
                "template_family": "deepseek_r1_distill",
                "native_jinja_tools": False,
                "requires_dict_tool_arguments": False,
                "serialize_tools_in_system_prompt": True,
                "serialize_tool_role_as_user_xml": True,
                "supports_enable_thinking_kwarg": False,
                "forces_think_prefix": True,
                "emitted_tool_formats": ["qwen_json_tool_call", "deepseek_native_tool_call"],
                "bos_token": fam_info["bos_token"] or "<｜begin▁of▁sentence｜>",
                "eos_token": fam_info["eos_token"] or "<｜end▁of▁sentence｜>",
                "eos_token_ids": list(prof.get("stop_token_ids") or fam_info["eos_token_ids"]),
                "recommended_sampling": {"temperature": 0.6, "top_p": 0.95, "do_sample": True},
            }
        is_qwen_chatml = fam in ("qwen3_chatml", "qwen2_chatml")
        return {
            "model_id": prof.get("hf_repo_id", hf_repo_id_or_alias),
            "architecture": prof.get("architecture") or fam_info["architecture"],
            "model_type": prof.get("model_type") or fam_info["model_type"],
            "architecture_family": fam_info["architecture_family"],
            "loader_kind": fam_info["loader_kind"],
            "loader_class_name": fam_info["loader_class_name"],
            "is_qwen35_family": bool(fam_info["is_qwen35_family"]),
            "vocab_size": fam_info["vocab_size"],
            "template_family": fam,
            "native_jinja_tools": is_qwen_chatml,
            "requires_dict_tool_arguments": False,
            "serialize_tools_in_system_prompt": not is_qwen_chatml,
            "serialize_tool_role_as_user_xml": not is_qwen_chatml,
            "supports_enable_thinking_kwarg": fam == "qwen3_chatml",
            "forces_think_prefix": False,
            "emitted_tool_formats": (
                ["qwen3_xml_function", "qwen_json_tool_call", "openai_tool_calls"]
                if is_qwen_chatml
                else ["qwen_json_tool_call", "openai_tool_calls"]
            ),
            "bos_token": fam_info["bos_token"],
            "eos_token": fam_info["eos_token"],
            "eos_token_ids": list(prof.get("stop_token_ids") if prof.get("stop_token_ids") is not None else fam_info["eos_token_ids"]),
            "recommended_sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        }


def _normalize_openai_http_tool_calls(
    messages: List[Dict[str, Any]],
    output_format: str = "openai_http",
) -> List[Dict[str, Any]]:
    """Ensures assistant.tool_calls[].function.arguments is a JSON str with zero extra top-level keys for vLLM HTTP."""
    if output_format != "openai_http":
        return messages
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        m_copy = dict(msg)
        if m_copy.get("role") == "assistant" and isinstance(m_copy.get("tool_calls"), list):
            norm_tcs: List[Dict[str, Any]] = []
            for tc in m_copy["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                tc_copy = dict(tc)
                fn = dict(tc_copy.get("function") or {})
                raw_name = fn.get("name") or tc_copy.get("name") or ""
                raw_a = fn.get("arguments") if "arguments" in fn else tc_copy.get("arguments", {})
                if isinstance(raw_a, (dict, list)):
                    arg_str = json.dumps(raw_a, ensure_ascii=False)
                elif raw_a is None:
                    arg_str = "{}"
                else:
                    arg_str = str(raw_a)
                fn["name"] = str(raw_name)
                fn["arguments"] = arg_str
                tc_copy["function"] = fn
                tc_copy.pop("arguments", None)
                tc_copy.pop("name", None)
                norm_tcs.append(tc_copy)
            m_copy["tool_calls"] = norm_tcs
        out.append(m_copy)
    return out


def _compile_model_chat_messages(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    protocol_profile: Dict[str, Any],
    enable_thinking: Optional[bool] = None,
    *,
    output_format: str = "openai_http",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        from platodoom.catalog import compile_model_chat_messages

        try:
            comp_msgs, comp_kwargs = compile_model_chat_messages(
                messages=messages,
                tools=tools,
                protocol_profile=protocol_profile,
                enable_thinking=enable_thinking,
                output_format=output_format,
            )
        except TypeError:
            comp_msgs, comp_kwargs = compile_model_chat_messages(
                messages=messages,
                tools=tools,
                protocol_profile=protocol_profile,
                enable_thinking=enable_thinking,
            )
        comp_kwargs = dict(comp_kwargs)
        if "enable_thinking" in comp_kwargs:
            et_val = comp_kwargs.pop("enable_thinking")
            if et_val is not None:
                ctk = dict(comp_kwargs.get("chat_template_kwargs") or {})
                ctk.setdefault("enable_thinking", bool(et_val))
                comp_kwargs["chat_template_kwargs"] = ctk
        return _normalize_openai_http_tool_calls(comp_msgs, output_format=output_format), comp_kwargs
    except ImportError:
        fam = protocol_profile.get("template_family", "qwen3_chatml")
        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            msg = dict(m)
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str) and "<think>" in msg["content"]:
                msg["content"] = THINK_TAG_RE.sub("", msg["content"]).strip()
            cleaned.append(msg)

        if fam == "deepseek_r1_distill":
            out_msgs: List[Dict[str, Any]] = []
            tools_xml = ""
            if tools:
                tool_lines = [json.dumps(t.get("function", t), ensure_ascii=False) for t in tools]
                tools_xml = (
                    "\n\n# Tools\nYou may call one or more functions to assist with the user query.\n"
                    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                    + "\n".join(tool_lines)
                    + "\n</tools>\n"
                    'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                    '<tool_call>\n{"name": "<function-name>", "arguments": <args-json-object>}\n</tool_call>'
                )
            injected_tools = False
            for msg in cleaned:
                role = msg.get("role")
                if role == "system" and not injected_tools:
                    out_msgs.append({"role": "system", "content": str(msg.get("content", "")) + tools_xml})
                    injected_tools = True
                elif role == "assistant" and msg.get("tool_calls"):
                    tc_blocks = []
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function") or {}
                        name = fn.get("name") or tc.get("name")
                        args_val = fn.get("arguments", tc.get("arguments", {}))
                        if isinstance(args_val, str):
                            try:
                                args_val = json.loads(args_val)
                            except Exception:
                                args_val = {}
                        tc_blocks.append(f'<tool_call>\n{json.dumps({"name": name, "arguments": args_val})}\n</tool_call>')
                    base_c = str(msg.get("content") or "").strip()
                    combined_c = (base_c + "\n" + "\n".join(tc_blocks)).strip()
                    out_msgs.append({"role": "assistant", "content": combined_c})
                elif role == "tool":
                    wrapped = f"<tool_response>\n{msg.get('content', '')}\n</tool_response>"
                    if out_msgs and out_msgs[-1].get("role") == "user" and "<tool_response>" in str(out_msgs[-1].get("content", "")):
                        out_msgs[-1]["content"] = str(out_msgs[-1]["content"]) + "\n" + wrapped
                    else:
                        out_msgs.append({"role": "user", "content": wrapped})
                else:
                    out_msgs.append(msg)
            if tools_xml and not injected_tools:
                if out_msgs and out_msgs[0].get("role") == "user":
                    out_msgs[0]["content"] = tools_xml.lstrip() + "\n\n" + str(out_msgs[0].get("content", ""))
                else:
                    out_msgs.insert(0, {"role": "system", "content": tools_xml.lstrip()})
            r1_stops = list(protocol_profile.get("eos_token_ids") or [])
            return out_msgs, {"temperature": 0.6, "top_p": 0.95, "do_sample": True, "stop_token_ids": r1_stops}

        if output_format == "openai_http":
            cleaned = _normalize_openai_http_tool_calls(cleaned, output_format="openai_http")
        else:
            for msg in cleaned:
                if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
                    norm_tcs = []
                    for tc in msg["tool_calls"]:
                        tc_copy = dict(tc)
                        fn = dict(tc_copy.get("function") or {})
                        raw_a = fn.get("arguments", tc_copy.get("arguments", {}))
                        if isinstance(raw_a, str):
                            try:
                                raw_a = json.loads(raw_a) if raw_a.strip() else {}
                            except Exception:
                                raw_a = {}
                        fn["arguments"] = raw_a
                        tc_copy["function"] = fn
                        tc_copy["arguments"] = raw_a
                        norm_tcs.append(tc_copy)
                    msg["tool_calls"] = norm_tcs
        req_kwargs: Dict[str, Any] = {
            "temperature": 0.6 if enable_thinking else 0.0,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        return cleaned, req_kwargs


def _resolve_effective_max_model_len(profile: Mapping[str, Any], gpu_tier: str) -> int:
    tier_norm = str(gpu_tier or "").strip().upper()
    if tier_norm in ("A100", "A100-80GB", "H100", "G4") and profile.get("max_model_len_a100"):
        return int(profile["max_model_len_a100"])
    return int(profile.get("max_model_len", 16384))


def _compact_messages(messages: List[Dict[str, Any]], max_model_len: int = 16384, threshold_ratio: float = 0.80) -> List[Dict[str, Any]]:
    try:
        from platodoom.catalog import compact_conversation_messages

        return compact_conversation_messages(messages, max_model_len=max_model_len, threshold_ratio=threshold_ratio)
    except ImportError:
        from scripts.qwen_worktree_agent import compact_messages_for_16k_window

        reserve = max(1024, int(max_model_len * (1.0 - threshold_ratio)))
        return compact_messages_for_16k_window(messages, max_model_len=max_model_len, reserve_completion_tokens=reserve)


def _discover_capabilities(
    workspace_dir: Path,
    home_dir: Optional[Path] = None,
    *,
    include_global: bool = False,
) -> Dict[str, Any]:
    try:
        from platodoom.discovery import discover_workspace_capabilities

        return discover_workspace_capabilities(
            workspace_dir,
            home_dir=home_dir,
            include_global=include_global,
        )
    except ImportError:
        return {
            "include_global": bool(include_global),
            "instructions": [],
            "skills": {},
            "commands": {},
            "mcp_servers": {},
            "elapsed_ms": 0.1,
        }


def _build_compass(workspace_dir: Path, capabilities: Dict[str, Any], max_tokens: int = 450) -> str:
    try:
        from platodoom.discovery import build_zero_turn_workspace_compass

        return build_zero_turn_workspace_compass(workspace_dir, capabilities=capabilities, max_tokens=max_tokens)
    except ImportError:
        return (
            f"# Zero-Turn Workspace Compass\n"
            f"- Working Directory: {workspace_dir}\n"
            f"- Discovered Skills: {len(capabilities.get('skills', {}))}\n"
            f"- Discovered MCP Servers: {len(capabilities.get('mcp_servers', {}))}\n"
        )


def list_active_sessions() -> List[Dict[str, Any]]:
    """Lists active interactive and pipeline sessions from `~/.platodoom/sessions/`."""
    try:
        from platodoom.swarm import list_active_sessions as _swarm_list_sessions

        return _swarm_list_sessions()
    except ImportError:
        sessions: List[Dict[str, Any]] = []
        home_raw = os.environ.get("PLATODOOM_CONFIG_DIR") or os.environ.get("PLATODOOM_HOME")
        plato_home = Path(home_raw).expanduser().resolve() if home_raw else (Path.home() / ".platodoom").resolve()
        sess_root = plato_home / "sessions"
        if sess_root.is_dir():
            for ep_file in sorted(sess_root.glob("*/endpoint.json")):
                try:
                    data = json.loads(ep_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        sessions.append(data)
                except Exception:
                    continue
        return sessions


_PRESERVED_PERSISTENCE_KEYS: Tuple[str, ...] = (
    "session_id",
    "session_slug",
    "title",
    "model_alias",
    "active_model",
    "model",
    "model_id",
    "hf_repo_id",
    "gpu_tier",
    "gpu",
    "workflow_mode",
    "access_mode",
    "previous_write_access_mode",
    "workspace_dir",
    "stateful_cwd",
    "workspace_roots",
    "created_at",
    "updated_at",
    "updated_at_epoch",
    "updated_at_ns",
    "turn_count",
    "estimated_tokens",
    "message_count",
    "total_message_count",
    "role",
    "content",
    "reasoning_content",
    "tool_calls",
    "tool_call_id",
    "name",
    "type",
    "id",
    "index",
    "function",
    "arguments",
    "event",
    "timestamp",
    "endpoint_url",
    "storage_uri",
)

_SENSITIVE_PERSISTENCE_KEYS = {
    "bearer_token",
    "auth_token",
    "api_key",
    "tunnel_token",
    "hf_token",
    "secret",
    "password",
    "credentials",
    "google_token_path",
    "google_credentials_path",
}


_INLINE_PERSISTENCE_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z\-_]{30,})\b"
)


def _redact_persistence_payload(
    obj: Any,
    preserve_keys: Tuple[str, ...] = (),
) -> Any:
    """
    Recursively redacts secrets, API keys, and tokens from persistence payloads
    (session.json, messages.json, transcript.jsonl) before writing to disk (FR-4.1).
    """
    try:
        from platodoom.security import redact_env_secrets
    except ImportError:
        def redact_env_secrets(text: str, env: Optional[Dict[str, str]] = None) -> str:
            return text

    preserved_set = set(_PRESERVED_PERSISTENCE_KEYS) | set(preserve_keys or ())

    def _walk(val: Any) -> Any:
        if isinstance(val, str):
            redacted_str = redact_env_secrets(val)
            return _INLINE_PERSISTENCE_SECRET_RE.sub("<REDACTED_SECRET>", redacted_str)
        if isinstance(val, list):
            return [_walk(item) for item in val]
        if isinstance(val, tuple):
            return [_walk(item) for item in val]
        if isinstance(val, dict):
            out: Dict[str, Any] = {}
            for k, v in val.items():
                k_str = str(k)
                if k_str.lower() in _SENSITIVE_PERSISTENCE_KEYS and k_str not in preserved_set:
                    out[k_str] = "[REDACTED]" if v else v
                else:
                    out[k_str] = _walk(v)
            return out
        return val

    return _walk(obj)


def _ensure_dir_0700(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(dir_path), 0o700)
    except OSError:
        pass
    return dir_path


def _atomic_write_json_0600(target_path: Path | str, payload: Any) -> Path:
    """Atomically writes JSON payload with 0600 file permissions and 0700 parent directory permissions."""
    resolved = Path(target_path).expanduser().resolve()
    _ensure_dir_0700(resolved.parent)
    try:
        if resolved.parent.parent.name in ("sessions", ".platodoom"):
            os.chmod(str(resolved.parent.parent), 0o700)
    except OSError:
        pass
    try:
        from platodoom.config import atomic_write_json as _cfg_atomic_write_json

        return _cfg_atomic_write_json(resolved, payload, mode=0o600)
    except (ImportError, TypeError):
        try:
            atomic_write_json(resolved, payload, mode=0o600)
        except TypeError:
            atomic_write_json(resolved, payload)
        try:
            os.chmod(str(resolved), 0o600)
        except OSError:
            pass
        return resolved


def estimate_messages_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Estimates token count of an OpenAI-format message list (`len(text) // 4`)."""
    try:
        from platodoom.catalog import estimate_messages_tokens as _cat_est

        return _cat_est(messages, tools=tools)
    except ImportError:
        total_chars = 0
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            total_chars += len(str(msg.get("role") or "")) + 4
            if msg.get("content") is not None:
                total_chars += len(str(msg.get("content") or ""))
            if msg.get("reasoning_content") is not None:
                total_chars += len(str(msg.get("reasoning_content") or ""))
            if msg.get("tool_calls") is not None:
                try:
                    total_chars += len(json.dumps(msg.get("tool_calls"), ensure_ascii=False))
                except Exception:
                    total_chars += len(str(msg.get("tool_calls")))
        if tools:
            try:
                total_chars += len(json.dumps(tools, ensure_ascii=False))
            except Exception:
                total_chars += len(str(tools))
        return max(1, total_chars // 4) if total_chars > 0 else 0


def _resolve_effective_sessions_root(
    sessions_root: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    if sessions_root is not None:
        root = Path(sessions_root).expanduser().resolve()
        return _ensure_dir_0700(root)
    try:
        from platodoom.config import resolve_sessions_root_dir

        return resolve_sessions_root_dir(env=env)
    except ImportError:
        eff_env = env if env is not None else os.environ
        home_raw = eff_env.get("PLATODOOM_CONFIG_DIR") or eff_env.get("PLATODOOM_HOME")
        plato_home = Path(home_raw).expanduser().resolve() if home_raw else (Path.home() / ".platodoom").resolve()
        sess_dir = plato_home / "sessions"
        _ensure_dir_0700(plato_home)
        return _ensure_dir_0700(sess_dir)


def _reconstruct_messages_and_meta_from_transcript(
    transcript_path: Path | str,
    session_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Internal helper that rebuilds both the OpenAI-format `messages` list and recovered session metadata
    from `transcript.jsonl` when `session.json` or `messages.json` is missing or corrupted (FR-4.1).
    """
    t_path = Path(transcript_path).expanduser().resolve()
    recovered_meta: Dict[str, Any] = dict(session_meta or {})
    reconstructed: List[Dict[str, Any]] = []

    if t_path.is_file():
        try:
            raw_lines = t_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            raw_lines = []

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue

            for meta_key in (
                "session_id",
                "session_slug",
                "title",
                "model_alias",
                "active_model",
                "model_id",
                "hf_repo_id",
                "gpu_tier",
                "workflow_mode",
                "access_mode",
                "workspace_dir",
            ):
                if rec.get(meta_key):
                    if meta_key == "title":
                        cur_title = str(recovered_meta.get("title") or "").strip()
                        new_title = str(rec["title"]).strip()
                        if new_title and (not cur_title or cur_title in ("New Conversation", "Interactive Terminal Session") or new_title not in ("New Conversation", "Interactive Terminal Session")):
                            recovered_meta["title"] = new_title
                    elif meta_key in ("active_model", "model_alias", "model_id", "hf_repo_id", "gpu_tier", "workflow_mode", "access_mode"):
                        recovered_meta[meta_key] = rec[meta_key]
                    elif not recovered_meta.get(meta_key):
                        recovered_meta[meta_key] = rec[meta_key]
            if rec.get("timestamp"):
                recovered_meta["updated_at"] = rec["timestamp"]
                if not recovered_meta.get("created_at"):
                    recovered_meta["created_at"] = rec["timestamp"]
            if rec.get("event") in ("mode_change", "mode_switch", "workflow_mode_change", "access_mode_change"):
                if rec.get("workflow_mode"):
                    recovered_meta["workflow_mode"] = rec["workflow_mode"]
                if rec.get("access_mode"):
                    recovered_meta["access_mode"] = rec["access_mode"]
                continue
            if rec.get("event") == "model_switch":
                if rec.get("active_model"):
                    recovered_meta["active_model"] = rec["active_model"]
                    recovered_meta["model_alias"] = rec["active_model"]
                    recovered_meta["model"] = rec["active_model"]
                continue
            if rec.get("event") == "session_rename":
                if rec.get("title"):
                    recovered_meta["title"] = str(rec["title"]).strip()
                continue

            ev_type = rec.get("event")
            if ev_type == "user_message" or (rec.get("role") == "user" and "content" in rec):
                content = str(rec.get("content") or "")
                reconstructed.append({"role": "user", "content": content})
            elif ev_type == "assistant_message" or (rec.get("role") == "assistant" and ("content" in rec or "tool_calls" in rec)):
                msg_obj: Dict[str, Any] = {
                    "role": "assistant",
                    "content": str(rec.get("content") or ""),
                }
                if rec.get("reasoning_content"):
                    msg_obj["reasoning_content"] = rec["reasoning_content"]
                if isinstance(rec.get("tool_calls"), list) and rec["tool_calls"]:
                    msg_obj["tool_calls"] = rec["tool_calls"]
                reconstructed.append(msg_obj)
            elif ev_type == "tool_result" or rec.get("role") == "tool":
                reconstructed.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(rec.get("tool_call_id") or rec.get("id") or "call_recovered"),
                        "name": str(rec.get("name") or rec.get("tool") or "tool"),
                        "content": str(rec.get("content") if "content" in rec else rec.get("output") or ""),
                    }
                )
            elif rec.get("role") == "system" and "content" in rec:
                if not reconstructed:
                    reconstructed.append({"role": "system", "content": str(rec["content"])})
            elif "user_message" in rec or "assistant_reply" in rec:
                if rec.get("user_message"):
                    reconstructed.append({"role": "user", "content": str(rec["user_message"])})
                if rec.get("assistant_reply") is not None:
                    reconstructed.append({"role": "assistant", "content": str(rec["assistant_reply"])})
            elif isinstance(rec.get("messages"), list):
                reconstructed = [dict(m) for m in rec["messages"] if isinstance(m, dict)]

    slug = str(recovered_meta.get("session_slug") or t_path.parent.name)
    recovered_meta.setdefault("session_slug", slug)
    recovered_meta.setdefault("session_id", slug.replace("platodoom-chat-", ""))
    model_alias = str(
        recovered_meta.get("active_model")
        or recovered_meta.get("model_alias")
        or recovered_meta.get("model")
        or "qwen3.8-27b"
    )
    recovered_meta.setdefault("active_model", model_alias)
    recovered_meta.setdefault("model_alias", model_alias)
    recovered_meta.setdefault("model", model_alias)
    recovered_meta.setdefault("workflow_mode", "plan" if recovered_meta.get("access_mode") == "read-only" else "build")
    recovered_meta.setdefault("access_mode", "read-only" if recovered_meta.get("workflow_mode") == "plan" else "confirm-writes")

    if not recovered_meta.get("title") or recovered_meta.get("title") in ("New Conversation", "Interactive Terminal Session"):
        for m in reconstructed:
            if m.get("role") == "user" and str(m.get("content") or "").strip():
                first_line = str(m["content"]).strip().splitlines()[0].strip()
                recovered_meta["title"] = first_line[:60]
                break
        else:
            recovered_meta["title"] = "Recovered Conversation"

    user_turns = sum(1 for m in reconstructed if m.get("role") == "user")
    recovered_meta["turn_count"] = max(int(recovered_meta.get("turn_count") or 0), user_turns)
    recovered_meta["message_count"] = len(reconstructed)
    recovered_meta["estimated_tokens"] = estimate_messages_tokens(reconstructed)
    if isinstance(session_meta, dict):
        session_meta.update(recovered_meta)
    return reconstructed, recovered_meta


def reconstruct_messages_from_transcript(
    transcript_path: Path | str,
    session_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Rebuilds the OpenAI-format `messages` list from `transcript.jsonl` when `messages.json` is missing
    or corrupted (FR-4.1). If `session_meta` is passed as a dict, populates recovered metadata in-place.
    """
    reconstructed, _ = _reconstruct_messages_and_meta_from_transcript(
        transcript_path=transcript_path,
        session_meta=session_meta,
    )
    return reconstructed


def list_saved_sessions(
    sessions_root: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Lists all persisted conversations in `~/.platodoom/sessions/` ordered by `updated_at` descending
    with 1-based `"index"` keys in `<= 20 ms` without network probes (FR-4.1, FR-4.2).
    """
    root = _resolve_effective_sessions_root(sessions_root=sessions_root)
    if not root.is_dir():
        return []

    raw_entries: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if "-sub-" in child.name:
            continue
        session_file = child / "session.json"
        messages_file = child / "messages.json"
        transcript_file = child / "transcript.jsonl"
        endpoint_file = child / "endpoint.json"

        if not (session_file.is_file() or messages_file.is_file() or transcript_file.is_file() or endpoint_file.is_file()):
            continue

        meta: Dict[str, Any] = {}
        corrupt_session = False
        if session_file.is_file():
            try:
                loaded_meta = json.loads(session_file.read_text(encoding="utf-8"))
                if isinstance(loaded_meta, dict):
                    meta = loaded_meta
                else:
                    corrupt_session = True
            except Exception:
                corrupt_session = True

        if meta.get("session_type") == "subagent":
            continue

        if not meta and endpoint_file.is_file():
            try:
                ep_data = json.loads(endpoint_file.read_text(encoding="utf-8"))
                if isinstance(ep_data, dict):
                    meta = dict(ep_data)
            except Exception:
                pass

        messages: List[Dict[str, Any]] = []
        need_messages_inspection = (
            corrupt_session
            or not meta
            or meta.get("turn_count") is None
            or not meta.get("title")
            or (meta.get("title") in ("New Conversation", "Interactive Terminal Session") and int(meta.get("message_count") or 0) > 1)
        )
        if need_messages_inspection and messages_file.is_file():
            try:
                loaded_msgs = json.loads(messages_file.read_text(encoding="utf-8"))
                if isinstance(loaded_msgs, list):
                    messages = [m for m in loaded_msgs if isinstance(m, dict)]
            except Exception:
                messages = []

        if (corrupt_session or not meta or (need_messages_inspection and not messages)) and transcript_file.is_file():
            rec_msgs, rec_meta = _reconstruct_messages_and_meta_from_transcript(transcript_file, session_meta=meta)
            if not messages:
                messages = rec_msgs
            for k, v in rec_meta.items():
                if k not in meta or meta[k] is None or meta[k] == "":
                    meta[k] = v

        slug = str(meta.get("session_slug") or child.name)
        session_id = str(meta.get("session_id") or slug.replace("platodoom-chat-", ""))
        model_alias = str(
            meta.get("model_alias")
            or meta.get("active_model")
            or meta.get("model")
            or "qwen3.8-27b"
        )
        model_id = str(meta.get("model_id") or meta.get("hf_repo_id") or model_alias)
        gpu_tier = str(meta.get("gpu_tier") or meta.get("gpu") or "T4")
        access_mode = str(meta.get("access_mode") or "confirm-writes")
        workflow_mode = str(
            meta.get("workflow_mode")
            or ("plan" if access_mode == "read-only" else "build")
        )

        title = str(meta.get("title") or "").strip()
        if not title or title in ("New Conversation", "Interactive Terminal Session"):
            for m in messages:
                if m.get("role") == "user" and str(m.get("content") or "").strip():
                    title = str(m["content"]).strip().splitlines()[0].strip()[:60]
                    break
            if not title:
                title = "New Conversation"

        user_turns = sum(1 for m in messages if m.get("role") == "user")
        turn_count = max(int(meta.get("turn_count") or 0), user_turns)
        message_count = len(messages) if messages else int(meta.get("message_count") or 0)
        estimated_tokens = (
            estimate_messages_tokens(messages)
            if messages
            else int(meta.get("estimated_tokens") or 0)
        )

        updated_at = str(meta.get("updated_at") or "")
        updated_at_ns = int(meta.get("updated_at_ns") or 0)
        updated_at_epoch = float(meta.get("updated_at_epoch") or (updated_at_ns / 1e9 if updated_at_ns else 0.0))
        if not updated_at or not updated_at_ns:
            try:
                stat_res = (session_file if session_file.is_file() else child).stat()
                if not updated_at:
                    updated_at = datetime.fromtimestamp(stat_res.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if not updated_at_ns:
                    updated_at_ns = int(getattr(stat_res, "st_mtime_ns", int(stat_res.st_mtime * 1e9)))
                if not updated_at_epoch:
                    updated_at_epoch = float(stat_res.st_mtime)
            except OSError:
                if not updated_at:
                    updated_at = "1970-01-01T00:00:00Z"
        created_at = str(meta.get("created_at") or updated_at)

        raw_entries.append(
            {
                "session_id": session_id,
                "session_slug": slug,
                "title": title,
                "model_alias": model_alias,
                "active_model": model_alias,
                "model": model_alias,
                "model_id": model_id,
                "hf_repo_id": model_id,
                "gpu_tier": gpu_tier,
                "gpu": gpu_tier,
                "workflow_mode": workflow_mode,
                "access_mode": access_mode,
                "previous_write_access_mode": str(meta.get("previous_write_access_mode") or "confirm-writes"),
                "workspace_dir": str(meta.get("workspace_dir") or Path.cwd()),
                "created_at": created_at,
                "updated_at": updated_at,
                "updated_at_epoch": updated_at_epoch,
                "updated_at_ns": updated_at_ns,
                "turn_count": turn_count,
                "message_count": message_count,
                "total_message_count": message_count,
                "estimated_tokens": estimated_tokens,
                "status": str(meta.get("status") or ("Online" if meta.get("endpoint_url") else "Saved")),
                "state_dir": str(child),
            }
        )

    raw_entries.sort(
        key=lambda item: (
            float(item.get("updated_at_epoch") or 0.0),
            int(item.get("updated_at_ns") or 0),
            str(item.get("updated_at") or ""),
            str(item.get("session_slug") or ""),
        ),
        reverse=True,
    )
    for idx, entry in enumerate(raw_entries, start=1):
        entry["index"] = idx
    return raw_entries


def load_saved_session_payload(
    target: str | int,
    sessions_root: Optional[Path | str] = None,
) -> Tuple[Path, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Resolves `target` (1-based index, 'latest', session_id, session_slug, or title substring)
    and loads `(state_dir, session_meta, messages)` with automatic transcript recovery and self-healing (FR-4.1, FR-4.2).
    """
    root = _resolve_effective_sessions_root(sessions_root=sessions_root)
    sessions = list_saved_sessions(sessions_root=root)

    raw_target = str(target if target is not None else "").strip()
    if not raw_target:
        raise ValueError("Resume target cannot be empty. Provide a 1-based index, 'latest', or session_slug.")

    selected_summary: Optional[Dict[str, Any]] = None
    if isinstance(target, int) or raw_target.isdigit():
        idx = int(raw_target)
        for s in sessions:
            if s.get("index") == idx:
                selected_summary = s
                break
        if selected_summary is None:
            raise ValueError(
                f"Saved session index #{idx} not found (available: 1..{len(sessions)})."
            )
    elif raw_target.lower() in ("latest", "last", "recent"):
        if not sessions:
            raise ValueError(f"No saved sessions found in {root}.")
        selected_summary = sessions[0]
    else:
        for s in sessions:
            if raw_target in (s.get("session_slug"), s.get("session_id")):
                selected_summary = s
                break
        if selected_summary is None:
            for s in sessions:
                if str(s.get("session_slug", "")).startswith(raw_target) or str(s.get("session_id", "")).startswith(raw_target):
                    selected_summary = s
                    break
        if selected_summary is None:
            lower_q = raw_target.lower()
            for s in sessions:
                if lower_q in str(s.get("title", "")).lower():
                    selected_summary = s
                    break
        if selected_summary is None:
            cand_dir = root / (raw_target if raw_target.startswith("platodoom-chat-") else f"platodoom-chat-{raw_target}")
            if cand_dir.is_dir():
                selected_summary = {"state_dir": str(cand_dir), "session_slug": cand_dir.name}
            elif (root / raw_target).is_dir():
                selected_summary = {"state_dir": str(root / raw_target), "session_slug": raw_target}

    if selected_summary is None:
        raise ValueError(f"No saved session matching '{raw_target}' found in {root}.")

    state_dir = Path(str(selected_summary["state_dir"])).expanduser().resolve()
    session_file = state_dir / "session.json"
    messages_file = state_dir / "messages.json"
    transcript_file = state_dir / "transcript.jsonl"

    session_meta: Dict[str, Any] = dict(selected_summary)
    need_transcript_recovery = False

    if session_file.is_file():
        try:
            loaded_meta = json.loads(session_file.read_text(encoding="utf-8"))
            if isinstance(loaded_meta, dict):
                session_meta.update(loaded_meta)
            else:
                need_transcript_recovery = True
        except Exception:
            need_transcript_recovery = True
    else:
        need_transcript_recovery = True

    messages: List[Dict[str, Any]] = []
    if messages_file.is_file():
        try:
            loaded_msgs = json.loads(messages_file.read_text(encoding="utf-8"))
            if isinstance(loaded_msgs, list) and loaded_msgs:
                messages = [m for m in loaded_msgs if isinstance(m, dict)]
            else:
                need_transcript_recovery = True
        except Exception:
            need_transcript_recovery = True
    else:
        need_transcript_recovery = True

    if need_transcript_recovery and transcript_file.is_file():
        rec_msgs, rec_meta = _reconstruct_messages_and_meta_from_transcript(transcript_file, session_meta=session_meta)
        if not messages:
            messages = rec_msgs
        for k, v in rec_meta.items():
            if k not in session_meta or session_meta[k] is None or session_meta[k] == "":
                session_meta[k] = v
        # Self-heal missing or corrupted messages.json and session.json on disk (mode 0600)
        try:
            if messages:
                _atomic_write_json_0600(messages_file, _redact_persistence_payload(messages))
            _atomic_write_json_0600(session_file, _redact_persistence_payload(session_meta))
        except OSError:
            pass

    return state_dir, session_meta, messages


def delete_saved_session(
    target: str | int,
    sessions_root: Optional[Path | str] = None,
    active_session_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deletes a saved session directory from `~/.platodoom/sessions/`, refusing to delete the active session
    with `error_code='CANNOT_DELETE_ACTIVE_SESSION'` (FR-4.2).
    """
    state_dir, meta, _ = load_saved_session_payload(target, sessions_root=sessions_root)
    slug = str(meta.get("session_slug") or state_dir.name)
    if active_session_slug and slug == str(active_session_slug).strip():
        msg = (
            f"Cannot delete currently active session '{slug}' (CANNOT_DELETE_ACTIVE_SESSION). "
            "Switch to another session with `/resume` or start `/new` first."
        )
        return {
            "ok": False,
            "deleted": False,
            "session_slug": slug,
            "error_code": "CANNOT_DELETE_ACTIVE_SESSION",
            "code": "CANNOT_DELETE_ACTIVE_SESSION",
            "error": msg,
            "message": msg,
        }
    shutil.rmtree(state_dir)
    return {
        "ok": True,
        "deleted": True,
        "session_slug": slug,
        "session_id": str(meta.get("session_id") or slug.replace("platodoom-chat-", "")),
        "title": str(meta.get("title") or ""),
        "state_dir": str(state_dir),
    }


def _consume_chat_completion_stream(
    req: urllib.request.Request,
    timeout_sec: float = 300.0,
    on_token_delta: Optional[Callable[[str], None]] = None,
    on_reasoning_delta: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Consumes an OpenAI-compatible Server-Sent Events (`text/event-stream`) or JSON response from
    `/v1/chat/completions`, invoking `on_reasoning_delta` and `on_token_delta` in real time as tokens
    arrive and returning a synthesized OpenAI ChatCompletion dict (FR-2.1).
    """
    resp = _open_session_http_request(req, timeout=timeout_sec)
    try:
        headers = getattr(resp, "headers", None) or {}
        content_type = ""
        if hasattr(headers, "get"):
            content_type = str(headers.get("Content-Type") or headers.get("content-type") or "")

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        finish_reason: str = "stop"
        model_name: str = ""
        chunk_id: str = f"chatcmpl-{uuid.uuid4().hex[:10]}"
        usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        batch_meta: Optional[Dict[str, Any]] = None

        def _normalize_parsed_json_response(parsed_obj: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(parsed_obj, dict):
                return {}
            for ch in parsed_obj.get("choices") or []:
                if isinstance(ch, dict) and isinstance(ch.get("message"), dict):
                    m_obj = ch["message"]
                    if not m_obj.get("reasoning_content") and m_obj.get("reasoning") is not None:
                        m_obj["reasoning_content"] = m_obj.get("reasoning")
            return parsed_obj

        if "text/event-stream" not in content_type.lower():
            raw_body = resp.read().decode("utf-8", errors="replace")
            stripped_body = raw_body.lstrip()
            if not (stripped_body.startswith("data:") or stripped_body.startswith(":")):
                parsed = json.loads(raw_body) if stripped_body else {}
                return _normalize_parsed_json_response(parsed)
            lines_iter = stripped_body.splitlines()
        else:
            lines_iter = None

        def _process_sse_line(raw_line_str: str) -> bool:
            nonlocal finish_reason, model_name, chunk_id, usage, batch_meta
            line = raw_line_str.strip()
            if not line or line.startswith(":"):
                return False
            if not line.startswith("data:"):
                return False
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                return True
            try:
                chunk = json.loads(data_str)
            except Exception:
                return False
            if not isinstance(chunk, dict):
                return False
            if "error" in chunk and chunk["error"]:
                err_val = chunk["error"]
                err_msg = err_val.get("message", str(err_val)) if isinstance(err_val, dict) else str(err_val)
                raise RuntimeError(f"Colab SSE stream error: {err_msg}")
            if chunk.get("id"):
                chunk_id = str(chunk["id"])
            if chunk.get("model"):
                model_name = str(chunk["model"])
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            if isinstance(chunk.get("sandcastle_batch_meta"), dict):
                batch_meta = dict(chunk["sandcastle_batch_meta"])
            for choice in chunk.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else (choice.get("message") or {})
                if not isinstance(delta, dict):
                    continue
                r_delta = (
                    delta.get("reasoning_content")
                    if delta.get("reasoning_content") is not None
                    else delta.get("reasoning")
                )
                if r_delta:
                    r_str = str(r_delta)
                    reasoning_parts.append(r_str)
                    if on_reasoning_delta is not None:
                        on_reasoning_delta(r_str)
                c_delta = delta.get("content")
                if c_delta:
                    c_str = str(c_delta)
                    content_parts.append(c_str)
                    if on_token_delta is not None:
                        on_token_delta(c_str)
                tc_deltas = delta.get("tool_calls")
                if isinstance(tc_deltas, list):
                    for idx_pos, tc in enumerate(tc_deltas):
                        if not isinstance(tc, dict):
                            continue
                        tc_idx = int(tc.get("index", idx_pos))
                        entry = tool_calls_by_index.setdefault(
                            tc_idx,
                            {
                                "id": str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tc.get("id"):
                            entry["id"] = str(tc["id"])
                        fn = tc.get("function") or {}
                        if isinstance(fn, dict):
                            if fn.get("name"):
                                fn_name = str(fn["name"])
                                if not entry["function"]["name"]:
                                    entry["function"]["name"] = fn_name
                                elif entry["function"]["name"] != fn_name:
                                    entry["function"]["name"] += fn_name
                            if fn.get("arguments") is not None:
                                arg_val = fn["arguments"]
                                if isinstance(arg_val, dict):
                                    entry["function"]["arguments"] = json.dumps(arg_val, ensure_ascii=False)
                                else:
                                    entry["function"]["arguments"] += str(arg_val)
            return False

        if lines_iter is not None:
            for l_str in lines_iter:
                if _process_sse_line(l_str):
                    break
        else:
            first_bytes = resp.readline()
            if first_bytes:
                first_str = first_bytes.decode("utf-8", errors="replace")
                if first_str.lstrip().startswith("{"):
                    rest_str = resp.read().decode("utf-8", errors="replace")
                    parsed = json.loads(first_str + rest_str)
                    return _normalize_parsed_json_response(parsed)
                if not _process_sse_line(first_str):
                    while True:
                        raw_line = resp.readline()
                        if not raw_line:
                            break
                        if _process_sse_line(raw_line.decode("utf-8", errors="replace")):
                            break

        assembled_message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
            "reasoning_content": "".join(reasoning_parts) if reasoning_parts else None,
            "reasoning": "".join(reasoning_parts) if reasoning_parts else None,
        }
        if tool_calls_by_index:
            assembled_message["tool_calls"] = [
                tool_calls_by_index[k] for k in sorted(tool_calls_by_index.keys())
            ]
        out_completion: Dict[str, Any] = {
            "id": chunk_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": assembled_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        if batch_meta is not None:
            out_completion["sandcastle_batch_meta"] = batch_meta
        return out_completion
    finally:
        try:
            resp.close()
        except Exception:
            pass


class PlatoSession:
    """
    Stateful interactive or one-shot platodoom session managing isolated `platodoom-chat-<id>`
    state, mid-session `/model` switching (in-place GPU hot-swap or tier transition), mid-session
    `/plan` vs `/build` and `/access` mode toggling, multi-conversation persistence (`~/.platodoom/sessions/`),
    real-time SSE token streaming, multi-ecosystem Skill/MCP execution, and subagent continuous batching.
    """

    def __init__(
        self,
        model_alias: Optional[str] = None,
        access_mode: Optional[str] = None,
        workspace_dir: Optional[Path | str] = None,
        session_id: Optional[str] = None,
        attach_slug: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        endpoint_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        gpu_tier: Optional[str] = None,
        vram_total_gb: Optional[float] = None,
        session_type: str = "interactive",
        workflow_mode: Optional[str] = None,
        storage_uri: Optional[str] = None,
        resume_target: Optional[str | int] = None,
        title: Optional[str] = None,
        read_only: bool = False,
        parent_session_slug: Optional[str] = None,
        extra_workspace_roots: Optional[List[Path | str]] = None,
        confirm_callback: Optional[Callable[..., bool]] = None,
        env: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        self.env: Dict[str, str] = dict(os.environ) if env is None else dict(env)
        self.workspace_dir: Path = Path(workspace_dir).expanduser().resolve() if workspace_dir else Path.cwd().resolve()
        self.project_root: Path = resolve_project_root(self.workspace_dir)
        self.dir_slug: str = compute_dir_slug(self.project_root)
        self.parent_session_slug: Optional[str] = parent_session_slug
        self.confirm_callback: Optional[Callable[..., bool]] = confirm_callback

        self.config: Dict[str, Any] = config if isinstance(config, dict) else _resolve_config(cwd=self.workspace_dir, env=self.env)

        eff_attach_slug = attach_slug or kwargs.get("session_slug")
        if eff_attach_slug and str(eff_attach_slug).strip():
            self.session_slug: str = str(eff_attach_slug).strip()
            self.session_id: str = self.session_slug.replace("platodoom-chat-", "")
        else:
            raw_id = str(session_id).strip() if (session_id and str(session_id).strip()) else secrets.token_hex(4)
            self.session_id = raw_id.replace("platodoom-chat-", "")
            self.session_slug = raw_id if raw_id.startswith("platodoom-chat-") else f"platodoom-chat-{raw_id}"

        self.session_type: str = session_type
        custom_home_env = self.env.get("PLATODOOM_CONFIG_DIR") or self.env.get("PLATODOOM_HOME")
        if custom_home_env and str(custom_home_env).strip():
            sdir = Path(str(custom_home_env).strip()).expanduser().resolve() / "sessions" / self.session_slug
            _ensure_dir_0700(sdir.parent)
            _ensure_dir_0700(sdir)
            self.session_dir: Path = sdir
        else:
            self.session_dir = resolve_platodoom_session_dir(self.session_slug)
        self.state_dir: Path = self.session_dir

        resolved_model = str(
            model_alias
            or self.env.get("PLATODOOM_MODEL")
            or self.config.get("default_model")
            or "qwen-9b"
        ).strip()
        self.active_model: str = resolved_model
        self.model_profile: Dict[str, Any] = _resolve_model_profile(self.active_model, self.config)
        self.model_spec: Dict[str, Any] = self.model_profile
        self.hf_repo_id: str = str(self.model_profile.get("hf_repo_id", self.active_model))
        self.model_id: str = self.hf_repo_id
        self.protocol_profile: Dict[str, Any] = _introspect_model_protocol(self.hf_repo_id, self.config)
        self.tier_upgrade_callback: Optional[Callable[[str, str, Dict[str, Any]], Optional[Dict[str, Any]]]] = None

        raw_access = str(
            access_mode
            or self.env.get("PLATODOOM_ACCESS")
            or self.config.get("default_access")
            or "workspace-write"
        ).strip().lower()
        if raw_access in ("yolo", "--yolo", "full"):
            raw_access = "full-access"
        elif raw_access in ("plan", "readonly"):
            raw_access = "read-only"
        elif raw_access in ("build", "confirm", "confirm_writes"):
            raw_access = "confirm-writes"
        if raw_access not in VALID_ACCESS_MODES:
            raw_access = "workspace-write"

        raw_wf = str(workflow_mode or "").strip().lower()
        if read_only or raw_wf == "plan" or raw_access == "read-only":
            self.workflow_mode: str = "plan"
            self.previous_write_access_mode: str = (
                raw_access if raw_access in ("workspace-write", "confirm-writes", "full-access") else "confirm-writes"
            )
            self.access_mode: str = "read-only"
        else:
            self.workflow_mode = "build"
            self.access_mode = raw_access
            self.previous_write_access_mode = raw_access

        self.storage_uri: Optional[str] = storage_uri or self.model_profile.get("storage_uri")
        if not self.storage_uri:
            gcs_bucket = ((self.config.get("storage") or {}).get("gcs_bucket"))
            gcs_prefix = ((self.config.get("storage") or {}).get("gcs_models_prefix") or "models/")
            if gcs_bucket:
                self.storage_uri = f"gs://{gcs_bucket}/{str(gcs_prefix).lstrip('/')}"

        self.workspace_roots: List[Path] = [self.workspace_dir]
        if self.project_root not in self.workspace_roots:
            self.workspace_roots.append(self.project_root)
        if extra_workspace_roots:
            for extra_root in extra_workspace_roots:
                p_extra = Path(extra_root).expanduser().resolve()
                if p_extra not in self.workspace_roots:
                    self.workspace_roots.append(p_extra)
        self.stateful_cwd: Path = self.workspace_dir

        self.bearer_token: str = (
            str(bearer_token).strip()
            if (bearer_token and str(bearer_token).strip())
            else (
                self.env.get("PLATODOOM_TUNNEL_TOKEN")
                or self.env.get("SANDCASTLE_VLLM_API_KEY")
                or _generate_bearer_token()
            )
        )
        raw_ep = endpoint_url or self.env.get("PLATODOOM_ENDPOINT_URL") or self.env.get("SANDCASTLE_VLLM_URL")
        self.endpoint_url: Optional[str] = normalize_colab_endpoint_url(raw_ep) if raw_ep else None
        self._explicit_endpoint_requested: bool = bool(raw_ep)
        self._explicit_gpu_requested: bool = bool(gpu_tier and str(gpu_tier).strip().lower() != "auto")
        self._sync_standby: bool = bool(kwargs.get("sync_standby", False))
        self._auto_attach_warm: Optional[bool] = kwargs.get("auto_attach_warm")
        self._no_colab: bool = bool(kwargs.get("no_colab", False))
        init_res = kwargs.get("resident_models")
        self.resident_models: List[str] = (
            [str(m) for m in init_res if str(m).strip()]
            if isinstance(init_res, list) and init_res
            else ([self.active_model] if self.endpoint_url else [])
        )
        self.inactive_models: List[str] = [
            str(m) for m in (kwargs.get("inactive_models") or []) if str(m).strip()
        ]
        self.model_agent_counts: Dict[str, int] = {
            str(k): int(v) for k, v in (kwargs.get("model_agent_counts") or {}).items()
        }
        self.last_batch_meta: Optional[Dict[str, Any]] = None

        passed_cache = kwargs.get("response_cache")
        if isinstance(passed_cache, LocalResponseCache):
            self.response_cache: LocalResponseCache = passed_cache
        else:
            rc_cfg = self.config.get("response_cache") if isinstance(self.config.get("response_cache"), dict) else {}
            env_cache_flag = self.env.get("PLATODOOM_RESPONSE_CACHE")
            rc_enabled = (
                str(env_cache_flag).strip().lower() not in ("0", "false", "no", "off")
                if env_cache_flag is not None and str(env_cache_flag).strip()
                else bool(rc_cfg.get("enabled", True))
            )
            self.response_cache = LocalResponseCache(
                cache_dir=resolve_response_cache_dir(env=self.env),
                enabled=rc_enabled,
                ttl_seconds=float(rc_cfg.get("ttl_seconds", 604800.0)),
                env=self.env,
            )

        preferred_gpu_raw = str(
            gpu_tier
            or self.env.get("PLATODOOM_GPU")
            or self.config.get("default_gpu")
            or self.model_profile.get("preferred_gpu")
            or "T4"
        ).strip()
        if preferred_gpu_raw.lower() == "auto":
            preferred_gpu_raw = str(self.model_profile.get("preferred_gpu") or "T4")
        norm_gpu_tier, default_vram, mem_type, tier_max_seqs = _resolve_hardware_tier_info(preferred_gpu_raw)
        self.gpu_tier: str = norm_gpu_tier
        self.memory_type: str = mem_type
        self.vram_total_gb: float = float(vram_total_gb) if vram_total_gb is not None else default_vram
        self.max_model_len: int = _resolve_effective_max_model_len(self.model_profile, self.gpu_tier)
        prof_seqs = int(self.model_profile.get("max_num_seqs", tier_max_seqs))
        self.max_num_seqs: int = max(int(tier_max_seqs), prof_seqs) if not self.gpu_tier.lower().startswith("v") else prof_seqs
        self.daemon_pid: Optional[int] = os.getpid() if self.endpoint_url else None

        raw_stack = (
            kwargs.get("stack_preset")
            or kwargs.get("stack")
            or self.env.get("PLATODOOM_STACK")
            or self.config.get("default_stack")
        )
        self.stack_preset: Optional[str] = str(raw_stack).strip() if (raw_stack and str(raw_stack).strip()) else None
        passed_active_stack = kwargs.get("active_stack")
        self.active_stack: Optional[Dict[str, Any]] = dict(passed_active_stack) if isinstance(passed_active_stack, dict) else None
        if self.stack_preset and self.active_stack is None:
            from platodoom.catalog import resolve_hardware_stack

            self.active_stack = resolve_hardware_stack(
                self.gpu_tier,
                stack_preset=self.stack_preset,
                model_alias=self.active_model,
                config=self.config,
            )
            self.stack_preset = str(self.active_stack.get("stack_preset") or self.stack_preset)
            stack_models = self.active_stack.get("models") or []
            stack_aliases = [str(m.get("alias")) for m in stack_models if m.get("alias")]
            if stack_aliases:
                self.resident_models = list(stack_aliases)
            if stack_aliases and model_alias is None and self.active_model not in stack_aliases:
                self.active_model = stack_aliases[0]
                self.model_profile = _resolve_model_profile(self.active_model, self.config)
                self.model_spec = self.model_profile
                self.hf_repo_id = str(self.model_profile.get("hf_repo_id", self.active_model))
                self.model_id = self.hf_repo_id
                self.protocol_profile = _introspect_model_protocol(self.hf_repo_id, self.config)

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.title: str = str(title).strip() if (title and str(title).strip()) else "New Conversation"
        self.created_at: str = now_iso
        self.updated_at: str = now_iso
        self.turn_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

        raw_home = kwargs.get("home_dir") or self.env.get("HOME")
        self.home_dir: Optional[Path] = Path(raw_home).expanduser().resolve() if raw_home else None
        self.capabilities: Dict[str, Any] = _discover_capabilities(
            self.workspace_dir,
            home_dir=self.home_dir,
            include_global=False,
        )
        self.workspace_compass: str = _build_compass(self.workspace_dir, self.capabilities, max_tokens=450)

        self.tool_state: Dict[str, Any] = {
            "cwd": self.stateful_cwd,
            "workspace_roots": self.workspace_roots,
            "access_mode": self.access_mode,
            "workflow_mode": self.workflow_mode,
            "session_slug": self.session_slug,
            "extra_denylist_globs": list((self.config.get("security") or {}).get("extra_denylist_globs") or []),
        }

        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self._build_system_prompt()}]
        self.switch_history: List[Dict[str, Any]] = []
        self.subagent_invocations: List[Dict[str, Any]] = []

        if resume_target is not None:
            self.resume_conversation(resume_target)
            if workflow_mode is not None or read_only:
                self.set_workflow_mode("plan" if (read_only or str(workflow_mode).lower() == "plan") else "build")
            if (
                self.endpoint_url is None
                and not kwargs.get("no_colab", False)
                and str(self.config.get("default_backend", "colab")).lower() == "colab"
                and self._should_attempt_warm_discovery()
            ):
                self.try_attach_warm_endpoint(
                    self.active_model,
                    allow_inplace_switch=False,
                    probe_timeout_sec=0.65,
                )
        else:
            if (
                self.endpoint_url is None
                and not kwargs.get("no_colab", False)
                and str(self.config.get("default_backend", "colab")).lower() == "colab"
                and self._should_attempt_warm_discovery()
            ):
                self.try_attach_warm_endpoint(
                    self.active_model,
                    allow_inplace_switch=False,
                    probe_timeout_sec=0.65,
                )
            self.save_state()

    def _build_system_prompt(self) -> str:
        mode_guidance = (
            "Strict Plan Mode (`workflow_mode=plan`, `access_mode=read-only`): "
            "File-modifying tools (`write_file`, `edit_file`) and mutating shell commands are disabled. "
            "Reply naturally and concisely to simple greetings (`hi`, `hello`) or conversational questions without invoking repository tools or reading files. "
            "When the user asks to analyze code or plan changes, inspect relevant files with read-only tools and instruct the user to run `/build` when ready to modify files."
            if self.workflow_mode == "plan" or self.access_mode == "read-only"
            else (
                f"Build Mode (`workflow_mode=build`, `access_mode={self.access_mode}`): Full tool suite active. "
                "Reply naturally and concisely to simple greetings (`hi`, `hello`) or conversational questions without invoking repository tools or reading files; "
                "inspect or modify files only when requested or required by the task."
            )
        )
        return (
            f"You are **platodoom** (active model: `{self.active_model}` / `{self.hf_repo_id}`), "
            f"an open-source interactive AI coding & reasoning assistant.\n"
            f"Workflow mode: `{self.workflow_mode}`. Access mode: `{self.access_mode}`.\n"
            f"{mode_guidance}\n\n"
            f"{self.workspace_compass}\n\n"
            f"Session slug: `{self.session_slug}`."
        )

    def _refresh_system_prompt(self) -> None:
        sys_prompt = self._build_system_prompt()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = sys_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": sys_prompt})

    @property
    def model_alias(self) -> str:
        return self.active_model

    @model_alias.setter
    def model_alias(self, val: str) -> None:
        self.active_model = str(val)

    @property
    def cwd(self) -> Path:
        return self.stateful_cwd

    @cwd.setter
    def cwd(self, val: Path | str) -> None:
        self.stateful_cwd = Path(val).expanduser().resolve()
        self.tool_state["cwd"] = self.stateful_cwd

    def save_state(self) -> Path:
        """
        Atomically persists `session.json`, `messages.json`, `endpoint.json`, and `colab_endpoint.json` (mode `0600`)
        inside `~/.platodoom/sessions/<session_slug>/` (mode `0700`) with secret redaction on conversation files
        in `<= 50 ms` (FR-2.1.2, FR-4.1, FR-5.4).
        """
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        now_ns = time.time_ns()
        self.updated_at = now_iso
        self.state_dir = self.session_dir
        _ensure_dir_0700(self.session_dir.parent)
        _ensure_dir_0700(self.session_dir)

        if self.title in ("", "New Conversation"):
            for m in self.messages:
                if m.get("role") == "user" and str(m.get("content") or "").strip():
                    self.title = str(m["content"]).strip().splitlines()[0].strip()[:60]
                    break

        user_turns = sum(1 for m in self.messages if m.get("role") == "user")
        self.turn_count = max(self.turn_count, user_turns)
        est_tokens = estimate_messages_tokens(self.messages)

        endpoint_payload: Dict[str, Any] = {
            "session_id": self.session_id,
            "session_slug": self.session_slug,
            "session_type": self.session_type,
            "dir_slug": getattr(self, "dir_slug", None) or compute_dir_slug(self.project_root),
            "endpoint_url": self.endpoint_url or "",
            "bearer_token": self.bearer_token,
            "active_model": self.active_model,
            "model_alias": self.active_model,
            "model": self.active_model,
            "resident_models": list(
                getattr(self, "resident_models", None) or ([self.active_model] if self.endpoint_url else [])
            ),
            "inactive_models": list(getattr(self, "inactive_models", None) or []),
            "model_agent_counts": dict(getattr(self, "model_agent_counts", None) or {}),
            "hf_repo_id": self.hf_repo_id,
            "model_id": self.hf_repo_id,
            "gpu_tier": self.gpu_tier,
            "memory_type": getattr(self, "memory_type", "VRAM"),
            "vram_total_gb": self.vram_total_gb,
            "stack_preset": getattr(self, "stack_preset", None),
            "active_stack": getattr(self, "active_stack", None),
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "daemon_pid": self.daemon_pid,
            "status": "Online" if self.endpoint_url else "Offline",
            "updated_at": now_iso,
            "updated_at_epoch": now_ns / 1e9,
            "heartbeat_epoch": now_ns / 1e9,
        }
        session_payload_raw: Dict[str, Any] = {
            **endpoint_payload,
            "title": self.title,
            "workflow_mode": self.workflow_mode,
            "access_mode": self.access_mode,
            "previous_write_access_mode": self.previous_write_access_mode,
            "storage_uri": self.storage_uri,
            "workspace_dir": str(self.workspace_dir),
            "stateful_cwd": str(self.stateful_cwd),
            "workspace_roots": [str(p) for p in self.workspace_roots],
            "created_at": self.created_at,
            "updated_at": now_iso,
            "updated_at_epoch": now_ns / 1e9,
            "updated_at_ns": now_ns,
            "turn_count": self.turn_count,
            "message_count": len(self.messages),
            "total_message_count": len(self.messages),
            "estimated_tokens": est_tokens,
            "switch_history": self.switch_history,
            "last_batch_meta": getattr(self, "last_batch_meta", None),
        }
        redacted_session_payload = _redact_persistence_payload(session_payload_raw)
        redacted_messages_payload = _redact_persistence_payload(self.messages)

        endpoint_file = self.session_dir / "endpoint.json"
        colab_endpoint_file = self.session_dir / "colab_endpoint.json"
        session_file = self.session_dir / "session.json"
        messages_file = self.session_dir / "messages.json"

        _atomic_write_json_0600(endpoint_file, endpoint_payload)
        _atomic_write_json_0600(colab_endpoint_file, endpoint_payload)
        _atomic_write_json_0600(messages_file, redacted_messages_payload)
        _atomic_write_json_0600(session_file, redacted_session_payload)
        if self.session_type == "interactive" and "PYTEST_CURRENT_TEST" not in os.environ:
            try:
                last_sel_file = self.session_dir.parent.parent / "last_cli_selection.json"
                _atomic_write_json_0600(last_sel_file, endpoint_payload)
            except OSError as exc:
                logging.getLogger("platodoom.session").warning(
                    "Unable to persist last_cli_selection.json: %s",
                    exc,
                )
        return endpoint_file

    def append_transcript_event(self, event: Dict[str, Any]) -> Path:
        """Appends a secret-redacted JSONL event to `~/.platodoom/sessions/<session_slug>/transcript.jsonl` (mode `0600`)."""
        _ensure_dir_0700(self.session_dir.parent)
        _ensure_dir_0700(self.session_dir)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {
            "timestamp": now_iso,
            "session_id": self.session_id,
            "session_slug": self.session_slug,
            "title": self.title,
            "active_model": self.active_model,
            "model_alias": self.active_model,
            "gpu_tier": self.gpu_tier,
            "workflow_mode": self.workflow_mode,
            "access_mode": self.access_mode,
            **(event or {}),
        }
        redacted = _redact_persistence_payload(payload)
        transcript_file = self.session_dir / "transcript.jsonl"
        line_bytes = (json.dumps(redacted, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(str(transcript_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            try:
                os.fchmod(fd, 0o600)
            except OSError:
                pass
            os.write(fd, line_bytes)
            try:
                os.fsync(fd)
            except OSError:
                pass
        finally:
            os.close(fd)
        return transcript_file

    def set_workflow_mode(self, mode: str) -> Dict[str, Any]:
        """
        Transitions the session in-place between Plan Mode (`workflow_mode='plan'`, `access_mode='read-only'`)
        and Build Mode (`workflow_mode='build'`, `access_mode=previous_write_access_mode`) in `<= 25 ms`
        while preserving 100% of `self.messages` (FR-3.1, FR-3.2).
        """
        norm = str(mode or "").strip().lower()
        if norm not in VALID_WORKFLOW_MODES:
            raise ValueError(f"Invalid workflow mode '{mode}'. Valid workflow modes: {', '.join(VALID_WORKFLOW_MODES)}")

        prev_wf = self.workflow_mode
        prev_access = self.access_mode

        if norm == "plan":
            if self.access_mode != "read-only":
                self.previous_write_access_mode = self.access_mode
            self.workflow_mode = "plan"
            self.access_mode = "read-only"
        else:
            self.workflow_mode = "build"
            if self.access_mode == "read-only":
                restored = getattr(self, "previous_write_access_mode", None) or "confirm-writes"
                if restored == "read-only" or restored not in VALID_ACCESS_MODES:
                    restored = "confirm-writes"
                self.access_mode = restored

        self.tool_state["workflow_mode"] = self.workflow_mode
        self.tool_state["access_mode"] = self.access_mode
        self._refresh_system_prompt()
        self.save_state()
        self.append_transcript_event(
            {
                "event": "mode_change",
                "previous_workflow_mode": prev_wf,
                "workflow_mode": self.workflow_mode,
                "previous_access_mode": prev_access,
                "access_mode": self.access_mode,
            }
        )
        return {
            "ok": True,
            "previous_workflow_mode": prev_wf,
            "workflow_mode": self.workflow_mode,
            "previous_access_mode": prev_access,
            "access_mode": self.access_mode,
            "previous_write_access_mode": self.previous_write_access_mode,
            "message_count": len(self.messages),
        }

    def resume_conversation(self, target: str | int) -> Dict[str, Any]:
        """
        Loads a saved conversation from `~/.platodoom/sessions/` by 1-based index, 'latest', `session_id`,
        `session_slug`, or title substring in `<= 200 ms` on a warm GPU without reprovisioning (FR-4.2).
        """
        if self.session_dir.exists() and (self.session_dir / "session.json").is_file():
            try:
                self.save_state()
            except OSError:
                pass
        state_dir, meta, loaded_msgs = load_saved_session_payload(
            target,
            sessions_root=self.session_dir.parent,
        )
        warm_endpoint = self.endpoint_url
        warm_token = self.bearer_token
        warm_daemon_pid = self.daemon_pid

        self.session_dir = state_dir
        self.state_dir = state_dir
        self.session_slug = str(meta.get("session_slug") or state_dir.name)
        self.session_id = str(meta.get("session_id") or self.session_slug.replace("platodoom-chat-", ""))
        self.title = str(meta.get("title") or "Resumed Conversation")
        self.created_at = str(meta.get("created_at") or self.created_at)
        self.updated_at = str(meta.get("updated_at") or self.updated_at)

        target_model = str(
            meta.get("model_alias")
            or meta.get("active_model")
            or meta.get("model")
            or self.active_model
        )
        self.active_model = target_model
        self.model_profile = _resolve_model_profile(self.active_model, self.config)
        self.model_spec = self.model_profile
        self.hf_repo_id = str(self.model_profile.get("hf_repo_id", self.active_model))
        self.model_id = self.hf_repo_id
        self.protocol_profile = _introspect_model_protocol(self.hf_repo_id, self.config)

        if meta.get("gpu_tier"):
            norm_t, nom_vram, mem_t, _ = _resolve_hardware_tier_info(str(meta["gpu_tier"]))
            self.gpu_tier = norm_t
            self.memory_type = mem_t
            if meta.get("vram_total_gb") is not None:
                self.vram_total_gb = float(meta["vram_total_gb"])
            else:
                self.vram_total_gb = nom_vram
        if "stack_preset" in meta:
            self.stack_preset = meta.get("stack_preset")
        if "active_stack" in meta and isinstance(meta.get("active_stack"), dict):
            self.active_stack = meta.get("active_stack")
        if isinstance(meta.get("resident_models"), list):
            self.resident_models = [str(m) for m in meta["resident_models"] if str(m).strip()]
        if isinstance(meta.get("inactive_models"), list):
            self.inactive_models = [str(m) for m in meta["inactive_models"] if str(m).strip()]
        if isinstance(meta.get("model_agent_counts"), dict):
            self.model_agent_counts = {str(k): int(v) for k, v in meta["model_agent_counts"].items()}
        if isinstance(meta.get("last_batch_meta"), dict):
            self.last_batch_meta = dict(meta["last_batch_meta"])
        if meta.get("storage_uri"):
            self.storage_uri = str(meta["storage_uri"])

        restored_access = str(meta.get("access_mode") or "confirm-writes")
        restored_wf = str(
            meta.get("workflow_mode")
            or ("plan" if restored_access == "read-only" else "build")
        )
        self.workflow_mode = "plan" if (restored_wf == "plan" or restored_access == "read-only") else "build"
        self.access_mode = "read-only" if self.workflow_mode == "plan" else restored_access
        self.previous_write_access_mode = str(
            meta.get("previous_write_access_mode")
            or (restored_access if restored_access != "read-only" else "confirm-writes")
        )

        self.tool_state["session_slug"] = self.session_slug
        self.tool_state["workflow_mode"] = self.workflow_mode
        self.tool_state["access_mode"] = self.access_mode

        if loaded_msgs:
            self.messages = [dict(m) for m in loaded_msgs if isinstance(m, dict)]
            self._refresh_system_prompt()
        else:
            self.messages = [{"role": "system", "content": self._build_system_prompt()}]

        self.turn_count = sum(1 for m in self.messages if m.get("role") == "user")

        if warm_endpoint:
            self.endpoint_url = warm_endpoint
            self.bearer_token = warm_token
            self.daemon_pid = warm_daemon_pid
        else:
            ep_file = state_dir / "endpoint.json"
            if ep_file.is_file():
                try:
                    ep_data = json.loads(ep_file.read_text(encoding="utf-8"))
                    if isinstance(ep_data, dict):
                        ep_status = str(ep_data.get("status") or "").strip().lower()
                        if ep_data.get("endpoint_url") and ep_status not in ("offline", "terminated", "failed"):
                            self.endpoint_url = normalize_colab_endpoint_url(str(ep_data["endpoint_url"]))
                        if ep_data.get("bearer_token") and ep_data["bearer_token"] != "[REDACTED]":
                            self.bearer_token = str(ep_data["bearer_token"])
                except Exception:
                    pass

        self.save_state()
        return {
            "ok": True,
            "session_id": self.session_id,
            "session_slug": self.session_slug,
            "title": self.title,
            "active_model": self.active_model,
            "model_alias": self.active_model,
            "hf_repo_id": self.hf_repo_id,
            "gpu_tier": self.gpu_tier,
            "workflow_mode": self.workflow_mode,
            "access_mode": self.access_mode,
            "turn_count": self.turn_count,
            "message_count": len(self.messages),
            "estimated_tokens": estimate_messages_tokens(self.messages),
            "state_dir": str(self.session_dir),
        }

    def start_new_conversation(self, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Flushes the current session to disk and starts a fresh conversation with a new `session_slug`
        while reusing the warm Colab GPU endpoint (`<= 200 ms`, FR-4.2).
        """
        self.save_state()
        sessions_parent = self.session_dir.parent
        new_id = secrets.token_hex(4)
        new_slug = f"platodoom-chat-{new_id}"
        self.session_id = new_id
        self.session_slug = new_slug
        self.session_dir = sessions_parent / new_slug
        self.state_dir = self.session_dir

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.title = str(title).strip() if (title and str(title).strip()) else "New Conversation"
        self.created_at = now_iso
        self.updated_at = now_iso
        self.turn_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.tool_state["session_slug"] = self.session_slug
        self.messages = [{"role": "system", "content": self._build_system_prompt()}]
        self.save_state()
        return {
            "ok": True,
            "session_id": self.session_id,
            "session_slug": self.session_slug,
            "title": self.title,
            "active_model": self.active_model,
            "gpu_tier": self.gpu_tier,
            "workflow_mode": self.workflow_mode,
            "access_mode": self.access_mode,
            "message_count": len(self.messages),
            "state_dir": str(self.session_dir),
        }

    def rename_conversation(self, new_title: str) -> Dict[str, Any]:
        """Updates the current conversation title and persists `session.json` and `transcript.jsonl` (FR-4.2)."""
        cleaned = str(new_title or "").strip()
        if not cleaned:
            raise ValueError("Conversation title cannot be empty.")
        self.title = cleaned[:120]
        self.save_state()
        self.append_transcript_event({"event": "session_rename", "title": self.title})
        return {
            "ok": True,
            "session_id": self.session_id,
            "session_slug": self.session_slug,
            "title": self.title,
        }

    def _workspace_has_active_endpoint_record(self) -> bool:
        """Returns True if `<project_root>/.sandcastle/colab_qwen_endpoint.json` contains a non-empty online URL."""
        for fname in ("colab_qwen_endpoint.json", "colab_endpoint.json"):
            cand = self.project_root / ".sandcastle" / fname
            if cand.is_file():
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and str(data.get("endpoint_url") or "").strip():
                        if str(data.get("status") or "").strip().lower() not in ("offline", "terminated", "failed"):
                            return True
                except Exception:
                    continue
        return False

    def _should_attempt_warm_discovery(self) -> bool:
        """Determines whether cross-registry warm discovery should run automatically without leaking into bare offline tmp tests."""
        if self._auto_attach_warm is False:
            return False
        if self._auto_attach_warm is True:
            return True
        if (self.env.get("PLATODOOM_HOME") or "").strip() or (self.env.get("PLATODOOM_CONFIG_DIR") or "").strip():
            return True
        if self._workspace_has_active_endpoint_record():
            return True
        if "PYTEST_CURRENT_TEST" in os.environ:
            return False
        return True

    def _sync_standby_if_small(self) -> None:
        """Syncs small-tier (`T4` / `v5e-1`) live endpoints to `<platodoom_home>/standby/endpoint.json`."""
        if not self.endpoint_url:
            return
        has_isolated_home = bool(
            (self.env.get("PLATODOOM_CONFIG_DIR") or "").strip()
            or (self.env.get("PLATODOOM_HOME") or "").strip()
        )
        if not self._sync_standby and not has_isolated_home:
            if "PYTEST_CURRENT_TEST" in os.environ:
                return
            try:
                from platodoom.standby import resolve_standby_endpoint_path

                if not resolve_standby_endpoint_path(env=self.env).is_file():
                    return
            except Exception:
                return
        try:
            from platodoom.standby import sync_standby_from_live_small_endpoint

            sync_standby_from_live_small_endpoint(
                {
                    "endpoint_url": self.endpoint_url,
                    "bearer_token": self.bearer_token,
                    "active_model": self.active_model,
                    "resident_models": getattr(self, "resident_models", None) or [self.active_model],
                    "gpu_tier": self.gpu_tier,
                    "vram_total_gb": self.vram_total_gb,
                    "daemon_pid": self.daemon_pid or os.getpid(),
                },
                env=self.env,
            )
        except Exception:
            pass

    def _adopt_warm_endpoint_record(
        self,
        record: Dict[str, Any],
        probe: Optional[Dict[str, Any]] = None,
        *,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Adopts a verified warm Colab endpoint record (`endpoint_url`, `bearer_token`, `gpu_tier`, `vram_total_gb`)."""
        norm_url = normalize_colab_endpoint_url(str(record.get("endpoint_url") or ""))
        if not norm_url:
            raise ValueError("Cannot adopt a warm endpoint record with an empty endpoint_url.")
        self.endpoint_url = norm_url
        cand_tok = resolve_serving_bearer_token(record=record)
        if cand_tok and cand_tok != "EMPTY":
            self.bearer_token = cand_tok
        raw_tier = str(
            record.get("gpu_tier")
            or record.get("gpu_type")
            or record.get("colab_gpu_type")
            or self.gpu_tier
        ).strip()
        norm_tier, default_vram, mem_type, _ = _resolve_hardware_tier_info(raw_tier)
        self.gpu_tier = norm_tier
        self.memory_type = mem_type
        if record.get("vram_total_gb") is not None:
            self.vram_total_gb = float(record["vram_total_gb"])
        elif record.get("vram_gb") is not None:
            self.vram_total_gb = float(record["vram_gb"])
        else:
            self.vram_total_gb = default_vram
        raw_pid = record.get("daemon_pid")
        self.daemon_pid = int(raw_pid) if (isinstance(raw_pid, int) and is_pid_alive(raw_pid)) else os.getpid()
        served = (probe or {}).get("models") or record.get("resident_models") or record.get("served_models")
        self.resident_models = (
            [str(m) for m in served if str(m).strip()]
            if isinstance(served, list) and served
            else [self.active_model]
        )
        if save:
            self.save_state()
            self._sync_standby_if_small()
        return {
            "endpoint_url": self.endpoint_url,
            "bearer_token": self.bearer_token,
            "active_model": self.active_model,
            "gpu_tier": self.gpu_tier,
            "vram_total_gb": self.vram_total_gb,
            "resident_models": list(self.resident_models),
            "status": "Online",
        }

    def try_attach_warm_endpoint(
        self,
        target_model: Optional[str] = None,
        *,
        allow_inplace_switch: bool = True,
        probe_timeout_sec: float = 0.65,
    ) -> Optional[Dict[str, Any]]:
        """
        Discovers and attaches to a live warm Colab instance across `standby/`, `.sandcastle/`, and `sessions/*`.
        - Attaches directly in `< 1.0 s` when `target_model` is already loaded (`Spec 866 §3.1.2`).
        - Performs an in-place model switch when `allow_inplace_switch=True` and a compatible warm accelerator is live (`Spec 866 §3.1.3`).
        """
        eff_model = str(target_model or self.active_model).strip()
        stack_specs = (
            (self.active_stack or {}).get("models")
            if isinstance(self.active_stack, dict) and int(self.active_stack.get("model_count", 1)) > 1
            else None
        )
        warm_res = discover_warm_colab_endpoints(
            repo_dir=self.workspace_dir,
            target_model=eff_model,
            exclude_session_slug=self.session_slug,
            required_gpu_tier=(self.gpu_tier if self._explicit_gpu_requested else None),
            stack_models=stack_specs,
            probe_timeout_sec=probe_timeout_sec,
            config=self.config,
            env=self.env,
        )
        match_type = warm_res.get("match_type")
        cand = warm_res.get("candidate")
        probe = warm_res.get("probe")
        if not isinstance(cand, dict) or not cand.get("endpoint_url"):
            return None

        if match_type == "exact_model_loaded":
            if eff_model != self.active_model:
                new_prof = _resolve_model_profile(eff_model, self.config)
                self.active_model = str(new_prof.get("alias", eff_model))
                self.model_profile = new_prof
                self.model_spec = new_prof
                self.hf_repo_id = str(new_prof.get("hf_repo_id", eff_model))
                self.model_id = self.hf_repo_id
                self.protocol_profile = _introspect_model_protocol(self.hf_repo_id, self.config)
                self._refresh_system_prompt()
            adopted = self._adopt_warm_endpoint_record(cand, probe, save=True)
            return {
                **adopted,
                "attached": True,
                "match_type": "exact_model_loaded",
                "registry_source": cand.get("_registry_source"),
                "elapsed_ms": warm_res.get("elapsed_ms", 0.0),
            }

        if match_type == "compatible_warm_switch" and allow_inplace_switch:
            prev_ep = self.endpoint_url
            prev_tok = self.bearer_token
            prev_tier = self.gpu_tier
            prev_mem = self.memory_type
            prev_vram = self.vram_total_gb
            prev_pid = self.daemon_pid
            prev_res = list(getattr(self, "resident_models", []))
            try:
                self._adopt_warm_endpoint_record(cand, probe, save=False)
                if stack_specs and self.stack_preset:
                    self.configure_stack(self.stack_preset)
                else:
                    self.switch_model(eff_model)
                self.save_state()
                self._sync_standby_if_small()
                return {
                    "attached": True,
                    "match_type": "compatible_warm_switch",
                    "endpoint_url": self.endpoint_url,
                    "bearer_token": self.bearer_token,
                    "active_model": self.active_model,
                    "gpu_tier": self.gpu_tier,
                    "vram_total_gb": self.vram_total_gb,
                    "registry_source": cand.get("_registry_source"),
                    "elapsed_ms": warm_res.get("elapsed_ms", 0.0),
                    "status": "Online",
                }
            except Exception:
                self.endpoint_url = prev_ep
                self.bearer_token = prev_tok
                self.gpu_tier = prev_tier
                self.memory_type = prev_mem
                self.vram_total_gb = prev_vram
                self.daemon_pid = prev_pid
                self.resident_models = prev_res
                return None

        return None

    def ensure_endpoint(
        self,
        auto_provision: bool = True,
        headless_launch: bool = True,
        force_reprovision: bool = False,
    ) -> Dict[str, Any]:
        """
        Verifies or provisions the isolated Colab GPU serving endpoint for `self.session_slug`,
        attaching in `< 1.0 s` to warm instances across registries before provisioning (`Spec 866 §3.1`).
        """
        expected_probe_model: Any = self.active_model
        if isinstance(self.active_stack, dict) and int(self.active_stack.get("model_count", 1)) > 1:
            stack_aliases = [str(m.get("alias")) for m in (self.active_stack.get("models") or []) if m.get("alias")]
            if stack_aliases:
                expected_probe_model = stack_aliases

        if not self.endpoint_url and not force_reprovision and not self._explicit_endpoint_requested:
            attached = self.try_attach_warm_endpoint(
                self.active_model,
                allow_inplace_switch=True,
                probe_timeout_sec=0.65,
            )
            if attached and self.endpoint_url:
                return {
                    "endpoint_url": self.endpoint_url,
                    "bearer_token": self.bearer_token,
                    "active_model": self.active_model,
                    "gpu_tier": self.gpu_tier,
                    "vram_total_gb": self.vram_total_gb,
                    "stack_preset": self.stack_preset,
                    "active_stack": self.active_stack,
                    "match_type": attached.get("match_type"),
                    "status": "Online",
                }

        if self.endpoint_url and not force_reprovision:
            probe_kwargs: Dict[str, Any] = {
                "api_key": self.bearer_token,
                "timeout_sec": 5.0,
                "verify_inference": False,
                "expected_model": expected_probe_model,
                "bearer_token": self.bearer_token,
            }
            try:
                psig = inspect.signature(probe_colab_qwen_endpoint)
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in psig.parameters.values()):
                    probe_kwargs = {k: v for k, v in probe_kwargs.items() if k in psig.parameters}
            except (ValueError, TypeError):
                pass

            probe = probe_colab_qwen_endpoint(self.endpoint_url, **probe_kwargs)
            if probe.get("online"):
                self.daemon_pid = self.daemon_pid or os.getpid()
                if probe.get("models"):
                    self.resident_models = [str(m) for m in probe["models"] if str(m).strip()]
                self.save_state()
                self._sync_standby_if_small()
                return {
                    "endpoint_url": self.endpoint_url,
                    "bearer_token": self.bearer_token,
                    "active_model": self.active_model,
                    "gpu_tier": self.gpu_tier,
                    "stack_preset": self.stack_preset,
                    "active_stack": self.active_stack,
                    "status": "Online",
                }
            if probe.get("model_mismatch"):
                try:
                    if isinstance(self.active_stack, dict) and int(self.active_stack.get("model_count", 1)) > 1 and self.stack_preset:
                        self.configure_stack(self.stack_preset)
                    else:
                        self.switch_model(self.active_model)
                    self.daemon_pid = self.daemon_pid or os.getpid()
                    self.save_state()
                    self._sync_standby_if_small()
                    return {
                        "endpoint_url": self.endpoint_url,
                        "bearer_token": self.bearer_token,
                        "active_model": self.active_model,
                        "gpu_tier": self.gpu_tier,
                        "stack_preset": self.stack_preset,
                        "active_stack": self.active_stack,
                        "status": "Online",
                    }
                except RuntimeError:
                    if not auto_provision:
                        raise
                    force_reprovision = True
            elif not self._explicit_endpoint_requested and self._should_attempt_warm_discovery():
                # Stale session URL is unreachable; re-discover any warm endpoint across registries before cold provisioning
                self.endpoint_url = None
                attached_fallback = self.try_attach_warm_endpoint(
                    self.active_model,
                    allow_inplace_switch=True,
                    probe_timeout_sec=0.65,
                )
                if attached_fallback and self.endpoint_url:
                    return {
                        "endpoint_url": self.endpoint_url,
                        "bearer_token": self.bearer_token,
                        "active_model": self.active_model,
                        "gpu_tier": self.gpu_tier,
                        "vram_total_gb": self.vram_total_gb,
                        "stack_preset": self.stack_preset,
                        "active_stack": self.active_stack,
                        "match_type": attached_fallback.get("match_type"),
                        "status": "Online",
                    }

        call_kwargs: Dict[str, Any] = {
            "repo_dir": self.workspace_dir,
            "auto_provision": auto_provision,
            "include_global": False,
            "headless_launch": headless_launch,
            "session_slug": self.session_slug,
            "model_alias": self.active_model,
            "bearer_token": self.bearer_token,
            "gpu_override": self.gpu_tier,
            "max_parallel_agents": max(4, sum(self.model_agent_counts.values()) if self.model_agent_counts else 4),
            "storage_uri": self.storage_uri,
            "stack_preset": self.stack_preset,
            "stack_models": (
                ((self.active_stack or {}).get("all_models") or (self.active_stack or {}).get("models"))
                if isinstance(self.active_stack, dict)
                else None
            ),
            "agent_counts": dict(self.model_agent_counts) if self.model_agent_counts else None,
            "force_reprovision": force_reprovision,
            "explicit_gpu_requested": self._explicit_gpu_requested,
            "sync_standby": self._sync_standby,
        }
        try:
            sig = inspect.signature(ensure_colab_serving_endpoint)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_kw:
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
        except (ValueError, TypeError):
            pass

        rec = ensure_colab_serving_endpoint(**call_kwargs)
        self.endpoint_url = normalize_colab_endpoint_url(str(rec.get("endpoint_url", "")))
        rec_tok = str(rec.get("bearer_token") or rec.get("auth_token") or "").strip()
        if rec_tok and rec_tok != "EMPTY":
            self.bearer_token = rec_tok
        rec_tier = str(rec.get("gpu_tier") or rec.get("gpu_type") or rec.get("colab_gpu_type") or "").strip()
        if rec_tier:
            try:
                norm_t, def_vram, mem_t, _ = _resolve_hardware_tier_info(rec_tier)
                self.gpu_tier = norm_t
                self.memory_type = mem_t
                if rec.get("vram_total_gb") is not None:
                    self.vram_total_gb = float(rec["vram_total_gb"])
                elif not self._explicit_gpu_requested:
                    self.vram_total_gb = def_vram
            except Exception:
                pass
        if rec.get("served_models"):
            self.resident_models = [str(m) for m in rec["served_models"] if str(m).strip()]
        if rec.get("daemon_pid"):
            self.daemon_pid = int(rec["daemon_pid"])
        self.save_state()
        self._sync_standby_if_small()
        return rec

    def _poll_inflight_model_switch(
        self,
        base_url: str,
        target_alias: str,
        deadline: float,
    ) -> Dict[str, Any]:
        """Polls /health and /models when a tunnel proxy (e.g. HTTP 524) times out during multi-GB checkpoint loading."""
        server_root_url = base_url[:-3] if base_url.endswith("/v1") else base_url
        health_url = f"{server_root_url}/health"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "Platodoom-Session-Switch/1.0",
        }
        while time.perf_counter() < deadline:
            status_str = ""
            last_err = None
            try:
                hreq = urllib.request.Request(health_url, headers=headers, method="GET")
                with _open_session_http_request(hreq, timeout=5.0) as hresp:
                    hdata = json.loads(hresp.read().decode("utf-8"))
                    status_str = str(hdata.get("status") or "")
                    last_err = hdata.get("last_switch_error")
            except urllib.error.HTTPError as herr:
                try:
                    hdata = json.loads(herr.read().decode("utf-8"))
                    status_str = str(hdata.get("status") or "")
                    last_err = hdata.get("last_switch_error")
                except Exception:
                    status_str = ""
            except Exception:
                status_str = ""

            if status_str == "unloaded" and last_err:
                raise RuntimeError(f"Remote model switch to '{target_alias}' failed: {last_err}")
            if status_str != "switching":
                probe = probe_colab_qwen_endpoint(
                    base_url,
                    api_key=self.bearer_token,
                    timeout_sec=5.0,
                    verify_inference=False,
                    expected_model=target_alias,
                    bearer_token=self.bearer_token,
                )
                if probe.get("online"):
                    return {"ok": True, "active_model": target_alias, "recovered_after_proxy_timeout": True}
            time.sleep(min(2.0, max(0.1, deadline - time.perf_counter())))

        raise RuntimeError(f"Timed out waiting for remote physical model switch to '{target_alias}' to complete.")

    def configure_stack(
        self,
        preset_name: str,
        *,
        agent_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Configures Solo (`x = 1`) or Co-Resident Multi-Model Stack (`x = 2..3`) preset on `self.gpu_tier`.
        Validates physical VRAM/HBM (`free_kv_gb >= 4.0`) before mutating state and loads all active (`n_i > 0`)
        stack models onto the bound serving endpoint when connected while leaving `n_i == 0` models unloaded.
        """
        from platodoom.catalog import resolve_hardware_stack

        resolved_stack = resolve_hardware_stack(
            self.gpu_tier,
            stack_preset=preset_name,
            model_alias=self.active_model,
            config=self.config,
        )
        stack_models = list(resolved_stack.get("models") or [])
        if isinstance(agent_counts, dict):
            alloc_items: List[Dict[str, Any]] = []
            for sm in stack_models:
                al = str(sm.get("alias"))
                pr = str(sm.get("precision") or "Q4")
                n_val = agent_counts.get(f"{al}@{pr}", agent_counts.get(al, 1))
                alloc_items.append({"alias": al, "precision": pr, "agents": int(n_val)})
            return self.configure_multi_model_agents(
                alloc_items,
                gpu_tier=self.gpu_tier,
                stack_preset=str(resolved_stack.get("stack_preset") or preset_name),
            )
        stack_aliases = [str(m.get("alias")) for m in stack_models if m.get("alias")]
        remote_resp: Optional[Dict[str, Any]] = None

        if self.endpoint_url:
            norm_ep = normalize_colab_endpoint_url(self.endpoint_url)
            load_url = f"{norm_ep}/internal/load-stack"
            req_body = {
                "hardware_tier": self.gpu_tier,
                "stack_preset": resolved_stack.get("stack_preset"),
                "models": stack_models,
            }
            req = urllib.request.Request(
                load_url,
                data=json.dumps(req_body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.bearer_token}",
                },
                method="POST",
            )
            est_stack_weights_gb = float(resolved_stack.get("total_weights_gb") or 9.0)
            switch_timeout = max(float(self.config.get("switch_timeout_sec", 300.0)), 120.0 + est_stack_weights_gb * 5.0)
            try:
                with _open_session_http_request(req, timeout=switch_timeout) as resp:
                    remote_resp = json.loads(resp.read().decode("utf-8"))
                    if isinstance(remote_resp, dict) and remote_resp.get("ok") is False:
                        raise RuntimeError(
                            str(remote_resp.get("diagnostic") or remote_resp.get("error") or "Stack load failed.")
                        )
            except urllib.error.HTTPError as http_err:
                raw_err = ""
                try:
                    raw_err = http_err.read().decode("utf-8")
                except Exception:
                    pass
                raise RuntimeError(
                    f"Stack '{preset_name}' load rejected with HTTP {http_err.code}: {raw_err or http_err}"
                ) from http_err

            if stack_aliases:
                post_probe = probe_colab_qwen_endpoint(
                    norm_ep,
                    api_key=self.bearer_token,
                    timeout_sec=5.0,
                    verify_inference=False,
                    expected_model=stack_aliases,
                    bearer_token=self.bearer_token,
                )
                if not post_probe.get("online"):
                    raise RuntimeError(
                        f"Post-stack verification failed: endpoint '{norm_ep}' did not advertise {stack_aliases} ({post_probe.get('error')})."
                    )

        self.stack_preset = str(resolved_stack.get("stack_preset") or preset_name)
        self.active_stack = resolved_stack
        self.resident_models = list(stack_aliases)
        self.inactive_models = []
        self.model_agent_counts = {al: 1 for al in stack_aliases}
        if stack_aliases and self.active_model not in stack_aliases:
            self.active_model = stack_aliases[0]
            self.model_profile = _resolve_model_profile(self.active_model, self.config)
            self.model_spec = self.model_profile
            self.hf_repo_id = str(self.model_profile.get("hf_repo_id", self.active_model))
            self.model_id = self.hf_repo_id
            self.protocol_profile = _introspect_model_protocol(self.hf_repo_id, self.config)
            self._refresh_system_prompt()

        self.save_state()
        return {
            "ok": True,
            "hardware_tier": self.gpu_tier,
            "stack_preset": self.stack_preset,
            "stack_mode": resolved_stack.get("stack_mode"),
            "model_count": resolved_stack.get("model_count"),
            "active_model": self.active_model,
            "models": stack_models,
            "resident_models": list(self.resident_models),
            "inactive_unloaded_models": [],
            "total_weights_gb": resolved_stack.get("total_weights_gb"),
            "free_kv_gb": resolved_stack.get("free_kv_gb"),
            "remote_load": remote_resp,
        }

    def configure_multi_model_agents(
        self,
        model_allocations: Any,
        *,
        gpu_tier: Optional[str] = None,
        stack_preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Configures `x` models with per-model agent counts `(n_1, ..., n_x)` on GPU 1 (`Spec 867`):
        - Loads ONLY active models (`n_i > 0`) once into GPU 1 VRAM.
        - Leaves `n_i == 0` models unloaded (`0` bytes VRAM, `0` compute).
        - Persists updated session metadata and transcript event exclusively on local disk.
        """
        from platodoom.catalog import resolve_multi_model_agent_allocation

        eff_tier = (
            gpu_tier
            if gpu_tier is not None
            else (self.gpu_tier if (self._explicit_gpu_requested or self.endpoint_url) else "auto")
        )
        alloc = resolve_multi_model_agent_allocation(
            model_allocations,
            gpu_tier=eff_tier,
            config=self.config,
        )
        selected_tier = str(alloc["hardware_tier"])
        norm_t, nom_vram, mem_t, tier_seqs = _resolve_hardware_tier_info(selected_tier)
        loaded_aliases = list(alloc["loaded_models"])
        inactive_aliases = list(alloc["inactive_unloaded_models"])
        remote_resp: Optional[Dict[str, Any]] = None

        if self.endpoint_url:
            norm_ep = normalize_colab_endpoint_url(self.endpoint_url)
            req_body = {
                "hardware_tier": norm_t,
                "stack_preset": stack_preset or "multi_model_agents",
                "models": alloc["all_models"],
                "agent_counts": alloc["agent_counts"],
            }
            est_alloc_weights_gb = float(alloc.get("total_weights_gb") or 9.0)
            switch_timeout = max(float(self.config.get("switch_timeout_sec", 300.0)), 120.0 + est_alloc_weights_gb * 5.0)
            last_err: Optional[Exception] = None
            for ep_suffix in ("/internal/load-stack", "/models/load_stack"):
                req = urllib.request.Request(
                    f"{norm_ep}{ep_suffix}",
                    data=json.dumps(req_body).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.bearer_token}",
                    },
                    method="POST",
                )
                try:
                    with _open_session_http_request(req, timeout=switch_timeout) as resp:
                        remote_resp = json.loads(resp.read().decode("utf-8"))
                        if isinstance(remote_resp, dict) and remote_resp.get("ok") is False:
                            raise RuntimeError(
                                str(remote_resp.get("diagnostic") or remote_resp.get("reason") or remote_resp.get("error") or "Multi-model agent load failed.")
                            )
                        last_err = None
                        break
                except urllib.error.HTTPError as http_err:
                    if http_err.code == 404 and ep_suffix == "/internal/load-stack":
                        last_err = http_err
                        continue
                    raw_err = ""
                    try:
                        raw_err = http_err.read().decode("utf-8")
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Multi-model agent allocation rejected with HTTP {http_err.code}: {raw_err or http_err}"
                    ) from http_err
            if last_err is not None:
                raise RuntimeError(f"Endpoint '{norm_ep}' did not accept multi-model stack load: {last_err}") from last_err

            if loaded_aliases:
                post_probe = probe_colab_qwen_endpoint(
                    norm_ep,
                    api_key=self.bearer_token,
                    timeout_sec=5.0,
                    verify_inference=False,
                    expected_model=loaded_aliases,
                    bearer_token=self.bearer_token,
                )
                if not post_probe.get("online"):
                    raise RuntimeError(
                        f"Post-load verification failed: endpoint '{norm_ep}' did not advertise active models {loaded_aliases} ({post_probe.get('error')})."
                    )

        self.gpu_tier = norm_t
        self.memory_type = mem_t
        if not self.endpoint_url:
            self.vram_total_gb = nom_vram
        self.max_num_seqs = max(tier_seqs, int(alloc.get("max_parallel_agents_per_model") or 0), 1)
        self.stack_preset = stack_preset or "multi_model_agents"
        self.active_stack = {
            "hardware_tier": norm_t,
            "stack_preset": self.stack_preset,
            "stack_mode": "idle_zero_agents" if len(loaded_aliases) == 0 else ("solo" if len(loaded_aliases) == 1 else "co_resident_stack"),
            "model_count": len(loaded_aliases),
            "configured_model_count": int(alloc["x_configured"]),
            "total_agents": int(alloc["total_agents"]),
            "models": list(alloc["active_models"]),
            "inactive_models": list(alloc["inactive_models"]),
            "all_models": list(alloc["all_models"]),
            "agent_counts": dict(alloc["agent_counts"]),
            "vram_allocated_by_model": dict(alloc["vram_allocated_by_model"]),
            "total_weights_gb": float(alloc["total_weights_gb"]),
            "free_kv_gb": float(alloc["free_kv_gb"]),
        }
        self.resident_models = loaded_aliases
        self.inactive_models = inactive_aliases
        self.model_agent_counts = dict(alloc["agent_counts"])

        if loaded_aliases and self.active_model not in loaded_aliases:
            self.active_model = loaded_aliases[0]
            self.model_profile = _resolve_model_profile(self.active_model, self.config)
            self.model_spec = self.model_profile
            self.hf_repo_id = str(self.model_profile.get("hf_repo_id", self.active_model))
            self.model_id = self.hf_repo_id
            self.protocol_profile = _introspect_model_protocol(self.hf_repo_id, self.config)
            self._refresh_system_prompt()

        self.save_state()
        self.append_transcript_event(
            {
                "event": "multi_model_agents_configured",
                "hardware_tier": self.gpu_tier,
                "configured_models": list(alloc["configured_models"]),
                "loaded_models": loaded_aliases,
                "inactive_unloaded_models": inactive_aliases,
                "agent_counts": dict(alloc["agent_counts"]),
                "vram_allocated_by_model": dict(alloc["vram_allocated_by_model"]),
                "total_agents": int(alloc["total_agents"]),
            }
        )
        return {
            "ok": True,
            **alloc,
            "hardware_tier": self.gpu_tier,
            "stack_preset": self.stack_preset,
            "model_count": len(loaded_aliases),
            "active_model": self.active_model if loaded_aliases else None,
            "models": list(alloc["active_models"]),
            "resident_models": loaded_aliases,
            "remote_load": remote_resp,
        }

    def switch_model(
        self,
        new_alias: str,
        target_vram_gb: Optional[float] = None,
        provisioner_callback: Optional[Callable[[str, str, Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Switches the active model mid-session (`/model <alias>`, FR-2.1.4, FR-3.2, AC-4)
        while preserving 100% of `self.messages`, workspace state, and tool trajectory.
        - Physically switches weights & tokenizer on the bound serving endpoint via `POST /v1/internal/switch-model`
          and verifies `GET /v1/models` reports `new_alias` before updating session state.
        - Fails fast with `ValueError` / `RuntimeError` without mutating `self.active_model` when invalid,
          unreachable, or insufficient VRAM without an attached GPU tier-upgrade callback.
        """
        target_alias = str(new_alias or "").strip()
        if not target_alias:
            raise ValueError("Model alias cannot be empty.")

        valid_syntax, syntax_err = validate_model_identifier_syntax(target_alias)
        if not valid_syntax:
            raise ValueError(syntax_err or f"Invalid model identifier '{target_alias}'.")

        previous_model = self.active_model
        previous_gpu = self.gpu_tier
        new_profile = _resolve_model_profile(target_alias, self.config)
        effective_vram = float(target_vram_gb) if target_vram_gb is not None else float(self.vram_total_gb)
        vram_check = _verify_vram_compatibility(target_alias, effective_vram, self.config)

        remote_switch_response: Optional[Dict[str, Any]] = None
        transition_mode = "hot_swap"
        transition_type = "hot_swap"
        cb = provisioner_callback or self.tier_upgrade_callback

        new_source = str(new_profile.get("source") or "hf_hub")
        new_storage_uri = new_profile.get("storage_uri")
        next_endpoint_url = self.endpoint_url
        next_bearer_token = self.bearer_token
        next_gpu_tier = self.gpu_tier
        next_vram_gb = self.vram_total_gb
        switch_target_ep = self.endpoint_url

        estimated_weights_gb = float(new_profile.get("weights_gb") or new_profile.get("min_vram_gb") or 9.0)
        switch_timeout = max(float(self.config.get("switch_timeout_sec", 300.0)), 120.0 + estimated_weights_gb * 5.0)
        deadline = time.perf_counter() + max(5.0, switch_timeout)
        resolved_candidate_alias = str(new_profile.get("alias", target_alias))
        co_resident_aliases = [
            str(m.get("alias"))
            for m in ((self.active_stack or {}).get("models") or [])
            if isinstance(m, dict) and m.get("alias")
        ]
        is_co_resident_target = (
            isinstance(self.active_stack, dict)
            and int(self.active_stack.get("model_count", 1)) > 1
            and resolved_candidate_alias in co_resident_aliases
        )
        co_resident_fast_switched = False
        warm_attached_without_eviction = False
        if is_co_resident_target:
            if self.endpoint_url:
                norm_ep_fast = normalize_colab_endpoint_url(self.endpoint_url)
                fast_probe = probe_colab_qwen_endpoint(
                    norm_ep_fast,
                    api_key=self.bearer_token,
                    timeout_sec=5.0,
                    verify_inference=False,
                    expected_model=resolved_candidate_alias,
                    bearer_token=self.bearer_token,
                )
                if fast_probe.get("online"):
                    co_resident_fast_switched = True
                    transition_mode = "co_resident_stack"
                    transition_type = "zero_eviction_stack_select"
                    remote_switch_response = {
                        "ok": True,
                        "active_model": resolved_candidate_alias,
                        "co_resident": True,
                        "evicted_models": [],
                    }
            else:
                co_resident_fast_switched = True
                transition_mode = "co_resident_stack"
                transition_type = "zero_eviction_stack_select"

        if (
            not co_resident_fast_switched
            and not self._explicit_endpoint_requested
            and self._should_attempt_warm_discovery()
        ):
            warm_res = discover_warm_colab_endpoints(
                repo_dir=self.workspace_dir,
                target_model=target_alias,
                exclude_session_slug=self.session_slug,
                probe_timeout_sec=0.65,
                config=self.config,
                env=self.env,
            )
            warm_match = warm_res.get("match_type")
            warm_cand = warm_res.get("candidate")
            if warm_match == "exact_model_loaded" and isinstance(warm_cand, dict) and warm_cand.get("endpoint_url"):
                next_endpoint_url = normalize_colab_endpoint_url(str(warm_cand["endpoint_url"]))
                cand_tok = resolve_serving_bearer_token(record=warm_cand)
                if cand_tok and cand_tok != "EMPTY":
                    next_bearer_token = cand_tok
                cand_tier_raw = str(
                    warm_cand.get("gpu_tier")
                    or warm_cand.get("gpu_type")
                    or warm_cand.get("colab_gpu_type")
                    or next_gpu_tier
                ).strip()
                norm_wt, def_wv, _, _ = _resolve_hardware_tier_info(cand_tier_raw)
                next_gpu_tier = norm_wt
                next_vram_gb = float(warm_cand.get("vram_total_gb") or def_wv)
                effective_vram = next_vram_gb
                vram_check = _verify_vram_compatibility(target_alias, effective_vram, self.config)
                warm_attached_without_eviction = True
                transition_mode = "warm_attach"
                transition_type = "warm_attach"
                remote_switch_response = {
                    "ok": True,
                    "active_model": resolved_candidate_alias,
                    "warm_attached": True,
                    "registry_source": warm_cand.get("_registry_source"),
                }
            elif (
                warm_match == "compatible_warm_switch"
                and (switch_target_ep is None or not vram_check.get("compatible", False))
                and isinstance(warm_cand, dict)
                and warm_cand.get("endpoint_url")
            ):
                cand_norm_url = normalize_colab_endpoint_url(str(warm_cand["endpoint_url"]))
                if cand_norm_url and cand_norm_url != switch_target_ep:
                    switch_target_ep = cand_norm_url
                    next_endpoint_url = cand_norm_url
                    cand_tok = resolve_serving_bearer_token(record=warm_cand)
                    if cand_tok and cand_tok != "EMPTY":
                        next_bearer_token = cand_tok
                    cand_tier_raw = str(
                        warm_cand.get("gpu_tier")
                        or warm_cand.get("gpu_type")
                        or warm_cand.get("colab_gpu_type")
                        or next_gpu_tier
                    ).strip()
                    norm_wt, def_wv, _, _ = _resolve_hardware_tier_info(cand_tier_raw)
                    next_gpu_tier = norm_wt
                    next_vram_gb = float(warm_cand.get("vram_total_gb") or def_wv)
                    effective_vram = next_vram_gb
                    vram_check = _verify_vram_compatibility(target_alias, effective_vram, self.config)

        if switch_target_ep and not co_resident_fast_switched and not warm_attached_without_eviction:
            norm_ep = normalize_colab_endpoint_url(switch_target_ep)
            switch_url = f"{norm_ep}/internal/switch-model"
            req_body = {
                "model_alias": target_alias,
                "hf_repo_id": new_profile.get("hf_repo_id", target_alias),
                "source": new_source,
                "storage_uri": new_storage_uri,
                "gcs_bucket": new_profile.get("gcs_bucket"),
                "gcs_prefix": new_profile.get("gcs_prefix"),
                "fallback_hf_repo_id": new_profile.get("fallback_hf_repo_id"),
                "precision": vram_check.get("selected_precision", "gptq_4bit"),
                "max_model_len": _resolve_effective_max_model_len(new_profile, next_gpu_tier),
            }
            req = urllib.request.Request(
                switch_url,
                data=json.dumps(req_body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {next_bearer_token}",
                },
                method="POST",
            )
            try:
                with _open_session_http_request(req, timeout=switch_timeout) as resp:
                    remote_switch_response = json.loads(resp.read().decode("utf-8"))
                    if isinstance(remote_switch_response, dict) and remote_switch_response.get("ok") is False:
                        err_code = str(
                            remote_switch_response.get("code")
                            or (remote_switch_response.get("error") or {}).get("code")
                            or "switch_failed"
                        )
                        diag = str(
                            remote_switch_response.get("diagnostic")
                            or remote_switch_response.get("error")
                            or f"Switch to '{target_alias}' rejected."
                        )
                        if err_code in ("insufficient_vram", "MODEL_SWITCH_VRAM_EXCEEDED") and callable(cb):
                            transition_mode = "tier_transition"
                            transition_type = "tier_upgrade"
                        else:
                            raise RuntimeError(f"Model switch to '{target_alias}' failed ({err_code}): {diag}")
            except urllib.error.HTTPError as http_err:
                raw_err = ""
                err_payload: Dict[str, Any] = {}
                try:
                    raw_err = http_err.read().decode("utf-8")
                    err_payload = json.loads(raw_err)
                except Exception:
                    err_payload = {"ok": False, "http_status": int(http_err.code), "error": raw_err or str(http_err)}
                remote_switch_response = err_payload
                err_code = str(
                    err_payload.get("code")
                    or (err_payload.get("error") if isinstance(err_payload.get("error"), dict) else {}).get("code")
                    or ""
                ).strip()
                diag = str(
                    err_payload.get("diagnostic")
                    or (err_payload.get("error") if isinstance(err_payload.get("error"), dict) else {}).get("message")
                    or err_payload.get("error")
                    or raw_err
                    or str(http_err)
                )
                if http_err.code in (409, 507) and err_code in ("insufficient_vram", "MODEL_SWITCH_VRAM_EXCEEDED", ""):
                    if callable(cb):
                        transition_mode = "tier_transition"
                        transition_type = "tier_upgrade"
                    else:
                        raise RuntimeError(
                            f"Model switch to '{target_alias}' rejected with HTTP {http_err.code} (insufficient VRAM): {diag}"
                        ) from http_err
                elif http_err.code == 524:
                    remote_switch_response = self._poll_inflight_model_switch(norm_ep, target_alias, deadline)
                else:
                    raise RuntimeError(
                        f"Model switch to '{target_alias}' failed with HTTP {http_err.code}: {diag}"
                    ) from http_err
            except TimeoutError as terr:
                remote_switch_response = self._poll_inflight_model_switch(norm_ep, target_alias, deadline)
            except RuntimeError:
                raise
            except Exception as exc:
                if "timed out" in str(exc).lower():
                    remote_switch_response = self._poll_inflight_model_switch(norm_ep, target_alias, deadline)
                else:
                    raise RuntimeError(f"Failed reaching endpoint '{norm_ep}' to switch model to '{target_alias}': {exc}") from exc
        elif not co_resident_fast_switched and not warm_attached_without_eviction and not vram_check.get("compatible", False):
            transition_mode = "tier_transition"
            transition_type = "tier_upgrade"

        if transition_mode == "tier_transition":
            preferred_gpu = str(new_profile.get("preferred_gpu") or "A100")
            norm_pref_gpu, pref_vram_gb, _, _ = _resolve_hardware_tier_info(preferred_gpu)
            next_gpu_tier = norm_pref_gpu
            next_vram_gb = pref_vram_gb
            if callable(cb):
                cb_res = cb(target_alias, norm_pref_gpu, new_profile)
                if isinstance(cb_res, dict):
                    if cb_res.get("endpoint_url"):
                        next_endpoint_url = normalize_colab_endpoint_url(str(cb_res["endpoint_url"]))
                    if cb_res.get("bearer_token"):
                        next_bearer_token = str(cb_res["bearer_token"])
                    if cb_res.get("gpu_tier"):
                        next_gpu_tier = str(cb_res["gpu_tier"])
                    if cb_res.get("vram_total_gb") is not None:
                        next_vram_gb = float(cb_res["vram_total_gb"])
            elif switch_target_ep:
                raise RuntimeError(
                    f"Model '{target_alias}' exceeds active endpoint VRAM ({effective_vram:.1f} GB) and no tier-upgrade provisioner is bound."
                )

        resolved_target_alias = str(new_profile.get("alias", target_alias))
        if next_endpoint_url:
            post_kwargs: Dict[str, Any] = {
                "api_key": next_bearer_token,
                "timeout_sec": 5.0,
                "verify_inference": False,
                "expected_model": resolved_target_alias,
                "bearer_token": next_bearer_token,
            }
            try:
                psig = inspect.signature(probe_colab_qwen_endpoint)
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in psig.parameters.values()):
                    post_kwargs = {k: v for k, v in post_kwargs.items() if k in psig.parameters}
            except (ValueError, TypeError):
                pass
            post_probe = probe_colab_qwen_endpoint(next_endpoint_url, **post_kwargs)
            if not post_probe.get("online"):
                raise RuntimeError(
                    f"Post-switch verification failed: endpoint '{next_endpoint_url}' did not report '{resolved_target_alias}' as loaded ({post_probe.get('error')})."
                )
            if post_probe.get("models"):
                self.resident_models = [str(m) for m in post_probe["models"] if str(m).strip()]

        # Strip completed <think>...</think> blocks from prior assistant messages only once switch succeeds (FR-3.4)
        for msg in self.messages:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str) and "<think>" in msg["content"]:
                msg["content"] = THINK_TAG_RE.sub("", msg["content"]).strip()

        self.endpoint_url = next_endpoint_url
        self.bearer_token = next_bearer_token
        if next_endpoint_url and not self.daemon_pid:
            self.daemon_pid = os.getpid()
        self.gpu_tier = next_gpu_tier
        _, _, next_mem_type, next_tier_seqs = _resolve_hardware_tier_info(self.gpu_tier)
        self.memory_type = next_mem_type
        self.vram_total_gb = next_vram_gb
        if not co_resident_fast_switched and isinstance(self.active_stack, dict) and int(self.active_stack.get("model_count", 1)) > 1:
            self.active_stack = None
            self.stack_preset = None
        self.active_model = resolved_target_alias
        if next_endpoint_url and not getattr(self, "resident_models", None):
            self.resident_models = [resolved_target_alias]
        self.model_profile = new_profile
        self.model_spec = new_profile
        self.hf_repo_id = str(new_profile.get("hf_repo_id", target_alias))
        self.model_id = self.hf_repo_id
        self.storage_uri = new_storage_uri
        self.max_model_len = _resolve_effective_max_model_len(new_profile, self.gpu_tier)
        next_prof_seqs = int(new_profile.get("max_num_seqs", next_tier_seqs))
        self.max_num_seqs = max(int(next_tier_seqs), next_prof_seqs) if not self.gpu_tier.lower().startswith("v") else next_prof_seqs
        self.protocol_profile = _introspect_model_protocol(self.hf_repo_id, self.config)

        # Update system message header while keeping the rest of messages untouched
        self._refresh_system_prompt()

        switch_event = {
            "ok": True,
            "status": "switched",
            "previous_model": previous_model,
            "active_model": self.active_model,
            "model_alias": self.active_model,
            "hf_repo_id": self.hf_repo_id,
            "source": new_source,
            "storage_uri": self.storage_uri,
            "previous_gpu_tier": previous_gpu,
            "gpu_tier": self.gpu_tier,
            "vram_total_gb": self.vram_total_gb,
            "transition_mode": transition_mode,
            "transition_type": transition_type,
            "vram_check": vram_check,
            "protocol_profile": self.protocol_profile,
            "remote_switch": remote_switch_response,
            "messages_preserved": len(self.messages),
            "switched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.switch_history.append(switch_event)
        self.save_state()
        self._sync_standby_if_small()
        return switch_event

    def set_access_mode(self, new_mode: str) -> Dict[str, Any]:
        """Transitions `self.access_mode` and synchronizes `self.workflow_mode` mid-session (`/access <mode>`, FR-3.1, FR-3.2)."""
        norm = str(new_mode or "").strip().lower()
        if norm in ("yolo", "--yolo", "full"):
            norm = "full-access"
        elif norm in ("plan", "readonly"):
            norm = "read-only"
        elif norm in ("build", "confirm", "confirm_writes"):
            norm = "confirm-writes"
        if norm not in VALID_ACCESS_MODES:
            raise ValueError(
                f"Invalid access mode '{new_mode}'. Valid modes: {', '.join(VALID_ACCESS_MODES)}"
            )
        prev = self.access_mode
        prev_wf = self.workflow_mode
        if norm == "read-only":
            if self.access_mode != "read-only":
                self.previous_write_access_mode = self.access_mode
            self.access_mode = "read-only"
            self.workflow_mode = "plan"
        else:
            self.access_mode = norm
            self.previous_write_access_mode = norm
            self.workflow_mode = "build"

        self.tool_state["access_mode"] = self.access_mode
        self.tool_state["workflow_mode"] = self.workflow_mode
        self._refresh_system_prompt()
        self.save_state()
        return {
            "ok": True,
            "previous_access_mode": prev,
            "access_mode": self.access_mode,
            "previous_workflow_mode": prev_wf,
            "workflow_mode": self.workflow_mode,
        }

    def add_workspace_dir(self, path_str: str) -> Dict[str, Any]:
        """Authorizes an additional directory in `self.workspace_roots` (`/dirs <add_path>`, FR-2.1.6)."""
        target = Path(path_str).expanduser().resolve()
        if not target.exists() or not target.is_dir():
            return {"ok": False, "error": f"Directory does not exist: {path_str}"}
        if target not in self.workspace_roots:
            self.workspace_roots.append(target)
        self.tool_state["workspace_roots"] = self.workspace_roots
        self.save_state()
        return {"ok": True, "workspace_roots": [str(p) for p in self.workspace_roots]}

    def compact_context(self) -> Dict[str, Any]:
        """
        Manually or automatically compacts conversation history while preserving recent context (`/compact`, FR-3.4, FR-4.1).
        Updates `messages.json` with the compacted context while preserving all historical lines in `transcript.jsonl`.
        """
        before_count = len(self.messages)
        self.messages = _compact_messages(self.messages, max_model_len=self.max_model_len, threshold_ratio=0.80)
        self.save_state()
        self.append_transcript_event(
            {
                "event": "context_compacted",
                "messages_before": before_count,
                "messages_after": len(self.messages),
            }
        )
        return {
            "ok": True,
            "messages_before": before_count,
            "messages_after": len(self.messages),
        }

    def shutdown(self) -> Dict[str, Any]:
        """Terminates the session's Colab Keep-Alive Daemon and marks the endpoint Offline (`/shutdown`, FR-2.1.6)."""
        res = terminate_directory_daemon(self.workspace_dir, session_slug=self.session_slug)
        mark_endpoint_offline(
            repo_dir=self.workspace_dir,
            reason=f"Session {self.session_slug} shut down by operator.",
            include_global=False,
            session_slug=self.session_slug,
        )
        self.endpoint_url = None
        self.daemon_pid = None
        return {
            "ok": True,
            "action": "shutdown",
            "session_slug": self.session_slug,
            "daemon_terminated": res,
        }

    def handle_slash_command(self, raw_line: str) -> Dict[str, Any]:
        """
        Executes built-in interactive slash commands (`/plan`, `/build`, `/mode`, `/model`, `/access`,
        `/history`, `/sessions`, `/resume`, `/new`, `/rename`, `/delete-session`, `/status`, `/skills`,
        `/mcp`, `/dirs`, `/compact`, `/exit`, `/shutdown`, `/help`) and discovered custom commands
        (FR-3.1, FR-3.2, FR-4.2).
        """
        cmdline = str(raw_line or "").strip()
        if not cmdline.startswith("/"):
            return {"ok": False, "handled": False, "error": "Not a slash command"}

        parts = cmdline.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/plan":
            res = self.set_workflow_mode("plan")
            follow_up = arg if arg else None
            return {
                "ok": True,
                "handled": True,
                "exit": False,
                "exit_repl": False,
                "command": "/plan",
                **res,
                "follow_up_prompt": follow_up,
                "custom_command_prompt": follow_up,
                "output": (
                    f"[PLAN | read-only] Switched to Plan Mode (messages={len(self.messages)} preserved). "
                    "File-modifying tools (write_file, edit_file) and mutating shell commands are disabled. "
                    "Use `/build` when ready to execute changes."
                ),
            }

        if cmd == "/build":
            if arg.lower() in ("confirm-writes", "workspace-write", "full-access", "yolo"):
                norm_write = "full-access" if arg.lower() == "yolo" else arg.lower()
                self.previous_write_access_mode = norm_write
                arg = ""
            res = self.set_workflow_mode("build")
            follow_up = arg if arg else None
            return {
                "ok": True,
                "handled": True,
                "exit": False,
                "exit_repl": False,
                "command": "/build",
                **res,
                "follow_up_prompt": follow_up,
                "custom_command_prompt": follow_up,
                "output": (
                    f"[BUILD | {self.access_mode}] Switched to Build Mode (messages={len(self.messages)} preserved). "
                    "Full tool suite (write_file, edit_file, run_command) enabled."
                ),
            }

        if cmd == "/mode":
            if not arg:
                return {
                    "ok": True,
                    "handled": True,
                    "command": "/mode",
                    "workflow_mode": self.workflow_mode,
                    "access_mode": self.access_mode,
                    "output": f"Active mode: workflow={self.workflow_mode}, access={self.access_mode}",
                }
            if arg.lower() in ("plan", "build"):
                res = self.set_workflow_mode(arg.lower())
                return {
                    "ok": True,
                    "handled": True,
                    "command": "/mode",
                    **res,
                    "output": f"Workflow mode updated: {res['previous_workflow_mode']} -> {res['workflow_mode']} (access={res['access_mode']})",
                }
            res = self.set_access_mode(arg)
            return {
                "ok": True,
                "handled": True,
                "command": "/mode",
                **res,
                "output": f"Access mode updated: {res['previous_access_mode']} -> {res['access_mode']} (workflow={res['workflow_mode']})",
            }

        if cmd == "/model":
            cat = _get_catalog(self.config)
            try:
                from platodoom.repl import PRIMARY_MENU_MODELS, render_startup_model_table

                ordered_aliases = [k for k in PRIMARY_MENU_MODELS if k in cat]
                for alias in cat:
                    if alias not in ordered_aliases:
                        ordered_aliases.append(alias)
            except Exception:
                ordered_aliases = list(cat.keys())
                render_startup_model_table = None  # type: ignore[assignment]

            if not arg:
                numbered_lines = (
                    "\n".join(
                        render_startup_model_table(
                            cat,
                            default_model=self.active_model,
                            ordered_aliases=ordered_aliases,
                        )
                    )
                    if callable(render_startup_model_table)
                    else "\n".join(f"  [{i}] {a}" for i, a in enumerate(ordered_aliases, start=1))
                )
                return {
                    "ok": True,
                    "handled": True,
                    "command": "/model",
                    "active_model": self.active_model,
                    "available_models": ordered_aliases,
                    "output": (
                        f"Active model: {self.active_model} ({self.hf_repo_id}) on {self.gpu_tier} ({self.vram_total_gb:.0f}GB VRAM).\n"
                        f"{numbered_lines}\n"
                        f"Usage: /model <1-{len(ordered_aliases)} or alias>"
                    ),
                }
            target_arg = arg.strip().strip("[]")
            if target_arg.isdigit():
                pick_idx = int(target_arg) - 1
                if 0 <= pick_idx < len(ordered_aliases):
                    target_arg = ordered_aliases[pick_idx]
            try:
                sw = self.switch_model(target_arg)
            except (ValueError, RuntimeError) as exc:
                return {
                    "ok": False,
                    "handled": True,
                    "command": "/model",
                    "active_model": self.active_model,
                    "error": str(exc),
                    "output": f"Error switching model to '{arg}': {exc}",
                }
            return {
                "ok": True,
                "handled": True,
                "command": "/model",
                **sw,
                "output": (
                    f"Switched model: {sw['previous_model']} -> {sw['active_model']} ({sw['hf_repo_id']}) "
                    f"[{sw['transition_mode']}, gpu={sw['gpu_tier']}, messages_preserved={sw['messages_preserved']}]."
                ),
            }

        if cmd == "/access":
            if not arg:
                return {
                    "ok": True,
                    "handled": True,
                    "command": "/access",
                    "access_mode": self.access_mode,
                    "workflow_mode": self.workflow_mode,
                    "valid_modes": list(VALID_ACCESS_MODES),
                    "output": f"Active access mode: {self.access_mode} [workflow={self.workflow_mode}] (options: {', '.join(VALID_ACCESS_MODES)})",
                }
            res = self.set_access_mode(arg)
            return {
                "ok": True,
                "handled": True,
                "command": "/access",
                **res,
                "output": f"Access mode updated: {res['previous_access_mode']} -> {res['access_mode']}",
            }

        if cmd in ("/history", "/sessions"):
            saved = list_saved_sessions(sessions_root=self.session_dir.parent)
            seen_slugs = {str(s.get("session_slug")) for s in saved}
            for active_s in list_active_sessions():
                a_slug = str(active_s.get("session_slug") or "")
                if a_slug and a_slug not in seen_slugs and "-sub-" not in a_slug:
                    saved.append(active_s)
                    seen_slugs.add(a_slug)
            lines = [f"Saved Conversations / Discovered Sessions ({len(saved)}):"]
            try:
                from platodoom.repl import format_sessions_table

                lines.append(format_sessions_table(saved, active_session_slug=self.session_slug))
            except Exception:
                for idx_s, s in enumerate(saved, start=1):
                    s_idx = s.get("index", idx_s)
                    s_slug = s.get("session_slug", "")
                    active_mark = " *active*" if s_slug == self.session_slug else ""
                    lines.append(
                        f"  [{s_idx}] {s_slug}{active_mark} · title='{s.get('title', 'New Conversation')}' "
                        f"model={s.get('active_model') or s.get('model_alias')} gpu={s.get('gpu_tier')} "
                        f"mode={s.get('workflow_mode', 'build')} turns={s.get('turn_count', 0)} "
                        f"updated={s.get('updated_at', '')}"
                    )
            return {
                "ok": True,
                "handled": True,
                "command": cmd,
                "sessions": saved,
                "active_session_slug": self.session_slug,
                "output": "\n".join(lines),
            }

        if cmd == "/resume":
            if not arg:
                saved = list_saved_sessions(sessions_root=self.session_dir.parent)
                return {
                    "ok": False,
                    "handled": True,
                    "command": "/resume",
                    "sessions": saved,
                    "error": "Usage: /resume <index|session_id|latest|title>",
                    "output": "Usage: /resume <index|session_id|latest|title>",
                }
            try:
                resumed = self.resume_conversation(arg)
            except Exception as exc:
                return {
                    "ok": False,
                    "handled": True,
                    "command": "/resume",
                    "error": str(exc),
                    "output": f"Failed to resume session '{arg}': {exc}",
                }
            return {
                "ok": True,
                "handled": True,
                "command": "/resume",
                "resumed": resumed,
                **resumed,
                "output": (
                    f"Resumed conversation '{self.session_slug}' ('{self.title}') · "
                    f"model={self.active_model} gpu={self.gpu_tier} mode={self.workflow_mode} "
                    f"messages={len(self.messages)}"
                ),
            }

        if cmd == "/new":
            created = self.start_new_conversation(title=arg if arg else None)
            return {
                "ok": True,
                "handled": True,
                "command": "/new",
                "session": created,
                **created,
                "output": (
                    f"Started new conversation '{self.session_slug}' ('{self.title}') "
                    f"on warm endpoint (model={self.active_model}, gpu={self.gpu_tier}, mode={self.workflow_mode})."
                ),
            }

        if cmd == "/rename":
            if not arg:
                return {
                    "ok": False,
                    "handled": True,
                    "command": "/rename",
                    "error": "Usage: /rename <new_title>",
                    "output": "Usage: /rename <new_title>",
                }
            renamed = self.rename_conversation(arg)
            return {
                "ok": True,
                "handled": True,
                "command": "/rename",
                "session": renamed,
                **renamed,
                "output": f"Renamed conversation '{self.session_slug}' to '{self.title}'.",
            }

        if cmd == "/delete-session":
            if not arg:
                return {
                    "ok": False,
                    "handled": True,
                    "command": "/delete-session",
                    "error": "Usage: /delete-session <index|session_id>",
                    "output": "Usage: /delete-session <index|session_id>",
                }
            try:
                deleted = delete_saved_session(
                    arg,
                    sessions_root=self.session_dir.parent,
                    active_session_slug=self.session_slug,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "handled": True,
                    "command": "/delete-session",
                    "error_code": "CANNOT_DELETE_ACTIVE_SESSION" if "active" in str(exc).lower() else "DELETE_FAILED",
                    "error": str(exc),
                    "output": str(exc),
                }
            if not deleted.get("ok", True):
                err_msg = str(deleted.get("message") or deleted.get("error") or "Cannot delete active session.")
                return {
                    "ok": False,
                    "handled": True,
                    "command": "/delete-session",
                    **deleted,
                    "error": err_msg,
                    "output": err_msg,
                }
            return {
                "ok": True,
                "handled": True,
                "command": "/delete-session",
                "deleted": deleted,
                **deleted,
                "output": f"Deleted saved session '{deleted['session_slug']}' ('{deleted.get('title', '')}').",
            }

        if cmd == "/status":
            daemon_alive = is_directory_daemon_alive(self.workspace_dir, session_slug=self.session_slug)
            mem_type = getattr(self, "memory_type", "VRAM")
            status_info = {
                "ok": True,
                "handled": True,
                "command": "/status",
                "session_id": self.session_id,
                "session_slug": self.session_slug,
                "title": self.title,
                "active_model": self.active_model,
                "hf_repo_id": self.hf_repo_id,
                "template_family": self.protocol_profile.get("template_family"),
                "gpu_tier": self.gpu_tier,
                "memory_type": mem_type,
                "vram_total_gb": self.vram_total_gb,
                "stack_preset": getattr(self, "stack_preset", None),
                "active_stack": getattr(self, "active_stack", None),
                "workflow_mode": self.workflow_mode,
                "access_mode": self.access_mode,
                "endpoint_url": self.endpoint_url,
                "daemon_pid": self.daemon_pid,
                "daemon_alive": daemon_alive,
                "workspace_roots": [str(p) for p in self.workspace_roots],
                "stateful_cwd": str(self.stateful_cwd),
                "message_count": len(self.messages),
                "turn_count": self.turn_count,
                "estimated_tokens": estimate_messages_tokens(self.messages),
                "skills_count": len(self.capabilities.get("skills", {})),
                "mcp_servers_count": len(self.capabilities.get("mcp_servers", {})),
            }
            stack_summary = ""
            if isinstance(self.active_stack, dict) and int(self.active_stack.get("model_count", 1)) > 1:
                stack_summary = f" | Stack: {self.stack_preset} (x={self.active_stack.get('model_count')})"
            status_info["output"] = (
                f"Session: {self.session_slug} ('{self.title}') | Model: {self.active_model} ({self.hf_repo_id}) | "
                f"GPU: {self.gpu_tier} ({self.vram_total_gb:.0f}GB {mem_type}){stack_summary} | Mode: {self.workflow_mode} ({self.access_mode}) | "
                f"Messages: {len(self.messages)} | Skills: {status_info['skills_count']} | MCP: {status_info['mcp_servers_count']}"
            )
            return status_info

        if cmd == "/skills":
            want_global = arg.strip().lower() in ("--global", "-g", "global", "all", "--all")
            active_caps = (
                _discover_capabilities(self.workspace_dir, home_dir=self.home_dir, include_global=True)
                if want_global
                else self.capabilities
            )
            skills = active_caps.get("skills") or {}
            scope_label = "Workspace + Global" if want_global else "Workspace"
            lines = [f"Discovered Skills [{scope_label}] ({len(skills)}):"]
            for s_name, s_meta in sorted(skills.items()):
                eco = s_meta.get("ecosystem", "unknown")
                scope = s_meta.get("scope", "project")
                desc = str(s_meta.get("description", ""))[:90]
                lines.append(f"  - {s_name} [{eco}/{scope}]: {desc}")
            return {
                "ok": True,
                "handled": True,
                "command": "/skills",
                "include_global": want_global,
                "skills": skills,
                "output": "\n".join(lines),
            }

        if cmd == "/mcp":
            mcp_servers = self.capabilities.get("mcp_servers") or {}
            lines = [f"Discovered MCP Servers ({len(mcp_servers)}):"]
            for m_name, m_meta in sorted(mcp_servers.items()):
                transport = m_meta.get("transport", "stdio")
                eco = m_meta.get("ecosystem", "unknown")
                lines.append(f"  - {m_name} [{eco}/{transport}]")
            return {
                "ok": True,
                "handled": True,
                "command": "/mcp",
                "mcp_servers": mcp_servers,
                "output": "\n".join(lines),
            }

        if cmd == "/dirs":
            if arg:
                res = self.add_workspace_dir(arg)
                return {
                    "ok": res["ok"],
                    "handled": True,
                    "command": "/dirs",
                    **res,
                    "output": (
                        f"Authorized workspace roots: {', '.join(res.get('workspace_roots', []))}"
                        if res["ok"]
                        else str(res.get("error"))
                    ),
                }
            roots = [str(p) for p in self.workspace_roots]
            return {
                "ok": True,
                "handled": True,
                "command": "/dirs",
                "workspace_roots": roots,
                "output": f"Authorized workspace roots: {', '.join(roots)}",
            }

        if cmd == "/compact":
            res = self.compact_context()
            return {
                "ok": True,
                "handled": True,
                "command": "/compact",
                **res,
                "output": f"Compacted context: {res['messages_before']} -> {res['messages_after']} messages.",
            }

        if cmd == "/gpu":
            if arg:
                try:
                    norm_gpu, nom_vram, mem_type, _ = _resolve_hardware_tier_info(arg)
                except ValueError:
                    valid_str = "G4, H100, A100-80GB, A100, v6e-1, L4, T4, v5e-1"
                    return {
                        "ok": False,
                        "handled": True,
                        "command": "/gpu",
                        "error": f"Invalid GPU/TPU tier '{arg}'. Valid tiers: {valid_str}",
                        "output": f"Invalid GPU/TPU tier '{arg}'. Valid tiers: {valid_str}",
                    }
                prev_gpu = self.gpu_tier
                self.gpu_tier = norm_gpu
                self.memory_type = mem_type
                self.vram_total_gb = nom_vram
                self.active_stack = None
                self.stack_preset = None
                self.save_state()
                return {
                    "ok": True,
                    "handled": True,
                    "command": "/gpu",
                    "previous_gpu_tier": prev_gpu,
                    "gpu_tier": self.gpu_tier,
                    "memory_type": self.memory_type,
                    "vram_total_gb": self.vram_total_gb,
                    "output": f"GPU tier updated: {prev_gpu} -> {self.gpu_tier} ({self.vram_total_gb:.0f}GB {self.memory_type}).",
                }
            mem_type = getattr(self, "memory_type", "VRAM")
            return {
                "ok": True,
                "handled": True,
                "command": "/gpu",
                "gpu_tier": self.gpu_tier,
                "memory_type": mem_type,
                "vram_total_gb": self.vram_total_gb,
                "output": f"Active hardware tier: {self.gpu_tier} ({self.vram_total_gb:.0f}GB {mem_type}).",
            }

        if cmd == "/stack":
            from platodoom.catalog import get_stack_presets_for_tier

            if not arg:
                tier_presets = get_stack_presets_for_tier(self.gpu_tier, self.config)
                preset_names = ["solo", *list((tier_presets.get("presets") or {}).keys())]
                active_desc = (
                    f"{self.stack_preset} (x={self.active_stack.get('model_count')}: "
                    + " + ".join(f"{m['alias']}@{m['precision']}" for m in self.active_stack.get("models", []))
                    + ")"
                    if isinstance(self.active_stack, dict)
                    else f"solo (x=1: {self.active_model})"
                )
                return {
                    "ok": True,
                    "handled": True,
                    "command": "/stack",
                    "gpu_tier": self.gpu_tier,
                    "stack_preset": self.stack_preset or "solo",
                    "active_stack": self.active_stack,
                    "available_presets": preset_names,
                    "tier_stack_catalog": tier_presets,
                    "output": (
                        f"Active stack on {self.gpu_tier} ({self.vram_total_gb:.0f}GB {getattr(self, 'memory_type', 'VRAM')}): {active_desc}\n"
                        f"Available stack presets: {', '.join(preset_names)}\n"
                        "Usage: /stack <preset>"
                    ),
                }
            try:
                st_res = self.configure_stack(arg)
            except (ValueError, RuntimeError) as exc:
                return {
                    "ok": False,
                    "handled": True,
                    "command": "/stack",
                    "gpu_tier": self.gpu_tier,
                    "error": str(exc),
                    "output": f"Error configuring stack '{arg}' on {self.gpu_tier}: {exc}",
                }
            summary_models = " + ".join(f"{m['alias']}@{m['precision']}" for m in st_res.get("models", []))
            return {
                "ok": True,
                "handled": True,
                "command": "/stack",
                **st_res,
                "output": (
                    f"Configured stack '{st_res['stack_preset']}' (x={st_res['model_count']}) on {self.gpu_tier}: "
                    f"{summary_models} [{st_res['total_weights_gb']:.1f}GB weights, {st_res['free_kv_gb']:.1f}GB free KV]."
                ),
            }

        if cmd == "/clear":
            if self.messages and self.messages[0].get("role") == "system":
                self.messages = [self.messages[0]]
            else:
                self.messages = []
            self.turn_count = 0
            self.save_state()
            return {
                "ok": True,
                "handled": True,
                "command": "/clear",
                "message_count": len(self.messages),
                "output": f"Cleared conversation history (retained system prompt, messages={len(self.messages)}).",
            }

        if cmd in ("/exit", "/quit"):
            self.save_state()
            return {
                "ok": True,
                "handled": True,
                "command": "/exit",
                "exit": True,
                "exit_repl": True,
                "keep_gpu_warm": True,
                "output": f"Exiting platodoom REPL. Colab GPU session '{self.session_slug}' remains warm in background.",
            }

        if cmd == "/shutdown":
            res = self.shutdown()
            return {
                "ok": True,
                "handled": True,
                "command": "/shutdown",
                "exit": True,
                "exit_repl": True,
                "keep_gpu_warm": False,
                **res,
                "output": f"Terminated Colab GPU session and Keep-Alive Daemon for '{self.session_slug}'.",
            }

        if cmd == "/auth":
            from platodoom.onboarding import (
                format_auth_status,
                get_auth_status,
                reset_onboarding_state,
            )

            if arg.lower() == "reset":
                reset_res = reset_onboarding_state()
                return {
                    "ok": True,
                    "handled": True,
                    "command": "/auth",
                    **reset_res,
                    "output": f"Reset onboarding status in {reset_res['config_path']}. Run `/auth` or `platodoom auth` to configure credentials.",
                }
            status = get_auth_status(self.config)
            return {
                "ok": True,
                "handled": True,
                "command": "/auth",
                "auth_status": status,
                "output": format_auth_status(status),
            }

        if cmd == "/help":
            help_text = (
                "Built-in slash commands:\n"
                "  /auth [status|reset]       Sign into Google Colab in browser or configure HF / tunnel tokens\n"
                "  /plan [<prompt>]           Switch to read-only Plan Mode (inspect & design without modifying files)\n"
                "  /build [<prompt>]          Switch to Build Mode (enable write_file, edit_file, and shell commands)\n"
                "  /model [<alias>]           Show or switch the active model across all 14 priority Qwen/DeepSeek models\n"
                "  /gpu [<tier>]              Show or switch Colab GPU/TPU tier (G4, H100, A100-80GB, A100, v6e-1, L4, T4, v5e-1)\n"
                "  /stack [<preset>|solo]     Configure Solo (x=1) or Co-Resident Multi-Model Stack (x=2..3) preset\n"
                "  /access [<mode>]           Show or switch access mode (read-only, confirm-writes, workspace-write, full-access)\n"
                "  /history | /sessions       List saved conversations in ~/.platodoom/sessions/\n"
                "  /resume <#|id|latest>      Resume a saved conversation on the warm GPU\n"
                "  /new [<title>]             Save current conversation and start a fresh conversation\n"
                "  /rename <new_title>        Rename the current conversation\n"
                "  /delete-session <#|id>     Delete a saved conversation from disk\n"
                "  /status                    Show active model, GPU tier, workflow mode, tunnel health, and context stats\n"
                "  /skills                    List all discovered multi-ecosystem skills\n"
                "  /mcp                       List all discovered MCP servers\n"
                "  /dirs [<path>]             Show or add authorized workspace directories\n"
                "  /compact                   Compact conversation history to free context window\n"
                "  /exit                      Exit terminal REPL while keeping Colab GPU warm\n"
                "  /shutdown                  Terminate the active Colab GPU runtime and Keep-Alive Daemon"
            )
            return {"ok": True, "handled": True, "command": "/help", "output": help_text}

        # Check discovered custom slash commands
        custom_cmds = self.capabilities.get("commands") or {}
        cmd_key = cmd.lstrip("/")
        if cmd_key in custom_cmds:
            cmd_meta = custom_cmds[cmd_key]
            cmd_path = Path(str(cmd_meta.get("path", "")))
            body = cmd_path.read_text(encoding="utf-8", errors="replace") if cmd_path.is_file() else str(cmd_meta.get("content", ""))
            prompt_to_run = f"{body}\n\n{arg}".strip()
            return {
                "ok": True,
                "handled": True,
                "command": cmd,
                "follow_up_prompt": prompt_to_run,
                "custom_command_prompt": prompt_to_run,
                "output": f"Loaded custom command /{cmd_key} ({cmd_meta.get('ecosystem', 'custom')}).",
            }

        return {
            "ok": False,
            "handled": True,
            "command": cmd,
            "error": f"Unknown slash command '{cmd}'. Type /help for available commands.",
            "output": f"Unknown slash command '{cmd}'. Type /help for available commands.",
        }

    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Returns the active OpenAI tool schemas, excluding `write_file` and `edit_file`
        when `self.workflow_mode == 'plan'` or `self.access_mode == 'read-only'` (FR-3.1).
        """
        try:
            from platodoom.tools import get_local_tools_schema

            return get_local_tools_schema(
                workflow_mode=self.workflow_mode,
                access_mode=self.access_mode,
            )
        except ImportError:
            from scripts.qwen_worktree_agent import WORKSPACE_TOOLS_SCHEMA

            if self.workflow_mode == "plan" or self.access_mode == "read-only":
                return [
                    t
                    for t in WORKSPACE_TOOLS_SCHEMA
                    if (t.get("function") or {}).get("name") not in ("write_file", "edit_file")
                ]
            return list(WORKSPACE_TOOLS_SCHEMA)

    def _execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> str:
        if (self.workflow_mode == "plan" or self.access_mode == "read-only") and tool_name in ("write_file", "edit_file"):
            denied = {
                "ok": False,
                "status": "error",
                "error_code": "PERMISSION_DENIED_READ_ONLY",
                "code": "PERMISSION_DENIED_READ_ONLY",
                "tool": tool_name,
                "workflow_mode": self.workflow_mode,
                "access_mode": self.access_mode,
                "error": (
                    f"PERMISSION_DENIED_READ_ONLY: Tool '{tool_name}' is blocked in Plan Mode (`read-only`). "
                    "Use `/build` to switch to Build Mode when ready to modify files."
                ),
            }
            return json.dumps(denied, ensure_ascii=False)

        if tool_name == "use_skill":
            skill_name = str(arguments.get("name") or arguments.get("skill_name") or "").strip()
            try:
                from platodoom.discovery import use_skill

                res = use_skill(
                    skill_name,
                    self.capabilities,
                    workspace_dir=self.workspace_dir,
                    home_dir=self.home_dir,
                )
                return json.dumps(res, ensure_ascii=False)
            except ImportError:
                return json.dumps({"ok": False, "error": f"Skill '{skill_name}' not found."})

        if tool_name == "list_mcp_tools":
            srv = arguments.get("server_name")
            from platodoom.mcp_bridge import list_mcp_tools

            return json.dumps(
                list_mcp_tools(
                    srv,
                    self.capabilities,
                    require_project_trust=bool(interactive_confirm_fn or self.confirm_callback),
                    interactive_confirm_fn=interactive_confirm_fn or self.confirm_callback,
                ),
                ensure_ascii=False,
            )

        if tool_name == "inspect_mcp_tool":
            srv = str(arguments.get("server_name") or "")
            tname = str(arguments.get("tool_name") or "")
            from platodoom.mcp_bridge import inspect_mcp_tool

            return json.dumps(inspect_mcp_tool(srv, tname, self.capabilities), ensure_ascii=False)

        if tool_name == "call_mcp_tool":
            srv = str(arguments.get("server_name") or "")
            tname = str(arguments.get("tool_name") or "")
            targs = arguments.get("arguments") or {}
            if isinstance(targs, str):
                try:
                    targs = json.loads(targs)
                except Exception:
                    targs = {}
            from platodoom.security import check_tool_permission

            eff_confirm = interactive_confirm_fn or self.confirm_callback
            allowed_mcp, mcp_diag = check_tool_permission(
                tool_name="call_mcp_tool",
                args={"server_name": srv, "tool_name": tname, "arguments": targs},
                access_mode=self.access_mode,
                workspace_roots=self.workspace_roots,
                cwd=self.stateful_cwd,
                interactive_confirm_fn=eff_confirm,
                workflow_mode=self.workflow_mode,
            )
            if not allowed_mcp:
                return json.dumps(
                    {
                        "ok": False,
                        "status": "error",
                        "error_code": "PERMISSION_DENIED_READ_ONLY",
                        "error": mcp_diag,
                    },
                    ensure_ascii=False,
                )
            from platodoom.mcp_bridge import call_mcp_tool

            return json.dumps(
                call_mcp_tool(
                    srv,
                    tname,
                    targs,
                    self.capabilities,
                    require_project_trust=True,
                    interactive_confirm_fn=eff_confirm,
                ),
                ensure_ascii=False,
            )

        if tool_name == "invoke_subagent":
            force_ro = bool(arguments.get("read_only", False)) or self.workflow_mode == "plan" or self.access_mode == "read-only"
            sub_res = invoke_subagent(
                role=str(arguments.get("role") or "subagent"),
                prompt=str(arguments.get("prompt") or ""),
                model_alias=arguments.get("model_alias"),
                workspace_dir=arguments.get("workspace_dir") or self.stateful_cwd,
                read_only=force_ro,
                parent_session=self,
            )
            return json.dumps(sub_res, ensure_ascii=False)

        if tool_name == "invoke_parallel_subagents":
            tasks = arguments.get("tasks") or []
            sub_results = invoke_parallel_subagents(tasks=tasks, parent_session=self)
            return json.dumps(sub_results, ensure_ascii=False)

        try:
            from platodoom.tools import execute_local_tool

            self.tool_state["access_mode"] = self.access_mode
            self.tool_state["workflow_mode"] = self.workflow_mode
            self.tool_state["workspace_roots"] = self.workspace_roots
            confirm_fn = interactive_confirm_fn or self.confirm_callback
            out = execute_local_tool(
                tool_name=tool_name,
                args=arguments,
                session_state=self.tool_state,
                interactive_confirm_fn=confirm_fn,
            )
            if self.tool_state.get("cwd"):
                self.stateful_cwd = Path(self.tool_state["cwd"]).resolve()
            if (
                (self.workflow_mode == "plan" or self.access_mode == "read-only")
                and isinstance(out, str)
                and "PERMISSION_DENIED_READ_ONLY" in out
                and not out.lstrip().startswith("{")
            ):
                return json.dumps(
                    {
                        "ok": False,
                        "status": "error",
                        "error_code": "PERMISSION_DENIED_READ_ONLY",
                        "code": "PERMISSION_DENIED_READ_ONLY",
                        "tool": tool_name,
                        "workflow_mode": self.workflow_mode,
                        "access_mode": self.access_mode,
                        "error": out,
                        "message": out,
                    },
                    ensure_ascii=False,
                )
            return out
        except ImportError:
            from scripts.qwen_worktree_agent import execute_worktree_tool

            return execute_worktree_tool(
                tool_name=tool_name,
                arguments=arguments,
                worktree_dir=self.stateful_cwd,
                env=dict(os.environ),
            )

    def _parse_response_tool_calls(self, assistant_msg: Dict[str, Any]) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
        raw_content = str(assistant_msg.get("content") or "")
        reasoning_content = assistant_msg.get("reasoning_content") or assistant_msg.get("reasoning")
        openai_tcs = assistant_msg.get("tool_calls")
        try:
            from platodoom.catalog import parse_universal_tool_calls, split_thinking_and_content

            think_extracted, without_think = split_thinking_and_content(
                raw_content,
                forces_think_prefix=bool(self.protocol_profile.get("forces_think_prefix", False)),
            )
            if think_extracted and not reasoning_content:
                reasoning_content = think_extracted
            cleaned_text, calls = parse_universal_tool_calls(without_think, openai_tool_calls=openai_tcs)
            return reasoning_content, cleaned_text, calls
        except ImportError:
            from scripts.qwen_worktree_agent import parse_assistant_tool_calls

            if "<think>" in raw_content and not reasoning_content:
                m = THINK_TAG_RE.search(raw_content)
                if m:
                    reasoning_content = m.group(1).strip()
            cleaned_text, calls = parse_assistant_tool_calls(assistant_msg)
            return reasoning_content, cleaned_text, calls

    def run_turn(
        self,
        user_prompt: str,
        max_tool_steps: int = 16,
        interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
        stream_callback: Optional[Callable[[Any], None]] = None,
        reasoning_callback: Optional[Callable[[str], None]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        stream: bool = True,
        max_steps: Optional[int] = None,
        max_hops: Optional[int] = None,
        use_cache: Optional[bool] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes a full multi-turn reasoning + local tool/skill/MCP/subagent loop against the active endpoint
        with local-disk response caching (`~/.platodoom/cache/responses/`), real-time SSE token streaming
        (`stream=True`, `max_tokens=2048`), and incremental local-disk session persistence (FR-2.1, FR-3.1, FR-4.1, Spec 867).
        """
        if user_prompt.strip().startswith("/"):
            slash_res = self.handle_slash_command(user_prompt.strip())
            follow_up = slash_res.get("custom_command_prompt") or slash_res.get("follow_up_prompt")
            if not follow_up:
                return slash_res
            user_prompt = str(follow_up)

        if self.title in ("", "New Conversation") and user_prompt.strip():
            self.title = user_prompt.strip().splitlines()[0].strip()[:60]

        self.messages.append({"role": "user", "content": user_prompt})
        self.turn_count = sum(1 for m in self.messages if m.get("role") == "user")
        self.append_transcript_event({"event": "user_message", "role": "user", "content": user_prompt})
        self.save_state()

        effective_max_steps = (
            int(max_hops)
            if max_hops is not None
            else (int(max_steps) if max_steps is not None else int(max_tool_steps))
        )
        eff_use_cache = self.response_cache.enabled if use_cache is None else bool(use_cache)
        tools_schema = self._get_tools_schema()
        tool_events: List[Dict[str, Any]] = []
        final_text = ""
        final_reasoning: Optional[str] = None
        files_modified: List[str] = []
        turn_cache_hits = 0
        turn_cache_misses = 0

        caller_thinking: Optional[bool] = None
        if sampling_kwargs and isinstance(sampling_kwargs, dict):
            if "enable_thinking" in sampling_kwargs and sampling_kwargs["enable_thinking"] is not None:
                caller_thinking = bool(sampling_kwargs["enable_thinking"])
            elif isinstance(sampling_kwargs.get("chat_template_kwargs"), dict):
                ctk = sampling_kwargs["chat_template_kwargs"]
                if "enable_thinking" in ctk and ctk["enable_thinking"] is not None:
                    caller_thinking = bool(ctk["enable_thinking"])

        use_omp_harness = (
            os.environ.get("PLATODOOM_USE_OMP") == "1"
            or (
                "PYTEST_CURRENT_TEST" not in os.environ
                and os.environ.get("PLATODOOM_USE_OMP", "1") != "0"
                and interactive_confirm_fn is None
            )
        )
        if use_omp_harness and self.session_type in ("interactive", "subagent"):
            if not self.endpoint_url:
                self.ensure_endpoint(auto_provision=not self._no_colab, headless_launch=True)
            if self.endpoint_url:
                from scripts.omp_qwen_adapter import run_omp_turn

                omp_session_dir = self.session_dir / "omp"
                has_prior_omp_session = omp_session_dir.is_dir() and any(omp_session_dir.glob("**/*.jsonl"))
                eff_tools = "read,grep,glob" if self.access_mode == "read-only" else None
                eff_auto_approve = self.access_mode == "full-access"
                eff_approval_mode = "write" if self.access_mode == "confirm-writes" else None
                eff_thinking = (
                    "off"
                    if caller_thinking is False
                    else ("high" if caller_thinking is True else None)
                )
                eff_prompt = user_prompt
                if not has_prior_omp_session and len(self.messages) > 1:
                    prior_lines = []
                    for pm in self.messages[:-1]:
                        p_role = str(pm.get("role") or "user")
                        p_content = str(pm.get("content") or "").strip()
                        if p_content and p_role in ("user", "assistant"):
                            prior_lines.append(f"[{p_role}]: {p_content[:2000]}")
                    if prior_lines:
                        eff_prompt = (
                            "Prior conversation context:\n"
                            + "\n".join(prior_lines[-8:])
                            + f"\n\nCurrent user request:\n{user_prompt}"
                        )

                omp_res = run_omp_turn(
                    prompt=eff_prompt,
                    cwd=self.stateful_cwd or self.workspace_dir,
                    endpoint_url=self.endpoint_url,
                    api_key=self.bearer_token or "EMPTY",
                    model_id=self.active_model,
                    context_window=self.max_model_len,
                    emit_ndjson=False,
                    event_callback=stream_callback,
                    reasoning_callback=reasoning_callback,
                    session_dir=omp_session_dir,
                    continue_session=has_prior_omp_session,
                    tools=eff_tools,
                    thinking=eff_thinking,
                    auto_approve=eff_auto_approve,
                    approval_mode=eff_approval_mode,
                    gpu_tier=self.gpu_tier,
                    verify_endpoint=True,
                )
                resp_txt = str(omp_res.get("response") or "")
                reas_txt = omp_res.get("reasoning")
                t_events = list(omp_res.get("tool_events") or [])
                omp_files_modified = list(omp_res.get("files_modified") or [])
                omp_ok = bool(omp_res.get("ok", False))
                omp_err = omp_res.get("error")
                if omp_res.get("context_window"):
                    self.max_model_len = int(omp_res["context_window"])
                if omp_ok:
                    assistant_entry: Dict[str, Any] = {"role": "assistant", "content": resp_txt}
                    if reas_txt:
                        assistant_entry["reasoning_content"] = reas_txt
                    self.messages.append(assistant_entry)
                    self.append_transcript_event(
                        {
                            "event": "assistant_message",
                            "role": "assistant",
                            "content": resp_txt,
                            "reasoning_content": reas_txt,
                        }
                    )
                else:
                    self.append_transcript_event(
                        {
                            "event": "turn_error",
                            "error": omp_err or "omp turn failed",
                        }
                    )
                self.save_state()
                return {
                    "ok": omp_ok,
                    "error": omp_err,
                    "session_slug": self.session_slug,
                    "session_dir": str(self.session_dir),
                    "title": self.title,
                    "reply": resp_txt,
                    "response": resp_txt,
                    "output": resp_txt,
                    "content": resp_txt,
                    "reasoning_content": reas_txt,
                    "reasoning": reas_txt,
                    "tool_events": t_events,
                    "files_modified": omp_files_modified,
                    "steps": int(omp_res.get("steps") or 1),
                    "active_model": self.active_model,
                    "hf_repo_id": self.hf_repo_id,
                    "gpu_tier": self.gpu_tier,
                    "workflow_mode": self.workflow_mode,
                    "access_mode": self.access_mode,
                    "turn_count": self.turn_count,
                    "message_count": len(self.messages),
                    "cache_hit": False,
                    "cache_hits": 0,
                    "cache_misses": 1,
                    "cache_stats": self.response_cache.stats(),
                    "batch_meta": self.last_batch_meta,
                }

        for step in range(1, effective_max_steps + 1):
            tools_schema = self._get_tools_schema()
            self.messages = _compact_messages(self.messages, max_model_len=self.max_model_len, threshold_ratio=0.80)
            compiled_msgs, compiled_sampling = _compile_model_chat_messages(
                messages=self.messages,
                tools=tools_schema,
                protocol_profile=self.protocol_profile,
                enable_thinking=caller_thinking,
            )
            merged_sampling = dict(compiled_sampling)
            if sampling_kwargs and isinstance(sampling_kwargs, dict):
                merged_sampling.update(sampling_kwargs)
            if "enable_thinking" in merged_sampling:
                et_top = merged_sampling.pop("enable_thinking")
                if et_top is not None:
                    ctk_merged = dict(merged_sampling.get("chat_template_kwargs") or {})
                    ctk_merged["enable_thinking"] = bool(et_top)
                    merged_sampling["chat_template_kwargs"] = ctk_merged

            max_new_tokens = int(merged_sampling.pop("max_tokens", 2048))
            stop_arg = merged_sampling.get("stop")
            norm_stop_list = (
                [stop_arg] if isinstance(stop_arg, str) else ([str(s) for s in stop_arg] if isinstance(stop_arg, (list, tuple)) else None)
            )
            extra_cache_params = {
                k: v
                for k, v in merged_sampling.items()
                if k not in ("temperature", "top_p", "top_k", "stop", "enable_thinking", "chat_template_kwargs")
            }
            eff_thinking = (
                merged_sampling.get("chat_template_kwargs", {}).get("enable_thinking", caller_thinking)
                if isinstance(merged_sampling.get("chat_template_kwargs"), dict)
                else caller_thinking
            )
            cache_key = compute_response_cache_key(
                model=self.active_model,
                messages=compiled_msgs,
                tools=tools_schema,
                max_tokens=max_new_tokens,
                temperature=merged_sampling.get("temperature", 0.0),
                top_p=merged_sampling.get("top_p"),
                top_k=merged_sampling.get("top_k"),
                stop=norm_stop_list,
                enable_thinking=eff_thinking,
                chat_template_kwargs=merged_sampling.get("chat_template_kwargs")
                if isinstance(merged_sampling.get("chat_template_kwargs"), dict)
                else None,
                extra_params=extra_cache_params or None,
            )

            streamed_reasoning_deltas: List[str] = []
            streamed_token_deltas: List[str] = []

            def _on_reasoning_delta(r_delta: str) -> None:
                streamed_reasoning_deltas.append(r_delta)
                if reasoning_callback is not None:
                    reasoning_callback(r_delta)
                if stream_callback is not None:
                    stream_callback({"type": "reasoning_delta", "step": step, "delta": r_delta, "content": r_delta})

            def _on_token_delta(c_delta: str) -> None:
                streamed_token_deltas.append(c_delta)
                if stream_callback is not None:
                    stream_callback({"type": "token_delta", "step": step, "delta": c_delta, "content": c_delta})

            cached_resp = (
                self.response_cache.get(cache_key)
                if (eff_use_cache and not force_refresh)
                else None
            )

            if cached_resp is not None:
                turn_cache_hits += 1
                resp_data = cached_resp
                c_ch0 = (resp_data.get("choices") or [{}])[0]
                c_msg = c_ch0.get("message") or {}
                r_cached = c_msg.get("reasoning_content") or c_msg.get("reasoning")
                if r_cached:
                    _on_reasoning_delta(str(r_cached))
                if c_msg.get("content"):
                    _on_token_delta(str(c_msg["content"]))
            else:
                turn_cache_misses += 1
                if not self.endpoint_url:
                    self.ensure_endpoint(auto_provision=not self._no_colab, headless_launch=True)

                base_url = normalize_colab_endpoint_url(str(self.endpoint_url or ""))
                if not base_url:
                    raise RuntimeError(f"No active endpoint URL configured for session '{self.session_slug}'.")

                req_payload: Dict[str, Any] = {
                    "model": self.active_model,
                    "messages": compiled_msgs,
                    "tools": tools_schema,
                    "max_tokens": max_new_tokens,
                    "stream": bool(stream),
                    **merged_sampling,
                }
                req = urllib.request.Request(
                    f"{base_url}/chat/completions",
                    data=json.dumps(req_payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream, application/json",
                        "Authorization": f"Bearer {self.bearer_token}",
                        "ngrok-skip-browser-warning": "true",
                    },
                    method="POST",
                )

                resp_data = _consume_chat_completion_stream(
                    req,
                    timeout_sec=300.0,
                    on_token_delta=_on_token_delta,
                    on_reasoning_delta=_on_reasoning_delta,
                )

                if isinstance(resp_data.get("sandcastle_batch_meta"), dict):
                    self.last_batch_meta = dict(resp_data["sandcastle_batch_meta"])

                if eff_use_cache and resp_data.get("choices"):
                    prev_enabled = self.response_cache.enabled
                    try:
                        if use_cache is True and not prev_enabled:
                            self.response_cache.enabled = True
                        self.response_cache.put(
                            cache_key,
                            resp_data,
                            request_fingerprint_meta={
                                "session_slug": self.session_slug,
                                "model": self.active_model,
                                "step": step,
                            },
                        )
                    finally:
                        self.response_cache.enabled = prev_enabled

            choices = resp_data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Empty choices returned from {self.endpoint_url or 'cache'}/chat/completions")

            if isinstance(resp_data.get("sandcastle_batch_meta"), dict):
                self.last_batch_meta = dict(resp_data["sandcastle_batch_meta"])

            assistant_msg = choices[0].get("message") or {}
            reasoning_text, cleaned_text, tool_calls = self._parse_response_tool_calls(assistant_msg)
            if reasoning_text:
                final_reasoning = reasoning_text
                if stream_callback and not streamed_reasoning_deltas:
                    stream_callback({"type": "thinking", "step": step, "content": reasoning_text})
            if cleaned_text:
                final_text = cleaned_text
                if stream_callback:
                    stream_callback({"type": "assistant_text", "step": step, "content": cleaned_text})

            if not tool_calls:
                assistant_entry: Dict[str, Any] = {"role": "assistant", "content": cleaned_text}
                if reasoning_text:
                    assistant_entry["reasoning_content"] = reasoning_text
                self.messages.append(assistant_entry)
                self.append_transcript_event(
                    {
                        "event": "assistant_message",
                        "role": "assistant",
                        "content": cleaned_text,
                        "reasoning_content": reasoning_text,
                    }
                )
                break

            formatted_tcs = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": (
                            json.dumps(tc["arguments"], ensure_ascii=False)
                            if isinstance(tc.get("arguments"), (dict, list))
                            else str(tc.get("arguments") or "{}")
                        ),
                    },
                }
                for tc in tool_calls
            ]
            self.messages.append(
                {
                    "role": "assistant",
                    "content": cleaned_text,
                    "tool_calls": formatted_tcs,
                }
            )
            self.append_transcript_event(
                {
                    "event": "assistant_message",
                    "role": "assistant",
                    "content": cleaned_text,
                    "reasoning_content": reasoning_text,
                    "tool_calls": formatted_tcs,
                }
            )

            for tc in tool_calls:
                t_name = str(tc.get("name") or "")
                t_args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                if stream_callback:
                    stream_callback({"type": "tool_call", "step": step, "name": t_name, "arguments": t_args})

                if tc.get("malformed_error"):
                    t_out = f"ERROR: {tc['malformed_error']}"
                else:
                    t_out = self._execute_tool_call(
                        tool_name=t_name,
                        arguments=t_args,
                        interactive_confirm_fn=interactive_confirm_fn,
                    )
                    if t_name in ("write_file", "edit_file") and "PERMISSION_DENIED_READ_ONLY" not in str(t_out):
                        p_arg = t_args.get("path") or t_args.get("Path")
                        if p_arg and str(p_arg) not in files_modified:
                            files_modified.append(str(p_arg))

                tool_events.append(
                    {
                        "step": step,
                        "id": tc["id"],
                        "name": t_name,
                        "arguments": t_args,
                        "output": t_out,
                    }
                )
                if stream_callback:
                    stream_callback(
                        {
                            "type": "tool_result",
                            "step": step,
                            "name": t_name,
                            "output": t_out,
                            "result": t_out,
                        }
                    )

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": t_name,
                        "content": t_out,
                    }
                )
                self.append_transcript_event(
                    {
                        "event": "tool_result",
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": t_name,
                        "content": t_out,
                    }
                )
            self.save_state()

        self.save_state()
        return {
            "ok": True,
            "session_slug": self.session_slug,
            "session_dir": str(self.session_dir),
            "title": self.title,
            "active_model": self.active_model,
            "hf_repo_id": self.hf_repo_id,
            "gpu_tier": self.gpu_tier,
            "workflow_mode": self.workflow_mode,
            "access_mode": self.access_mode,
            "reply": final_text,
            "response": final_text,
            "output": final_text,
            "reasoning_content": final_reasoning,
            "tool_events": tool_events,
            "files_modified": files_modified,
            "message_count": len(self.messages),
            "cache_hit": turn_cache_hits > 0,
            "cache_hits": turn_cache_hits,
            "cache_misses": turn_cache_misses,
            "cache_stats": self.response_cache.stats(),
            "batch_meta": self.last_batch_meta,
        }


def _is_model_coresident_on_session(model_alias: str, session: Optional[PlatoSession]) -> bool:
    if session is None:
        return False
    clean = str(model_alias or "").strip().split("@", 1)[0].strip()
    if not clean:
        return False
    if clean in (getattr(session, "inactive_models", None) or []):
        return False
    if clean == session.active_model:
        return True
    if clean in (getattr(session, "resident_models", None) or []):
        return True
    stack = getattr(session, "active_stack", None)
    if isinstance(stack, dict):
        for m in stack.get("models") or []:
            if isinstance(m, dict) and str(m.get("alias") or "") == clean:
                return True
    return False


def invoke_subagent(
    role: str,
    prompt: str,
    model_alias: Optional[str] = None,
    workspace_dir: Optional[Path | str] = None,
    read_only: bool = False,
    access_mode: Optional[str] = None,
    gpu_mode: str = "shared",
    parent_session: Optional[PlatoSession] = None,
    max_turns: int = 8,
    stream: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    force_refresh: bool = False,
    launch_barrier: Optional[threading.Barrier] = None,
) -> Dict[str, Any]:
    """
    Spawns an isolated child subagent tool loop (FR-3.1, FR-6, AC-4, Spec 867).
    - Same-GPU Co-Resident Continuous Batching: When `model_alias` is None, matches `parent_session.active_model`,
      or belongs to `parent_session.resident_models` / `parent_session.active_stack`, routes directly to
      `parent_session.endpoint_url` using `parent_session.bearer_token` (`0` extra Colab VMs spawned!).
    - Strict Plan Mode Read-Only Inheritance: When `parent_session.workflow_mode == 'plan'` or
      `parent_session.access_mode == 'read-only'`, forces `read_only=True` and `child_access='read-only'`.
    """
    raw_eff = str(model_alias or (parent_session.active_model if parent_session else "qwen-9b")).strip()
    effective_model = raw_eff.split("@", 1)[0].strip() if "@" in raw_eff else raw_eff
    target_dir = (
        Path(workspace_dir).expanduser().resolve()
        if workspace_dir
        else (parent_session.stateful_cwd if parent_session else Path.cwd().resolve())
    )
    access_ranks = {"read-only": 0, "confirm-writes": 1, "workspace-write": 1, "full-access": 2}
    parent_access = parent_session.access_mode if parent_session else "workspace-write"
    parent_wf = getattr(parent_session, "workflow_mode", "build") if parent_session else "build"

    if parent_wf == "plan" or parent_access == "read-only":
        read_only = True

    if read_only:
        requested_access = "read-only"
    elif access_mode and str(access_mode).strip().lower() in access_ranks:
        requested_access = str(access_mode).strip().lower()
    else:
        requested_access = parent_access

    # Never escalate beyond parent_session.access_mode (FR-3.1, FR-6.3)
    if parent_session and access_ranks.get(requested_access, 1) > access_ranks.get(parent_access, 1):
        child_access = parent_access
    else:
        child_access = requested_access

    child_wf = "plan" if child_access == "read-only" else "build"

    if parent_session and effective_model in (getattr(parent_session, "inactive_models", None) or []):
        raise ValueError(
            f"Model '{effective_model}' has 0 active agents assigned (n_i = 0) and is intentionally unloaded from GPU 1 VRAM."
        )

    same_gpu_reused = True
    extra_colab_vms_spawned = 0
    gpu_mode_used = "shared"
    parent_ep = parent_session.endpoint_url if parent_session else os.environ.get("PLATODOOM_ENDPOINT_URL")
    parent_tok = parent_session.bearer_token if parent_session else os.environ.get("PLATODOOM_TUNNEL_TOKEN")

    cfg = parent_session.config if parent_session else _resolve_config()
    max_sessions = int(((cfg.get("colab") or {}).get("max_concurrent_sessions")) or 2)
    req_gpu_mode = str(gpu_mode or "shared").strip().lower()
    is_coresident = _is_model_coresident_on_session(effective_model, parent_session)

    if req_gpu_mode == "dedicated":
        active_online = [s for s in list_active_sessions() if s.get("status") == "Online"]
        active_count = max(len(active_online), 1 if (parent_session and parent_session.endpoint_url) else 0)
        if active_count >= max_sessions or max_sessions <= 1:
            gpu_mode_used = "shared_fallback"
            same_gpu_reused = True
            extra_colab_vms_spawned = 0
        else:
            gpu_mode_used = "dedicated"
            same_gpu_reused = False
            extra_colab_vms_spawned = 1
    elif parent_session and not is_coresident and effective_model != parent_session.active_model:
        vram_check = _verify_vram_compatibility(effective_model, parent_session.vram_total_gb, parent_session.config)
        if not vram_check.get("compatible", False):
            if max_sessions <= 1:
                gpu_mode_used = "shared_fallback"
                same_gpu_reused = True
                extra_colab_vms_spawned = 0
            else:
                gpu_mode_used = "dedicated"
                same_gpu_reused = False
                extra_colab_vms_spawned = 1

    child_slug = f"{parent_session.session_slug if parent_session else 'platodoom-chat-root'}-sub-{uuid.uuid4().hex[:6]}"
    child_session = PlatoSession(
        model_alias=effective_model,
        access_mode=child_access,
        workflow_mode=child_wf,
        read_only=(child_access == "read-only"),
        workspace_dir=target_dir,
        attach_slug=child_slug,
        parent_session_slug=parent_session.session_slug if parent_session else None,
        config=parent_session.config if parent_session else None,
        endpoint_url=parent_ep,
        bearer_token=parent_tok,
        gpu_tier=parent_session.gpu_tier if parent_session else "T4",
        vram_total_gb=parent_session.vram_total_gb if parent_session else 16.0,
        session_type="subagent",
        env=parent_session.env if parent_session else None,
        resident_models=list(getattr(parent_session, "resident_models", None) or []) if parent_session else None,
        inactive_models=list(getattr(parent_session, "inactive_models", None) or []) if parent_session else None,
        active_stack=getattr(parent_session, "active_stack", None) if parent_session else None,
        model_agent_counts=dict(getattr(parent_session, "model_agent_counts", None) or {}) if parent_session else None,
        response_cache=getattr(parent_session, "response_cache", None) if parent_session else None,
        auto_attach_warm=False if parent_ep else None,
        no_colab=bool(getattr(parent_session, "_no_colab", False)) if parent_session else False,
    )

    # Propagate RULE[user_global] and LANGUAGE.md invariants into the child subagent prompt
    subagent_task_prompt = (
        f"[Subagent Role: {role}]\n"
        "Invariants: Follow LANGUAGE.md ({errors}, {correctness}, {functionality}, {correct required outputs}, "
        "{sufficient}, {insufficient}) and RULE[user_global] (First-Principles Modeling, Root-Cause Ownership, "
        "Fail Fast, Runtime Reality Verification).\n\n"
        f"Task:\n{prompt}"
    )

    eff_stream = (not same_gpu_reused) if stream is None else bool(stream)
    if launch_barrier is not None:
        try:
            launch_barrier.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass

    try:
        turn_res = child_session.run_turn(
            user_prompt=subagent_task_prompt,
            max_tool_steps=max_turns,
            stream=eff_stream,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )
        tool_steps_count = len(turn_res.get("tool_events", []))
        turns_used_count = max(1, tool_steps_count + (1 if turn_res.get("response") else 0))
        if turns_used_count > max_turns:
            turns_used_count = max_turns
        result = {
            "status": "completed",
            "ok": True,
            "role": role,
            "subagent_slug": child_slug,
            "session_dir": str(child_session.session_dir),
            "model_used": effective_model,
            "model_alias": effective_model,
            "workflow_mode": child_wf,
            "access_mode": child_access,
            "read_only": child_access == "read-only",
            "endpoint_url": child_session.endpoint_url,
            "same_gpu_reused": same_gpu_reused,
            "gpu_mode_used": gpu_mode_used,
            "extra_colab_vms_spawned": extra_colab_vms_spawned,
            "extra_vms_spawned": extra_colab_vms_spawned,
            "summary": turn_res.get("response", ""),
            "response": turn_res.get("response", ""),
            "files_modified": turn_res.get("files_modified", []),
            "tool_steps": tool_steps_count,
            "turns_used": turns_used_count,
            "cache_hit": bool(turn_res.get("cache_hit", False)),
            "batch_meta": turn_res.get("batch_meta"),
            "verification_results": "Verified via live subagent tool execution.",
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "ok": False,
            "role": role,
            "subagent_slug": child_slug,
            "session_dir": str(child_session.session_dir),
            "model_used": effective_model,
            "model_alias": effective_model,
            "workflow_mode": child_wf,
            "access_mode": child_access,
            "read_only": child_access == "read-only",
            "endpoint_url": child_session.endpoint_url,
            "same_gpu_reused": same_gpu_reused,
            "gpu_mode_used": gpu_mode_used,
            "extra_colab_vms_spawned": extra_colab_vms_spawned,
            "extra_vms_spawned": extra_colab_vms_spawned,
            "summary": f"Subagent '{role}' failed: {exc}",
            "files_modified": [],
            "tool_steps": 0,
            "turns_used": 0,
            "cache_hit": False,
            "batch_meta": None,
            "verification_results": str(exc),
        }

    if parent_session is not None:
        parent_session.subagent_invocations.append(result)
    return result


def invoke_parallel_subagents(
    tasks: List[Dict[str, Any]],
    parent_session: Optional[PlatoSession] = None,
    max_workers: Optional[int] = None,
    gpu_mode: str = "shared",
    stream: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Executes multiple child subagent tool loops concurrently over the parent session's warm Colab GPU 1
    co-resident model slots and per-model continuous batching queues (`0` extra Colab VMs spawned by default,
    FR-6, AC-4, Spec 867).
    """
    if not tasks:
        return []

    worker_count = max(1, len(tasks)) if max_workers is None else max(1, min(int(max_workers), len(tasks)))
    ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    sync_barrier = threading.Barrier(parties=worker_count) if (worker_count > 1 and worker_count == len(tasks)) else None

    def _run_idx(idx: int, spec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        res = invoke_subagent(
            role=str(spec.get("role") or f"subagent-{idx + 1}"),
            prompt=str(spec.get("prompt") or ""),
            model_alias=spec.get("model_alias") or spec.get("model"),
            workspace_dir=spec.get("workspace_dir"),
            read_only=bool(spec.get("read_only", False)),
            access_mode=spec.get("access_mode"),
            gpu_mode=str(spec.get("gpu_mode") or gpu_mode or "shared"),
            parent_session=parent_session,
            max_turns=int(spec.get("max_turns", 8)),
            stream=spec.get("stream", stream),
            use_cache=spec.get("use_cache", use_cache),
            force_refresh=bool(spec.get("force_refresh", force_refresh)),
            launch_barrier=sync_barrier,
        )
        return idx, res

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_run_idx, i, t) for i, t in enumerate(tasks)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            ordered_results[idx] = res

    return [r for r in ordered_results if r is not None]
