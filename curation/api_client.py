"""
Authoritative API Client and SDK Wrappers for Sandcastle Distillation Suite.
Document ID: api_client_847_v1
Authoring Agent: implementer-3 (Stack, API, MCP & Library Usage Implementer)
Governing Evaluation Standard: RULE[user_global] (Strict contract fidelity, fail-fast, zero silent swallows)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("curation.api_client")

_ORIG_STDLIB_URLOPEN = urllib.request.urlopen
_SHARED_HTTP_CLIENT_LOCK = threading.Lock()
_SHARED_HTTP_CLIENT: Any = None


class _PooledHttpResponseAdapter:
    """Context-manager response wrapper matching urllib.response.addinfourl over pooled httpx."""

    def __init__(self, status: int, body_bytes: bytes, headers: Dict[str, str], url: str) -> None:
        self.status = int(status)
        self.code = int(status)
        self._body = body_bytes
        self._stream = io.BytesIO(body_bytes)
        self.headers = headers
        self.url = url

    def read(self, amt: Optional[int] = None) -> bytes:
        return self._stream.read() if amt is None else self._stream.read(amt)

    def readline(self, limit: int = -1) -> bytes:
        return self._stream.readline(limit)

    def readlines(self, hint: int = -1) -> List[bytes]:
        return self._stream.readlines(hint)

    def __iter__(self) -> Any:
        return iter(self._stream)

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "_PooledHttpResponseAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
        return None


def _get_shared_http_client() -> Any:
    """Returns a process-wide keep-alive httpx.Client for low-latency connection reuse."""
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is not None:
        return _SHARED_HTTP_CLIENT
    with _SHARED_HTTP_CLIENT_LOCK:
        if _SHARED_HTTP_CLIENT is None:
            try:
                import httpx

                _SHARED_HTTP_CLIENT = httpx.Client(
                    limits=httpx.Limits(
                        max_connections=64,
                        max_keepalive_connections=32,
                        keepalive_expiry=60.0,
                    ),
                    follow_redirects=True,
                )
            except Exception:
                _SHARED_HTTP_CLIENT = False
        return _SHARED_HTTP_CLIENT


def _open_http_request(req: urllib.request.Request, timeout: float = 10.0) -> Any:
    """Executes an HTTP request using pooled keep-alive httpx when urllib.request.urlopen is unpatched."""
    if urllib.request.urlopen is not _ORIG_STDLIB_URLOPEN:
        return urllib.request.urlopen(req, timeout=timeout)

    headers = dict(req.header_items())
    accept_hdr = str(headers.get("Accept") or headers.get("accept") or "").lower()
    content = req.data
    if "text/event-stream" in accept_hdr or (
        isinstance(content, (bytes, bytearray)) and (b'"stream": true' in content or b'"stream":true' in content)
    ):
        return urllib.request.urlopen(req, timeout=timeout)

    client = _get_shared_http_client()
    if not client:
        return urllib.request.urlopen(req, timeout=timeout)

    import httpx

    url = req.full_url
    method = req.get_method()
    try:
        resp = client.request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            timeout=float(timeout),
        )
    except httpx.TimeoutException as exc:
        raise socket.timeout(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise urllib.error.URLError(str(exc)) from exc

    resp_headers = {str(k): str(v) for k, v in resp.headers.items()}
    if resp.status_code >= 400:
        fp = io.BytesIO(resp.content)
        raise urllib.error.HTTPError(
            url=url,
            code=int(resp.status_code),
            msg=resp.reason_phrase or f"HTTP {resp.status_code}",
            hdrs=resp.headers,  # type: ignore[arg-type]
            fp=fp,
        )
    return _PooledHttpResponseAdapter(
        status=int(resp.status_code),
        body_bytes=resp.content,
        headers=resp_headers,
        url=url,
    )

BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
UNIFIED_MODEL_ALIAS = "qwen3.8-27b"
GEMINI_HARNESS_MODEL = "gemini-2.5-pro"

SUPPORTED_HERMES_MODELS: List[str] = [
    "Qwen/Qwen3.5-9B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

QWEN35_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
]

QWEN35_FAST_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LEGACY_QWEN25_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_LINEAR_ATTN_RECURRENT_LEAVES = {
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
}

_EXCLUDED_MODULE_SUBSTRINGS = (
    "visual",
    "vision_tower",
    "vision_model",
    "mm_projector",
    "multi_modal_projector",
    "audio_tower",
)

_EXCLUDED_LEAF_NAMES = {
    "lm_head",
    "embed_tokens",
    "wte",
    "wpe",
    "score",
    "in_proj_a",
    "in_proj_b",
}

_KNOWN_PROJECTION_LEAVES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "wqkv",
    "wo",
    "w1",
    "w2",
    "w3",
    "c_attn",
    "c_proj",
    "fc1",
    "fc2",
}


def discover_qlora_linear_modules(
    model: Any,
    include_linear_attn: Optional[bool] = None,
) -> List[str]:
    """Inspect torch.nn.Linear modules on any Hermes/HF model (excluding lm_head and vision towers).

    Adapts all attention, linear-attention, and MLP projection layers automatically without
    manual per-architecture module lists (Spec 851 FR-3.3), while supporting Spec 852 FR-3.1
    high-throughput 7-module filtering (include_linear_attn=False).
    """
    if include_linear_attn is None:
        env_toggle = os.environ.get("SANDCASTLE_INCLUDE_LINEAR_ATTN_LORA", "").strip().lower()
        if env_toggle in ("0", "false", "no"):
            include_linear_attn = False
        elif env_toggle in ("1", "true", "yes"):
            include_linear_attn = True
        else:
            include_linear_attn = True

    default_fallback = list(QWEN35_TARGET_MODULES) if include_linear_attn else list(QWEN35_FAST_TARGET_MODULES)
    if model is None or not hasattr(model, "named_modules"):
        return default_fallback

    try:
        import torch.nn as nn
        linear_base_types: Tuple[type, ...] = (nn.Linear,)
    except Exception:
        linear_base_types = ()

    discovered: set[str] = set()
    for full_name, module in model.named_modules():
        if not full_name:
            continue
        lower_path = full_name.lower()
        if any(excl in lower_path for excl in _EXCLUDED_MODULE_SUBSTRINGS):
            continue
        parts = full_name.split(".")
        leaf = parts[-1]
        if leaf in _EXCLUDED_LEAF_NAMES or "lm_head" in parts:
            continue
        if not include_linear_attn and (
            leaf in _LINEAR_ATTN_RECURRENT_LEAVES or "linear_attn" in lower_path
        ):
            continue

        cls_name = module.__class__.__name__
        is_linear = bool(linear_base_types and isinstance(module, linear_base_types))
        if not is_linear and cls_name.endswith(("Linear", "Linear4bit", "Linear8bitLt")):
            is_linear = True
        if not is_linear:
            weight = getattr(module, "weight", None)
            shape = getattr(weight, "shape", None)
            if shape is not None and len(shape) == 2 and (
                leaf.endswith("_proj") or leaf in _KNOWN_PROJECTION_LEAVES
            ):
                is_linear = True

        if is_linear:
            discovered.add(leaf)

    return sorted(discovered) if discovered else default_fallback


def select_qwen35_precision_tier(
    total_vram_gb: float,
    force_4bit: Optional[bool] = None,
) -> Dict[str, Any]:
    """Selects hardware-aware precision tier for Qwen/Qwen3.5-9B fine-tuning (Spec 852 FR-2.1).

    - High-VRAM Tier (>= 38.0 GB, e.g., A100 40GB/80GB, H100): Native bfloat16 (load_in_4bit=False)
      to eliminate 100% of 4-bit NF4 dequantization overhead.
    - Constrained-VRAM Tier (15.0 <= VRAM < 38.0 GB, e.g., L4 24GB, V100/T4 16GB): 4-bit NF4
      quantization with double quant and bfloat16 compute dtype.
    - < 15.0 GB: Fails fast with RuntimeError per RULE[user_global].
    """
    if total_vram_gb < 14.0:
        raise RuntimeError(
            f"Insufficient GPU VRAM ({total_vram_gb:.2f} GB < 14.0 GB minimum) for Qwen/Qwen3.5-9B fine-tuning."
        )

    if force_4bit is None:
        env_force_4bit = os.environ.get("SANDCASTLE_FORCE_4BIT", "").strip().lower()
        env_force_bf16 = os.environ.get("SANDCASTLE_FORCE_BF16", "").strip().lower()
        if env_force_4bit in ("1", "true", "yes"):
            force_4bit = True
        elif env_force_bf16 in ("1", "true", "yes"):
            force_4bit = False
        else:
            force_4bit = total_vram_gb < 38.0

    if not force_4bit and total_vram_gb >= 38.0:
        return {
            "precision_tier": "bfloat16_native",
            "load_in_4bit": False,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "total_vram_gb": float(total_vram_gb),
        }

    return {
        "precision_tier": "nf4_4bit",
        "load_in_4bit": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "total_vram_gb": float(total_vram_gb),
    }


MAX_PARALLEL_AGENTS: int = 12
MAX_MODEL_LEN: int = 16384
MAX_NUM_BATCHED_TOKENS: int = 16384
DEFAULT_COLAB_ENDPOINT_FILENAME: str = "colab_qwen_endpoint.json"
DEFAULT_GPU_ROUTING_FILENAME: str = "colab_gpu_routing.json"

PIPELINE_STAGE_AGENT_GROUPS: Dict[str, List[str]] = {
    "stage1-discovery": [
        "codebase-mapper",
        "api-schema-mapper",
        "mcp-provisioner",
        "library-installer",
        "function-mapper",
        "spec-refiner",
    ],
    "stage2-planning": [
        "planning-coordinator",
        "architecture-planner",
        "storage-planner",
        "stack-planner",
        "optimization-planner",
        "visual-planner",
        "implementation-coordinator",
    ],
    "stage3-implementation": [
        "architecture-implementer",
        "storage-implementer",
        "stack-implementer",
        "optimization-implementer",
        "visual-implementer",
        "verification-coordinator",
    ],
    "stage4-testing": [
        "architecture-tester",
        "storage-tester",
        "stack-tester",
        "optimization-tester",
        "visual-tester",
        "e2e-validator",
    ],
    "stage5-merger": ["branch-merger"],
}


def _estimate_model_kv_and_state_footprint(
    model_alias_or_config: Any,
    *,
    max_model_len: int = MAX_MODEL_LEN,
    kv_cache_dtype: str = "auto",
) -> Dict[str, Any]:
    """Computes exact first-principles KV-cache and GDN/MLA recurrent state footprint.

    Delegates to platodoom.catalog.estimate_model_kv_and_state_footprint when present,
    with a deterministic first-principles fallback matching Qwen3.5 (head_dim=256,
    float32 SSM recurrent state + fp16/bf16 conv state), Qwen3/Qwen2 (head_dim=128),
    and DeepSeek-V2-Lite MLA (kv_lora_rank=512, qk_rope_head_dim=64).
    """
    try:
        from platodoom.catalog import estimate_model_kv_and_state_footprint as _cat_est  # type: ignore[attr-defined]

        return _cat_est(
            model_alias_or_config,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
        )
    except Exception:
        pass

    alias = (
        str(model_alias_or_config.get("model_alias") or model_alias_or_config.get("alias") or "")
        if isinstance(model_alias_or_config, dict)
        else str(model_alias_or_config or "")
    ).strip().lower()

    elem_bytes = 1 if str(kv_cache_dtype or "auto").strip().lower().startswith("fp8") else 2
    seq_len = max(1, int(max_model_len))

    # DeepSeek-V2-Lite MLA: 27 layers * (512 + 64) * elem_bytes
    if "deepseek-v2-lite" in alias:
        kv_bpt = float(27 * (512 + 64) * elem_bytes)
        kv_gb = (kv_bpt * seq_len) / float(1024 ** 3)
        return {
            "kv_bytes_per_token": kv_bpt,
            "kv_gb_per_seq_at_max_len": round(kv_gb, 6),
            "gdn_bytes_per_seq": 0.0,
            "gdn_gb_per_seq": 0.0,
            "is_hybrid_gdn": False,
            "is_mla": True,
        }

    # Qwen3.5 / Qwen3.6 / Qwen3.8 hybrid GatedDeltaNet + full-attention (head_dim=256)
    gdn_profiles: Dict[str, Tuple[int, int, int, int, int, int, int]] = {
        # (full_attn_layers, kv_heads, head_dim, gdn_layers, v_heads, k_dim, v_dim)
        "qwen-9b": (8, 4, 256, 24, 32, 128, 128),
        "qwen-27b": (16, 4, 256, 48, 48, 128, 128),
        "qwen3.8-27b": (16, 4, 256, 48, 48, 128, 128),
        "qwen-27b-38": (16, 4, 256, 48, 48, 128, 128),
        "qwen-35b-moe": (10, 2, 256, 30, 32, 128, 128),
        "qwen-122b-moe": (12, 2, 256, 36, 64, 128, 128),
    }
    for key, (l_full, n_kv, d_head, l_gdn, n_v, d_k, d_v) in gdn_profiles.items():
        if alias == key or key in alias:
            kv_bpt = float(l_full * 2 * n_kv * d_head * elem_bytes)
            kv_gb = (kv_bpt * seq_len) / float(1024 ** 3)
            conv_dim = 2 * (16 * d_k) + (n_v * d_v)
            gdn_bps = float(l_gdn * ((n_v * d_k * d_v * 4) + (conv_dim * 3 * 2)))
            gdn_gb = gdn_bps / float(1024 ** 3)
            return {
                "kv_bytes_per_token": kv_bpt,
                "kv_gb_per_seq_at_max_len": round(kv_gb, 6),
                "gdn_bytes_per_seq": gdn_bps,
                "gdn_gb_per_seq": round(gdn_gb, 6),
                "is_hybrid_gdn": True,
                "is_mla": False,
            }

    # Standard GQA models (head_dim=128)
    gqa_profiles: Dict[str, Tuple[int, int, int]] = {
        "r1-7b": (28, 4, 128),
        "qwen-4b": (36, 8, 128),
        "qwen-8b": (36, 8, 128),
        "r1-14b": (48, 8, 128),
        "qwen-14b": (40, 8, 128),
        "r1-32b": (64, 8, 128),
        "qwen-30b-moe": (48, 4, 128),
        "qwen-coder-30b-moe": (48, 4, 128),
    }
    l_full, n_kv, d_head = gqa_profiles.get(alias, (36, 8, 128))
    kv_bpt = float(l_full * 2 * n_kv * d_head * elem_bytes)
    kv_gb = (kv_bpt * seq_len) / float(1024 ** 3)
    return {
        "kv_bytes_per_token": kv_bpt,
        "kv_gb_per_seq_at_max_len": round(kv_gb, 6),
        "gdn_bytes_per_seq": 0.0,
        "gdn_gb_per_seq": 0.0,
        "is_hybrid_gdn": False,
        "is_mla": False,
    }


def _compute_dynamic_max_num_seqs(
    free_vram_gb: float,
    weights_gb: float,
    *,
    model_alias: str = "",
    max_model_len: int = MAX_MODEL_LEN,
    kv_cache_dtype: str = "auto",
    requested_agents: int = 1,
    active_slot_count: int = 1,
) -> int:
    """Computes dynamic vLLM continuous-batching max_num_seqs from live KV + GDN/MLA headroom."""
    try:
        from platodoom.catalog import compute_dynamic_max_num_seqs as _cat_dyn  # type: ignore[attr-defined]

        return int(
            _cat_dyn(
                free_vram_gb,
                weights_gb,
                model_alias=model_alias,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
                requested_agents=requested_agents,
                active_slot_count=active_slot_count,
            )
        )
    except Exception:
        pass

    footprint = _estimate_model_kv_and_state_footprint(
        model_alias or UNIFIED_MODEL_ALIAS,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
    )
    slots = max(1, int(active_slot_count))
    slot_vram_budget = (float(free_vram_gb) * (0.88 if slots == 1 else 0.90)) / float(slots)
    kv_pool_gb = max(0.5, slot_vram_budget - float(weights_gb) / float(slots) - 1.50) if slot_vram_budget > float(weights_gb) / float(slots) + 1.50 else max(0.5, float(free_vram_gb) / float(slots) - 1.20)
    # Average active sequence length under prefix caching (~4096 tokens) + full per-seq GDN state
    avg_seq_Ratio = min(1.0, 4096.0 / float(max(1, int(max_model_len))))
    per_seq_gb = max(
        0.02,
        float(footprint["kv_gb_per_seq_at_max_len"]) * avg_seq_Ratio + float(footprint["gdn_gb_per_seq"]),
    )
    computed = max(1, int(kv_pool_gb // per_seq_gb))
    if computed < 16:
        return max(1, min(256, computed))
    floor_seqs = min(256, max(16, int(requested_agents)))
    return min(256, max(floor_seqs, computed))


def is_platodoom_model_id(
    model_id: Optional[str],
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Returns True if `model_id` is any platodoom-hosted model alias (catalog, unified, or custom), not a cloud agy model."""
    raw = str(model_id or "").strip()
    if not raw:
        return False
    base = raw.split("@", 1)[0].strip() if ("@" in raw and not Path(raw).expanduser().exists()) else raw
    lower = base.lower()
    try:
        from platodoom.catalog import get_model_catalog

        catalog = get_model_catalog(config)
        if base in catalog:
            return True
        if any(k.lower() == lower for k in catalog):
            return True
    except Exception:
        pass
    return lower in _KNOWN_DISTINCT_ALIASES


def route_gpu_for_model_count(
    num_models: Optional[int] = None,
    peak_stage_models: Optional[int] = None,
    stage_models: Optional[Dict[str, str]] = None,
    gpu_override: Optional[str] = None,
    enabled_agents: Optional[List[str]] = None,
    repo_dir: Optional[Path | str] = None,
    stack_preset: Optional[str] = None,
    model_alias: Optional[str] = None,
    stack_models: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Routes GPU type, GPU VRAM (GB), System RAM (GB), precision tier, and parallel batch slots based on enabled platodoom/Colab models (Spec 854, Spec 856, Spec 872)."""
    per_stage_counts: Dict[str, int] = {}
    stage_agent_counts: Dict[str, int] = {}
    enabled_set = set(enabled_agents) if enabled_agents is not None else None

    try:
        from platodoom.catalog import (
            GPU_TIER_SPECS as _SSOT_GPU_TIER_SPECS,
            get_model_catalog as _get_ssot_catalog,
            normalize_hardware_tier,
            resolve_hardware_stack,
            resolve_model_profile,
            resolve_multi_model_agent_allocation,
        )
    except ImportError:
        _SSOT_GPU_TIER_SPECS = None  # type: ignore
        _get_ssot_catalog = None  # type: ignore
        normalize_hardware_tier = None  # type: ignore
        resolve_hardware_stack = None  # type: ignore
        resolve_model_profile = None  # type: ignore
        resolve_multi_model_agent_allocation = None  # type: ignore

    def _record_stage_platodoom_model(agent_name: str, raw_val: str) -> bool:
        if not is_platodoom_model_id(raw_val):
            return False
        base_name = raw_val.split("@", 1)[0].strip() if "@" in raw_val else raw_val.strip()
        canonical_key = raw_val.strip()
        if callable(resolve_model_profile):
            try:
                prof = resolve_model_profile(base_name)
                if str(prof.get("modality") or "text").lower() == "image":
                    raise ValueError(
                        f"Auxiliary image model '{raw_val}' assigned to text Blackboard agent '{agent_name}' "
                        f"cannot execute Blackboard coding tool loops."
                    )
                prof_alias = str(prof.get("alias") or base_name)
                if "@" in raw_val:
                    _, prec_part = raw_val.split("@", 1)
                    canonical_key = f"{prof_alias}@{prec_part.strip()}"
                else:
                    canonical_key = prof_alias
            except ValueError:
                raise
            except Exception:
                pass
        stage_agent_counts[canonical_key] = stage_agent_counts.get(canonical_key, 0) + 1
        return True

    if stage_models is not None:
        total_qwen = 0
        peak_stage = 0
        known_agents: set[str] = set()
        for stage_id, agents in PIPELINE_STAGE_AGENT_GROUPS.items():
            c = 0
            for ag in agents:
                known_agents.add(ag)
                if enabled_set is not None and ag not in enabled_set:
                    continue
                val = str(stage_models.get(ag, "") or "").strip()
                if _record_stage_platodoom_model(ag, val):
                    c += 1
                    total_qwen += 1
            per_stage_counts[stage_id] = c
            if c > peak_stage:
                peak_stage = c
        for ag, val_raw in stage_models.items():
            if ag not in known_agents:
                if enabled_set is not None and ag not in enabled_set:
                    continue
                val = str(val_raw or "").strip()
                if _record_stage_platodoom_model(ag, val):
                    total_qwen += 1
                    if peak_stage == 0:
                        peak_stage = 1
        n_total = total_qwen if num_models is None else max(0, int(num_models))
        n_peak = peak_stage if peak_stage_models is None else max(0, int(peak_stage_models))
    else:
        n_total = max(0, int(num_models or 0))
        if peak_stage_models is not None:
            n_peak = max(0, int(peak_stage_models))
        else:
            n_peak = min(n_total, 7)

    if model_alias and callable(resolve_model_profile):
        base_ma = str(model_alias).split("@", 1)[0].strip() if "@" in str(model_alias) else str(model_alias).strip()
        try:
            ma_prof = resolve_model_profile(base_ma)
            if str(ma_prof.get("modality") or "text").lower() == "image":
                raise ValueError(
                    f"Auxiliary image model '{model_alias}' cannot execute text/coding Blackboard tool loops."
                )
        except ValueError:
            raise
        except Exception:
            pass

    if callable(normalize_hardware_tier):
        norm_override = normalize_hardware_tier(gpu_override, allow_auto=True)
    else:
        raw_ovr = str(gpu_override or "auto").strip()
        if raw_ovr.lower() in ("", "auto"):
            norm_override = "auto"
        elif raw_ovr in ("G4", "H100", "A100-80GB", "A100", "v6e-1", "L4", "T4", "v5e-1"):
            norm_override = raw_ovr
        else:
            raise ValueError(
                f"Unsupported Colab hardware tier {gpu_override!r}. Valid tiers: G4, H100, A100-80GB, A100, v6e-1, L4, T4, v5e-1."
            )

    active_alloc_res: Optional[Dict[str, Any]] = None
    active_model_aliases: List[str] = []
    if n_total > 0 or stack_models or (stage_agent_counts and norm_override != "auto"):
        alloc_input: Any = None
        if stage_agent_counts:
            alloc_input = dict(stage_agent_counts)
        elif stack_models and not stack_preset:
            alloc_input = list(stack_models)
        elif model_alias and str(model_alias).strip():
            alloc_input = {str(model_alias).strip(): max(1, n_total)}

        if alloc_input is not None and callable(resolve_multi_model_agent_allocation):
            active_alloc_res = resolve_multi_model_agent_allocation(
                alloc_input,
                gpu_tier=norm_override,
            )
            active_model_aliases = [str(m["alias"]) for m in active_alloc_res.get("active_models", [])]

    norm_alias = str(model_alias or (active_model_aliases[0] if active_model_aliases else "") or "").strip().lower()
    use_prequant_tier = bool(norm_alias)
    low_vram_precision_tier = "gptq_4bit" if use_prequant_tier else "nf4_4bit"

    def _enrich_dual_schema(base_profile: Dict[str, Any]) -> Dict[str, Any]:
        accel_type = str(base_profile.setdefault("accelerator_type", "GPU" if base_profile["gpu_type"] != "None" else "None"))
        mem_type = str(base_profile.setdefault("memory_type", "VRAM" if accel_type == "GPU" else ("HBM" if accel_type == "TPU" else "None")))
        colab_accel = str(base_profile.setdefault("colab_accelerator", accel_type))
        colab_tpu = base_profile.setdefault("colab_tpu_type", None)

        if active_alloc_res is not None and base_profile["gpu_type"] != "None" and not stack_preset:
            base_profile["stack_mode"] = active_alloc_res["stack_mode"]
            base_profile["stack_preset"] = None
            base_profile["model_count"] = active_alloc_res["active_model_count"]
            base_profile["stack_models"] = active_alloc_res["active_models"]
            base_profile["estimated_weights_gb"] = active_alloc_res["estimated_weights_gb"]
            base_profile["kv_headroom_gb"] = active_alloc_res["free_kv_gb"]
            base_profile["agent_counts"] = active_alloc_res["agent_counts"]
            base_profile["active_models"] = active_model_aliases
            base_profile["model"] = active_model_aliases[0] if active_model_aliases else UNIFIED_MODEL_ALIAS
            base_profile["model_alias"] = base_profile["model"]
        elif base_profile["gpu_type"] != "None" and callable(resolve_hardware_stack) and (
            stack_preset or stack_models or model_alias
        ):
            resolved_st = resolve_hardware_stack(
                base_profile["gpu_type"],
                stack_preset=stack_preset,
                model_alias=model_alias or UNIFIED_MODEL_ALIAS,
                stack_models=stack_models,
            )
            base_profile["stack_mode"] = resolved_st["stack_mode"]
            base_profile["stack_preset"] = resolved_st["stack_preset"]
            base_profile["model_count"] = resolved_st["model_count"]
            base_profile["stack_models"] = resolved_st["models"]
            base_profile["estimated_weights_gb"] = resolved_st["total_weights_gb"]
            base_profile["kv_headroom_gb"] = resolved_st["free_kv_gb"]
            base_profile["active_models"] = [str(m.get("alias", "")) for m in resolved_st["models"] if m.get("alias")]
            base_profile["model"] = model_alias or (base_profile["active_models"][0] if base_profile["active_models"] else UNIFIED_MODEL_ALIAS)
            base_profile["model_alias"] = base_profile["model"]
        else:
            base_profile.setdefault("stack_mode", "cloud_only" if base_profile["gpu_type"] == "None" else "solo")
            base_profile.setdefault("stack_preset", None if base_profile["gpu_type"] == "None" else stack_preset)
            base_profile.setdefault("model_count", 0 if base_profile["gpu_type"] == "None" else 1)
            base_profile.setdefault("stack_models", [])
            base_profile.setdefault("estimated_weights_gb", 0.0 if base_profile["gpu_type"] == "None" else 5.0)
            base_profile.setdefault(
                "kv_headroom_gb",
                0.0 if base_profile["gpu_type"] == "None" else round(max(4.0, float(base_profile["vram_gb"]) - 5.0), 2),
            )
            base_profile.setdefault("agent_counts", dict(stage_agent_counts))
            base_profile.setdefault("active_models", [] if base_profile["gpu_type"] == "None" else [model_alias or UNIFIED_MODEL_ALIAS])
            base_profile.setdefault("model", None if base_profile["gpu_type"] == "None" else (model_alias or UNIFIED_MODEL_ALIAS))
            base_profile.setdefault("model_alias", base_profile["model"])

        if active_model_aliases and callable(resolve_model_profile):
            try:
                lens: List[int] = []
                for a in active_model_aliases:
                    p = resolve_model_profile(a)
                    if base_profile["gpu_type"] in ("A100", "A100-80GB", "H100", "G4", "v6e-1"):
                        lens.append(int(p.get("max_model_len_a100", p.get("max_model_len", MAX_MODEL_LEN))))
                    else:
                        lens.append(int(p.get("max_model_len", MAX_MODEL_LEN)))
                if lens:
                    base_profile["max_model_len"] = min(lens)
            except Exception:
                pass

        base_profile.setdefault("total_platodoom_models", base_profile["total_qwen_models"])
        base_profile.update(
            {
                "tierId": base_profile["tier_id"],
                "gpuType": base_profile["gpu_type"],
                "acceleratorType": accel_type,
                "memoryType": mem_type,
                "colabAccelerator": colab_accel,
                "colabGpuType": base_profile["colab_gpu_type"],
                "colabTpuType": colab_tpu,
                "gpuLabel": base_profile["gpu_label"],
                "vramGb": base_profile["vram_gb"],
                "systemRamGb": base_profile["system_ram_gb"],
                "colabMachineShape": base_profile["colab_machine_shape"],
                "precisionTier": base_profile["precision_tier"],
                "loadIn4Bit": base_profile["load_in_4bit"],
                "maxNumSeqs": base_profile["max_num_seqs"],
                "maxParallelAgents": base_profile["max_parallel_agents"],
                "maxModelLen": base_profile["max_model_len"],
                "maxNumBatchedTokens": base_profile["max_num_batched_tokens"],
                "estimatedPeakVramGib": base_profile["estimated_peak_vram_gib"],
                "totalQwenModels": base_profile["total_qwen_models"],
                "totalPlatodoomModels": base_profile["total_platodoom_models"],
                "peakStageQwenModels": base_profile["peak_stage_qwen_models"],
                "perStageCounts": base_profile["per_stage_counts"],
                "overrideMode": base_profile["override_mode"],
                "stackMode": base_profile["stack_mode"],
                "stackPreset": base_profile["stack_preset"],
                "modelCount": base_profile["model_count"],
                "stackModels": base_profile["stack_models"],
                "activeModels": base_profile["active_models"],
                "agentCounts": base_profile.get("agent_counts", {}),
                "modelAlias": base_profile["model_alias"],
                "estimatedWeightsGb": base_profile["estimated_weights_gb"],
                "kvHeadroomGb": base_profile["kv_headroom_gb"],
                "coResidencyValid": True,
            }
        )
        if repo_dir is not None:
            proj = resolve_project_root(repo_dir)
            base_profile["project_root"] = str(proj)
            base_profile["dir_slug"] = compute_dir_slug(proj)
        return base_profile

    if n_total == 0 and norm_override == "auto" and not stack_preset and not stack_models:
        return _enrich_dual_schema(
            {
                "tier_id": "tier0_cloud_only",
                "gpu_type": "None",
                "colab_gpu_type": "None",
                "gpu_label": "None (Cloud API Only)",
                "vram_gb": 0.0,
                "system_ram_gb": 0.0,
                "colab_machine_shape": "standard",
                "precision_tier": "cloud_api",
                "load_in_4bit": False,
                "max_num_seqs": 0,
                "max_parallel_agents": 0,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": 0,
                "estimated_peak_vram_gib": 0.0,
                "total_qwen_models": 0,
                "peak_stage_qwen_models": 0,
                "per_stage_counts": per_stage_counts,
                "override_mode": "auto",
                "rationale": "n=0 platodoom/Qwen models assigned — all enabled tasks route through cloud APIs; no Colab GPU required.",
            }
        )

    if norm_override != "auto":
        selected_tier = norm_override
    else:
        if n_total <= 2 and n_peak <= 2:
            concurrency_tier = "T4"
        elif n_total <= 5 and n_peak <= 4:
            concurrency_tier = "L4"
        elif n_total <= 12:
            concurrency_tier = "A100"
        else:
            concurrency_tier = "A100-80GB"

        selected_tier = concurrency_tier
        if active_alloc_res is not None and isinstance(_SSOT_GPU_TIER_SPECS, dict):
            model_tier = str(active_alloc_res.get("hardware_tier") or "T4")
            conc_rank = int((_SSOT_GPU_TIER_SPECS.get(concurrency_tier) or {}).get("rank", 2))
            mod_rank = int((_SSOT_GPU_TIER_SPECS.get(model_tier) or {}).get("rank", 2))
            if mod_rank > conc_rank:
                selected_tier = model_tier
            elif conc_rank > mod_rank and callable(resolve_multi_model_agent_allocation):
                # Re-evaluate stack weights/precision on the concurrency-escalated tier
                alloc_input_re = dict(stage_agent_counts) if stage_agent_counts else (list(stack_models) if stack_models else {str(model_alias): max(1, n_total)})
                try:
                    active_alloc_res = resolve_multi_model_agent_allocation(alloc_input_re, gpu_tier=selected_tier)
                except ValueError:
                    selected_tier = model_tier

    if selected_tier == "v5e-1":
        max_seqs = 4
        est_peak = round(6.20 + max_seqs * (0.25 + 0.58 / 12.0) + 2.20, 2)
        return _enrich_dual_schema(
            {
                "tier_id": "tier1_v5e_1_16gb",
                "gpu_type": "v5e-1",
                "accelerator_type": "TPU",
                "memory_type": "HBM",
                "colab_accelerator": "TPU",
                "colab_gpu_type": None,
                "colab_tpu_type": "v5e-1",
                "gpu_label": "Google Cloud TPU v5e-1 (16GB HBM · 32.0GB High-RAM)",
                "vram_gb": 16.0,
                "system_ram_gb": 32.0,
                "colab_machine_shape": "hm",
                "precision_tier": "int4_quantized",
                "load_in_4bit": True,
                "max_num_seqs": max_seqs,
                "max_parallel_agents": MAX_PARALLEL_AGENTS,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": 8192,
                "estimated_peak_vram_gib": est_peak,
                "total_qwen_models": n_total,
                "peak_stage_qwen_models": n_peak,
                "per_stage_counts": per_stage_counts,
                "override_mode": norm_override,
                "rationale": f"n={n_total} Qwen model(s) ({n_peak} peak stage) -> Routed to Google Cloud TPU v5e-1 (16GB HBM, 32.0GB System RAM, Q4 quantized, {max_seqs} active batch slots @ 16k).",
            }
        )

    if selected_tier == "T4":
        max_seqs = 4
        est_peak = round(6.20 + max_seqs * (0.25 + 0.58 / 12.0) + 2.40, 2)
        return _enrich_dual_schema(
            {
                "tier_id": "tier1_t4_16gb",
                "gpu_type": "T4",
                "colab_gpu_type": "T4",
                "gpu_label": "NVIDIA T4 (16GB VRAM · 25.5GB High-RAM)",
                "vram_gb": 16.0,
                "system_ram_gb": 25.5,
                "colab_machine_shape": "hm",
                "precision_tier": low_vram_precision_tier,
                "load_in_4bit": True,
                "max_num_seqs": max_seqs,
                "max_parallel_agents": MAX_PARALLEL_AGENTS,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": 8192,
                "estimated_peak_vram_gib": est_peak,
                "total_qwen_models": n_total,
                "peak_stage_qwen_models": n_peak,
                "per_stage_counts": per_stage_counts,
                "override_mode": norm_override,
                "rationale": f"n={n_total} Qwen model(s) ({n_peak} peak stage) -> Routed to NVIDIA T4 (16GB VRAM, 25.5GB System RAM, {low_vram_precision_tier}, {max_seqs} active batch slots @ 16k).",
            }
        )

    if selected_tier == "L4":
        max_seqs = 6
        est_peak = round(6.20 + max_seqs * (0.25 + 0.58 / 12.0) + 2.40, 2)
        return _enrich_dual_schema(
            {
                "tier_id": "tier2_l4_24gb",
                "gpu_type": "L4",
                "colab_gpu_type": "L4",
                "gpu_label": "NVIDIA L4 (24GB VRAM · 53.0GB High-RAM)",
                "vram_gb": 24.0,
                "system_ram_gb": 53.0,
                "colab_machine_shape": "hm",
                "precision_tier": low_vram_precision_tier,
                "load_in_4bit": True,
                "max_num_seqs": max_seqs,
                "max_parallel_agents": MAX_PARALLEL_AGENTS,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": 8192,
                "estimated_peak_vram_gib": est_peak,
                "total_qwen_models": n_total,
                "peak_stage_qwen_models": n_peak,
                "per_stage_counts": per_stage_counts,
                "override_mode": norm_override,
                "rationale": f"n={n_total} Qwen models ({n_peak} peak stage) -> Routed to NVIDIA L4 (24GB VRAM, 53.0GB System RAM, {low_vram_precision_tier}, {max_seqs} active batch slots @ 16k).",
            }
        )

    if selected_tier == "v6e-1":
        max_seqs = 8
        est_peak = round(16.80 + max_seqs * 0.25 + 0.58 + 3.00, 2)
        return _enrich_dual_schema(
            {
                "tier_id": "tier2b_v6e_1_32gb",
                "gpu_type": "v6e-1",
                "accelerator_type": "TPU",
                "memory_type": "HBM",
                "colab_accelerator": "TPU",
                "colab_gpu_type": None,
                "colab_tpu_type": "v6e-1",
                "gpu_label": "Google Cloud TPU v6e-1 Trillium (32GB HBM · 64.0GB High-RAM)",
                "vram_gb": 32.0,
                "system_ram_gb": 64.0,
                "colab_machine_shape": "hm",
                "precision_tier": "int4_or_int8",
                "load_in_4bit": True,
                "max_num_seqs": max_seqs,
                "max_parallel_agents": MAX_PARALLEL_AGENTS,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "estimated_peak_vram_gib": est_peak,
                "total_qwen_models": n_total,
                "peak_stage_qwen_models": n_peak,
                "per_stage_counts": per_stage_counts,
                "override_mode": norm_override,
                "rationale": f"n={n_total} Qwen models ({n_peak} peak stage) -> Routed to Google Cloud TPU v6e-1 Trillium (32GB HBM, 64.0GB System RAM, Q4/INT8, {max_seqs} parallel slots @ 16k).",
            }
        )

    if selected_tier == "A100":
        max_seqs = 12
        est_peak = round(17.55 + max_seqs * 0.25 + 0.58 + 4.20, 2)
        return _enrich_dual_schema(
            {
                "tier_id": "tier3_a100_40gb",
                "gpu_type": "A100",
                "colab_gpu_type": "A100",
                "gpu_label": "NVIDIA A100-SXM4-40GB (40GB VRAM · 83.5GB High-RAM)",
                "vram_gb": 40.0,
                "system_ram_gb": 83.5,
                "colab_machine_shape": "hm",
                "precision_tier": "bfloat16_native",
                "load_in_4bit": False,
                "max_num_seqs": max_seqs,
                "max_parallel_agents": MAX_PARALLEL_AGENTS,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "estimated_peak_vram_gib": est_peak,
                "total_qwen_models": n_total,
                "peak_stage_qwen_models": n_peak,
                "per_stage_counts": per_stage_counts,
                "override_mode": norm_override,
                "rationale": f"n={n_total} Qwen models ({n_peak} peak stage) -> Routed to NVIDIA A100 40GB (40GB VRAM, 83.5GB System RAM, native bfloat16, {max_seqs} parallel slots @ 16k).",
            }
        )

    if selected_tier == "H100":
        max_seqs = 16
        est_peak = round(17.55 + max_seqs * 0.25 + (16.0 / 12.0) * 0.58 + 4.20, 2)
        return _enrich_dual_schema(
            {
                "tier_id": "tier5_h100_80gb",
                "gpu_type": "H100",
                "colab_gpu_type": "H100",
                "gpu_label": "NVIDIA H100 80GB HBM3 (80GB VRAM · 170.0GB High-RAM · Native FP8/BF16)",
                "vram_gb": 80.0,
                "system_ram_gb": 170.0,
                "colab_machine_shape": "hm",
                "precision_tier": "bfloat16_or_fp8_native",
                "load_in_4bit": False,
                "max_num_seqs": max_seqs,
                "max_parallel_agents": max_seqs,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "estimated_peak_vram_gib": est_peak,
                "total_qwen_models": n_total,
                "peak_stage_qwen_models": n_peak,
                "per_stage_counts": per_stage_counts,
                "override_mode": norm_override,
                "rationale": f"n={n_total} Qwen models ({n_peak} peak stage) -> Routed to NVIDIA H100 80GB (80GB VRAM, 170.0GB System RAM, native FP8/BF16, {max_seqs} parallel slots).",
            }
        )

    if selected_tier == "G4":
        max_seqs = 16
        est_peak = round(35.50 + max_seqs * 0.25 + (16.0 / 12.0) * 0.58 + 4.50, 2)
        return _enrich_dual_schema(
            {
                "tier_id": "tier6_g4_96gb",
                "gpu_type": "G4",
                "colab_gpu_type": "G4",
                "gpu_label": "NVIDIA RTX PRO 6000 Blackwell G4 (96GB VRAM · 192.0GB High-RAM · Native FP8/BF16)",
                "vram_gb": 96.0,
                "system_ram_gb": 192.0,
                "colab_machine_shape": "hm",
                "precision_tier": "bfloat16_or_fp8_native",
                "load_in_4bit": False,
                "max_num_seqs": max_seqs,
                "max_parallel_agents": max_seqs,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "estimated_peak_vram_gib": est_peak,
                "total_qwen_models": n_total,
                "peak_stage_qwen_models": n_peak,
                "per_stage_counts": per_stage_counts,
                "override_mode": norm_override,
                "rationale": f"n={n_total} Qwen models ({n_peak} peak stage) -> Routed to NVIDIA G4 96GB (96GB VRAM, 192.0GB System RAM, native FP8/BF16, {max_seqs} parallel slots).",
            }
        )

    max_seqs = 16
    est_peak = round(17.55 + max_seqs * 0.25 + (16.0 / 12.0) * 0.58 + 4.20, 2)
    return _enrich_dual_schema(
        {
            "tier_id": "tier4_a100_80gb",
            "gpu_type": "A100-80GB",
            "colab_gpu_type": "A100",
            "gpu_label": "NVIDIA A100 80GB / H100 (80GB VRAM · 167.0GB High-RAM · A100-40GB 12x Compatible)",
            "vram_gb": 80.0,
            "system_ram_gb": 167.0,
            "colab_machine_shape": "hm",
            "precision_tier": "bfloat16_native",
            "load_in_4bit": False,
            "max_num_seqs": max_seqs,
            "max_parallel_agents": max_seqs,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "estimated_peak_vram_gib": est_peak,
            "total_qwen_models": n_total,
            "peak_stage_qwen_models": n_peak,
            "per_stage_counts": per_stage_counts,
            "override_mode": norm_override,
            "rationale": f"n={n_total} Qwen models ({n_peak} peak stage across multi-stage fleet) -> Routed to NVIDIA A100 80GB / H100 (80GB VRAM, 167.0GB System RAM, native bfloat16, {max_seqs} parallel slots @ 16k; A100 40GB 12-slot compatible).",
        }
    )


def route_gpu_for_multi_model_agents(
    model_allocations: Any,
    *,
    gpu_override: Optional[str] = "auto",
    repo_dir: Optional[Union[str, Path]] = None,
    min_kv_headroom_gb: float = 4.0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Routes GPU 1 for x configured models and n parallel agents (Spec 867 / Spec 869).

    Loads only models with n_i > 0 into GPU 1 VRAM (once per active model);
    models with n_i == 0 are excluded from stack_models and allocated 0.0 GB VRAM.
    Sizes GPU 1 across all 8 Spec 865 hardware tiers (T4, L4, A100-40GB,
    A100-80GB, H100, G4, B200, V6E-1) with dynamic KV-aware max_num_seqs.
    """
    from platodoom.catalog import GPU_TIER_SPECS, resolve_multi_model_agent_allocation

    custom_models = (config or {}).get("custom_models") if isinstance(config, dict) else None
    resolved = resolve_multi_model_agent_allocation(
        model_allocations,
        gpu_tier=gpu_override or "auto",
        min_kv_headroom_gb=min_kv_headroom_gb,
        custom_models=custom_models,
        config=config,
    )

    hw_tier = resolved.get("hardware_tier") or resolved.get("gpu_tier")
    total_agents = int(resolved["total_agents"])
    active_models = list(resolved["active_models"])
    inactive_models = list(resolved["inactive_models"])
    agent_counts = dict(resolved["agent_counts"])
    vram_by_model = dict(resolved["vram_allocated_by_model"])

    if hw_tier is None or (total_agents == 0 and str(gpu_override or "auto").strip().lower() == "auto"):
        base_profile: Dict[str, Any] = {
            "tier_id": "tier0_cloud_only",
            "gpu_type": "None",
            "accelerator_type": "None",
            "memory_type": "None",
            "colab_accelerator": "None",
            "colab_gpu_type": "None",
            "colab_tpu_type": None,
            "gpu_label": "None (Zero Active Agents — 0 GB GPU VRAM)",
            "vram_gb": 0.0,
            "system_ram_gb": 0.0,
            "colab_machine_shape": "standard",
            "precision_tier": "none",
            "load_in_4bit": False,
            "max_num_seqs": 0,
            "max_parallel_agents": 0,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": 0,
            "estimated_peak_vram_gib": 0.0,
            "total_qwen_models": 0,
            "peak_stage_qwen_models": 0,
            "per_stage_counts": {},
            "override_mode": str(gpu_override or "auto"),
            "stack_mode": resolved["stack_mode"],
            "stack_preset": None,
            "model_count": 0,
            "active_model_count": 0,
            "inactive_model_count": len(inactive_models),
            "total_agents": 0,
            "stack_models": [],
            "inactive_models": inactive_models,
            "loaded_models": [],
            "inactive_unloaded_models": list(resolved["inactive_unloaded_models"]),
            "agent_counts": agent_counts,
            "vram_allocated_by_model": vram_by_model,
            "estimated_weights_gb": 0.0,
            "kv_headroom_gb": 0.0,
            "rationale": "All configured models have n_i == 0 active agents; 0 models loaded and 0 GB VRAM required.",
        }
    else:
        tier_spec = GPU_TIER_SPECS[hw_tier]
        is_tpu = tier_spec.get("accelerator_type") == "TPU"
        vram_gb = float(tier_spec["vram_gb"])
        sys_ram_gb = float(tier_spec["system_ram_gb"])
        all_q4 = bool(active_models) and all(str(m.get("precision", "")).upper() == "Q4" for m in active_models)

        tier_meta_map: Dict[str, Tuple[str, str, Optional[str], Optional[str], int]] = {
            "v5e-1": ("tier1_v5e_1_16gb", "v5e-1", None, "v5e-1", 4),
            "T4": ("tier1_t4_16gb", "T4", "T4", None, 32),
            "L4": ("tier2_l4_24gb", "L4", "L4", None, 64),
            "v6e-1": ("tier8_v6e_1_32gb", "v6e-1", None, "v6e-1", 8),
            "V6E-1": ("tier8_v6e_1_32gb", "v6e-1", None, "v6e-1", 8),
            "A100": ("tier3_a100_40gb", "A100", "A100", None, 96),
            "A100-40GB": ("tier3_a100_40gb", "A100", "A100", None, 96),
            "A100-80GB": ("tier4_a100_80gb", "A100-80GB", "A100", None, 128),
            "H100": ("tier5_h100_80gb", "H100", "H100", None, 160),
            "G4": ("tier6_g4_96gb", "G4", "G4", None, 192),
            "B200": ("tier7_b200_180gb", "B200", "B200", None, 256),
        }
        tier_id, gpu_type_str, colab_gpu, colab_tpu, hw_default_seqs = tier_meta_map.get(
            hw_tier,
            (f"tier_{hw_tier.lower()}", hw_tier, hw_tier, None, 32),
        )
        if is_tpu:
            max_seqs = max(hw_default_seqs, min(16, max((int(m["agent_count"]) for m in active_models), default=1)))
        else:
            primary_alias = str(active_models[0]["alias"]) if active_models else UNIFIED_MODEL_ALIAS
            kv_dt = "fp8" if hw_tier in ("L4", "H100", "G4", "B200") else "auto"
            max_seqs = _compute_dynamic_max_num_seqs(
                free_vram_gb=vram_gb,
                weights_gb=float(resolved["total_weights_gb"]),
                model_alias=primary_alias,
                max_model_len=MAX_MODEL_LEN,
                kv_cache_dtype=kv_dt,
                requested_agents=max(1, total_agents),
                active_slot_count=max(1, len(active_models)),
            )

        est_peak = min(vram_gb, round(float(resolved["estimated_peak_vram_gb"]), 2))
        active_summary = ", ".join(
            f"{m['alias']}@{m['precision']}:{m['agent_count']}" for m in active_models
        ) or "none"
        inactive_summary = ", ".join(
            f"{m['alias']}:0" for m in inactive_models
        ) or "none"

        precision_tier_str = (
            "int4_quantized"
            if (is_tpu and all_q4)
            else ("gptq_4bit" if all_q4 else "bfloat16_or_fp8_native")
        )

        base_profile = {
            "tier_id": tier_id,
            "gpu_type": gpu_type_str,
            "hardware_tier": hw_tier,
            "accelerator_type": "TPU" if is_tpu else "GPU",
            "memory_type": str(tier_spec.get("memory_type", "HBM" if is_tpu else "VRAM")),
            "colab_accelerator": "TPU" if is_tpu else "GPU",
            "colab_gpu_type": colab_gpu,
            "colab_tpu_type": colab_tpu,
            "gpu_label": f"{tier_spec.get('label') or (f'Cloud TPU {hw_tier}' if is_tpu else f'NVIDIA {hw_tier}')} ({vram_gb:.1f}GB {tier_spec.get('memory_type', 'VRAM')} · {sys_ram_gb:.1f}GB High-RAM)",
            "vram_gb": vram_gb,
            "system_ram_gb": sys_ram_gb,
            "colab_machine_shape": "hm",
            "precision_tier": precision_tier_str,
            "load_in_4bit": all_q4,
            "max_num_seqs": max_seqs,
            "max_parallel_agents": max(max_seqs, total_agents),
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": 8192 if vram_gb <= 24.0 else MAX_NUM_BATCHED_TOKENS,
            "estimated_peak_vram_gib": est_peak,
            "total_qwen_models": len(active_models),
            "peak_stage_qwen_models": total_agents,
            "per_stage_counts": agent_counts,
            "override_mode": str(gpu_override or "auto"),
            "stack_mode": resolved["stack_mode"],
            "stack_preset": None,
            "model_count": len(active_models),
            "active_model_count": len(active_models),
            "inactive_model_count": len(inactive_models),
            "total_agents": total_agents,
            "stack_models": active_models,
            "inactive_models": inactive_models,
            "loaded_models": list(resolved["loaded_models"]),
            "inactive_unloaded_models": list(resolved["inactive_unloaded_models"]),
            "agent_counts": agent_counts,
            "vram_allocated_by_model": vram_by_model,
            "estimated_weights_gb": float(resolved["total_weights_gb"]),
            "kv_headroom_gb": float(resolved["free_kv_gb"]),
            "rationale": (
                f"GPU 1 multi-model allocation (active=[{active_summary}], unloaded_zero_agent=[{inactive_summary}], "
                f"total_agents={total_agents}) -> Routed to {hw_tier} ({vram_gb:.1f}GB, "
                f"weights={resolved['total_weights_gb']}GB, free_kv={resolved['free_kv_gb']}GB)."
            ),
        }

    base_profile.update(
        {
            "tierId": base_profile["tier_id"],
            "gpuType": base_profile["gpu_type"],
            "acceleratorType": base_profile["accelerator_type"],
            "memoryType": base_profile["memory_type"],
            "colabAccelerator": base_profile["colab_accelerator"],
            "colabGpuType": base_profile["colab_gpu_type"],
            "colabTpuType": base_profile["colab_tpu_type"],
            "gpuLabel": base_profile["gpu_label"],
            "vramGb": base_profile["vram_gb"],
            "systemRamGb": base_profile["system_ram_gb"],
            "colabMachineShape": base_profile["colab_machine_shape"],
            "precisionTier": base_profile["precision_tier"],
            "loadIn4Bit": base_profile["load_in_4bit"],
            "maxNumSeqs": base_profile["max_num_seqs"],
            "maxParallelAgents": base_profile["max_parallel_agents"],
            "maxModelLen": base_profile["max_model_len"],
            "maxNumBatchedTokens": base_profile["max_num_batched_tokens"],
            "estimatedPeakVramGib": base_profile["estimated_peak_vram_gib"],
            "totalQwenModels": base_profile["total_qwen_models"],
            "peakStageQwenModels": base_profile["peak_stage_qwen_models"],
            "perStageCounts": base_profile["per_stage_counts"],
            "overrideMode": base_profile["override_mode"],
            "stackMode": base_profile["stack_mode"],
            "stackPreset": base_profile["stack_preset"],
            "modelCount": base_profile["model_count"],
            "activeModelCount": base_profile["active_model_count"],
            "inactiveModelCount": base_profile["inactive_model_count"],
            "totalAgents": base_profile["total_agents"],
            "stackModels": base_profile["stack_models"],
            "inactiveModels": base_profile["inactive_models"],
            "agentCounts": base_profile["agent_counts"],
            "vramAllocatedByModel": base_profile["vram_allocated_by_model"],
            "estimatedWeightsGb": base_profile["estimated_weights_gb"],
            "kvHeadroomGb": base_profile["kv_headroom_gb"],
        }
    )
    if repo_dir is not None:
        proj = resolve_project_root(repo_dir)
        base_profile["project_root"] = str(proj)
        base_profile["dir_slug"] = compute_dir_slug(proj)
    return base_profile


def compute_serving_concurrency_config(
    total_vram_gb: float,
    max_model_len: int = MAX_MODEL_LEN,
    target_parallel_agents: int = MAX_PARALLEL_AGENTS,
    force_4bit: Optional[bool] = None,
    *,
    model_alias: str = UNIFIED_MODEL_ALIAS,
    kv_cache_dtype: str = "auto",
) -> Dict[str, Any]:
    """Computes first-principles Single-GPU Parallel Serving parameters (Spec 853 FR-3 / Spec 869).

    Uses exact architecture KV-cache and GatedDeltaNet / MLA recurrent state sizing
    (_estimate_model_kv_and_state_footprint) and dynamic vLLM continuous-batching
    concurrency (_compute_dynamic_max_num_seqs).
    """
    if total_vram_gb < 14.0:
        raise RuntimeError(
            f"Insufficient GPU VRAM ({total_vram_gb:.2f} GB < 14.0 GB minimum) for {model_alias} serving."
        )

    tier_info = select_qwen35_precision_tier(total_vram_gb=total_vram_gb, force_4bit=force_4bit)
    load_in_4bit = bool(tier_info["load_in_4bit"])
    footprint = _estimate_model_kv_and_state_footprint(
        model_alias,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
    )
    kv_per_agent_gib = float(footprint["kv_gb_per_seq_at_max_len"])
    recurrent_per_agent_gib = float(footprint["gdn_gb_per_seq"])

    if not load_in_4bit and total_vram_gb >= 38.0:
        base_weights_gib = 17.55
        active_seqs = int(target_parallel_agents)
        kv_total_gib = round(active_seqs * kv_per_agent_gib, 3)
        recurrent_total_gib = round(active_seqs * recurrent_per_agent_gib, 3)
        workspace_gib = 1.20
        peak_vram_gib = round(base_weights_gib + kv_total_gib + recurrent_total_gib + workspace_gib, 2)
        dynamic_seqs = _compute_dynamic_max_num_seqs(
            free_vram_gb=total_vram_gb,
            weights_gb=base_weights_gib,
            model_alias=model_alias,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
            requested_agents=target_parallel_agents,
        )
        return {
            "model": model_alias,
            "precision_tier": "bfloat16_native",
            "dtype": "bfloat16",
            "load_in_4bit": False,
            "tf32": True,
            "max_num_seqs": active_seqs,
            "dynamic_max_num_seqs": dynamic_seqs,
            "max_parallel_agents": int(target_parallel_agents),
            "max_model_len": int(max_model_len),
            "max_num_batched_tokens": int(max_model_len),
            "enable_chunked_prefill": True,
            "last_token_only_lm_head": True,
            "gpu_memory_utilization": 0.90,
            "queue_timeout_sec": 300.0,
            "base_weights_gib": base_weights_gib,
            "kv_cache_gib": kv_total_gib,
            "recurrent_state_gib": recurrent_total_gib,
            "kv_bytes_per_token": float(footprint["kv_bytes_per_token"]),
            "gdn_bytes_per_seq": float(footprint["gdn_bytes_per_seq"]),
            "workspace_gib": workspace_gib,
            "estimated_peak_vram_gib": peak_vram_gib,
            "total_vram_gb": float(total_vram_gb),
        }

    # Constrained-VRAM fallback (15.0 <= VRAM < 38.0 GB)
    base_weights_gib = 6.20
    workspace_gib = 2.40
    budget_gib = 0.85 * float(total_vram_gb)
    available_for_seqs = max(0.5, budget_gib - base_weights_gib - workspace_gib)
    per_seq_cost = max(0.05, kv_per_agent_gib + recurrent_per_agent_gib)
    active_seqs = max(2, min(int(target_parallel_agents), int(available_for_seqs // per_seq_cost)))
    kv_total_gib = round(active_seqs * kv_per_agent_gib, 3)
    recurrent_total_gib = round(active_seqs * recurrent_per_agent_gib, 3)
    peak_vram_gib = round(base_weights_gib + kv_total_gib + recurrent_total_gib + workspace_gib, 2)
    dynamic_seqs = _compute_dynamic_max_num_seqs(
        free_vram_gb=total_vram_gb,
        weights_gb=base_weights_gib,
        model_alias=model_alias,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        requested_agents=target_parallel_agents,
    )
    norm_alias = str(model_alias or "").strip().lower()
    constrained_prec = (
        "gptq_4bit"
        if bool(norm_alias)
        else "nf4_4bit"
    )
    return {
        "model": model_alias,
        "precision_tier": constrained_prec,
        "dtype": "bfloat16",
        "load_in_4bit": True,
        "tf32": True,
        "max_num_seqs": active_seqs,
        "dynamic_max_num_seqs": dynamic_seqs,
        "max_parallel_agents": int(target_parallel_agents),
        "max_model_len": int(max_model_len),
        "max_num_batched_tokens": min(int(max_model_len), 8192),
        "enable_chunked_prefill": True,
        "last_token_only_lm_head": True,
        "gpu_memory_utilization": 0.85,
        "queue_timeout_sec": 300.0,
        "base_weights_gib": base_weights_gib,
        "kv_cache_gib": kv_total_gib,
        "recurrent_state_gib": recurrent_total_gib,
        "kv_bytes_per_token": float(footprint["kv_bytes_per_token"]),
        "gdn_bytes_per_seq": float(footprint["gdn_bytes_per_seq"]),
        "workspace_gib": workspace_gib,
        "estimated_peak_vram_gib": peak_vram_gib,
        "total_vram_gb": float(total_vram_gb),
    }


def normalize_colab_endpoint_url(raw_url: str) -> str:
    """Normalizes any Colab tunnel URL into an OpenAI-compatible /v1 base URL."""
    cleaned = (raw_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")]
    elif cleaned.endswith("/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")]
        if not cleaned.endswith("/v1"):
            cleaned += "/v1"
    elif not cleaned.endswith("/v1"):
        cleaned += "/v1"
    return cleaned


def resolve_project_root(path: Optional[Path | str] = None) -> Path:
    """Resolve any repo, .sandcastle dir, or .sandcastle/worktrees/<branch> path to canonical <project_root> (FR-3.1)."""
    raw = Path(path).expanduser().resolve() if path is not None and str(path).strip() else Path.cwd().resolve()
    if raw.is_file():
        raw = raw.parent
    parts = raw.parts
    for i in range(len(parts) - 1):
        if parts[i] == ".sandcastle" and parts[i + 1] == "worktrees":
            if i > 0:
                return Path(*parts[:i]).resolve()
    if raw.name == ".sandcastle":
        return raw.parent.resolve()
    if raw.name == "runtime" and raw.parent.name == ".platodoom":
        return raw.parent.parent.resolve()
    for parent in raw.parents:
        if parent.name == ".sandcastle":
            return parent.parent.resolve()
        if parent.name == "runtime" and parent.parent.name == ".platodoom":
            return parent.parent.parent.resolve()
    return raw


def compute_dir_slug(path: Optional[Path | str] = None) -> str:
    """Computes deterministic <repo_basename>-<sha1_8_of_project_root> (Section 2.0.2, FR-3.1)."""
    project_root = resolve_project_root(path)
    base = re.sub(r"[^a-zA-Z0-9]+", "-", project_root.name).strip("-").lower() or "repo"
    digest = hashlib.sha1(str(project_root).encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def resolve_directory_sandcastle_dir(path: Optional[Path | str] = None) -> Path:
    """Returns <project_root>/.sandcastle for storing per-directory registry, routing, lock, and PID files (FR-3.1)."""
    return resolve_project_root(path) / ".sandcastle"


def atomic_write_json(
    target_path: Path | str,
    data: Dict[str, Any],
    mode: Optional[int] = None,
) -> None:
    """Write JSON payload atomically via same-directory temporary file, chmod, fsync, and os.replace (Storage Architecture §3.1)."""
    resolved_target = Path(target_path).expanduser().resolve()
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if mode is not None:
        effective_mode = int(mode)
    elif (
        isinstance(data, dict)
        and (
            bool(data.get("bearer_token"))
            or bool(data.get("auth_token"))
            or resolved_target.name == "config.json"
        )
    ):
        effective_mode = 0o600
    else:
        effective_mode = 0o644

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{resolved_target.name}.",
        suffix=f".{os.getpid()}.tmp",
        dir=str(resolved_target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, effective_mode)
        except (AttributeError, OSError):
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(resolved_target))
        try:
            os.chmod(str(resolved_target), effective_mode)
        except OSError:
            pass
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _resolve_platodoom_sessions_root() -> Path:
    custom_home = os.environ.get("PLATODOOM_CONFIG_DIR") or os.environ.get("PLATODOOM_HOME")
    if custom_home and str(custom_home).strip():
        return Path(str(custom_home).strip()).expanduser().resolve() / "sessions"
    return (Path.home() / ".platodoom" / "sessions").resolve()


def resolve_colab_endpoint_registry_paths(
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    session_slug: Optional[str] = None,
    project_root: Optional[Path | str] = None,
    target_dir: Optional[Path | str] = None,
    sync_global_fallback: bool = False,
) -> List[Path]:
    """Resolves per-directory or per-session endpoint registry paths without cross-directory pollution (FR-2.1.2, FR-3.1, FR-3.2)."""
    if session_slug is not None and str(session_slug).strip():
        cleaned_slug = str(session_slug).strip()
        session_dir = (_resolve_platodoom_sessions_root() / cleaned_slug).resolve()
        return [
            (session_dir / "endpoint.json").resolve(),
            (session_dir / "colab_endpoint.json").resolve(),
        ]

    eff_repo_dir = repo_dir if repo_dir is not None else (target_dir if target_dir is not None else project_root)
    resolved_proj_root = resolve_project_root(eff_repo_dir)
    dir_slug = compute_dir_slug(resolved_proj_root)
    paths: List[Path] = [
        resolved_proj_root / ".sandcastle" / DEFAULT_COLAB_ENDPOINT_FILENAME,
    ]
    if (resolved_proj_root / "multi-stage-blackboard-pipeline.mts").is_file() and not (
        resolved_proj_root / ".sandcastle" / "multi-stage-blackboard-pipeline.mts"
    ).is_file():
        paths.append(resolved_proj_root / DEFAULT_COLAB_ENDPOINT_FILENAME)
    paths.append(Path.home() / ".sandcastle" / "daemons" / dir_slug / DEFAULT_COLAB_ENDPOINT_FILENAME)

    if include_global or sync_global_fallback:
        swarm_root = resolve_project_root(Path(__file__).resolve().parent.parent)
        paths.append(swarm_root / ".sandcastle" / DEFAULT_COLAB_ENDPOINT_FILENAME)
        paths.append(Path.home() / ".sandcastle" / DEFAULT_COLAB_ENDPOINT_FILENAME)

    deduped: List[Path] = []
    seen: set[str] = set()
    for p in paths:
        rp = str(p.expanduser().resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(Path(rp))
    return deduped


def load_colab_endpoint_record(
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    session_slug: Optional[str] = None,
    project_root: Optional[Path | str] = None,
    target_dir: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """Loads the persisted Colab endpoint registry record for the target project root or interactive session_slug (FR-2.1.2, FR-3.1, FR-3.2)."""
    for reg_path in resolve_colab_endpoint_registry_paths(
        repo_dir,
        include_global=include_global,
        session_slug=session_slug,
        project_root=project_root,
        target_dir=target_dir,
    ):
        if reg_path.is_file():
            try:
                data = json.loads(reg_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("registry_path", str(reg_path))
                    return data
            except Exception as exc:
                logger.debug("Could not read endpoint registry from %s: %s", reg_path, exc)
                continue
    return None


def register_colab_qwen_endpoint(
    endpoint_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
    session_slug: Optional[str] = None,
    project_root: Optional[Path | str] = None,
    target_dir: Optional[Path | str] = None,
    sync_global_fallback: bool = False,
) -> Dict[str, Any]:
    """Persists the live HTTPS Colab endpoint URL and serving metadata to the per-directory or per-session endpoint registry (FR-2.1.2, FR-2.2, FR-3.2)."""
    base_url = normalize_colab_endpoint_url(endpoint_url)
    if not base_url:
        raise ValueError("Cannot register an empty Colab endpoint URL.")

    effective_session_slug = (
        str(session_slug).strip()
        if session_slug is not None and str(session_slug).strip()
        else None
    )
    if effective_session_slug is None and isinstance(metadata, dict):
        meta_slug = str(metadata.get("session_slug") or "").strip()
        if meta_slug.startswith("platodoom-chat-"):
            effective_session_slug = meta_slug

    eff_repo_dir = repo_dir if repo_dir is not None else (target_dir if target_dir is not None else project_root)
    resolved_proj_root = resolve_project_root(eff_repo_dir)
    dir_slug = effective_session_slug if effective_session_slug else compute_dir_slug(resolved_proj_root)
    now_epoch = time.time()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_epoch))

    record: Dict[str, Any] = {
        "endpoint_url": base_url,
        "chat_completions_url": f"{base_url}/chat/completions",
        "models_url": f"{base_url}/models",
        "model": UNIFIED_MODEL_ALIAS,
        "status": "Online",
        "offline_reason": None,
        "dir_slug": dir_slug,
        "project_root": str(resolved_proj_root),
        "max_parallel_agents": MAX_PARALLEL_AGENTS,
        "max_num_seqs": MAX_PARALLEL_AGENTS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "checked_at": now_iso,
        "updated_at": now_iso,
        "updated_at_epoch": now_epoch,
        "heartbeat_epoch": now_epoch,
    }
    if effective_session_slug:
        record["session_slug"] = effective_session_slug
        record["session_type"] = "interactive" if effective_session_slug.startswith("platodoom-chat-") else "pipeline"
    if metadata:
        record.update(metadata)
        record["endpoint_url"] = base_url
        record["chat_completions_url"] = f"{base_url}/chat/completions"
        record["models_url"] = f"{base_url}/models"
        if not record.get("dir_slug"):
            record["dir_slug"] = dir_slug
        if not record.get("project_root"):
            record["project_root"] = str(resolved_proj_root)
        if effective_session_slug:
            record["session_slug"] = effective_session_slug
        record.setdefault("updated_at", now_iso)
        record.setdefault("updated_at_epoch", now_epoch)
        record.setdefault("heartbeat_epoch", now_epoch)

    write_mode = (
        0o600
        if (
            bool(record.get("bearer_token"))
            or bool(record.get("auth_token"))
            or effective_session_slug is not None
        )
        else None
    )

    written_paths: List[str] = []
    for reg_path in resolve_colab_endpoint_registry_paths(
        eff_repo_dir,
        include_global=(include_global or sync_global_fallback),
        session_slug=effective_session_slug,
    ):
        try:
            atomic_write_json(reg_path, record, mode=write_mode)
            written_paths.append(str(reg_path))
        except Exception as exc:
            logger.debug("Could not write endpoint registry to %s: %s", reg_path, exc)

    record["registry_paths"] = written_paths
    return record


def resolve_gpu_routing_registry_paths(
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
) -> List[Path]:
    """Resolves per-directory GPU hardware routing profile paths without cross-directory pollution (FR-3.2)."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    paths: List[Path] = [
        project_root / ".sandcastle" / DEFAULT_GPU_ROUTING_FILENAME,
    ]
    if (project_root / "multi-stage-blackboard-pipeline.mts").is_file() and not (
        project_root / ".sandcastle" / "multi-stage-blackboard-pipeline.mts"
    ).is_file():
        paths.append(project_root / DEFAULT_GPU_ROUTING_FILENAME)
    paths.append(Path.home() / ".sandcastle" / "daemons" / dir_slug / DEFAULT_GPU_ROUTING_FILENAME)

    if include_global:
        swarm_root = resolve_project_root(Path(__file__).resolve().parent.parent)
        paths.append(swarm_root / ".sandcastle" / DEFAULT_GPU_ROUTING_FILENAME)
        paths.append(Path.home() / ".sandcastle" / DEFAULT_GPU_ROUTING_FILENAME)

    deduped: List[Path] = []
    seen: set[str] = set()
    for p in paths:
        rp = str(p.expanduser().resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(Path(rp))
    return deduped


def write_gpu_routing_profile(
    profile: Dict[str, Any],
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
) -> Dict[str, Any]:
    """Atomically persists the per-directory GPU hardware routing profile (Storage Architecture §6.2)."""
    project_root = resolve_project_root(repo_dir)
    dir_slug = compute_dir_slug(project_root)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = dict(profile)
    payload["dir_slug"] = dir_slug
    payload["project_root"] = str(project_root)
    payload.setdefault("updated_at", now_iso)

    written_paths: List[str] = []
    for routing_path in resolve_gpu_routing_registry_paths(repo_dir, include_global=include_global):
        try:
            atomic_write_json(routing_path, payload)
            written_paths.append(str(routing_path))
        except Exception as exc:
            logger.debug("Could not write GPU routing profile to %s: %s", routing_path, exc)
    payload["registry_paths"] = written_paths
    return payload


def load_gpu_routing_profile(
    repo_dir: Optional[Path | str] = None,
    include_global: bool = False,
) -> Optional[Dict[str, Any]]:
    """Loads the persisted per-directory GPU hardware routing profile if present."""
    for routing_path in resolve_gpu_routing_registry_paths(repo_dir, include_global=include_global):
        if routing_path.is_file():
            try:
                data = json.loads(routing_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("registry_path", str(routing_path))
                    return data
            except Exception as exc:
                logger.debug("Could not read GPU routing profile from %s: %s", routing_path, exc)
                continue
    return None


_RECOGNIZED_SERVED_MODEL_IDS = {
    UNIFIED_MODEL_ALIAS,
    "qwen-9b",
    "r1-14b",
    "qwen-27b",
    "qwen3.8-27b",
    "qwen-27b-38",
    "qwen-35b-moe",
    "r1-32b",
    "qwen-30b-moe",
    "qwen-coder-30b-moe",
    "qwen-14b",
    "qwen-8b",
    "qwen-4b",
    "qwen-122b-moe",
    "r1-7b",
    "deepseek-v2-lite",
    "flux-klein-4b",
    "flux-klein-9b",
    "z-image-turbo",
}

_RECOGNIZED_SERVED_MODEL_SUBSTRINGS = (
    "qwen3.5",
    "qwen3.6",
    "qwen3.8",
    "qwen3-",
    "qwen/qwen3",
    "deepseek-r1-distill-qwen",
    "deepseek-v2-lite",
    "sandcastle-qwen",
)


_MODEL_EQUIVALENCE_GROUPS: Tuple[Set[str], ...] = (
    {
        "qwen-9b",
        "qwen/qwen3.5-9b",
        "qwen/qwen3.5-9b-gptq-int4",
        "cyankiwi/qwen3.5-9b-awq-4bit",
        "mssfj/qwen3.5-9b-gptq-int4",
        "quanttrio/qwen3.5-9b-awq",
    },
    {
        "r1-14b",
        "deepseek-ai/deepseek-r1-distill-qwen-14b",
        "casperhansen/deepseek-r1-distill-qwen-14b-awq",
    },
    {
        "qwen-27b",
        "qwen/qwen3.5-27b",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.5-27b-gptq-int4",
        "qwen/qwen3.5-27b-awq",
        "cyankiwi/qwen3.5-27b-awq-4bit",
        "alennnndy/qwen3.6-27b-awq-4bit",
        "qwen/qwen3.5-27b-fp8",
    },
    {
        "qwen3.8-27b",
        "qwen-27b-38",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.8-27b-fp8",
        "cyankiwi/qwen3.8-27b-awq-int4",
    },
    {
        "qwen-35b-moe",
        "qwen/qwen3.5-35b-a3b",
        "qwen/qwen3.5-35b-a3b-gptq-int4",
        "qwen/qwen3.5-35b-a3b-fp8",
        "qwen/qwen3.6-35b-a3b",
        "qwen/qwen3.6-35b-a3b-gptq-int4",
        "quanttrio/qwen3.6-35b-a3b-awq",
        "cyankiwi/qwen3.6-35b-a3b-awq-4bit",
    },
    {
        "r1-32b",
        "deepseek-ai/deepseek-r1-distill-qwen-32b",
        "casperhansen/deepseek-r1-distill-qwen-32b-awq",
    },
    {
        "qwen-30b-moe",
        "qwen/qwen3-30b-a3b",
        "qwen/qwen3-30b-a3b-awq",
        "qwen/qwen3-30b-a3b-gptq-int4",
        "stelterlab/qwen3-30b-a3b-instruct-2507-awq",
        "qwen/qwen3-30b-a3b-fp8",
    },
    {
        "qwen-coder-30b-moe",
        "qwen/qwen3-coder-30b-a3b-instruct",
        "cyankiwi/qwen3-coder-30b-a3b-instruct-awq-4bit",
        "qwen/qwen3-coder-30b-a3b-instruct-awq",
        "qwen/qwen3-coder-30b-a3b-instruct-gptq-int4",
        "qwen/qwen3-coder-30b-a3b-instruct-fp8",
    },
    {
        "qwen-14b",
        "qwen/qwen3-14b",
        "qwen/qwen3-14b-awq",
        "qwen/qwen3-14b-fp8",
    },
    {
        "qwen-8b",
        "qwen/qwen3-8b",
        "qwen/qwen3-8b-awq",
        "qwen/qwen3-8b-fp8",
    },
    {
        "qwen-4b",
        "qwen/qwen3-4b",
        "qwen/qwen3-4b-awq",
        "qwen/qwen3-4b-fp8",
    },
    {
        "qwen-122b-moe",
        "qwen/qwen3.5-122b-a10b",
        "qwen/qwen3.5-122b-a10b-gptq-int4",
        "qwen/qwen3.5-122b-a10b-fp8",
    },
    {
        "r1-7b",
        "deepseek-ai/deepseek-r1-distill-qwen-7b",
        "casperhansen/deepseek-r1-distill-qwen-7b-awq",
    },
    {
        "deepseek-v2-lite",
        "deepseek-ai/deepseek-v2-lite-chat",
        "deepseek-ai/deepseek-v2-lite",
        "techxgenus/deepseek-v2-lite-chat-awq",
        "casperhansen/deepseek-v2-lite-chat-awq",
    },
    {"flux-klein-4b", "black-forest-labs/flux.2-klein-4b", "black-forest-labs/flux.2-klein-base-4b", "black-forest-labs/flux.2-klein-4b-fp8"},
    {"flux-klein-9b", "black-forest-labs/flux.2-klein-9b", "unsloth/flux.2-klein-9b-gguf"},
    {"z-image-turbo", "tongyi-mai/z-image-turbo", "unsloth/z-image-turbo-fp8"},
)

try:
    from platodoom.catalog import BUILTIN_MODEL_CATALOG as _API_SSOT_CATALOG

    for _cat_alias, _cat_prof in _API_SSOT_CATALOG.items():
        _RECOGNIZED_SERVED_MODEL_IDS.add(_cat_alias)
        for _repo_key in ("hf_repo_id", "quantized_hf_repo_id", "fallback_hf_repo_id"):
            _rv = str(_cat_prof.get(_repo_key) or "").strip()
            if _rv:
                _RECOGNIZED_SERVED_MODEL_IDS.add(_rv)
        for _pv in (_cat_prof.get("precision_variants") or {}).values():
            if isinstance(_pv, dict) and _pv.get("hf_repo_id"):
                _RECOGNIZED_SERVED_MODEL_IDS.add(str(_pv["hf_repo_id"]).strip())
except Exception:
    pass

_KNOWN_DISTINCT_ALIASES: Set[str] = {
    "qwen-9b",
    "r1-14b",
    "qwen-27b",
    "qwen3.8-27b",
    "qwen-27b-38",
    "qwen-35b-moe",
    "r1-32b",
    "qwen-30b-moe",
    "qwen-coder-30b-moe",
    "qwen-14b",
    "qwen-8b",
    "qwen-4b",
    "qwen-122b-moe",
    "r1-7b",
    "deepseek-v2-lite",
    "flux-klein-4b",
    "flux-klein-9b",
    "z-image-turbo",
}

_DNS_ERROR_MARKERS: Tuple[str, ...] = (
    "nodename nor servname provided",
    "name or service not known",
    "temporary failure in name resolution",
    "no address associated with hostname",
    "getaddrinfo failed",
    "[errno 8]",
    "[errno -2]",
    "[errno -3]",
    "[errno -5]",
)


def resolve_serving_bearer_token(
    bearer_token: Optional[str] = None,
    api_key: Optional[str] = None,
    record: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolves serving Authorization bearer token honoring both PLATODOOM_TUNNEL_TOKEN and SANDCASTLE_VLLM_API_KEY."""
    rec_token: Optional[str] = None
    if isinstance(record, dict):
        rec_token = str(record.get("bearer_token") or record.get("auth_token") or "").strip() or None

    for candidate in (
        bearer_token,
        api_key,
        rec_token,
        os.environ.get("PLATODOOM_TUNNEL_TOKEN"),
        os.environ.get("SANDCASTLE_VLLM_API_KEY"),
        os.environ.get("VLLM_API_KEY"),
    ):
        if candidate is not None:
            cleaned = str(candidate).strip()
            if cleaned and cleaned != "EMPTY":
                return cleaned
    return "EMPTY"


def resolve_expected_model_identifiers(expected_model: str) -> Set[str]:
    """Resolves a strict closed set of lowercase identifiers (alias, canonical repo, quantized repo) for expected_model."""
    raw = str(expected_model or "").strip()
    if not raw:
        return set()
    norm = raw.lower()
    accepted: Set[str] = {norm}

    for grp in _MODEL_EQUIVALENCE_GROUPS:
        if norm in grp:
            accepted.update(grp)

    try:
        from platodoom.catalog import BUILTIN_MODEL_CATALOG, resolve_model_profile

        def _collect_profile_ids(cat_alias: str, cat_prof: Dict[str, Any]) -> Set[str]:
            ids: Set[str] = {cat_alias.lower()}
            for r_key in ("hf_repo_id", "quantized_hf_repo_id", "fallback_hf_repo_id"):
                r_val = str(cat_prof.get(r_key) or "").strip().lower()
                if r_val:
                    ids.add(r_val)
            for pv in (cat_prof.get("precision_variants") or {}).values():
                if isinstance(pv, dict):
                    pv_repo = str(pv.get("hf_repo_id") or "").strip().lower()
                    if pv_repo:
                        ids.add(pv_repo)
                    for fb_r in pv.get("fallback_hf_repo_ids") or ():
                        fb_s = str(fb_r or "").strip().lower()
                        if fb_s:
                            ids.add(fb_s)
            return ids

        if norm in BUILTIN_MODEL_CATALOG:
            accepted.update(_collect_profile_ids(norm, BUILTIN_MODEL_CATALOG[norm]))
        else:
            for c_alias, c_prof in BUILTIN_MODEL_CATALOG.items():
                c_ids = _collect_profile_ids(c_alias, c_prof)
                if norm in c_ids:
                    accepted.update(c_ids)
                    for grp in _MODEL_EQUIVALENCE_GROUPS:
                        if c_alias.lower() in grp:
                            accepted.update(grp)

        if norm not in _KNOWN_DISTINCT_ALIASES:
            prof = resolve_model_profile(raw)
            prof_alias = str(prof.get("model_alias") or "").strip().lower()
            prof_repo = str(prof.get("hf_repo_id") or "").strip().lower()
            if prof_alias and prof_alias == norm:
                accepted.add(prof_alias)
                if prof_repo:
                    accepted.add(prof_repo)
    except Exception:
        pass

    return accepted


def _normalize_served_model_candidate(raw_id: str) -> Set[str]:
    cleaned = str(raw_id or "").strip()
    if not cleaned:
        return set()
    lower = cleaned.lower()
    out: Set[str] = {lower}
    if "/" in cleaned and (cleaned.startswith("/") or cleaned.startswith("./")):
        leaf = Path(cleaned).name.lower()
        out.add(leaf)
        if "--" in leaf:
            out.add(leaf.replace("--", "/", 1))
    return out


def _normalize_expected_model_list(expected_model: Optional[str | Sequence[str]]) -> List[str]:
    if expected_model is None:
        return []
    if isinstance(expected_model, str):
        cleaned = expected_model.strip()
        if not cleaned:
            return []
        if "," in cleaned:
            return [part.strip().split("@", 1)[0] for part in cleaned.split(",") if part.strip()]
        return [cleaned.split("@", 1)[0].strip()]
    out: List[str] = []
    for item in expected_model:
        s = str(item or "").strip()
        if s:
            out.append(s.split("@", 1)[0].strip())
    return out


def _matches_single_expected_model(
    models: List[str],
    single_expected: str,
    model_roots: Optional[Dict[str, str]] = None,
) -> bool:
    accepted = resolve_expected_model_identifiers(single_expected)
    if not accepted:
        return False
    for m in models:
        m_candidates = _normalize_served_model_candidate(m)
        if m_candidates & accepted:
            return True
        m_lower = str(m or "").strip().lower()
        if model_roots and m in model_roots and m_lower not in _KNOWN_DISTINCT_ALIASES:
            root_candidates = _normalize_served_model_candidate(model_roots[m])
            if root_candidates & accepted:
                return True
    return False


def _is_recognized_served_model(
    models: List[str],
    expected_model: Optional[str | Sequence[str]] = None,
    model_roots: Optional[Dict[str, str]] = None,
) -> bool:
    if not models:
        return False
    expected_list = _normalize_expected_model_list(expected_model)
    if expected_list:
        return all(_matches_single_expected_model(models, exp, model_roots=model_roots) for exp in expected_list)

    for m in models:
        if m in _RECOGNIZED_SERVED_MODEL_IDS:
            return True
        ml = m.lower()
        if any(sub in ml for sub in _RECOGNIZED_SERVED_MODEL_SUBSTRINGS):
            return True
    return False


def is_transient_tunnel_dns_error(exc: BaseException | str, endpoint_url: str = "") -> bool:
    """Returns True iff `exc` represents a transient DNS hostname resolution failure on a non-loopback tunnel URL."""
    if endpoint_url:
        parsed_host = (urllib.parse.urlparse(endpoint_url).hostname or "").strip().lower()
        if parsed_host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
            return False

    if isinstance(exc, str):
        msg = exc.lower()
        return any(marker in msg for marker in _DNS_ERROR_MARKERS)

    visited: Set[int] = set()
    cur: Optional[Any] = exc
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if isinstance(cur, socket.gaierror):
            return True
        msg = str(cur).lower()
        if any(marker in msg for marker in _DNS_ERROR_MARKERS):
            return True
        reason = getattr(cur, "reason", None)
        if isinstance(reason, socket.gaierror):
            return True
        if reason is not None and any(marker in str(reason).lower() for marker in _DNS_ERROR_MARKERS):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def probe_colab_qwen_endpoint(
    endpoint_url: str,
    api_key: Optional[str] = None,
    timeout_sec: float = 5.0,
    verify_inference: bool = False,
    inference_timeout_sec: float = 15.0,
    expected_model: Optional[str | Sequence[str]] = None,
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
    base_url = normalize_colab_endpoint_url(endpoint_url)
    if not base_url:
        return {
            "online": False,
            "ok": False,
            "inference_verified": False,
            "endpoint_url": "",
            "models": [],
            "served_models": [],
            "latency_ms": 0.0,
            "inference_latency_ms": 0.0,
            "error": "Empty endpoint URL",
        }

    strict_timeout = min(max(float(timeout_sec), 0.2), 5.0)
    token = resolve_serving_bearer_token(bearer_token=bearer_token, api_key=api_key)
    overall_t0 = time.perf_counter()
    dns_deadline = time.monotonic() + max(0.0, float(dns_retry_window_sec))
    retry_interval = max(0.05, float(dns_retry_interval_sec))

    active_base_url = base_url
    used_loopback_fallback = False
    dns_attempts = 0
    models: List[str] = []
    model_roots: Dict[str, str] = {}
    model_details: Dict[str, Dict[str, Any]] = {}
    models_latency_ms = 0.0

    while True:
        models_url = f"{active_base_url}/models"
        req = urllib.request.Request(
            models_url,
            headers={
                "Authorization": f"Bearer {token}",
                "ngrok-skip-browser-warning": "true",
                "User-Agent": "Sandcastle-Colab-Probe/1.0",
            },
            method="GET",
        )
        req_t0 = time.perf_counter()
        try:
            with _open_http_request(req, timeout=strict_timeout) as resp:
                models_latency_ms = round((time.perf_counter() - req_t0) * 1000.0, 2)
                if resp.status == 401:
                    return {
                        "online": False,
                        "ok": False,
                        "alive": False,
                        "auth_error": True,
                        "status_code": 401,
                        "http_status": 401,
                        "error_code": "COLAB_ENDPOINT_UNAUTHORIZED",
                        "inference_verified": False,
                        "endpoint_url": active_base_url,
                        "models": [],
                        "served_models": [],
                        "dns_attempts": dns_attempts,
                        "latency_ms": models_latency_ms,
                        "inference_latency_ms": 0.0,
                        "error": "HTTP 401 Unauthorized",
                    }
                if resp.status != 200:
                    return {
                        "online": False,
                        "ok": False,
                        "status_code": int(resp.status),
                        "http_status": int(resp.status),
                        "inference_verified": False,
                        "endpoint_url": active_base_url,
                        "models": [],
                        "served_models": [],
                        "dns_attempts": dns_attempts,
                        "latency_ms": models_latency_ms,
                        "inference_latency_ms": 0.0,
                        "error": f"HTTP {resp.status}",
                    }
                payload = json.loads(resp.read().decode("utf-8"))
                raw_data = payload.get("data", []) if isinstance(payload, dict) else []
                models = [
                    str(m.get("id", ""))
                    for m in raw_data
                    if isinstance(m, dict) and m.get("id")
                ]
                model_roots = {
                    str(m.get("id", "")): str(m.get("root", ""))
                    for m in raw_data
                    if isinstance(m, dict) and m.get("id") and m.get("root")
                }
                model_details = {
                    str(m.get("id", "")): dict(m)
                    for m in raw_data
                    if isinstance(m, dict) and m.get("id")
                }
                serves_model = _is_recognized_served_model(
                    models,
                    expected_model=expected_model,
                    model_roots=model_roots,
                )
                if not serves_model:
                    expected_items = _normalize_expected_model_list(expected_model)
                    target_label = ", ".join(expected_items) if expected_items else UNIFIED_MODEL_ALIAS
                    return {
                        "online": False,
                        "ok": False,
                        "alive": True,
                        "status_code": 200,
                        "http_status": 200,
                        "model_mismatch": bool(expected_items),
                        "expected_model": expected_model,
                        "inference_verified": False,
                        "endpoint_url": active_base_url,
                        "models": models,
                        "served_models": models,
                        "model_details": model_details,
                        "dns_attempts": dns_attempts,
                        "latency_ms": models_latency_ms,
                        "inference_latency_ms": 0.0,
                        "error": f"Model '{target_label}' not listed in {models}",
                    }
                break
        except urllib.error.HTTPError as http_err:
            models_latency_ms = round((time.perf_counter() - req_t0) * 1000.0, 2)
            if http_err.code == 401:
                return {
                    "online": False,
                    "ok": False,
                    "alive": False,
                    "auth_error": True,
                    "status_code": 401,
                    "http_status": 401,
                    "error_code": "COLAB_ENDPOINT_UNAUTHORIZED",
                    "inference_verified": False,
                    "endpoint_url": active_base_url,
                    "models": [],
                    "served_models": [],
                    "dns_attempts": dns_attempts,
                    "latency_ms": models_latency_ms,
                    "inference_latency_ms": 0.0,
                    "error": "HTTP 401 Unauthorized",
                }
            return {
                "online": False,
                "ok": False,
                "alive": False,
                "status_code": int(http_err.code),
                "http_status": int(http_err.code),
                "inference_verified": False,
                "endpoint_url": active_base_url,
                "models": [],
                "served_models": [],
                "dns_attempts": dns_attempts,
                "latency_ms": models_latency_ms,
                "inference_latency_ms": 0.0,
                "error": str(http_err),
            }
        except Exception as exc:
            models_latency_ms = round((time.perf_counter() - overall_t0) * 1000.0, 2)
            if not used_loopback_fallback and is_transient_tunnel_dns_error(exc, active_base_url):
                dns_attempts += 1
                now = time.monotonic()
                if now < dns_deadline:
                    wait_sec = min(retry_interval, max(0.01, dns_deadline - now))
                    _sleep_fn(wait_sec)
                    continue
                if allow_local_loopback_fallback:
                    fb_url = normalize_colab_endpoint_url(local_loopback_url)
                    if fb_url and fb_url != active_base_url:
                        active_base_url = fb_url
                        used_loopback_fallback = True
                        continue
            return {
                "online": False,
                "ok": False,
                "model_mismatch": False,
                "dns_error": is_transient_tunnel_dns_error(exc, active_base_url),
                "dns_attempts": dns_attempts,
                "loopback_fallback": used_loopback_fallback,
                "loopback_fallback_used": used_loopback_fallback,
                "inference_verified": False,
                "endpoint_url": active_base_url,
                "tunnel_url": base_url,
                "models": [],
                "served_models": [],
                "latency_ms": models_latency_ms,
                "inference_latency_ms": 0.0,
                "error": str(exc),
            }

    if not verify_inference:
        return {
            "online": True,
            "ok": True,
            "model_mismatch": False,
            "dns_attempts": dns_attempts,
            "loopback_fallback": used_loopback_fallback,
            "loopback_fallback_used": used_loopback_fallback,
            "inference_verified": False,
            "endpoint_url": active_base_url,
            "tunnel_url": base_url,
            "models": models,
            "served_models": models,
            "model_details": model_details,
            "latency_ms": models_latency_ms,
            "inference_latency_ms": 0.0,
            "error": None,
        }

    warmup_model = models[0] if models else UNIFIED_MODEL_ALIAS
    expected_items = _normalize_expected_model_list(expected_model)
    if expected_items:
        accepted_ids = resolve_expected_model_identifiers(expected_items[0])
        for m in models:
            if _normalize_served_model_candidate(m) & accepted_ids:
                warmup_model = m
                break
    elif UNIFIED_MODEL_ALIAS in models:
        warmup_model = UNIFIED_MODEL_ALIAS

    inf_timeout = max(0.5, float(inference_timeout_sec))
    warmup_temp = 0.0
    include_thinking_kwarg = True
    try:
        from platodoom.catalog import introspect_model_protocol

        probe_target_alias = expected_items[0] if expected_items else warmup_model
        proto = introspect_model_protocol(str(probe_target_alias), allow_remote=False)
        tmpl_fam = str(proto.get("template_family") or "").lower()
        include_thinking_kwarg = bool(proto.get("supports_enable_thinking_kwarg", True))
        if tmpl_fam == "deepseek_r1_distill":
            include_thinking_kwarg = False
            warmup_temp = 0.6
        elif tmpl_fam == "deepseek_v2_chat":
            include_thinking_kwarg = False
    except Exception:
        lower_wm = str(warmup_model).lower()
        if "r1-" in lower_wm or "deepseek-r1" in lower_wm:
            include_thinking_kwarg = False
            warmup_temp = 0.6
        elif "deepseek" in lower_wm:
            include_thinking_kwarg = False

    warmup_payload: Dict[str, Any] = {
        "model": warmup_model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 16,
        "temperature": warmup_temp,
        "stream": False,
    }
    if include_thinking_kwarg:
        warmup_payload["chat_template_kwargs"] = {"enable_thinking": False}
    warmup_body = json.dumps(warmup_payload).encode("utf-8")
    inf_req = urllib.request.Request(
        f"{active_base_url}/chat/completions",
        data=warmup_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "Sandcastle-Colab-Probe/1.0",
        },
        method="POST",
    )
    inf_t0 = time.perf_counter()
    try:
        with _open_http_request(inf_req, timeout=inf_timeout) as inf_resp:
            inf_latency_ms = round((time.perf_counter() - inf_t0) * 1000.0, 2)
            total_latency_ms = round((time.perf_counter() - overall_t0) * 1000.0, 2)
            if inf_resp.status != 200:
                return {
                    "online": False,
                    "ok": False,
                    "dns_attempts": dns_attempts,
                    "loopback_fallback": used_loopback_fallback,
                    "inference_verified": False,
                    "endpoint_url": active_base_url,
                    "tunnel_url": base_url,
                    "models": models,
                    "served_models": models,
                    "latency_ms": total_latency_ms,
                    "inference_latency_ms": inf_latency_ms,
                    "error": f"Inference warm-up probe HTTP {inf_resp.status}",
                }
            inf_payload = json.loads(inf_resp.read().decode("utf-8"))
            choices = inf_payload.get("choices") if isinstance(inf_payload, dict) else None
            if not isinstance(choices, list) or len(choices) == 0:
                return {
                    "online": False,
                    "ok": False,
                    "dns_attempts": dns_attempts,
                    "loopback_fallback": used_loopback_fallback,
                    "inference_verified": False,
                    "endpoint_url": active_base_url,
                    "tunnel_url": base_url,
                    "models": models,
                    "served_models": models,
                    "latency_ms": total_latency_ms,
                    "inference_latency_ms": inf_latency_ms,
                    "error": "Inference warm-up probe returned empty choices",
                }
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            msg = first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
            has_assistant_output = bool(
                str(msg.get("content") or "").strip()
                or str(msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                or msg.get("tool_calls")
                or str(first_choice.get("text") or "").strip()
            )
            if not has_assistant_output:
                return {
                    "online": False,
                    "ok": False,
                    "dns_attempts": dns_attempts,
                    "loopback_fallback": used_loopback_fallback,
                    "inference_verified": False,
                    "endpoint_url": active_base_url,
                    "tunnel_url": base_url,
                    "models": models,
                    "served_models": models,
                    "latency_ms": total_latency_ms,
                    "inference_latency_ms": inf_latency_ms,
                    "error": "Inference warm-up probe returned empty assistant message",
                }
            return {
                "online": True,
                "ok": True,
                "model_mismatch": False,
                "dns_attempts": dns_attempts,
                "loopback_fallback": used_loopback_fallback,
                "loopback_fallback_used": used_loopback_fallback,
                "inference_verified": True,
                "endpoint_url": active_base_url,
                "tunnel_url": base_url,
                "models": models,
                "served_models": models,
                "model_details": model_details,
                "latency_ms": total_latency_ms,
                "inference_latency_ms": inf_latency_ms,
                "error": None,
            }
    except urllib.error.HTTPError as http_err:
        inf_latency_ms = round((time.perf_counter() - inf_t0) * 1000.0, 2)
        total_latency_ms = round((time.perf_counter() - overall_t0) * 1000.0, 2)
        if http_err.code == 401:
            return {
                "online": False,
                "ok": False,
                "auth_error": True,
                "status_code": 401,
                "http_status": 401,
                "dns_attempts": dns_attempts,
                "loopback_fallback": used_loopback_fallback,
                "inference_verified": False,
                "endpoint_url": active_base_url,
                "tunnel_url": base_url,
                "models": models,
                "served_models": models,
                "model_details": model_details,
                "latency_ms": total_latency_ms,
                "inference_latency_ms": inf_latency_ms,
                "error": "HTTP 401 Unauthorized",
            }
        return {
            "online": False,
            "ok": False,
            "status_code": int(http_err.code),
            "http_status": int(http_err.code),
            "dns_attempts": dns_attempts,
            "loopback_fallback": used_loopback_fallback,
            "inference_verified": False,
            "endpoint_url": active_base_url,
            "tunnel_url": base_url,
            "models": models,
            "served_models": models,
            "latency_ms": total_latency_ms,
            "inference_latency_ms": inf_latency_ms,
            "error": f"Inference warm-up probe failed: {http_err}",
        }
    except Exception as exc:
        inf_latency_ms = round((time.perf_counter() - inf_t0) * 1000.0, 2)
        total_latency_ms = round((time.perf_counter() - overall_t0) * 1000.0, 2)
        return {
            "online": False,
            "ok": False,
            "dns_attempts": dns_attempts,
            "loopback_fallback": used_loopback_fallback,
            "inference_verified": False,
            "endpoint_url": active_base_url,
            "tunnel_url": base_url,
            "models": models,
            "served_models": models,
            "latency_ms": total_latency_ms,
            "inference_latency_ms": inf_latency_ms,
            "error": f"Inference warm-up probe failed: {exc}",
        }

ROLE_MODEL_ROUTING: Dict[str, str] = {
    # Families
    "scanner": UNIFIED_MODEL_ALIAS,
    "planner": UNIFIED_MODEL_ALIAS,
    "implementer": UNIFIED_MODEL_ALIAS,
    "tester": UNIFIED_MODEL_ALIAS,
    "merger": UNIFIED_MODEL_ALIAS,
    # Stage 1 & Discovery Scanners
    "codebase-mapper": UNIFIED_MODEL_ALIAS,
    "api-schema-mapper": UNIFIED_MODEL_ALIAS,
    "function-mapper": UNIFIED_MODEL_ALIAS,
    "library-installer": UNIFIED_MODEL_ALIAS,
    "mcp-provisioner": UNIFIED_MODEL_ALIAS,
    "spec-refiner": UNIFIED_MODEL_ALIAS,
    "repo-scanner": UNIFIED_MODEL_ALIAS,
    "code-scanner": UNIFIED_MODEL_ALIAS,
    "config-scanner": UNIFIED_MODEL_ALIAS,
    "test-scanner": UNIFIED_MODEL_ALIAS,
    "Test-scanner": UNIFIED_MODEL_ALIAS,
    "dep-scanner": UNIFIED_MODEL_ALIAS,
    "doc-scanner": UNIFIED_MODEL_ALIAS,
    "surface-scanner": UNIFIED_MODEL_ALIAS,
    "schema-scanner": UNIFIED_MODEL_ALIAS,
    "runtime-scanner": UNIFIED_MODEL_ALIAS,
    "security-scanner": UNIFIED_MODEL_ALIAS,
    "Security-scanner": UNIFIED_MODEL_ALIAS,
    "plan-scan-1": UNIFIED_MODEL_ALIAS,
    "plan-scan-2": UNIFIED_MODEL_ALIAS,
    # Stage 2 Planners & Coordinators (plan-1 and plan-2 stay on Gemini at live inference)
    "plan-1": GEMINI_HARNESS_MODEL,
    "plan-2": GEMINI_HARNESS_MODEL,
    "plan": UNIFIED_MODEL_ALIAS,
    "wayfinder": UNIFIED_MODEL_ALIAS,
    "planning-coordinator": UNIFIED_MODEL_ALIAS,
    "architecture-planner": UNIFIED_MODEL_ALIAS,
    "storage-planner": UNIFIED_MODEL_ALIAS,
    "stack-planner": UNIFIED_MODEL_ALIAS,
    "optimization-planner": UNIFIED_MODEL_ALIAS,
    "visual-planner": UNIFIED_MODEL_ALIAS,
    "implementation-coordinator": UNIFIED_MODEL_ALIAS,
    "verification-coordinator": UNIFIED_MODEL_ALIAS,
    "planner-2.1": UNIFIED_MODEL_ALIAS,
    "planner-2.2": UNIFIED_MODEL_ALIAS,
    "planner-2.3": UNIFIED_MODEL_ALIAS,
    "planner-2.4": UNIFIED_MODEL_ALIAS,
    "planner-2.5": UNIFIED_MODEL_ALIAS,
    # Stage 3 Implementers
    "architecture-implementer": UNIFIED_MODEL_ALIAS,
    "storage-implementer": UNIFIED_MODEL_ALIAS,
    "stack-implementer": UNIFIED_MODEL_ALIAS,
    "optimization-implementer": UNIFIED_MODEL_ALIAS,
    "visual-implementer": UNIFIED_MODEL_ALIAS,
    "implementer-1": UNIFIED_MODEL_ALIAS,
    "implementer-2": UNIFIED_MODEL_ALIAS,
    "implementer-3": UNIFIED_MODEL_ALIAS,
    "implementer-4": UNIFIED_MODEL_ALIAS,
    "implementer-5": UNIFIED_MODEL_ALIAS,
    # Stage 4 Testers & Stage 5 Merger
    "architecture-tester": UNIFIED_MODEL_ALIAS,
    "storage-tester": UNIFIED_MODEL_ALIAS,
    "stack-tester": UNIFIED_MODEL_ALIAS,
    "optimization-tester": UNIFIED_MODEL_ALIAS,
    "visual-tester": UNIFIED_MODEL_ALIAS,
    "tester-1": UNIFIED_MODEL_ALIAS,
    "tester-2": UNIFIED_MODEL_ALIAS,
    "tester-3": UNIFIED_MODEL_ALIAS,
    "tester-4": UNIFIED_MODEL_ALIAS,
    "tester-5": UNIFIED_MODEL_ALIAS,
    "reviewer": UNIFIED_MODEL_ALIAS,
    "branch-merger": UNIFIED_MODEL_ALIAS,
    # Validator stays on Gemini at live inference
    "validator": GEMINI_HARNESS_MODEL,
    "e2e-validator": GEMINI_HARNESS_MODEL,
}


def resolve_unified_model_for_role(role: str) -> str:
    """
    Resolves the live-inference model ID for a Sandcastle role under Spec 850:
    - 'plan-1', 'plan-2', 'validator', 'e2e-validator' -> Gemini ('gemini-2.5-pro')
    - All other coding roles -> 'qwen3.8-27b'
    """
    if role in ROLE_MODEL_ROUTING:
        return ROLE_MODEL_ROUTING[role]
    return UNIFIED_MODEL_ALIAS


class CheckpointArchitectureMetadata(TypedDict):
    source: str
    metadata_origin: str
    architectures: List[str]
    primary_architecture: str
    model_type: str
    text_model_type: str
    architecture_family: str
    loader_kind: str
    loader_class_name: str
    is_qwen35_family: bool
    vocab_size: Optional[int]
    hidden_size: Optional[int]
    num_hidden_layers: Optional[int]
    eos_token_ids: List[int]
    pad_token_id: Optional[int]
    quantization_config: Optional[Dict[str, Any]]
    quant_method: Optional[str]
    estimated_bf16_weights_gb: Optional[float]


_QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"})
_QWEN35_MOE_MODEL_TYPES = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})
_QWEN35_ARCHITECTURES = frozenset({"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"})
_CHECKPOINT_ARCH_CACHE_MAXSIZE = 256
_CHECKPOINT_ARCH_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}
_CHECKPOINT_ARCH_CACHE_LOCK = threading.Lock()


def clear_checkpoint_architecture_cache() -> None:
    """Clears the in-memory checkpoint architecture metadata cache."""
    with _CHECKPOINT_ARCH_CACHE_LOCK:
        _CHECKPOINT_ARCH_CACHE.clear()


def find_local_hf_snapshot_file(repo_id: str, filename: str = "config.json") -> Optional[Path]:
    """Resolves `filename` from local HuggingFace Hub snapshot cache with zero network I/O (< 0.4 ms)."""
    clean_repo = str(repo_id or "").strip().strip("/")
    if not clean_repo or "/" not in clean_repo or clean_repo.startswith((".", "/")):
        return None

    cache_roots: List[Path] = []
    env_hub_cache = os.environ.get("HF_HUB_CACHE")
    if env_hub_cache and env_hub_cache.strip():
        cache_roots.append(Path(env_hub_cache).expanduser())
    env_hf_home = os.environ.get("HF_HOME")
    if env_hf_home and env_hf_home.strip():
        cache_roots.append(Path(env_hf_home).expanduser() / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    cache_dir_name = "models--" + clean_repo.replace("/", "--")
    seen_roots: set[str] = set()
    for root in cache_roots:
        root_key = str(root)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        model_cache_dir = root / cache_dir_name
        if not model_cache_dir.is_dir():
            continue
        snapshots_dir = model_cache_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue

        ref_main = model_cache_dir / "refs" / "main"
        if ref_main.is_file():
            try:
                rev = ref_main.read_text(encoding="utf-8").strip()
                if rev:
                    ref_candidate = snapshots_dir / rev / filename
                    if ref_candidate.is_file():
                        return ref_candidate
            except Exception:
                pass

        try:
            snap_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            snap_dirs.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
            for snap in snap_dirs:
                candidate = snap / filename
                if candidate.is_file():
                    return candidate
        except Exception:
            continue
    return None


def _extract_config_dict_from_obj(model_or_config: Any) -> Optional[Dict[str, Any]]:
    """Extracts a normalized dictionary from a dict, PretrainedConfig, SimpleNamespace, or loaded model."""
    if model_or_config is None:
        return None
    if isinstance(model_or_config, dict):
        return model_or_config

    cfg_obj = getattr(model_or_config, "config", None)
    if cfg_obj is not None and cfg_obj is not model_or_config:
        return _extract_config_dict_from_obj(cfg_obj)

    # Reject raw model instances that lack a `.config` attribute
    if hasattr(model_or_config, "named_modules") or hasattr(model_or_config, "generate"):
        return None

    if hasattr(model_or_config, "to_dict") and callable(model_or_config.to_dict):
        try:
            as_dict = model_or_config.to_dict()
            if isinstance(as_dict, dict):
                return as_dict
        except Exception:
            pass

    known_attrs = (
        "architectures",
        "model_type",
        "text_config",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "eos_token_id",
        "pad_token_id",
        "quantization_config",
    )
    if any(hasattr(model_or_config, attr) for attr in known_attrs):
        extracted: Dict[str, Any] = {}
        for attr in known_attrs:
            if hasattr(model_or_config, attr):
                val = getattr(model_or_config, attr)
                if attr == "text_config" and val is not None and not isinstance(val, dict):
                    val = _extract_config_dict_from_obj(val) or {}
                extracted[attr] = val
        return extracted
    return None


def is_qwen35_architecture(model_or_config: Any) -> bool:
    """Strict boolean gate returning True IFF explicit configuration metadata identifies
    the checkpoint as belonging to `qwen3_5` or `qwen3_5_moe`.
    """
    cfg_dict = _extract_config_dict_from_obj(model_or_config)
    if not cfg_dict:
        return False

    top_model_type = str(cfg_dict.get("model_type") or "").strip().lower()
    text_cfg = cfg_dict.get("text_config")
    if not isinstance(text_cfg, dict) and text_cfg is not None:
        text_cfg = _extract_config_dict_from_obj(text_cfg)
    text_model_type = str((text_cfg or {}).get("model_type") or "").strip().lower() if isinstance(text_cfg, dict) else ""

    if top_model_type in _QWEN35_MODEL_TYPES or text_model_type in _QWEN35_MODEL_TYPES:
        return True

    raw_archs = cfg_dict.get("architectures") or []
    if isinstance(raw_archs, (list, tuple, set)):
        for arch in raw_archs:
            if str(arch).strip() in _QWEN35_ARCHITECTURES:
                return True
    return False


def _normalize_eos_ids(raw_eos: Any) -> List[int]:
    if isinstance(raw_eos, int) and not isinstance(raw_eos, bool):
        return [int(raw_eos)]
    if isinstance(raw_eos, (list, tuple, set)):
        out: List[int] = []
        for item in raw_eos:
            if isinstance(item, int) and not isinstance(item, bool) and int(item) not in out:
                out.append(int(item))
        return out
    return []


def _estimate_bf16_weights_gb_from_config(cfg_dict: Dict[str, Any]) -> Optional[float]:
    text_cfg = cfg_dict.get("text_config") if isinstance(cfg_dict.get("text_config"), dict) else cfg_dict
    hidden_size = text_cfg.get("hidden_size") or cfg_dict.get("hidden_size")
    num_layers = text_cfg.get("num_hidden_layers") or cfg_dict.get("num_hidden_layers")
    vocab_size = text_cfg.get("vocab_size") or cfg_dict.get("vocab_size")
    if not (
        isinstance(hidden_size, int)
        and hidden_size > 0
        and isinstance(num_layers, int)
        and num_layers > 0
        and isinstance(vocab_size, int)
        and vocab_size > 0
    ):
        return None

    tie_embeddings = bool(text_cfg.get("tie_word_embeddings", cfg_dict.get("tie_word_embeddings", False)))
    embed_params = (1 if tie_embeddings else 2) * vocab_size * hidden_size

    num_heads = text_cfg.get("num_attention_heads") or cfg_dict.get("num_attention_heads") or 32
    num_kv_heads = text_cfg.get("num_key_value_heads") or cfg_dict.get("num_key_value_heads") or num_heads
    head_dim = text_cfg.get("head_dim") or cfg_dict.get("head_dim") or max(1, int(hidden_size // max(1, int(num_heads))))
    q_dim = int(num_heads) * int(head_dim)
    kv_dim = int(num_kv_heads) * int(head_dim)
    attn_params_per_layer = hidden_size * (q_dim + 2 * kv_dim + q_dim)

    num_experts = (
        text_cfg.get("num_experts")
        or text_cfg.get("n_routed_experts")
        or text_cfg.get("num_local_experts")
        or cfg_dict.get("num_experts")
        or cfg_dict.get("n_routed_experts")
        or 0
    )
    moe_inter = (
        text_cfg.get("moe_intermediate_size")
        or cfg_dict.get("moe_intermediate_size")
        or text_cfg.get("intermediate_size")
        or cfg_dict.get("intermediate_size")
        or (hidden_size * 4)
    )
    dense_inter = text_cfg.get("intermediate_size") or cfg_dict.get("intermediate_size") or (hidden_size * 4)
    shared_inter = text_cfg.get("shared_expert_intermediate_size") or cfg_dict.get("shared_expert_intermediate_size") or 0

    if isinstance(num_experts, int) and num_experts > 1:
        mlp_params_per_layer = int(num_experts) * 3 * hidden_size * int(moe_inter) + (
            3 * hidden_size * int(shared_inter) if isinstance(shared_inter, int) and shared_inter > 0 else 0
        )
    else:
        mlp_params_per_layer = 3 * hidden_size * int(dense_inter)

    total_params = embed_params + int(num_layers) * (attn_params_per_layer + mlp_params_per_layer)
    vision_extra_gb = 0.85 if isinstance(cfg_dict.get("vision_config"), dict) else 0.0
    return round((total_params * 2.0) / (1024.0 ** 3) + vision_extra_gb, 2)


def _build_checkpoint_metadata_from_dicts(
    cfg_dict: Dict[str, Any],
    *,
    source: str,
    metadata_origin: str,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> CheckpointArchitectureMetadata:
    text_cfg = cfg_dict.get("text_config") if isinstance(cfg_dict.get("text_config"), dict) else {}
    raw_archs = cfg_dict.get("architectures") or []
    architectures = [str(a).strip() for a in raw_archs if str(a).strip()] if isinstance(raw_archs, (list, tuple)) else []

    model_type = str(cfg_dict.get("model_type") or "").strip()
    text_model_type = str(text_cfg.get("model_type") or model_type).strip()
    mt_lower = model_type.lower()
    tmt_lower = text_model_type.lower()
    arch_set = set(architectures)

    tok_cfg = tokenizer_config if isinstance(tokenizer_config, dict) else {}
    raw_bos = tok_cfg.get("bos_token")
    bos_str = raw_bos.get("content", "") if isinstance(raw_bos, dict) else str(raw_bos or "")
    chat_tmpl = str(tok_cfg.get("chat_template") or "")
    source_lower = str(source or "").lower()

    is_q35_moe = (
        mt_lower in _QWEN35_MOE_MODEL_TYPES
        or tmt_lower in _QWEN35_MOE_MODEL_TYPES
        or "Qwen3_5MoeForConditionalGeneration" in arch_set
    )
    is_q35_dense = (
        not is_q35_moe
        and (
            mt_lower in {"qwen3_5", "qwen3_5_text"}
            or tmt_lower in {"qwen3_5", "qwen3_5_text"}
            or "Qwen3_5ForConditionalGeneration" in arch_set
        )
    )
    is_q35 = is_q35_moe or is_q35_dense

    cls_name_str = str(cfg_dict.get("_class_name") or "").strip()
    explicit_diffusion = (
        cls_name_str.endswith("Pipeline")
        or str(cfg_dict.get("loader_kind") or "").strip() == "diffusion_pipeline"
        or mt_lower in {"flux2_klein", "z_image", "diffusion_image"}
        or any(a.endswith("Pipeline") for a in architectures)
    )
    if explicit_diffusion:
        is_q35 = False

    explicit_causal_lm = bool(architectures) and all(a.endswith("ForCausalLM") for a in architectures)
    explicit_vlm = any(
        a.endswith(("ForConditionalGeneration", "ForVision2Seq", "ImageTextToText"))
        for a in architectures
    )

    is_r1_marker = (
        "begin▁of▁sentence" in bos_str
        or "<｜User｜>" in chat_tmpl
        or "<｜Assistant｜>" in chat_tmpl
        or "deepseek-r1" in source_lower
        or "r1-distill" in source_lower
        or source_lower in {"r1-7b", "r1-14b", "r1-32b", "r1-70b"}
    )

    if explicit_diffusion:
        architecture_family = "diffusion_image"
        loader_kind = "diffusion_pipeline"
        loader_class_name = str(
            cls_name_str
            or cfg_dict.get("loader_class_name")
            or (architectures[0] if architectures else "DiffusionPipeline")
        )
        primary_arch = architectures[0] if architectures else loader_class_name
    elif is_q35_moe:
        architecture_family = "qwen3_5_moe"
        if explicit_causal_lm:
            loader_kind = "causal_lm"
            loader_class_name = "AutoModelForCausalLM"
        else:
            loader_kind = "conditional_generation"
            loader_class_name = "Qwen3_5MoeForConditionalGeneration"
        primary_arch = architectures[0] if architectures else "Qwen3_5MoeForConditionalGeneration"
    elif is_q35_dense:
        architecture_family = "qwen3_5"
        if explicit_causal_lm:
            loader_kind = "causal_lm"
            loader_class_name = "AutoModelForCausalLM"
        else:
            loader_kind = "conditional_generation"
            loader_class_name = "Qwen3_5ForConditionalGeneration"
        primary_arch = architectures[0] if architectures else "Qwen3_5ForConditionalGeneration"
    elif explicit_vlm:
        architecture_family = "multimodal_vlm"
        loader_kind = "conditional_generation"
        loader_class_name = "AutoModelForImageTextToText"
        primary_arch = architectures[0] if architectures else "AutoModelForImageTextToText"
    elif mt_lower == "qwen3_moe" or "Qwen3MoeForCausalLM" in arch_set:
        architecture_family = "qwen3_moe"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        primary_arch = architectures[0] if architectures else "Qwen3MoeForCausalLM"
    elif mt_lower == "qwen3" or "Qwen3ForCausalLM" in arch_set:
        architecture_family = "qwen3"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        primary_arch = architectures[0] if architectures else "Qwen3ForCausalLM"
    elif (mt_lower in {"qwen2", "llama"} or "Qwen2ForCausalLM" in arch_set or "LlamaForCausalLM" in arch_set) and is_r1_marker:
        architecture_family = "deepseek_r1"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        default_r1_arch = "LlamaForCausalLM" if (mt_lower == "llama" or "LlamaForCausalLM" in arch_set) else "Qwen2ForCausalLM"
        primary_arch = architectures[0] if architectures else default_r1_arch
    elif mt_lower == "qwen2" or "Qwen2ForCausalLM" in arch_set:
        architecture_family = "qwen2"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        primary_arch = architectures[0] if architectures else "Qwen2ForCausalLM"
    elif mt_lower == "deepseek_v2" or "DeepseekV2ForCausalLM" in arch_set:
        architecture_family = "deepseek_v2"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        primary_arch = architectures[0] if architectures else "DeepseekV2ForCausalLM"
    elif mt_lower == "deepseek_v3" or "DeepseekV3ForCausalLM" in arch_set:
        architecture_family = "deepseek_v3"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        primary_arch = architectures[0] if architectures else "DeepseekV3ForCausalLM"
    elif mt_lower == "llama" or "LlamaForCausalLM" in arch_set:
        architecture_family = "llama"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        primary_arch = architectures[0] if architectures else "LlamaForCausalLM"
    elif mt_lower in {"mistral", "mixtral"} or "MistralForCausalLM" in arch_set or "MixtralForCausalLM" in arch_set:
        architecture_family = "mistral"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        primary_arch = architectures[0] if architectures else "MistralForCausalLM"
    else:
        architecture_family = mt_lower if mt_lower else "causal_lm"
        if architecture_family not in {
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3",
            "qwen3_moe",
            "qwen2",
            "deepseek_r1",
            "deepseek_v2",
            "deepseek_v3",
            "llama",
            "mistral",
            "multimodal_vlm",
        }:
            architecture_family = "causal_lm"
        loader_kind = "causal_lm"
        loader_class_name = "AutoModelForCausalLM"
        primary_arch = architectures[0] if architectures else "AutoModelForCausalLM"

    raw_vocab = text_cfg.get("vocab_size", cfg_dict.get("vocab_size"))
    vocab_size = int(raw_vocab) if isinstance(raw_vocab, int) and raw_vocab > 0 else None
    raw_hidden = text_cfg.get("hidden_size", cfg_dict.get("hidden_size"))
    hidden_size = int(raw_hidden) if isinstance(raw_hidden, int) and raw_hidden > 0 else None
    raw_layers = text_cfg.get("num_hidden_layers", cfg_dict.get("num_hidden_layers"))
    num_hidden_layers = int(raw_layers) if isinstance(raw_layers, int) and raw_layers > 0 else None

    eos_ids = _normalize_eos_ids(text_cfg.get("eos_token_id", cfg_dict.get("eos_token_id")))
    raw_pad = text_cfg.get("pad_token_id", cfg_dict.get("pad_token_id"))
    pad_token_id = int(raw_pad) if isinstance(raw_pad, int) and not isinstance(raw_pad, bool) else None

    raw_qcfg = cfg_dict.get("quantization_config")
    quantization_config = dict(raw_qcfg) if isinstance(raw_qcfg, dict) and raw_qcfg else None
    quant_method = str(quantization_config.get("quant_method") or "").strip().lower() or None if quantization_config else None
    raw_td = cfg_dict.get("torch_dtype") or text_cfg.get("torch_dtype")
    torch_dtype = str(raw_td).strip() if raw_td else None

    return {
        "source": str(source),
        "metadata_origin": metadata_origin,
        "architectures": architectures,
        "primary_architecture": primary_arch,
        "model_type": model_type,
        "text_model_type": text_model_type,
        "architecture_family": architecture_family,
        "loader_kind": loader_kind,
        "loader_class_name": loader_class_name,
        "is_qwen35_family": bool(is_q35),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "eos_token_ids": eos_ids,
        "pad_token_id": pad_token_id,
        "quantization_config": quantization_config,
        "quant_method": quant_method,
        "torch_dtype": torch_dtype,
        "estimated_bf16_weights_gb": _estimate_bf16_weights_gb_from_config(cfg_dict),
    }


def _store_arch_cache(cache_key: Tuple[str, int], meta: Dict[str, Any]) -> None:
    with _CHECKPOINT_ARCH_CACHE_LOCK:
        if len(_CHECKPOINT_ARCH_CACHE) >= _CHECKPOINT_ARCH_CACHE_MAXSIZE:
            oldest_key = next(iter(_CHECKPOINT_ARCH_CACHE))
            _CHECKPOINT_ARCH_CACHE.pop(oldest_key, None)
        _CHECKPOINT_ARCH_CACHE[cache_key] = dict(meta)


def inspect_checkpoint_architecture(
    source: Any,
    *,
    token: Optional[str] = None,
    allow_remote: bool = False,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> CheckpointArchitectureMetadata:
    """Deterministically inspects checkpoint configuration metadata (< 50ms cold, < 0.05ms cached)
    to determine architecture family, loader class, quantization metadata, and Qwen3.5 gating.
    """
    in_mem_cfg = _extract_config_dict_from_obj(source) if not isinstance(source, (str, Path)) else None
    if in_mem_cfg is not None:
        return _build_checkpoint_metadata_from_dicts(
            in_mem_cfg,
            source="<in-memory-config>",
            metadata_origin="in_memory_config",
            tokenizer_config=tokenizer_config,
        )

    if not isinstance(source, (str, Path)):
        return _build_checkpoint_metadata_from_dicts(
            {},
            source=str(source),
            metadata_origin="unverified_local_path",
            tokenizer_config=tokenizer_config,
        )

    raw_str = str(source).strip()
    if not raw_str:
        return _build_checkpoint_metadata_from_dicts(
            {},
            source="",
            metadata_origin="unverified_local_path",
            tokenizer_config=tokenizer_config,
        )

    candidate_path = Path(raw_str).expanduser()
    cfg_file: Optional[Path] = None
    if candidate_path.is_file() and candidate_path.name in ("config.json", "model_index.json"):
        cfg_file = candidate_path
    elif candidate_path.is_dir():
        if (candidate_path / "config.json").is_file():
            cfg_file = candidate_path / "config.json"
        elif (candidate_path / "model_index.json").is_file():
            cfg_file = candidate_path / "model_index.json"

    if cfg_file is not None:
        resolved_cfg = cfg_file.resolve()
        mtime_ns = resolved_cfg.stat().st_mtime_ns
        cache_key = (str(resolved_cfg), mtime_ns)
        if tokenizer_config is None:
            with _CHECKPOINT_ARCH_CACHE_LOCK:
                cached = _CHECKPOINT_ARCH_CACHE.get(cache_key)
            if cached is not None:
                return dict(cached)  # type: ignore[return-value]

        cfg_dict = json.loads(resolved_cfg.read_text(encoding="utf-8"))
        if isinstance(cfg_dict, dict) and resolved_cfg.name == "config.json":
            idx_path = resolved_cfg.parent / "model_index.json"
            if idx_path.is_file():
                try:
                    idx_loaded = json.loads(idx_path.read_text(encoding="utf-8"))
                    if isinstance(idx_loaded, dict):
                        for k, v in idx_loaded.items():
                            cfg_dict.setdefault(k, v)
                except Exception:
                    pass
        tok_dict = tokenizer_config
        if tok_dict is None:
            tok_path = resolved_cfg.parent / "tokenizer_config.json"
            if tok_path.is_file():
                try:
                    tok_dict = json.loads(tok_path.read_text(encoding="utf-8"))
                except Exception:
                    tok_dict = None
        meta = _build_checkpoint_metadata_from_dicts(
            cfg_dict if isinstance(cfg_dict, dict) else {},
            source=str(resolved_cfg.parent),
            metadata_origin="local_dir",
            tokenizer_config=tok_dict,
        )
        if tokenizer_config is None:
            _store_arch_cache(cache_key, meta)
        return meta

    looks_like_local_fs_path = (
        raw_str.startswith(("/", "./", "../", "~"))
        or os.path.isabs(raw_str)
        or candidate_path.exists()
    )
    if not looks_like_local_fs_path and "/" in raw_str:
        snap_cfg = find_local_hf_snapshot_file(raw_str, "config.json") or find_local_hf_snapshot_file(raw_str, "model_index.json")
        if snap_cfg is not None:
            resolved_snap = snap_cfg.resolve()
            mtime_ns = resolved_snap.stat().st_mtime_ns
            cache_key = (str(resolved_snap), mtime_ns)
            if tokenizer_config is None:
                with _CHECKPOINT_ARCH_CACHE_LOCK:
                    cached = _CHECKPOINT_ARCH_CACHE.get(cache_key)
                if cached is not None:
                    return dict(cached)  # type: ignore[return-value]
            cfg_dict = json.loads(resolved_snap.read_text(encoding="utf-8"))
            tok_dict = tokenizer_config
            if tok_dict is None:
                snap_tok = resolved_snap.parent / "tokenizer_config.json"
                if snap_tok.is_file():
                    try:
                        tok_dict = json.loads(snap_tok.read_text(encoding="utf-8"))
                    except Exception:
                        tok_dict = None
            meta = _build_checkpoint_metadata_from_dicts(
                cfg_dict if isinstance(cfg_dict, dict) else {},
                source=raw_str,
                metadata_origin="hf_cache",
                tokenizer_config=tok_dict,
            )
            if tokenizer_config is None:
                _store_arch_cache(cache_key, meta)
            return meta

        try:
            from platodoom.catalog import _load_local_or_cached_metadata

            try:
                loaded_meta_tuple = _load_local_or_cached_metadata(raw_str, allow_remote=False)  # type: ignore[call-arg]
            except TypeError:
                loaded_meta_tuple = _load_local_or_cached_metadata(raw_str)
            cat_mod_cfg = (
                loaded_meta_tuple[0]
                if isinstance(loaded_meta_tuple, tuple) and len(loaded_meta_tuple) >= 1
                else {}
            )
            cat_tok_cfg = (
                loaded_meta_tuple[1]
                if isinstance(loaded_meta_tuple, tuple) and len(loaded_meta_tuple) >= 2
                else None
            )
            if isinstance(cat_mod_cfg, dict) and cat_mod_cfg:
                meta = _build_checkpoint_metadata_from_dicts(
                    cat_mod_cfg,
                    source=raw_str,
                    metadata_origin="bundled_catalog",
                    tokenizer_config=tokenizer_config if tokenizer_config is not None else cat_tok_cfg,
                )
                return meta
        except Exception:
            pass

        if allow_remote:
            remote_key = (f"remote::{raw_str}", 0)
            if tokenizer_config is None:
                with _CHECKPOINT_ARCH_CACHE_LOCK:
                    cached = _CHECKPOINT_ARCH_CACHE.get(remote_key)
                if cached is not None:
                    return dict(cached)  # type: ignore[return-value]
            remote_cfg_dict: Optional[Dict[str, Any]] = None
            try:
                from transformers import AutoConfig

                ac_kwargs: Dict[str, Any] = {"trust_remote_code": True}
                if token:
                    ac_kwargs["token"] = token
                cfg_obj = AutoConfig.from_pretrained(raw_str, **ac_kwargs)
                remote_cfg_dict = _extract_config_dict_from_obj(cfg_obj)
            except Exception:
                try:
                    remote_cfg_dict = HuggingFaceConfigClient(model_id=raw_str, token=token).fetch_model_config()
                except Exception:
                    remote_cfg_dict = None
            if isinstance(remote_cfg_dict, dict) and remote_cfg_dict:
                meta = _build_checkpoint_metadata_from_dicts(
                    remote_cfg_dict,
                    source=raw_str,
                    metadata_origin="remote_autoconfig",
                    tokenizer_config=tokenizer_config,
                )
                if tokenizer_config is None:
                    _store_arch_cache(remote_key, meta)
                return meta

    return _build_checkpoint_metadata_from_dicts(
        {},
        source=raw_str,
        metadata_origin="unverified_local_path",
        tokenizer_config=tokenizer_config,
    )


def validate_qwen35_architecture(config_dict: Dict[str, Any]) -> bool:
    """
    Validates that a model configuration dict belongs to the Qwen3.5 dense or MoE architecture
    family (`qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`,
    `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`) with Qwen3.5
    vocabulary (`248044` or `248320`).
    """
    if not isinstance(config_dict, dict):
        return False
    if not is_qwen35_architecture(config_dict):
        return False

    text_cfg = config_dict.get("text_config") if isinstance(config_dict.get("text_config"), dict) else {}
    vocab_size = text_cfg.get("vocab_size", config_dict.get("vocab_size", 0))
    return vocab_size in (248044, 248320)


class CompletionUsage(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionChoice(BaseModel):
    model_config = ConfigDict(strict=True)

    index: int = 0
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    object: str = "chat.completion"
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class VLLMMultiLoRAClient:
    """
    OpenAI-compatible client for Google Colab Pro+ vLLM serving.
    Supports both unified Spec 850 serving ('qwen3.8-27b' with
    plan-1, plan-2, and validator kept on Gemini at live inference) and legacy
    Multi-LoRA adapter routing.
    """

    UNIFIED_MODEL_ALIAS: str = UNIFIED_MODEL_ALIAS
    BASE_MODEL_ID: str = BASE_MODEL_ID
    ROLE_MODEL_ROUTING: Dict[str, str] = ROLE_MODEL_ROUTING

    DEFAULT_ADAPTER_MAP: Dict[str, str] = {
        # Scanners
        "codebase-mapper": "distilled:scanners-lora",
        "api-schema-mapper": "distilled:scanners-lora",
        "function-mapper": "distilled:scanners-lora",
        "library-installer": "distilled:scanners-lora",
        "mcp-provisioner": "distilled:scanners-lora",
        # Planners
        "plan-1": "distilled:planners-lora",
        "plan-2": "distilled:planners-lora",
        "planner-2.1": "distilled:planners-lora",
        "planner-2.2": "distilled:planners-lora",
        "planner-2.3": "distilled:planners-lora",
        "planner-2.4": "distilled:planners-lora",
        "planner-2.5": "distilled:planners-lora",
        # Implementers
        "implementer": "distilled:implementers-lora",
        "implementer-1": "distilled:implementers-lora",
        "implementer-2": "distilled:implementers-lora",
        "implementer-3": "distilled:implementers-lora",
        "implementer-4": "distilled:implementers-lora",
        "implementer-5": "distilled:implementers-lora",
        # Testers & Merger
        "tester": "distilled:testers-merger-lora",
        "tester-1": "distilled:testers-merger-lora",
        "tester-2": "distilled:testers-merger-lora",
        "tester-3": "distilled:testers-merger-lora",
        "tester-4": "distilled:testers-merger-lora",
        "tester-5": "distilled:testers-merger-lora",
        "merger": "distilled:testers-merger-lora",
        # Validator
        "validator": "distilled:validator-lora",
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        unified: bool = False,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VLLM_API_BASE") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY") or "EMPTY"
        self.timeout = timeout
        self.max_retries = max_retries
        self.unified = unified

    def resolve_unified_model_for_role(self, role: str) -> str:
        """Resolves the Spec 850 unified model or Gemini live-inference routing for a role."""
        return resolve_unified_model_for_role(role)

    def resolve_model_for_role(self, role: str) -> str:
        """Resolves the canonical model/adapter ID for a Sandcastle role."""
        if self.unified:
            return self.resolve_unified_model_for_role(role)
        if role in self.DEFAULT_ADAPTER_MAP:
            return self.DEFAULT_ADAPTER_MAP[role]
        if role.startswith("distilled:") or role == UNIFIED_MODEL_ALIAS:
            return role
        if role in ROLE_MODEL_ROUTING:
            return ROLE_MODEL_ROUTING[role]
        # Default fallback to implementers adapter
        return "distilled:implementers-lora"

    def health_check(self) -> bool:
        """Verifies vLLM server liveness via /health endpoint."""
        url = f"{self.base_url}/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with _open_http_request(req, timeout=5) as response:
                return response.status == 200
        except Exception as err:
            logger.warning("vLLM health check probe failed: %s", err)
            return False

    def list_models(self) -> List[str]:
        """Lists active models and LoRA adapters served by vLLM."""
        url = f"{self.base_url}/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with _open_http_request(req, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return [m.get("id", "") for m in payload.get("data", [])]
        except Exception as err:
            raise RuntimeError(f"Failed to query vLLM models at {url}: {err}") from err

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        role_or_model: str = UNIFIED_MODEL_ALIAS,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        enable_thinking: bool = False,
    ) -> ChatCompletionResponse:
        """
        Executes a synchronous OpenAI-compatible ChatML completion request with
        enable_thinking=False for Qwen/Qwen3.5-9B direct-action parity.
        Enforces exponential backoff on transient network faults (max 3 retries).
        """
        if role_or_model.startswith("distilled:") or role_or_model == UNIFIED_MODEL_ALIAS:
            model = role_or_model
        else:
            model = self.resolve_model_for_role(role_or_model)

        normalized_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            m_copy = dict(msg)
            raw_tcs = m_copy.get("tool_calls")
            if isinstance(raw_tcs, list):
                norm_tcs: List[Dict[str, Any]] = []
                for tc in raw_tcs:
                    if not isinstance(tc, dict):
                        continue
                    tc_copy = dict(tc)
                    tc_copy.pop("arguments", None)
                    tc_copy.pop("name", None)
                    fn_obj = tc_copy.get("function")
                    if isinstance(fn_obj, dict):
                        fn_copy = dict(fn_obj)
                        raw_args = fn_copy.get("arguments")
                        if isinstance(raw_args, (dict, list)):
                            fn_copy["arguments"] = json.dumps(raw_args, ensure_ascii=False)
                        elif raw_args is None:
                            fn_copy["arguments"] = "{}"
                        else:
                            fn_copy["arguments"] = str(raw_args)
                        tc_copy["function"] = fn_copy
                    norm_tcs.append(tc_copy)
                m_copy["tool_calls"] = norm_tcs
            normalized_messages.append(m_copy)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if tools:
            payload["tools"] = tools
        if stop:
            payload["stop"] = stop

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req_url = f"{self.base_url}/chat/completions"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                request = urllib.request.Request(req_url, data=body, headers=headers, method="POST")
                with _open_http_request(request, timeout=self.timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    choices: List[CompletionChoice] = []
                    for c in resp_data.get("choices", []):
                        c_msg = dict(c.get("message") or {})
                        if not c_msg.get("reasoning_content") and c_msg.get("reasoning"):
                            c_msg["reasoning_content"] = c_msg.get("reasoning")
                        choices.append(
                            CompletionChoice(
                                index=c.get("index", 0),
                                message=c_msg,
                                finish_reason=c.get("finish_reason"),
                            )
                        )
                    raw_usage = resp_data.get("usage", {})
                    usage = CompletionUsage(
                        prompt_tokens=raw_usage.get("prompt_tokens", 0),
                        completion_tokens=raw_usage.get("completion_tokens", 0),
                        total_tokens=raw_usage.get("total_tokens", 0),
                    )
                    return ChatCompletionResponse(
                        id=resp_data.get("id", f"vllm-{int(time.time())}"),
                        object=resp_data.get("object", "chat.completion"),
                        model=resp_data.get("model", model),
                        choices=choices,
                        usage=usage,
                    )
            except urllib.error.HTTPError as http_err:
                err_content = http_err.read().decode("utf-8", errors="replace")
                logger.error(
                    "HTTP %s from vLLM chat completions on attempt %d/%d: %s",
                    http_err.code,
                    attempt,
                    self.max_retries,
                    err_content,
                )
                last_error = RuntimeError(f"vLLM HTTP {http_err.code}: {err_content}")
                if http_err.code in (400, 404, 422):
                    raise last_error
            except Exception as conn_err:
                logger.warning(
                    "Connection error to vLLM on attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    conn_err,
                )
                last_error = conn_err

            if attempt < self.max_retries:
                backoff_sec = 2 ** attempt
                time.sleep(backoff_sec)

        raise RuntimeError(
            f"Exhausted {self.max_retries} attempts connecting to vLLM at {req_url}: {last_error}"
        ) from last_error


class VLLMUnifiedClient(VLLMMultiLoRAClient):
    """
    Spec 850 Unified Qwen/Qwen3.5-9B vLLM client ('qwen3.8-27b').
    Routes all distilled coding roles to 'qwen3.8-27b' while keeping
    'plan-1', 'plan-2', and 'validator' / 'e2e-validator' on Gemini at live inference.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            unified=True,
        )


class GitHubAPIClient:
    """
    Authenticated GitHub API Client supporting direct REST queries and GitHub CLI (`gh`).
    Implements rate-limit tracking, dynamic account owner resolution, default branch resolution,
    and commit diff extraction.
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, token: Optional[str] = None, owner: Optional[str] = None) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not self.token:
            try:
                res = subprocess.run(
                    ["gh", "auth", "token"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.token = res.stdout.strip()
            except Exception:
                self.token = None
        self.owner = owner or os.environ.get("GITHUB_OWNER")

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Sandcastle-Distillation-Fleet/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get_authenticated_owner(self) -> str:
        """Resolves the GitHub owner via --github-owner / GITHUB_OWNER or authenticated user."""
        if self.owner:
            return self.owner
        try:
            res = subprocess.run(
                ["gh", "api", "/user", "--jq", ".login"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            login = res.stdout.strip()
            if login:
                self.owner = login
                return login
        except Exception:
            pass
        url = f"{self.BASE_URL}/user"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.owner = data.get("login", "")
            return self.owner

    def get_rate_limit(self) -> Dict[str, Any]:
        """Queries GitHub API rate limits."""
        url = f"{self.BASE_URL}/rate_limit"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as err:
            logger.warning("Failed to fetch rate limit from GitHub API: %s", err)
            return {"rate": {"remaining": -1, "limit": -1}}

    def get_repo_metadata(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Retrieves repository details including dynamic default_branch.
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_commit_with_patch(self, owner: str, repo: str, ref: str) -> Dict[str, Any]:
        """
        Fetches commit details including file-level stats and unified diff hunks.
        Schema matches GET /repos/{owner}/{repo}/commits/{ref}.
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits/{ref}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_commits(
        self,
        owner: str,
        repo: str,
        sha: Optional[str] = None,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """Lists recent commits on the repository default branch or given ref."""
        params = {"per_page": str(per_page)}
        if sha:
            params["sha"] = sha
        query_string = urllib.parse.urlencode(params)
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits?{query_string}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """Lists pull requests for a repository (GET /repos/{owner}/{repo}/pulls)."""
        params = {"state": state, "per_page": str(per_page)}
        query_string = urllib.parse.urlencode(params)
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls?{query_string}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, list) else []

    def get_issue(self, owner: str, repo: str, issue_number: int) -> Dict[str, Any]:
        """Fetches a single issue or pull request metadata (GET /repos/{owner}/{repo}/issues/{issue_number})."""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues/{int(issue_number)}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, dict) else {}

    def list_pr_reviews(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """Lists reviews on a pull request (GET /repos/{owner}/{repo}/pulls/{pr_number}/reviews)."""
        query_string = urllib.parse.urlencode({"per_page": str(per_page)})
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{int(pr_number)}/reviews?{query_string}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, list) else []

    def list_pr_comments(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """Lists inline code review comments on a pull request diff (GET /repos/{owner}/{repo}/pulls/{pr_number}/comments)."""
        query_string = urllib.parse.urlencode({"per_page": str(per_page)})
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{int(pr_number)}/comments?{query_string}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, list) else []

    def list_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """Lists timeline comments on an issue or PR (GET /repos/{owner}/{repo}/issues/{issue_number}/comments)."""
        query_string = urllib.parse.urlencode({"per_page": str(per_page)})
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues/{int(issue_number)}/comments?{query_string}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, list) else []

    def list_check_runs(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """Lists GitHub Actions check runs for a commit SHA (GET /repos/{owner}/{repo}/commits/{commit_sha}/check-runs)."""
        query_string = urllib.parse.urlencode({"per_page": str(per_page)})
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits/{commit_sha}/check-runs?{query_string}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("check_runs"), list):
                return list(data["check_runs"])
            return data if isinstance(data, list) else []

    def get_job_logs(self, owner: str, repo: str, job_id: int) -> str:
        """Downloads raw GitHub Actions workflow job logs (GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs)."""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/actions/jobs/{int(job_id)}/logs"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")


class HuggingFaceConfigClient:
    """
    Validates base model hyperparameters and tokenizer configurations on Hugging Face Hub
    or local Hugging Face cache.
    Primary target: Qwen/Qwen3.5-9B (Qwen3_5ForConditionalGeneration, vocab_size=248044).
    """

    HUB_BASE_URL = "https://huggingface.co"

    EXPECTED_TARGET_MODULES = QWEN35_TARGET_MODULES

    def __init__(self, model_id: str = BASE_MODEL_ID, token: Optional[str] = None) -> None:
        self.model_id = model_id
        self.token = token

    def _local_snapshot_file(self, filename: str) -> Optional[Path]:
        return find_local_hf_snapshot_file(self.model_id, filename)

    def fetch_model_config(self) -> Dict[str, Any]:
        """Fetches config.json from local HF cache or Hugging Face Hub raw endpoint."""
        local_cfg = self._local_snapshot_file("config.json")
        if local_cfg is not None:
            return json.loads(local_cfg.read_text(encoding="utf-8"))
        url = f"{self.HUB_BASE_URL}/{self.model_id}/raw/main/config.json"
        headers: Dict[str, str] = {"User-Agent": "Sandcastle/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def fetch_tokenizer_config(self) -> Dict[str, Any]:
        """Fetches tokenizer_config.json from local HF cache or Hugging Face Hub raw endpoint."""
        local_tok = self._local_snapshot_file("tokenizer_config.json")
        if local_tok is not None:
            return json.loads(local_tok.read_text(encoding="utf-8"))
        url = f"{self.HUB_BASE_URL}/{self.model_id}/raw/main/tokenizer_config.json"
        headers: Dict[str, str] = {"User-Agent": "Sandcastle/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def validate_architecture(self, config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validates checkpoint hyperparameters strictly from `config.json` metadata
        across Qwen3.5 (dense & MoE), Qwen3 (dense & MoE), Qwen2/Qwen2.5, DeepSeek-R1/V2/V3,
        Llama, Mistral, and general causal language models without model nickname guessing.
        """
        cfg = config_dict if config_dict is not None else self.fetch_model_config()
        if not isinstance(cfg, dict) or not (
            cfg.get("model_type")
            or cfg.get("architectures")
            or cfg.get("text_config")
            or cfg.get("vocab_size")
        ):
            raise ValueError(
                f"Invalid checkpoint configuration for '{self.model_id}': missing architecture metadata."
            )
        meta = inspect_checkpoint_architecture(cfg)
        text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
        raw_vocab_size = int(meta.get("vocab_size") or 0)
        hidden_size = int(meta.get("hidden_size") or 0)
        num_layers = int(meta.get("num_hidden_layers") or 0)
        is_q35 = bool(meta.get("is_qwen35_family"))

        if is_q35:
            eos_token_id = text_cfg.get("eos_token_id", cfg.get("eos_token_id"))
            vocab_size = (
                248044
                if raw_vocab_size in (248044, 248320) and (eos_token_id == 248044 or eos_token_id is None)
                else raw_vocab_size
            )
            is_valid_35 = validate_qwen35_architecture(cfg)
            target_modules = list(QWEN35_TARGET_MODULES)
            is_valid_q25_14b = is_valid_35
        else:
            vocab_size = raw_vocab_size
            is_valid_35 = False
            target_modules = list(LEGACY_QWEN25_TARGET_MODULES)
            is_valid_q25_14b = bool(vocab_size == 152064 and hidden_size == 5120 and num_layers == 48)

        is_valid_arch = bool(raw_vocab_size > 0 and hidden_size > 0 and num_layers > 0)
        return {
            "model_id": self.model_id,
            "architecture_family": meta["architecture_family"],
            "loader_kind": meta["loader_kind"],
            "loader_class_name": meta["loader_class_name"],
            "is_qwen35_family": is_q35,
            "model_type": meta["model_type"] or ("qwen3_5" if is_q35 else "causal_lm"),
            "architectures": meta["architectures"] or [meta["primary_architecture"]],
            "vocab_size": vocab_size,
            "padded_vocab_size": raw_vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "eos_token_ids": list(meta["eos_token_ids"]),
            "pad_token_id": meta["pad_token_id"],
            "quantization_config": meta["quantization_config"],
            "is_valid_architecture": is_valid_arch,
            "is_valid_qwen35_9b": is_valid_35,
            "is_valid_qwen25_coder_14b": is_valid_q25_14b,
            "target_modules": target_modules,
        }


def _resolve_dynamic_gcs_defaults(
    project_id: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    resolved_project = project_id.strip() if isinstance(project_id, str) and project_id.strip() else None
    resolved_bucket = bucket_name.strip() if isinstance(bucket_name, str) and bucket_name.strip() else None

    if not resolved_bucket:
        env_bucket = os.environ.get("PLATODOOM_GCS_BUCKET") or os.environ.get("GCS_BUCKET_NAME")
        if env_bucket and env_bucket.strip():
            resolved_bucket = env_bucket.strip()
        else:
            env_uri = os.environ.get("PLATODOOM_STORAGE_URI") or ""
            if env_uri.startswith("gs://"):
                uri_body = env_uri[len("gs://") :].strip("/")
                if uri_body:
                    resolved_bucket = uri_body.split("/", 1)[0].strip()

    if not resolved_project:
        env_proj = os.environ.get("PLATODOOM_GCS_PROJECT") or os.environ.get("GCP_PROJECT_ID")
        if env_proj and env_proj.strip():
            resolved_project = env_proj.strip()

    if not resolved_bucket or not resolved_project:
        cfg_dir = os.environ.get("PLATODOOM_CONFIG_DIR") or os.environ.get("PLATODOOM_HOME")
        cfg_path = (
            Path(cfg_dir).expanduser().resolve() / "config.json"
            if cfg_dir and str(cfg_dir).strip()
            else Path.home() / ".platodoom" / "config.json"
        )
        if cfg_path.is_file():
            try:
                cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
                storage_cfg = cfg_data.get("storage") if isinstance(cfg_data, dict) else {}
                if isinstance(storage_cfg, dict):
                    if not resolved_bucket and storage_cfg.get("gcs_bucket"):
                        resolved_bucket = str(storage_cfg["gcs_bucket"]).strip()
                    if not resolved_project and storage_cfg.get("gcs_project_id"):
                        resolved_project = str(storage_cfg["gcs_project_id"]).strip()
            except Exception:
                pass

    if not resolved_bucket or not resolved_project:
        repo_root = resolve_project_root(Path(__file__).resolve().parent.parent)
        for cand_manifest in (
            Path.cwd() / "datasets" / "v1" / "manifest_v1.json",
            repo_root / "datasets" / "v1" / "manifest_v1.json",
        ):
            if cand_manifest.is_file():
                try:
                    m_data = json.loads(cand_manifest.read_text(encoding="utf-8"))
                    m_bucket = str(m_data.get("bucket") or "").strip() if isinstance(m_data, dict) else ""
                    if m_bucket and not resolved_bucket:
                        resolved_bucket = m_bucket
                    if not resolved_project and resolved_bucket and "-" in resolved_bucket:
                        b_parts = resolved_bucket.split("-")
                        if len(b_parts) >= 3 and b_parts[-1].isdigit():
                            resolved_project = "-".join(b_parts[-3:])
                    break
                except Exception:
                    continue

    return resolved_project, resolved_bucket


class GCSClientWrapper:
    """
    High-level wrapper over Google Cloud Storage API.
    Provides bucket verification, object listing, and integrity verification against manifests.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ) -> None:
        resolved_proj, resolved_bkt = _resolve_dynamic_gcs_defaults(
            project_id=project_id,
            bucket_name=bucket_name,
        )
        self.project_id = resolved_proj
        self.bucket_name = resolved_bkt
        self._storage_client = None

    def _get_client(self) -> Any:
        if self._storage_client is None:
            from google.cloud import storage
            self._storage_client = storage.Client(project=self.project_id)
        return self._storage_client

    def ensure_bucket_exists(self, location: str = "us-central1") -> bool:
        """
        Checks if the target bucket exists. If not, provisions it with Uniform Bucket-Level Access.
        """
        client = self._get_client()
        bucket = client.bucket(self.bucket_name)
        if bucket.exists():
            return True

        logger.info("Bucket %s does not exist; creating in location %s...", self.bucket_name, location)
        client.create_bucket(self.bucket_name, location=location)
        return True

    def list_blobs(self, prefix: str = "") -> List[Dict[str, Any]]:
        """Lists blobs in the target bucket under the given prefix."""
        client = self._get_client()
        bucket = client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [
            {
                "name": b.name,
                "size": b.size,
                "updated": b.updated.isoformat() if b.updated else None,
                "md5_hash": b.md5_hash,
                "metadata": dict(b.metadata) if b.metadata else {},
            }
            for b in blobs
        ]

    def verify_remote_manifest(
        self,
        local_manifest_entries: List[Dict[str, Any]],
        remote_prefix: str = "datasets/v2/",
    ) -> Dict[str, Any]:
        """
        Verifies that every entry in the local manifest exists on GCS with the exact expected byte size.
        """
        remote_blobs = {b["name"]: b["size"] for b in self.list_blobs(prefix=remote_prefix)}
        matched = 0
        missing = []
        size_mismatches = []

        for entry in local_manifest_entries:
            expected_remote_path = f"{remote_prefix}{entry['relative_path']}"
            if expected_remote_path not in remote_blobs:
                missing.append(expected_remote_path)
            elif remote_blobs[expected_remote_path] != entry["size_bytes"]:
                size_mismatches.append({
                    "path": expected_remote_path,
                    "expected_bytes": entry["size_bytes"],
                    "remote_bytes": remote_blobs[expected_remote_path],
                })
            else:
                matched += 1

        all_ok = len(missing) == 0 and len(size_mismatches) == 0
        return {
            "all_verified": all_ok,
            "total_local_files": len(local_manifest_entries),
            "matched_files": matched,
            "missing_remote_files": missing,
            "size_mismatches": size_mismatches,
        }
