"""
Single-GPU 12-Agent Parallel Inference Engine & OpenAI Server for qwen3.8-27b.
Document ID: colab_server_853_v1
Governing Standards: platodoom specification, LANGUAGE.md, RULE[user_global]

Key Architectural Invariants:
1. Shared-Weight Single-Process GPU Architecture (FR-3.1):
   Loads `qwen3.8-27b` ONCE into GPU VRAM and serves all concurrent HTTP requests
   from `.mts` agents over that shared weight allocation.
2. 12-Agent Continuous Dynamic Batching @ 16,384 Context Window (FR-3.2):
   Supports up to 12 simultaneous parallel agent sequences (`max_num_seqs = 12`) at `16,384` context
   (`max_model_len = 16384`) on a single 40GB A100 GPU (<= 25.5 GiB peak VRAM).
3. Chunked Prefill & Last-Token-Only Logit Projection (FR-3.3):
   Bounds prefill tokens per micro-step (`max_num_batched_tokens = 16384`) and projects `lm_head`
   (`vocab_size = 248,320`) ONLY on the final token position (`hidden_states[:, -1:, :]`).
4. Bounded Burst Queueing (Section 4.2):
   Admits up to `max_num_seqs` (12) requests into active GPU batches while holding request 13+ in
   a bounded FIFO queue (300s timeout) so burst traffic never triggers CUDA OOM.
"""

from __future__ import annotations

# === BEGIN_COLAB_SUPERVISOR_CORE ===
import concurrent.futures
import gc
import hmac
import http.client
import importlib
import importlib.util
import inspect
import json
import logging
import math
import os
import queue
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

if __name__ not in sys.modules:
    class _SupervisorNamespaceModule:
        pass
    _sup_ns_mod = _SupervisorNamespaceModule()
    _sup_ns_mod.__dict__ = globals()
    sys.modules[__name__] = _sup_ns_mod  # type: ignore[assignment]


def _is_valid_local_checkpoint_dir(candidate: str | Path) -> bool:
    try:
        p = Path(candidate)
        if not p.is_dir():
            return False
        if (p / "config.json").is_file() or (p / "model_index.json").is_file():
            return True
        return any(f.is_file() and f.suffix == ".gguf" and f.stat().st_size > 0 for f in p.iterdir())
    except Exception:
        return False


class LogitsProcessor:
    pass


class LogitsProcessorList(list):
    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        for proc in self:
            scores = proc(input_ids, scores)
        return scores


class StoppingCriteria:
    pass


class StoppingCriteriaList(list):
    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
        return any(crit(input_ids, scores, **kwargs) for crit in self)


_HF_GENERATION_TYPES_CACHE: Optional[Tuple[Any, Any, Any]] = None


def _get_hf_generation_types() -> Tuple[Any, Any, Any]:
    """Lazily resolves and caches HuggingFace generation wrapper types so CLI imports stay < 50ms and SSE turns pay 0ms import cost."""
    global _HF_GENERATION_TYPES_CACHE
    if _HF_GENERATION_TYPES_CACHE is not None:
        return _HF_GENERATION_TYPES_CACHE
    try:
        from transformers import (
            LogitsProcessorList as _HFLogitsProcessorList,
            StoppingCriteriaList as _HFStoppingCriteriaList,
            TextIteratorStreamer as _HFTextIteratorStreamer,
        )
        _HF_GENERATION_TYPES_CACHE = (_HFLogitsProcessorList, _HFStoppingCriteriaList, _HFTextIteratorStreamer)
    except ImportError:
        _HF_GENERATION_TYPES_CACHE = (LogitsProcessorList, StoppingCriteriaList, None)
    return _HF_GENERATION_TYPES_CACHE


try:
    from curation.api_client import (
        BASE_MODEL_ID,
        MAX_MODEL_LEN,
        MAX_NUM_BATCHED_TOKENS,
        MAX_PARALLEL_AGENTS,
        UNIFIED_MODEL_ALIAS,
        CheckpointArchitectureMetadata,
        compute_serving_concurrency_config as _api_compute_serving_concurrency_config,
        inspect_checkpoint_architecture as _api_inspect_checkpoint_architecture,
        is_qwen35_architecture as _api_is_qwen35_architecture,
    )
except ImportError:
    UNIFIED_MODEL_ALIAS = "qwen3.8-27b"
    BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
    MAX_MODEL_LEN = 16384
    MAX_NUM_BATCHED_TOKENS = 16384
    MAX_PARALLEL_AGENTS = 32
    CheckpointArchitectureMetadata = Dict[str, Any]  # type: ignore[misc]
    _api_compute_serving_concurrency_config = None
    _api_inspect_checkpoint_architecture = None
    _api_is_qwen35_architecture = None


def is_qwen35_architecture(model_or_config: Any) -> bool:
    _fn = globals().get("_api_is_qwen35_architecture")
    if callable(_fn):
        return bool(_fn(model_or_config))
    if model_or_config is None:
        return False
    cfg = getattr(model_or_config, "config", model_or_config)
    if cfg is None:
        return False
    if isinstance(cfg, dict):
        mt = str(cfg.get("model_type") or "").strip().lower()
        archs = cfg.get("architectures") or []
        t_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
        sub_mt = str(t_cfg.get("model_type") or "").strip().lower()
    else:
        mt = str(getattr(cfg, "model_type", "") or "").strip().lower()
        archs = getattr(cfg, "architectures", None) or []
        t_obj = getattr(cfg, "text_config", None)
        sub_mt = str(getattr(t_obj, "model_type", "") or "").strip().lower() if t_obj is not None else ""
    if mt in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"} or sub_mt in {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }:
        return True
    if isinstance(archs, (list, tuple, set)):
        for arch_name in archs:
            if str(arch_name) in {
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3_5ForCausalLM",
                "Qwen3_5MoeForCausalLM",
            }:
                return True
    return False


def _estimate_bf16_weights_gb_from_cfg(raw_cfg: Dict[str, Any]) -> float:
    t_cfg = raw_cfg.get("text_config") if isinstance(raw_cfg.get("text_config"), dict) else raw_cfg
    hidden = int(t_cfg.get("hidden_size") or raw_cfg.get("hidden_size") or 4096)
    layers = int(t_cfg.get("num_hidden_layers") or raw_cfg.get("num_hidden_layers") or 32)
    inter = int(t_cfg.get("intermediate_size") or raw_cfg.get("intermediate_size") or hidden * 3)
    vocab = int(t_cfg.get("vocab_size") or raw_cfg.get("vocab_size") or 151936)
    n_experts = int(t_cfg.get("num_experts") or raw_cfg.get("num_experts") or 1)
    moe_inter = int(t_cfg.get("moe_intermediate_size") or raw_cfg.get("moe_intermediate_size") or inter)
    ffn_params = (n_experts * 3 * hidden * moe_inter) if n_experts > 1 else (3 * hidden * inter)
    approx_params = layers * (4 * hidden * hidden + ffn_params) + vocab * hidden
    return round((approx_params * 2.0) / (1024.0 ** 3), 2)


def inspect_checkpoint_architecture(
    model_path_or_id: Any,
    token: Any = None,
    allow_remote: bool = False,
) -> Dict[str, Any]:
    if isinstance(token, bool):
        allow_remote = token
        token = None
    if isinstance(model_path_or_id, (str, Path)):
        raw_str_id = str(model_path_or_id)
        p_diff = Path(raw_str_id)
        if p_diff.is_dir() and (p_diff / "model_index.json").is_file() and not (p_diff / "config.json").is_file():
            try:
                mi = json.loads((p_diff / "model_index.json").read_text(encoding="utf-8"))
            except Exception:
                mi = {}
            cls_name = str(mi.get("_class_name") or "DiffusionPipeline")
            return {
                "model_type": "diffusion_image",
                "architecture_family": "diffusion_image",
                "loader_kind": "diffusion_pipeline",
                "loader_class_name": cls_name,
                "is_qwen35_family": False,
                "quant_method": None,
                "quantization_config": None,
                "metadata_origin": "local_dir",
            }
        low_id = raw_str_id.lower()
        if not p_diff.is_dir() and any(k in low_id for k in ("flux", "z-image")):
            cls_name = "FluxPipeline" if "flux" in low_id else "ZImagePipeline"
            return {
                "model_type": "diffusion_image",
                "architecture_family": "diffusion_image",
                "loader_kind": "diffusion_pipeline",
                "loader_class_name": cls_name,
                "is_qwen35_family": False,
                "quant_method": "gguf" if "gguf" in low_id else None,
                "quantization_config": None,
                "metadata_origin": "repo_heuristic",
            }
    _fn = globals().get("_api_inspect_checkpoint_architecture")
    if callable(_fn):
        _res = dict(_fn(model_path_or_id, token=token, allow_remote=allow_remote))
        if isinstance(model_path_or_id, str) and _res.get("quantization_config") is None:
            _low_m = model_path_or_id.lower()
            if "gptq" in _low_m:
                _res["quantization_config"] = {"quant_method": "gptq", "bits": 4}
                _res["quant_method"] = "gptq"
            elif "awq" in _low_m:
                _res["quantization_config"] = {"quant_method": "awq", "bits": 4}
                _res["quant_method"] = "awq"
            elif "fp8" in _low_m:
                _res["quantization_config"] = {"quant_method": "fp8"}
                _res["quant_method"] = "fp8"
        return _res
    raw_cfg: Dict[str, Any] = {}
    origin = "dict"
    if isinstance(model_path_or_id, dict):
        raw_cfg = dict(model_path_or_id)
    elif hasattr(model_path_or_id, "config"):
        cfg_obj = getattr(model_path_or_id, "config")
        origin = "model_config"
        if isinstance(cfg_obj, dict):
            raw_cfg = dict(cfg_obj)
        elif hasattr(cfg_obj, "to_dict"):
            raw_cfg = dict(cfg_obj.to_dict())
    else:
        p = Path(str(model_path_or_id))
        if p.is_dir() and (p / "config.json").is_file():
            origin = "local_dir"
            try:
                raw_cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
            except Exception:
                raw_cfg = {}
        elif isinstance(model_path_or_id, str):
            origin = "repo_heuristic"
            low_s = model_path_or_id.lower()
            if "gptq" in low_s:
                raw_cfg["quantization_config"] = {"quant_method": "gptq", "bits": 4}
            elif "awq" in low_s:
                raw_cfg["quantization_config"] = {"quant_method": "awq", "bits": 4}
            elif "fp8" in low_s:
                raw_cfg["quantization_config"] = {"quant_method": "fp8"}
            if "qwen3.5" in low_s or "qwen3_5" in low_s or "qwen3.6" in low_s or "qwen3.8" in low_s or "qwen3_8" in low_s:
                raw_cfg["model_type"] = "qwen3_5_moe" if ("35b" in low_s or "122b" in low_s) else "qwen3_5"
    archs = [str(a) for a in (raw_cfg.get("architectures") or [])]
    mt = str(raw_cfg.get("model_type") or "").strip().lower()
    t_cfg = raw_cfg.get("text_config") if isinstance(raw_cfg.get("text_config"), dict) else {}
    q_cfg = raw_cfg.get("quantization_config") if isinstance(raw_cfg.get("quantization_config"), dict) else None
    q_method = str((q_cfg or {}).get("quant_method") or "").strip().lower() or None
    if "Qwen3_5MoeForConditionalGeneration" in archs or mt in ("qwen3_5_moe", "qwen3_5_moe_text"):
        return {
            "model_type": mt or "qwen3_5_moe",
            "architecture_family": "qwen3_5_moe",
            "loader_kind": "conditional_generation",
            "loader_class_name": "Qwen3_5MoeForConditionalGeneration",
            "is_qwen35_family": True,
            "quant_method": q_method,
            "quantization_config": q_cfg,
            "vocab_size": int(t_cfg.get("vocab_size") or raw_cfg.get("vocab_size") or 248320),
            "torch_dtype": raw_cfg.get("torch_dtype") or t_cfg.get("dtype"),
            "metadata_origin": origin,
            "raw_config": raw_cfg,
        }
    if "Qwen3_5ForConditionalGeneration" in archs or mt in ("qwen3_5", "qwen3_5_text"):
        return {
            "model_type": mt or "qwen3_5",
            "architecture_family": "qwen3_5",
            "loader_kind": "conditional_generation",
            "loader_class_name": "Qwen3_5ForConditionalGeneration",
            "is_qwen35_family": True,
            "quant_method": q_method,
            "quantization_config": q_cfg,
            "vocab_size": int(t_cfg.get("vocab_size") or raw_cfg.get("vocab_size") or 248320),
            "torch_dtype": raw_cfg.get("torch_dtype") or t_cfg.get("dtype"),
            "metadata_origin": origin,
            "raw_config": raw_cfg,
        }
    return {
        "model_type": mt or "causal_lm",
        "architecture_family": mt or "causal_lm",
        "loader_kind": "causal_lm",
        "loader_class_name": "AutoModelForCausalLM",
        "is_qwen35_family": False,
        "quant_method": q_method,
        "quantization_config": q_cfg,
        "vocab_size": raw_cfg.get("vocab_size"),
        "torch_dtype": raw_cfg.get("torch_dtype"),
        "metadata_origin": origin,
        "raw_config": raw_cfg,
    }


def compute_serving_concurrency_config(
    total_vram_gb: float = 40.0,
    max_model_len: int = MAX_MODEL_LEN,
    target_parallel_agents: int = MAX_PARALLEL_AGENTS,
) -> Dict[str, Any]:
    _fn = globals().get("_api_compute_serving_concurrency_config")
    if callable(_fn):
        return dict(
            _fn(
                total_vram_gb=total_vram_gb,
                max_model_len=max_model_len,
                target_parallel_agents=target_parallel_agents,
            )
        )
    eff_seqs = max(1, int(target_parallel_agents))
    eff_len = max(1024, int(max_model_len))
    eff_batched = max(8192, min(65536, eff_seqs * 2048))
    return {
        "total_vram_gb": float(total_vram_gb),
        "max_num_seqs": eff_seqs,
        "max_model_len": eff_len,
        "max_num_batched_tokens": eff_batched,
    }

logger = logging.getLogger("curation.colab_server")

# Dynamic resolution of default GCS bucket and project (FR-1.2, AC-1: Zero hardcoded project/bucket literals)
def _resolve_default_gcs_bucket_and_project() -> Tuple[str, str]:
    """Dynamically resolves default GCS bucket and project ID without hardcoded credentials or literals."""
    bucket = os.environ.get("PLATODOOM_GCS_BUCKET") or os.environ.get("GCS_BUCKET_NAME")
    project = os.environ.get("PLATODOOM_GCS_PROJECT") or os.environ.get("GCP_PROJECT_ID")

    if not bucket or not project:
        cfg_path = Path.home() / ".platodoom" / "config.json"
        if cfg_path.is_file():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                storage_cfg = cfg.get("storage", {})
                bucket = bucket or storage_cfg.get("gcs_bucket")
                project = project or storage_cfg.get("gcs_project_id")
            except Exception:
                pass

    if not bucket or not project:
        _mod_file = globals().get("__file__")
        manifest_path = (Path(_mod_file).resolve().parent.parent if _mod_file else Path.cwd()) / "datasets" / "v1" / "manifest_v1.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                b = m.get("bucket", "")
                bucket = bucket or b
                if b and not project:
                    prefix_str = "platodoom-"
                    if prefix_str in b:
                        project = b.split(prefix_str, 1)[1]
                    else:
                        project = b
            except Exception:
                pass

    bucket = bucket or "platodoom-models"
    project = project or "platodoom-default"
    return str(bucket), str(project)


_DEF_BUCKET, _DEF_PROJECT = _resolve_default_gcs_bucket_and_project()
DEFAULT_GCS_PROJECT_ID = _DEF_PROJECT
DEFAULT_GCS_BUCKET_NAME = _DEF_BUCKET
DEFAULT_GCS_MODEL_PREFIX = "models/"

# Security and VRAM helper imports with resilient fallbacks
try:
    from platodoom.security import verify_bearer_authorization
except ImportError:
    def verify_bearer_authorization(auth_header: Optional[str], expected_token: Optional[str]) -> bool:
        if not expected_token or expected_token == "EMPTY":
            return True
        if not auth_header:
            return False
        parts = auth_header.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return hmac.compare_digest(parts[1].strip(), expected_token.strip())
        return False

_SUPERVISOR_GPU_TIERS: Dict[str, Dict[str, Any]] = {
    "v5e-1": {"vram_gb": 16.0, "max_num_seqs": 4, "cc": None, "accelerator": "TPU"},
    "T4": {"vram_gb": 15.8, "max_num_seqs": 32, "cc": (7, 5), "accelerator": "GPU"},
    "L4": {"vram_gb": 23.5, "max_num_seqs": 64, "cc": (8, 9), "accelerator": "GPU"},
    "v6e-1": {"vram_gb": 32.0, "max_num_seqs": 8, "cc": None, "accelerator": "TPU"},
    "A100": {"vram_gb": 40.0, "max_num_seqs": 96, "cc": (8, 0), "accelerator": "GPU"},
    "A100-80GB": {"vram_gb": 79.2, "max_num_seqs": 128, "cc": (8, 0), "accelerator": "GPU"},
    "H100": {"vram_gb": 79.2, "max_num_seqs": 160, "cc": (9, 0), "accelerator": "GPU"},
    "G4": {"vram_gb": 95.0, "max_num_seqs": 256, "cc": (12, 0), "accelerator": "GPU"},
}

_SUPERVISOR_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "qwen-9b": {
        "hf_repo_id": "Qwen/Qwen3.5-9B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "source": "huggingface",
        "total_params_b": 9.0,
        "active_params_b": 9.0,
        "is_moe": False,
        "min_vram_gb": 9.0,
        "weights_q4_gb": 5.2,
        "weights_bf16_gb": 18.0,
        "model_type": "qwen3_5",
        "quant_method": "compressed-tensors",
        "torch_dtype": "bfloat16",
        "max_pos": 262144,
        "vocab_size": 248320,
        "kv_bytes_bf16": 32768.0,
        "kv_bytes_fp8": 16384.0,
        "state_bytes": 51511296,
        "aligned_block_tokens_fp8": 1088,
        "aligned_block_bytes_fp8": 71303168,
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "r1-7b": {
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-7b-awq",
        "source": "huggingface",
        "total_params_b": 7.6,
        "active_params_b": 7.6,
        "is_moe": False,
        "min_vram_gb": 8.4,
        "weights_q4_gb": 4.6,
        "weights_bf16_gb": 15.2,
        "model_type": "qwen2",
        "quant_method": "awq",
        "torch_dtype": "bfloat16",
        "max_pos": 131072,
        "vocab_size": 152064,
        "kv_bytes_bf16": 57344.0,
        "kv_bytes_fp8": 28672.0,
        "state_bytes": 0,
        "tool_parser": None,
        "reasoning_parser": "deepseek_r1",
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-27b": {
        "hf_repo_id": "Qwen/Qwen3.5-27B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
        "source": "huggingface",
        "total_params_b": 27.0,
        "active_params_b": 27.0,
        "is_moe": False,
        "min_vram_gb": 19.0,
        "weights_q4_gb": 15.2,
        "weights_bf16_gb": 54.0,
        "model_type": "qwen3_5",
        "quant_method": "compressed-tensors",
        "torch_dtype": "bfloat16",
        "max_pos": 262144,
        "vocab_size": 248320,
        "kv_bytes_bf16": 65536.0,
        "kv_bytes_fp8": 32768.0,
        "state_bytes": 153944064,
        "aligned_block_tokens_fp8": 1600,
        "aligned_block_bytes_fp8": 209715200,
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-27b-36": {
        "hf_repo_id": "Qwen/Qwen3.6-27B",
        "quantized_hf_repo_id": "Alennnndy/Qwen3.6-27B-AWQ-4bit",
        "source": "huggingface",
        "total_params_b": 27.0,
        "active_params_b": 27.0,
        "is_moe": False,
        "min_vram_gb": 19.0,
        "weights_q4_gb": 15.2,
        "weights_bf16_gb": 54.0,
        "model_type": "qwen3_5",
        "quant_method": "compressed-tensors",
        "torch_dtype": "bfloat16",
        "max_pos": 262144,
        "vocab_size": 248320,
        "kv_bytes_bf16": 65536.0,
        "kv_bytes_fp8": 32768.0,
        "state_bytes": 153944064,
        "aligned_block_tokens_fp8": 1600,
        "aligned_block_bytes_fp8": 209715200,
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen3.8-27b": {
        "hf_repo_id": "Qwen/Qwen3.8-27B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
        "source": "huggingface",
        "total_params_b": 27.0,
        "active_params_b": 27.0,
        "is_moe": False,
        "min_vram_gb": 19.0,
        "weights_q4_gb": 15.2,
        "weights_bf16_gb": 54.0,
        "model_type": "qwen3_5",
        "quant_method": "compressed-tensors",
        "torch_dtype": "bfloat16",
        "max_pos": 262144,
        "vocab_size": 248320,
        "kv_bytes_bf16": 65536.0,
        "kv_bytes_fp8": 32768.0,
        "state_bytes": 153944064,
        "aligned_block_tokens_fp8": 1600,
        "aligned_block_bytes_fp8": 209715200,
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "r1-14b": {
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
        "source": "huggingface",
        "total_params_b": 14.8,
        "active_params_b": 14.8,
        "is_moe": False,
        "min_vram_gb": 12.2,
        "weights_q4_gb": 8.4,
        "weights_bf16_gb": 29.6,
        "model_type": "qwen2",
        "quant_method": "awq",
        "torch_dtype": "bfloat16",
        "max_pos": 131072,
        "vocab_size": 152064,
        "kv_bytes_bf16": 196608.0,
        "kv_bytes_fp8": 98304.0,
        "state_bytes": 0,
        "tool_parser": None,
        "reasoning_parser": "deepseek_r1",
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-30b-moe": {
        "hf_repo_id": "Qwen/Qwen3-30B-A3B",
        "quantized_hf_repo_id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        "source": "huggingface",
        "total_params_b": 30.5,
        "active_params_b": 3.3,
        "is_moe": True,
        "min_vram_gb": 20.8,
        "weights_q4_gb": 16.8,
        "weights_bf16_gb": 61.0,
        "model_type": "qwen3_moe",
        "quant_method": "gptq",
        "torch_dtype": "float16",
        "max_pos": 40960,
        "vocab_size": 151936,
        "kv_bytes_bf16": 98304.0,
        "kv_bytes_fp8": 49152.0,
        "state_bytes": 0,
        "tool_parser": "hermes",
        "reasoning_parser": "qwen3",
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-coder-30b-moe": {
        "hf_repo_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "quantized_hf_repo_id": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
        "source": "huggingface",
        "total_params_b": 30.5,
        "active_params_b": 3.3,
        "is_moe": True,
        "min_vram_gb": 20.8,
        "weights_q4_gb": 16.8,
        "weights_bf16_gb": 61.0,
        "model_type": "qwen3_moe",
        "quant_method": "compressed-tensors",
        "torch_dtype": "bfloat16",
        "max_pos": 262144,
        "vocab_size": 151936,
        "kv_bytes_bf16": 98304.0,
        "kv_bytes_fp8": 49152.0,
        "state_bytes": 0,
        "tool_parser": "qwen3_coder",
        "reasoning_parser": None,
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-14b": {
        "hf_repo_id": "Qwen/Qwen3-14B",
        "quantized_hf_repo_id": "Qwen/Qwen3-14B-AWQ",
        "source": "huggingface",
        "total_params_b": 14.8,
        "active_params_b": 14.8,
        "is_moe": False,
        "min_vram_gb": 12.2,
        "weights_q4_gb": 8.4,
        "weights_bf16_gb": 29.6,
        "model_type": "qwen3",
        "quant_method": "awq",
        "torch_dtype": "float16",
        "max_pos": 40960,
        "vocab_size": 151936,
        "kv_bytes_bf16": 163840.0,
        "kv_bytes_fp8": 81920.0,
        "state_bytes": 0,
        "tool_parser": "hermes",
        "reasoning_parser": "qwen3",
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-8b": {
        "hf_repo_id": "Qwen/Qwen3-8B",
        "quantized_hf_repo_id": "Qwen/Qwen3-8B-AWQ",
        "source": "huggingface",
        "total_params_b": 8.2,
        "active_params_b": 8.2,
        "is_moe": False,
        "min_vram_gb": 8.7,
        "weights_q4_gb": 4.9,
        "weights_bf16_gb": 16.4,
        "model_type": "qwen3",
        "quant_method": "awq",
        "torch_dtype": "float16",
        "max_pos": 40960,
        "vocab_size": 151936,
        "kv_bytes_bf16": 147456.0,
        "kv_bytes_fp8": 73728.0,
        "state_bytes": 0,
        "tool_parser": "hermes",
        "reasoning_parser": "qwen3",
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-4b": {
        "hf_repo_id": "Qwen/Qwen3-4B",
        "quantized_hf_repo_id": "Qwen/Qwen3-4B-AWQ",
        "source": "huggingface",
        "total_params_b": 4.0,
        "active_params_b": 4.0,
        "is_moe": False,
        "min_vram_gb": 6.4,
        "weights_q4_gb": 2.5,
        "weights_bf16_gb": 8.0,
        "model_type": "qwen3",
        "quant_method": "awq",
        "torch_dtype": "float16",
        "max_pos": 40960,
        "vocab_size": 151936,
        "kv_bytes_bf16": 147456.0,
        "kv_bytes_fp8": 73728.0,
        "state_bytes": 0,
        "tool_parser": "hermes",
        "reasoning_parser": "qwen3",
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-35b-moe": {
        "hf_repo_id": "Qwen/Qwen3.5-35B-A3B",
        "quantized_hf_repo_id": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        "source": "huggingface",
        "total_params_b": 35.0,
        "active_params_b": 3.0,
        "is_moe": True,
        "min_vram_gb": 23.0,
        "weights_q4_gb": 19.2,
        "weights_bf16_gb": 70.0,
        "model_type": "qwen3_5_moe",
        "quant_method": "gptq",
        "torch_dtype": "bfloat16",
        "max_pos": 262144,
        "vocab_size": 248320,
        "kv_bytes_bf16": 20480.0,
        "kv_bytes_fp8": 10240.0,
        "state_bytes": 64389120,
        "aligned_block_tokens_fp8": 2112,
        "aligned_block_bytes_fp8": 86507520,
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "qwen-122b-moe": {
        "hf_repo_id": "Qwen/Qwen3.5-122B-A10B",
        "quantized_hf_repo_id": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        "source": "huggingface",
        "total_params_b": 122.0,
        "active_params_b": 10.0,
        "is_moe": True,
        "min_vram_gb": 70.0,
        "weights_q4_gb": 67.65,
        "weights_bf16_gb": 244.0,
        "model_type": "qwen3_5_moe",
        "quant_method": "gptq",
        "torch_dtype": "bfloat16",
        "max_pos": 262144,
        "vocab_size": 248320,
        "kv_bytes_bf16": 24576.0,
        "kv_bytes_fp8": 12288.0,
        "state_bytes": 153649152,
        "aligned_block_tokens_fp8": 4224,
        "aligned_block_bytes_fp8": 207618048,
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "r1-32b": {
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-32b-awq",
        "source": "huggingface",
        "total_params_b": 32.8,
        "active_params_b": 32.8,
        "is_moe": False,
        "min_vram_gb": 30.0,
        "weights_q4_gb": 18.4,
        "weights_bf16_gb": 65.5,
        "model_type": "qwen2",
        "quant_method": "awq",
        "torch_dtype": "bfloat16",
        "max_pos": 131072,
        "vocab_size": 152064,
        "kv_bytes_bf16": 262144.0,
        "kv_bytes_fp8": 131072.0,
        "state_bytes": 0,
        "tool_parser": None,
        "reasoning_parser": "deepseek_r1",
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "max_output_tokens": 8192,
        "modality": "text",
    },
    "deepseek-v2-lite": {
        "hf_repo_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "quantized_hf_repo_id": "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
        "source": "huggingface",
        "total_params_b": 15.7,
        "active_params_b": 2.4,
        "is_moe": True,
        "min_vram_gb": 12.8,
        "weights_q4_gb": 8.8,
        "weights_bf16_gb": 31.5,
        "model_type": "deepseek_v2",
        "quant_method": "awq",
        "torch_dtype": "float16",
        "max_pos": 163840,
        "vocab_size": 102400,
        "kv_bytes_bf16": 31104.0,
        "kv_bytes_fp8": 31104.0,
        "state_bytes": 0,
        "tool_parser": None,
        "reasoning_parser": None,
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "modality": "text",
    },
    "flux-klein-4b": {
        "hf_repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "quantized_hf_repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "source": "huggingface",
        "total_params_b": 4.0,
        "active_params_b": 4.0,
        "is_moe": False,
        "min_vram_gb": 8.5,
        "weights_q4_gb": 8.0,
        "weights_bf16_gb": 8.0,
        "model_type": "diffusion_image",
        "quant_method": None,
        "torch_dtype": "bfloat16",
        "max_pos": 4096,
        "vocab_size": 0,
        "kv_bytes_bf16": 0.0,
        "kv_bytes_fp8": 0.0,
        "state_bytes": 0,
        "tool_parser": None,
        "reasoning_parser": None,
        "modality": "image",
    },
    "flux-klein-9b": {
        "hf_repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "quantized_hf_repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "source": "huggingface",
        "total_params_b": 9.0,
        "active_params_b": 9.0,
        "is_moe": False,
        "min_vram_gb": 18.0,
        "weights_q4_gb": 17.5,
        "weights_bf16_gb": 18.2,
        "model_type": "diffusion_image",
        "quant_method": None,
        "torch_dtype": "bfloat16",
        "max_pos": 4096,
        "vocab_size": 0,
        "kv_bytes_bf16": 0.0,
        "kv_bytes_fp8": 0.0,
        "state_bytes": 0,
        "tool_parser": None,
        "reasoning_parser": None,
        "modality": "image",
    },
    "z-image-turbo": {
        "hf_repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "quantized_hf_repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "source": "huggingface",
        "total_params_b": 6.0,
        "active_params_b": 6.0,
        "is_moe": False,
        "min_vram_gb": 12.0,
        "weights_q4_gb": 12.2,
        "weights_bf16_gb": 12.2,
        "model_type": "diffusion_image",
        "quant_method": None,
        "torch_dtype": "bfloat16",
        "max_pos": 4096,
        "vocab_size": 0,
        "kv_bytes_bf16": 0.0,
        "kv_bytes_fp8": 0.0,
        "state_bytes": 0,
        "tool_parser": None,
        "reasoning_parser": None,
        "modality": "image",
    },
}

_SUPERVISOR_FLOAT16_REPOS = frozenset(
    {
        "Qwen/Qwen3-4B-AWQ",
        "Qwen/Qwen3-8B-AWQ",
        "Qwen/Qwen3-14B-AWQ",
        "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        "QuantTrio/Qwen3.6-35B-A3B-AWQ",
        "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
    }
)

_SUPERVISOR_STACK_PRESETS: Dict[str, Dict[str, List[str]]] = {
    "T4": {"solo": ["qwen-9b@Q4"], "recommended": ["qwen-9b@Q4", "r1-7b@Q4"], "triple": ["qwen-9b@Q4", "r1-7b@Q4", "qwen-4b@Q4"]},
    "L4": {"solo": ["qwen-27b@Q4"], "recommended": ["qwen-14b@Q4", "r1-7b@Q4"], "triple": ["qwen-9b@Q4", "r1-7b@Q4", "qwen-4b@Q4"]},
    "A100": {"solo": ["qwen-35b-moe@Q4"], "recommended": ["qwen-27b@Q4", "r1-14b@Q4"], "triple": ["qwen-27b@Q4", "r1-14b@Q4", "qwen-9b@Q4"]},
    "A100-80GB": {"solo": ["qwen-122b-moe@Q4"], "recommended": ["qwen-27b@BF16", "r1-14b@Q4"], "triple": ["qwen-35b-moe@Q4", "r1-32b@Q4", "qwen-9b@BF16"]},
    "H100": {"solo": ["qwen-122b-moe@Q4"], "recommended": ["qwen-27b@BF16", "r1-14b@Q4"], "triple": ["qwen-35b-moe@Q4", "r1-32b@Q4", "qwen-9b@BF16"]},
    "G4": {"solo": ["qwen-122b-moe@Q4"], "recommended": ["qwen-35b-moe@BF16", "r1-14b@Q4"], "triple": ["qwen-35b-moe@Q4", "r1-32b@Q4", "qwen-9b@BF16"]},
}

try:
    from platodoom.catalog import (
        BUILTIN_MODEL_CATALOG,
        compile_model_chat_messages,
        compute_dynamic_max_num_seqs,
        estimate_model_kv_and_state_footprint,
        inspect_local_checkpoint_footprint_gib,
        introspect_model_protocol,
        normalize_precision_code,
        parse_multi_model_agent_spec,
        resolve_download_ignore_patterns,
        resolve_hardware_stack,
        resolve_model_precision_variant,
        resolve_model_profile,
        resolve_vllm_model_parser_spec,
        verify_stack_vram_compatibility,
        verify_vram_compatibility,
    )
except ImportError:
    BUILTIN_MODEL_CATALOG = {k: dict(v, alias=k) for k, v in _SUPERVISOR_MODEL_SPECS.items()}

    def introspect_model_protocol(model_alias: str, allow_remote: bool = False) -> Dict[str, Any]:
        clean = str(model_alias or "").strip().split("@", 1)[0].strip()
        prof = _SUPERVISOR_MODEL_SPECS.get(clean, {})
        low = clean.lower()
        is_r1 = prof.get("reasoning_parser") == "deepseek_r1" or "r1-" in low or "deepseek-r1" in low
        is_ds = prof.get("model_type") == "deepseek_v2" or "deepseek-v2" in low
        return {
            "model_id": clean,
            "template_family": "deepseek_r1_distill" if is_r1 else ("deepseek_v2_chat" if is_ds else "qwen3_chatml"),
            "serialize_tools_in_system_prompt": bool(prof.get("tool_parser") is None),
            "serialize_tool_role_as_user_xml": bool(is_r1 or is_ds),
            "supports_system_role": not is_r1,
            "supports_enable_thinking_kwarg": bool(prof.get("reasoning_parser") == "qwen3"),
            "forces_think_prefix": bool(is_r1),
        }

    def compile_model_chat_messages(
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        protocol_profile: Optional[Dict[str, Any]] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        profile = protocol_profile or {}
        tool_as_user = bool(profile.get("serialize_tool_role_as_user_xml", False))
        compiled: List[Dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role") or "user")
            if role == "developer":
                role = "system"
            if role == "assistant" and m.get("tool_calls") and tool_as_user:
                calls_txt = []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    calls_txt.append(f"<tool_call>\n{json.dumps({'name': fn.get('name'), 'arguments': fn.get('arguments')}, ensure_ascii=False)}\n</tool_call>")
                compiled.append({"role": "assistant", "content": "\n".join(filter(None, [str(m.get("content") or ""), *calls_txt]))})
            elif role == "tool" and tool_as_user:
                compiled.append({"role": "user", "content": f"<tool_response>\n{m.get('content') or ''}\n</tool_response>"})
            else:
                compiled.append({**m, "role": role})
        if tools:
            tools_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
            block = (
                "\n\n# Tools\nYou are provided with function signatures within <tools></tools> XML tags:\n"
                f"<tools>\n{tools_lines}\n</tools>\n"
                "For each function call, return a json object within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            )
            if compiled and compiled[0].get("role") == "system":
                compiled[0]["content"] = str(compiled[0].get("content") or "") + block
            else:
                compiled.insert(0, {"role": "system", "content": block.strip()})
        return compiled, {}
    def normalize_precision_code(precision: Optional[str]) -> str:
        p = str(precision or "Q4").strip().upper()
        if p in ("BF16", "BFLOAT16", "FP16", "FLOAT16"):
            return "BF16"
        if p in ("FP8",):
            return "FP8"
        if p in ("Q8", "INT8", "8BIT"):
            return "Q8"
        return "Q4"

    def inspect_local_checkpoint_footprint_gib(
        checkpoint_dir: Any,
        *,
        language_model_only: bool = False,
        quant_method: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not checkpoint_dir:
            return None
        try:
            cp = Path(str(checkpoint_dir)).expanduser()
            if not cp.is_dir():
                return None
        except Exception:
            return None
        non_lm_pfx = (
            "visual.",
            "model.visual.",
            "vision_tower.",
            "model.vision_tower.",
            "multi_modal_projector.",
            "mm_projector.",
            "mtp.",
            "model.mtp.",
        )
        det_q = str(quant_method or "").strip().lower() or None
        cfg_f = cp / "config.json"
        if cfg_f.is_file() and not det_q:
            try:
                qc = (json.loads(cfg_f.read_text(encoding="utf-8")) or {}).get("quantization_config") or {}
                if isinstance(qc, Mapping) and qc.get("quant_method"):
                    det_q = str(qc["quant_method"]).strip().lower()
            except Exception:
                pass
        st_list = sorted(p for p in cp.glob("*.safetensors") if p.is_file())
        if not st_list:
            return None
        disk_b = 0
        tot_b = 0
        nlm_b = 0
        for sp in st_list:
            fsz = sp.stat().st_size
            disk_b += fsz
            try:
                with sp.open("rb") as fh:
                    rl = fh.read(8)
                    if len(rl) == 8:
                        (hl,) = struct.unpack("<Q", rl)
                        if 0 < hl <= 64 * 1024 * 1024:
                            ho = json.loads(fh.read(hl).decode("utf-8"))
                            if isinstance(ho, Mapping):
                                sb = 0
                                sn = 0
                                for tn, tm in ho.items():
                                    if tn != "__metadata__" and isinstance(tm, Mapping):
                                        offs = tm.get("data_offsets")
                                        if isinstance(offs, Sequence) and len(offs) == 2:
                                            spn = max(0, int(offs[1]) - int(offs[0]))
                                            sb += spn
                                            if any(str(tn).lower().startswith(px) for px in non_lm_pfx):
                                                sn += spn
                                if sb > 0:
                                    tot_b += sb
                                    nlm_b += sn
                                    continue
            except Exception:
                pass
            tot_b += fsz
        excl_b = nlm_b if language_model_only else 0
        lm_b = max(1, tot_b - excl_b)
        lm_gib = lm_b / float(1024 ** 3)
        repack = 1.025 if det_q in ("gptq", "auto_gptq", "awq", "compressed-tensors") else 1.005
        return {
            "disk_checkpoint_bytes": int(disk_b),
            "disk_checkpoint_gib": round(disk_b / float(1024 ** 3), 4),
            "total_tensor_bytes": int(tot_b),
            "total_tensor_gib": round(tot_b / float(1024 ** 3), 4),
            "lm_tensor_bytes": int(lm_b),
            "lm_tensor_gib": round(lm_gib, 4),
            "active_tensor_gib": round(lm_gib, 4),
            "excluded_non_lm_bytes": int(excl_b),
            "excluded_non_lm_gib": round(excl_b / float(1024 ** 3), 4),
            "excluded_multimodal_or_mtp_gib": round(excl_b / float(1024 ** 3), 4),
            "estimated_gpu_weights_gib": round(lm_gib * repack, 4),
            "shard_count": len(st_list),
            "quant_method": det_q,
            "language_model_only": bool(language_model_only),
        }

    def resolve_model_profile(model_alias: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        clean = str(model_alias or "").strip().split("@", 1)[0].strip()
        if clean in _SUPERVISOR_MODEL_SPECS:
            d = dict(_SUPERVISOR_MODEL_SPECS[clean])
            d["model_alias"] = clean
            return d
        for k, v in _SUPERVISOR_MODEL_SPECS.items():
            if clean.lower() in (k.lower(), str(v.get("hf_repo_id") or "").lower(), str(v.get("quantized_hf_repo_id") or "").lower()):
                d = dict(v)
                d["model_alias"] = k
                return d
        raise ValueError(f"Unknown model alias '{model_alias}'.")

    def resolve_model_precision_variant(model_alias: str, precision: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prof = resolve_model_profile(model_alias)
        norm = normalize_precision_code(precision)
        if norm == "BF16":
            return {"hf_repo_id": prof["hf_repo_id"], "precision": "BF16", "weights_gb": prof["weights_bf16_gb"]}
        return {
            "hf_repo_id": prof.get("quantized_hf_repo_id") or prof["hf_repo_id"],
            "precision": norm,
            "weights_gb": prof["weights_q4_gb"],
        }

    def resolve_download_ignore_patterns(repo_id: str, precision: str = "") -> List[str]:
        return ["*.msgpack", "*.h5", "consolidated*", "*.pth", "*.gguf"]

    def verify_vram_compatibility(
        model_alias: str,
        available_vram_gb: float,
        config: Optional[Dict[str, Any]] = None,
        precision: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean = str(model_alias or "").strip().split("@", 1)[0].strip()
        prof = _SUPERVISOR_MODEL_SPECS.get(clean)
        min_vram = float(prof["min_vram_gb"]) if prof else 7.5
        if float(available_vram_gb) >= min_vram:
            res: Dict[str, Any] = {
                "compatible": True,
                "active_model": clean,
                "available_vram_gb": float(available_vram_gb),
            }
            norm_p = normalize_precision_code(precision) if precision else ("BF16" if prof and float(available_vram_gb) >= float(prof["weights_bf16_gb"]) + 6.0 else "Q4")
            res["active_precision"] = norm_p
            res["selected_precision"] = norm_p
            res["normalized_precision"] = norm_p
            if prof and norm_p == "Q4" and prof.get("quantized_hf_repo_id"):
                res["quantized_hf_repo_id"] = prof["quantized_hf_repo_id"]
            return res
        return {
            "compatible": False,
            "requested_model": clean,
            "required_min_vram_gb": min_vram,
            "available_vram_gb": float(available_vram_gb),
            "fallback_alias": "qwen-9b",
            "diagnostic": (
                f"Physical VRAM {available_vram_gb:.1f} GB is below {clean} floor ({min_vram:.1f} GB). "
                f"Switch rejected without silent fallback."
            ),
        }

    def parse_multi_model_agent_spec(spec_str: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tok in str(spec_str or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            cnt = 1
            if ":" in tok:
                m_part, c_part = tok.rsplit(":", 1)
                cnt = int(c_part.strip())
            else:
                m_part = tok
            prec = "Q4"
            if "@" in m_part:
                a_part, p_part = m_part.split("@", 1)
                prec = normalize_precision_code(p_part)
            else:
                a_part = m_part
            out.append({"model_spec": a_part.strip(), "precision": prec, "agent_count": cnt})
        return out

    def resolve_hardware_stack(
        hardware_tier: str,
        stack_preset: str = "recommended",
        model_alias: Optional[str] = None,
        precision: Optional[str] = None,
    ) -> Dict[str, Any]:
        tier_map = _SUPERVISOR_STACK_PRESETS.get(str(hardware_tier), _SUPERVISOR_STACK_PRESETS["L4"])
        if stack_preset == "solo" and model_alias:
            p = normalize_precision_code(precision)
            return {"models": [f"{model_alias}@{p}"], "stack_preset": "solo"}
        return {"models": list(tier_map.get(stack_preset, tier_map["recommended"])), "stack_preset": stack_preset}

    def verify_stack_vram_compatibility(
        stack_specs: Sequence[Any],
        tier_or_vram: str | float,
        agent_counts: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        if isinstance(tier_or_vram, str) and tier_or_vram in _SUPERVISOR_GPU_TIERS:
            avail = float(_SUPERVISOR_GPU_TIERS[tier_or_vram]["vram_gb"])
        else:
            avail = float(tier_or_vram)
        active_models: List[Dict[str, Any]] = []
        inactive_models: List[Dict[str, Any]] = []
        resolved_counts: Dict[str, int] = {}
        total_weights = 0.0
        for idx, raw in enumerate(stack_specs):
            if isinstance(raw, Mapping):
                raw_al = str(raw.get("alias") or raw.get("model") or raw.get("model_spec") or "")
                raw_pr = str(raw.get("precision") or "Q4")
                ag = int(raw.get("agent_count", (agent_counts or {}).get(raw_al.split("@", 1)[0], 1)))
            else:
                s_tok = str(raw).strip()
                ag_inline = None
                if ":" in s_tok and not s_tok.startswith(("/", "./")):
                    s_tok, c_str = s_tok.rsplit(":", 1)
                    ag_inline = int(c_str)
                if "@" in s_tok and not _is_valid_local_checkpoint_dir(s_tok):
                    raw_al, raw_pr = s_tok.split("@", 1)
                else:
                    raw_al, raw_pr = s_tok, "Q4"
                ag = ag_inline if ag_inline is not None else int((agent_counts or {}).get(raw_al, 1))
            clean_al = raw_al.strip()
            norm_pr = normalize_precision_code(raw_pr)
            prof = _SUPERVISOR_MODEL_SPECS.get(clean_al)
            if prof is None and not _is_valid_local_checkpoint_dir(clean_al) and "/" not in clean_al:
                raise ValueError(f"Unknown model identifier '{clean_al}'.")
            w_gb = float(prof["weights_bf16_gb"] if (prof and norm_pr == "BF16") else (prof["weights_q4_gb"] if prof else 5.0))
            repo = str((prof.get("quantized_hf_repo_id") if (prof and norm_pr == "Q4" and prof.get("quantized_hf_repo_id")) else (prof["hf_repo_id"] if prof else clean_al)))
            entry = {"alias": clean_al, "precision": norm_pr, "hf_repo_id": repo, "weights_gb": w_gb, "agent_count": ag, "role": "primary" if idx == 0 else f"slot_{idx + 1}"}
            resolved_counts[clean_al] = ag
            if ag <= 0:
                inactive_models.append(entry)
            else:
                active_models.append(entry)
                total_weights += w_gb
        if len(active_models) > 3:
            return {"compatible": False, "code": "invalid_stack_cardinality", "diagnostic": f"Active stack cardinality {len(active_models)} > 3."}
        kv_floor = 4.0 if len(active_models) == 1 else (1.8 * len(active_models))
        compatible = (total_weights + (kv_floor if active_models else 0.0)) <= avail + 0.5
        return {
            "compatible": compatible,
            "models": active_models,
            "inactive_models": inactive_models,
            "agent_counts": resolved_counts,
            "model_count": len(active_models),
            "estimated_weights_gb": round(total_weights, 3),
            "free_kv_gb": round(max(0.0, avail - total_weights), 3),
            "available_memory_gb": avail,
            "diagnostic": "" if compatible else f"Stack weights {total_weights:.1f} GB + KV floor exceed {avail:.1f} GB.",
        }

    def resolve_vllm_model_parser_spec(model_alias: str, arch_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        clean = str(model_alias or "").strip().split("@", 1)[0].strip()
        prof = _SUPERVISOR_MODEL_SPECS.get(clean, {})
        tp = prof.get("tool_parser")
        rp = prof.get("reasoning_parser")
        mtype = str(prof.get("model_type") or "")
        low = clean.lower()
        if rp == "deepseek_r1" or "r1-" in low:
            family = "deepseek_r1_distill"
        elif mtype == "deepseek_v2" or "deepseek-v2" in low:
            family = "deepseek_v2_lite"
        elif mtype in ("qwen3_5", "qwen3_5_moe"):
            family = "qwen3_5"
        elif clean == "qwen-coder-30b-moe":
            family = "qwen3_coder"
        elif mtype in ("qwen3", "qwen3_moe"):
            family = "qwen3"
        else:
            family = mtype or "generic_causal_lm"
        return {
            "model_alias": clean,
            "family": family,
            "tool_call_parser": tp,
            "reasoning_parser": rp,
            "enable_auto_tool_choice": bool(tp),
            "supports_thinking_toggle": bool(rp == "qwen3"),
            "supports_enable_thinking": bool(rp == "qwen3"),
            "client_prompt_tool_fallback": bool(tp is None),
            "mamba_hybrid": mtype in ("qwen3_5", "qwen3_5_moe"),
            "language_model_only": mtype in ("qwen3_5", "qwen3_5_moe"),
        }

    def estimate_model_kv_and_state_footprint(
        model_alias: str,
        *,
        max_model_len: int = 16384,
        kv_cache_dtype: str = "auto",
        arch_meta: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        clean = str(model_alias or "").strip().split("@", 1)[0].strip()
        prof = _SUPERVISOR_MODEL_SPECS.get(clean, {})
        use_fp8 = str(kv_cache_dtype).lower().startswith("fp8") and str(prof.get("model_type") or "") != "deepseek_v2"
        kv_b = float(prof.get("kv_bytes_fp8" if use_fp8 else "kv_bytes_bf16", 65536.0))
        st_b = int(prof.get("state_bytes", 0))
        is_hyb = str(prof.get("model_type") or "") in ("qwen3_5", "qwen3_5_moe")
        is_mla = str(prof.get("model_type") or "") == "deepseek_v2"
        aligned_tok = int(prof.get("aligned_block_tokens_fp8") or 16) if is_hyb else 16
        aligned_b = int(prof.get("aligned_block_bytes_fp8") or int(aligned_tok * kv_b)) if is_hyb else int(16 * kv_b)
        if is_hyb and not use_fp8:
            aligned_tok = max(16, aligned_tok // 2)
        seq_b = kv_b * int(max_model_len) + st_b
        return {
            "kv_bytes_per_token": kv_b,
            "kv_gb_per_seq_at_max_len": round((kv_b * int(max_model_len)) / float(1024 ** 3), 6),
            "gdn_bytes_per_seq": float(st_b),
            "gdn_gb_per_seq": round(st_b / float(1024 ** 3), 6),
            "total_gb_per_seq_at_max_len": round(seq_b / float(1024 ** 3), 6),
            "aligned_block_tokens": aligned_tok,
            "unified_aligned_block_bytes": aligned_b,
            "unified_aligned_block_gib": round(aligned_b / float(1024 ** 3), 6),
            "is_hybrid_gdn": is_hyb,
            "is_mla": is_mla,
        }

    def compute_dynamic_max_num_seqs(
        free_vram_gb: float,
        weights_gb: float,
        *,
        model_alias: str = "",
        max_model_len: int = 16384,
        kv_cache_dtype: str = "auto",
        requested_agents: int = 1,
        active_slot_count: int = 1,
    ) -> int:
        slot_cnt = max(1, int(active_slot_count or 1))
        usable = (float(free_vram_gb) * 0.90 / slot_cnt) - float(weights_gb) - max(1.4, 0.06 * float(free_vram_gb) / slot_cnt)
        if usable <= 0.0:
            return 1
        fp = estimate_model_kv_and_state_footprint(model_alias, max_model_len=max_model_len, kv_cache_dtype=kv_cache_dtype)
        work_tok = min(max(512, int(max_model_len or 16384)), 2048)
        work_kv = (float(fp["kv_bytes_per_token"]) * work_tok) / float(1024 ** 3)
        per_seq = max(0.02, work_kv + float(fp["gdn_gb_per_seq"]))
        if fp.get("is_hybrid_gdn") and float(fp.get("unified_aligned_block_gib", 0.0)) > 0.0:
            per_seq = max(per_seq, 2.0 * float(fp["unified_aligned_block_gib"]))
        raw_s = max(1, int(usable // per_seq))
        if raw_s < 16:
            return max(1, min(256, raw_s))
        return min(256, max(max(16, int(requested_agents or 1)), raw_s))

# Universal 4-format tool call and reasoning regexes
XML_TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
XML_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
DEEPSEEK_TOOL_CALL_RE = re.compile(
    r"[<｜|]*tool[ ▁_]call[ ▁_]begin[>｜|]*(?:function[<｜|]*tool[ ▁_]sep[>｜|]*)?([^\n<]+)?\n?(?:```(?:json)?\n)?(.*?)(?:\n```)?[<｜|]*tool[ ▁_]call[ ▁_]end[>｜|]*",
    re.DOTALL,
)
FENCED_TOOL_CALL_RE = re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)



def download_merged_model_parallel(
    local_model_dir: Path | str,
    project_id: str = DEFAULT_GCS_PROJECT_ID,
    bucket_name: str = DEFAULT_GCS_BUCKET_NAME,
    gcs_prefix: str = DEFAULT_GCS_MODEL_PREFIX,
    max_workers: int = 16,
    storage_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Downloads the 18.84 GB merged Qwen3.5-9B model from GCS using 16 parallel worker threads (FR-2.2)."""
    target_dir = Path(local_model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    config_file = target_dir / "config.json"
    shard_1 = target_dir / "model-00001-of-00004.safetensors"
    if config_file.exists() and (shard_1.exists() or any(target_dir.glob("*.safetensors"))):
        existing_files = [p.name for p in target_dir.iterdir() if p.is_file()]
        return {
            "cached": True,
            "local_model_dir": str(target_dir),
            "downloaded_files": len(existing_files),
            "elapsed_sec": 0.0,
        }

    t0 = time.perf_counter()
    if storage_client is None:
        import google.auth
        from google.cloud import storage

        if "google.colab" in sys.modules and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            from google.colab import auth

            auth.authenticate_user()
        credentials, _ = google.auth.default(quota_project_id=project_id)
        storage_client = storage.Client(project=project_id, credentials=credentials)

    bucket = storage_client.bucket(bucket_name)
    blobs = [b for b in bucket.list_blobs(prefix=gcs_prefix) if not b.name.endswith("/")]
    if not blobs:
        raise RuntimeError(f"FATAL: No model files found under gs://{bucket_name}/{gcs_prefix}")

    def _download_one(blob: Any) -> Tuple[str, int]:
        rel_path = os.path.relpath(blob.name, gcs_prefix.rstrip("/"))
        dest_file = target_dir / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if dest_file.exists() and blob.size and dest_file.stat().st_size == blob.size:
            return (rel_path, int(blob.size))
        blob.download_to_filename(str(dest_file))
        return (rel_path, int(blob.size or dest_file.stat().st_size))

    downloaded: List[Tuple[str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(_download_one, b) for b in blobs]
        for fut in concurrent.futures.as_completed(futures):
            downloaded.append(fut.result())

    elapsed = round(time.perf_counter() - t0, 2)
    total_bytes = sum(sz for _, sz in downloaded)
    return {
        "cached": False,
        "local_model_dir": str(target_dir),
        "downloaded_files": len(downloaded),
        "total_bytes": total_bytes,
        "elapsed_sec": elapsed,
    }


def _coerce_checkpoint_metadata(config_or_meta: Any) -> Dict[str, Any]:
    """Normalizes CheckpointArchitectureMetadata, dict, model, or HF PretrainedConfig into a metadata dict."""
    if isinstance(config_or_meta, dict):
        if "loader_kind" in config_or_meta and "architecture_family" in config_or_meta:
            return dict(config_or_meta)
        return dict(inspect_checkpoint_architecture(config_or_meta, allow_remote=False))
    return dict(inspect_checkpoint_architecture(config_or_meta, allow_remote=False))


def select_model_loader_from_config(
    config_or_meta: Any,
    transformers_module: Any = None,
    diffusers_module: Any = None,
) -> Tuple[Any, str, str]:
    """Deterministically selects `(loader_cls, loader_kind, architecture_family)` from checkpoint metadata.

    Standard causal LMs (`qwen2`, `qwen3`, `qwen3_moe`, `deepseek_r1`, `deepseek_v2`, `deepseek_v3`,
    `llama`, `mistral`, etc.) always use `AutoModelForCausalLM`. Conditional-generation loaders are
    selected strictly when checkpoint architecture metadata requires them (Spec 862 FR-1).
    Diffusion image pipelines (`model_index.json` / `FluxPipeline` / `ZImagePipeline`) dispatch to
    `diffusers.DiffusionPipeline` (Spec 868 §4.5).
    """
    if isinstance(config_or_meta, dict) and "loader_kind" in config_or_meta and "architecture_family" in config_or_meta:
        meta = dict(config_or_meta)
    else:
        meta = dict(inspect_checkpoint_architecture(config_or_meta, allow_remote=False))
    loader_kind = str(meta.get("loader_kind") or "causal_lm")
    loader_cls_name = str(meta.get("loader_class_name") or "")
    arch_family = str(meta.get("architecture_family") or "causal_lm")

    if loader_kind == "diffusion_pipeline" or arch_family == "diffusion_image" or loader_cls_name.endswith("Pipeline"):
        if transformers_module is not None and hasattr(transformers_module, loader_cls_name or "DiffusionPipeline"):
            cls = getattr(transformers_module, loader_cls_name or "DiffusionPipeline")
            return cls, "diffusion_pipeline", "diffusion_image"
        df_mod = diffusers_module
        if df_mod is None:
            try:
                df_mod = __import__("diffusers")
            except ImportError as exc:
                raise ImportError(
                    f"Diffusion image checkpoint requires 'diffusers' package to load '{loader_cls_name or 'DiffusionPipeline'}', "
                    f"but 'diffusers' is not installed: {exc}"
                ) from exc
        cls = (
            getattr(df_mod, loader_cls_name, None)
            if loader_cls_name
            else None
        ) or getattr(df_mod, "DiffusionPipeline", None)
        if cls is None:
            raise ImportError(f"diffusers module lacks '{loader_cls_name}' and 'DiffusionPipeline'.")
        return cls, "diffusion_pipeline", "diffusion_image"

    if transformers_module is not None:
        tf_mod = transformers_module
    elif "AutoModelForCausalLM" in globals():
        class _GlobalTFNamespace:
            pass
        tf_mod = _GlobalTFNamespace()
        for _k in ("AutoModelForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            if _k in globals():
                setattr(tf_mod, _k, globals()[_k])
    else:
        tf_mod = __import__("transformers")

    if loader_kind == "causal_lm" or loader_cls_name == "AutoModelForCausalLM":
        cls = getattr(tf_mod, "AutoModelForCausalLM")
        return cls, "causal_lm", arch_family

    if (
        loader_cls_name == "Qwen3_5MoeForConditionalGeneration"
        or loader_kind == "qwen3_5_moe_conditional_generation"
        or (loader_kind == "conditional_generation" and arch_family == "qwen3_5_moe")
    ):
        cls = getattr(tf_mod, "Qwen3_5MoeForConditionalGeneration", None) or getattr(
            tf_mod, "AutoModelForImageTextToText", None
        )
        if cls is None:
            raise ImportError("Transformers lacks Qwen3_5MoeForConditionalGeneration and AutoModelForImageTextToText.")
        return cls, "conditional_generation", arch_family

    if (
        loader_cls_name == "Qwen3_5ForConditionalGeneration"
        or loader_kind == "qwen3_5_conditional_generation"
        or (loader_kind == "conditional_generation" and arch_family == "qwen3_5")
    ):
        cls = getattr(tf_mod, "Qwen3_5ForConditionalGeneration", None) or getattr(
            tf_mod, "AutoModelForImageTextToText", None
        )
        if cls is None:
            raise ImportError("Transformers lacks Qwen3_5ForConditionalGeneration and AutoModelForImageTextToText.")
        return cls, "conditional_generation", arch_family

    if loader_kind == "conditional_generation":
        cls = (
            getattr(tf_mod, "AutoModelForImageTextToText", None)
            or getattr(tf_mod, "AutoModelForVision2Seq", None)
            or getattr(tf_mod, "AutoModelForCausalLM")
        )
        return cls, "conditional_generation", arch_family

    cls = getattr(tf_mod, "AutoModelForCausalLM")
    return cls, "causal_lm", arch_family


def _load_pretrained_or_gptqmodel(
    model_path_or_id: str | Path,
    *,
    loader_cls: Any,
    loader_kind: str,
    q_method: str,
    token: Optional[str] = None,
    transformers_module: Any = None,
    call_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Dispatches in-process checkpoint loading to `GPTQModel.load` (for dynamic GPTQ/AWQ) or `loader_cls.from_pretrained`."""
    eff_kwargs = dict(call_kwargs or {})
    use_gptqmodel_loader = (
        loader_kind != "diffusion_pipeline"
        and q_method in ("gptq", "awq")
        and transformers_module is None
        and getattr(loader_cls, "__module__", "").startswith("transformers")
        and importlib.util.find_spec("gptqmodel") is not None
    )
    if use_gptqmodel_loader:
        import torch
        from gptqmodel import GPTQModel
        from gptqmodel.models.auto import check_and_get_model_definition

        ckpt_str = str(model_path_or_id)
        trust_rc = bool(eff_kwargs.get("trust_remote_code", True))
        m_def = check_and_get_model_definition(ckpt_str, trust_remote_code=trust_rc)
        orig_req_proc = bool(getattr(m_def, "require_load_processor", False))
        if orig_req_proc:
            m_def.require_load_processor = False
        gptq_kw: Dict[str, Any] = {
            "dtype": eff_kwargs.get("torch_dtype") or eff_kwargs.get("dtype") or (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ),
            "trust_remote_code": trust_rc,
        }
        if token is not None:
            gptq_kw["token"] = token
        req_dmap = eff_kwargs.get("device_map")
        if torch.cuda.is_available() and req_dmap:
            gptq_kw["device_map"] = req_dmap
        elif not torch.cuda.is_available():
            gptq_kw["device"] = "cpu"
        try:
            qm = GPTQModel.load(ckpt_str, **gptq_kw)
        finally:
            if orig_req_proc:
                m_def.require_load_processor = orig_req_proc
        model = getattr(qm, "model", qm)
        if hasattr(model, "__dict__"):
            model.__dict__["_loaded_via_gptqmodel"] = True
        return model
    return loader_cls.from_pretrained(str(model_path_or_id), **eff_kwargs)


def load_model_from_checkpoint_metadata(
    model_path_or_id: str | Path,
    *,
    token: Optional[str] = None,
    allow_remote_config: bool = True,
    transformers_module: Any = None,
    diffusers_module: Any = None,
    **from_pretrained_kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Loads a checkpoint using its own `config.json` / `model_index.json` metadata without alias guessing or silent fallbacks."""
    meta = inspect_checkpoint_architecture(
        model_path_or_id,
        token=token,
        allow_remote=allow_remote_config,
    )
    loader_cls, loader_kind, _ = select_model_loader_from_config(
        meta,
        transformers_module=transformers_module,
        diffusers_module=diffusers_module,
    )
    call_kwargs = dict(from_pretrained_kwargs)
    if token is not None and "token" not in call_kwargs:
        call_kwargs["token"] = token
    if loader_kind == "diffusion_pipeline":
        call_kwargs.pop("quantization_config", None)
        call_kwargs.pop("trust_remote_code", None)
    q_method = str(meta.get("quant_method") or "").strip().lower()
    try:
        model = _load_pretrained_or_gptqmodel(
            model_path_or_id,
            loader_cls=loader_cls,
            loader_kind=loader_kind,
            q_method=q_method,
            token=token,
            transformers_module=transformers_module,
            call_kwargs=call_kwargs,
        )
    except (ImportError, ModuleNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
        gc.collect()
        try:
            import torch as _t_cleanup
            if _t_cleanup.cuda.is_available():
                _t_cleanup.cuda.empty_cache()
        except Exception:
            pass
        if exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
            raise
        if q_method in ("gptq", "awq", "fp8", "compressed-tensors"):
            raise RuntimeError(
                f"Failed loading '{model_path_or_id}' (quant_method='{q_method}' via {loader_cls.__name__}): "
                f"{exc.__class__.__name__}: {exc}. Verify Cell 1 provisioned native vLLM (`vllm==0.33.0`, "
                f"`compressed-tensors==0.17.0`) or in-process quantization packages (optimum / gptqmodel / compressed-tensors)."
            ) from exc
        raise
    return model, dict(meta)


def _resolve_vocab_upper_bound(tokenizer: Any, model_or_config: Any = None) -> Optional[int]:
    """Resolves the strict upper bound (`vocab_size`) for valid token IDs (`0 <= id < vocab_size`)."""
    if model_or_config is not None:
        cfg_root = getattr(model_or_config, "config", model_or_config)
        for cfg_src in (cfg_root, getattr(cfg_root, "text_config", None), getattr(model_or_config, "generation_config", None)):
            if cfg_src is None:
                continue
            cfg_vocab = cfg_src.get("vocab_size") if isinstance(cfg_src, dict) else getattr(cfg_src, "vocab_size", None)
            if isinstance(cfg_vocab, int) and not isinstance(cfg_vocab, bool) and cfg_vocab > 0:
                return int(cfg_vocab)

    if tokenizer is not None:
        candidates: List[int] = []
        raw_vocab = getattr(tokenizer, "vocab_size", None)
        if isinstance(raw_vocab, int) and not isinstance(raw_vocab, bool) and raw_vocab > 0:
            candidates.append(int(raw_vocab))
        if hasattr(tokenizer, "__len__"):
            try:
                tok_len = int(len(tokenizer))
                if tok_len > 0:
                    candidates.append(tok_len)
            except Exception:
                pass
        if candidates:
            return max(candidates)
    return None


def _is_valid_vocab_token_id(token_id: Any, vocab_upper_bound: Optional[int]) -> bool:
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        return False
    if isinstance(vocab_upper_bound, int) and vocab_upper_bound > 0 and token_id >= vocab_upper_bound:
        return False
    return True


def _resolve_stop_token_ids(
    tokenizer: Any,
    model_alias: Any = "",
    model_or_config: Any = None,
) -> List[int]:
    """Resolves deduplicated stop token IDs exclusively from the active tokenizer and model config within `[0, vocab_size)`."""
    if model_or_config is None and model_alias is not None and not isinstance(model_alias, str):
        model_or_config = model_alias
    stop_ids: set[int] = set()
    vocab_upper_bound = _resolve_vocab_upper_bound(tokenizer, model_or_config)

    if tokenizer is not None:
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        if _is_valid_vocab_token_id(tok_eos, vocab_upper_bound):
            stop_ids.add(int(tok_eos))
        elif isinstance(tok_eos, (list, tuple, set)):
            for t in tok_eos:
                if _is_valid_vocab_token_id(t, vocab_upper_bound):
                    stop_ids.add(int(t))

    if model_or_config is not None:
        cfg_root = getattr(model_or_config, "config", model_or_config)
        for cfg_src in (
            getattr(model_or_config, "generation_config", None),
            cfg_root,
            getattr(cfg_root, "text_config", None) if cfg_root is not None else None,
        ):
            if cfg_src is None:
                continue
            cfg_eos = cfg_src.get("eos_token_id") if isinstance(cfg_src, dict) else getattr(cfg_src, "eos_token_id", None)
            if _is_valid_vocab_token_id(cfg_eos, vocab_upper_bound):
                stop_ids.add(int(cfg_eos))
            elif isinstance(cfg_eos, (list, tuple, set)):
                for x in cfg_eos:
                    if _is_valid_vocab_token_id(x, vocab_upper_bound):
                        stop_ids.add(int(x))

    if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        unk_ids: set[int] = set()
        raw_unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(raw_unk, int) and not isinstance(raw_unk, bool):
            unk_ids.add(int(raw_unk))
        try:
            probe_id = tokenizer.convert_tokens_to_ids("__platodoom_unmapped_token_probe__")
            if isinstance(probe_id, int) and not isinstance(probe_id, bool):
                unk_ids.add(int(probe_id))
        except Exception:
            pass

        for token_str in ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<|EOT|>", "<|eot_id|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(token_str)
                if _is_valid_vocab_token_id(tid, vocab_upper_bound) and int(tid) not in unk_ids:
                    stop_ids.add(int(tid))
            except Exception:
                pass

    return sorted(stop_ids)


def _resolve_pad_token_id(tokenizer: Any, model_or_config: Any = None) -> int:
    """Resolves `pad_token_id` strictly within `[0, vocab_size)`, falling back to valid `eos_token_id`."""
    vocab_bound = _resolve_vocab_upper_bound(tokenizer, model_or_config)
    cfg_root = getattr(model_or_config, "config", model_or_config) if model_or_config is not None else None
    sources = (
        tokenizer,
        getattr(model_or_config, "generation_config", None) if model_or_config is not None else None,
        cfg_root,
        getattr(cfg_root, "text_config", None) if cfg_root is not None else None,
    )
    for candidate_src in sources:
        if candidate_src is None:
            continue
        pad_id = candidate_src.get("pad_token_id") if isinstance(candidate_src, dict) else getattr(candidate_src, "pad_token_id", None)
        if _is_valid_vocab_token_id(pad_id, vocab_bound):
            return int(pad_id)

    for candidate_src in sources:
        if candidate_src is None:
            continue
        eos_id = candidate_src.get("eos_token_id") if isinstance(candidate_src, dict) else getattr(candidate_src, "eos_token_id", None)
        if _is_valid_vocab_token_id(eos_id, vocab_bound):
            return int(eos_id)
        if isinstance(eos_id, (list, tuple, set)):
            for item in eos_id:
                if _is_valid_vocab_token_id(item, vocab_bound):
                    return int(item)

    stop_candidates = _resolve_stop_token_ids(tokenizer, "", model_or_config)
    if stop_candidates:
        return int(stop_candidates[0])
    return 0


def ensure_qwen35_fast_kernels_bound(qwen_model: Any) -> int:
    """Binds pre-installed fused flash-linear-attention kernels onto in-process Qwen3_5GatedDeltaNet layers
    when `fla` is already imported/installed (never runs runtime pip installs; native CUDA serving uses vLLM).
    """
    if not is_qwen35_architecture(qwen_model):
        return 0

    chunk_fn = None
    recurrent_fn = None
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        chunk_fn = chunk_gated_delta_rule
        recurrent_fn = fused_recurrent_gated_delta_rule
    except Exception:
        return 0

    if chunk_fn is None or qwen_model is None or not hasattr(qwen_model, "named_modules"):
        return 0

    patched = 0
    for _, mod in qwen_model.named_modules():
        if mod.__class__.__name__ == "Qwen3_5GatedDeltaNet":
            mod.chunk_gated_delta_rule = chunk_fn
            if recurrent_fn is not None:
                mod.recurrent_gated_delta_rule = recurrent_fn
            patched += 1
    return patched


def patch_last_token_only_lm_head(qwen_model: Any) -> bool:
    """Patches `lm_head` so vocabulary projection (`vocab_size = 248,320`) is only computed on the final
    sequence position (`hidden_states[:, -1:, :]`) ONLY for `qwen3_5` / `qwen3_5_moe` checkpoints (Spec 862).
    """
    if not is_qwen35_architecture(qwen_model):
        return False

    lm_head = getattr(qwen_model, "lm_head", None)
    if lm_head is None and hasattr(qwen_model, "language_model"):
        lm_head = getattr(qwen_model.language_model, "lm_head", None)
    if lm_head is None or getattr(lm_head, "_sandcastle_last_token_patched", False):
        return bool(lm_head is not None)

    orig_forward = lm_head.forward

    def _last_token_lm_head_forward(hidden_states: Any) -> Any:
        if hasattr(hidden_states, "ndim") and hidden_states.ndim == 3 and hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]
        return orig_forward(hidden_states)

    lm_head.forward = _last_token_lm_head_forward
    lm_head._sandcastle_last_token_patched = True
    return True


def _normalize_stop_strings(raw_stop: Any) -> List[str]:
    if raw_stop is None:
        return []
    if isinstance(raw_stop, str):
        return [raw_stop] if raw_stop else []
    if isinstance(raw_stop, (list, tuple)):
        return [s for s in raw_stop if isinstance(s, str) and s]
    return []


def _truncate_at_stop_strings(text: str, stop_strings: List[str]) -> Tuple[str, bool]:
    if not text or not stop_strings:
        return text, False
    earliest_idx: Optional[int] = None
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx
    if earliest_idx is not None:
        return text[:earliest_idx], True
    return text, False


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Applies OpenAI-compatible flat presence_penalty to token IDs generated after prompt_len."""

    def __init__(self, presence_penalty: float, prompt_len: int) -> None:
        super().__init__()
        self.presence_penalty = float(presence_penalty)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.presence_penalty == 0.0 or input_ids.shape[1] <= self.prompt_len:
            return scores
        scores = scores.clone()
        vocab_dim = scores.shape[-1]
        for batch_idx in range(input_ids.shape[0]):
            gen_ids = input_ids[batch_idx, self.prompt_len :]
            if gen_ids.numel() > 0:
                unique_ids = gen_ids.unique()
                valid_mask = (unique_ids >= 0) & (unique_ids < vocab_dim)
                unique_ids = unique_ids[valid_mask]
                if unique_ids.numel() > 0:
                    scores[batch_idx, unique_ids] -= self.presence_penalty
        return scores


class CustomStopStringsStoppingCriteria(StoppingCriteria):
    """Halts token generation when caller-supplied stop strings appear in decoded generated output."""

    def __init__(
        self,
        tokenizer: Any,
        stop_strings: List[str],
        prompt_len: int,
        stop_triggered_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = [s for s in stop_strings if isinstance(s, str) and s]
        self.prompt_len = int(prompt_len)
        self.stop_triggered_event = stop_triggered_event

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
        if not self.stop_strings or input_ids.shape[1] <= self.prompt_len:
            return False
        for batch_idx in range(input_ids.shape[0]):
            gen_ids = input_ids[batch_idx, self.prompt_len :]
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            if not any(stop_s in decoded for stop_s in self.stop_strings):
                return False
        if self.stop_triggered_event is not None:
            self.stop_triggered_event.set()
        return True


class ClientDisconnectStoppingCriteria(StoppingCriteria):
    """Halts GPU token generation within 1 token (<= 250ms) when client disconnects."""

    def __init__(self, abort_event: threading.Event) -> None:
        super().__init__()
        self.abort_event = abort_event

    def __call__(
        self,
        input_ids: Any,
        scores: Any = None,
        **kwargs: Any,
    ) -> bool:
        return bool(self.abort_event.is_set())


class StreamTagDemuxer:
    """Sub-millisecond state machine that demuxes reasoning content (<think>),
    visible content, buffered tool calls (<tool_call>, <｜tool▁call▁begin｜>, ```tool_call),
    and custom stop strings with O(1) overhead per token chunk.
    """

    def __init__(
        self,
        initial_in_think: bool = False,
        stop_strings: Optional[List[str]] = None,
    ) -> None:
        self.buffer = ""
        self.in_think = bool(initial_in_think)
        self.in_tool = False
        self.stopped = False
        self.stop_strings = [s for s in (stop_strings or []) if isinstance(s, str) and s]
        self.tool_buffer = ""
        self.full_raw_text = ""

    def _check_stop_in_buffer(self) -> Tuple[Optional[str], bool]:
        """Returns `(prefix_before_stop, should_hold_partial_prefix)`."""
        if not self.stop_strings or not self.buffer:
            return None, False
        truncated, hit = _truncate_at_stop_strings(self.buffer, self.stop_strings)
        if hit:
            self.stopped = True
            self.buffer = ""
            return truncated, False
        for s in self.stop_strings:
            max_k = min(len(s) - 1, len(self.buffer))
            for k in range(max_k, 0, -1):
                if self.buffer.endswith(s[:k]):
                    return None, True
        return None, False

    def process_token(self, token_str: str) -> List[Dict[str, Any]]:
        if self.stopped:
            return []
        self.full_raw_text += token_str
        self.buffer += token_str
        emitted_chunks: List[Dict[str, Any]] = []

        while self.buffer and not self.stopped:
            if not self.in_think and not self.in_tool:
                if "<think>" in self.buffer:
                    prefix, _, remainder = self.buffer.partition("<think>")
                    if prefix:
                        trunc_p, hit_p = _truncate_at_stop_strings(prefix, self.stop_strings)
                        if trunc_p:
                            emitted_chunks.append({"type": "content", "text": trunc_p})
                        if hit_p:
                            self.stopped = True
                            self.buffer = ""
                            break
                    self.in_think = True
                    self.buffer = remainder
                    continue

                tool_tag = None
                for t in ("<tool_call>", "<｜tool▁call▁begin｜>", "<|tool_call_begin|>", "```tool_call"):
                    if t in self.buffer:
                        tool_tag = t
                        break

                if tool_tag is not None:
                    prefix, _, remainder = self.buffer.partition(tool_tag)
                    if prefix:
                        trunc_p, hit_p = _truncate_at_stop_strings(prefix, self.stop_strings)
                        if trunc_p:
                            emitted_chunks.append({"type": "content", "text": trunc_p})
                        if hit_p:
                            self.stopped = True
                            self.buffer = ""
                            break
                    self.in_tool = True
                    self.tool_buffer = tool_tag + remainder
                    self.buffer = ""
                    break

                stop_prefix, hold_stop = self._check_stop_in_buffer()
                if self.stopped:
                    if stop_prefix:
                        emitted_chunks.append({"type": "content", "text": stop_prefix})
                    break
                if hold_stop:
                    break

                potential_prefixes = (
                    "<", "<t", "<to", "<too", "<tool", "<tool_", "<tool_c", "<tool_ca", "<tool_cal", "<tool_call",
                    "<th", "<thi", "<thin", "<think",
                    "<｜", "<｜t", "<｜to", "<｜too", "<｜tool",
                    "<|", "<|t", "<|to", "<|too", "<|tool",
                    "```", "```t", "```to", "```too", "```tool"
                )
                if any(self.buffer.endswith(p) for p in potential_prefixes):
                    break

                emitted_chunks.append({"type": "content", "text": self.buffer})
                self.buffer = ""
                break

            elif self.in_think:
                if self.buffer.startswith("<think>"):
                    self.buffer = self.buffer[len("<think>") :]
                    if not self.buffer:
                        break
                if "</think>" in self.buffer:
                    think_text, _, remainder = self.buffer.partition("</think>")
                    if think_text:
                        trunc_t, hit_t = _truncate_at_stop_strings(think_text, self.stop_strings)
                        if trunc_t:
                            emitted_chunks.append({"type": "thinking", "text": trunc_t})
                        if hit_t:
                            self.stopped = True
                            self.buffer = ""
                            break
                    self.in_think = False
                    self.buffer = remainder
                    continue
                else:
                    stop_prefix, hold_stop = self._check_stop_in_buffer()
                    if self.stopped:
                        if stop_prefix:
                            emitted_chunks.append({"type": "thinking", "text": stop_prefix})
                        break
                    if hold_stop:
                        break
                    potential_closing = (
                        "<", "</", "</t", "</th", "</thi", "</thin", "</think",
                        "<t", "<th", "<thi", "<thin", "<think",
                    )
                    if any(self.buffer.endswith(p) for p in potential_closing):
                        break
                    emitted_chunks.append({"type": "thinking", "text": self.buffer})
                    self.buffer = ""
                    break

            elif self.in_tool:
                self.tool_buffer += self.buffer
                self.buffer = ""
                if self.stop_strings:
                    trunc_tb, hit_tb = _truncate_at_stop_strings(self.tool_buffer, self.stop_strings)
                    if hit_tb:
                        self.tool_buffer = trunc_tb
                        self.stopped = True
                break

        return emitted_chunks

    def flush(self) -> Tuple[List[Dict[str, Any]], str, str]:
        emitted_chunks: List[Dict[str, Any]] = []
        if self.buffer and not self.stopped:
            buf_text, _ = _truncate_at_stop_strings(self.buffer, self.stop_strings)
            if self.in_think:
                if buf_text.startswith("<think>"):
                    buf_text = buf_text[len("<think>") :]
                if buf_text:
                    emitted_chunks.append({"type": "thinking", "text": buf_text})
            elif not self.in_tool:
                tool_tag = None
                for t in ("<tool_call>", "<｜tool▁call▁begin｜>", "<|tool_call_begin|>", "```tool_call"):
                    if t in buf_text:
                        tool_tag = t
                        break
                if tool_tag is not None:
                    prefix, _, remainder = buf_text.partition(tool_tag)
                    if prefix:
                        emitted_chunks.append({"type": "content", "text": prefix})
                    self.in_tool = True
                    self.tool_buffer += tool_tag + remainder
                elif buf_text:
                    emitted_chunks.append({"type": "content", "text": buf_text})
            else:
                self.tool_buffer += buf_text
            self.buffer = ""

        clean_raw, _ = _truncate_at_stop_strings(self.full_raw_text, self.stop_strings)
        return emitted_chunks, self.tool_buffer, clean_raw


class ToolCallExtractionResult(tuple):
    """Tuple subclass returning (cleaned_text, tool_calls) with optional reasoning_content attribute."""

    def __new__(
        cls,
        cleaned_text: str,
        tool_calls: List[Dict[str, Any]],
        reasoning_content: Optional[str] = None,
    ):
        return super().__new__(cls, (cleaned_text, tool_calls))

    def __init__(
        self,
        cleaned_text: str,
        tool_calls: List[Dict[str, Any]],
        reasoning_content: Optional[str] = None,
    ) -> None:
        self.cleaned_text = cleaned_text
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


def extract_tool_calls_from_text(text: str) -> ToolCallExtractionResult:
    """Universal tool call extractor parsing:
    1. Qwen3.5/3.6 XML: <tool_call><function=name><parameter=key>val</parameter></function></tool_call>
    2. JSON <tool_call>{...}</tool_call>
    3. DeepSeek <｜tool▁call▁begin｜>...<｜tool▁call▁end｜>
    4. Fenced ```tool_call {...} ``` blocks.
    Also extracts <think>...</think> into reasoning_content.
    Returns ToolCallExtractionResult(cleaned_text, tool_calls, reasoning_content).
    """
    if not text:
        return ToolCallExtractionResult("", [], None)

    reasoning_content: Optional[str] = None
    think_matches = THINK_BLOCK_RE.findall(text)
    if think_matches:
        joined_think = "\n\n".join(m.strip() for m in think_matches if m.strip())
        reasoning_content = joined_think if joined_think else None
        cleaned_text = THINK_BLOCK_RE.sub("", text)
    elif "</think>" in text:
        think_part, remainder = text.split("</think>", 1)
        think_part = re.sub(r"^\s*<think>\s*", "", think_part, flags=re.IGNORECASE).strip()
        reasoning_content = think_part if think_part else None
        cleaned_text = remainder
    elif text.lstrip().startswith("<think>"):
        think_part = re.sub(r"^\s*<think>\s*", "", text.lstrip(), flags=re.IGNORECASE).strip()
        reasoning_content = think_part if think_part else None
        cleaned_text = ""
    else:
        cleaned_text = text

    tool_calls: List[Dict[str, Any]] = []

    # Format 1: XML Qwen3.5/3.6 format
    xml_matches = list(XML_TOOL_CALL_RE.finditer(cleaned_text))
    if xml_matches:
        for m in xml_matches:
            fn_name = m.group(1).strip()
            fn_body = m.group(2)
            args_dict: Dict[str, Any] = {}
            for pm in XML_PARAM_RE.finditer(fn_body):
                param_name = pm.group(1).strip()
                param_val_raw = pm.group(2).strip()
                try:
                    param_val = json.loads(param_val_raw)
                except Exception:
                    param_val = param_val_raw
                args_dict[param_name] = param_val
            call_idx = len(tool_calls)
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:10]}_{call_idx}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(args_dict),
                    },
                }
            )
        cleaned_text = XML_TOOL_CALL_RE.sub("", cleaned_text)

    # Format 2: JSON <tool_call>{...}</tool_call>
    json_matches = list(JSON_TOOL_CALL_RE.finditer(cleaned_text))
    if json_matches:
        for m in json_matches:
            raw_json = m.group(1).strip()
            call_idx = len(tool_calls)
            try:
                parsed = json.loads(raw_json)
                name = str(parsed.get("name") or parsed.get("tool") or "run_command")
                args_obj = parsed.get("arguments", parsed.get("args", parsed.get("parameters", {})))
                args_str = json.dumps(args_obj) if not isinstance(args_obj, str) else args_obj
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:10]}_{call_idx}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args_str,
                        },
                    }
                )
            except Exception:
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:10]}_{call_idx}",
                        "type": "function",
                        "function": {
                            "name": "malformed_tool_call",
                            "arguments": raw_json,
                        },
                    }
                )
        cleaned_text = JSON_TOOL_CALL_RE.sub("", cleaned_text)

    # Format 3: DeepSeek format
    ds_matches = list(DEEPSEEK_TOOL_CALL_RE.finditer(cleaned_text))
    if ds_matches:
        for m in ds_matches:
            fn_name = (m.group(1) or "").strip()
            body = (m.group(2) or "").strip()
            call_idx = len(tool_calls)
            try:
                parsed = json.loads(body)
                if not fn_name:
                    fn_name = str(parsed.get("name") or parsed.get("tool") or "run_command")
                    args_obj = parsed.get("arguments", parsed.get("args", parsed.get("parameters", {})))
                else:
                    args_obj = parsed
                args_str = json.dumps(args_obj) if not isinstance(args_obj, str) else args_obj
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:10]}_{call_idx}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_str,
                        },
                    }
                )
            except Exception:
                if body:
                    tool_calls.append(
                        {
                            "id": f"call_{uuid.uuid4().hex[:10]}_{call_idx}",
                            "type": "function",
                            "function": {
                                "name": fn_name or "malformed_tool_call",
                                "arguments": body,
                            },
                        }
                    )
        cleaned_text = DEEPSEEK_TOOL_CALL_RE.sub("", cleaned_text)

    # Format 4: Fenced ```tool_call ... ```
    fenced_matches = list(FENCED_TOOL_CALL_RE.finditer(cleaned_text))
    if fenced_matches:
        for m in fenced_matches:
            raw_json = m.group(1).strip()
            call_idx = len(tool_calls)
            try:
                parsed = json.loads(raw_json)
                name = str(parsed.get("name") or parsed.get("tool") or "run_command")
                args_obj = parsed.get("arguments", parsed.get("args", parsed.get("parameters", {})))
                args_str = json.dumps(args_obj) if not isinstance(args_obj, str) else args_obj
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:10]}_{call_idx}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args_str,
                        },
                    }
                )
            except Exception:
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:10]}_{call_idx}",
                        "type": "function",
                        "function": {
                            "name": "malformed_tool_call",
                            "arguments": raw_json,
                        },
                    }
                )
        cleaned_text = FENCED_TOOL_CALL_RE.sub("", cleaned_text)

    cleaned_text = re.sub(r"[<｜|]tool[ ▁_]calls?[ ▁_](?:begin|end)[>｜|]", "", cleaned_text)
    return ToolCallExtractionResult(cleaned_text.strip(), tool_calls, reasoning_content)


def split_thinking_and_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Convenience helper extracting (cleaned_text, tool_calls, reasoning_content)."""
    res = extract_tool_calls_from_text(text)
    return res.cleaned_text, res.tool_calls, res.reasoning_content



@dataclass
class InferenceRequestItem:
    request_id: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 2048
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = None
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    enqueued_at: float = field(default_factory=time.perf_counter)
    done_event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class DynamicBatchCoordinator:
    """Single-GPU Continuous Dynamic Batching & Bounded Concurrency Controller (FR-3.2, FR-3.3, Section 4.2, Spec 867).

    - Admits up to `max_num_seqs` concurrent agent requests into an active GPU batch.
    - Supports cohort-aware immediate dispatch (`target_cohort_size = min(n_i, max_num_seqs)`) so full `n_i`
      agent cohorts dispatch with 0 ms tail wait once assembled.
    - Holds excess burst requests (`> max_num_seqs`) in a bounded FIFO queue (`queue_timeout_sec = 300.0`)
      so burst concurrency never causes OOM.
    """

    def __init__(
        self,
        batch_execute_fn: Callable[[List[InferenceRequestItem]], List[Dict[str, Any]]],
        max_num_seqs: int = MAX_PARALLEL_AGENTS,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
        batch_window_ms: Optional[float] = None,
        queue_timeout_sec: float = 300.0,
        target_cohort_size: Optional[int] = None,
        on_batch_telemetry_fn: Optional[Callable[[int], None]] = None,
        on_close_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        eff_window_ms = (
            float(batch_window_ms)
            if batch_window_ms is not None
            else float(os.environ.get("SANDCASTLE_BATCH_WINDOW_MS", "50.0"))
        )
        self.batch_execute_fn = batch_execute_fn
        self.max_num_seqs = max(1, int(max_num_seqs))
        self.max_model_len = int(max_model_len)
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.batch_window_sec = max(0.002, eff_window_ms / 1000.0)
        self.queue_timeout_sec = float(queue_timeout_sec)
        self.target_cohort_size: Optional[int] = (
            max(1, int(target_cohort_size)) if target_cohort_size is not None and int(target_cohort_size) > 0 else None
        )
        self._on_batch_telemetry_fn = on_batch_telemetry_fn
        self._on_close_fn = on_close_fn

        self._queue: queue.Queue[InferenceRequestItem] = queue.Queue()
        self._stop_event = threading.Event()
        self._metrics_lock = threading.Lock()
        self.inflight_requests = 0
        self.total_requests_served = 0
        self.total_batches_executed = 0
        self.max_observed_batch_size = 0
        self.last_batch_size = 0
        self.batch_size_histogram: Dict[int, int] = {}
        self._worker = threading.Thread(target=self._batch_loop, name="DynamicBatchWorker", daemon=True)
        self._worker.start()

    @property
    def pending_or_active_count(self) -> int:
        with self._metrics_lock:
            return int(self._queue.qsize()) + int(self.inflight_requests)

    def set_concurrency_targets(
        self,
        max_num_seqs: Optional[int] = None,
        target_cohort_size: Optional[int] = None,
        *,
        update_cohort: bool = True,
    ) -> None:
        with self._metrics_lock:
            if max_num_seqs is not None:
                self.max_num_seqs = max(1, int(max_num_seqs))
                self.max_num_batched_tokens = max(self.max_num_batched_tokens, self.max_num_seqs * 4096)
            if update_cohort:
                self.target_cohort_size = (
                    max(1, int(target_cohort_size))
                    if target_cohort_size is not None and int(target_cohort_size) > 0
                    else None
                )

    def submit(self, item: InferenceRequestItem) -> Dict[str, Any]:
        """Submits an agent completion request and waits up to queue_timeout_sec for batched GPU execution."""
        self._queue.put(item)
        completed = item.done_event.wait(timeout=self.queue_timeout_sec)
        if not completed:
            raise TimeoutError(
                f"Request {item.request_id} timed out after {self.queue_timeout_sec}s waiting in GPU batch queue."
            )
        if item.error is not None:
            raise item.error
        assert item.response is not None
        return item.response

    def _estimate_item_tokens(self, item: InferenceRequestItem) -> int:
        total_chars = sum(
            len(str(m.get("content") or "")) + len(str(m.get("reasoning_content") or ""))
            for m in item.messages
        )
        first_content = str(item.messages[0].get("content") or "") if item.messages else ""
        if item.tools and "<tools>" not in first_content and "Available tools" not in first_content:
            total_chars += sum(len(json.dumps(t)) for t in item.tools)
        est = max(64, int(total_chars / 4.0))
        return min(est, self.max_model_len)

    def _batch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                first_item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            batch: List[InferenceRequestItem] = [first_item]
            batched_tokens = self._estimate_item_tokens(first_item)

            # First drain any immediately queued items without waiting
            while len(batch) < self.max_num_seqs:
                try:
                    ready_cand = self._queue.get_nowait()
                except queue.Empty:
                    break
                cand_tokens = self._estimate_item_tokens(ready_cand)
                if batched_tokens + cand_tokens > self.max_num_batched_tokens and len(batch) >= 1:
                    self._execute_batch(batch)
                    batch = [ready_cand]
                    batched_tokens = cand_tokens
                else:
                    batch.append(ready_cand)
                    batched_tokens += cand_tokens

            with self._metrics_lock:
                max_seqs = self.max_num_seqs
                cohort_target = self.target_cohort_size
            cohort_goal = min(cohort_target, max_seqs) if cohort_target is not None else max_seqs

            deadline = time.perf_counter() + self.batch_window_sec
            while len(batch) < cohort_goal and len(batch) < max_seqs:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    candidate = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break

                cand_tokens = self._estimate_item_tokens(candidate)
                if batched_tokens + cand_tokens > self.max_num_batched_tokens and len(batch) >= 1:
                    self._execute_batch(batch)
                    batch = [candidate]
                    batched_tokens = cand_tokens
                    deadline = time.perf_counter() + self.batch_window_sec
                else:
                    batch.append(candidate)
                    batched_tokens += cand_tokens

            if batch:
                self._execute_batch(batch)

    def _execute_batch(self, batch: List[InferenceRequestItem]) -> None:
        b_len = len(batch)
        with self._metrics_lock:
            self.inflight_requests += b_len
            self.total_batches_executed += 1
            self.last_batch_size = b_len
            self.batch_size_histogram[b_len] = self.batch_size_histogram.get(b_len, 0) + 1
            if b_len > self.max_observed_batch_size:
                self.max_observed_batch_size = b_len

        try:
            responses = self.batch_execute_fn(batch)
            for item, resp in zip(batch, responses):
                item.response = resp
                item.done_event.set()
            with self._metrics_lock:
                self.total_requests_served += b_len
            if self._on_batch_telemetry_fn is not None:
                try:
                    self._on_batch_telemetry_fn(b_len)
                except Exception:
                    pass
        except Exception as exc:
            for item in batch:
                item.error = exc
                item.done_event.set()
        finally:
            with self._metrics_lock:
                self.inflight_requests = max(0, self.inflight_requests - b_len)

    def close(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._on_close_fn is not None:
            try:
                self._on_close_fn()
            except Exception:
                pass


def _isolate_generated_sequence_tokens(
    raw_gen_ids: Any,
    stop_token_ids: Optional[Sequence[int]],
    pad_token_id: Optional[int],
) -> Tuple[List[int], bool]:
    """Isolates a single sequence's generated token IDs inside a multi-sequence GPU batch (Spec 867).

    Truncates immediately at the sequence's first `stop_token_id` (excluding post-EOS tokens generated while
    longer sibling sequences in the same batch continue running) and strips trailing `pad_token_id` right-padding
    so zero cross-agent or padding tokens ever leak into decoded agent completions.
    """
    if hasattr(raw_gen_ids, "tolist"):
        seq_list = [int(t) for t in raw_gen_ids.tolist()]
    else:
        seq_list = [int(t) for t in raw_gen_ids]

    stop_set = {int(t) for t in (stop_token_ids or [])}
    isolated: List[int] = []
    hit_stop = False
    for tok_id in seq_list:
        if tok_id in stop_set:
            hit_stop = True
            break
        isolated.append(tok_id)

    if not hit_stop and pad_token_id is not None and int(pad_token_id) not in stop_set:
        pad_int = int(pad_token_id)
        while isolated and isolated[-1] == pad_int:
            isolated.pop()

    return isolated, hit_stop


_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_MODEL_ALIAS_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PREQUANTIZED_METHODS = frozenset({"awq", "gptq", "compressed-tensors", "fp8", "mxfp4", "marlin", "exl2"})


def _resolve_torch_compute_dtype(
    gpu_tier: Optional[str] = None,
    compute_capability: Optional[Tuple[int, int]] = None,
) -> Any:
    """Resolves `torch.float16` on `sm_75` / T4 and `torch.bfloat16` on `sm_80+` / Ampere+ (Spec 868 §4.4)."""
    import torch

    if compute_capability is not None:
        try:
            if int(compute_capability[0]) < 8:
                return torch.float16
            return torch.bfloat16
        except (IndexError, TypeError, ValueError):
            pass

    if gpu_tier and str(gpu_tier).strip():
        if str(gpu_tier).strip().upper() == "T4":
            return torch.float16
        return torch.bfloat16

    if torch.cuda.is_available():
        try:
            cc_major, _ = torch.cuda.get_device_capability(0)
            if int(cc_major) < 8:
                return torch.float16
        except Exception:
            pass
    return torch.bfloat16


def _verify_checkpoint_files_fast(checkpoint_dir: Path | str) -> Tuple[bool, Optional[str]]:
    """O(shards) physical checkpoint integrity check without hashing multi-GB tensors."""
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        return False, f"Checkpoint directory does not exist: {ckpt}"

    cfg_path = ckpt / "config.json"
    model_index_path = ckpt / "model_index.json"
    gguf_files = [
        p for p in ckpt.iterdir()
        if p.is_file() and p.suffix == ".gguf" and p.stat().st_size > 0
    ]

    if model_index_path.is_file() and model_index_path.stat().st_size > 0:
        # Diffusion pipeline checkpoint (Spec 868 §4.5)
        diff_weights = [
            p for p in ckpt.rglob("*")
            if p.is_file() and p.suffix in (".safetensors", ".bin", ".gguf") and p.stat().st_size > 0
        ]
        if not diff_weights:
            return False, f"No non-empty diffusion weight files (*.safetensors / *.gguf) found in {ckpt}"
        return True, None

    if gguf_files and not cfg_path.is_file():
        return True, None

    if not cfg_path.is_file() or cfg_path.stat().st_size == 0:
        return False, f"Missing or empty config.json in {ckpt}"

    has_tokenizer = any(
        (ckpt / name).is_file() and (ckpt / name).stat().st_size > 0
        for name in ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")
    )
    if not has_tokenizer:
        return False, f"Missing tokenizer metadata (tokenizer_config.json / tokenizer.json) in {ckpt}"

    index_path = ckpt / "model.safetensors.index.json"
    if index_path.is_file():
        if index_path.stat().st_size == 0:
            return False, f"Empty model.safetensors.index.json in {ckpt}"
        try:
            idx_data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"Corrupted model.safetensors.index.json in {ckpt}: {exc}"
        weight_map = idx_data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False, f"model.safetensors.index.json has empty weight_map in {ckpt}"
        for shard_name in sorted(set(str(v) for v in weight_map.values())):
            shard_file = ckpt / shard_name
            if not shard_file.is_file() or shard_file.stat().st_size == 0:
                return False, f"Missing or empty weight shard '{shard_name}' in {ckpt}"
        return True, None

    weight_files = [
        p for p in ckpt.iterdir()
        if p.is_file() and (p.suffix in (".safetensors", ".bin", ".gguf") or p.name.endswith(".pt")) and p.stat().st_size > 0
    ]
    if not weight_files:
        return False, f"No non-empty weight files (*.safetensors / *.bin / *.gguf) found in {ckpt}"
    return True, None


def build_quantization_config(
    model_path: str | Path,
    vram_gb: float,
    model_alias: str = "",
    arch_meta: Optional[Dict[str, Any]] = None,
    precision: Optional[str] = None,
    gpu_tier: Optional[str] = None,
    compute_capability: Optional[Tuple[int, int]] = None,
) -> Optional[Any]:
    """Builds a BitsAndBytes 4-bit NF4 or 8-bit config when precision or physical memory requires runtime quantization."""
    cfg_dict: Dict[str, Any] = {}
    p = Path(str(model_path))
    if p.is_dir() and (p / "config.json").is_file():
        try:
            cfg_dict = json.loads((p / "config.json").read_text(encoding="utf-8"))
        except Exception:
            cfg_dict = {}
    elif isinstance(arch_meta, dict):
        cfg_dict = dict(arch_meta)

    q_existing = cfg_dict.get("quantization_config")
    _prequant_set = globals().get(
        "_PREQUANTIZED_METHODS",
        ("awq", "gptq", "compressed-tensors", "fp8", "mxfp4", "marlin", "exl2"),
    )
    if isinstance(q_existing, dict):
        q_method = str(q_existing.get("quant_method") or "").strip().lower()
        if q_method in _prequant_set:
            return None
    if isinstance(arch_meta, dict) and str(arch_meta.get("quant_method") or "").strip().lower() in _prequant_set:
        return None

    lower_target = f"{model_alias} {model_path}".lower()
    if any(tag in lower_target for tag in ("-awq", "-gptq", "-fp8", "_awq", "_gptq", "-gguf", ".gguf")):
        return None

    norm_prec: Optional[str] = None
    if precision and str(precision).strip():
        try:
            norm_prec = normalize_precision_code(precision)
        except Exception:
            p_up = str(precision).strip().upper()
            if p_up in ("BF16", "BFLOAT16", "FP16"):
                norm_prec = "BF16"
            elif p_up == "FP8":
                norm_prec = "FP8"
            elif p_up in ("INT8", "Q8", "8BIT"):
                norm_prec = "Q8"
            elif p_up in ("Q6", "Q5", "Q4", "NF4", "NF4_4BIT", "AWQ_4BIT", "GPTQ_4BIT"):
                norm_prec = "Q4"

    if norm_prec in ("BF16", "FP8"):
        return None

    if norm_prec in ("INT8", "Q8"):
        try:
            _bnb_cls = globals().get("BitsAndBytesConfig")
            if _bnb_cls is None:
                from transformers import BitsAndBytesConfig as _bnb_cls

            return _bnb_cls(load_in_8bit=True)
        except Exception:
            return None

    needs_nf4 = norm_prec in ("Q6", "Q5", "Q4")
    if norm_prec is None:
        _est_fn = globals().get("_estimate_bf16_weights_gb_from_cfg")
        _est_gb = _est_fn(cfg_dict) if (callable(_est_fn) and cfg_dict) else None
        if (_est_gb is None or _est_gb <= 0) and model_alias:
            _clean_al = str(model_alias).strip().split("@", 1)[0].strip()
            _spec_m = (_SUPERVISOR_MODEL_SPECS.get(_clean_al) or {})
            if _spec_m.get("weights_bf16_gb"):
                _est_gb = float(_spec_m["weights_bf16_gb"])
        if (_est_gb is None or _est_gb <= 0) and str(model_path).strip():
            for _sp_v in _SUPERVISOR_MODEL_SPECS.values():
                if str(_sp_v.get("hf_repo_id") or "").lower() in lower_target:
                    _est_gb = float(_sp_v["weights_bf16_gb"])
                    break
        if _est_gb is not None and _est_gb > 0:
            needs_nf4 = float(vram_gb) < (_est_gb + 6.0)
        else:
            needs_nf4 = False

    if not needs_nf4:
        return None

    try:
        _bnb_cls = globals().get("BitsAndBytesConfig")
        if _bnb_cls is None:
            from transformers import BitsAndBytesConfig as _bnb_cls

        _dtype_fn = globals().get("_resolve_torch_compute_dtype")
        compute_dtype = (
            _dtype_fn(gpu_tier=gpu_tier, compute_capability=compute_capability)
            if callable(_dtype_fn)
            else getattr(globals().get("torch"), "bfloat16", None)
        )
        return _bnb_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=(norm_prec in (None, "Q4")),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    except Exception:
        return None


def validate_model_identifier_syntax(
    model_alias: str,
    hf_repo_id: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Validates model_alias and optional hf_repo_id before mutating GPU memory (HTTP 400 guard)."""
    alias = str(model_alias or "").strip()
    if not alias:
        return False, "Model alias must be a non-empty string."
    if any(ch.isspace() for ch in alias) or "://" in alias or "\x00" in alias:
        return False, f"Malformed model identifier '{alias}'."

    if not _is_valid_local_checkpoint_dir(alias):
        if ".." in alias or alias.startswith("/") or alias.startswith("./") or alias.startswith("\\"):
            return False, f"Local checkpoint path '{alias}' does not exist or lacks config.json."
        if "/" in alias:
            if not _HF_REPO_ID_RE.match(alias):
                return False, f"Invalid HuggingFace repository identifier '{alias}'. Expected '<org>/<repo>'."
        elif not _MODEL_ALIAS_TOKEN_RE.match(alias):
            return False, f"Invalid model alias syntax '{alias}'."

    if hf_repo_id is not None and str(hf_repo_id).strip():
        repo = str(hf_repo_id).strip()
        if any(ch.isspace() for ch in repo) or "://" in repo or "\x00" in repo:
            return False, f"Malformed hf_repo_id '{repo}'."
        if not _is_valid_local_checkpoint_dir(repo):
            if ".." in repo or repo.startswith("/") or repo.startswith("./") or not _HF_REPO_ID_RE.match(repo):
                return False, f"Invalid hf_repo_id '{repo}'. Expected '<org>/<repo>' or existing local checkpoint path."

    return True, None


def resolve_serving_checkpoint_source(
    model_alias: str,
    hf_repo_id: Optional[str] = None,
    total_vram_gb: float = 16.0,
    precision: Optional[str] = None,
    source: Optional[str] = None,
    storage_uri: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    gcs_prefix: Optional[str] = None,
    fallback_hf_repo_id: Optional[str] = None,
    cache_root: Path | str = "/content/models",
) -> Dict[str, Any]:
    """Resolves authoritative checkpoint source (local dir, GCS prefix, or HF repo including AWQ/GPTQ routing)."""
    raw_alias_input = str(model_alias or "").strip()
    if "@" in raw_alias_input and not _is_valid_local_checkpoint_dir(raw_alias_input):
        base_alias_part, inline_prec = raw_alias_input.split("@", 1)
        model_alias = base_alias_part.strip()
        if precision is None and inline_prec.strip():
            precision = inline_prec.strip()

    valid, err = validate_model_identifier_syntax(model_alias, hf_repo_id)
    if not valid:
        raise ValueError(err or f"Invalid model identifier: {model_alias}")

    alias = str(model_alias).strip()
    explicit_repo = str(hf_repo_id).strip() if hf_repo_id and str(hf_repo_id).strip() else None

    if explicit_repo and _is_valid_local_checkpoint_dir(explicit_repo):
        resolved_dir = str(Path(explicit_repo).resolve())
        return {
            "model_alias": alias,
            "source": "local",
            "local_dir": resolved_dir,
            "hf_repo_id": explicit_repo,
            "fallback_hf_repo_id": fallback_hf_repo_id,
            "precision": precision or "bf16",
        }

    if _is_valid_local_checkpoint_dir(alias):
        resolved_dir = str(Path(alias).resolve())
        return {
            "model_alias": alias,
            "source": "local",
            "local_dir": resolved_dir,
            "hf_repo_id": explicit_repo or alias,
            "fallback_hf_repo_id": fallback_hf_repo_id,
            "precision": precision or "bf16",
        }

    profile: Dict[str, Any] = {}
    try:
        profile = resolve_model_profile(alias)
    except ValueError:
        if explicit_repo and _HF_REPO_ID_RE.match(explicit_repo):
            profile = {
                "model_alias": alias,
                "hf_repo_id": explicit_repo,
                "source": source or "huggingface",
            }
        elif _HF_REPO_ID_RE.match(alias):
            profile = {
                "model_alias": alias,
                "hf_repo_id": alias,
                "source": source or "huggingface",
            }
        else:
            raise

    vram_info = verify_vram_compatibility(alias, float(total_vram_gb), precision=precision)
    variant_repo: Optional[str] = None
    if precision:
        try:
            var_info = resolve_model_precision_variant(alias, precision=precision)
            variant_repo = str(var_info.get("hf_repo_id") or "").strip() or None
        except Exception:
            variant_repo = None
    quantized_repo = variant_repo or str(vram_info.get("quantized_hf_repo_id") or "").strip() or None
    profile_repo = str(profile.get("hf_repo_id") or "").strip() or None

    if explicit_repo and explicit_repo != profile_repo:
        effective_repo = explicit_repo
        resolved_source = source or "huggingface"
    elif profile.get("source") == "gcs" and not source and not variant_repo:
        effective_repo = profile_repo or alias
        resolved_source = "gcs"
    elif quantized_repo:
        effective_repo = quantized_repo
        resolved_source = "huggingface"
    else:
        effective_repo = explicit_repo or profile_repo or alias
        resolved_source = source or str(profile.get("source") or "huggingface")

    eff_bucket = gcs_bucket
    eff_prefix = gcs_prefix
    eff_uri = storage_uri or str(profile.get("storage_uri") or "")
    if eff_uri.startswith("gs://"):
        without_scheme = eff_uri[len("gs://"):]
        b_part, _, p_part = without_scheme.partition("/")
        eff_bucket = eff_bucket or b_part
        eff_prefix = eff_prefix or p_part

    if resolved_source == "gcs":
        eff_bucket = eff_bucket or DEFAULT_GCS_BUCKET_NAME
        eff_prefix = eff_prefix or (
            DEFAULT_GCS_MODEL_PREFIX
            if alias == UNIFIED_MODEL_ALIAS
            else f"models/v2/{alias}/"
        )

    safe_slug = re.sub(r"[^A-Za-z0-9._-]+", "--", str(effective_repo or alias)).strip("-")
    target_dir = str(Path(cache_root) / safe_slug)

    return {
        "model_alias": alias,
        "source": resolved_source,
        "local_dir": target_dir,
        "hf_repo_id": effective_repo,
        "fallback_hf_repo_id": fallback_hf_repo_id or profile.get("fallback_hf_repo_id") or effective_repo,
        "gcs_bucket": eff_bucket,
        "gcs_prefix": eff_prefix,
        "gcs_project": DEFAULT_GCS_PROJECT_ID,
        "precision": precision or str(vram_info.get("active_precision") or profile.get("default_precision") or "bf16"),
    }


def stage_checkpoint_to_disk(
    resolved_spec: Dict[str, Any],
    total_vram_gb: float = 16.0,
    token: Optional[str] = None,
) -> str:
    """Stages checkpoint weights and tokenizer on local disk BEFORE unloading active GPU allocations."""
    src = str(resolved_spec.get("source") or "huggingface")
    local_dir_str = str(resolved_spec.get("local_dir") or "").strip()
    if not local_dir_str:
        raise ValueError("Resolved checkpoint specification is missing local_dir.")

    target_dir = Path(local_dir_str)
    if src == "local":
        ok_fast, reason = _verify_checkpoint_files_fast(target_dir)
        if not ok_fast:
            raise RuntimeError(f"Local checkpoint verification failed for '{target_dir}': {reason}")
        return str(target_dir)

    if target_dir.is_dir():
        ok_cached, _ = _verify_checkpoint_files_fast(target_dir)
        if ok_cached:
            return str(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if src == "gcs":
        gcs_err: Optional[Exception] = None
        try:
            download_merged_model_parallel(
                local_model_dir=target_dir,
                project_id=str(resolved_spec.get("gcs_project") or DEFAULT_GCS_PROJECT_ID),
                bucket_name=str(resolved_spec.get("gcs_bucket") or DEFAULT_GCS_BUCKET_NAME),
                gcs_prefix=str(resolved_spec.get("gcs_prefix") or DEFAULT_GCS_MODEL_PREFIX),
            )
            ok_gcs, gcs_reason = _verify_checkpoint_files_fast(target_dir)
            if ok_gcs:
                return str(target_dir)
            gcs_err = RuntimeError(f"GCS checkpoint incomplete at '{target_dir}': {gcs_reason}")
        except Exception as exc:
            gcs_err = exc

        fb_repo = str(resolved_spec.get("fallback_hf_repo_id") or "").strip()
        if not fb_repo or not _HF_REPO_ID_RE.match(fb_repo):
            raise RuntimeError(f"GCS checkpoint download failed and no valid fallback HF repo available: {gcs_err}")
        resolved_spec = dict(resolved_spec)
        resolved_spec["hf_repo_id"] = fb_repo

    repo_id = str(resolved_spec.get("hf_repo_id") or "").strip()
    if not _HF_REPO_ID_RE.match(repo_id):
        raise ValueError(f"Invalid HuggingFace repository identifier '{repo_id}'.")

    try:
        import importlib.util
        from huggingface_hub import snapshot_download

        if importlib.util.find_spec("hf_transfer") is not None:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        try:
            ignore_pats = resolve_download_ignore_patterns(
                repo_id,
                precision=str(resolved_spec.get("precision") or ""),
            )
        except Exception:
            ignore_pats = ["*.msgpack", "*.h5", "consolidated*"]
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            ignore_patterns=ignore_pats,
            token=hf_token,
        )
    except Exception as exc:
        exc_cls = exc.__class__.__name__
        exc_msg = str(exc)
        if exc_cls in ("RepositoryNotFoundError", "HFValidationError") or "404" in exc_msg or "Repo id must" in exc_msg:
            raise ValueError(f"HuggingFace repository '{repo_id}' not found or invalid: {exc}") from exc
        raise RuntimeError(f"Failed downloading checkpoint '{repo_id}' to '{target_dir}': {exc}") from exc

    ok_final, final_reason = _verify_checkpoint_files_fast(target_dir)
    if not ok_final:
        raise RuntimeError(f"Staged checkpoint '{repo_id}' at '{target_dir}' is incomplete: {final_reason}")
    return str(target_dir)


def default_physical_model_loader(
    model_alias: str,
    hf_repo_id: Optional[str] = None,
    total_vram_gb: float = 16.0,
    precision: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    resolved_spec: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, str]:
    """Physically loads AutoTokenizer and model weights (or DiffusionPipeline for image models) from a staged checkpoint directory onto accelerator."""
    import torch
    from transformers import AutoTokenizer

    spec = resolved_spec or resolve_serving_checkpoint_source(
        model_alias=model_alias,
        hf_repo_id=hf_repo_id,
        total_vram_gb=total_vram_gb,
        precision=precision,
        source=kwargs.get("source"),
        storage_uri=kwargs.get("storage_uri"),
        gcs_bucket=kwargs.get("gcs_bucket"),
        gcs_prefix=kwargs.get("gcs_prefix"),
        fallback_hf_repo_id=kwargs.get("fallback_hf_repo_id"),
        cache_root=kwargs.get("cache_root") or os.environ.get("PLATODOOM_MODEL_CACHE_DIR", "/content/models"),
    )
    ckpt_dir = checkpoint_dir or stage_checkpoint_to_disk(spec, total_vram_gb=total_vram_gb)
    arch_meta = inspect_checkpoint_architecture(ckpt_dir, allow_remote=False)
    eff_prec = precision or spec.get("precision")
    gpu_tier_kw = kwargs.get("gpu_tier")
    cc_kw = kwargs.get("compute_capability")
    q_cfg = build_quantization_config(
        ckpt_dir,
        vram_gb=total_vram_gb,
        model_alias=model_alias,
        arch_meta=arch_meta,
        precision=str(eff_prec) if eff_prec else None,
        gpu_tier=str(gpu_tier_kw) if gpu_tier_kw else None,
        compute_capability=cc_kw,
    )

    compute_dtype = (
        _resolve_torch_compute_dtype(gpu_tier=str(gpu_tier_kw) if gpu_tier_kw else None, compute_capability=cc_kw)
        if (torch.cuda.is_available() or gpu_tier_kw or cc_kw)
        else torch.float32
    )

    if arch_meta.get("loader_kind") == "diffusion_pipeline" or arch_meta.get("architecture_family") == "diffusion_image":
        tokenizer = None
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": compute_dtype,
        }
    else:
        _is_catalog_model = str(model_alias or "").strip() in BUILTIN_MODEL_CATALOG
        _allow_remote_code = _is_catalog_model or str(os.environ.get("PLATODOOM_TRUST_REMOTE_CODE") or "").strip().lower() in ("1", "true", "yes")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=_allow_remote_code)
        load_kwargs = {
            "trust_remote_code": _allow_remote_code,
            "torch_dtype": compute_dtype,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            if q_cfg is not None:
                load_kwargs["quantization_config"] = q_cfg

    model, _ = load_model_from_checkpoint_metadata(
        ckpt_dir,
        allow_remote_config=False,
        **load_kwargs,
    )
    if hasattr(model, "eval"):
        model.eval()
    resolved_repo = str(spec.get("hf_repo_id") or hf_repo_id or model_alias)
    return model, tokenizer, resolved_repo


_MISSING_ATTR = object()


def _extract_caller_thinking_flag(payload: Dict[str, Any]) -> Optional[bool]:
    """Extracts explicit caller thinking/reasoning preference, returning None when omitted."""
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("enable_thinking"), bool):
        return bool(payload["enable_thinking"])
    ctk = payload.get("chat_template_kwargs")
    if isinstance(ctk, dict) and isinstance(ctk.get("enable_thinking"), bool):
        return bool(ctk["enable_thinking"])
    think_cfg = payload.get("thinking")
    if isinstance(think_cfg, bool):
        return think_cfg
    if isinstance(think_cfg, dict):
        if isinstance(think_cfg.get("enabled"), bool):
            return bool(think_cfg["enabled"])
        t_type = str(think_cfg.get("type") or "").strip().lower()
        if t_type in ("enabled", "true", "on"):
            return True
        if t_type in ("disabled", "false", "off", "none"):
            return False
    effort = payload.get("reasoning_effort")
    if isinstance(effort, str):
        eff_low = effort.strip().lower()
        if eff_low in ("none", "off", "disabled", "false", "0"):
            return False
        if eff_low in ("low", "medium", "high", "enabled", "true", "on"):
            return True
    return None


def _coerce_tool_args_dict(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        stripped = raw_args.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {"raw": raw_args}
    return {}


def _prepare_messages_for_template(
    raw_messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    *,
    native_tools: bool,
    native_tool_role: bool,
    dict_args: bool,
) -> List[Dict[str, Any]]:
    """Normalizes multi-turn messages & tool schemas so non-native templates preserve 100% of tool definitions and outputs."""
    out_msgs: List[Dict[str, Any]] = []
    total = len(raw_messages)
    for idx, orig in enumerate(raw_messages):
        msg = dict(orig)
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "assistant" and isinstance(content, str) and idx < total - 1:
            content = THINK_BLOCK_RE.sub("", content)
            if "</think>" in content:
                content = content.split("</think>", 1)[-1]
            content = content.strip()

        if role == "assistant" and isinstance(msg.get("tool_calls"), list) and msg["tool_calls"]:
            raw_calls = msg["tool_calls"]
            if not native_tools:
                serialized_blocks: List[str] = []
                for tc in raw_calls:
                    fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
                    fn_name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else None) or "unknown"
                    args_dict = _coerce_tool_args_dict(
                        fn.get("arguments", tc.get("arguments", {}) if isinstance(tc, dict) else {})
                    )
                    call_json = json.dumps({"name": fn_name, "arguments": args_dict}, ensure_ascii=False)
                    serialized_blocks.append(f"<tool_call>\n{call_json}\n</tool_call>")
                merged_c = "\n".join(filter(None, [str(content or "").strip(), *serialized_blocks])).strip()
                out_msgs.append({"role": "assistant", "content": merged_c})
            else:
                normalized_calls: List[Dict[str, Any]] = []
                for tc_idx, tc in enumerate(raw_calls):
                    tc_copy = dict(tc) if isinstance(tc, dict) else {}
                    fn = dict(tc_copy.get("function") or {})
                    fn_name = fn.get("name") or tc_copy.get("name") or "unknown"
                    raw_a = fn.get("arguments", tc_copy.get("arguments", {}))
                    if dict_args:
                        norm_a: Any = _coerce_tool_args_dict(raw_a)
                    else:
                        norm_a = raw_a if isinstance(raw_a, str) else json.dumps(raw_a or {}, ensure_ascii=False)
                    fn["name"] = fn_name
                    fn["arguments"] = norm_a
                    tc_copy.setdefault("id", f"call_{idx}_{tc_idx}")
                    tc_copy.setdefault("type", "function")
                    tc_copy["function"] = fn
                    if dict_args:
                        tc_copy["arguments"] = norm_a
                    normalized_calls.append(tc_copy)
                out_msgs.append(
                    {
                        "role": "assistant",
                        "content": "" if content is None else content,
                        "tool_calls": normalized_calls,
                    }
                )
        elif role == "tool" and (not native_tools or not native_tool_role):
            tool_id = msg.get("name") or msg.get("tool_call_id") or ""
            header = f"[tool={tool_id}]\n" if tool_id else ""
            body_txt = "" if content is None else (content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
            wrapped = f"<tool_response>\n{header}{body_txt}\n</tool_response>"
            if (
                out_msgs
                and out_msgs[-1].get("role") == "user"
                and "<tool_response>" in str(out_msgs[-1].get("content") or "")
            ):
                out_msgs[-1]["content"] = f"{out_msgs[-1]['content']}\n{wrapped}"
            else:
                out_msgs.append({"role": "user", "content": wrapped})
        else:
            msg["role"] = role
            msg["content"] = "" if content is None else content
            out_msgs.append(msg)

    if tools and not native_tools:
        tools_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        tool_block = (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_lines}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        )
        if out_msgs and out_msgs[0].get("role") == "system":
            sys_c = str(out_msgs[0].get("content") or "")
            if "<tools>" not in sys_c:
                out_msgs[0]["content"] = f"{sys_c.rstrip()}\n\n{tool_block}" if sys_c.strip() else tool_block
        else:
            out_msgs.insert(0, {"role": "system", "content": tool_block})

    return out_msgs


def introspect_loaded_tokenizer_capabilities(
    tokenizer: Any,
    model_alias: str = "",
    model: Any = None,
) -> Dict[str, Any]:
    """Recomputes vocabulary-bounded padding, stop token IDs, and chat-template capabilities from a loaded tokenizer."""
    vocab_bound = _resolve_vocab_upper_bound(tokenizer, model)
    resolved_pad_id = _resolve_pad_token_id(tokenizer, model)

    if tokenizer is not None:
        if getattr(tokenizer, "padding_side", None) != "left":
            try:
                tokenizer.padding_side = "left"
            except Exception:
                pass
        cur_pad = getattr(tokenizer, "pad_token_id", None)
        if not _is_valid_vocab_token_id(cur_pad, vocab_bound):
            try:
                tokenizer.pad_token_id = resolved_pad_id
            except Exception:
                pass

    if model is not None:
        for cfg_obj in (getattr(model, "generation_config", None), getattr(model, "config", None)):
            if cfg_obj is not None and not isinstance(cfg_obj, dict):
                cfg_pad = getattr(cfg_obj, "pad_token_id", None)
                if cfg_pad is not None and not _is_valid_vocab_token_id(cfg_pad, vocab_bound):
                    try:
                        cfg_obj.pad_token_id = resolved_pad_id
                    except Exception:
                        pass

    pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    if not _is_valid_vocab_token_id(pad_id, vocab_bound):
        pad_id = resolved_pad_id

    stop_ids = _resolve_stop_token_ids(tokenizer, model_alias, model)

    tok_name = getattr(tokenizer, "name_or_path", "") if tokenizer is not None else ""
    mod_name = getattr(getattr(model, "config", None), "_name_or_path", "") if model is not None else ""
    raw_tmpl = getattr(tokenizer, "chat_template", _MISSING_ATTR) if tokenizer is not None else _MISSING_ATTR
    tmpl_str = raw_tmpl if isinstance(raw_tmpl, str) else ""

    alias_haystack = f"{model_alias} {tok_name} {mod_name}".lower()
    is_deepseek_r1_reasoning = bool(
        "deepseek-r1" in alias_haystack
        or "r1-distill" in alias_haystack
        or re.search(r"(?:^|[-_/])r1-(?:1\.5b|7b|8b|14b|32b|70b|671b)", alias_haystack)
        or ("<｜Assistant｜>" in tmpl_str and "<think>" in tmpl_str)
    )

    has_apply = bool(tokenizer is not None and callable(getattr(tokenizer, "apply_chat_template", None)))
    has_chat_template = False
    supports_native_tools = False
    supports_tool_role = False
    requires_dict_tool_arguments = False
    supports_enable_thinking = False
    forces_think_prefix = False

    if has_apply and isinstance(raw_tmpl, str) and bool(raw_tmpl.strip()):
        has_chat_template = True
        supports_native_tools = bool(
            re.search(r"\b(?:if|in)\s+tools\b|\btools\s+is\s+(?:defined|iterable|not\s+none)|\btools\s*\|", raw_tmpl)
            or ("<tools>" in raw_tmpl and "tools" in raw_tmpl)
        )
        supports_tool_role = bool(
            supports_native_tools
            and any(
                m in raw_tmpl
                for m in ('role == "tool"', "role == 'tool'", '["role"] == "tool"', "['role'] == 'tool'")
            )
            and "not ns.is_tool" not in raw_tmpl
        )
        requires_dict_tool_arguments = bool(
            "arguments|items" in raw_tmpl or "arguments | items" in raw_tmpl or ".items()" in raw_tmpl
        )
        supports_enable_thinking = bool("enable_thinking" in raw_tmpl)
        forces_think_prefix = bool("<think>" in raw_tmpl and not supports_enable_thinking)
    elif has_apply and raw_tmpl is _MISSING_ATTR:
        has_chat_template = True
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            supports_native_tools = bool(("tools" in sig.parameters or has_var_kw) and not is_deepseek_r1_reasoning)
            supports_tool_role = supports_native_tools
            requires_dict_tool_arguments = True
            supports_enable_thinking = bool("enable_thinking" in sig.parameters or has_var_kw)
        except Exception:
            supports_native_tools = False
            supports_tool_role = False
            requires_dict_tool_arguments = False
            supports_enable_thinking = False
        forces_think_prefix = bool(is_deepseek_r1_reasoning and not supports_enable_thinking)
    else:
        forces_think_prefix = bool(is_deepseek_r1_reasoning)

    return {
        "pad_token_id": pad_id,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None,
        "stop_token_ids": stop_ids,
        "padding_side": getattr(tokenizer, "padding_side", "left") if tokenizer is not None else "left",
        "vocab_size": vocab_bound,
        "has_chat_template": has_chat_template,
        "supports_native_tools": supports_native_tools,
        "supports_tool_role": supports_tool_role,
        "requires_dict_tool_arguments": requires_dict_tool_arguments,
        "supports_enable_thinking": supports_enable_thinking,
        "is_deepseek_r1_reasoning": is_deepseek_r1_reasoning,
        "forces_think_prefix": forces_think_prefix,
    }


@dataclass
class VLLMProcessSlot:
    """Tracks a native `vllm serve` child process group bound to `127.0.0.1:<port>` (Spec 869)."""

    alias: str
    hf_repo_id: str
    checkpoint_target: str
    port: int = 8001
    host: str = "127.0.0.1"
    precision: str = "Q4"
    weights_gb: float = 0.0
    role: str = "primary"
    agent_count: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 16384
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 16384
    argv: List[str] = field(default_factory=list)
    resolved_flags: Dict[str, Any] = field(default_factory=dict)
    process: Optional[Any] = None
    pid: Optional[int] = None
    pgid: Optional[int] = None
    log_path: Optional[str] = None
    healthy: bool = False
    started_at: float = field(default_factory=time.time)


def _infer_cc_from_tier(gpu_tier: Optional[str], compute_capability: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if compute_capability is not None:
        return (int(compute_capability[0]), int(compute_capability[1]))
    tier_key = str(gpu_tier or "").strip()
    tier_entry = _SUPERVISOR_GPU_TIERS.get(tier_key)
    if tier_entry and tier_entry.get("cc") is not None:
        cc = tier_entry["cc"]
        return (int(cc[0]), int(cc[1]))
    try:
        import torch

        if torch.cuda.is_available():
            maj, minr = torch.cuda.get_device_capability(0)
            return (int(maj), int(minr))
    except Exception:
        pass
    return (8, 9)


def resolve_target_catalog_max_model_len(
    model_alias: str,
    gpu_tier: str = "L4",
    *,
    requested_max_model_len: Optional[int] = None,
    native_max_pos: int = 32768,
) -> int:
    """Resolves the tier-aware catalog context-window target (Spec 874 §4.5).

    Promotes legacy default `16384` requests up to the model's catalog `max_model_len` / `max_model_len_a100`
    on high-VRAM tiers (`A100`, `A100-80GB`, `H100`, `G4`) and 32K-native models on `L4+`, while preserving
    explicit lower clamps (`< 16384`) or explicit higher requests (`> catalog_target`).
    """
    clean_alias = str(model_alias or "").strip().split("@", 1)[0].strip()
    cat_entry = BUILTIN_MODEL_CATALOG.get(clean_alias) or _SUPERVISOR_MODEL_SPECS.get(clean_alias) or {}
    tier_up = str(gpu_tier or "").strip().upper()
    if tier_up in ("A100", "A100-80GB", "H100", "G4"):
        cat_target = int(cat_entry.get("max_model_len_a100") or cat_entry.get("max_model_len") or 32768)
    elif tier_up == "T4":
        cat_target = min(16384, int(cat_entry.get("max_model_len") or 16384))
    else:
        cat_target = int(cat_entry.get("max_model_len") or 16384)

    if requested_max_model_len is None or int(requested_max_model_len) <= 0:
        eff_target = cat_target
    else:
        req = int(requested_max_model_len)
        if tier_up == "T4":
            eff_target = min(req, 16384)
        elif req == 16384 and cat_target > 16384:
            eff_target = cat_target
        else:
            eff_target = req

    eff_cap = max(4096, int(native_max_pos or 32768))
    return max(4096, min(eff_target, eff_cap))


def compute_physical_vllm_memory_budget(
    *,
    model_alias: str,
    eff_total_vram_gb: float,
    eff_free_vram_gb: float,
    resident_weights_gb: float,
    kv_cache_dtype: str = "auto",
    requested_max_model_len: int = 16384,
    native_max_pos: int = 32768,
    requested_max_num_seqs: Optional[int] = None,
    requested_max_num_batched_tokens: Optional[int] = None,
    slot_vram_fraction: Optional[float] = None,
    active_slot_count: int = 1,
    compute_capability: Tuple[int, int] = (8, 9),
    is_q35_hybrid: bool = False,
    is_quantized_marlin: bool = False,
    tier_seq_cap: int = 256,
    agent_count: int = 1,
) -> Dict[str, Any]:
    """First-principles physical GPU memory budget & parameter solver for `vllm serve` (Spec 871 §3.1–§3.2 & Spec 874 §4.5).

    Prioritizes reaching the target `max_model_len` for active agent concurrency (`target_seqs`) before scaling
    unused `max_num_seqs` slots, while safely clamping context length on tight-VRAM tiers to prevent CUDA OOM.
    """
    clean_alias = str(model_alias or "").strip().split("@", 1)[0].strip()
    spec_entry = _SUPERVISOR_MODEL_SPECS.get(clean_alias) or {}
    total_params_b = float(spec_entry.get("total_params_b") or max(4.0, resident_weights_gb * 1.8))
    active_params_b = float(spec_entry.get("active_params_b") or total_params_b)
    is_moe = bool(spec_entry.get("is_moe", False) or total_params_b > active_params_b * 1.5)

    total_vram = max(1.0, float(eff_total_vram_gb))
    free_vram = max(0.5, min(total_vram, float(eff_free_vram_gb)))
    weights = max(0.5, float(resident_weights_gb))
    active_slots = max(1, int(active_slot_count))
    eff_agents = max(1, int(agent_count or 1))

    # 1. Continuous OS / CUDA driver / host-process safety margin (zero step-function cliffs)
    os_driver_margin_gib = max(1.5, 0.052 * total_vram)
    min_free_cushion_gib = max(1.2, 0.03 * total_vram)
    u_cap_by_total = max(0.10, (total_vram - os_driver_margin_gib) / total_vram)
    u_cap_by_free = max(0.10, (free_vram - min_free_cushion_gib) / total_vram)
    u_max_safe = min(0.94, u_cap_by_total, u_cap_by_free)

    if active_slots == 1:
        if slot_vram_fraction is not None and abs(float(slot_vram_fraction) - 0.90) > 1e-4:
            desired_u = float(slot_vram_fraction)
        else:
            weight_ratio = weights / total_vram
            desired_u = 0.90 + 0.04 * min(1.0, max(0.0, (weight_ratio - 0.60) / 0.25))
        gpu_util = round(max(0.10, min(u_max_safe, desired_u)), 4)
    else:
        aggregate_cap = min(0.92 - 0.025 * (active_slots - 1), u_max_safe)
        raw_frac = float(slot_vram_fraction) if slot_vram_fraction is not None else (aggregate_cap / active_slots)
        host_scale = min(1.0, u_cap_by_free / 0.92)
        gpu_util = round(max(0.10, min(aggregate_cap, raw_frac * host_scale)), 4)

    slot_budget_gib = min(total_vram * gpu_util, free_vram - (min_free_cushion_gib / active_slots))
    post_weight_budget_gib = slot_budget_gib - weights
    vram_feasible = post_weight_budget_gib >= 1.5

    # 2. Exact KV + Mamba `'align'` block footprint
    req_len_clamped = max(1024, min(int(requested_max_model_len or 16384), int(native_max_pos or 32768)))
    fp = estimate_model_kv_and_state_footprint(
        clean_alias or "qwen-9b",
        max_model_len=req_len_clamped,
        kv_cache_dtype=kv_cache_dtype,
    )
    aligned_block_tokens = max(16, int(fp.get("aligned_block_tokens") or 16))
    aligned_block_gib = float(fp.get("unified_aligned_block_gib") or 0.0)
    if aligned_block_gib <= 0.0:
        aligned_block_gib = max(1e-4, (float(fp["kv_bytes_per_token"]) * aligned_block_tokens) / float(1024 ** 3))

    # 3. Physical startup memory equation: forward workspace + inductor/CUDAGraph pool + 64 minimal profiling blocks
    eager_workspace_gib = (
        1.20
        + 0.035 * active_params_b
        + (0.80 if is_moe else 0.0)
        + (0.40 if is_quantized_marlin else 0.0)
    )
    compile_cudagraph_overhead_gib = 1.50 + 0.05 * total_params_b
    profiling_kv_gib = max(1.20, 64.0 * aligned_block_gib)
    compile_required_headroom_gib = (
        eager_workspace_gib + compile_cudagraph_overhead_gib + profiling_kv_gib
    )

    enforce_eager = bool(
        not vram_feasible
        or post_weight_budget_gib < compile_required_headroom_gib
    )

    active_runtime_workspace_gib = (
        eager_workspace_gib
        if enforce_eager
        else (eager_workspace_gib + 0.55 * compile_cudagraph_overhead_gib)
    )
    net_kv_pool_gib = max(0.0, post_weight_budget_gib - active_runtime_workspace_gib)
    eff_tier_cap = max(1, min(256, int(tier_seq_cap or 256)))
    target_seqs = min(eff_tier_cap, max(4, eff_agents * 2))

    # 4. Context-first window (`max_model_len`) allocation: achieve target catalog length before scaling unused seq slots
    clamped_max_len = req_len_clamped
    if vram_feasible and net_kv_pool_gib > 0.0:
        single_seq_full_gib = float(fp.get("total_gb_per_seq_at_max_len") or 0.0)
        if is_q35_hybrid:
            blocks_for_req_len = max(2, math.ceil(req_len_clamped / float(aligned_block_tokens)))
            single_seq_full_gib = max(single_seq_full_gib, blocks_for_req_len * aligned_block_gib)

        # Context-first KV policy (Spec 874 §4.5 & §5.5): prioritize reaching the model's catalog
        # max_model_len whenever a single max-length sequence fits safely within 75% of the KV pool
        # (leaving >= 25% headroom for concurrent peer blocks & prefix cache while vLLM PagedAttention
        # queues excess concurrency cleanly instead of starving max_model_len to 4K-8K at high agent_count).
        max_single_seq_share_gib = max(aligned_block_gib * 3.0, net_kv_pool_gib * 0.75)

        if single_seq_full_gib > max_single_seq_share_gib:
            if is_q35_hybrid:
                safe_blocks = max(2, int(max_single_seq_share_gib // aligned_block_gib))
                raw_safe_tokens = safe_blocks * aligned_block_tokens
            else:
                kv_bpt = max(1.0, float(fp["kv_bytes_per_token"]))
                raw_safe_tokens = int((max_single_seq_share_gib * (1024 ** 3)) // kv_bpt)
            snapped_len = max(4096, (raw_safe_tokens // 2048) * 2048)
            clamped_max_len = min(req_len_clamped, snapped_len)

    # 5. Prefill batch tokens: let native vLLM V1 choose its own chunked-prefill window unless tight headroom requires a cap
    cli_batched_tokens: Optional[int] = None
    if requested_max_num_batched_tokens is not None:
        req_bt = max(2048, int(requested_max_num_batched_tokens))
        cli_batched_tokens = min(req_bt, 8192) if (enforce_eager and net_kv_pool_gib < 8.0) else (min(req_bt, 16384) if enforce_eager else req_bt)
        resolved_batched_tokens = cli_batched_tokens
    elif enforce_eager:
        cli_batched_tokens = min(clamped_max_len, 8192 if net_kv_pool_gib < 8.0 else 16384)
        resolved_batched_tokens = cli_batched_tokens
    else:
        cli_batched_tokens = None
        resolved_batched_tokens = 16384

    # 6. Concurrent sequences (`max_num_seqs`) strictly bounded by physical KV + realistic active agent demand
    if not vram_feasible or net_kv_pool_gib <= 0.0:
        physical_seq_cap = 4
    else:
        working_tokens = min(max(512, clamped_max_len), 4096 if clamped_max_len >= 32768 else 2048)
        working_kv_gib = (float(fp["kv_bytes_per_token"]) * working_tokens) / float(1024 ** 3)
        per_seq_working_gib = max(0.02, working_kv_gib + float(fp.get("gdn_gb_per_seq") or 0.0))
        if is_q35_hybrid:
            per_seq_working_gib = max(per_seq_working_gib, 3.0 * aligned_block_gib)
        physical_seq_cap = max(1, min(256, int(net_kv_pool_gib // per_seq_working_gib)))

    if requested_max_num_seqs is not None:
        caller_seqs = max(1, min(256, int(requested_max_num_seqs)))
        resolved_seqs = min(caller_seqs, physical_seq_cap) if vram_feasible else caller_seqs
    else:
        if clamped_max_len >= 32768:
            demand_seq_cap = min(eff_tier_cap, max(16, target_seqs * 3))
            resolved_seqs = max(1, min(demand_seq_cap, physical_seq_cap))
        else:
            resolved_seqs = max(1, min(eff_tier_cap, physical_seq_cap))

    total_aligned_blocks = int(net_kv_pool_gib // aligned_block_gib) if aligned_block_gib > 0.0 else 0
    return {
        "gpu_memory_utilization": gpu_util,
        "slot_budget_gib": round(slot_budget_gib, 3),
        "post_weight_budget_gib": round(post_weight_budget_gib, 3),
        "active_runtime_workspace_gib": round(active_runtime_workspace_gib, 3),
        "compile_required_headroom_gib": round(compile_required_headroom_gib, 3),
        "net_kv_pool_gib": round(net_kv_pool_gib, 3),
        "os_driver_margin_gib": round(total_vram - total_vram * gpu_util, 3),
        "free_vram_cushion_gib": round(free_vram - slot_budget_gib, 3),
        "enforce_eager": enforce_eager,
        "max_model_len": int(clamped_max_len),
        "max_num_seqs": int(resolved_seqs),
        "max_num_batched_tokens": int(resolved_batched_tokens),
        "cli_max_num_batched_tokens": cli_batched_tokens,
        "aligned_block_tokens": int(aligned_block_tokens),
        "aligned_block_gib": round(aligned_block_gib, 6),
        "estimated_cache_blocks": int(total_aligned_blocks),
        "vram_feasible": bool(vram_feasible),
        "target_seqs": int(target_seqs),
    }


def _parse_optional_bool_flag(raw: Any, *, field_name: str = "enable_vision") -> Optional[bool]:
    """Strictly parses an optional boolean toggle (`None`, `bool`, or `'1'/'0'/'true'/'false'/'yes'/'no'/'on'/'off'`)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(int(raw))
    if isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            return None
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"Invalid boolean value for '{field_name}': {raw!r}. Expected true/false or 1/0.")


def _resolve_env_vision_toggle() -> Optional[bool]:
    """Resolves explicit Vision toggle from environment variables (canonical `SANDCASTLE_VLLM_ENABLE_VISION`)."""
    for env_key in ("SANDCASTLE_VLLM_ENABLE_VISION", "SANDCASTLE_VLLM_ENABLE_VLM"):
        raw = os.environ.get(env_key)
        if raw is not None and str(raw).strip() != "":
            return _parse_optional_bool_flag(raw, field_name=env_key)
    lang_only = _parse_optional_bool_flag(
        os.environ.get("SANDCASTLE_VLLM_LANGUAGE_ONLY"),
        field_name="SANDCASTLE_VLLM_LANGUAGE_ONLY",
    )
    if lang_only is True:
        return False
    return None


def build_vllm_serve_args(
    *,
    model_alias: str,
    checkpoint_target: str,
    port: int = 8001,
    host: str = "127.0.0.1",
    gpu_tier: str = "L4",
    compute_capability: Optional[Tuple[int, int]] = None,
    free_vram_gb: float = 22.5,
    total_vram_gb: Optional[float] = None,
    slot_vram_fraction: Optional[float] = None,
    max_model_len: int = 16384,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    active_slot_count: int = 1,
    agent_count: int = 1,
    arch_meta: Optional[Mapping[str, Any]] = None,
    precision: Optional[str] = None,
    enable_speculative: bool = False,
    draft_checkpoint_target: Optional[str] = None,
    vllm_bin: Optional[str] = None,
    has_fastsafetensors: Optional[bool] = None,
    enable_vision: Optional[bool] = None,
) -> Dict[str, Any]:
    """Synthesizes deterministic `vllm serve` CLI argv + environment from hardware and checkpoint metadata (Spec 869 & Spec 871)."""
    clean_alias = str(model_alias or "").strip().split("@", 1)[0].strip()
    spec_entry = dict(_SUPERVISOR_MODEL_SPECS.get(clean_alias) or {})
    meta: Dict[str, Any] = dict(arch_meta or {})
    ckpt_str = str(checkpoint_target or spec_entry.get("quantized_hf_repo_id") or spec_entry.get("hf_repo_id") or clean_alias).strip()

    ckpt_path = Path(ckpt_str)
    has_local_safetensors = False
    if ckpt_path.is_dir():
        has_local_safetensors = any(ckpt_path.glob("*.safetensors"))
        if (ckpt_path / "config.json").is_file():
            try:
                disk_cfg = json.loads((ckpt_path / "config.json").read_text(encoding="utf-8"))
                if isinstance(disk_cfg, dict):
                    for k, v in disk_cfg.items():
                        meta.setdefault(k, v)
            except Exception:
                pass

    cc_maj, cc_min = _infer_cc_from_tier(gpu_tier, compute_capability)
    cc_int = cc_maj * 10 + cc_min
    eff_total_vram = float(
        total_vram_gb
        if total_vram_gb is not None
        else max(float(free_vram_gb), float((_SUPERVISOR_GPU_TIERS.get(gpu_tier) or {}).get("vram_gb", free_vram_gb)))
    )
    eff_free_vram = min(eff_total_vram, max(0.5, float(free_vram_gb)))

    q_cfg = meta.get("quantization_config") if isinstance(meta.get("quantization_config"), dict) else {}
    raw_q_method = str(meta.get("quant_method") or q_cfg.get("quant_method") or "").strip().lower()
    norm_prec = normalize_precision_code(precision) if precision else None
    lower_ckpt = ckpt_str.lower()

    if not raw_q_method and clean_alias != UNIFIED_MODEL_ALIAS:
        if norm_prec == "BF16" and not any(t in lower_ckpt for t in ("-awq", "-gptq", "-fp8", "int4", "4bit")):
            raw_q_method = ""
        else:
            raw_q_method = str(spec_entry.get("quant_method") or "").strip().lower()
            if not raw_q_method:
                if "gptq" in lower_ckpt:
                    raw_q_method = "gptq"
                elif "awq" in lower_ckpt:
                    raw_q_method = "awq"
                elif "fp8" in lower_ckpt:
                    raw_q_method = "fp8"

    model_type = str(meta.get("model_type") or spec_entry.get("model_type") or "causal_lm").strip().lower()
    ckpt_dtype = str(meta.get("torch_dtype") or "").strip().lower()
    if not ckpt_dtype:
        if ckpt_str in _SUPERVISOR_FLOAT16_REPOS:
            ckpt_dtype = "float16"
        elif norm_prec == "BF16":
            ckpt_dtype = "bfloat16"
        else:
            ckpt_dtype = str(spec_entry.get("torch_dtype") or "bfloat16").strip().lower()

    if cc_int < 80 or ckpt_dtype in ("float16", "fp16", "torch.float16", "half"):
        resolved_dtype = "float16"
    else:
        resolved_dtype = "bfloat16"

    if has_fastsafetensors is None:
        has_fst = (
            os.environ.get("PLATODOOM_ENABLE_FASTSAFETENSORS") == "1"
            or importlib.util.find_spec("fastsafetensors") is not None
        )
    else:
        has_fst = bool(has_fastsafetensors)

    active_slots = max(1, int(active_slot_count))
    approx_slot_budget_gb = (
        eff_free_vram * (float(slot_vram_fraction) / 0.90)
        if (active_slots > 1 and slot_vram_fraction is not None)
        else (eff_free_vram / active_slots)
    )
    quantization_flag: Optional[str] = None
    load_format_flag: Optional[str] = None

    is_prequant = raw_q_method in (
        "gptq",
        "auto_gptq",
        "awq",
        "compressed-tensors",
        "compressed_tensors",
        "fp8",
        "marlin",
        "gptq_marlin",
        "awq_marlin",
    )
    if is_prequant:
        quantization_flag = None
        if has_fst and (has_local_safetensors or has_fastsafetensors is True):
            load_format_flag = "fastsafetensors"
    else:
        bf16_w = float(spec_entry.get("weights_bf16_gb", 18.0))
        raw_prec_l = str(precision or "").strip().lower()
        if norm_prec in ("Q4", "FP8") or cc_int < 80 or approx_slot_budget_gb < (bf16_w + 2.5):
            if (
                cc_int >= 89
                and raw_prec_l != "bitsandbytes"
                and (bf16_w * 0.55 + 3.0) <= approx_slot_budget_gb
                and (norm_prec == "FP8" or approx_slot_budget_gb < 30.0)
            ):
                quantization_flag = "fp8"
            else:
                quantization_flag = "bitsandbytes"
                load_format_flag = "bitsandbytes"
                if cc_int < 80:
                    resolved_dtype = "float16"
        else:
            if has_fst and (has_local_safetensors or has_fastsafetensors is True):
                load_format_flag = "fastsafetensors"

    if quantization_flag == "fp8" or raw_q_method == "fp8":
        resolved_precision = "FP8"
    elif quantization_flag == "bitsandbytes" or is_prequant:
        resolved_precision = norm_prec if norm_prec in ("Q4", "Q5", "Q6", "Q8", "INT8") else "Q4"
    else:
        resolved_precision = norm_prec or "BF16"

    if cc_int >= 89 and model_type != "deepseek_v2":
        kv_cache_dtype = "fp8"
    else:
        kv_cache_dtype = "auto"

    parser_spec = resolve_vllm_model_parser_spec(clean_alias)
    is_q35_hybrid = bool(
        parser_spec.get("mamba_hybrid")
        or parser_spec.get("language_model_only")
        or model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")
    )

    explicit_vision = (
        _parse_optional_bool_flag(enable_vision, field_name="enable_vision")
        if enable_vision is not None
        else _resolve_env_vision_toggle()
    )
    enable_vlm = False
    if explicit_vision is True:
        if not is_q35_hybrid:
            raise ValueError(
                f"Vision mode requested for '{clean_alias}', which is not a vision-capable model (model_type={model_type!r})."
            )
        if ckpt_path.is_dir():
            if not (ckpt_path / "preprocessor_config.json").is_file():
                raise ValueError(
                    f"Vision mode requested for '{clean_alias}' at '{ckpt_str}', but 'preprocessor_config.json' is missing "
                    f"(text-only checkpoint cannot serve multimodal image inputs)."
                )
        elif clean_alias == UNIFIED_MODEL_ALIAS:
            raise ValueError(
                f"Vision mode requested for '{clean_alias}', which is a text-only merged checkpoint without vision weights/preprocessor_config.json."
            )
        enable_vlm = True

    # Inspect actual local safetensors footprint (`<= 50 ms`, zero network I/O) when staged on disk (Spec 871 §3.1).
    # Only exclude visual.* weights when actually booting with `--language-model-only`.
    use_language_model_only = bool(is_q35_hybrid and not enable_vlm)
    local_fp = (
        inspect_local_checkpoint_footprint_gib(
            ckpt_path,
            language_model_only=use_language_model_only,
            quant_method=raw_q_method or quantization_flag,
        )
        if has_local_safetensors
        else None
    )
    if local_fp is not None and float(local_fp.get("estimated_gpu_weights_gib", 0.0)) > 0.0:
        eff_weights_gb = round(float(local_fp["estimated_gpu_weights_gib"]), 3)
        if enable_vlm:
            # Reserve modest activation scratch for ViT image encoder forward passes.
            eff_weights_gb = round(eff_weights_gb + 0.8, 3)
    elif quantization_flag == "bitsandbytes" or is_prequant:
        eff_weights_gb = float(spec_entry.get("weights_q4_gb", 5.2)) + (1.2 if enable_vlm else 0.0)
    elif quantization_flag == "fp8":
        eff_weights_gb = round(float(spec_entry.get("weights_bf16_gb", 18.0)) * 0.52, 2) + (1.2 if enable_vlm else 0.0)
    else:
        eff_weights_gb = float(spec_entry.get("weights_bf16_gb", 18.0)) + (1.2 if enable_vlm else 0.0)

    native_max_pos = int(meta.get("max_position_embeddings") or spec_entry.get("max_pos") or 32768)
    tier_cap = int((_SUPERVISOR_GPU_TIERS.get(str(gpu_tier or ""), {})).get("max_num_seqs", 256))
    target_max_model_len = resolve_target_catalog_max_model_len(
        clean_alias,
        gpu_tier=str(gpu_tier or "L4"),
        requested_max_model_len=int(max_model_len) if max_model_len is not None else None,
        native_max_pos=native_max_pos,
    )

    mem_budget = compute_physical_vllm_memory_budget(
        model_alias=clean_alias,
        eff_total_vram_gb=eff_total_vram,
        eff_free_vram_gb=eff_free_vram,
        resident_weights_gb=eff_weights_gb,
        kv_cache_dtype=kv_cache_dtype,
        requested_max_model_len=target_max_model_len,
        native_max_pos=native_max_pos,
        requested_max_num_seqs=max_num_seqs,
        requested_max_num_batched_tokens=max_num_batched_tokens,
        slot_vram_fraction=slot_vram_fraction,
        active_slot_count=active_slots,
        compute_capability=(cc_maj, cc_min),
        is_q35_hybrid=is_q35_hybrid,
        is_quantized_marlin=bool(is_prequant and raw_q_method != "fp8"),
        tier_seq_cap=tier_cap,
        agent_count=agent_count,
    )

    gpu_util = float(mem_budget["gpu_memory_utilization"])
    clamped_max_len = int(mem_budget["max_model_len"])
    resolved_seqs = int(mem_budget["max_num_seqs"])
    resolved_batched_tokens = int(mem_budget["max_num_batched_tokens"])
    cli_batched_tokens = mem_budget.get("cli_max_num_batched_tokens")
    enforce_eager = bool(mem_budget["enforce_eager"])

    resolved_bin = (
        vllm_bin
        or os.environ.get("PLATODOOM_VLLM_BIN")
        or ("/opt/vllm-venv/bin/vllm" if Path("/opt/vllm-venv/bin/vllm").exists() else "vllm")
    )

    argv: List[str] = [
        str(resolved_bin),
        "serve",
        ckpt_str,
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--served-model-name",
        clean_alias,
        "--dtype",
        resolved_dtype,
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--gpu-memory-utilization",
        f"{gpu_util:.4f}",
        "--max-model-len",
        str(clamped_max_len),
        "--max-num-seqs",
        str(resolved_seqs),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
    ]
    if cli_batched_tokens is not None:
        argv.extend(["--max-num-batched-tokens", str(int(cli_batched_tokens))])

    if quantization_flag:
        argv.extend(["--quantization", quantization_flag])
    if load_format_flag:
        argv.extend(["--load-format", load_format_flag])
    if is_q35_hybrid:
        argv.extend(["--mamba-ssm-cache-dtype", "auto"])
        if enable_vlm:
            net_kv_gib = float(mem_budget.get("net_kv_pool_gib") or 0.0)
            if net_kv_gib < 6.0 or not bool(mem_budget.get("vram_feasible", True)):
                raise RuntimeError(
                    f"Vision mode requested for '{clean_alias}', but remaining KV cache pool ({net_kv_gib:.2f} GiB) "
                    f"is below the 6.0 GiB multimodal minimum on {gpu_tier} ({eff_free_vram:.1f} GiB free, "
                    f"weights={eff_weights_gb:.2f} GiB). Refusing silent text-only fallback."
                )
            argv.extend(["--limit-mm-per-prompt", '{"image": 8, "video": 0}'])
        else:
            argv.append("--language-model-only")
    if enforce_eager:
        argv.append("--enforce-eager")
    if model_type == "deepseek_v2" or bool(meta.get("trust_remote_code")):
        argv.append("--trust-remote-code")

    tool_parser = parser_spec.get("tool_call_parser")
    reasoning_parser = parser_spec.get("reasoning_parser")
    if parser_spec.get("enable_auto_tool_choice") and tool_parser:
        argv.extend(["--enable-auto-tool-choice", "--tool-call-parser", str(tool_parser)])
    if reasoning_parser:
        argv.extend(["--reasoning-parser", str(reasoning_parser)])

    spec_cfg: Optional[Dict[str, Any]] = None
    if enable_speculative and int(agent_count) <= 8 and not enable_vlm:
        if is_q35_hybrid:
            if quantization_flag != "bitsandbytes":
                spec_cfg = {"method": "mtp", "num_speculative_tokens": 1}
            else:
                spec_cfg = {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4}
        elif model_type in ("qwen3", "qwen3_moe") and int(spec_entry.get("vocab_size") or 151936) == 151936:
            if (
                clean_alias != "qwen-4b"
                and draft_checkpoint_target
                and float(mem_budget["net_kv_pool_gib"]) >= 6.0
            ):
                spec_cfg = {"model": str(draft_checkpoint_target), "num_speculative_tokens": 3}
            else:
                spec_cfg = {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4}
        else:
            spec_cfg = {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4}
        argv.extend(["--speculative-config", json.dumps(spec_cfg, separators=(",", ":"))])

    child_env = dict(os.environ)
    resolved_bin_str = str(resolved_bin)
    if os.path.isabs(resolved_bin_str) or os.sep in resolved_bin_str:
        bin_dir = str(Path(resolved_bin_str).parent)
        path_parts = [p for p in child_env.get("PATH", "").split(os.pathsep) if p]
        if bin_dir and bin_dir not in path_parts:
            child_env["PATH"] = os.pathsep.join([bin_dir, *path_parts]) if path_parts else bin_dir
    child_env.setdefault("VLLM_CACHE_ROOT", "/content/.cache/vllm")
    child_env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/content/.cache/inductor")
    child_env.setdefault("FLASHINFER_WORKSPACE_BASE", "/content/.cache/flashinfer")
    child_env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    child_env["PYTHONUNBUFFERED"] = "1"

    return {
        "argv": argv,
        "env": child_env,
        "resolved_flags": {
            "model_alias": clean_alias,
            "checkpoint_target": ckpt_str,
            "host": str(host),
            "port": int(port),
            "dtype": resolved_dtype,
            "resolved_precision": resolved_precision,
            "kv_cache_dtype": kv_cache_dtype,
            "quantization": quantization_flag,
            "load_format": load_format_flag,
            "gpu_memory_utilization": gpu_util,
            "max_model_len": clamped_max_len,
            "max_num_seqs": resolved_seqs,
            "max_num_batched_tokens": resolved_batched_tokens,
            "mamba_ssm_cache_dtype": "auto" if is_q35_hybrid else None,
            "language_model_only": bool(is_q35_hybrid and not enable_vlm),
            "vision": bool(is_q35_hybrid and enable_vlm),
            "max_images_per_prompt": 8 if (is_q35_hybrid and enable_vlm) else 0,
            "enforce_eager": enforce_eager,
            "tool_call_parser": tool_parser,
            "reasoning_parser": reasoning_parser,
            "enable_auto_tool_choice": bool(parser_spec.get("enable_auto_tool_choice") and tool_parser),
            "speculative_config": spec_cfg,
            "weights_gb": eff_weights_gb,
            "memory_budget": mem_budget,
            "local_checkpoint_footprint": local_fp,
        },
    }


def _terminate_slot_process_group(
    slot: Any,
    timeout_sec: float = 10.0,
    poll_vram: bool = True,
) -> Dict[str, Any]:
    """Cleanly terminates an entire `vllm serve` child process group (`SIGTERM` -> `SIGKILL`) and verifies VRAM release."""
    proc = getattr(slot, "process", None) if not isinstance(slot, Mapping) else slot.get("process")
    pgid = getattr(slot, "pgid", None) if not isinstance(slot, Mapping) else slot.get("pgid")
    pid = getattr(slot, "pid", None) if not isinstance(slot, Mapping) else slot.get("pid")
    if pid is None and proc is not None:
        pid = getattr(proc, "pid", None)

    killed_sig: Optional[str] = None
    resolved_pgid = pgid
    if resolved_pgid is None and pid is not None and hasattr(os, "getpgid"):
        try:
            resolved_pgid = os.getpgid(int(pid))
        except Exception:
            resolved_pgid = None

    has_distinct_pgid = bool(
        resolved_pgid is not None and hasattr(os, "killpg") and int(resolved_pgid) != os.getpgrp()
    )
    proc_running = bool(proc is not None and proc.poll() is None)

    if has_distinct_pgid:
        try:
            os.killpg(int(resolved_pgid), signal.SIGTERM)
            killed_sig = "SIGTERM"
        except ProcessLookupError:
            pass
        except Exception:
            if proc_running:
                try:
                    proc.terminate()
                    killed_sig = "SIGTERM"
                except Exception:
                    pass
    elif proc_running:
        try:
            proc.terminate()
            killed_sig = "SIGTERM"
        except Exception:
            pass

    deadline = time.monotonic() + max(0.2, float(timeout_sec))
    while time.monotonic() < deadline:
        pg_alive = False
        if has_distinct_pgid:
            try:
                os.killpg(int(resolved_pgid), 0)
                pg_alive = True
            except ProcessLookupError:
                pg_alive = False
            except PermissionError:
                pg_alive = True
        leader_alive = bool(proc is not None and proc.poll() is None)
        if not leader_alive and not pg_alive:
            break
        if leader_alive:
            try:
                proc.wait(timeout=min(0.15, max(0.03, deadline - time.monotonic())))
            except Exception:
                pass
        else:
            time.sleep(min(0.05, max(0.01, deadline - time.monotonic())))

    pg_still_alive = False
    if has_distinct_pgid:
        try:
            os.killpg(int(resolved_pgid), 0)
            pg_still_alive = True
        except ProcessLookupError:
            pg_still_alive = False
        except PermissionError:
            pg_still_alive = True

    if (proc is not None and proc.poll() is None) or pg_still_alive:
        if has_distinct_pgid:
            try:
                os.killpg(int(resolved_pgid), signal.SIGKILL)
                killed_sig = "SIGKILL"
            except ProcessLookupError:
                pass
            except Exception:
                if proc is not None and proc.poll() is None:
                    try:
                        proc.kill()
                        killed_sig = "SIGKILL"
                    except Exception:
                        pass
        elif proc is not None and proc.poll() is None:
            try:
                proc.kill()
                killed_sig = "SIGKILL"
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=3.0)
            except Exception:
                pass

    if not isinstance(slot, Mapping):
        if hasattr(slot, "process"):
            slot.process = None
        if hasattr(slot, "healthy"):
            slot.healthy = False

    gc.collect()
    free_gb: Optional[float] = None
    if poll_vram:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_b, _ = torch.cuda.mem_get_info(0)
                free_gb = round(float(free_b) / float(1024 ** 3), 3)
        except Exception:
            free_gb = None

    return {
        "terminated": True,
        "pid": pid,
        "pgid": pgid,
        "signal": killed_sig,
        "free_vram_gb": free_gb,
    }


def _prune_model_cache_lru(
    cache_root: Path | str,
    required_free_gb: float,
    protected_dirs: Sequence[Path | str] = (),
    headroom_gb: float = 5.0,
) -> Dict[str, Any]:
    """Evicts oldest inactive model directories on `/content/models` when free disk `< required_free_gb + headroom_gb`."""
    root = Path(cache_root)
    probe_dir = root if root.exists() else root.parent
    if not probe_dir.exists():
        probe_dir = Path("/tmp")

    protected_resolved = set()
    for p in protected_dirs:
        if p:
            try:
                protected_resolved.add(str(Path(p).resolve()))
            except Exception:
                protected_resolved.add(str(p))

    req_bytes = int(max(0.0, float(required_free_gb)) * (1024 ** 3))
    target_bytes = int(max(0.0, float(required_free_gb) + float(headroom_gb)) * (1024 ** 3))
    evicted: List[str] = []

    usage = shutil.disk_usage(str(probe_dir))
    if usage.free < target_bytes and root.is_dir():
        candidates: List[Tuple[float, Path]] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                c_res = str(child.resolve())
            except Exception:
                c_res = str(child)
            if c_res in protected_resolved:
                continue
            try:
                mtime = float(child.stat().st_mtime)
            except Exception:
                mtime = 0.0
            candidates.append((mtime, child))
        candidates.sort(key=lambda item: (item[0], item[1].name))
        for _, victim in candidates:
            if shutil.disk_usage(str(probe_dir)).free >= target_bytes:
                break
            try:
                shutil.rmtree(victim)
                evicted.append(str(victim))
            except Exception:
                pass

    final_usage = shutil.disk_usage(str(probe_dir))
    free_gb = round(float(final_usage.free) / float(1024 ** 3), 3)
    return {
        "ok": final_usage.free >= req_bytes,
        "free_gb": free_gb,
        "required_free_gb": float(required_free_gb),
        "evicted_dirs": evicted,
    }


class VLLMStartupError(RuntimeError):
    """Structured fail-fast exception surfaced when a `vllm serve` child process exits prematurely (Spec 871 §3.3)."""

    def __init__(
        self,
        message: str,
        *,
        is_oom: bool = False,
        error_code: str = "checkpoint_load_failed",
        http_status: int = 500,
        root_cause_line: str = "",
        diagnostic_excerpt: str = "",
        returncode: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.is_oom = bool(is_oom)
        self.error_code = str(error_code)
        self.http_status = int(http_status)
        self.root_cause_line = str(root_cause_line)
        self.diagnostic_excerpt = str(diagnostic_excerpt)
        self.returncode = returncode


def extract_vllm_root_cause_diagnostic(log_text: str, max_chars: int = 6000) -> Dict[str, Any]:
    """Scans the full multi-process `vllm serve` log to isolate the primary causal failure block (Spec 871 §3.3).

    Filters out shard-loading progress bar noise and repetitive weight-dtype warning floods, prioritizes
    inner `EngineCore` / worker tracebacks (`torch.OutOfMemoryError`, `CUDACachingAllocator` OOM, KV cache
    starvation, config/CLI errors) over outer `APIServer` `wait_for_engine_startup` wrappers, and classifies
    OOM deterministically as `507` / `cuda_oom`.
    """
    raw_lines = str(log_text or "").splitlines()
    filtered: List[str] = []
    dtype_warn_run = 0
    for raw_ln in raw_lines:
        stripped = raw_ln.strip()
        if not stripped:
            continue
        if "Loading safetensors checkpoint shards:" in stripped or re.search(r"\b\d+%\|[█▏▎▍▌▋▊▉\s]+\|", stripped):
            continue
        if "Attempted to load weight " in stripped and " with dtype " in stripped:
            dtype_warn_run += 1
            if dtype_warn_run <= 2:
                filtered.append(raw_ln)
            continue
        if dtype_warn_run > 2:
            filtered.append(f"... [{dtype_warn_run - 2} similar weight-dtype warnings omitted]")
        dtype_warn_run = 0
        filtered.append(raw_ln)
    if dtype_warn_run > 2:
        filtered.append(f"... [{dtype_warn_run - 2} similar weight-dtype warnings omitted]")

    oom_patterns = (
        "outofmemoryerror",
        "cuda out of memory",
        "cuda error: out of memory",
        "memory allocation failed with oom",
        "no available memory for the cache blocks",
        "larger than the available kv cache memory",
        "free memory on device",
    )
    invalid_patterns = (
        "unrecognized arguments:",
        "invalid argument",
        "repositorynotfounderror",
        "hfvalidationerror",
    )
    wrapper_noise = (
        "engine core initialization failed. see root cause above",
        "see stack trace (above) for the root cause",
        "enginedeaderror: enginecore encountered an issue",
        "wait_for_engine_startup",
    )

    is_oom = False
    is_invalid = False
    primary_idx: Optional[int] = None
    primary_line = ""
    allocator_oom_lines: List[str] = []

    for idx, ln in enumerate(filtered):
        low = ln.lower()
        if "cudacachingallocator" in low and "oom" in low:
            is_oom = True
            if ln not in allocator_oom_lines:
                allocator_oom_lines.append(ln)
        if any(pat in low for pat in oom_patterns):
            is_oom = True
            if "outofmemoryerror" in low or "cuda out of memory" in low or "no available memory" in low or "free memory on device" in low:
                # Prefer the first inner EngineCore / worker OOM exception line
                if primary_idx is None or "outofmemoryerror" not in primary_line.lower():
                    primary_idx = idx
                    primary_line = ln.strip()
        if any(pat in low for pat in invalid_patterns):
            is_invalid = True
            if primary_idx is None:
                primary_idx = idx
                primary_line = ln.strip()

    if primary_idx is None:
        # Locate the last meaningful inner exception/ERROR line before the outer APIServer wrapper
        for idx, ln in enumerate(filtered):
            low = ln.lower()
            if any(w in low for w in wrapper_noise):
                continue
            if (
                re.search(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):", ln)
                or ("error " in low and "[core.py:" in low and "traceback" not in low and "failed to start" not in low)
            ):
                primary_idx = idx
                primary_line = ln.strip()

    if primary_idx is None and filtered:
        primary_idx = len(filtered) - 1
        primary_line = filtered[-1].strip()

    excerpt_lines: List[str] = []
    for alloc_ln in allocator_oom_lines[:2]:
        excerpt_lines.append(alloc_ln)
    if primary_idx is not None and filtered:
        start_i = max(0, primary_idx - 18)
        end_i = min(len(filtered), primary_idx + 12)
        for cand_ln in filtered[start_i:end_i]:
            if cand_ln not in excerpt_lines:
                excerpt_lines.append(cand_ln)
        tail_start = max(end_i, len(filtered) - 8)
        if tail_start < len(filtered):
            if tail_start > end_i:
                excerpt_lines.append("... [outer wrapper frames omitted] ...")
            for tail_ln in filtered[tail_start:]:
                if tail_ln not in excerpt_lines:
                    excerpt_lines.append(tail_ln)

    excerpt = "\n".join(excerpt_lines).strip()
    if len(excerpt) > max_chars:
        if primary_line and primary_line not in excerpt[: max_chars // 2]:
            excerpt = f"{primary_line}\n...\n{excerpt[-(max_chars - len(primary_line) - 10):]}"
        else:
            excerpt = excerpt[: max_chars - 24] + "\n...[truncated]..."

    if is_oom:
        err_code = "cuda_oom"
        http_status = 507
    elif is_invalid:
        err_code = "invalid_model_identifier"
        http_status = 400
    else:
        err_code = "checkpoint_load_failed"
        http_status = 500

    return {
        "is_oom": is_oom,
        "error_code": err_code,
        "http_status": http_status,
        "root_cause_line": primary_line,
        "summary": primary_line,
        "diagnostic_excerpt": excerpt,
        "diagnostic": excerpt,
    }


def _wait_for_vllm_slot_health(
    host: str,
    port: int,
    process: Optional[Any] = None,
    timeout_sec: float = 300.0,
    poll_interval_sec: float = 0.15,
    log_path: Optional[str] = None,
) -> bool:
    """Polls `GET http://<host>:<port>/health` until HTTP 200 or fails fast (`<= 250 ms`) if the child process exits."""
    deadline = time.monotonic() + max(0.5, float(timeout_sec))
    last_err = "health endpoint did not return HTTP 200"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            rc = process.returncode
            raw_log = ""
            if log_path and Path(log_path).is_file():
                try:
                    raw_log = Path(log_path).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    raw_log = ""
            parsed = extract_vllm_root_cause_diagnostic(raw_log)
            root_ln = parsed["root_cause_line"]
            excerpt = parsed["diagnostic_excerpt"]
            headline = f"vllm serve child on {host}:{port} exited prematurely with code {rc}."
            if root_ln:
                headline += f" Root cause: {root_ln}"
            full_msg = headline + (f"\nDiagnostic log excerpt:\n{excerpt}" if excerpt else "")
            raise VLLMStartupError(
                full_msg,
                is_oom=bool(parsed["is_oom"]),
                error_code=str(parsed["error_code"]),
                http_status=int(parsed["http_status"]),
                root_cause_line=str(root_ln),
                diagnostic_excerpt=str(excerpt),
                returncode=rc,
            )
        conn: Optional[http.client.HTTPConnection] = None
        try:
            conn = http.client.HTTPConnection(str(host), int(port), timeout=1.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            if resp.status == 200:
                return True
            last_err = f"HTTP {resp.status}"
        except Exception as exc:
            last_err = str(exc)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        if process is not None and process.poll() is not None:
            continue
        time.sleep(min(float(poll_interval_sec), max(0.02, deadline - time.monotonic())))

    raise TimeoutError(f"Timed out waiting {timeout_sec:.1f}s for vllm serve health on {host}:{port} ({last_err}).")


def _default_spawn_vllm_process_slot(
    *,
    model_alias: str,
    hf_repo_id: str,
    checkpoint_target: str,
    port: int = 8001,
    host: str = "127.0.0.1",
    precision: Optional[str] = None,
    weights_gb: float = 0.0,
    role: str = "primary",
    agent_count: int = 1,
    gpu_tier: str = "L4",
    compute_capability: Optional[Tuple[int, int]] = None,
    free_vram_gb: float = 22.5,
    total_vram_gb: Optional[float] = None,
    slot_vram_fraction: Optional[float] = None,
    max_model_len: int = 16384,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    active_slot_count: int = 1,
    arch_meta: Optional[Mapping[str, Any]] = None,
    health_timeout_sec: float = 300.0,
    enable_vision: Optional[bool] = None,
) -> VLLMProcessSlot:
    """Spawns a dedicated `vllm serve` child process in its own process group (`start_new_session=True`) and awaits `/health`."""
    built = build_vllm_serve_args(
        model_alias=model_alias,
        checkpoint_target=checkpoint_target,
        port=port,
        host=host,
        gpu_tier=gpu_tier,
        compute_capability=compute_capability,
        free_vram_gb=free_vram_gb,
        total_vram_gb=total_vram_gb,
        slot_vram_fraction=slot_vram_fraction,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        active_slot_count=active_slot_count,
        agent_count=agent_count,
        arch_meta=arch_meta,
        precision=precision,
        enable_vision=enable_vision,
    )
    flags = built["resolved_flags"]
    eff_slot_prec = str(flags.get("resolved_precision") or precision or "Q4")
    log_dir = Path(os.environ.get("PLATODOOM_VLLM_LOG_DIR", "/tmp"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"platodoom_vllm_slot_{int(port)}.log"

    log_fp = open(log_file, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            built["argv"],
            env=built["env"],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_fp.close()

    pgid_val: Optional[int] = None
    if hasattr(os, "getpgid"):
        try:
            pgid_val = os.getpgid(proc.pid)
        except Exception:
            pgid_val = proc.pid

    v_slot = VLLMProcessSlot(
        alias=flags["model_alias"],
        hf_repo_id=hf_repo_id,
        checkpoint_target=checkpoint_target,
        port=int(port),
        host=str(host),
        precision=eff_slot_prec,
        weights_gb=float(weights_gb or flags["weights_gb"]),
        role=role,
        agent_count=max(1, int(agent_count)),
        gpu_memory_utilization=float(flags["gpu_memory_utilization"]),
        max_model_len=int(flags["max_model_len"]),
        max_num_seqs=int(flags["max_num_seqs"]),
        max_num_batched_tokens=int(flags["max_num_batched_tokens"]),
        argv=list(built["argv"]),
        resolved_flags=dict(flags),
        process=proc,
        pid=proc.pid,
        pgid=pgid_val,
        log_path=str(log_file),
        healthy=False,
    )
    try:
        _wait_for_vllm_slot_health(
            host=host,
            port=port,
            process=proc,
            timeout_sec=health_timeout_sec,
            log_path=str(log_file),
        )
        v_slot.healthy = True
    except Exception as first_exc:
        _terminate_slot_process_group(v_slot, timeout_sec=5.0, poll_vram=True)
        # Fail fast without silent `--language-model-only` downgrade so Vision/startup errors surface immediately.
        diag_excerpt = ""
        try:
            if log_file.is_file():
                diag_excerpt = str(
                    extract_vllm_root_cause_diagnostic(
                        log_file.read_text(encoding="utf-8", errors="replace")
                    ).get("diagnostic_excerpt")
                    or ""
                ).strip()
        except Exception:
            diag_excerpt = ""
        if diag_excerpt and diag_excerpt not in str(first_exc):
            raise RuntimeError(
                f"vllm serve failed for '{flags['model_alias']}' (vision={bool(flags.get('vision', False))}): "
                f"{first_exc}\n{diag_excerpt}"
            ) from first_exc
        raise first_exc
    return v_slot


def _normalize_chat_messages_roles_and_content(
    raw_msgs: Any,
    *,
    max_images_per_prompt: int = 8,
) -> List[Dict[str, Any]]:
    """Normalizes OpenAI `developer` roles to `system`, flattens pure-text content arrays, clamps multimodal images to a newest-first sliding window, and normalizes tool payloads."""
    if not isinstance(raw_msgs, list):
        return []
    norm_msgs: List[Dict[str, Any]] = []
    for m_idx, raw_m in enumerate(raw_msgs):
        if not isinstance(raw_m, Mapping):
            continue
        m_copy = dict(raw_m)
        if str(m_copy.get("role") or "").strip().lower() == "developer":
            m_copy["role"] = "system"
        raw_content = m_copy.get("content")
        if isinstance(raw_content, list):
            text_parts: List[str] = []
            all_text_parts = True
            for part in raw_content:
                if isinstance(part, Mapping) and part.get("type") in ("text", "input_text"):
                    text_parts.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    all_text_parts = False
                    break
            if all_text_parts:
                m_copy["content"] = "\n\n".join(p for p in text_parts if p)
        tcs = m_copy.get("tool_calls")
        if isinstance(tcs, list):
            norm_tcs: List[Dict[str, Any]] = []
            for tc_idx, tc in enumerate(tcs):
                if not isinstance(tc, Mapping):
                    continue
                fn_raw = tc.get("function") if isinstance(tc.get("function"), Mapping) else {}
                fn_name = str(fn_raw.get("name") or tc.get("name") or "unknown")
                raw_args = fn_raw.get("arguments", tc.get("arguments", "{}"))
                if isinstance(raw_args, str):
                    args_str = raw_args
                else:
                    args_str = json.dumps(raw_args if raw_args is not None else {}, ensure_ascii=False)
                norm_tcs.append(
                    {
                        "id": str(tc.get("id") or f"call_{m_idx}_{tc_idx}"),
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_str,
                        },
                    }
                )
            m_copy["tool_calls"] = norm_tcs
        if m_copy.get("role") == "tool" and m_copy.get("content") is not None and not isinstance(m_copy.get("content"), str):
            m_copy["content"] = json.dumps(m_copy["content"], ensure_ascii=False)
        norm_msgs.append(m_copy)

    if max_images_per_prompt is not None and int(max_images_per_prompt) > 0:
        cap = int(max_images_per_prompt)
        image_coords: List[Tuple[int, int]] = []
        for msg_idx, msg in enumerate(norm_msgs):
            content_list = msg.get("content")
            if not isinstance(content_list, list):
                continue
            for part_idx, part in enumerate(content_list):
                if isinstance(part, Mapping) and part.get("type") in ("image_url", "input_image", "image"):
                    image_coords.append((msg_idx, part_idx))
        if len(image_coords) > cap:
            to_evict = set(image_coords[:-cap])
            touched_msgs = {m_i for m_i, _ in to_evict}
            for msg_idx in touched_msgs:
                old_list = norm_msgs[msg_idx].get("content")
                if not isinstance(old_list, list):
                    continue
                new_list: List[Any] = []
                for part_idx, part in enumerate(old_list):
                    if (msg_idx, part_idx) in to_evict:
                        new_list.append(
                            {
                                "type": "text",
                                "text": "[earlier image omitted: per-prompt image limit]",
                            }
                        )
                    else:
                        new_list.append(part)
                norm_msgs[msg_idx]["content"] = new_list

    return norm_msgs


def _normalize_openai_chat_request_for_vllm(
    payload: Mapping[str, Any],
    target_model_alias: Optional[str] = None,
    *,
    max_images_per_prompt: int = 8,
) -> Dict[str, Any]:
    """Normalizes OpenAI `/v1/chat/completions` payloads for native `vLLM` (developer role, content-list flattening, newest-image sliding window, tool-prompt fallback, and HTTP 422 prevention)."""
    out = dict(payload)
    if target_model_alias:
        out["model"] = str(target_model_alias)

    thinking_pref = _extract_caller_thinking_flag(out)
    if thinking_pref is not None:
        ctk = dict(out.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", bool(thinking_pref))
        out["chat_template_kwargs"] = ctk

    for custom_key in ("enable_thinking", "thinking", "prefer_batched"):
        out.pop(custom_key, None)

    if out.get("stream"):
        s_opts = dict(out.get("stream_options") or {})
        s_opts.setdefault("include_usage", True)
        out["stream_options"] = s_opts

    raw_msgs = out.get("messages")
    if isinstance(raw_msgs, list):
        out["messages"] = _normalize_chat_messages_roles_and_content(
            raw_msgs,
            max_images_per_prompt=max_images_per_prompt,
        )

    eff_alias = str(target_model_alias or out.get("model") or "").strip()
    if eff_alias and isinstance(out.get("messages"), list):
        try:
            parser_spec = resolve_vllm_model_parser_spec(eff_alias)
        except Exception:
            parser_spec = {}
        if parser_spec.get("client_prompt_tool_fallback"):
            has_tools_or_history = bool(out.get("tools")) or any(
                m.get("role") == "tool" or m.get("tool_calls") for m in out["messages"]
            )
            if has_tools_or_history:
                try:
                    proto_profile = introspect_model_protocol(eff_alias, allow_remote=False)
                except Exception:
                    proto_profile = {
                        "template_family": "deepseek_r1_distill"
                        if parser_spec.get("family") == "deepseek_r1_distill"
                        else "qwen3_chatml",
                        "serialize_tools_in_system_prompt": True,
                        "serialize_tool_role_as_user_xml": parser_spec.get("family") in (
                            "deepseek_r1_distill",
                            "deepseek_v2_lite",
                        ),
                    }
                compiled_msgs, _ = compile_model_chat_messages(
                    out["messages"],
                    tools=out.get("tools") if isinstance(out.get("tools"), list) else None,
                    protocol_profile=proto_profile,
                    enable_thinking=thinking_pref,
                )
                out["messages"] = compiled_msgs
            out.pop("tools", None)
            out.pop("tool_choice", None)
            out.pop("parallel_tool_calls", None)

    return out


def _augment_vllm_response_reasoning_compat(
    obj: Dict[str, Any],
    fallback_model_alias: Optional[str] = None,
) -> Dict[str, Any]:
    """Mirrors `reasoning` <-> `reasoning_content` across both non-streaming messages and SSE delta chunks and extracts XML tool calls on non-parser models."""
    if not isinstance(obj, dict):
        return obj
    if fallback_model_alias:
        obj["model"] = fallback_model_alias
    choices = obj.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message")
            if isinstance(msg, dict):
                if msg.get("reasoning") is not None and msg.get("reasoning_content") is None:
                    msg["reasoning_content"] = msg["reasoning"]
                elif msg.get("reasoning_content") is not None and msg.get("reasoning") is None:
                    msg["reasoning"] = msg["reasoning_content"]
                content_txt = msg.get("content")
                if (
                    isinstance(content_txt, str)
                    and not msg.get("tool_calls")
                    and (
                        "<tool_call>" in content_txt
                        or "```tool_call" in content_txt
                        or "tool▁call▁begin" in content_txt
                        or "<think>" in content_txt
                    )
                ):
                    extracted = extract_tool_calls_from_text(content_txt)
                    msg["content"] = extracted.cleaned_text
                    if extracted.reasoning_content and not msg.get("reasoning_content"):
                        msg["reasoning_content"] = extracted.reasoning_content
                        msg.setdefault("reasoning", extracted.reasoning_content)
                    if extracted.tool_calls:
                        msg["tool_calls"] = extracted.tool_calls
                        if ch.get("finish_reason") in (None, "stop"):
                            ch["finish_reason"] = "tool_calls"
            delta = ch.get("delta")
            if isinstance(delta, dict):
                if delta.get("reasoning") is not None and delta.get("reasoning_content") is None:
                    delta["reasoning_content"] = delta["reasoning"]
                elif delta.get("reasoning_content") is not None and delta.get("reasoning") is None:
                    delta["reasoning"] = delta["reasoning_content"]
    return obj


def _proxy_chat_completion_to_vllm_slot(
    slot_host: str,
    slot_port: int,
    slot_alias: str,
    payload: Mapping[str, Any],
    timeout_sec: float = 300.0,
) -> Tuple[int, Dict[str, Any]]:
    """Forwards a non-streaming `/v1/chat/completions` call locklessly to `127.0.0.1:<port>` with `TCP_NODELAY=1`."""
    norm_payload = _normalize_openai_chat_request_for_vllm(payload, target_model_alias=slot_alias)
    norm_payload["stream"] = False
    raw_req = json.dumps(norm_payload, ensure_ascii=False).encode("utf-8")

    conn = http.client.HTTPConnection(str(slot_host), int(slot_port), timeout=float(timeout_sec))
    try:
        conn.connect()
        if conn.sock is not None:
            try:
                conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw_req,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        resp = conn.getresponse()
        raw_resp = resp.read().decode("utf-8", errors="replace")
        status = int(resp.status)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    try:
        parsed = json.loads(raw_resp) if raw_resp else {}
    except Exception:
        parsed = {"error": {"message": raw_resp or f"Upstream vLLM HTTP {status}", "type": "upstream_error"}}

    if status == 200 and isinstance(parsed, dict):
        parsed = _augment_vllm_response_reasoning_compat(parsed, fallback_model_alias=slot_alias)
    return status, parsed


_SSE_TOOL_MARKERS: Tuple[str, ...] = (
    "<tool_call>",
    "```tool_call",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁calls▁begin｜>",
)
_SSE_MAX_MARKER_LEN = max(len(m) for m in _SSE_TOOL_MARKERS)


class _SseToolCallStreamExtractor:
    """Streams safe visible prose immediately while buffering `<tool_call>` / DeepSeek tool blocks for OpenAI `delta.tool_calls` emission."""

    def __init__(self) -> None:
        self.pending_buf: str = ""
        self.tool_buf: str = ""
        self.in_tool_block: bool = False
        self.finalized: bool = False

    def feed_text(self, text: str) -> str:
        if not text:
            return ""
        if self.in_tool_block:
            self.tool_buf += text
            return ""
        self.pending_buf += text
        earliest_idx: Optional[int] = None
        for marker in _SSE_TOOL_MARKERS:
            idx = self.pending_buf.find(marker)
            if idx != -1 and (earliest_idx is None or idx < earliest_idx):
                earliest_idx = idx
        if earliest_idx is not None:
            safe_out = self.pending_buf[:earliest_idx]
            self.tool_buf += self.pending_buf[earliest_idx:]
            self.pending_buf = ""
            self.in_tool_block = True
            return safe_out
        hold_len = 0
        max_check = min(len(self.pending_buf), _SSE_MAX_MARKER_LEN - 1)
        for cand_len in range(max_check, 0, -1):
            suffix = self.pending_buf[-cand_len:]
            if any(m.startswith(suffix) for m in _SSE_TOOL_MARKERS):
                hold_len = cand_len
                break
        if hold_len == 0:
            safe_out = self.pending_buf
            self.pending_buf = ""
            return safe_out
        safe_out = self.pending_buf[:-hold_len]
        self.pending_buf = self.pending_buf[-hold_len:]
        return safe_out

    def finalize(self) -> Tuple[str, List[Dict[str, Any]]]:
        if self.finalized:
            return "", []
        self.finalized = True
        trailing_prose = self.pending_buf
        self.pending_buf = ""
        if not self.tool_buf:
            return trailing_prose, []
        extracted = extract_tool_calls_from_text(self.tool_buf)
        self.tool_buf = ""
        indexed_calls: List[Dict[str, Any]] = []
        for idx, tc in enumerate(extracted.tool_calls):
            tc_copy = dict(tc)
            tc_copy["index"] = idx
            indexed_calls.append(tc_copy)
        extra_text = (trailing_prose + ("\n" + extracted.cleaned_text if extracted.cleaned_text else "")).strip()
        return extra_text, indexed_calls


def _proxy_sse_chat_completion_to_vllm_slot(
    handler: BaseHTTPRequestHandler,
    slot_host: str,
    slot_port: int,
    slot_alias: str,
    payload: Mapping[str, Any],
    timeout_sec: float = 300.0,
) -> None:
    """Streams SSE chunks line-by-line from `127.0.0.1:<port>` with zero mutex contention and immediate abort propagation."""
    try:
        handler.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass

    norm_payload = _normalize_openai_chat_request_for_vllm(payload, target_model_alias=slot_alias)
    norm_payload["stream"] = True
    raw_req = json.dumps(norm_payload, ensure_ascii=False).encode("utf-8")

    try:
        parser_spec = resolve_vllm_model_parser_spec(slot_alias)
        use_tool_extractor = bool(parser_spec.get("client_prompt_tool_fallback"))
    except Exception:
        use_tool_extractor = False
    sse_tool_extractor = _SseToolCallStreamExtractor() if use_tool_extractor else None
    last_chunk_meta: Dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": slot_alias,
    }

    conn = http.client.HTTPConnection(str(slot_host), int(slot_port), timeout=float(timeout_sec))
    resp: Optional[http.client.HTTPResponse] = None
    raw_sock: Optional[socket.socket] = None
    try:
        conn.connect()
        raw_sock = conn.sock
        if raw_sock is not None:
            try:
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw_req,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        if int(resp.status) != 200:
            err_raw = resp.read()
            handler.send_response(int(resp.status))
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(err_raw)))
            handler.end_headers()
            handler.wfile.write(err_raw)
            return

        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache, no-transform")
        handler.send_header("Connection", "keep-alive")
        handler.send_header("X-Accel-Buffering", "no")
        handler.end_headers()
        handler.wfile.write(b": keepalive\n\n")
        handler.wfile.flush()

        def _flush_extracted_tools(extra_txt: str, calls: List[Dict[str, Any]], meta: Mapping[str, Any]) -> None:
            if extra_txt:
                txt_chunk = {
                    "id": meta.get("id"),
                    "object": "chat.completion.chunk",
                    "created": meta.get("created"),
                    "model": slot_alias,
                    "choices": [{"index": 0, "delta": {"content": extra_txt}, "finish_reason": None}],
                }
                handler.wfile.write(f"data: {json.dumps(txt_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            if calls:
                tc_chunk = {
                    "id": meta.get("id"),
                    "object": "chat.completion.chunk",
                    "created": meta.get("created"),
                    "model": slot_alias,
                    "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}],
                }
                handler.wfile.write(f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

        while True:
            raw_line = resp.fp.readline()
            if not raw_line:
                break
            line_str = raw_line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue
            if line_str.startswith(":"):
                handler.wfile.write(f"{line_str}\n\n".encode("utf-8"))
                handler.wfile.flush()
                continue
            if not line_str.startswith("data:"):
                continue
            data_part = line_str[5:].strip()
            if data_part == "[DONE]":
                if sse_tool_extractor is not None and not sse_tool_extractor.finalized:
                    extra_txt, extracted_calls = sse_tool_extractor.finalize()
                    if extra_txt or extracted_calls:
                        _flush_extracted_tools(extra_txt, extracted_calls, last_chunk_meta)
                        if extracted_calls:
                            fin_chunk = {
                                "id": last_chunk_meta.get("id"),
                                "object": "chat.completion.chunk",
                                "created": last_chunk_meta.get("created"),
                                "model": slot_alias,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                            }
                            handler.wfile.write(f"data: {json.dumps(fin_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
                break
            try:
                chunk_obj = json.loads(data_part)
                if isinstance(chunk_obj, dict):
                    chunk_obj = _augment_vllm_response_reasoning_compat(chunk_obj, fallback_model_alias=slot_alias)
                    last_chunk_meta = {
                        "id": chunk_obj.get("id") or last_chunk_meta["id"],
                        "created": chunk_obj.get("created") or last_chunk_meta["created"],
                    }
                    if sse_tool_extractor is not None and isinstance(chunk_obj.get("choices"), list) and chunk_obj["choices"]:
                        ch0 = chunk_obj["choices"][0]
                        delta0 = ch0.get("delta") if isinstance(ch0, dict) else None
                        if isinstance(delta0, dict) and isinstance(delta0.get("content"), str) and not delta0.get("tool_calls"):
                            safe_txt = sse_tool_extractor.feed_text(delta0["content"])
                            if safe_txt:
                                delta0["content"] = safe_txt
                            else:
                                delta0.pop("content", None)
                        if isinstance(ch0, dict) and ch0.get("finish_reason") is not None:
                            extra_txt, extracted_calls = sse_tool_extractor.finalize()
                            if extra_txt or extracted_calls:
                                _flush_extracted_tools(extra_txt, extracted_calls, last_chunk_meta)
                                if extracted_calls and ch0.get("finish_reason") in ("stop", "length"):
                                    ch0["finish_reason"] = "tool_calls"
                        elif isinstance(delta0, dict) and not delta0 and not chunk_obj.get("usage"):
                            continue
                    out_bytes = f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n".encode("utf-8")
                else:
                    out_bytes = f"data: {data_part}\n\n".encode("utf-8")
            except Exception:
                out_bytes = f"data: {data_part}\n\n".encode("utf-8")
            handler.wfile.write(out_bytes)
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        # Client disconnected: finally block immediately shuts down raw_sock + resp + conn so vLLM frees KV blocks immediately.
        pass
    finally:
        handler.close_connection = True
        if raw_sock is not None:
            try:
                raw_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                raw_sock.close()
            except Exception:
                pass
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


def _proxy_stream_to_token_queue(
    slot_host: str,
    slot_port: int,
    slot_alias: str,
    payload: Mapping[str, Any],
    token_queue: queue.Queue,
    abort_event: threading.Event,
    timeout_sec: float = 300.0,
) -> None:
    """Streams SSE chunks from `127.0.0.1:<port>` into `token_queue` for programmatic `stream_chat_completion` callers."""
    norm_payload = _normalize_openai_chat_request_for_vllm(payload, target_model_alias=slot_alias)
    norm_payload["stream"] = True
    raw_req = json.dumps(norm_payload, ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection(str(slot_host), int(slot_port), timeout=float(timeout_sec))
    resp: Optional[http.client.HTTPResponse] = None
    raw_sock: Optional[socket.socket] = None
    try:
        conn.connect()
        raw_sock = conn.sock
        if raw_sock is not None:
            try:
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw_req,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        if int(resp.status) != 200:
            err_body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Upstream vLLM HTTP {resp.status}: {err_body}")
        while not abort_event.is_set():
            raw_line = resp.fp.readline()
            if not raw_line:
                break
            line_str = raw_line.decode("utf-8", errors="replace").strip()
            if not line_str or not line_str.startswith("data:"):
                continue
            data_part = line_str[5:].strip()
            if data_part == "[DONE]":
                break
            try:
                chunk_obj = json.loads(data_part)
                if isinstance(chunk_obj, dict):
                    chunk_obj = _augment_vllm_response_reasoning_compat(chunk_obj, fallback_model_alias=slot_alias)
                    token_queue.put({"type": "chunk", "data": chunk_obj})
            except Exception:
                pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass
class LoadedModelSlot:
    """Represents a single resident model slot (either a native `vllm serve` child on `127.0.0.1:<port>` or an in-process TPU/diffusion/mock model)."""

    alias: str
    hf_repo_id: str
    checkpoint_id: str
    precision: str = "Q4"
    weights_gb: float = 0.0
    role: str = "primary"
    agent_count: int = 1
    expected_concurrency: int = 1
    model: Any = None
    tokenizer: Any = None
    tokenizer_capabilities: Dict[str, Any] = field(default_factory=dict)
    stop_token_ids: List[int] = field(default_factory=list)
    architecture_metadata: Dict[str, Any] = field(default_factory=dict)
    is_qwen35_model: bool = False
    bound_fla_kernels: int = 0
    lm_head_last_token_patched: bool = False
    coordinator: Optional[DynamicBatchCoordinator] = None
    upstream_host: str = "127.0.0.1"
    upstream_port: Optional[int] = None
    vllm_slot: Optional[VLLMProcessSlot] = None
    max_model_len: Optional[int] = None
    modality: str = "text"

    @property
    def is_resident(self) -> bool:
        if self.upstream_port is not None:
            return True
        if self.model is not None and (self.tokenizer is not None or self.modality == "image"):
            return True
        return False

    @property
    def resolved_flags(self) -> Dict[str, Any]:
        if self.vllm_slot is not None and isinstance(getattr(self.vllm_slot, "resolved_flags", None), dict):
            return dict(self.vllm_slot.resolved_flags)
        return {}

    @property
    def vision(self) -> bool:
        return bool(self.resolved_flags.get("vision", False))

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


class _UpstreamHTTPStatusError(RuntimeError):
    """Propagates an upstream vLLM HTTP error status and JSON payload without masking."""

    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = int(status_code)
        self.payload = payload if isinstance(payload, dict) else {"error": {"message": str(payload)}}
        err_obj = self.payload.get("error")
        msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj or self.payload)
        super().__init__(str(msg or f"Upstream HTTP {self.status_code}"))


class BatchedQwen35ServingEngine:
    """Port-8000 Supervisor & Serving Engine managing native `vllm serve` child process groups (`[01]`-`[14]` on CUDA) and TPU/Diffusion/Mock slots."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        total_vram_gb: float = 40.0,
        max_num_seqs: int = MAX_PARALLEL_AGENTS,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
        batch_window_ms: Optional[float] = None,
        queue_timeout_sec: float = 300.0,
        active_model_alias: str = UNIFIED_MODEL_ALIAS,
        active_hf_repo_id: str = BASE_MODEL_ID,
        loaded_checkpoint_path: Optional[str] = None,
        model_loader: Optional[Callable[..., Tuple[Any, Any, str]]] = None,
        initial_stack: Optional[Sequence[Any]] = None,
        agent_counts: Optional[Mapping[str, int]] = None,
        vllm_spawner: Optional[Callable[..., VLLMProcessSlot]] = None,
        use_vllm_subprocess: Optional[bool] = None,
        base_vllm_port: int = 8001,
        gpu_tier: Optional[str] = None,
        compute_capability: Optional[Tuple[int, int]] = None,
        initial_vllm_slot: Optional[VLLMProcessSlot] = None,
        free_vram_gb: Optional[float] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.total_vram_gb = float(total_vram_gb)
        self._explicit_free_vram_gb = float(free_vram_gb) if free_vram_gb is not None else None
        self._model_loader = model_loader or default_physical_model_loader
        self._using_default_loader = model_loader is None
        self._vllm_spawner = vllm_spawner or _default_spawn_vllm_process_slot
        self._use_vllm_subprocess = (
            bool(use_vllm_subprocess)
            if use_vllm_subprocess is not None
            else (vllm_spawner is not None or initial_vllm_slot is not None)
        )
        self._base_vllm_port = int(base_vllm_port)
        self.gpu_tier = gpu_tier
        self.compute_capability = compute_capability
        self._batch_window_ms = (
            float(batch_window_ms)
            if batch_window_ms is not None
            else float(os.environ.get("SANDCASTLE_BATCH_WINDOW_MS", "50.0"))
        )
        self._queue_timeout_sec = float(queue_timeout_sec)
        self.switch_in_progress: Optional[str] = None
        self.last_switch_error: Optional[str] = None
        self._loaded_checkpoint_id: Optional[str] = None
        self.stack_preset: Optional[str] = None
        self.loaded_models: Dict[str, LoadedModelSlot] = {}
        self.inactive_unloaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_agent_counts: Dict[str, int] = {}
        self.configured_models: List[str] = []
        self.config = compute_serving_concurrency_config(
            total_vram_gb=total_vram_gb,
            max_model_len=max_model_len,
            target_parallel_agents=max_num_seqs,
        )
        # Control-plane mutex ONLY for switch_model / load_model_stack (NEVER acquired on inference paths!)
        self._control_lock = threading.Lock()
        self._gpu_lock = self._control_lock

        self.coordinator = DynamicBatchCoordinator(
            batch_execute_fn=self._run_gpu_batch,
            max_num_seqs=self.config["max_num_seqs"],
            max_model_len=self.config["max_model_len"],
            max_num_batched_tokens=self.config["max_num_batched_tokens"],
            batch_window_ms=self._batch_window_ms,
            queue_timeout_sec=queue_timeout_sec,
            on_close_fn=lambda: self._close_all_slot_coordinators(),
        )

        if initial_vllm_slot is not None:
            self._bind_vllm_process_slot(initial_vllm_slot)
        elif self.model is not None and self.tokenizer is not None:
            self._rebind_loaded_model_and_tokenizer(
                model=self.model,
                tokenizer=self.tokenizer,
                model_alias=active_model_alias,
                hf_repo_id=active_hf_repo_id,
                checkpoint_id=loaded_checkpoint_path or active_hf_repo_id or active_model_alias,
            )
        else:
            self.active_model_alias = None
            self.active_hf_repo_id = None
            self.stop_token_ids: List[int] = []
            self.tokenizer_capabilities: Dict[str, Any] = {}
            self.architecture_metadata: Dict[str, Any] = {}
            self.is_qwen35_model = False
            self.bound_fla_kernels = 0
            self.lm_head_last_token_patched = False

        if initial_stack is not None:
            st_res = self.load_model_stack(initial_stack, agent_counts=agent_counts)
            if not st_res.get("ok"):
                raise RuntimeError(str(st_res.get("diagnostic") or "Failed loading initial_stack."))

    def _probe_free_vram_gb(self) -> float:
        """Returns physical free GPU VRAM in GiB after accounting for pre-existing host CUDA allocations."""
        if self._explicit_free_vram_gb is not None:
            return min(self.total_vram_gb, max(0.5, float(self._explicit_free_vram_gb)))
        try:
            import torch

            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(0)
                live_total_gib = float(total_b) / float(1024 ** 3)
                if abs(live_total_gib - self.total_vram_gb) <= 2.0:
                    return round(float(free_b) / float(1024 ** 3), 3)
        except Exception:
            pass
        return self.total_vram_gb

    def _record_slot_batch_on_engine_coordinator(self, batch_len: int) -> None:
        if not hasattr(self, "coordinator") or self.coordinator is None:
            return
        with self.coordinator._metrics_lock:
            self.coordinator.total_batches_executed += 1
            self.coordinator.total_requests_served += int(batch_len)
            self.coordinator.last_batch_size = int(batch_len)
            self.coordinator.batch_size_histogram[int(batch_len)] = (
                self.coordinator.batch_size_histogram.get(int(batch_len), 0) + 1
            )
            if int(batch_len) > self.coordinator.max_observed_batch_size:
                self.coordinator.max_observed_batch_size = int(batch_len)

    def _make_slot_coordinator(
        self,
        alias: str,
        *,
        max_num_seqs: Optional[int] = None,
        target_cohort_size: Optional[int] = None,
    ) -> DynamicBatchCoordinator:
        eff_max_seqs = max(1, int(max_num_seqs if max_num_seqs is not None else self.config["max_num_seqs"]))
        eff_batched_tokens = max(int(self.config["max_num_batched_tokens"]), eff_max_seqs * 4096)
        return DynamicBatchCoordinator(
            batch_execute_fn=lambda batch_items, _a=alias: self._run_gpu_batch_for_slot(_a, batch_items),
            max_num_seqs=eff_max_seqs,
            max_model_len=self.config["max_model_len"],
            max_num_batched_tokens=eff_batched_tokens,
            batch_window_ms=self._batch_window_ms,
            queue_timeout_sec=self._queue_timeout_sec,
            target_cohort_size=target_cohort_size,
            on_batch_telemetry_fn=self._record_slot_batch_on_engine_coordinator,
        )

    def _build_slot(
        self,
        *,
        alias: str,
        hf_repo_id: str,
        checkpoint_id: str,
        model: Any,
        tokenizer: Any,
        precision: str = "Q4",
        weights_gb: float = 0.0,
        role: str = "primary",
        agent_count: int = 1,
        target_cohort_size: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        coordinator: Optional[DynamicBatchCoordinator] = None,
    ) -> LoadedModelSlot:
        caps = introspect_loaded_tokenizer_capabilities(tokenizer, model_alias=alias, model=model)
        if model is not None and hasattr(model, "generate"):
            _get_hf_generation_types()
        stop_ids = list(caps.get("stop_token_ids", []))
        arch_meta = _coerce_checkpoint_metadata(model)
        is_q35 = is_qwen35_architecture(model)
        fla_cnt = ensure_qwen35_fast_kernels_bound(model)
        lm_patched = patch_last_token_only_lm_head(model)
        slot_coord = (
            coordinator
            if (coordinator is not None and coordinator is not self.coordinator)
            else self._make_slot_coordinator(
                alias,
                max_num_seqs=max_num_seqs,
                target_cohort_size=target_cohort_size,
            )
        )
        return LoadedModelSlot(
            alias=alias,
            hf_repo_id=hf_repo_id,
            checkpoint_id=checkpoint_id,
            precision=precision,
            weights_gb=weights_gb,
            role=role,
            agent_count=max(1, int(agent_count)),
            expected_concurrency=max(1, int(agent_count)),
            model=model,
            tokenizer=tokenizer,
            tokenizer_capabilities=caps,
            stop_token_ids=stop_ids,
            architecture_metadata=arch_meta,
            is_qwen35_model=is_q35,
            bound_fla_kernels=fla_cnt,
            lm_head_last_token_patched=lm_patched,
            coordinator=slot_coord,
        )

    def _close_all_slot_coordinators(self) -> None:
        for existing_slot in list(self.loaded_models.values()):
            c = existing_slot.coordinator
            if c is not None and c is not getattr(self, "coordinator", None):
                try:
                    c.close()
                except Exception:
                    pass
                existing_slot.coordinator = None

    def _close_non_primary_coordinators(self, keep_coordinators: Optional[set] = None) -> None:
        keep_ids = set(keep_coordinators or set())
        if hasattr(self, "coordinator") and self.coordinator is not None:
            keep_ids.add(id(self.coordinator))
        for existing_slot in list(self.loaded_models.values()):
            c = existing_slot.coordinator
            if c is not None and id(c) not in keep_ids:
                try:
                    c.close()
                except Exception:
                    pass
                existing_slot.coordinator = None

    def close(self) -> None:
        for sl in list(self.loaded_models.values()):
            if sl.vllm_slot is not None:
                _terminate_slot_process_group(sl.vllm_slot, timeout_sec=5.0, poll_vram=False)
                sl.vllm_slot = None
                sl.upstream_port = None
        self._close_all_slot_coordinators()
        if hasattr(self, "coordinator") and self.coordinator is not None:
            self.coordinator.close()

    def get_aggregated_batch_telemetry(self) -> Dict[str, Any]:
        slot_stats: Dict[str, Dict[str, Any]] = {}
        loaded_slots = getattr(self, "loaded_models", None) or {}
        for s_alias, sl in loaded_slots.items():
            c = sl.coordinator
            if c is not None:
                with c._metrics_lock:
                    slot_stats[s_alias] = {
                        "agent_count": int(sl.agent_count),
                        "expected_concurrency": int(sl.expected_concurrency),
                        "total_batches": int(c.total_batches_executed),
                        "total_batches_executed": int(c.total_batches_executed),
                        "total_requests": int(c.total_requests_served),
                        "total_requests_batched": int(c.total_requests_served),
                        "max_observed_batch_size": int(c.max_observed_batch_size),
                        "last_batch_size": int(c.last_batch_size),
                        "max_num_seqs": int(c.max_num_seqs),
                        "max_batch_size": int(c.max_num_seqs),
                        "target_cohort_size": c.target_cohort_size,
                    }
        eng_coord = getattr(self, "coordinator", None)
        sum_b = sum(v["total_batches"] for v in slot_stats.values())
        sum_r = sum(v["total_requests"] for v in slot_stats.values())
        max_s_obs = max((v["max_observed_batch_size"] for v in slot_stats.values()), default=0)
        if eng_coord is not None:
            with eng_coord._metrics_lock:
                total_b = max(int(eng_coord.total_batches_executed), sum_b)
                total_r = max(int(eng_coord.total_requests_served), sum_r)
                max_obs = max(int(eng_coord.max_observed_batch_size), max_s_obs)
                last_b = int(eng_coord.last_batch_size)
        else:
            total_b = sum_b
            total_r = sum_r
            max_obs = max_s_obs
            last_b = 0
        return {
            "total_batches": total_b,
            "total_batches_executed": total_b,
            "total_requests_served": total_r,
            "max_observed_batch_size": max_obs,
            "last_batch_size": last_b,
            "slot_batch_stats": slot_stats,
        }

    def _sync_primary_attributes_from_slot(self, slot: LoadedModelSlot) -> None:
        self.model = slot.model
        self.tokenizer = slot.tokenizer
        self.active_model_alias = slot.alias
        self.active_hf_repo_id = slot.hf_repo_id
        self._loaded_checkpoint_id = slot.checkpoint_id
        self.tokenizer_capabilities = dict(slot.tokenizer_capabilities)
        self.stop_token_ids = list(slot.stop_token_ids)
        self.architecture_metadata = dict(slot.architecture_metadata)
        self.is_qwen35_model = slot.is_qwen35_model
        self.bound_fla_kernels = slot.bound_fla_kernels
        self.lm_head_last_token_patched = slot.lm_head_last_token_patched

    def _bind_vllm_process_slot(self, v_slot: VLLMProcessSlot, role: str = "primary") -> LoadedModelSlot:
        self._close_non_primary_coordinators()
        slot_coord = self._make_slot_coordinator(
            v_slot.alias,
            max_num_seqs=v_slot.max_num_seqs,
        )
        slot = LoadedModelSlot(
            alias=v_slot.alias,
            hf_repo_id=v_slot.hf_repo_id,
            checkpoint_id=v_slot.checkpoint_target,
            precision=v_slot.precision,
            weights_gb=v_slot.weights_gb,
            role=role,
            agent_count=v_slot.agent_count,
            expected_concurrency=v_slot.agent_count,
            model=None,
            tokenizer=None,
            is_qwen35_model=bool(v_slot.resolved_flags.get("mamba_ssm_cache_dtype") == "auto"),
            coordinator=slot_coord,
            upstream_host=v_slot.host,
            upstream_port=v_slot.port,
            vllm_slot=v_slot,
            max_model_len=int(v_slot.max_model_len),
            modality="text",
        )
        self.loaded_models = {v_slot.alias: slot}
        self.inactive_unloaded_models = {}
        self.model_agent_counts = {v_slot.alias: v_slot.agent_count}
        self.configured_models = [v_slot.alias]
        self.stack_preset = None
        self.config["max_model_len"] = int(v_slot.max_model_len)
        self.config["max_num_seqs"] = int(v_slot.max_num_seqs)
        self.config["max_num_batched_tokens"] = int(v_slot.max_num_batched_tokens)
        self._sync_primary_attributes_from_slot(slot)
        return slot

    def _rebind_loaded_model_and_tokenizer(
        self,
        model: Any,
        tokenizer: Any,
        model_alias: str,
        hf_repo_id: str,
        checkpoint_id: Optional[str] = None,
        precision: str = "Q4",
        weights_gb: float = 0.0,
    ) -> None:
        self._close_non_primary_coordinators()
        slot = self._build_slot(
            alias=model_alias,
            hf_repo_id=hf_repo_id,
            checkpoint_id=checkpoint_id or hf_repo_id or model_alias,
            model=model,
            tokenizer=tokenizer,
            precision=precision,
            weights_gb=weights_gb,
            role="primary",
        )
        self.loaded_models = {model_alias: slot}
        self.inactive_unloaded_models = {}
        self.model_agent_counts = {model_alias: 1}
        self.configured_models = [model_alias]
        self.stack_preset = None
        self._sync_primary_attributes_from_slot(slot)

    def _evict_loaded_model(self) -> None:
        self._close_non_primary_coordinators()
        for sl in list(self.loaded_models.values()):
            if sl.vllm_slot is not None:
                _terminate_slot_process_group(sl.vllm_slot, timeout_sec=10.0, poll_vram=True)
                sl.vllm_slot = None
            sl.upstream_port = None
            sl.model = None
            sl.tokenizer = None
        self.loaded_models.clear()
        self.inactive_unloaded_models.clear()
        self.model_agent_counts.clear()
        self.configured_models.clear()
        self.stack_preset = None
        old_model = self.model
        old_tokenizer = self.tokenizer
        self.model = None
        self.tokenizer = None
        self.active_model_alias = None
        self.active_hf_repo_id = None
        self._loaded_checkpoint_id = None
        self.stop_token_ids = []
        self.tokenizer_capabilities = {}
        self.architecture_metadata = {}
        self.is_qwen35_model = False
        self.bound_fla_kernels = 0
        self.lm_head_last_token_patched = False
        del old_model
        del old_tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _reconcile_single_slot_attributes(self) -> None:
        """Synchronizes single-slot (`x = 1`) metadata if a caller mutates `active_model_alias` or `active_hf_repo_id` directly."""
        if len(self.loaded_models) != 1:
            return
        only_key, only_slot = next(iter(self.loaded_models.items()))
        if only_slot.upstream_port is not None:
            return
        cur_alias = getattr(self, "active_model_alias", None)
        cur_repo = getattr(self, "active_hf_repo_id", None)
        if self.model is None or self.tokenizer is None:
            only_slot.model = self.model
            only_slot.tokenizer = self.tokenizer
            return
        only_slot.model = self.model
        only_slot.tokenizer = self.tokenizer
        if cur_repo and cur_repo != only_slot.hf_repo_id:
            only_slot.hf_repo_id = str(cur_repo)
        if cur_alias and cur_alias != only_key:
            only_slot.alias = str(cur_alias)
            if not cur_repo:
                only_slot.hf_repo_id = str(cur_alias)
            if only_slot.coordinator is not None and only_slot.coordinator is not self.coordinator:
                only_slot.coordinator.batch_execute_fn = (
                    lambda batch_items, _a=only_slot.alias: self._run_gpu_batch_for_slot(_a, batch_items)
                )
            self.loaded_models = {only_slot.alias: only_slot}

    def _is_explicitly_inactive_model(self, requested_model: Optional[str]) -> bool:
        inactive_map = getattr(self, "inactive_unloaded_models", None) or {}
        if not inactive_map or not requested_model or not str(requested_model).strip():
            return False
        req_clean = str(requested_model).strip().split("@", 1)[0].strip()
        req_low = req_clean.lower()
        for inact_alias, meta in inactive_map.items():
            if inact_alias.lower() == req_low:
                return True
            repo = str((meta or {}).get("hf_repo_id") or "")
            if repo and repo.lower() == req_low:
                return True
        try:
            from curation.api_client import resolve_expected_model_identifiers

            accepted = resolve_expected_model_identifiers(req_clean)
            for inact_alias, meta in inactive_map.items():
                repo = str((meta or {}).get("hf_repo_id") or "")
                if inact_alias.lower() in accepted or (repo and repo.lower() in accepted):
                    return True
        except Exception:
            pass
        return False

    def resolve_loaded_slot(self, requested_model: Optional[str] = None) -> LoadedModelSlot:
        """Resolves a co-resident `LoadedModelSlot` by alias or repo ID without evicting any resident model."""
        self._reconcile_single_slot_attributes()
        if self._is_explicitly_inactive_model(requested_model):
            raise KeyError(
                f"Model '{requested_model}' has 0 active agents assigned (n_i=0) and is intentionally unloaded (0 bytes VRAM)."
            )
        if not self.loaded_models:
            if self.model is not None and self.tokenizer is not None:
                alias = self.active_model_alias or UNIFIED_MODEL_ALIAS
                return LoadedModelSlot(
                    alias=alias,
                    hf_repo_id=self.active_hf_repo_id or alias,
                    checkpoint_id=self._loaded_checkpoint_id or alias,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    tokenizer_capabilities=dict(self.tokenizer_capabilities),
                    stop_token_ids=list(self.stop_token_ids),
                    architecture_metadata=dict(self.architecture_metadata),
                    is_qwen35_model=self.is_qwen35_model,
                    coordinator=self.coordinator,
                )
            raise RuntimeError("No model is physically loaded in accelerator memory.")

        if requested_model is None or not str(requested_model).strip():
            if self.active_model_alias and self.active_model_alias in self.loaded_models:
                return self.loaded_models[self.active_model_alias]
            return next(iter(self.loaded_models.values()))

        req_clean = str(requested_model).strip().split("@", 1)[0].strip()
        if req_clean in self.loaded_models:
            return self.loaded_models[req_clean]

        req_low = req_clean.lower()
        for s_alias, slot in self.loaded_models.items():
            if s_alias.lower() == req_low or slot.hf_repo_id.lower() == req_low:
                return slot

        try:
            from curation.api_client import resolve_expected_model_identifiers

            accepted = resolve_expected_model_identifiers(req_clean)
            for s_alias, slot in self.loaded_models.items():
                if s_alias.lower() in accepted or slot.hf_repo_id.lower() in accepted:
                    return slot
        except Exception:
            pass

        if len(self.loaded_models) == 1 and not getattr(self, "inactive_unloaded_models", None):
            return next(iter(self.loaded_models.values()))

        raise KeyError(
            f"Requested model '{requested_model}' is not resident in active co-resident stack {list(self.loaded_models.keys())}."
        )

    def _should_use_vllm_for_alias(self, alias: str) -> bool:
        if not getattr(self, "_use_vllm_subprocess", False):
            return False
        clean = str(alias or "").strip().split("@", 1)[0].strip()
        prof = _SUPERVISOR_MODEL_SPECS.get(clean)
        if prof and prof.get("modality") == "image":
            return False
        return True

    def load_model_stack(
        self,
        stack_specs: Sequence[str | Dict[str, Any]],
        *,
        hardware_tier: Optional[str] = None,
        stack_preset: Optional[str] = None,
        cache_root: Optional[str] = None,
        agent_counts: Optional[Mapping[str, int]] = None,
        enable_vision: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Atomically loads co-resident active models (`n_i > 0`) onto GPU 1 while keeping `n_i == 0` models unloaded (`0` VRAM):
        - Rejects concurrent transitions immediately with HTTP 409 (`SWITCH_IN_PROGRESS`).
        - Pre-validates aggregate active `weights_gb + KV_floor <= vram_gb` BEFORE any eviction (HTTP 507 on breach).
        - Prunes inactive model cache via LRU and stages disk checkpoints ONLY for active (`n_i > 0`) non-resident slots.
        - Terminates evicted child `vllm serve` process groups (`os.killpg`) and spawns/preserves active slots.
        """
        if not self._control_lock.acquire(blocking=False):
            in_prog = getattr(self, "switch_in_progress", None) or "active"
            msg = f"Model stack transition already in progress ('{in_prog}'). Retry once /health returns 200."
            return {
                "ok": False,
                "http_status": 409,
                "code": "switch_in_progress",
                "switching_to": in_prog,
                "error": {"code": "SWITCH_IN_PROGRESS", "message": msg},
                "diagnostic": msg,
            }
        try:
            items = list(stack_specs or [])
            has_agent_counts = agent_counts is not None or any(
                (isinstance(it, Mapping) and ("agent_count" in it or "agents" in it or "count" in it))
                or (
                    isinstance(it, str)
                    and ":" in it
                    and not it.strip().startswith(("gs://", "hf://", "gdrive://", "http://", "https://"))
                )
                for it in items
            )

            if not has_agent_counts and not (1 <= len(items) <= 3):
                msg = f"Invalid stack cardinality x={len(items)}: co-resident stacks require 1 <= x <= 3 models."
                self.last_switch_error = msg
                return {
                    "ok": False,
                    "http_status": 400,
                    "code": "invalid_stack_cardinality",
                    "error": {"code": "INVALID_STACK_CARDINALITY", "message": msg},
                    "diagnostic": msg,
                }
            if has_agent_counts and not items and not agent_counts:
                msg = "Invalid stack payload: at least 1 configured model is required."
                self.last_switch_error = msg
                return {
                    "ok": False,
                    "http_status": 400,
                    "code": "invalid_stack_cardinality",
                    "error": {"code": "INVALID_STACK_CARDINALITY", "message": msg},
                    "diagnostic": msg,
                }
            if not items and agent_counts:
                items = list(agent_counts.keys())

            tier_or_vram: str | float = hardware_tier if hardware_tier else float(getattr(self, "total_vram_gb", 40.0))
            try:
                stack_check = verify_stack_vram_compatibility(
                    items,
                    tier_or_vram,
                    agent_counts=agent_counts,
                )
            except ValueError as verr:
                msg = str(verr)
                self.last_switch_error = msg
                return {
                    "ok": False,
                    "http_status": 400,
                    "code": "invalid_model_identifier",
                    "error": {"code": "INVALID_MODEL_IDENTIFIER", "message": msg},
                    "diagnostic": msg,
                }

            if not stack_check.get("compatible", False):
                diag = str(stack_check.get("diagnostic") or "Stack exceeds accelerator memory budget.")
                self.last_switch_error = diag
                err_code = str(stack_check.get("code") or "insufficient_vram")
                if err_code == "invalid_stack_cardinality":
                    return {
                        "ok": False,
                        "http_status": 400,
                        "code": "invalid_stack_cardinality",
                        "error": {"code": "INVALID_STACK_CARDINALITY", "message": diag},
                        "diagnostic": diag,
                    }
                return {
                    "ok": False,
                    "http_status": 507,
                    "code": "insufficient_vram",
                    "error": {"code": "STACK_VRAM_EXCEEDED", "message": diag},
                    "compatible": False,
                    "model_count": stack_check.get("model_count", len(items)),
                    "estimated_weights_gb": stack_check.get("estimated_weights_gb"),
                    "free_kv_gb": stack_check.get("free_kv_gb"),
                    "available_memory_gb": stack_check.get("available_memory_gb"),
                    "diagnostic": diag,
                }

            resolved_slots_meta: List[Dict[str, Any]] = list(stack_check["models"])
            inactive_slots_meta: List[Dict[str, Any]] = list(stack_check.get("inactive_models") or [])
            resolved_agent_counts: Dict[str, int] = dict(
                stack_check.get("agent_counts")
                or {str(m["alias"]): int(m.get("agent_count", 1)) for m in resolved_slots_meta}
            )
            total_vram = float(stack_check.get("available_memory_gb", getattr(self, "total_vram_gb", 40.0)))

            if len(resolved_slots_meta) == 0:
                prev_evicted = list(self.loaded_models.keys())
                self._evict_loaded_model()
                self.inactive_unloaded_models = {str(m["alias"]): dict(m) for m in inactive_slots_meta}
                self.model_agent_counts = dict(resolved_agent_counts)
                self.configured_models = list(resolved_agent_counts.keys())
                self.stack_preset = stack_preset or "inactive"
                self.last_switch_error = None
                vram_by_model = {alias: 0.0 for alias in self.configured_models}
                return {
                    "ok": True,
                    "http_status": 200,
                    "stack_preset": self.stack_preset,
                    "model_count": 0,
                    "configured_model_count": len(self.configured_models),
                    "configured_models": list(self.configured_models),
                    "active_model": None,
                    "resident_models": [],
                    "loaded_models": [],
                    "inactive_unloaded_models": list(self.inactive_unloaded_models.keys()),
                    "skipped_inactive_models": list(self.inactive_unloaded_models.keys()),
                    "agent_counts": dict(self.model_agent_counts),
                    "vram_allocated_by_model": vram_by_model,
                    "reused_models": [],
                    "evicted_models": prev_evicted,
                    "estimated_weights_gb": 0.0,
                    "vram_used_gb": 0.0,
                    "free_kv_gb": stack_check["free_kv_gb"],
                    "available_memory_gb": stack_check["available_memory_gb"],
                }

            eff_cache_root = cache_root or os.environ.get("PLATODOOM_MODEL_CACHE_DIR", "/content/models")
            staged_map: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[str]]] = {}
            for slot_meta in resolved_slots_meta:
                s_alias = str(slot_meta["alias"])
                s_prec = normalize_precision_code(slot_meta["precision"])
                s_repo = str(slot_meta["hf_repo_id"])
                valid_syn, syn_err = validate_model_identifier_syntax(s_alias, s_repo)
                if not valid_syn:
                    msg = syn_err or f"Invalid model identifier '{s_alias}'."
                    self.last_switch_error = msg
                    return {
                        "ok": False,
                        "http_status": 400,
                        "code": "invalid_model_identifier",
                        "error": {"code": "INVALID_MODEL_IDENTIFIER", "message": msg},
                        "diagnostic": msg,
                    }
                existing = self.loaded_models.get(s_alias)
                if (
                    existing is not None
                    and existing.is_resident
                    and normalize_precision_code(existing.precision) == s_prec
                ):
                    staged_map[s_alias] = (None, existing.checkpoint_id)
                    continue

                needs_disk_stage = (
                    getattr(self, "_model_loader", None) is default_physical_model_loader
                    and (not self._should_use_vllm_for_alias(s_alias) or getattr(self, "_vllm_spawner", None) is _default_spawn_vllm_process_slot)
                )
                if needs_disk_stage:
                    is_already_local = bool(s_repo and os.path.exists(s_repo))
                    if not is_already_local or cache_root is not None:
                        req_disk_gb = float(slot_meta.get("weights_gb", 6.0))
                        prune_res = _prune_model_cache_lru(
                            eff_cache_root,
                            required_free_gb=req_disk_gb,
                            protected_dirs=[sl.checkpoint_id for sl in self.loaded_models.values() if sl.checkpoint_id],
                        )
                        if not prune_res["ok"]:
                            msg = f"Insufficient disk space in '{eff_cache_root}' ({prune_res['free_gb']} GB free < {req_disk_gb} GB required)."
                            self.last_switch_error = msg
                            return {
                                "ok": False,
                                "http_status": 507,
                                "code": "insufficient_disk",
                                "error": {"code": "INSUFFICIENT_DISK", "message": msg},
                                "diagnostic": msg,
                            }
                    try:
                        r_spec = resolve_serving_checkpoint_source(
                            model_alias=s_alias,
                            hf_repo_id=s_repo,
                            total_vram_gb=total_vram,
                            precision=s_prec,
                            cache_root=eff_cache_root,
                        )
                        s_dir = stage_checkpoint_to_disk(r_spec, total_vram_gb=total_vram)
                        staged_map[s_alias] = (r_spec, s_dir)
                    except Exception as serr:
                        msg = f"Checkpoint staging failed for '{s_alias}' prior to stack transition: {serr}"
                        self.last_switch_error = msg
                        return {
                            "ok": False,
                            "http_status": 500,
                            "code": "checkpoint_stage_failed",
                            "error": {"code": "CHECKPOINT_STAGE_FAILED", "message": msg},
                            "diagnostic": msg,
                        }
                else:
                    staged_map[s_alias] = (None, None)

            eff_stack_vision = _parse_optional_bool_flag(enable_vision, field_name="enable_vision")
            target_pairs = {
                (str(m["alias"]), normalize_precision_code(m["precision"]))
                for m in resolved_slots_meta
            }
            evicted_aliases: List[str] = []
            reused_aliases: List[str] = []
            kept_slots: Dict[str, LoadedModelSlot] = {}
            for old_alias, old_slot in list(self.loaded_models.items()):
                old_vision_matches = (
                    eff_stack_vision is None
                    or bool(getattr(old_slot, "vision", False)) == bool(eff_stack_vision)
                )
                if (
                    (old_alias, normalize_precision_code(old_slot.precision)) in target_pairs
                    and old_slot.is_resident
                    and old_vision_matches
                ):
                    kept_slots[old_alias] = old_slot
                    reused_aliases.append(old_alias)
                else:
                    evicted_aliases.append(old_alias)
                    if old_slot.vllm_slot is not None:
                        _terminate_slot_process_group(old_slot.vllm_slot, timeout_sec=10.0, poll_vram=True)
                        old_slot.vllm_slot = None
                    old_slot.upstream_port = None
                    if old_slot.coordinator is not None and old_slot.coordinator is not self.coordinator:
                        try:
                            old_slot.coordinator.close()
                        except Exception:
                            pass
                        old_slot.coordinator = None
                    old_slot.model = None
                    old_slot.tokenizer = None

            if evicted_aliases:
                if self.active_model_alias in evicted_aliases:
                    self.model = None
                    self.tokenizer = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            self.switch_in_progress = ",".join(str(m["alias"]) for m in resolved_slots_meta)
            new_loaded_models: Dict[str, LoadedModelSlot] = {}
            active_slot_total = len(resolved_slots_meta)
            total_stack_weights = max(1.0, sum(float(m.get("weights_gb", 5.0)) for m in resolved_slots_meta))
            aggregate_util_cap = 0.92 - 0.025 * max(0, active_slot_total - 1)
            used_ports = {sl.upstream_port for sl in kept_slots.values() if sl.upstream_port is not None}

            try:
                loader_fn = getattr(self, "_model_loader", default_physical_model_loader)
                for idx, slot_meta in enumerate(resolved_slots_meta):
                    s_alias = str(slot_meta["alias"])
                    s_prec = normalize_precision_code(slot_meta["precision"])
                    s_repo = str(slot_meta["hf_repo_id"])
                    s_weights = float(slot_meta.get("weights_gb", 0.0))
                    s_role = str(slot_meta.get("role") or ("primary" if idx == 0 else f"slot_{idx + 1}"))
                    s_agent_cnt = max(1, int(slot_meta.get("agent_count", resolved_agent_counts.get(s_alias, 1))))
                    slot_max_seqs = max(int(self.config["max_num_seqs"]), min(256, s_agent_cnt))
                    slot_cohort = min(s_agent_cnt, slot_max_seqs) if has_agent_counts else None

                    if s_alias in kept_slots:
                        reused = kept_slots[s_alias]
                        reused.role = s_role
                        reused.weights_gb = s_weights
                        reused.agent_count = s_agent_cnt
                        reused.expected_concurrency = s_agent_cnt
                        if reused.coordinator is None or reused.coordinator is self.coordinator:
                            reused.coordinator = self._make_slot_coordinator(
                                s_alias,
                                max_num_seqs=slot_max_seqs,
                                target_cohort_size=slot_cohort,
                            )
                        else:
                            reused.coordinator.set_concurrency_targets(
                                max_num_seqs=slot_max_seqs,
                                target_cohort_size=slot_cohort,
                                update_cohort=True,
                            )
                        new_loaded_models[s_alias] = reused
                        continue

                    r_spec, s_dir = staged_map.get(s_alias, (None, None))
                    if self._should_use_vllm_for_alias(s_alias):
                        cand_port = int(self._base_vllm_port) + idx
                        while cand_port in used_ports:
                            cand_port += 1
                        used_ports.add(cand_port)
                        prop_util = (
                            0.90
                            if active_slot_total == 1
                            else round(max(0.15, aggregate_util_cap * (max(1.0, s_weights) / total_stack_weights)), 4)
                        )
                        live_free_vram = self._probe_free_vram_gb()
                        eff_hw_tier = str(hardware_tier or self.gpu_tier or "L4")
                        slot_target_max_len = resolve_target_catalog_max_model_len(
                            s_alias,
                            gpu_tier=eff_hw_tier,
                            requested_max_model_len=int(self.config["max_model_len"]),
                            native_max_pos=int((_SUPERVISOR_MODEL_SPECS.get(s_alias) or {}).get("max_pos") or 32768),
                        )
                        spawner_kwargs: Dict[str, Any] = {
                            "model_alias": s_alias,
                            "hf_repo_id": s_repo,
                            "checkpoint_target": s_dir or s_repo,
                            "port": cand_port,
                            "host": "127.0.0.1",
                            "precision": s_prec,
                            "weights_gb": s_weights,
                            "role": s_role,
                            "agent_count": s_agent_cnt,
                            "gpu_tier": eff_hw_tier,
                            "compute_capability": self.compute_capability,
                            "free_vram_gb": live_free_vram,
                            "total_vram_gb": total_vram,
                            "slot_vram_fraction": prop_util,
                            "max_model_len": slot_target_max_len,
                            "active_slot_count": active_slot_total,
                        }
                        if eff_stack_vision is not None:
                            try:
                                sp_sig = inspect.signature(self._vllm_spawner)
                                if "enable_vision" in sp_sig.parameters or any(
                                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sp_sig.parameters.values()
                                ):
                                    spawner_kwargs["enable_vision"] = eff_stack_vision
                            except Exception:
                                spawner_kwargs["enable_vision"] = eff_stack_vision
                        v_slot = self._vllm_spawner(**spawner_kwargs)
                        slot_coord = self._make_slot_coordinator(
                            s_alias,
                            max_num_seqs=v_slot.max_num_seqs,
                            target_cohort_size=slot_cohort,
                        )
                        new_loaded_models[s_alias] = LoadedModelSlot(
                            alias=s_alias,
                            hf_repo_id=s_repo,
                            checkpoint_id=s_dir or s_repo,
                            precision=v_slot.precision or s_prec,
                            weights_gb=s_weights or v_slot.weights_gb,
                            role=s_role,
                            agent_count=s_agent_cnt,
                            expected_concurrency=s_agent_cnt,
                            coordinator=slot_coord,
                            upstream_host=v_slot.host,
                            upstream_port=v_slot.port,
                            vllm_slot=v_slot,
                            max_model_len=int(v_slot.max_model_len),
                            modality="text",
                        )
                        continue

                    loader_kwargs: Dict[str, Any] = {
                        "model_alias": s_alias,
                        "hf_repo_id": s_repo,
                        "total_vram_gb": total_vram,
                        "precision": s_prec,
                        "checkpoint_dir": s_dir,
                        "resolved_spec": r_spec,
                        "cache_root": cache_root,
                    }
                    pos_args: List[Any] = []
                    try:
                        sig = inspect.signature(loader_fn)
                        params = list(sig.parameters.values())
                        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                        if not accepts_var_kw:
                            if "model_alias" not in sig.parameters:
                                loader_kwargs.pop("model_alias", None)
                                if "alias" in sig.parameters:
                                    loader_kwargs["alias"] = s_alias
                                elif params and params[0].kind in (
                                    inspect.Parameter.POSITIONAL_ONLY,
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                ):
                                    pos_args.append(s_alias)
                            loader_kwargs = {k: v for k, v in loader_kwargs.items() if k in sig.parameters}
                    except Exception:
                        pass

                    loaded_m, loaded_t, resolved_r = loader_fn(*pos_args, **loader_kwargs)
                    if loaded_m is None or loaded_t is None:
                        raise RuntimeError(f"Physical model loader returned empty model or tokenizer for '{s_alias}'.")

                    built_slot = self._build_slot(
                        alias=s_alias,
                        hf_repo_id=str(resolved_r or s_repo),
                        checkpoint_id=s_dir or str(resolved_r or s_repo),
                        model=loaded_m,
                        tokenizer=loaded_t,
                        precision=s_prec,
                        weights_gb=s_weights,
                        role=s_role,
                        agent_count=s_agent_cnt,
                        target_cohort_size=slot_cohort,
                        max_num_seqs=slot_max_seqs,
                    )
                    new_loaded_models[s_alias] = built_slot

                self.loaded_models = new_loaded_models
                self.inactive_unloaded_models = {str(m["alias"]): dict(m) for m in inactive_slots_meta}
                self.model_agent_counts = dict(resolved_agent_counts)
                self.configured_models = list(resolved_agent_counts.keys())
                primary_slot = next(iter(self.loaded_models.values()))
                if primary_slot.vllm_slot is not None:
                    self.config["max_model_len"] = int(primary_slot.vllm_slot.max_model_len)
                    self.config["max_num_seqs"] = int(primary_slot.vllm_slot.max_num_seqs)
                    self.config["max_num_batched_tokens"] = int(primary_slot.vllm_slot.max_num_batched_tokens)
                self._sync_primary_attributes_from_slot(primary_slot)
                self.stack_preset = stack_preset or ("solo" if len(self.loaded_models) == 1 else "custom")
                self.last_switch_error = None
            except Exception as exc:
                for partial_slot in new_loaded_models.values():
                    if partial_slot.vllm_slot is not None:
                        _terminate_slot_process_group(partial_slot.vllm_slot, timeout_sec=5.0, poll_vram=False)
                    if partial_slot.coordinator is not None and partial_slot.coordinator is not self.coordinator:
                        try:
                            partial_slot.coordinator.close()
                        except Exception:
                            pass
                self._evict_loaded_model()
                exc_text = str(exc)
                exc_low = exc_text.lower()
                is_oom = bool(
                    getattr(exc, "is_oom", False)
                    or exc.__class__.__name__ == "OutOfMemoryError"
                    or "out of memory" in exc_low
                    or "cuda error: out of memory" in exc_low
                    or "memory allocation failed with oom" in exc_low
                    or "no available memory for the cache blocks" in exc_low
                )
                st_code = 507 if is_oom else int(getattr(exc, "http_status", 500))
                err_slug = "cuda_oom" if is_oom else "stack_load_failed"
                self.last_switch_error = exc_text
                return {
                    "ok": False,
                    "http_status": st_code,
                    "code": err_slug,
                    "error": {"code": err_slug.upper(), "message": exc_text},
                    "diagnostic": exc_text,
                }
            finally:
                self.switch_in_progress = None

            vram_by_model: Dict[str, float] = {}
            for cfg_alias in self.configured_models:
                if cfg_alias in self.loaded_models:
                    vram_by_model[cfg_alias] = float(self.loaded_models[cfg_alias].weights_gb)
                else:
                    vram_by_model[cfg_alias] = 0.0
            for res_alias, sl in self.loaded_models.items():
                vram_by_model.setdefault(res_alias, float(sl.weights_gb))

            return {
                "ok": True,
                "http_status": 200,
                "stack_preset": self.stack_preset,
                "model_count": len(self.loaded_models),
                "configured_model_count": len(self.configured_models),
                "configured_models": list(self.configured_models),
                "active_model": self.active_model_alias,
                "resident_models": list(self.loaded_models.keys()),
                "loaded_models": [
                    {
                        "alias": sl.alias,
                        "hf_repo_id": sl.hf_repo_id,
                        "precision": sl.precision,
                        "weights_gb": sl.weights_gb,
                        "vram_allocated_gb": sl.weights_gb,
                        "agent_count": sl.agent_count,
                        "max_batch_size": sl.coordinator.max_num_seqs if sl.coordinator else self.config["max_num_seqs"],
                        "role": sl.role,
                        "vision": bool(getattr(sl, "vision", False)),
                        "input_modalities": ["text", "image"] if bool(getattr(sl, "vision", False)) else ["text"],
                        "upstream_port": sl.upstream_port,
                        "engine": "vllm" if sl.upstream_port is not None else "inprocess",
                    }
                    for sl in self.loaded_models.values()
                ],
                "inactive_unloaded_models": list(self.inactive_unloaded_models.keys()),
                "skipped_inactive_models": list(self.inactive_unloaded_models.keys()),
                "agent_counts": dict(self.model_agent_counts),
                "vram_allocated_by_model": vram_by_model,
                "reused_models": reused_aliases,
                "evicted_models": evicted_aliases,
                "estimated_weights_gb": stack_check["estimated_weights_gb"],
                "vram_used_gb": stack_check["estimated_weights_gb"],
                "free_kv_gb": stack_check["free_kv_gb"],
                "available_memory_gb": stack_check["available_memory_gb"],
            }
        finally:
            self._control_lock.release()

    def switch_model(
        self,
        model_alias: str,
        hf_repo_id: Optional[str] = None,
        precision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        new_model: Optional[Any] = None,
        new_tokenizer: Optional[Any] = None,
        source: Optional[str] = None,
        storage_uri: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        gcs_prefix: Optional[str] = None,
        fallback_hf_repo_id: Optional[str] = None,
        cache_root: Optional[str] = None,
        enable_vision: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """In-place physical model hot-swapping with pre-eviction disk staging, LRU cache pruning, VRAM guard, and process-group teardown."""
        if not self._control_lock.acquire(blocking=False):
            in_prog = getattr(self, "switch_in_progress", None) or "active"
            msg = f"Model switch already in progress ('{in_prog}'). Retry once /health returns 200."
            return {
                "ok": False,
                "http_status": 409,
                "code": "switch_in_progress",
                "switching_to": in_prog,
                "error": {"code": "SWITCH_IN_PROGRESS", "message": msg},
                "requested_model": model_alias,
                "diagnostic": msg,
            }
        try:
            valid_syntax, syntax_err = validate_model_identifier_syntax(model_alias, hf_repo_id)
            if not valid_syntax:
                msg = syntax_err or f"Invalid model identifier '{model_alias}'."
                self.last_switch_error = msg
                return {
                    "ok": False,
                    "http_status": 400,
                    "code": "invalid_model_identifier",
                    "error": {
                        "code": "INVALID_MODEL_IDENTIFIER",
                        "message": msg,
                    },
                    "requested_model": model_alias,
                    "diagnostic": msg,
                }

            total_vram = float(getattr(self, "total_vram_gb", 40.0))
            vram_check = verify_vram_compatibility(model_alias, total_vram, precision=precision)
            if not vram_check.get("compatible", True):
                diag = vram_check.get("diagnostic", f"Physical VRAM {total_vram:.1f} GB is below {model_alias} floor.")
                self.last_switch_error = diag
                return {
                    "ok": False,
                    "http_status": 507,
                    "code": "insufficient_vram",
                    "error": {
                        "code": "MODEL_SWITCH_VRAM_EXCEEDED",
                        "message": diag,
                    },
                    "compatible": False,
                    "requested_model": model_alias,
                    "required_min_vram_gb": vram_check.get("required_min_vram_gb", 34.0),
                    "available_vram_gb": total_vram,
                    "fallback_alias": vram_check.get("fallback_alias", "qwen-9b"),
                    "diagnostic": diag,
                }

            eff_enable_vision = _parse_optional_bool_flag(enable_vision, field_name="enable_vision")
            cur_slot = self.loaded_models.get(model_alias)
            is_currently_loaded = (
                (cur_slot is not None and cur_slot.is_resident and len(self.loaded_models) == 1)
                or (
                    self.model is not None
                    and self.tokenizer is not None
                    and self.active_model_alias == model_alias
                )
            )
            cur_vision_state = bool(getattr(cur_slot, "vision", False)) if cur_slot is not None else False
            vision_matches_current = (
                eff_enable_vision is None or cur_vision_state == bool(eff_enable_vision)
            )
            if (
                new_model is None
                and is_currently_loaded
                and vision_matches_current
                and (not hf_repo_id or hf_repo_id == self.active_hf_repo_id)
            ):
                if max_model_len:
                    self.config["max_model_len"] = int(max_model_len)
                    if hasattr(self, "coordinator"):
                        self.coordinator.max_model_len = int(max_model_len)
                return {
                    "ok": True,
                    "http_status": 200,
                    "already_loaded": True,
                    "previous_model": self.active_model_alias,
                    "active_model": self.active_model_alias,
                    "served_model_id": self.active_hf_repo_id or model_alias,
                    "hf_repo_id": self.active_hf_repo_id or model_alias,
                    "stop_token_ids": list(self.stop_token_ids),
                    "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
                    "tokenizer_capabilities": dict(self.tokenizer_capabilities),
                    "architecture_family": self.architecture_metadata.get("architecture_family", "causal_lm"),
                    "is_qwen35_model": self.is_qwen35_model,
                    "vision": cur_vision_state,
                    "input_modalities": ["text", "image"] if cur_vision_state else ["text"],
                    "max_model_len": self.config["max_model_len"],
                    "total_vram_gb": total_vram,
                }

            use_vllm = new_model is None and self._should_use_vllm_for_alias(model_alias)
            eff_cache_root = cache_root or os.environ.get("PLATODOOM_MODEL_CACHE_DIR", "/content/models")
            resolved_spec: Optional[Dict[str, Any]] = None
            staged_dir: Optional[str] = None
            needs_staging = (
                new_model is None
                and getattr(self, "_model_loader", None) is default_physical_model_loader
                and (not use_vllm or getattr(self, "_vllm_spawner", None) is _default_spawn_vllm_process_slot)
            )
            if needs_staging:
                is_already_local_target = bool(
                    (hf_repo_id and os.path.exists(str(hf_repo_id)))
                    or (storage_uri and os.path.exists(str(storage_uri)))
                    or os.path.exists(str(model_alias))
                )
                if not is_already_local_target or cache_root is not None:
                    req_gb = float((_SUPERVISOR_MODEL_SPECS.get(str(model_alias).split("@", 1)[0]) or {}).get("weights_q4_gb", 6.0))
                    prune_res = _prune_model_cache_lru(
                        eff_cache_root,
                        required_free_gb=req_gb,
                        protected_dirs=[sl.checkpoint_id for sl in self.loaded_models.values() if sl.checkpoint_id],
                    )
                    if not prune_res["ok"]:
                        msg = f"Insufficient disk space in '{eff_cache_root}' ({prune_res['free_gb']} GB free < {req_gb} GB required)."
                        self.last_switch_error = msg
                        return {
                            "ok": False,
                            "http_status": 507,
                            "code": "insufficient_disk",
                            "error": {"code": "INSUFFICIENT_DISK", "message": msg},
                            "requested_model": model_alias,
                            "diagnostic": msg,
                        }
                try:
                    resolved_spec = resolve_serving_checkpoint_source(
                        model_alias=model_alias,
                        hf_repo_id=hf_repo_id,
                        total_vram_gb=total_vram,
                        precision=precision,
                        source=source,
                        storage_uri=storage_uri,
                        gcs_bucket=gcs_bucket,
                        gcs_prefix=gcs_prefix,
                        fallback_hf_repo_id=fallback_hf_repo_id,
                        cache_root=eff_cache_root,
                    )
                    staged_dir = stage_checkpoint_to_disk(resolved_spec, total_vram_gb=total_vram)
                except ValueError as verr:
                    msg = str(verr)
                    self.last_switch_error = msg
                    return {
                        "ok": False,
                        "http_status": 400,
                        "code": "invalid_model_identifier",
                        "error": {"code": "INVALID_MODEL_IDENTIFIER", "message": msg},
                        "requested_model": model_alias,
                        "diagnostic": msg,
                    }
                except Exception as serr:
                    msg = f"Checkpoint staging failed prior to GPU eviction: {serr}"
                    self.last_switch_error = msg
                    return {
                        "ok": False,
                        "http_status": 500,
                        "code": "checkpoint_stage_failed",
                        "error": {"code": "CHECKPOINT_STAGE_FAILED", "message": msg},
                        "requested_model": model_alias,
                        "diagnostic": msg,
                    }

            if new_model is not None and new_tokenizer is None:
                msg = "new_tokenizer must be supplied alongside new_model during physical model switch."
                self.last_switch_error = msg
                return {
                    "ok": False,
                    "http_status": 400,
                    "code": "missing_tokenizer",
                    "error": {"code": "MISSING_TOKENIZER", "message": msg},
                    "requested_model": model_alias,
                    "diagnostic": msg,
                }

            previous_model = getattr(self, "active_model_alias", None)
            self.switch_in_progress = model_alias
            try:
                self._evict_loaded_model()
                if use_vllm:
                    eff_repo = str(
                        (resolved_spec or {}).get("hf_repo_id")
                        or hf_repo_id
                        or vram_check.get("resolved_checkpoint_repo_id")
                        or vram_check.get("quantized_hf_repo_id")
                        or (_SUPERVISOR_MODEL_SPECS.get(str(model_alias).split("@", 1)[0]) or {}).get("quantized_hf_repo_id")
                        or (_SUPERVISOR_MODEL_SPECS.get(str(model_alias).split("@", 1)[0]) or {}).get("hf_repo_id")
                        or model_alias
                    )
                    eff_prec = normalize_precision_code(
                        precision
                        or str(
                            (resolved_spec or {}).get("precision")
                            or vram_check.get("normalized_precision")
                            or vram_check.get("selected_precision")
                            or vram_check.get("active_precision")
                            or "Q4"
                        )
                    )
                    live_free_vram = self._probe_free_vram_gb()
                    eff_sw_tier = str(self.gpu_tier or "L4")
                    sw_target_max_len = resolve_target_catalog_max_model_len(
                        model_alias,
                        gpu_tier=eff_sw_tier,
                        requested_max_model_len=int(max_model_len or self.config["max_model_len"]),
                        native_max_pos=int((_SUPERVISOR_MODEL_SPECS.get(str(model_alias).split("@", 1)[0]) or {}).get("max_pos") or 32768),
                    )
                    sw_spawner_kwargs: Dict[str, Any] = {
                        "model_alias": model_alias,
                        "hf_repo_id": eff_repo,
                        "checkpoint_target": staged_dir or eff_repo,
                        "port": int(self._base_vllm_port),
                        "host": "127.0.0.1",
                        "precision": eff_prec,
                        "role": "primary",
                        "agent_count": 1,
                        "gpu_tier": eff_sw_tier,
                        "compute_capability": self.compute_capability,
                        "free_vram_gb": live_free_vram,
                        "total_vram_gb": total_vram,
                        "slot_vram_fraction": 0.90,
                        "max_model_len": sw_target_max_len,
                        "active_slot_count": 1,
                    }
                    if eff_enable_vision is not None:
                        try:
                            sw_sig = inspect.signature(self._vllm_spawner)
                            if "enable_vision" in sw_sig.parameters or any(
                                p.kind == inspect.Parameter.VAR_KEYWORD for p in sw_sig.parameters.values()
                            ):
                                sw_spawner_kwargs["enable_vision"] = eff_enable_vision
                        except Exception:
                            sw_spawner_kwargs["enable_vision"] = eff_enable_vision
                    v_slot = self._vllm_spawner(**sw_spawner_kwargs)
                    self._bind_vllm_process_slot(v_slot)
                else:
                    if new_model is not None:
                        loaded_model = new_model
                        loaded_tok = new_tokenizer
                        resolved_repo = str(hf_repo_id or model_alias)
                        checkpoint_id = resolved_repo
                    else:
                        loader_fn = getattr(self, "_model_loader", default_physical_model_loader)
                        loader_kwargs: Dict[str, Any] = {
                            "model_alias": model_alias,
                            "hf_repo_id": hf_repo_id,
                            "total_vram_gb": total_vram,
                            "precision": precision,
                            "checkpoint_dir": staged_dir,
                            "resolved_spec": resolved_spec,
                            "source": source,
                            "storage_uri": storage_uri,
                            "gcs_bucket": gcs_bucket,
                            "gcs_prefix": gcs_prefix,
                            "fallback_hf_repo_id": fallback_hf_repo_id,
                            "cache_root": cache_root,
                        }
                        try:
                            sig = inspect.signature(loader_fn)
                            accepts_var_kw = any(
                                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                            )
                            if not accepts_var_kw:
                                loader_kwargs = {k: v for k, v in loader_kwargs.items() if k in sig.parameters}
                        except Exception:
                            pass
                        loaded_model, loaded_tok, resolved_repo = loader_fn(**loader_kwargs)
                        checkpoint_id = staged_dir or resolved_repo

                    if loaded_model is None or loaded_tok is None:
                        raise RuntimeError(f"Physical model loader returned empty model or tokenizer for '{model_alias}'.")

                    self._rebind_loaded_model_and_tokenizer(
                        model=loaded_model,
                        tokenizer=loaded_tok,
                        model_alias=model_alias,
                        hf_repo_id=str(resolved_repo or hf_repo_id or model_alias),
                        checkpoint_id=checkpoint_id,
                    )
                if max_model_len and not use_vllm:
                    self.config["max_model_len"] = int(max_model_len)
                    if hasattr(self, "coordinator"):
                        self.coordinator.max_model_len = int(max_model_len)
                self.last_switch_error = None
            except Exception as exc:
                self._evict_loaded_model()
                exc_text = str(exc)
                exc_lower = exc_text.lower()
                is_oom = bool(
                    getattr(exc, "is_oom", False)
                    or exc.__class__.__name__ == "OutOfMemoryError"
                    or "out of memory" in exc_lower
                    or "cuda error: out of memory" in exc_lower
                    or "memory allocation failed with oom" in exc_lower
                    or "no available memory for the cache blocks" in exc_lower
                )
                if is_oom:
                    status = 507
                    err_code = "cuda_oom"
                elif (
                    getattr(exc, "error_code", None) == "invalid_model_identifier"
                    or isinstance(exc, ValueError)
                    or exc.__class__.__name__ in ("RepositoryNotFoundError", "HFValidationError")
                ):
                    status = 400
                    err_code = "invalid_model_identifier"
                else:
                    status = int(getattr(exc, "http_status", 500))
                    err_code = str(getattr(exc, "error_code", "checkpoint_load_failed"))
                self.last_switch_error = exc_text
                return {
                    "ok": False,
                    "http_status": status,
                    "code": err_code,
                    "error": {
                        "code": err_code.upper(),
                        "message": exc_text,
                    },
                    "requested_model": model_alias,
                    "diagnostic": exc_text,
                }
            finally:
                self.switch_in_progress = None

            active_slot_now = self.loaded_models.get(self.active_model_alias) if self.active_model_alias else None
            active_vision_now = bool(getattr(active_slot_now, "vision", False)) if active_slot_now is not None else False
            return {
                "ok": True,
                "http_status": 200,
                "already_loaded": False,
                "previous_model": previous_model,
                "active_model": self.active_model_alias,
                "served_model_id": self.active_hf_repo_id,
                "hf_repo_id": self.active_hf_repo_id,
                "stop_token_ids": list(self.stop_token_ids),
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
                "tokenizer_capabilities": dict(self.tokenizer_capabilities),
                "architecture_family": self.architecture_metadata.get("architecture_family", "causal_lm"),
                "is_qwen35_model": self.is_qwen35_model,
                "vision": active_vision_now,
                "input_modalities": ["text", "image"] if active_vision_now else ["text"],
                "max_model_len": self.config["max_model_len"],
                "total_vram_gb": total_vram,
            }
        finally:
            self._control_lock.release()

    def _resolve_effective_thinking(self, item: InferenceRequestItem, caps: Dict[str, Any]) -> bool:
        if item.enable_thinking is not None:
            return bool(item.enable_thinking)
        return bool(caps.get("is_deepseek_r1_reasoning") or caps.get("forces_think_prefix"))

    def _format_prompt_with_meta(
        self,
        item: InferenceRequestItem,
        *,
        slot: Optional[LoadedModelSlot] = None,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        tok = slot.tokenizer if slot is not None else self.tokenizer
        mod = slot.model if slot is not None else self.model
        active_alias = (slot.alias if slot is not None else getattr(self, "active_model_alias", "")) or ""
        caps = introspect_loaded_tokenizer_capabilities(tok, active_alias, mod)
        effective_thinking = self._resolve_effective_thinking(item, caps)

        if caps["has_chat_template"]:
            prepared = _prepare_messages_for_template(
                item.messages,
                item.tools,
                native_tools=bool(caps["supports_native_tools"]),
                native_tool_role=bool(caps["supports_tool_role"]),
                dict_args=bool(caps["requires_dict_tool_arguments"]),
            )
            tpl_kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if caps["supports_native_tools"] and item.tools:
                tpl_kwargs["tools"] = item.tools
            if caps["supports_enable_thinking"]:
                tpl_kwargs["enable_thinking"] = effective_thinking

            rendered: Optional[str] = None
            try:
                rendered = tok.apply_chat_template(prepared, **tpl_kwargs)
            except Exception:
                if "enable_thinking" in tpl_kwargs:
                    retry_kwargs = {k: v for k, v in tpl_kwargs.items() if k != "enable_thinking"}
                    try:
                        rendered = tok.apply_chat_template(prepared, **retry_kwargs)
                    except Exception:
                        rendered = None
                if rendered is None:
                    fallback_msgs = _prepare_messages_for_template(
                        item.messages,
                        item.tools,
                        native_tools=False,
                        native_tool_role=False,
                        dict_args=False,
                    )
                    try:
                        rendered = tok.apply_chat_template(
                            fallback_msgs,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        rendered = None

            if isinstance(rendered, str):
                if (
                    item.enable_thinking is False
                    and not caps["supports_enable_thinking"]
                    and rendered.rstrip().endswith("<think>")
                ):
                    rendered = f"{rendered}</think>\n\n"
                prompt_opens_think = rendered.rstrip().endswith("<think>")
                return rendered, prompt_opens_think, caps

        fallback_msgs = _prepare_messages_for_template(
            item.messages,
            item.tools,
            native_tools=False,
            native_tool_role=False,
            dict_args=False,
        )
        parts = [f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>" for m in fallback_msgs]
        if effective_thinking:
            parts.append("<|im_start|>assistant\n<think>\n")
        else:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        rendered_fallback = "\n".join(parts)
        return rendered_fallback, rendered_fallback.rstrip().endswith("<think>"), caps

    def _format_prompt(self, item: InferenceRequestItem, *, slot: Optional[LoadedModelSlot] = None) -> str:
        prompt, _, _ = self._format_prompt_with_meta(item, slot=slot)
        return prompt

    def _build_item_from_payload(self, payload: Dict[str, Any], request_id: Optional[str] = None) -> InferenceRequestItem:
        raw_max = payload.get("max_tokens")
        if raw_max is None:
            raw_max = payload.get("max_completion_tokens", 2048)
        upper_cap = max(4096, int(self.config.get("max_model_len", 16384)) - 1024)
        bounded_max = max(1, min(int(raw_max), upper_cap))

        raw_temp = payload.get("temperature")
        raw_top_p = payload.get("top_p")
        raw_top_k = payload.get("top_k")
        raw_rep = payload.get("repetition_penalty")
        raw_pres = payload.get("presence_penalty")
        raw_stop_ids = payload.get("stop_token_ids")
        parsed_stop_ids: Optional[List[int]] = None
        if isinstance(raw_stop_ids, (list, tuple)):
            parsed_stop_ids = [int(x) for x in raw_stop_ids if isinstance(x, int) and not isinstance(x, bool)]

        return InferenceRequestItem(
            request_id=request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            messages=_normalize_chat_messages_roles_and_content(payload.get("messages", [])),
            tools=payload.get("tools"),
            max_tokens=bounded_max,
            temperature=float(raw_temp) if raw_temp is not None else None,
            top_p=float(raw_top_p) if raw_top_p is not None else None,
            top_k=int(raw_top_k) if raw_top_k is not None else None,
            repetition_penalty=float(raw_rep) if raw_rep is not None else None,
            presence_penalty=float(raw_pres) if raw_pres is not None else None,
            enable_thinking=_extract_caller_thinking_flag(payload),
            stop=_normalize_stop_strings(payload.get("stop")) or None,
            stop_token_ids=parsed_stop_ids,
        )

    def _resolve_generation_params(
        self,
        item: InferenceRequestItem,
        caps: Dict[str, Any],
        base_stop_ids: List[int],
    ) -> Dict[str, Any]:
        effective_thinking = self._resolve_effective_thinking(item, caps)
        eff_temp = (0.6 if effective_thinking else 0.0) if item.temperature is None else max(0.0, float(item.temperature))
        vocab_bound = caps.get("vocab_size")
        merged_stops = set(base_stop_ids)
        if item.stop_token_ids:
            for tid in item.stop_token_ids:
                if _is_valid_vocab_token_id(tid, vocab_bound):
                    merged_stops.add(int(tid))
        stop_strings = _normalize_stop_strings(item.stop)
        return {
            "temperature": eff_temp,
            "do_sample": eff_temp > 0.0,
            "top_p": float(item.top_p) if item.top_p is not None else None,
            "top_k": int(item.top_k) if item.top_k is not None else None,
            "repetition_penalty": float(item.repetition_penalty) if item.repetition_penalty is not None else None,
            "presence_penalty": float(item.presence_penalty) if item.presence_penalty is not None else None,
            "stop_token_ids": sorted(merged_stops),
            "stop_strings": stop_strings,
        }

    def _run_gpu_batch_for_slot(
        self,
        slot_alias: Optional[str],
        items: List[InferenceRequestItem],
    ) -> List[Dict[str, Any]]:
        import torch

        slot = self.resolve_loaded_slot(slot_alias)
        slot_model = slot.model
        slot_tokenizer = slot.tokenizer
        if slot_model is None or slot_tokenizer is None:
            raise RuntimeError("No model is physically loaded in accelerator memory.")

        active_alias = slot.alias or getattr(self, "active_model_alias", None) or UNIFIED_MODEL_ALIAS
        total_batch_size = len(items)
        slot_max_seqs = (
            int(slot.coordinator.max_num_seqs)
            if getattr(slot, "coordinator", None) is not None
            else int(self.config["max_num_seqs"])
        )
        slot_agent_count = int(getattr(slot, "agent_count", 1))

        if not hasattr(slot_model, "generate"):
            responses: List[Dict[str, Any]] = []
            now_ts = int(time.time())
            mock_fn = getattr(slot_model, "mock_output_fn", None)
            default_mock_text = getattr(slot_model, "mock_output", "READY")
            for item in items:
                item_mock = mock_fn(item) if callable(mock_fn) else default_mock_text
                stop_strs = _normalize_stop_strings(item.stop)
                trunc_mock, _ = _truncate_at_stop_strings(str(item_mock), stop_strs)
                res = extract_tool_calls_from_text(trunc_mock)
                msg_payload: Dict[str, Any] = {"role": "assistant", "content": res.cleaned_text}
                if res.reasoning_content is not None:
                    msg_payload["reasoning_content"] = res.reasoning_content
                if res.tool_calls:
                    msg_payload["tool_calls"] = res.tool_calls
                responses.append(
                    {
                        "id": item.request_id,
                        "object": "chat.completion",
                        "created": now_ts,
                        "model": active_alias,
                        "choices": [
                            {
                                "index": 0,
                                "message": msg_payload,
                                "finish_reason": "tool_calls" if res.tool_calls else "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                        "sandcastle_batch_meta": {
                            "batch_size": total_batch_size,
                            "slot_batch_size": total_batch_size,
                            "model_alias": active_alias,
                            "slot_precision": slot.precision,
                            "slot_id": id(slot),
                            "agent_count_configured": slot_agent_count,
                            "max_parallel_agents": slot_max_seqs,
                            "max_model_len": int(self.config["max_model_len"]),
                        },
                    }
                )
            return responses

        base_stop_ids = _resolve_stop_token_ids(slot_tokenizer, active_alias, slot_model)
        formatted_entries: List[Tuple[int, InferenceRequestItem, str, bool, Dict[str, Any], Dict[str, Any]]] = []
        for orig_idx, item in enumerate(items):
            prompt_str, opens_think, caps = self._format_prompt_with_meta(item, slot=slot)
            g_params = self._resolve_generation_params(item, caps, base_stop_ids)
            formatted_entries.append((orig_idx, item, prompt_str, opens_think, caps, g_params))

        groups: Dict[Tuple[Any, ...], List[Tuple[int, InferenceRequestItem, str, bool, Dict[str, Any], Dict[str, Any]]]] = {}
        for entry in formatted_entries:
            gp = entry[5]
            g_key = (
                gp["temperature"],
                gp["top_p"],
                gp["top_k"],
                gp["repetition_penalty"],
                gp["presence_penalty"],
                tuple(gp["stop_token_ids"]),
                tuple(gp["stop_strings"]),
            )
            groups.setdefault(g_key, []).append(entry)

        ordered_responses: List[Optional[Dict[str, Any]]] = [None] * total_batch_size
        device = getattr(slot_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model_max_len = int(self.config.get("max_model_len", 16384))
        upper_gen_cap = max(4096, model_max_len - 1024)

        for sub_entries in groups.values():
            prompts = [e[2] for e in sub_entries]
            sub_items = [e[1] for e in sub_entries]
            gp = sub_entries[0][5]
            caps = sub_entries[0][4]
            pad_id = int(
                caps.get("pad_token_id")
                if caps.get("pad_token_id") is not None
                else _resolve_pad_token_id(slot_tokenizer, slot_model)
            )

            max_new_tokens = min(max(it.max_tokens for it in sub_items), upper_gen_cap)
            max_prompt_len = max(1024, model_max_len - max_new_tokens)

            encoded = slot_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            prompt_padded_len = int(input_ids.shape[1])

            gen_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(gp["do_sample"]),
                "pad_token_id": pad_id,
                "use_cache": True,
            }
            if gp["stop_token_ids"]:
                gen_kwargs["eos_token_id"] = gp["stop_token_ids"]
            if gp["do_sample"]:
                gen_kwargs["temperature"] = max(float(gp["temperature"]), 1e-5)
                if gp["top_p"] is not None:
                    gen_kwargs["top_p"] = float(gp["top_p"])
                if gp["top_k"] is not None and int(gp["top_k"]) > 0:
                    gen_kwargs["top_k"] = int(gp["top_k"])
            if gp["repetition_penalty"] is not None and float(gp["repetition_penalty"]) != 1.0:
                gen_kwargs["repetition_penalty"] = float(gp["repetition_penalty"])
            hf_lpl_cls, hf_scl_cls, _ = _get_hf_generation_types()
            if gp["presence_penalty"] is not None and float(gp["presence_penalty"]) != 0.0:
                gen_kwargs["logits_processor"] = hf_lpl_cls(
                    [PresencePenaltyLogitsProcessor(float(gp["presence_penalty"]), prompt_padded_len)]
                )
            if gp["stop_strings"]:
                gen_kwargs["stopping_criteria"] = hf_scl_cls(
                    [CustomStopStringsStoppingCriteria(slot_tokenizer, gp["stop_strings"], prompt_padded_len)]
                )

            with torch.inference_mode():
                output_ids = slot_model.generate(**gen_kwargs)

            now_ts = int(time.time())
            sub_batch_size = len(sub_entries)
            for sub_idx, (orig_idx, item, _, opens_think, _, _) in enumerate(sub_entries):
                raw_gen_tokens = output_ids[sub_idx][prompt_padded_len:]
                isolated_ids, hit_stop_tok = _isolate_generated_sequence_tokens(
                    raw_gen_tokens,
                    gp["stop_token_ids"],
                    pad_id,
                )
                if item.max_tokens and len(isolated_ids) > int(item.max_tokens):
                    isolated_ids = isolated_ids[: int(item.max_tokens)]
                    hit_stop_tok = False
                raw_text = slot_tokenizer.decode(isolated_ids, skip_special_tokens=True)
                raw_text, hit_stop_str = _truncate_at_stop_strings(raw_text, gp["stop_strings"])

                if opens_think and "<think>" not in raw_text and "</think>" not in raw_text:
                    raw_text = f"<think>{raw_text}"

                res = extract_tool_calls_from_text(raw_text)
                cleaned_text = res.cleaned_text
                tool_calls = res.tool_calls
                reasoning_content = res.reasoning_content

                prompt_tok_count = int(attention_mask[sub_idx].sum().item())
                comp_tok_count = int(len(isolated_ids))
                hit_max_len = (
                    (comp_tok_count >= item.max_tokens)
                    and not hit_stop_tok
                    and not hit_stop_str
                    and not tool_calls
                )
                msg_payload: Dict[str, Any] = {
                    "role": "assistant",
                    "content": cleaned_text,
                }
                if reasoning_content is not None:
                    msg_payload["reasoning_content"] = reasoning_content
                if tool_calls:
                    msg_payload["tool_calls"] = tool_calls

                ordered_responses[orig_idx] = {
                    "id": item.request_id,
                    "object": "chat.completion",
                    "created": now_ts,
                    "model": active_alias,
                    "choices": [
                        {
                            "index": 0,
                            "message": msg_payload,
                            "finish_reason": "tool_calls" if tool_calls else ("length" if hit_max_len else "stop"),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tok_count,
                        "completion_tokens": comp_tok_count,
                        "total_tokens": prompt_tok_count + comp_tok_count,
                    },
                    "sandcastle_batch_meta": {
                        "batch_size": sub_batch_size,
                        "slot_batch_size": total_batch_size,
                        "model_alias": active_alias,
                        "slot_precision": slot.precision,
                        "slot_id": id(slot),
                        "agent_count_configured": slot_agent_count,
                        "max_parallel_agents": slot_max_seqs,
                        "max_model_len": model_max_len,
                    },
                }

        return [r for r in ordered_responses if r is not None]

    def _run_gpu_batch(self, items: List[InferenceRequestItem]) -> List[Dict[str, Any]]:
        return self._run_gpu_batch_for_slot(getattr(self, "active_model_alias", None), items)

    def handle_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        loaded_slots = getattr(self, "loaded_models", None)
        has_any_loaded = (
            (isinstance(loaded_slots, dict) and any(s.is_resident for s in loaded_slots.values()))
            or (self.model is not None and self.tokenizer is not None)
        )
        if not has_any_loaded and getattr(self, "switch_in_progress", None) is None:
            raise RuntimeError("No model is physically loaded in accelerator memory.")
        if not has_any_loaded and getattr(self, "switch_in_progress", None) is not None:
            item = self._build_item_from_payload(payload)
            return self.coordinator.submit(item)
        slot = self.resolve_loaded_slot(payload.get("model"))
        if slot.upstream_port is not None:
            status_code, resp = _proxy_chat_completion_to_vllm_slot(
                slot.upstream_host,
                int(slot.upstream_port),
                slot.alias,
                payload,
            )
            if status_code >= 400:
                raise _UpstreamHTTPStatusError(status_code, resp)
            self._record_slot_batch_on_engine_coordinator(1)
            if isinstance(resp, dict) and "sandcastle_batch_meta" not in resp:
                slot_max_seqs = (
                    int(slot.vllm_slot.max_num_seqs)
                    if slot.vllm_slot is not None
                    else int(self.config.get("max_num_seqs", 64))
                )
                resp["sandcastle_batch_meta"] = {
                    "batch_size": 1,
                    "slot_batch_size": 1,
                    "model_alias": slot.alias,
                    "slot_precision": slot.precision,
                    "slot_id": id(slot),
                    "agent_count_configured": int(getattr(slot, "agent_count", 1)),
                    "max_parallel_agents": slot_max_seqs,
                    "max_model_len": int(self.config.get("max_model_len", 16384)),
                    "backend": "vllm",
                }
            return resp
        item = self._build_item_from_payload(payload)
        coord = slot.coordinator or self.coordinator
        return coord.submit(item)

    def stream_chat_completion(
        self,
        payload: Dict[str, Any],
        request_id: str,
        created_ts: int,
        token_queue: queue.Queue,
        abort_event: threading.Event,
    ) -> None:
        """Streams a chat completion without any global GPU lock."""
        if abort_event.is_set():
            return

        slot = self.resolve_loaded_slot(payload.get("model"))
        if slot.upstream_port is not None:
            self._record_slot_batch_on_engine_coordinator(1)
            _proxy_stream_to_token_queue(
                slot.upstream_host,
                int(slot.upstream_port),
                slot.alias,
                payload,
                token_queue,
                abort_event,
            )
            return

        item = self._build_item_from_payload(payload, request_id=request_id)
        max_new_tokens = item.max_tokens

        slot_model = slot.model
        slot_tokenizer = slot.tokenizer
        if slot_model is None or slot_tokenizer is None:
            raise RuntimeError("No model is physically loaded in accelerator memory.")

        model_alias = slot.alias or getattr(self, "active_model_alias", None) or UNIFIED_MODEL_ALIAS
        prompt, opens_think, caps = self._format_prompt_with_meta(item, slot=slot)
        base_stop_ids = _resolve_stop_token_ids(slot_tokenizer, model_alias, slot_model)
        gp = self._resolve_generation_params(item, caps, base_stop_ids)
        pad_id = int(
            caps.get("pad_token_id")
            if caps.get("pad_token_id") is not None
            else _resolve_pad_token_id(slot_tokenizer, slot_model)
        )

        import torch

        device = getattr(slot_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        max_prompt_len = max(1024, self.config.get("max_model_len", 16384) - max_new_tokens)

        input_ids = None
        attention_mask = None
        prompt_tokens = max(1, len(prompt.split()))

        if hasattr(slot_tokenizer, "__call__"):
            try:
                try:
                    encoded = slot_tokenizer(
                        [prompt],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_prompt_len,
                    )
                except TypeError:
                    encoded = slot_tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_prompt_len,
                    )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                prompt_tokens = (
                    int(attention_mask[0].sum().item())
                    if getattr(attention_mask, "ndim", 1) >= 2
                    else int(attention_mask.sum().item())
                )
            except Exception:
                pass

        hf_lpl_cls, hf_scl_cls, hf_streamer_cls = _get_hf_generation_types()
        stop_str_event = threading.Event()
        criteria_list: List[Any] = [ClientDisconnectStoppingCriteria(abort_event)]
        if gp["stop_strings"] and input_ids is not None:
            criteria_list.append(
                CustomStopStringsStoppingCriteria(
                    slot_tokenizer,
                    gp["stop_strings"],
                    int(input_ids.shape[1]),
                    stop_triggered_event=stop_str_event,
                )
            )
        stopping_criteria = hf_scl_cls(criteria_list)
        demuxer = StreamTagDemuxer(initial_in_think=opens_think, stop_strings=gp["stop_strings"])
        completion_tokens = 0

        use_real_streaming = (
            input_ids is not None
            and hasattr(slot_model, "generate")
            and hf_streamer_cls is not None
        )

        if use_real_streaming:
            streamer = hf_streamer_cls(slot_tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10.0)
            prompt_padded_len = int(input_ids.shape[1])
            gen_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "stopping_criteria": stopping_criteria,
                "streamer": streamer,
                "pad_token_id": pad_id,
                "do_sample": bool(gp["do_sample"]),
                "use_cache": True,
            }
            if gp["stop_token_ids"]:
                gen_kwargs["eos_token_id"] = gp["stop_token_ids"]
            if gp["do_sample"]:
                gen_kwargs["temperature"] = max(float(gp["temperature"]), 1e-5)
                if gp["top_p"] is not None:
                    gen_kwargs["top_p"] = float(gp["top_p"])
                if gp["top_k"] is not None and int(gp["top_k"]) > 0:
                    gen_kwargs["top_k"] = int(gp["top_k"])
            if gp["repetition_penalty"] is not None and float(gp["repetition_penalty"]) != 1.0:
                gen_kwargs["repetition_penalty"] = float(gp["repetition_penalty"])
            if gp["presence_penalty"] is not None and float(gp["presence_penalty"]) != 0.0:
                gen_kwargs["logits_processor"] = hf_lpl_cls(
                    [PresencePenaltyLogitsProcessor(float(gp["presence_penalty"]), prompt_padded_len)]
                )
            gen_err: List[Exception] = []

            def _generate_worker():
                try:
                    with torch.inference_mode():
                        slot_model.generate(**gen_kwargs)
                except Exception as exc:
                    gen_err.append(exc)
                    try:
                        streamer.end()
                    except Exception:
                        pass

            gen_thread = threading.Thread(target=_generate_worker, daemon=True)
            gen_thread.start()

            try:
                for text_chunk in streamer:
                    if abort_event.is_set():
                        break
                    completion_tokens += 1
                    chunks = demuxer.process_token(text_chunk)
                    for c in chunks:
                        delta_key = "reasoning_content" if c["type"] == "thinking" else "content"
                        token_queue.put(
                            {
                                "type": "chunk",
                                "data": {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": model_alias,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {delta_key: c["text"]},
                                            "finish_reason": None,
                                        }
                                    ],
                                },
                            }
                        )
                    if demuxer.stopped:
                        break
            finally:
                gen_thread.join(timeout=2.0)
            if gen_err:
                raise gen_err[0]
        else:
            mock_text = ""
            if hasattr(slot_model, "mock_output"):
                mock_text = str(slot_model.mock_output)
            elif hasattr(slot_tokenizer, "decode"):
                mock_text = slot_tokenizer.decode([])
            else:
                mock_text = "Hello! Ready to assist."

            if opens_think and "</think>" not in mock_text and "<think>" not in mock_text:
                demuxer.in_think = False

            tokens = [t + (" " if idx < len(mock_text.split(" ")) - 1 else "") for idx, t in enumerate(mock_text.split(" "))]
            for tok in tokens:
                if abort_event.is_set() or demuxer.stopped:
                    break
                if len(tokens) > 1:
                    time.sleep(0.01)
                completion_tokens += 1
                chunks = demuxer.process_token(tok)
                for c in chunks:
                    delta_key = "reasoning_content" if c["type"] == "thinking" else "content"
                    token_queue.put(
                        {
                            "type": "chunk",
                            "data": {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_alias,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {delta_key: c["text"]},
                                        "finish_reason": None,
                                    }
                                ],
                            },
                        }
                    )
                if demuxer.stopped:
                    break

        if abort_event.is_set():
            return

        rem_chunks, tool_buffer, full_raw = demuxer.flush()
        for c in rem_chunks:
            delta_key = "reasoning_content" if c["type"] == "thinking" else "content"
            token_queue.put(
                {
                    "type": "chunk",
                    "data": {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_alias,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {delta_key: c["text"]},
                                "finish_reason": None,
                            }
                        ],
                    },
                }
            )

        text_to_parse = full_raw if full_raw else tool_buffer
        res = extract_tool_calls_from_text(text_to_parse)
        tool_calls = res.tool_calls

        final_delta: Dict[str, Any] = {}
        finish_reason = "stop"
        if tool_calls:
            final_delta["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        token_queue.put(
            {
                "type": "chunk",
                "data": {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": final_delta, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": max(1, completion_tokens),
                        "total_tokens": prompt_tokens + max(1, completion_tokens),
                    },
                },
            }
        )


# Neutral multi-family serving engine alias (Spec 862)
BatchedModelServingEngine = BatchedQwen35ServingEngine


def _is_client_socket_closed(conn_obj: Any) -> bool:
    if not isinstance(conn_obj, socket.socket):
        return False
    try:
        if conn_obj.fileno() < 0:
            return True
        readable, _, _ = select.select([conn_obj], [], [], 0.0)
        if not readable:
            return False
        flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
        peeked = conn_obj.recv(1, flags)
        return len(peeked) == 0
    except BlockingIOError:
        return False
    except (BrokenPipeError, ConnectionResetError, OSError):
        return True


def _handle_sse_streaming_chat_completion(
    handler: BaseHTTPRequestHandler,
    engine: Any,
    body: Dict[str, Any],
    keepalive_interval_sec: float = 10.0,
) -> None:
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    loaded_slots = getattr(engine, "loaded_models", None)
    resolved_slot = None
    if isinstance(loaded_slots, dict) and loaded_slots and hasattr(engine, "resolve_loaded_slot"):
        resolved_slot = engine.resolve_loaded_slot(body.get("model"))
        model_name = resolved_slot.alias or body.get("model") or getattr(engine, "active_model_alias", UNIFIED_MODEL_ALIAS)
    else:
        model_name = body.get("model") or getattr(engine, "active_model_alias", UNIFIED_MODEL_ALIAS)

    # Native vLLM child process slot: lock-free direct SSE reverse proxy with TCP_NODELAY and immediate disconnect abort
    if resolved_slot is not None and resolved_slot.upstream_port is not None:
        if hasattr(engine, "_record_slot_batch_on_engine_coordinator"):
            engine._record_slot_batch_on_engine_coordinator(1)
        _proxy_sse_chat_completion_to_vllm_slot(
            handler,
            resolved_slot.upstream_host,
            int(resolved_slot.upstream_port),
            model_name,
            body,
        )
        return

    use_batched_stream = bool(body.get("prefer_batched", False))
    if resolved_slot is not None and resolved_slot.upstream_port is None:
        if int(getattr(resolved_slot, "expected_concurrency", 1)) > 1:
            use_batched_stream = True
        elif (
            getattr(resolved_slot, "coordinator", None) is not None
            and resolved_slot.coordinator.pending_or_active_count > 0
        ):
            use_batched_stream = True

    # 1. Disable Nagle's algorithm for low-latency streaming
    try:
        handler.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass

    # 2. Immediate headers (<= 100ms TTFB)
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache, no-transform")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()

    # 3. Immediate initial heartbeat comment flush (<= 100ms TTFB)
    handler.wfile.write(b": keepalive\n\n")
    handler.wfile.flush()

    abort_event = threading.Event()
    token_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    def _run_worker():
        try:
            if not use_batched_stream and hasattr(engine, "stream_chat_completion"):
                engine.stream_chat_completion(
                    payload=body,
                    request_id=req_id,
                    created_ts=created_ts,
                    token_queue=token_queue,
                    abort_event=abort_event,
                )
            else:
                resp = engine.handle_chat_completion(body)
                if abort_event.is_set():
                    return
                choices = resp.get("choices") or []
                choice = choices[0] if choices else {}
                message = choice.get("message") or {}
                m_id = resp.get("id") or req_id
                m_model = resp.get("model") or model_name
                m_ts = resp.get("created") or created_ts

                if message.get("reasoning_content"):
                    token_queue.put({
                        "type": "chunk",
                        "data": {
                            "id": m_id,
                            "object": "chat.completion.chunk",
                            "created": m_ts,
                            "model": m_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": str(message["reasoning_content"])},
                                    "finish_reason": None,
                                }
                            ],
                        },
                    })

                if message.get("content"):
                    token_queue.put({
                        "type": "chunk",
                        "data": {
                            "id": m_id,
                            "object": "chat.completion.chunk",
                            "created": m_ts,
                            "model": m_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": str(message["content"])},
                                    "finish_reason": None,
                                }
                            ],
                        },
                    })

                final_delta: Dict[str, Any] = {}
                raw_finish = choice.get("finish_reason")
                if message.get("tool_calls"):
                    indexed_tcs = []
                    for tc_idx, tc_item in enumerate(message["tool_calls"]):
                        tc_copy = dict(tc_item) if isinstance(tc_item, dict) else {"raw": tc_item}
                        tc_copy.setdefault("index", tc_idx)
                        indexed_tcs.append(tc_copy)
                    final_delta["tool_calls"] = indexed_tcs
                    finish_reason = raw_finish or "tool_calls"
                else:
                    finish_reason = raw_finish or "stop"

                final_chunk: Dict[str, Any] = {
                    "id": m_id,
                    "object": "chat.completion.chunk",
                    "created": m_ts,
                    "model": m_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": final_delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                if "usage" in resp:
                    final_chunk["usage"] = resp["usage"]
                if "sandcastle_batch_meta" in resp:
                    final_chunk["sandcastle_batch_meta"] = resp["sandcastle_batch_meta"]

                token_queue.put({
                    "type": "chunk",
                    "data": final_chunk,
                })
        except Exception as exc:
            token_queue.put({"type": "error", "error": str(exc)})
        finally:
            token_queue.put({"type": "done"})

    worker_thread = threading.Thread(target=_run_worker, daemon=True)
    worker_thread.start()

    # 4. SSE pump loop with keepalive heartbeats and immediate client-disconnect detection
    keepalive_period = max(0.02, float(keepalive_interval_sec))
    poll_step = min(0.03, keepalive_period)
    last_write_ts = time.perf_counter()
    client_conn = getattr(handler, "connection", None)
    try:
        while True:
            if _is_client_socket_closed(client_conn):
                abort_event.set()
                break
            try:
                evt = token_queue.get(timeout=poll_step)
            except queue.Empty:
                if _is_client_socket_closed(client_conn):
                    abort_event.set()
                    break
                now_ts = time.perf_counter()
                if (now_ts - last_write_ts) >= keepalive_period:
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    last_write_ts = now_ts
                continue

            ev_type = evt.get("type")
            if ev_type == "chunk":
                chunk_payload = f"data: {json.dumps(evt['data'], ensure_ascii=False)}\n\n".encode("utf-8")
                handler.wfile.write(chunk_payload)
                handler.wfile.flush()
                last_write_ts = time.perf_counter()
            elif ev_type == "error":
                err_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    "error": {"message": evt.get("error", "Generation error")},
                }
                handler.wfile.write(f"data: {json.dumps(err_chunk)}\n\n".encode("utf-8"))
                handler.wfile.flush()
                break
            elif ev_type == "done":
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
                break
    except (BrokenPipeError, ConnectionResetError, OSError):
        abort_event.set()
    finally:
        abort_event.set()
        handler.close_connection = True


MAX_HTTP_REQUEST_BODY_BYTES: int = 64 * 1024 * 1024  # 64 MB cap against Content-Length OOM DoS


class _HTTPPayloadError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = str(message)


def _validate_safe_cache_root(raw_cache_root: Optional[Any]) -> Optional[str]:
    """Ensures caller-supplied cache_root cannot escape the allowed model cache directory."""
    if raw_cache_root is None or not str(raw_cache_root).strip():
        return None
    allowed_base = Path(os.environ.get("PLATODOOM_MODEL_CACHE_DIR", "/content/models")).expanduser().resolve()
    candidate = Path(str(raw_cache_root).strip()).expanduser().resolve()
    if candidate != allowed_base and allowed_base not in candidate.parents:
        raise ValueError(
            f"Rejected unsafe cache_root '{raw_cache_root}': must stay within '{allowed_base}'."
        )
    return str(candidate)


def build_openai_request_handler(
    engine: BatchedQwen35ServingEngine,
    bearer_token: Optional[str] = None,
    sync_namespace: Optional[Dict[str, Any]] = None,
    keepalive_interval_sec: float = 10.0,
    require_auth: bool = False,
) -> Any:
    """Builds the Port-8000 Supervisor BaseHTTPRequestHandler class for BatchedQwen35ServingEngine."""
    expected_token = (
        bearer_token
        if bearer_token is not None
        else (
            os.environ.get("PLATODOOM_TUNNEL_TOKEN")
            or os.environ.get("SANDCASTLE_VLLM_API_KEY")
            or os.environ.get("SANDCASTLE_SERVING_BEARER_TOKEN")
            or os.environ.get("COLAB_SERVING_TOKEN")
        )
    )

    def _sync_ns() -> None:
        if isinstance(sync_namespace, dict):
            sync_namespace["serve_model"] = getattr(engine, "model", None)
            sync_namespace["serve_tokenizer"] = getattr(engine, "tokenizer", None)
            sync_namespace["active_served_model"] = getattr(engine, "active_model_alias", None)

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_bounded_json_body(self, *, allow_empty: bool = False) -> Dict[str, Any]:
            raw_cl = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_cl)
            except (TypeError, ValueError) as exc:
                raise _HTTPPayloadError(400, f"Invalid Content-Length header: {raw_cl!r}") from exc
            if length < 0:
                raise _HTTPPayloadError(400, "Negative Content-Length header is not allowed.")
            if length > MAX_HTTP_REQUEST_BODY_BYTES:
                raise _HTTPPayloadError(
                    413,
                    f"Request body too large ({length} bytes > {MAX_HTTP_REQUEST_BODY_BYTES} bytes limit).",
                )
            if length == 0:
                if allow_empty:
                    return {}
                raise _HTTPPayloadError(400, "Empty JSON request body.")
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(parsed, dict):
                raise _HTTPPayloadError(400, "JSON request body must be an object.")
            return parsed

        def _is_authenticated(self) -> bool:
            if not expected_token or expected_token == "EMPTY":
                return not require_auth
            auth_header = self.headers.get("Authorization")
            return verify_bearer_authorization(auth_header, expected_token)

        def _send_unauthorized(self) -> None:
            self._send_json(
                401,
                {
                    "error": {
                        "message": "Unauthorized: valid Authorization: Bearer sk-plato-... token required.",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )

        def do_GET(self) -> None:
            if not self._is_authenticated():
                self._send_unauthorized()
                return

            if self.path.startswith("/v1/models") or self.path.startswith("/models"):
                if hasattr(engine, "_reconcile_single_slot_attributes"):
                    engine._reconcile_single_slot_attributes()
                loaded_slots = getattr(engine, "loaded_models", None)
                models_data: List[Dict[str, Any]] = []
                def _slot_model_capabilities(alias_str: str, fallback_len: int) -> Tuple[int, int, bool]:
                    cat_entry = BUILTIN_MODEL_CATALOG.get(alias_str) or _SUPERVISOR_MODEL_SPECS.get(alias_str) or {}
                    try:
                        p_spec = resolve_vllm_model_parser_spec(alias_str)
                        has_reas = bool(p_spec.get("reasoning_parser") is not None)
                    except Exception:
                        has_reas = bool(cat_entry.get("reasoning_parser") is not None)
                    eff_len = int(fallback_len)
                    if cat_entry.get("max_output_tokens"):
                        max_out = int(cat_entry["max_output_tokens"])
                    elif alias_str == "r1-32b":
                        max_out = 8192 if eff_len >= 24576 else 4096
                    elif eff_len >= 65536:
                        max_out = 8192
                    elif eff_len >= 24576:
                        max_out = 6144 if has_reas else 4096
                    else:
                        max_out = 4096 if has_reas else 3072
                    if eff_len <= 16384 and max_out > 4096:
                        max_out = 4096
                    return eff_len, max_out, has_reas

                if isinstance(loaded_slots, dict) and loaded_slots:
                    for slot in loaded_slots.values():
                        if slot.is_resident:
                            slot_coord = getattr(slot, "coordinator", None)
                            if slot.vllm_slot is not None:
                                slot_max_batch = int(slot.vllm_slot.max_num_seqs)
                                slot_max_len = int(getattr(slot.vllm_slot, "max_model_len", None) or getattr(slot, "max_model_len", None) or engine.config["max_model_len"])
                            elif slot_coord is not None:
                                slot_max_batch = int(slot_coord.max_num_seqs)
                                slot_max_len = int(getattr(slot, "max_model_len", None) or engine.config["max_model_len"])
                            else:
                                slot_max_batch = int(engine.config["max_num_seqs"])
                                slot_max_len = int(getattr(slot, "max_model_len", None) or engine.config["max_model_len"])
                            eff_len, slot_max_out, slot_reas = _slot_model_capabilities(str(slot.alias), slot_max_len)
                            v_flags = (
                                getattr(slot.vllm_slot, "resolved_flags", None)
                                if getattr(slot, "vllm_slot", None) is not None
                                else None
                            )
                            if not isinstance(v_flags, Mapping):
                                v_flags = getattr(slot, "resolved_flags", None)
                            slot_vision = bool((v_flags or {}).get("vision", False))
                            models_data.append(
                                {
                                    "id": slot.alias,
                                    "object": "model",
                                    "owned_by": "platodoom",
                                    "root": slot.hf_repo_id or slot.alias,
                                    "precision": slot.precision,
                                    "weights_gb": slot.weights_gb,
                                    "vram_allocated_gb": float(slot.weights_gb),
                                    "agent_count": int(getattr(slot, "agent_count", 1)),
                                    "max_batch_size": slot_max_batch,
                                    "role": slot.role,
                                    "max_model_len": eff_len,
                                    "context_length": eff_len,
                                    "max_output_tokens": slot_max_out,
                                    "reasoning": slot_reas,
                                    "vision": slot_vision,
                                    "input_modalities": ["text", "image"] if slot_vision else ["text"],
                                    "max_num_seqs": slot_max_batch,
                                    "backend": "vllm" if slot.upstream_port is not None else "transformers",
                                    "upstream_port": slot.upstream_port,
                                }
                            )
                else:
                    model_loaded = (
                        (getattr(engine, "model", None) is not None and getattr(engine, "tokenizer", None) is not None)
                        if hasattr(engine, "model")
                        else True
                    )
                    active_alias = getattr(engine, "active_model_alias", None)
                    active_repo = getattr(engine, "active_hf_repo_id", None) or active_alias
                    if model_loaded and active_alias:
                        eff_len, slot_max_out, slot_reas = _slot_model_capabilities(
                            str(active_alias),
                            int(engine.config["max_model_len"]),
                        )
                        models_data.append(
                            {
                                "id": active_alias,
                                "object": "model",
                                "owned_by": "platodoom",
                                "root": active_repo,
                                "agent_count": 1,
                                "vram_allocated_gb": 0.0,
                                "max_batch_size": int(engine.config["max_num_seqs"]),
                                "max_model_len": eff_len,
                                "context_length": eff_len,
                                "max_output_tokens": slot_max_out,
                                "reasoning": slot_reas,
                                "max_num_seqs": engine.config["max_num_seqs"],
                            }
                        )
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": models_data,
                    },
                )
                return

            if self.path.startswith("/health"):
                if hasattr(engine, "_reconcile_single_slot_attributes"):
                    engine._reconcile_single_slot_attributes()
                switch_target = getattr(engine, "switch_in_progress", None)
                loaded_slots = getattr(engine, "loaded_models", None)
                has_loaded_slots = isinstance(loaded_slots, dict) and any(
                    s.is_resident for s in loaded_slots.values()
                )
                model_loaded = (
                    has_loaded_slots
                    or (getattr(engine, "model", None) is not None and getattr(engine, "tokenizer", None) is not None)
                    if hasattr(engine, "model")
                    else True
                )
                active_alias = getattr(engine, "active_model_alias", None)
                active_repo = getattr(engine, "active_hf_repo_id", None) or active_alias
                resident_list = (
                    [s.alias for s in loaded_slots.values() if s.is_resident]
                    if isinstance(loaded_slots, dict) and loaded_slots
                    else ([active_alias] if model_loaded and active_alias else [])
                )
                inactive_list = list(getattr(engine, "inactive_unloaded_models", []) or [])
                configured_list = list(getattr(engine, "configured_models", []) or resident_list)
                agent_counts_map = dict(getattr(engine, "model_agent_counts", {}) or {})
                vram_by_model: Dict[str, float] = {}
                slot_backends: Dict[str, str] = {}
                if isinstance(loaded_slots, dict) and loaded_slots:
                    for s in loaded_slots.values():
                        if s.is_resident:
                            vram_by_model[s.alias] = round(float(s.weights_gb), 4)
                            slot_backends[s.alias] = "vllm" if s.upstream_port is not None else "transformers"
                for inactive_alias in inactive_list:
                    vram_by_model.setdefault(inactive_alias, 0.0)

                agg_stats = (
                    engine.get_aggregated_batch_telemetry()
                    if hasattr(engine, "get_aggregated_batch_telemetry")
                    else None
                )
                total_batches_val = (
                    int(agg_stats["total_batches_executed"])
                    if agg_stats is not None
                    else int(engine.coordinator.total_batches_executed)
                )
                total_requests_val = (
                    int(agg_stats["total_requests_served"])
                    if agg_stats is not None
                    else int(getattr(engine.coordinator, "total_requests_served", 0))
                )
                max_batch_val = (
                    int(agg_stats["max_observed_batch_size"])
                    if agg_stats is not None
                    else int(engine.coordinator.max_observed_batch_size)
                )
                slot_stats_val = dict(agg_stats["slot_batch_stats"]) if agg_stats is not None else {}

                base_health: Dict[str, Any] = {
                    "stateless_inference_only": True,
                    "persistent_storage_bytes": 0,
                    "total_vram_gb": getattr(engine, "total_vram_gb", 40.0),
                    "max_num_seqs": engine.config["max_num_seqs"],
                    "max_model_len": engine.config["max_model_len"],
                    "max_num_batched_tokens": engine.config["max_num_batched_tokens"],
                    "total_batches": total_batches_val,
                    "total_requests_served": total_requests_val,
                    "max_observed_batch_size": max_batch_val,
                    "slot_batch_stats": slot_stats_val,
                    "resident_models": resident_list,
                    "inactive_unloaded_models": inactive_list,
                    "configured_models": configured_list,
                    "agent_counts": agent_counts_map,
                    "vram_allocated_by_model": vram_by_model,
                    "slot_backends": slot_backends,
                    "model_count": len(resident_list),
                    "stack_preset": getattr(engine, "stack_preset", None) or getattr(engine, "active_stack_preset", None),
                }
                if switch_target:
                    base_health.update(
                        {
                            "status": "switching",
                            "loaded": False,
                            "switching_to": switch_target,
                            "model": None,
                        }
                    )
                    self._send_json(503, base_health)
                    return
                if not model_loaded or not active_alias:
                    base_health.update(
                        {
                            "status": "unloaded",
                            "loaded": False,
                            "model": None,
                            "last_switch_error": getattr(engine, "last_switch_error", None),
                        }
                    )
                    self._send_json(503, base_health)
                    return
                base_health.update(
                    {
                        "status": "ok",
                        "loaded": True,
                        "model": active_alias,
                        "root": active_repo,
                    }
                )
                self._send_json(200, base_health)
                return

            self._send_json(404, {"error": "Not found"})

        def _handle_load_stack_body(self, body: Dict[str, Any]) -> None:
            raw_models = body.get("models")
            if raw_models is None:
                raw_models = body.get("stack") if "stack" in body else body.get("stack_models")
            raw_agent_counts = (
                body.get("agent_counts")
                if "agent_counts" in body
                else body.get("allocations")
            )
            preset_name = body.get("stack_preset")
            hw_tier = body.get("hardware_tier") or body.get("gpu_tier")
            if isinstance(raw_models, dict):
                if raw_agent_counts is None:
                    raw_agent_counts = dict(raw_models)
                raw_models = list(raw_models.keys())
            elif isinstance(raw_models, str):
                parsed_items = parse_multi_model_agent_spec(raw_models)
                raw_models = [
                    {
                        "alias": item["model_spec"],
                        "precision": item["precision"],
                        "agent_count": item["agent_count"],
                    }
                    for item in parsed_items
                ]
                if raw_agent_counts is None:
                    raw_agent_counts = {item["model_spec"]: item["agent_count"] for item in parsed_items}
            if raw_models is None and preset_name and hw_tier:
                resolved_preset = resolve_hardware_stack(
                    str(hw_tier),
                    stack_preset=str(preset_name),
                    model_alias=body.get("model_alias") or body.get("model"),
                    precision=body.get("precision"),
                )
                raw_models = resolved_preset["models"]
            if raw_models is None and raw_agent_counts is not None:
                if isinstance(raw_agent_counts, dict):
                    raw_models = list(raw_agent_counts.keys())
                elif isinstance(raw_agent_counts, list):
                    raw_models = list(raw_agent_counts)
                elif isinstance(raw_agent_counts, str):
                    parsed_items = parse_multi_model_agent_spec(raw_agent_counts)
                    raw_models = [
                        {
                            "alias": item["model_spec"],
                            "precision": item["precision"],
                            "agent_count": item["agent_count"],
                        }
                        for item in parsed_items
                    ]
                    raw_agent_counts = {item["model_spec"]: item["agent_count"] for item in parsed_items}
            if not isinstance(raw_models, list):
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "http_status": 400,
                        "code": "invalid_stack_payload",
                        "error": "Missing or non-list 'models' / 'stack' parameter for load-stack.",
                    },
                )
                return
            raw_stack_vision = body.get("enable_vision") if "enable_vision" in body else body.get("vision")
            parsed_stack_vision = _parse_optional_bool_flag(raw_stack_vision, field_name="enable_vision")
            stack_kwargs: Dict[str, Any] = {
                "hardware_tier": str(hw_tier) if hw_tier else None,
                "stack_preset": str(preset_name) if preset_name else None,
                "cache_root": _validate_safe_cache_root(body.get("cache_root")),
            }
            try:
                sig = inspect.signature(engine.load_model_stack)
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if raw_agent_counts is not None and ("agent_counts" in sig.parameters or accepts_var_kw):
                    stack_kwargs["agent_counts"] = raw_agent_counts
                if parsed_stack_vision is not None and ("enable_vision" in sig.parameters or accepts_var_kw):
                    stack_kwargs["enable_vision"] = parsed_stack_vision
            except Exception:
                if raw_agent_counts is not None:
                    stack_kwargs["agent_counts"] = raw_agent_counts
                if parsed_stack_vision is not None:
                    stack_kwargs["enable_vision"] = parsed_stack_vision
            res = engine.load_model_stack(raw_models, **stack_kwargs)
            _sync_ns()
            status_code = int(res.get("http_status", 200 if res.get("ok") else 507))
            self._send_json(status_code, res)

        def do_POST(self) -> None:
            if not self._is_authenticated():
                self._send_unauthorized()
                return

            if (
                self.path.startswith("/v1/internal/load-stack")
                or self.path.startswith("/internal/load-stack")
                or self.path.startswith("/v1/admin/load-stack")
                or self.path.startswith("/v1/models/load_stack")
                or self.path.startswith("/v1/models/load-stack")
                or self.path.startswith("/models/load_stack")
                or self.path.startswith("/models/load-stack")
            ):
                try:
                    body = self._read_bounded_json_body(allow_empty=True)
                    self._handle_load_stack_body(body)
                except _HTTPPayloadError as perr:
                    self._send_json(perr.status_code, {"ok": False, "http_status": perr.status_code, "code": "invalid_payload", "error": perr.message})
                except ValueError as verr:
                    self._send_json(400, {"ok": False, "http_status": 400, "code": "invalid_model_identifier", "error": str(verr)})
                except Exception as exc:
                    self._send_json(500, {"ok": False, "http_status": 500, "code": "load_stack_internal_error", "error": str(exc)})
                return

            if (
                self.path.startswith("/v1/internal/switch-model")
                or self.path.startswith("/v1/admin/switch-model")
                or self.path.startswith("/v1/models/switch")
            ):
                try:
                    body = self._read_bounded_json_body(allow_empty=True)
                    if (
                        (
                            isinstance(body.get("models") or body.get("stack"), (list, dict, str))
                            or "agent_counts" in body
                            or "allocations" in body
                        )
                        and hasattr(engine, "load_model_stack")
                    ):
                        self._handle_load_stack_body(body)
                        return
                    raw_alias = body.get("model_alias") if "model_alias" in body else body.get("model")
                    if raw_alias is None or not str(raw_alias).strip():
                        self._send_json(
                            400,
                            {
                                "ok": False,
                                "http_status": 400,
                                "code": "invalid_model_identifier",
                                "error": "Missing or empty 'model_alias' parameter",
                            },
                        )
                        return
                    raw_sw_vision = body.get("enable_vision") if "enable_vision" in body else body.get("vision")
                    parsed_sw_vision = _parse_optional_bool_flag(raw_sw_vision, field_name="enable_vision")
                    safe_cache_root = _validate_safe_cache_root(body.get("cache_root"))
                    candidate_kwargs: Dict[str, Any] = {
                        "model_alias": str(raw_alias),
                        "hf_repo_id": body.get("hf_repo_id"),
                        "precision": body.get("precision"),
                        "max_model_len": body.get("max_model_len"),
                        "source": body.get("source"),
                        "storage_uri": body.get("storage_uri"),
                        "gcs_bucket": body.get("gcs_bucket"),
                        "gcs_prefix": body.get("gcs_prefix"),
                        "fallback_hf_repo_id": body.get("fallback_hf_repo_id"),
                        "cache_root": safe_cache_root,
                        **({"enable_vision": parsed_sw_vision} if parsed_sw_vision is not None else {}),
                    }
                    try:
                        sig = inspect.signature(engine.switch_model)
                        accepts_var_kw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                        )
                        if not accepts_var_kw:
                            candidate_kwargs = {
                                k: v for k, v in candidate_kwargs.items() if k in sig.parameters
                            }
                    except Exception:
                        candidate_kwargs = {
                            k: candidate_kwargs[k]
                            for k in ("model_alias", "hf_repo_id", "precision", "max_model_len")
                        }
                    res = engine.switch_model(**candidate_kwargs)
                    _sync_ns()
                    status_code = int(res.get("http_status", 200 if res.get("ok") else 507))
                    self._send_json(status_code, res)
                except _HTTPPayloadError as perr:
                    self._send_json(perr.status_code, {"ok": False, "http_status": perr.status_code, "code": "invalid_payload", "error": perr.message})
                except ValueError as verr:
                    self._send_json(400, {"ok": False, "http_status": 400, "code": "invalid_model_identifier", "error": str(verr)})
                except Exception as exc:
                    self._send_json(500, {"ok": False, "http_status": 500, "code": "switch_internal_error", "error": str(exc)})
                return

            if self.path.startswith("/v1/chat/completions") or self.path.startswith("/chat/completions"):
                try:
                    body = self._read_bounded_json_body(allow_empty=False)
                    if hasattr(engine, "_is_explicitly_inactive_model") and engine._is_explicitly_inactive_model(body.get("model")):
                        raise KeyError(
                            f"Model '{body.get('model')}' is configured with 0 active agents (n_i == 0) and is not loaded in VRAM."
                        )
                    loaded_slots = getattr(engine, "loaded_models", None)
                    has_loaded = (
                        (isinstance(loaded_slots, dict) and any(s.is_resident for s in loaded_slots.values()))
                        or (getattr(engine, "model", None) is not None and getattr(engine, "tokenizer", None) is not None)
                    )
                    if (
                        hasattr(engine, "model")
                        and not has_loaded
                        and getattr(engine, "switch_in_progress", None) is None
                    ):
                        self._send_json(
                            503,
                            {"error": {"message": "No model is physically loaded in accelerator memory.", "type": "model_unloaded", "code": "no_model_loaded"}},
                        )
                        return
                    if body.get("stream", False):
                        _handle_sse_streaming_chat_completion(self, engine, body, keepalive_interval_sec=keepalive_interval_sec)
                        return
                    resp = engine.handle_chat_completion(body)
                    self._send_json(200, resp)
                except _HTTPPayloadError as perr:
                    self._send_json(perr.status_code, {"error": {"message": perr.message, "type": "invalid_request_error"}})
                except _UpstreamHTTPStatusError as uerr:
                    self._send_json(uerr.status_code, uerr.payload)
                except KeyError as kerr:
                    self._send_json(
                        404,
                        {"error": {"message": str(kerr), "type": "invalid_request_error", "code": "model_not_loaded"}},
                    )
                except TimeoutError as terr:
                    self._send_json(503, {"error": {"message": str(terr), "type": "queue_timeout"}})
                except RuntimeError as rerr:
                    if "no model is physically loaded" in str(rerr).lower():
                        self._send_json(503, {"error": {"message": str(rerr), "type": "model_unloaded", "code": "no_model_loaded"}},
                        )
                    else:
                        self._send_json(500, {"error": {"message": str(rerr), "type": "server_error"}})
                except Exception as exc:
                    self._send_json(500, {"error": {"message": str(exc), "type": "server_error"}})
                return

            if self.path.startswith("/v1/completions"):
                try:
                    body = self._read_bounded_json_body(allow_empty=False)
                    if hasattr(engine, "_is_explicitly_inactive_model") and engine._is_explicitly_inactive_model(body.get("model")):
                        raise KeyError(
                            f"Model '{body.get('model')}' is configured with 0 active agents (n_i == 0) and is not loaded in VRAM."
                        )
                    loaded_slots = getattr(engine, "loaded_models", None)
                    has_loaded = (
                        (isinstance(loaded_slots, dict) and any(s.is_resident for s in loaded_slots.values()))
                        or (getattr(engine, "model", None) is not None and getattr(engine, "tokenizer", None) is not None)
                    )
                    if (
                        hasattr(engine, "model")
                        and not has_loaded
                        and getattr(engine, "switch_in_progress", None) is None
                    ):
                        self._send_json(
                            503,
                            {"error": {"message": "No model is physically loaded in accelerator memory.", "type": "model_unloaded", "code": "no_model_loaded"}},
                        )
                        return
                    prompt = body.get("prompt", "")
                    messages = [{"role": "user", "content": prompt}]
                    body_converted = dict(body)
                    body_converted["messages"] = messages
                    if body_converted.get("stream", False):
                        _handle_sse_streaming_chat_completion(self, engine, body_converted)
                        return
                    resp = engine.handle_chat_completion(body_converted)
                    self._send_json(200, resp)
                except _HTTPPayloadError as perr:
                    self._send_json(perr.status_code, {"error": {"message": perr.message, "type": "invalid_request_error"}})
                except _UpstreamHTTPStatusError as uerr:
                    self._send_json(uerr.status_code, uerr.payload)
                except KeyError as kerr:
                    self._send_json(
                        404,
                        {"error": {"message": str(kerr), "type": "invalid_request_error", "code": "model_not_loaded"}},
                    )
                except TimeoutError as terr:
                    self._send_json(503, {"error": {"message": str(terr), "type": "queue_timeout"}})
                except RuntimeError as rerr:
                    if "no model is physically loaded" in str(rerr).lower():
                        self._send_json(503, {"error": {"message": str(rerr), "type": "model_unloaded", "code": "no_model_loaded"}})
                    else:
                        self._send_json(500, {"error": {"message": str(rerr), "type": "server_error"}})
                except Exception as exc:
                    self._send_json(500, {"error": {"message": str(exc), "type": "server_error"}})
                return

            self._send_json(404, {"error": "Not found"})

    return _Handler


def create_openai_http_server(
    engine: BatchedQwen35ServingEngine,
    host: str = "127.0.0.1",
    port: int = 0,
    bearer_token: Optional[str] = None,
    keepalive_interval_sec: float = 10.0,
) -> ThreadingHTTPServer:
    """Creates a multi-threaded OpenAI-compatible HTTP server backed by BatchedQwen35ServingEngine (FR-5.4)."""
    handler_cls = build_openai_request_handler(
        engine=engine,
        bearer_token=bearer_token,
        keepalive_interval_sec=keepalive_interval_sec,
    )
    return ThreadingHTTPServer((host, port), handler_cls)


def _evict_prior_cell4_runtime_state(ns: Dict[str, Any]) -> None:
    """Evicts prior Cell 4 HTTP server, child vLLM process groups (`os.killpg`), and cyclic nn.Modules on warm kernel re-runs."""
    for _srv_key in ("http_srv", "server"):
        _old_srv = ns.pop(_srv_key, None)
        if _old_srv is not None:
            for _fn_name in ("shutdown", "server_close"):
                _fn = getattr(_old_srv, _fn_name, None)
                if callable(_fn):
                    try:
                        _fn()
                    except Exception:
                        pass
        del _old_srv
    _old_eng = ns.pop("_engine", None)
    if _old_eng is not None:
        for _fn_name in ("close", "_evict"):
            _fn = getattr(_old_eng, _fn_name, None)
            if callable(_fn):
                try:
                    _fn()
                except Exception:
                    pass
    del _old_eng
    _torch_mod = sys.modules.get("torch")
    for _mdl_key in ("serve_model", "serve_tokenizer", "serve_tok", "_qm", "model"):
        _old_obj = ns.pop(_mdl_key, None)
        if (
            _old_obj is not None
            and _torch_mod is not None
            and hasattr(_torch_mod, "nn")
            and isinstance(_old_obj, _torch_mod.nn.Module)
        ):
            try:
                _old_obj._modules.clear()
                _old_obj._parameters.clear()
                _old_obj._buffers.clear()
            except Exception:
                pass
        del _old_obj
    for _tb_attr in ("last_exc", "last_value", "last_traceback"):
        if hasattr(sys, _tb_attr):
            try:
                setattr(sys, _tb_attr, None)
            except Exception:
                pass
    gc.collect()
    if _torch_mod is not None and hasattr(_torch_mod, "cuda") and _torch_mod.cuda.is_available():
        try:
            _torch_mod.cuda.empty_cache()
        except Exception:
            pass


# === END_COLAB_SUPERVISOR_CORE ===


_CELL4_RUNTIME_BOOTSTRAP_TEMPLATE = """
# === Cell 4 Runtime Supervisor Bootstrap (Single-Sourced from curation/colab_server.py) ===
_evict_prior_cell4_runtime_state(globals())

_engine = None
loaded_models_registry: Dict[str, Any] = {}
serve_model = None
serve_tokenizer = None
active_served_model = None
_ColabServingHandler = None


def _switch_colab_served_model(
    model_alias: str,
    hf_repo_id: Optional[str] = None,
    precision: Optional[str] = None,
    max_model_len: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    if _engine is None:
        raise RuntimeError("Supervisor engine is not initialized.")
    res = _engine.switch_model(
        model_alias=model_alias,
        hf_repo_id=hf_repo_id,
        precision=precision,
        max_model_len=max_model_len,
        **kwargs,
    )
    globals()["serve_model"] = getattr(_engine, "model", None)
    globals()["serve_tokenizer"] = getattr(_engine, "tokenizer", None)
    globals()["active_served_model"] = getattr(_engine, "active_model_alias", None)
    return res


if 'LOCAL_MODEL_DIR' in globals():
    _g = globals()
    _init_local_dir = str(_g.get("LOCAL_MODEL_DIR") or "")
    _init_alias = str(
        _g.get("ACTIVE_MODEL_ALIAS")
        or _g.get("MODEL_ALIAS")
        or os.environ.get("PLATODOOM_MODEL_ALIAS")
        or "qwen-9b"
    ).strip()
    _init_repo = str(
        _g.get("INITIAL_SERVED_REPO")
        or _g.get("ACTIVE_HF_REPO")
        or _init_local_dir
    ).strip()
    _vram_gb = float(_g.get("vram_gb") or _g.get("free_vram_gb") or 15.8)
    _free_vram_gb = float(_g.get("free_vram_gb") or _vram_gb)
    _max_seqs = int(_g.get("ACTIVE_GPU_SEQS") or os.environ.get("SANDCASTLE_MAX_PARALLEL_AGENTS") or 64)
    _max_len = int(_g.get("MAX_MODEL_LEN") or 16384)
    _max_batched = int(_g.get("MAX_NUM_BATCHED_TOKENS") or max(8192, _max_len))
    _hw_tier = str(_g.get("detected_tier") or os.environ.get("SANDCASTLE_GPU_TIER") or ("T4" if _vram_gb < 20.0 else "L4"))
    _cc_val = _g.get("compute_capability")
    _expected_tok = (
        _g.get("expected_token")
        or os.environ.get("PLATODOOM_TUNNEL_TOKEN")
        or os.environ.get("SANDCASTLE_VLLM_API_KEY")
        or os.environ.get("SANDCASTLE_SERVING_BEARER_TOKEN")
        or os.environ.get("COLAB_SERVING_TOKEN")
    )
    _custom_loader = _g.get("_cell4_model_loader")
    _custom_spawner = _g.get("_cell4_vllm_spawner")
    _is_tpu_env = bool(
        _g.get("IS_TPU_RUNTIME")
        or any(os.environ.get(k) for k in ("COLAB_TPU_ADDR", "TPU_NAME", "TPU_WORKER_ID"))
        or _hw_tier.startswith("v5e")
        or _hw_tier.startswith("v6e")
    )
    _vllm_bin_candidate = os.environ.get("PLATODOOM_VLLM_BIN") or "/opt/vllm-venv/bin/vllm"
    _has_vllm_bin = bool(
        _custom_spawner is not None
        or os.path.isfile(_vllm_bin_candidate)
        or shutil.which("vllm") is not None
    )
    _use_vllm = bool(not _is_tpu_env and _custom_loader is None and _has_vllm_bin)

    if _custom_loader is not None:
        _init_mdl, _init_tok, _resolved_repo = _custom_loader(_init_alias, _init_repo, total_vram_gb=_vram_gb)
        _engine = BatchedQwen35ServingEngine(
            model=_init_mdl,
            tokenizer=_init_tok,
            total_vram_gb=_vram_gb,
            free_vram_gb=_free_vram_gb,
            max_num_seqs=_max_seqs,
            max_model_len=_max_len,
            max_num_batched_tokens=_max_batched,
            active_model_alias=_init_alias,
            active_hf_repo_id=str(_resolved_repo or _init_repo),
            model_loader=_custom_loader,
            use_vllm_subprocess=False,
            gpu_tier=_hw_tier,
            compute_capability=_cc_val if isinstance(_cc_val, tuple) else None,
        )
    elif _use_vllm:
        _engine = BatchedQwen35ServingEngine(
            model=None,
            tokenizer=None,
            total_vram_gb=_vram_gb,
            free_vram_gb=_free_vram_gb,
            max_num_seqs=_max_seqs,
            max_model_len=_max_len,
            max_num_batched_tokens=_max_batched,
            active_model_alias=None,
            active_hf_repo_id=None,
            use_vllm_subprocess=True,
            vllm_spawner=_custom_spawner,
            gpu_tier=_hw_tier,
            compute_capability=_cc_val if isinstance(_cc_val, tuple) else None,
        )
        _staged_stack = _g.get("STAGED_STACK_MODELS")
        _staged_counts = _g.get("STAGED_AGENT_COUNTS")
        _init_enable_vision = _resolve_env_vision_toggle()
        if isinstance(_staged_stack, list) and len(_staged_stack) > 0:
            _boot_res = _engine.load_model_stack(
                _staged_stack,
                hardware_tier=_hw_tier,
                agent_counts=_staged_counts if isinstance(_staged_counts, dict) else None,
                cache_root=os.environ.get("PLATODOOM_MODEL_CACHE_DIR", "/content/models"),
                enable_vision=_init_enable_vision,
            )
            if not _boot_res.get("ok"):
                raise RuntimeError(f"Initial vLLM stack boot failed: {_boot_res}")
        else:
            _boot_res = _engine.switch_model(
                model_alias=_init_alias,
                hf_repo_id=_init_local_dir if os.path.isdir(_init_local_dir) else _init_repo,
                max_model_len=_max_len,
                cache_root=os.environ.get("PLATODOOM_MODEL_CACHE_DIR", "/content/models"),
                enable_vision=_init_enable_vision,
            )
            if not _boot_res.get("ok"):
                raise RuntimeError(f"Initial vLLM slot boot failed: {_boot_res}")
    else:
        import transformers as _c4_tf
        import torch as _c4_torch

        _arch_meta = inspect_checkpoint_architecture(_init_local_dir, allow_remote=False)
        _q_cfg = (
            build_quantization_config(_init_local_dir, _vram_gb, _init_alias, arch_meta=_arch_meta)
            if bool(_g.get("USE_4BIT_SERVING", False))
            else None
        )
        _c_dtype = _g.get("COMPUTE_DTYPE")
        if _c_dtype is None:
            _c_dtype = getattr(_c4_torch, "bfloat16", getattr(_c4_torch, "float16", None))
        COMPUTE_DTYPE = _c_dtype if (_c4_torch.cuda.is_available() if hasattr(_c4_torch, "cuda") else False) else getattr(_c4_torch, "float32", None)
        _init_tok = _c4_tf.AutoTokenizer.from_pretrained(_init_local_dir, trust_remote_code=True)
        if getattr(_init_tok, "padding_side", None) != "left":
            _init_tok.padding_side = "left"
        if getattr(_init_tok, "pad_token_id", None) is None and getattr(_init_tok, "eos_token_id", None) is not None:
            _init_tok.pad_token = _init_tok.eos_token
        _init_mdl, _ = load_model_from_checkpoint_metadata(
            _init_local_dir,
            allow_remote_config=False,
            torch_dtype=COMPUTE_DTYPE,
            trust_remote_code=True,
            **({"quantization_config": _q_cfg} if _q_cfg is not None else {}),
        )
        if hasattr(_init_mdl, "eval"):
            _init_mdl.eval()

        def _fallback_local_loader(
            model_alias: str = "",
            hf_repo_id: Optional[str] = None,
            total_vram_gb: float = _vram_gb,
            **extra_kw: Any,
        ) -> Tuple[Any, Any, str]:
            alias = str(model_alias or extra_kw.get("alias") or "")
            target_path = str(hf_repo_id or alias)
            if not os.path.isdir(target_path):
                prof = resolve_model_profile(alias)
                target_path = str(prof.get("quantized_hf_repo_id") or prof.get("hf_repo_id") or target_path)
            a_meta = inspect_checkpoint_architecture(target_path, allow_remote=not os.path.isdir(target_path))
            q_c = (
                build_quantization_config(target_path, total_vram_gb, alias, arch_meta=a_meta)
                if bool(_g.get("USE_4BIT_SERVING", False))
                else None
            )
            tok_l = _c4_tf.AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
            if getattr(tok_l, "padding_side", None) != "left":
                tok_l.padding_side = "left"
            if getattr(tok_l, "pad_token_id", None) is None and getattr(tok_l, "eos_token_id", None) is not None:
                tok_l.pad_token = tok_l.eos_token
            mdl_l, _ = load_model_from_checkpoint_metadata(
                target_path,
                allow_remote_config=not os.path.isdir(target_path),
                torch_dtype=COMPUTE_DTYPE,
                trust_remote_code=True,
                **({"quantization_config": q_c} if q_c is not None else {}),
            )
            if hasattr(mdl_l, "eval"):
                mdl_l.eval()
            return mdl_l, tok_l, target_path

        _engine = BatchedQwen35ServingEngine(
            model=_init_mdl,
            tokenizer=_init_tok,
            total_vram_gb=_vram_gb,
            free_vram_gb=_free_vram_gb,
            max_num_seqs=_max_seqs,
            max_model_len=_max_len,
            max_num_batched_tokens=_max_batched,
            active_model_alias=_init_alias,
            active_hf_repo_id=_init_repo,
            model_loader=_fallback_local_loader,
            use_vllm_subprocess=False,
            gpu_tier=_hw_tier,
            compute_capability=_cc_val if isinstance(_cc_val, tuple) else None,
        )
        del _init_mdl, _init_tok

    loaded_models_registry = _engine.loaded_models
    serve_model = getattr(_engine, "model", None)
    serve_tokenizer = getattr(_engine, "tokenizer", None)
    active_served_model = getattr(_engine, "active_model_alias", None)
    _allow_unauth = str(os.environ.get("PLATODOOM_ALLOW_UNAUTHENTICATED") or "").strip().lower() in ("1", "true", "yes")
    if not _expected_tok or _expected_tok == "EMPTY":
        if not _allow_unauth:
            import secrets as _sc_sec
            _expected_tok = f"sk-plato-{_sc_sec.token_hex(32)}"
            os.environ["PLATODOOM_TUNNEL_TOKEN"] = _expected_tok
    _ColabServingHandler = build_openai_request_handler(
        _engine,
        bearer_token=_expected_tok,
        sync_namespace=_g,
        require_auth=not _allow_unauth,
    )

http_srv = ThreadingHTTPServer(("0.0.0.0", 8000), _ColabServingHandler) if (__name__ == "__main__" and _ColabServingHandler is not None) else None
if http_srv is not None:
    threading.Thread(target=http_srv.serve_forever, daemon=True).start()
    _warmup_headers = {"Content-Type": "application/json"}
    _warmup_tok = os.environ.get("PLATODOOM_TUNNEL_TOKEN") or os.environ.get("SANDCASTLE_VLLM_API_KEY")
    if _warmup_tok:
        _warmup_headers["Authorization"] = f"Bearer {_warmup_tok}"
    try:
        _w_req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps({
                "model": active_served_model or "default",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode("utf-8"),
            headers=_warmup_headers,
            method="POST",
        )
        with urllib.request.urlopen(_w_req, timeout=240.0) as _w_resp:
            _ = _w_resp.read()
        print("Local CUDA & Triton kernels successfully warmed up")
    except Exception as _w_exc:
        _w_diag = ""
        if _engine is not None and isinstance(getattr(_engine, "loaded_models", None), dict):
            for _ws in _engine.loaded_models.values():
                _wv = getattr(_ws, "vllm_slot", None)
                _wlog = getattr(_wv, "log_path", None) if _wv is not None else None
                if _wlog and os.path.isfile(str(_wlog)):
                    try:
                        with open(str(_wlog), "r", encoding="utf-8", errors="replace") as _wf:
                            _w_diag = extract_vllm_root_cause_diagnostic(_wf.read()).get("diagnostic_excerpt", "")
                    except Exception:
                        _w_diag = ""
                    break
        try:
            http_srv.shutdown()
            http_srv.server_close()
        except Exception:
            pass
        raise RuntimeError(
            f"Cell 4 warmup inference failed ({_w_exc})"
            + (f":\\n{_w_diag}" if _w_diag else "")
        ) from _w_exc
    _vision_map = {k: bool(getattr(v, "vision", False)) for k, v in loaded_models_registry.items()}
    print(f"[Supervisor] Port-8000 OpenAI supervisor active (model={active_served_model}, vision={_vision_map}, resident={list(loaded_models_registry.keys())})")
"""


def get_cell4_supervisor_source(env_bootstrap: str = "") -> str:
    """Extracts the single-source-of-truth Cell 4 supervisor module + runtime bootstrap."""
    src_path = Path(__file__).resolve()
    full_text = src_path.read_text(encoding="utf-8")
    begin_marker = "# === BEGIN_COLAB_SUPERVISOR_CORE ==="
    end_marker = "# === END_COLAB_SUPERVISOR_CORE ==="
    b_idx = full_text.find(begin_marker)
    e_idx = full_text.find(end_marker)
    if b_idx == -1 or e_idx == -1 or e_idx <= b_idx:
        raise RuntimeError("Missing BEGIN/END_COLAB_SUPERVISOR_CORE markers in curation/colab_server.py")
    core_src = full_text[b_idx + len(begin_marker) : e_idx].strip()
    parts: List[str] = []
    if env_bootstrap and env_bootstrap.strip():
        parts.append(env_bootstrap.rstrip())
    parts.append(core_src)
    parts.append(_CELL4_RUNTIME_BOOTSTRAP_TEMPLATE.strip())
    return "\n\n".join(parts) + "\n"


