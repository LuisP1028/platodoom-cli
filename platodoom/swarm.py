"""
Multi-Directory Swarm Allocation, Per-Directory GPU Routing, Concurrency Cap Enforcement, and Session Registry Inspection for platodoom.
Document ID: platodoom_swarm_857_v1
Authoring Agent: storage-implementer
Governing Standards: refined_specification_857 (FR-2.1.2, FR-2.3, AC-4) & RULE[user_global]
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import urllib.error
import urllib.request

from curation.api_client import (
    DEFAULT_COLAB_ENDPOINT_FILENAME,
    _open_http_request,
    atomic_write_json,
    compute_dir_slug,
    load_colab_endpoint_record,
    normalize_colab_endpoint_url,
    resolve_colab_endpoint_registry_paths,
    resolve_project_root,
    route_gpu_for_model_count,
    route_gpu_for_multi_model_agents,
    write_gpu_routing_profile,
)
from platodoom.catalog import (
    parse_multi_model_agent_spec,
    resolve_multi_model_agent_allocation,
)
from platodoom.config import (
    resolve_platodoom_config,
    resolve_platodoom_home,
    resolve_response_cache_dir,
    resolve_session_state_dir,
)
from platodoom.response_cache import LocalResponseCache, redact_cache_payload
from platodoom.security import generate_ephemeral_tunnel_token

logger = logging.getLogger("platodoom.swarm")

_ORIG_SWARM_URLOPEN = urllib.request.urlopen


def _open_swarm_http_request(req: urllib.request.Request, timeout: float = 30.0) -> Any:
    if urllib.request.urlopen is not _ORIG_SWARM_URLOPEN:
        return urllib.request.urlopen(req, timeout=timeout)
    return _open_http_request(req, timeout=timeout)


_MODEL_MIN_GPU_TIER: Dict[str, Tuple[str, str, float]] = {
    "qwen-9b": ("T4", "cyankiwi/Qwen3.5-9B-AWQ-4bit", 16.0),
    "r1-14b": ("L4", "casperhansen/deepseek-r1-distill-qwen-14b-awq", 24.0),
    "qwen-27b": ("A100", "cyankiwi/Qwen3.5-27B-AWQ-4bit", 40.0),
    "qwen3.8-27b": ("A100", "cyankiwi/Qwen3.8-27B-AWQ-INT4", 40.0),
    "qwen-35b-moe": ("A100", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", 40.0),
    "r1-32b": ("A100", "casperhansen/deepseek-r1-distill-qwen-32b-awq", 40.0),
    "qwen-30b-moe": ("A100", "Qwen/Qwen3-30B-A3B-GPTQ-Int4", 40.0),
    "qwen-coder-30b-moe": ("A100", "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit", 40.0),
    "qwen-14b": ("L4", "Qwen/Qwen3-14B-AWQ", 24.0),
    "qwen-8b": ("T4", "Qwen/Qwen3-8B-AWQ", 16.0),
    "qwen-4b": ("T4", "Qwen/Qwen3-4B-AWQ", 16.0),
    "qwen-122b-moe": ("A100-80GB", "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4", 80.0),
    "r1-7b": ("T4", "casperhansen/deepseek-r1-distill-qwen-7b-awq", 16.0),
    "deepseek-v2-lite": ("T4", "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ", 16.0),
}

_GPU_TIER_RANK: Dict[str, int] = {
    "None": 0,
    "v5e-1": 1,
    "T4": 2,
    "L4": 3,
    "v6e-1": 4,
    "A100": 5,
    "A100-80GB": 6,
    "H100": 7,
    "G4": 8,
}


def _is_pid_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        pid_int = int(pid)
        if pid_int <= 0:
            return False
        os.kill(pid_int, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _normalize_gpu_tier(raw_tier: str) -> str:
    try:
        from platodoom.catalog import normalize_hardware_tier

        return normalize_hardware_tier(raw_tier, allow_auto=True)
    except ImportError:
        cleaned = (raw_tier or "auto").strip()
        upper = cleaned.upper()
        if upper in ("AUTO", ""):
            return "auto"
        mapping = {
            "V5E-1": "v5e-1",
            "T4": "T4",
            "L4": "L4",
            "V6E-1": "v6e-1",
            "A100": "A100",
            "A100-40GB": "A100",
            "A100_40GB": "A100",
            "A100-80GB": "A100-80GB",
            "A100_80GB": "A100-80GB",
            "H100": "H100",
            "G4": "G4",
        }
        if upper in mapping:
            return mapping[upper]
        raise ValueError(
            f"Invalid GPU/TPU tier '{raw_tier}'. Expected one of: auto, v5e-1, T4, L4, v6e-1, A100, A100-80GB, H100, G4."
        )


def _parse_single_swarm_allocation_spec(
    spec: str | Dict[str, Any],
    default_model: str = "qwen3.8-27b",
    base_cwd: Optional[Path | str] = None,
    allow_zero_agents: bool = False,
) -> Dict[str, Any]:
    """
    Parses a single '<directory_path>:<agent_count>:<gpu_tier>[:<model_alias>]' (or dict) into:
    {'directory_path': str, 'project_root': str, 'dir_slug': str,
     'agent_count': int, 'agents': int, 'gpu_tier': str, 'gpu': str, 'model_alias': str}.
    """
    eff_cwd = Path(base_cwd).expanduser().resolve() if base_cwd is not None else Path.cwd().resolve()
    min_agents = 0 if allow_zero_agents else 1

    def _resolve_dir(raw_p: str) -> Path:
        p = Path(raw_p).expanduser()
        if not p.is_absolute():
            p = eff_cwd / p
        return p.resolve()

    if isinstance(spec, dict):
        raw_dir = str(
            spec.get("directory_path")
            or spec.get("directory")
            or spec.get("dir")
            or spec.get("path")
            or ""
        ).strip()
        if not raw_dir:
            raise ValueError(f"Swarm allocation dict is missing 'directory_path': {spec}")
        raw_agents = spec.get("agent_count", spec.get("agents", 1))
        try:
            agent_count = int(raw_agents)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid agent_count '{raw_agents}' in {spec}") from exc
        if agent_count < min_agents:
            raise ValueError(f"agent_count must be >= {min_agents}, got {agent_count}")
        gpu_tier = _normalize_gpu_tier(str(spec.get("gpu_tier") or spec.get("gpu") or "auto"))
        model_alias = str(spec.get("model_alias") or spec.get("model") or default_model).strip() or default_model
        raw_stack = spec.get("stack_preset") or spec.get("stack")
        stack_preset = str(raw_stack).strip() if (raw_stack and str(raw_stack).strip()) else None
        resolved_dir = _resolve_dir(raw_dir)
        proj_root = resolve_project_root(resolved_dir)
        return {
            "directory_path": str(resolved_dir),
            "project_root": str(proj_root),
            "dir_slug": compute_dir_slug(proj_root),
            "agent_count": agent_count,
            "agents": agent_count,
            "gpu_tier": gpu_tier,
            "gpu": gpu_tier,
            "model_alias": model_alias,
            "stack_preset": stack_preset,
        }

    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Swarm allocation spec must be a non-empty string or dict.")

    cleaned = spec.strip()
    parts = cleaned.split(":")
    stack_preset = None
    if parts and parts[-1].strip().lower().startswith("stack="):
        stack_preset = parts.pop().strip().split("=", 1)[1].strip() or None

    if len(parts) < 3:
        raise ValueError(
            f"Invalid swarm allocation spec '{spec}'. Expected '<dir>:<agent_count>:<gpu_tier>[:<model_alias>][:stack=<preset>]'."
        )

    # Support optional trailing :model_alias vs 3-part <dir>:<agents>:<gpu>
    if len(parts) >= 4 and parts[-3].strip().lstrip("-").isdigit():
        model_alias = parts[-1].strip() or default_model
        raw_gpu = parts[-2].strip()
        raw_agents_str = parts[-3].strip()
        raw_dir = ":".join(parts[:-3]).strip()
    else:
        model_alias = default_model
        raw_gpu = parts[-1].strip()
        raw_agents_str = parts[-2].strip()
        raw_dir = ":".join(parts[:-2]).strip()

    if not raw_dir:
        raise ValueError(f"Empty directory path in swarm allocation spec '{spec}'.")
    try:
        agent_count = int(raw_agents_str)
    except ValueError as exc:
        raise ValueError(f"Invalid agent_count '{raw_agents_str}' in spec '{spec}'.") from exc
    if agent_count < min_agents:
        raise ValueError(f"agent_count must be >= {min_agents} in spec '{spec}', got {agent_count}.")

    gpu_tier = _normalize_gpu_tier(raw_gpu)
    resolved_dir = _resolve_dir(raw_dir)
    proj_root = resolve_project_root(resolved_dir)
    return {
        "directory_path": str(resolved_dir),
        "project_root": str(proj_root),
        "dir_slug": compute_dir_slug(proj_root),
        "agent_count": agent_count,
        "agents": agent_count,
        "gpu_tier": gpu_tier,
        "gpu": gpu_tier,
        "model_alias": model_alias,
        "stack_preset": stack_preset,
    }


def parse_swarm_allocation_spec(
    spec: str | Dict[str, Any] | List[str | Dict[str, Any]],
    default_model: str = "qwen3.8-27b",
    base_cwd: Optional[Path | str] = None,
    allow_zero_agents: bool = False,
) -> Any:
    """
    Parses a single '<dir>:<agents>:<gpu>[:<model>]' spec (returning a dict) OR a
    comma-separated / JSON-array / list of specs (returning a list of dicts when multiple
    specs are provided or when base_cwd / comma-separated / list input is used).
    """
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Swarm allocation spec list cannot be empty.")
        return [
            _parse_single_swarm_allocation_spec(
                item, default_model=default_model, base_cwd=base_cwd, allow_zero_agents=allow_zero_agents
            )
            for item in spec
        ]

    if isinstance(spec, str):
        stripped = spec.strip()
        if not stripped:
            raise ValueError("Swarm allocation spec must be a non-empty string.")
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                loaded = json.loads(stripped)
                if isinstance(loaded, list):
                    if not loaded:
                        raise ValueError("Swarm allocation JSON array cannot be empty.")
                    return [
                        _parse_single_swarm_allocation_spec(
                            item, default_model=default_model, base_cwd=base_cwd, allow_zero_agents=allow_zero_agents
                        )
                        for item in loaded
                    ]
                if isinstance(loaded, dict):
                    if isinstance(loaded.get("allocations"), list):
                        return [
                            _parse_single_swarm_allocation_spec(
                                item, default_model=default_model, base_cwd=base_cwd, allow_zero_agents=allow_zero_agents
                            )
                            for item in loaded["allocations"]
                        ]
                    single = _parse_single_swarm_allocation_spec(
                        loaded, default_model=default_model, base_cwd=base_cwd, allow_zero_agents=allow_zero_agents
                    )
                    return [single] if base_cwd is not None else single
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"Invalid JSON swarm allocation spec '{spec}': {exc}") from exc

        if "," in stripped:
            tokens = [t.strip() for t in stripped.split(",") if t.strip()]
            if not tokens:
                raise ValueError(f"Empty comma-separated swarm allocation spec '{spec}'.")
            return [
                _parse_single_swarm_allocation_spec(
                    tok, default_model=default_model, base_cwd=base_cwd, allow_zero_agents=allow_zero_agents
                )
                for tok in tokens
            ]

        single = _parse_single_swarm_allocation_spec(
            stripped, default_model=default_model, base_cwd=base_cwd, allow_zero_agents=allow_zero_agents
        )
        return [single] if base_cwd is not None else single

    if isinstance(spec, dict):
        if isinstance(spec.get("allocations"), list):
            return [
                _parse_single_swarm_allocation_spec(
                    item, default_model=default_model, base_cwd=base_cwd, allow_zero_agents=allow_zero_agents
                )
                for item in spec["allocations"]
            ]
        single = _parse_single_swarm_allocation_spec(
            spec, default_model=default_model, base_cwd=base_cwd, allow_zero_agents=allow_zero_agents
        )
        return [single] if base_cwd is not None else single

    raise ValueError("Swarm allocation spec must be a non-empty string, dict, or list.")


def _select_effective_gpu_tier(
    requested_gpu_tier: str,
    agent_count: int,
    model_alias: str,
    stack_preset: Optional[str] = None,
) -> Tuple[str, str]:
    min_tier, hf_repo_id, _ = _MODEL_MIN_GPU_TIER.get(
        model_alias,
        ("T4", model_alias, 16.0),
    )
    if model_alias not in _MODEL_MIN_GPU_TIER:
        try:
            from platodoom.catalog import resolve_model_profile

            prof = resolve_model_profile(model_alias)
            min_tier = str(prof.get("min_gpu_tier") or prof.get("preferred_gpu") or "T4")
            hf_repo_id = str(prof.get("quantized_hf_repo_id") or prof.get("hf_repo_id") or model_alias)
        except Exception:
            pass

    count_routed = route_gpu_for_model_count(
        num_models=agent_count,
        peak_stage_models=agent_count,
        gpu_override="auto" if requested_gpu_tier == "auto" else requested_gpu_tier,
    )
    count_tier = str(count_routed.get("gpu_type") or "T4")
    if requested_gpu_tier != "auto":
        if stack_preset:
            return requested_gpu_tier, hf_repo_id
        try:
            from platodoom.catalog import verify_vram_compatibility

            vram_res = verify_vram_compatibility(model_alias, requested_gpu_tier)
            if vram_res.get("compatible"):
                return requested_gpu_tier, str(vram_res.get("resolved_checkpoint_repo_id") or hf_repo_id)
        except Exception:
            pass
        if _GPU_TIER_RANK.get(requested_gpu_tier, 2) < _GPU_TIER_RANK.get(min_tier, 2):
            return min_tier, hf_repo_id
        return requested_gpu_tier, hf_repo_id

    effective_tier = (
        min_tier
        if _GPU_TIER_RANK.get(min_tier, 2) >= _GPU_TIER_RANK.get(count_tier, 2)
        else count_tier
    )
    return effective_tier, hf_repo_id


def allocate_directory_swarm(
    allocations: List[str | Dict[str, Any]] | str,
    config: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    auto_provision: bool = False,
    max_concurrent_sessions: Optional[int] = None,
    active_sessions_count: int = 0,
    base_cwd: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Enforces FR-2.3 & AC-4:
    - Resolves canonical project_root and deterministic dir_slug for each target directory.
    - Checks requested GPU VMs against config['colab']['max_concurrent_sessions'] (default 3),
      marking overflow allocations as 'queued_session_limit' with error_code='COLAB_SESSION_CAP_EXCEEDED'
      and a clear warning rather than failing silently.
    - Computes per-directory GPU routing and concurrency config (max_num_seqs >= agent_count),
      persists <project_root>/.sandcastle/colab_gpu_routing.json and ~/.platodoom/sessions/<dir_slug>/routing.json,
      and generates isolated bearer_token per directory session.
    """
    if not allocations:
        raise ValueError("allocations must be a non-empty list or string of directory swarm specifications.")

    cfg = config if isinstance(config, dict) else resolve_platodoom_config()
    default_model = str(cfg.get("default_model") or "qwen3.8-27b")
    max_concurrent = (
        int(max_concurrent_sessions)
        if max_concurrent_sessions is not None
        else int(cfg.get("colab", {}).get("max_concurrent_sessions") or 3)
    )

    parsed_items: List[Dict[str, Any]] = []
    raw_list = [allocations] if isinstance(allocations, (str, dict)) else list(allocations)
    for raw_item in raw_list:
        parsed = parse_swarm_allocation_spec(raw_item, default_model=default_model, base_cwd=base_cwd)
        if isinstance(parsed, list):
            parsed_items.extend(parsed)
        else:
            parsed_items.append(parsed)

    results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    allocated_gpu_sessions = max(0, int(active_sessions_count))
    newly_allocated_sessions = 0
    queued_gpu_sessions = 0
    now_epoch = time.time()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_epoch))

    for idx, item in enumerate(parsed_items):
        proj_root = Path(item["project_root"])
        dir_slug = item["dir_slug"]
        agent_count = int(item["agent_count"])
        model_alias = item["model_alias"]
        requested_gpu = item["gpu_tier"]
        stack_preset = item.get("stack_preset")

        effective_gpu, hf_repo_id = _select_effective_gpu_tier(
            requested_gpu_tier=requested_gpu,
            agent_count=agent_count,
            model_alias=model_alias,
            stack_preset=stack_preset,
        )

        routing_profile = route_gpu_for_model_count(
            num_models=agent_count,
            peak_stage_models=agent_count,
            gpu_override=effective_gpu,
            repo_dir=proj_root,
            stack_preset=stack_preset,
            model_alias=model_alias,
        )
        routing_profile["model_alias"] = model_alias
        routing_profile["hf_repo_id"] = hf_repo_id
        routing_profile["agent_count"] = agent_count
        routing_profile["requested_gpu_tier"] = requested_gpu
        routing_profile["max_num_seqs"] = max(int(routing_profile.get("max_num_seqs", 16)), agent_count)
        routing_profile["maxNumSeqs"] = routing_profile["max_num_seqs"]
        routing_profile["max_parallel_agents"] = max(
            int(routing_profile.get("max_parallel_agents", 12)),
            agent_count,
        )
        routing_profile["maxParallelAgents"] = routing_profile["max_parallel_agents"]

        if allocated_gpu_sessions < max_concurrent:
            allocation_status = "allocated"
            allocated_gpu_sessions += 1
            newly_allocated_sessions += 1
            entry_error_code = None
        else:
            allocation_status = "queued_session_limit"
            queued_gpu_sessions += 1
            entry_error_code = "COLAB_SESSION_CAP_EXCEEDED"
            warn_msg = (
                f"COLAB_SESSION_CAP_EXCEEDED: Directory '{proj_root}' ({dir_slug}) requested GPU #{allocated_gpu_sessions + queued_gpu_sessions} "
                f"({effective_gpu} for {agent_count} agents on {model_alias}), exceeding "
                f"colab.max_concurrent_sessions={max_concurrent}; marked as 'queued_session_limit'."
            )
            warnings.append(warn_msg)

        existing_rec = load_colab_endpoint_record(repo_dir=proj_root)
        bearer_token = (
            str(existing_rec.get("bearer_token"))
            if isinstance(existing_rec, dict) and existing_rec.get("bearer_token")
            else generate_ephemeral_tunnel_token()
        )

        session_dir = resolve_session_state_dir(dir_slug)
        session_routing_path = session_dir / "routing.json"
        session_endpoint_path = session_dir / "endpoint.json"
        session_colab_ep_path = session_dir / "colab_endpoint.json"
        dir_session_mirror_dir = resolve_session_state_dir(f"dir-{dir_slug}")
        sandcastle_routing_paths: List[str] = []

        record_Entry: Dict[str, Any] = {
            "index": idx,
            "directory_path": item["directory_path"],
            "project_root": str(proj_root),
            "dir_slug": dir_slug,
            "session_slug": dir_slug,
            "session_type": "swarm",
            "agent_count": agent_count,
            "agents": agent_count,
            "requested_gpu_tier": requested_gpu,
            "gpu_tier": effective_gpu,
            "gpu_type": effective_gpu,
            "accelerator_type": routing_profile.get("accelerator_type", "GPU"),
            "memory_type": routing_profile.get("memory_type", "VRAM"),
            "colab_gpu_type": routing_profile["colab_gpu_type"],
            "colab_tpu_type": routing_profile.get("colab_tpu_type"),
            "vram_gb": routing_profile["vram_gb"],
            "system_ram_gb": routing_profile["system_ram_gb"],
            "model_alias": model_alias,
            "active_model": model_alias,
            "hf_repo_id": hf_repo_id,
            "stack_preset": routing_profile.get("stack_preset"),
            "stack_mode": routing_profile.get("stack_mode"),
            "model_count": routing_profile.get("model_count", 1),
            "stack_models": routing_profile.get("stack_models", []),
            "precision_tier": routing_profile["precision_tier"],
            "max_num_seqs": routing_profile["max_num_seqs"],
            "max_model_len": routing_profile["max_model_len"],
            "allocation_status": allocation_status,
            "status": "Allocated" if allocation_status == "allocated" else "QueuedSessionLimit",
            "error_code": entry_error_code,
            "bearer_token": bearer_token,
            "endpoint_url": existing_rec.get("endpoint_url") if isinstance(existing_rec, dict) else None,
            "daemon_pid": existing_rec.get("daemon_pid") if isinstance(existing_rec, dict) else None,
            "session_routing_path": str(session_routing_path),
            "session_endpoint_path": str(session_endpoint_path),
            "updated_at": now_iso,
            "updated_at_epoch": now_epoch,
        }

        if not dry_run:
            written_routing = write_gpu_routing_profile(routing_profile, repo_dir=proj_root)
            sandcastle_routing_paths = list(written_routing.get("registry_paths", []))
            atomic_write_json(session_routing_path, routing_profile, mode=0o644)
            atomic_write_json(session_endpoint_path, record_Entry, mode=0o600)
            atomic_write_json(session_colab_ep_path, record_Entry, mode=0o600)
            atomic_write_json(dir_session_mirror_dir / "colab_endpoint.json", record_Entry, mode=0o600)
            atomic_write_json(proj_root / ".sandcastle" / "colab_endpoint.json", record_Entry, mode=0o600)

            for reg_path in resolve_colab_endpoint_registry_paths(repo_dir=proj_root):
                if isinstance(existing_rec, dict) and existing_rec.get("status") == "Online":
                    merged_ep = dict(existing_rec)
                    merged_ep.update(
                        {
                            "active_model": model_alias,
                            "hf_repo_id": hf_repo_id,
                            "agent_count": agent_count,
                            "bearer_token": bearer_token,
                        }
                    )
                    atomic_write_json(reg_path, merged_ep, mode=0o600)
                else:
                    atomic_write_json(reg_path, record_Entry, mode=0o600)

        record_Entry["routing_registry_paths"] = sandcastle_routing_paths
        results.append(record_Entry)

    cap_exceeded = queued_gpu_sessions > 0
    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "auto_provision": bool(auto_provision),
        "total_directories": len(results),
        "total_agents": sum(r["agent_count"] for r in results),
        "active_gpu_sessions": newly_allocated_sessions,
        "queued_gpu_sessions": queued_gpu_sessions,
        "max_concurrent_sessions": max_concurrent,
        "session_limit_exceeded": cap_exceeded,
        "error_code": "COLAB_SESSION_CAP_EXCEEDED" if cap_exceeded else None,
        "warnings": warnings,
        "allocations": results,
    }


def list_active_sessions(
    config: Optional[Dict[str, Any]] = None,
    include_dir_swarms: bool = True,
    project_root: Optional[Path | str] = None,
    probe_live: bool = False,
) -> List[Dict[str, Any]]:
    """Scans ~/.platodoom/sessions/*/endpoint.json and ~/.sandcastle/daemons/*/colab_qwen_endpoint.json for `platodoom sessions`."""
    platodoom_home = resolve_platodoom_home()
    sessions_root = platodoom_home / "sessions"
    discovered: Dict[str, Dict[str, Any]] = {}

    def _enrich_daemon_liveness(rec: Dict[str, Any]) -> None:
        daemon_pid = rec.get("daemon_pid")
        alive = _is_pid_alive(daemon_pid)
        rec["daemon_alive"] = alive
        rec["has_bearer_token"] = bool(rec.get("bearer_token"))
        if daemon_pid is not None and not alive:
            rec["liveness_status"] = "stale_daemon_dead"
            if str(rec.get("status") or "").lower() in ("online", "active", "allocated", ""):
                rec["status"] = "stale_daemon_dead"
        else:
            rec["liveness_status"] = "alive" if alive else "unattended"

    if sessions_root.is_dir():
        for child in sorted(sessions_root.iterdir()):
            if not child.is_dir():
                continue
            slug = child.name
            if slug.startswith("dir-"):
                base_slug = slug[len("dir-") :]
                if (sessions_root / base_slug).is_dir():
                    continue
            ep_file = child / "endpoint.json"
            if not ep_file.is_file():
                ep_file = child / "colab_endpoint.json"
            sess_file = child / "session.json"
            record: Dict[str, Any] = {}
            if ep_file.is_file():
                try:
                    loaded = json.loads(ep_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        record.update(loaded)
                        record["registry_path"] = str(ep_file)
                except Exception:
                    pass
            if sess_file.is_file():
                try:
                    sess_loaded = json.loads(sess_file.read_text(encoding="utf-8"))
                    if isinstance(sess_loaded, dict):
                        for k, v in sess_loaded.items():
                            if k != "messages" and k not in record:
                                record[k] = v
                        record["message_count"] = len(sess_loaded.get("messages", []))
                        record.setdefault("registry_path", str(sess_file))
                except Exception:
                    pass
            if not record:
                continue

            record.setdefault("session_slug", slug)
            record.setdefault("dir_slug", slug)
            record.setdefault(
                "session_type",
                "interactive" if slug.startswith("platodoom-chat-") else "pipeline",
            )
            _enrich_daemon_liveness(record)
            discovered[slug] = record

    standby_ep = platodoom_home / "standby" / "endpoint.json"
    if standby_ep.is_file():
        try:
            standby_loaded = json.loads(standby_ep.read_text(encoding="utf-8"))
            if isinstance(standby_loaded, dict):
                merged_standby = {**(discovered.get("platodoom-standby") or {}), **standby_loaded}
                merged_standby["session_slug"] = "platodoom-standby"
                merged_standby["dir_slug"] = "platodoom-standby"
                merged_standby["session_type"] = "standby"
                merged_standby["registry_path"] = str(standby_ep)
                if merged_standby.get("failure_state") == "MANUAL_REBOOT_REQUIRED":
                    merged_standby["liveness_status"] = "MANUAL_REBOOT_REQUIRED"
                    merged_standby["status"] = "Offline [MANUAL_REBOOT_REQUIRED]"
                else:
                    _enrich_daemon_liveness(merged_standby)
                discovered["platodoom-standby"] = merged_standby
        except Exception:
            pass

    if project_root is not None:
        resolved_proj = resolve_project_root(project_root)
        proj_slug = compute_dir_slug(resolved_proj)
        if proj_slug not in discovered:
            for cand_ep in (
                resolved_proj / ".sandcastle" / DEFAULT_COLAB_ENDPOINT_FILENAME,
                resolved_proj / ".sandcastle" / "colab_endpoint.json",
            ):
                if cand_ep.is_file():
                    try:
                        loaded = json.loads(cand_ep.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            loaded.setdefault("session_slug", proj_slug)
                            loaded.setdefault("dir_slug", proj_slug)
                            loaded.setdefault("session_type", "pipeline")
                            loaded["registry_path"] = str(cand_ep)
                            _enrich_daemon_liveness(loaded)
                            discovered[proj_slug] = loaded
                            break
                    except Exception:
                        pass

    if include_dir_swarms:
        daemons_root = Path.home() / ".sandcastle" / "daemons"
        if daemons_root.is_dir():
            for child in sorted(daemons_root.iterdir()):
                if not child.is_dir():
                    continue
                slug = child.name
                if slug in discovered:
                    continue
                ep_file = child / DEFAULT_COLAB_ENDPOINT_FILENAME
                if not ep_file.is_file():
                    continue
                try:
                    loaded = json.loads(ep_file.read_text(encoding="utf-8"))
                    if not isinstance(loaded, dict):
                        continue
                    loaded.setdefault("session_slug", slug)
                    loaded.setdefault("dir_slug", slug)
                    loaded.setdefault("session_type", "pipeline")
                    loaded["registry_path"] = str(ep_file)
                    _enrich_daemon_liveness(loaded)
                    discovered[slug] = loaded
                except Exception:
                    continue

    return list(discovered.values())


def _ensure_dir_0700_local(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(str(path), 0o700)
    except OSError:
        pass
    return path


def _append_jsonl_0600_local(path: Path, payload: Mapping[str, Any]) -> Path:
    _ensure_dir_0700_local(path.parent)
    redacted = redact_cache_payload(dict(payload))
    raw_line = (json.dumps(redacted, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        os.write(fd, raw_line)
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)
    return path


def _build_multi_model_agent_tasks(
    active_models: Sequence[Mapping[str, Any]],
    inactive_models: Sequence[Mapping[str, Any]],
    prompts_by_model_or_agent: Optional[Any],
    default_access_mode: str,
) -> List[Dict[str, Any]]:
    inactive_aliases = {str(m["alias"]) for m in inactive_models}
    inactive_specs = {str(m["spec"]) for m in inactive_models}
    total_agents = sum(int(m["agent_count"]) for m in active_models)
    tasks: List[Dict[str, Any]] = []

    if prompts_by_model_or_agent is None:
        for m in active_models:
            m_alias = str(m["alias"])
            m_spec = str(m["spec"])
            count = int(m["agent_count"])
            for idx in range(count):
                tasks.append(
                    {
                        "model_alias": m_alias,
                        "model_spec": m_spec,
                        "agent_index": idx + 1,
                        "model_agent_total": count,
                        "prompt": f"Agent {idx + 1}/{count} on {m_alias}: execute assigned task.",
                        "access_mode": default_access_mode,
                        "sampling_kwargs": None,
                    }
                )
        return tasks

    if isinstance(prompts_by_model_or_agent, Mapping):
        for m in active_models:
            m_alias = str(m["alias"])
            m_spec = str(m["spec"])
            m_hf = str(m.get("hf_repo_id") or "")
            count = int(m["agent_count"])
            raw_entry = None
            for key_cand in (m_alias, m_spec, m_hf):
                if key_cand and key_cand in prompts_by_model_or_agent:
                    raw_entry = prompts_by_model_or_agent[key_cand]
                    break
            if raw_entry is None:
                prompt_items: List[Any] = [
                    f"Agent {idx + 1}/{count} on {m_alias}: execute assigned task."
                    for idx in range(count)
                ]
            elif isinstance(raw_entry, str):
                prompt_items = [raw_entry for _ in range(count)]
            elif isinstance(raw_entry, Sequence) and not isinstance(raw_entry, (str, bytes)):
                seq_list = list(raw_entry)
                if len(seq_list) == count:
                    prompt_items = seq_list
                elif len(seq_list) == 1:
                    prompt_items = [seq_list[0] for _ in range(count)]
                else:
                    raise ValueError(
                        f"Model '{m_alias}' has agent_count={count}, but {len(seq_list)} prompts were provided."
                    )
            else:
                raise ValueError(
                    f"Invalid prompt specification for model '{m_alias}': expected str or sequence of {count} prompts."
                )

            for idx, item in enumerate(prompt_items):
                if isinstance(item, Mapping):
                    p_text = str(item.get("prompt") or item.get("task") or "").strip()
                    p_access = str(item.get("access_mode") or default_access_mode)
                    p_sampling = dict(item["sampling_kwargs"]) if isinstance(item.get("sampling_kwargs"), Mapping) else None
                else:
                    p_text = str(item).strip()
                    p_access = default_access_mode
                    p_sampling = None
                if not p_text:
                    raise ValueError(f"Empty prompt for model '{m_alias}' agent #{idx + 1}.")
                tasks.append(
                    {
                        "model_alias": m_alias,
                        "model_spec": m_spec,
                        "agent_index": idx + 1,
                        "model_agent_total": count,
                        "prompt": p_text,
                        "access_mode": p_access,
                        "sampling_kwargs": p_sampling,
                    }
                )
        return tasks

    if isinstance(prompts_by_model_or_agent, Sequence) and not isinstance(prompts_by_model_or_agent, (str, bytes)):
        flat_items = list(prompts_by_model_or_agent)
        if len(flat_items) != total_agents:
            raise ValueError(
                f"prompts_by_model_or_agent sequence has {len(flat_items)} items, expected total_agents={total_agents}."
            )
        # Validate no item explicitly targets an inactive 0-agent model
        for raw_it in flat_items:
            if isinstance(raw_it, Mapping):
                explicit_m = str(raw_it.get("model_alias") or raw_it.get("model") or "").strip()
                if explicit_m and (explicit_m in inactive_aliases or explicit_m in inactive_specs):
                    raise ValueError(
                        f"Task cannot target model '{explicit_m}' because its configured agent_count is 0 (unloaded)."
                    )

        cursor = 0
        for m in active_models:
            m_alias = str(m["alias"])
            m_spec = str(m["spec"])
            count = int(m["agent_count"])
            for idx in range(count):
                item = flat_items[cursor]
                cursor += 1
                if isinstance(item, Mapping):
                    p_text = str(item.get("prompt") or item.get("task") or "").strip()
                    p_access = str(item.get("access_mode") or default_access_mode)
                    p_sampling = dict(item["sampling_kwargs"]) if isinstance(item.get("sampling_kwargs"), Mapping) else None
                else:
                    p_text = str(item).strip()
                    p_access = default_access_mode
                    p_sampling = None
                if not p_text:
                    raise ValueError(f"Empty prompt at index {cursor - 1} for model '{m_alias}'.")
                tasks.append(
                    {
                        "model_alias": m_alias,
                        "model_spec": m_spec,
                        "agent_index": idx + 1,
                        "model_agent_total": count,
                        "prompt": p_text,
                        "access_mode": p_access,
                        "sampling_kwargs": p_sampling,
                    }
                )
        return tasks

    raise ValueError(
        "prompts_by_model_or_agent must be None, a Mapping[model_alias, prompts], or a Sequence of prompts."
    )


def run_single_gpu_multi_model_agents(
    model_allocations: Any,
    prompts_by_model_or_agent: Optional[Any] = None,
    *,
    gpu_tier: str = "auto",
    endpoint_url: Optional[str] = None,
    bearer_token: Optional[str] = None,
    workspace_dir: Optional[Path | str] = None,
    parent_session_slug: Optional[str] = None,
    access_mode: str = "read-only",
    max_turns: int = 8,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    use_cache: bool = True,
    force_refresh: bool = False,
    auto_provision: bool = False,
    env: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Spec 867 Orchestrator: Hosts `x` configured models on GPU 1, loads ONLY active models (`n_i > 0`)
    once into GPU 1 VRAM (`n_i == 0` models stay unloaded consuming 0 GB VRAM / 0 compute), executes
    all `n = sum(n_i)` agents concurrently in cohort-synchronized parallel GPU batches per model, and
    persists 100% of agent sessions, transcripts, and response caches on the user's local disk.
    """
    from platodoom.session import PlatoSession

    eff_env: Dict[str, str] = dict(os.environ) if env is None else dict(env)
    eff_workspace = (
        Path(workspace_dir).expanduser().resolve()
        if workspace_dir is not None
        else Path.cwd().resolve()
    )
    cfg = config if isinstance(config, dict) else resolve_platodoom_config(cwd=eff_workspace, env=eff_env)

    alloc = resolve_multi_model_agent_allocation(
        model_allocations,
        gpu_tier=gpu_tier,
        config=cfg,
    )
    resolved_gpu_tier = str(alloc["hardware_tier"])
    vram_gb = float(alloc["vram_gb"])
    all_models = list(alloc["all_models"])
    active_models = list(alloc["active_models"])
    inactive_models = list(alloc["inactive_models"])
    agent_counts: Dict[str, int] = dict(alloc["agent_counts"])
    vram_allocated_by_model: Dict[str, float] = dict(alloc["vram_allocated_by_model"])
    total_agents = int(alloc["total_agents"])

    platodoom_home = resolve_platodoom_home(env=eff_env)
    sessions_root = _ensure_dir_0700_local(platodoom_home / "sessions")
    swarm_slug = (
        str(parent_session_slug).strip()
        if (parent_session_slug and str(parent_session_slug).strip())
        else f"swarm-gpu1-{int(time.time() * 1000)}-{os.getpid()}"
    )
    orchestrator_session_dir = _ensure_dir_0700_local(sessions_root / swarm_slug)
    response_cache_dir = resolve_response_cache_dir(env=eff_env)
    shared_cache = LocalResponseCache(
        cache_dir=response_cache_dir,
        enabled=bool(use_cache),
        env=eff_env,
    )

    tasks = _build_multi_model_agent_tasks(
        active_models=active_models,
        inactive_models=inactive_models,
        prompts_by_model_or_agent=prompts_by_model_or_agent,
        default_access_mode=access_mode,
    )

    routing_profile = route_gpu_for_multi_model_agents(
        model_allocations=model_allocations,
        gpu_override=resolved_gpu_tier,
        config=cfg,
    )
    routing_path = orchestrator_session_dir / "routing.json"
    atomic_write_json(routing_path, redact_cache_payload(routing_profile), mode=0o600)

    eff_endpoint = str(endpoint_url).strip() if (endpoint_url and str(endpoint_url).strip()) else None
    eff_token = (
        str(bearer_token).strip()
        if (bearer_token and str(bearer_token).strip())
        else (
            eff_env.get("PLATODOOM_TUNNEL_TOKEN")
            or eff_env.get("SANDCASTLE_VLLM_API_KEY")
            or generate_ephemeral_tunnel_token()
        )
    )

    if not eff_endpoint and auto_provision and total_agents > 0:
        from scripts.provision_colab_serving import ensure_colab_serving_endpoint

        prov_res = ensure_colab_serving_endpoint(
            repo_dir=eff_workspace,
            gpu_override=resolved_gpu_tier,
            num_models=max(1, len(active_models)),
            peak_stage_models=max(1, total_agents),
            model_alias=str(active_models[0]["alias"]),
            stack_models=[str(m["spec"]) for m in active_models],
            agent_counts=agent_counts,
            model_allocations=model_allocations,
            headless_launch=True,
        )
        eff_endpoint = str(prov_res.get("endpoint_url") or "").strip() or None
        if prov_res.get("bearer_token"):
            eff_token = str(prov_res["bearer_token"])

    endpoint_reachable = False
    load_stack_response: Optional[Dict[str, Any]] = None
    if eff_endpoint:
        base_v1 = normalize_colab_endpoint_url(eff_endpoint)
        server_root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
        load_body = json.dumps(
            {
                "hardware_tier": resolved_gpu_tier,
                "models": [
                    {
                        "alias": m["alias"],
                        "spec": m["spec"],
                        "precision": m["precision"],
                        "hf_repo_id": m["hf_repo_id"],
                        "vram_gb": m["vram_gb"],
                        "agent_count": m["agent_count"],
                    }
                    for m in all_models
                ],
                "agent_counts": dict(agent_counts),
            }
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {eff_token}",
        }
        for load_route in ("/v1/internal/load-stack", "/v1/models/load_stack"):
            req = urllib.request.Request(
                f"{server_root}{load_route}",
                data=load_body,
                headers=headers,
                method="POST",
            )
            try:
                with _open_swarm_http_request(req, timeout=30.0) as resp:
                    load_stack_response = json.loads(resp.read().decode("utf-8"))
                endpoint_reachable = True
                break
            except urllib.error.HTTPError as http_err:
                if http_err.code == 404:
                    endpoint_reachable = True
                    continue
                err_detail = ""
                try:
                    err_detail = http_err.read().decode("utf-8", errors="replace")
                except Exception:
                    err_detail = str(http_err)
                raise RuntimeError(
                    f"GPU 1 load-stack failed with HTTP {http_err.code} on {load_route}: {err_detail}"
                ) from http_err
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                if use_cache and not force_refresh:
                    endpoint_reachable = False
                    break
                raise

    started_epoch = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_epoch))
    active_aliases = [str(m["alias"]) for m in active_models]
    inactive_aliases = [str(m["alias"]) for m in inactive_models]

    _append_jsonl_0600_local(
        orchestrator_session_dir / "transcript.jsonl",
        {
            "event": "swarm_multi_model_start",
            "timestamp": started_iso,
            "swarm_slug": swarm_slug,
            "gpu_tier": resolved_gpu_tier,
            "loaded_models": active_aliases,
            "inactive_unloaded_models": inactive_aliases,
            "agent_counts": agent_counts,
            "total_agents": total_agents,
        },
    )

    agent_sessions: List[Tuple[int, Dict[str, Any], PlatoSession]] = []
    for global_idx, task_spec in enumerate(tasks):
        m_alias = str(task_spec["model_alias"])
        a_idx = int(task_spec["agent_index"])
        agent_slug = f"{swarm_slug}-{m_alias}-agent-{a_idx}"
        sess = PlatoSession(
            model_alias=m_alias,
            access_mode=str(task_spec.get("access_mode") or access_mode),
            workspace_dir=eff_workspace,
            attach_slug=agent_slug,
            parent_session_slug=swarm_slug,
            session_type="subagent",
            config=cfg,
            endpoint_url=eff_endpoint,
            bearer_token=eff_token,
            gpu_tier=resolved_gpu_tier,
            vram_total_gb=vram_gb,
            env=eff_env,
            no_colab=not auto_provision,
            auto_attach_warm=False,
            sync_standby=False,
            resident_models=list(active_aliases),
            inactive_models=list(inactive_aliases),
            model_agent_counts=dict(agent_counts),
            response_cache=shared_cache,
        )
        agent_sessions.append((global_idx, task_spec, sess))

    barrier = threading.Barrier(parties=total_agents, timeout=15.0) if total_agents > 1 else None

    def _run_agent_worker(entry: Tuple[int, Dict[str, Any], PlatoSession]) -> Dict[str, Any]:
        g_idx, t_spec, a_sess = entry
        merged_samp: Dict[str, Any] = {"temperature": 0.0}
        if isinstance(sampling_kwargs, Mapping):
            merged_samp.update(dict(sampling_kwargs))
        if isinstance(t_spec.get("sampling_kwargs"), Mapping):
            merged_samp.update(dict(t_spec["sampling_kwargs"]))

        if barrier is not None:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass

        turn_out = a_sess.run_turn(
            user_prompt=str(t_spec["prompt"]),
            max_steps=int(max_turns),
            sampling_kwargs=merged_samp,
            stream=bool(stream),
            use_cache=bool(use_cache),
            force_refresh=bool(force_refresh),
        )
        b_meta = turn_out.get("batch_meta") if isinstance(turn_out.get("batch_meta"), dict) else None
        obs_bsize = int(b_meta["batch_size"]) if (b_meta and b_meta.get("batch_size") is not None) else None
        return {
            "global_agent_index": g_idx,
            "agent_id": f"{t_spec['model_alias']}-agent-{t_spec['agent_index']}",
            "model_alias": str(t_spec["model_alias"]),
            "model_spec": str(t_spec["model_spec"]),
            "agent_index": int(t_spec["agent_index"]),
            "session_slug": a_sess.session_slug,
            "session_dir": str(a_sess.session_dir),
            "prompt": str(t_spec["prompt"]),
            "ok": bool(turn_out.get("ok", True)),
            "response": str(turn_out.get("response") or turn_out.get("output") or ""),
            "cache_hit": bool(turn_out.get("cache_hit", False)),
            "cache_hits": int(turn_out.get("cache_hits", 0)),
            "cache_misses": int(turn_out.get("cache_misses", 0)),
            "batch_meta": b_meta,
            "observed_batch_size": obs_bsize,
        }

    agent_results: List[Dict[str, Any]] = []
    if total_agents > 0:
        try:
            with ThreadPoolExecutor(max_workers=total_agents) as pool:
                futures = [pool.submit(_run_agent_worker, ent) for ent in agent_sessions]
                for fut in futures:
                    agent_results.append(fut.result())
        except Exception as exc:
            err_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            fail_state = redact_cache_payload(
                {
                    "ok": False,
                    "session_slug": swarm_slug,
                    "session_type": "swarm_gpu1_multi_model",
                    "gpu_tier": resolved_gpu_tier,
                    "loaded_models": active_aliases,
                    "inactive_unloaded_models": inactive_aliases,
                    "agent_counts": agent_counts,
                    "error": str(exc),
                    "updated_at": err_iso,
                }
            )
            atomic_write_json(orchestrator_session_dir / "session.json", fail_state, mode=0o600)
            _append_jsonl_0600_local(
                orchestrator_session_dir / "transcript.jsonl",
                {"event": "swarm_multi_model_error", "timestamp": err_iso, "error": str(exc)},
            )
            raise

    agent_results.sort(key=lambda r: int(r["global_agent_index"]))

    server_health_snapshot: Optional[Dict[str, Any]] = None
    if eff_endpoint and endpoint_reachable:
        base_v1 = normalize_colab_endpoint_url(eff_endpoint)
        server_root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
        try:
            h_req = urllib.request.Request(
                f"{server_root}/health",
                headers={"Authorization": f"Bearer {eff_token}"},
                method="GET",
            )
            with _open_swarm_http_request(h_req, timeout=5.0) as h_resp:
                server_health_snapshot = json.loads(h_resp.read().decode("utf-8"))
        except Exception:
            server_health_snapshot = None

    per_model_observed_batch_sizes: Dict[str, Dict[str, Any]] = {}
    slot_stats_map = (
        server_health_snapshot.get("slot_batch_stats")
        if isinstance(server_health_snapshot, dict) and isinstance(server_health_snapshot.get("slot_batch_stats"), dict)
        else {}
    )
    for m in active_models:
        m_alias = str(m["alias"])
        m_results = [r for r in agent_results if r["model_alias"] == m_alias]
        obs_list = [int(r["observed_batch_size"]) for r in m_results if r.get("observed_batch_size") is not None]
        slot_stat = slot_stats_map.get(m_alias) if isinstance(slot_stats_map, dict) else None
        max_obs = max(obs_list) if obs_list else (
            int(slot_stat["max_observed_batch_size"])
            if isinstance(slot_stat, dict) and slot_stat.get("max_observed_batch_size") is not None
            else 0
        )
        per_model_observed_batch_sizes[m_alias] = {
            "configured_agents": int(m["agent_count"]),
            "observed_batch_sizes": obs_list,
            "max_observed_batch_size": max_obs,
            "all_cache_hits": all(bool(r.get("cache_hit")) for r in m_results) if m_results else False,
            "slot_batch_stats": slot_stat,
        }

    completed_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cache_stats = shared_cache.stats()
    local_persistence_paths = {
        "platodoom_home": str(platodoom_home),
        "orchestrator_session_dir": str(orchestrator_session_dir),
        "orchestrator_session_file": str(orchestrator_session_dir / "session.json"),
        "orchestrator_transcript_file": str(orchestrator_session_dir / "transcript.jsonl"),
        "orchestrator_routing_file": str(routing_path),
        "response_cache_dir": str(response_cache_dir),
        "agent_session_dirs": [r["session_dir"] for r in agent_results],
    }

    summary_payload: Dict[str, Any] = {
        "ok": all(bool(r.get("ok", True)) for r in agent_results),
        "swarm_slug": swarm_slug,
        "session_slug": swarm_slug,
        "session_type": "swarm_gpu1_multi_model",
        "gpu_tier": resolved_gpu_tier,
        "vram_gb": vram_gb,
        "configured_models": [str(m["alias"]) for m in all_models],
        "loaded_models": active_aliases,
        "inactive_unloaded_models": inactive_aliases,
        "agent_counts": agent_counts,
        "vram_allocated_by_model": vram_allocated_by_model,
        "total_agents": total_agents,
        "endpoint_reachable": endpoint_reachable,
        "load_stack_response": load_stack_response,
        "server_health": server_health_snapshot,
        "per_model_observed_batch_sizes": per_model_observed_batch_sizes,
        "agent_results": agent_results,
        "local_persistence_paths": local_persistence_paths,
        "cache_stats": cache_stats,
        "created_at": started_iso,
        "updated_at": completed_iso,
    }

    atomic_write_json(
        orchestrator_session_dir / "session.json",
        redact_cache_payload(summary_payload),
        mode=0o600,
    )
    _append_jsonl_0600_local(
        orchestrator_session_dir / "transcript.jsonl",
        {
            "event": "swarm_multi_model_complete",
            "timestamp": completed_iso,
            "swarm_slug": swarm_slug,
            "ok": summary_payload["ok"],
            "total_agents": total_agents,
            "cache_stats": cache_stats,
        },
    )
    return summary_payload
