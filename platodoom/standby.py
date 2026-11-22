"""
Deterministic, Agent-Free Standby Accelerator Registry & Watchdog (Spec 866).

Owns `~/.platodoom/standby/endpoint.json` (or `$PLATODOOM_HOME/standby/endpoint.json`),
enforces small-hardware-only background warming (`T4` / `v5e-1`), and executes a bounded,
100% programmatic state machine (`ONLINE` -> `RECOVERING` [max 1 reboot] -> `MANUAL_REBOOT_REQUIRED`)
with zero LLM agent invocations and guaranteed `exit code 0`.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from curation.api_client import (
    normalize_colab_endpoint_url,
    probe_colab_qwen_endpoint,
    resolve_serving_bearer_token,
)
from platodoom.catalog import (
    GPU_TIER_SPECS,
    normalize_hardware_tier,
    validate_model_gpu_compatibility,
)
from platodoom.config import (
    atomic_write_json,
    resolve_platodoom_config,
    resolve_platodoom_home,
)
from platodoom.security import generate_ephemeral_tunnel_token

SMALL_STANDBY_TIERS: FrozenSet[str] = frozenset({"T4", "v5e-1"})
FORBIDDEN_STANDBY_TIERS: FrozenSet[str] = frozenset(
    {"L4", "v6e-1", "A100", "A100-80GB", "H100", "G4"}
)
STANDBY_SESSION_SLUG: str = "platodoom-standby"
MANUAL_REBOOT_REQUIRED: str = "MANUAL_REBOOT_REQUIRED"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_standby_dir(env: Optional[Dict[str, str]] = None) -> Path:
    """Resolves `<platodoom_home>/standby` and enforces `0700` directory permissions."""
    home_dir = resolve_platodoom_home(env=env)
    standby_dir = home_dir / "standby"
    try:
        home_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(home_dir), 0o700)
        standby_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(standby_dir), 0o700)
    except OSError:
        pass
    return standby_dir


def resolve_standby_endpoint_path(env: Optional[Dict[str, str]] = None) -> Path:
    """Returns `<platodoom_home>/standby/endpoint.json`."""
    return resolve_standby_dir(env=env) / "endpoint.json"


def resolve_standby_lock_path(env: Optional[Dict[str, str]] = None) -> Path:
    """Returns `<platodoom_home>/standby/watchdog.lock`."""
    return resolve_standby_dir(env=env) / "watchdog.lock"


def read_standby_endpoint(env: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Reads `<platodoom_home>/standby/endpoint.json`, returning `None` if absent or unreadable."""
    ep_path = resolve_standby_endpoint_path(env=env)
    if not ep_path.is_file():
        return None
    try:
        loaded = json.loads(ep_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None


def is_small_standby_tier(gpu_tier: Optional[str]) -> bool:
    """Returns `True` iff `gpu_tier` canonicalizes to an allowed small standby accelerator (`T4` or `v5e-1`)."""
    if gpu_tier is None:
        return False
    cleaned = str(gpu_tier).strip()
    if not cleaned or cleaned.lower() == "auto":
        return False
    try:
        norm = normalize_hardware_tier(cleaned, allow_auto=False)
    except ValueError:
        return False
    return norm in SMALL_STANDBY_TIERS


def write_standby_endpoint(
    record: Dict[str, Any],
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Atomically persists `<platodoom_home>/standby/endpoint.json` with `0600` owner-only permissions."""
    now_iso = _utc_now_iso()
    now_epoch = time.time()
    raw_url = str(record.get("endpoint_url") or "").strip()
    norm_url = normalize_colab_endpoint_url(raw_url) if raw_url else ""

    raw_tier = str(
        record.get("gpu_tier")
        or record.get("gpu_type")
        or record.get("colab_gpu_type")
        or "T4"
    ).strip()
    try:
        norm_tier = normalize_hardware_tier(raw_tier, allow_auto=False)
    except ValueError:
        norm_tier = "T4"

    tier_spec = GPU_TIER_SPECS.get(norm_tier, GPU_TIER_SPECS["T4"])
    active_model = str(
        record.get("active_model")
        or record.get("model")
        or record.get("model_alias")
        or "qwen-9b"
    ).strip()
    resident_models_raw = record.get("resident_models") or record.get("served_models")
    if isinstance(resident_models_raw, list) and resident_models_raw:
        resident_models = [str(m) for m in resident_models_raw if str(m).strip()]
    else:
        resident_models = [active_model] if norm_url else []

    status = str(record.get("status") or ("Online" if norm_url else "Offline"))
    failure_state = record.get("failure_state")
    watchdog_state = str(
        record.get("watchdog_state")
        or (
            MANUAL_REBOOT_REQUIRED
            if failure_state == MANUAL_REBOOT_REQUIRED
            else ("ONLINE" if status == "Online" else ("RECOVERING" if status == "Recovering" else "OFFLINE"))
        )
    )

    payload: Dict[str, Any] = {
        **record,
        "session_slug": STANDBY_SESSION_SLUG,
        "dir_slug": STANDBY_SESSION_SLUG,
        "session_type": "standby",
        "endpoint_url": norm_url,
        "chat_completions_url": f"{norm_url}/chat/completions" if norm_url else "",
        "models_url": f"{norm_url}/models" if norm_url else "",
        "status": status,
        "watchdog_state": watchdog_state,
        "failure_state": failure_state,
        "offline_reason": record.get("offline_reason"),
        "last_error": record.get("last_error"),
        "model": active_model,
        "active_model": active_model,
        "model_alias": active_model,
        "resident_models": resident_models,
        "gpu_tier": norm_tier,
        "gpu_type": norm_tier,
        "colab_gpu_type": norm_tier,
        "accelerator_type": str(record.get("accelerator_type") or tier_spec["accelerator_type"]),
        "memory_type": str(record.get("memory_type") or tier_spec["memory_type"]),
        "vram_total_gb": float(
            record.get("vram_total_gb")
            if record.get("vram_total_gb") is not None
            else tier_spec.get("usable_vram_gb", tier_spec["vram_gb"])
        ),
        "vram_gb": int(record.get("vram_gb") or tier_spec["vram_gb"]),
        "bearer_token": str(record.get("bearer_token") or record.get("auth_token") or ""),
        "daemon_pid": record.get("daemon_pid"),
        "reboot_attempts_since_online": int(record.get("reboot_attempts_since_online", 0)),
        "checked_at": str(record.get("checked_at") or now_iso),
        "updated_at": now_iso,
        "updated_at_epoch": now_epoch,
        "heartbeat_epoch": float(
            record.get("heartbeat_epoch")
            if record.get("heartbeat_epoch") is not None
            else now_epoch
        ),
    }

    ep_path = resolve_standby_endpoint_path(env=env)
    atomic_write_json(ep_path, payload, mode=0o600)
    return payload


def clear_standby_manual_reboot_flag(
    env: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Clears `failure_state: MANUAL_REBOOT_REQUIRED` in `<platodoom_home>/standby/endpoint.json` if present."""
    existing = read_standby_endpoint(env=env)
    if not existing:
        return None
    if existing.get("failure_state") != MANUAL_REBOOT_REQUIRED and existing.get("watchdog_state") != MANUAL_REBOOT_REQUIRED:
        return existing
    updated = dict(existing)
    updated["failure_state"] = None
    updated["offline_reason"] = None
    updated["last_error"] = None
    if updated.get("status") != "Online":
        updated["watchdog_state"] = "OFFLINE"
    updated["reboot_attempts_since_online"] = 0
    return write_standby_endpoint(updated, env=env)


def resolve_standby_target_spec(
    config: Optional[Dict[str, Any]] = None,
    model_alias: Optional[str] = None,
    gpu_tier: Optional[str] = None,
    existing_record: Optional[Dict[str, Any]] = None,
    workspace_dir: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Resolves and validates the standby small-accelerator target (`T4` or `v5e-1` ONLY).
    Larger hardware tiers (`L4`, `v6e-1`, `A100`, `A100-80GB`, `H100`, `G4`) and models
    exceeding `16.0 GB` memory are strictly rejected from background auto-warming.
    """
    cfg = config if isinstance(config, dict) else resolve_platodoom_config(cwd=workspace_dir, env=env)
    rec = existing_record if isinstance(existing_record, dict) else {}

    target_model = str(
        model_alias
        or rec.get("active_model")
        or rec.get("model")
        or cfg.get("default_model")
        or "qwen-9b"
    ).strip()

    existing_tier = str(rec.get("gpu_tier") or rec.get("gpu_type") or rec.get("colab_gpu_type") or "").strip()
    cfg_gpu = str(cfg.get("default_gpu") or "").strip()

    if gpu_tier and str(gpu_tier).strip():
        raw_tier = str(gpu_tier).strip()
    elif existing_tier and is_small_standby_tier(existing_tier):
        raw_tier = existing_tier
    elif cfg_gpu and cfg_gpu.lower() != "auto":
        raw_tier = cfg_gpu
    else:
        raw_tier = "auto"

    if raw_tier.lower() == "auto":
        t4_ok, _ = validate_model_gpu_compatibility(target_model, "T4", config=cfg)
        if t4_ok:
            norm_tier = "T4"
        else:
            v5e_ok, v5e_diag = validate_model_gpu_compatibility(target_model, "v5e-1", config=cfg)
            if v5e_ok:
                norm_tier = "v5e-1"
            else:
                return {
                    "allowed": False,
                    "model_alias": target_model,
                    "gpu_tier": "T4",
                    "reason": (
                        f"Configured standby model '{target_model}' cannot fit on a small standby accelerator "
                        f"(T4 / v5e-1): {v5e_diag}"
                    ),
                }
    else:
        try:
            norm_tier = normalize_hardware_tier(raw_tier, allow_auto=False)
        except ValueError as exc:
            return {
                "allowed": False,
                "model_alias": target_model,
                "gpu_tier": raw_tier,
                "reason": str(exc),
            }

    if norm_tier in FORBIDDEN_STANDBY_TIERS or norm_tier not in SMALL_STANDBY_TIERS:
        return {
            "allowed": False,
            "model_alias": target_model,
            "gpu_tier": norm_tier,
            "reason": (
                f"Standby watchdog strictly monitors small accelerators ({sorted(SMALL_STANDBY_TIERS)}) only; "
                f"larger hardware tier '{norm_tier}' is never auto-warmed in the background."
            ),
        }

    compatible, diag = validate_model_gpu_compatibility(target_model, norm_tier, config=cfg)
    if not compatible:
        return {
            "allowed": False,
            "model_alias": target_model,
            "gpu_tier": norm_tier,
            "reason": (
                f"Standby model '{target_model}' is incompatible with small tier '{norm_tier}': {diag}"
            ),
        }

    tier_spec = GPU_TIER_SPECS[norm_tier]
    return {
        "allowed": True,
        "model_alias": target_model,
        "gpu_tier": norm_tier,
        "vram_total_gb": float(tier_spec.get("usable_vram_gb", tier_spec["vram_gb"])),
        "vram_gb": int(tier_spec["vram_gb"]),
        "memory_type": str(tier_spec["memory_type"]),
        "accelerator_type": str(tier_spec["accelerator_type"]),
        "reason": None,
    }


def sync_standby_from_live_small_endpoint(
    record: Dict[str, Any],
    env: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synchronizes a verified live small-tier (`T4` or `v5e-1`) Colab endpoint into
    `<platodoom_home>/standby/endpoint.json`, clearing any prior `MANUAL_REBOOT_REQUIRED` state.
    Larger tiers (`L4`, `v6e-1`, `A100`, `A100-80GB`, `H100`, `G4`) are ignored.
    """
    if not isinstance(record, dict):
        return None
    raw_url = str(record.get("endpoint_url") or "").strip()
    if not raw_url:
        return None
    raw_tier = (
        record.get("gpu_tier")
        or record.get("gpu_type")
        or record.get("colab_gpu_type")
    )
    if not is_small_standby_tier(str(raw_tier) if raw_tier is not None else None):
        return None

    norm_tier = normalize_hardware_tier(str(raw_tier), allow_auto=False)
    now_iso = _utc_now_iso()
    now_epoch = time.time()
    existing = read_standby_endpoint(env=env) or {}

    active_model = str(
        record.get("active_model")
        or record.get("model")
        or record.get("model_alias")
        or existing.get("active_model")
        or "qwen-9b"
    ).strip()
    served = record.get("resident_models") or record.get("served_models")
    resident_models = (
        [str(m) for m in served if str(m).strip()]
        if isinstance(served, list) and served
        else [active_model]
    )
    tok = str(
        record.get("bearer_token")
        or record.get("auth_token")
        or existing.get("bearer_token")
        or ""
    ).strip()

    merged: Dict[str, Any] = {
        **existing,
        "endpoint_url": raw_url,
        "status": "Online",
        "watchdog_state": "ONLINE",
        "failure_state": None,
        "offline_reason": None,
        "last_error": None,
        "model": active_model,
        "active_model": active_model,
        "model_alias": active_model,
        "resident_models": resident_models,
        "gpu_tier": norm_tier,
        "gpu_type": norm_tier,
        "colab_gpu_type": norm_tier,
        "vram_total_gb": record.get("vram_total_gb") or GPU_TIER_SPECS[norm_tier].get("usable_vram_gb", 15.8),
        "bearer_token": tok,
        "daemon_pid": record.get("daemon_pid") or os.getpid(),
        "reboot_attempts_since_online": 0,
        "checked_at": now_iso,
        "heartbeat_epoch": now_epoch,
    }
    return write_standby_endpoint(merged, env=env)


def provision_standby_small_instance(
    model_alias: str,
    gpu_tier: str = "T4",
    workspace_dir: Optional[Path | str] = None,
    timeout_sec: float = 420.0,
) -> Dict[str, Any]:
    """
    100% programmatic, zero-LLM single-attempt Colab standby provisioner for `T4` / `v5e-1`.
    """
    if not is_small_standby_tier(gpu_tier):
        raise ValueError(
            f"Refusing to provision standby instance on non-small tier {gpu_tier!r}; "
            f"only {sorted(SMALL_STANDBY_TIERS)} are permitted."
        )
    norm_tier = normalize_hardware_tier(gpu_tier, allow_auto=False)
    from scripts.provision_colab_serving import (
        ensure_colab_serving_endpoint,
        terminate_directory_daemon,
    )

    ephemeral_token = generate_ephemeral_tunnel_token()
    repo_root = Path(workspace_dir).expanduser().resolve() if workspace_dir else Path.cwd().resolve()
    bounded_wait_sec = max(1.0, float(timeout_sec))
    try:
        return ensure_colab_serving_endpoint(
            repo_dir=repo_root,
            timeout_sec=min(bounded_wait_sec, 5.0),
            auto_provision=True,
            include_global=False,
            headless_launch=True,
            session_slug=STANDBY_SESSION_SLUG,
            model_alias=model_alias,
            bearer_token=ephemeral_token,
            gpu_override=norm_tier,
            allow_cross_registry_warm=False,
            explicit_gpu_requested=True,
            max_wait_sec=bounded_wait_sec,
        )
    except BaseException:
        try:
            terminate_directory_daemon(repo_root, session_slug=STANDBY_SESSION_SLUG)
        except Exception:
            pass
        raise


def _is_probe_healthy_on_small_endpoint(probe: Dict[str, Any]) -> bool:
    """Returns `True` when `/v1/models` responded HTTP 200 with loaded models and valid auth."""
    if not isinstance(probe, dict):
        return False
    if probe.get("auth_error") or probe.get("unauthorized"):
        return False
    if probe.get("online"):
        return True
    if probe.get("alive") and int(probe.get("status_code") or 0) == 200 and bool(probe.get("models")):
        return True
    return False


def run_standby_watchdog_once(
    *,
    workspace_dir: Optional[Path | str] = None,
    model_alias: Optional[str] = None,
    gpu_tier: Optional[str] = None,
    probe_timeout_sec: float = 3.0,
    force_reboot: bool = False,
    check_only: bool = False,
    reboot_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    probe_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Executes one deterministic, zero-LLM standby watchdog cycle (`Spec 866 §3.2` & `§4.2`).

    States:
    - `ONLINE`: Standby endpoint responds healthy in `< 5 s` -> updates `heartbeat_epoch` and returns `exit_code=0`.
    - `RECOVERING`: Endpoint is down -> attempts **at most 1** bounded programmatic reboot on `T4` / `v5e-1`.
    - `MANUAL_REBOOT_REQUIRED`: Single reboot failed -> writes `status="Offline"` and
      `failure_state="MANUAL_REBOOT_REQUIRED"` to `~/.platodoom/standby/endpoint.json` and returns `exit_code=0`.
    """
    t0 = time.perf_counter()
    eff_probe = probe_fn or probe_colab_qwen_endpoint
    bounded_probe_timeout = min(max(float(probe_timeout_sec), 0.2), 3.5)

    lock_path = resolve_standby_lock_path(env=env)
    lock_fd: Optional[int] = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "ok": True,
                "status": "Locked",
                "watchdog_state": "LOCKED_SKIP",
                "failure_state": None,
                "reboot_attempted": False,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "exit_code": 0,
            }

        existing = read_standby_endpoint(env=env)
        spec = resolve_standby_target_spec(
            config=config,
            model_alias=model_alias,
            gpu_tier=gpu_tier,
            existing_record=existing,
            workspace_dir=workspace_dir,
            env=env,
        )
        if not spec["allowed"]:
            return {
                "ok": False,
                "status": "PolicySkipped",
                "watchdog_state": "POLICY_SKIPPED",
                "policy_rejected": True,
                "failure_state": (existing or {}).get("failure_state"),
                "model_alias": spec["model_alias"],
                "gpu_tier": spec["gpu_tier"],
                "reason": spec["reason"],
                "reboot_attempted": False,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "exit_code": 0,
            }

        target_model = str(spec["model_alias"])
        target_tier = str(spec["gpu_tier"])
        now_iso = _utc_now_iso()
        now_epoch = time.time()

        # 1. Probe existing standby endpoint if present and recorded on an allowed small tier
        existing_url = str((existing or {}).get("endpoint_url") or "").strip()
        existing_tier = (existing or {}).get("gpu_tier") or (existing or {}).get("gpu_type")
        if existing_url and (existing_tier is None or is_small_standby_tier(str(existing_tier))):
            tok = resolve_serving_bearer_token(record=existing)
            probe_token = None if tok == "EMPTY" else tok
            probe = eff_probe(
                existing_url,
                api_key=probe_token,
                bearer_token=probe_token,
                timeout_sec=bounded_probe_timeout,
                verify_inference=False,
                expected_model=target_model,
            )
            if _is_probe_healthy_on_small_endpoint(probe):
                served_models = probe.get("models") or []
                live_model = (
                    target_model
                    if probe.get("online")
                    else (str(served_models[0]) if served_models else target_model)
                )
                healthy_rec = write_standby_endpoint(
                    {
                        **(existing or {}),
                        "endpoint_url": existing_url,
                        "status": "Online",
                        "watchdog_state": "ONLINE",
                        "failure_state": None,
                        "offline_reason": None,
                        "last_error": None,
                        "model": live_model,
                        "active_model": live_model,
                        "resident_models": served_models or [live_model],
                        "gpu_tier": normalize_hardware_tier(str(existing_tier or target_tier), allow_auto=False),
                        "bearer_token": probe_token or "",
                        "reboot_attempts_since_online": 0,
                        "checked_at": now_iso,
                        "heartbeat_epoch": now_epoch,
                    },
                    env=env,
                )
                return {
                    "ok": True,
                    "status": "Online",
                    "watchdog_state": "ONLINE",
                    "failure_state": None,
                    "endpoint_url": healthy_rec["endpoint_url"],
                    "model_alias": healthy_rec["active_model"],
                    "gpu_tier": healthy_rec["gpu_tier"],
                    "reboot_attempted": False,
                    "heartbeat_epoch": healthy_rec["heartbeat_epoch"],
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "exit_code": 0,
                }

        # 2. If check_only was requested, report offline state without rebooting
        if check_only:
            cur_failure = (existing or {}).get("failure_state")
            return {
                "ok": False,
                "status": "Offline",
                "watchdog_state": MANUAL_REBOOT_REQUIRED if cur_failure == MANUAL_REBOOT_REQUIRED else "OFFLINE",
                "failure_state": cur_failure,
                "model_alias": target_model,
                "gpu_tier": target_tier,
                "reboot_attempted": False,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "exit_code": 0,
            }

        # 3. Guard against infinite retry loops when already halted in MANUAL_REBOOT_REQUIRED
        if (existing or {}).get("failure_state") == MANUAL_REBOOT_REQUIRED and not force_reboot:
            halted_rec = write_standby_endpoint(
                {
                    **(existing or {}),
                    "status": "Offline",
                    "watchdog_state": MANUAL_REBOOT_REQUIRED,
                    "failure_state": MANUAL_REBOOT_REQUIRED,
                    "model": target_model,
                    "active_model": target_model,
                    "gpu_tier": target_tier,
                    "checked_at": now_iso,
                },
                env=env,
            )
            return {
                "ok": False,
                "status": "Offline",
                "watchdog_state": MANUAL_REBOOT_REQUIRED,
                "failure_state": MANUAL_REBOOT_REQUIRED,
                "model_alias": target_model,
                "gpu_tier": target_tier,
                "reboot_attempted": False,
                "error": halted_rec.get("last_error") or halted_rec.get("offline_reason"),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "exit_code": 0,
            }

        # 4. Before spinning up a new VM, adopt any already-warm small tier (T4 / v5e-1) active in user sessions
        if reboot_fn is None and probe_fn is None:
            try:
                from scripts.provision_colab_serving import discover_warm_colab_endpoints

                warm_small = discover_warm_colab_endpoints(
                    repo_dir=workspace_dir,
                    target_model=target_model,
                    exclude_session_slug=STANDBY_SESSION_SLUG,
                    probe_timeout_sec=min(0.65, bounded_probe_timeout),
                    config=config,
                    allowed_tiers=SMALL_STANDBY_TIERS,
                )
                warm_cand = warm_small.get("candidate")
                if warm_small.get("match_type") in ("exact_model_loaded", "compatible_warm_switch") and isinstance(warm_cand, dict):
                    synced = sync_standby_from_live_small_endpoint(warm_cand, env=env)
                    if synced:
                        return {
                            "ok": True,
                            "status": "Online",
                            "watchdog_state": "ONLINE",
                            "failure_state": None,
                            "endpoint_url": synced["endpoint_url"],
                            "model_alias": synced["active_model"],
                            "gpu_tier": synced["gpu_tier"],
                            "adopted_existing_warm": True,
                            "reboot_attempted": False,
                            "heartbeat_epoch": synced["heartbeat_epoch"],
                            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "exit_code": 0,
                        }
            except Exception:
                pass

        # 5. Transition to RECOVERING and perform AT MOST ONE bounded programmatic reboot
        write_standby_endpoint(
            {
                **(existing or {}),
                "endpoint_url": "",
                "status": "Recovering",
                "watchdog_state": "RECOVERING",
                "failure_state": None,
                "offline_reason": "Standby endpoint unreachable; attempting single programmatic recovery.",
                "model": target_model,
                "active_model": target_model,
                "gpu_tier": target_tier,
                "reboot_attempts_since_online": 1,
                "checked_at": _utc_now_iso(),
            },
            env=env,
        )

        eff_reboot = reboot_fn or provision_standby_small_instance
        reboot_error: Optional[str] = None
        recovered_record: Optional[Dict[str, Any]] = None
        try:
            reboot_res = eff_reboot(
                model_alias=target_model,
                gpu_tier=target_tier,
                workspace_dir=workspace_dir,
            )
            if isinstance(reboot_res, dict) and str(reboot_res.get("endpoint_url") or "").strip():
                rec_url = str(reboot_res["endpoint_url"]).strip()
                rec_tok = resolve_serving_bearer_token(record=reboot_res)
                eff_rec_tok = None if rec_tok == "EMPTY" else rec_tok
                post_probe = eff_probe(
                    rec_url,
                    api_key=eff_rec_tok,
                    bearer_token=eff_rec_tok,
                    timeout_sec=bounded_probe_timeout,
                    verify_inference=False,
                    expected_model=target_model,
                )
                if _is_probe_healthy_on_small_endpoint(post_probe):
                    recovered_record = write_standby_endpoint(
                        {
                            **(existing or {}),
                            **reboot_res,
                            "endpoint_url": rec_url,
                            "status": "Online",
                            "watchdog_state": "ONLINE",
                            "failure_state": None,
                            "offline_reason": None,
                            "last_error": None,
                            "model": target_model,
                            "active_model": target_model,
                            "resident_models": post_probe.get("models") or [target_model],
                            "gpu_tier": target_tier,
                            "bearer_token": eff_rec_tok or "",
                            "daemon_pid": reboot_res.get("daemon_pid") or os.getpid(),
                            "reboot_attempts_since_online": 0,
                            "checked_at": _utc_now_iso(),
                            "heartbeat_epoch": time.time(),
                        },
                        env=env,
                    )
                else:
                    reboot_error = (
                        f"Post-reboot health probe failed for {rec_url}: "
                        f"{post_probe.get('error') or 'endpoint not serving model'}"
                    )
            else:
                reboot_error = "Programmatic standby reboot returned no live endpoint_url."
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            reboot_error = f"{type(exc).__name__}: {exc}"

        if recovered_record is not None:
            return {
                "ok": True,
                "status": "Online",
                "watchdog_state": "ONLINE",
                "failure_state": None,
                "endpoint_url": recovered_record["endpoint_url"],
                "model_alias": recovered_record["active_model"],
                "gpu_tier": recovered_record["gpu_tier"],
                "reboot_attempted": True,
                "heartbeat_epoch": recovered_record["heartbeat_epoch"],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "exit_code": 0,
            }

        # 6. Single reboot failed -> write MANUAL_REBOOT_REQUIRED and exit 0 cleanly without crashing
        diag_msg = str(reboot_error or "Standby programmatic recovery failed.")
        failed_rec = write_standby_endpoint(
            {
                **(existing or {}),
                "endpoint_url": "",
                "status": "Offline",
                "watchdog_state": MANUAL_REBOOT_REQUIRED,
                "failure_state": MANUAL_REBOOT_REQUIRED,
                "offline_reason": diag_msg,
                "last_error": diag_msg,
                "model": target_model,
                "active_model": target_model,
                "resident_models": [],
                "gpu_tier": target_tier,
                "reboot_attempts_since_online": 1,
                "checked_at": _utc_now_iso(),
            },
            env=env,
        )
        return {
            "ok": False,
            "status": "Offline",
            "watchdog_state": MANUAL_REBOOT_REQUIRED,
            "failure_state": MANUAL_REBOOT_REQUIRED,
            "endpoint_url": "",
            "model_alias": failed_rec["active_model"],
            "gpu_tier": failed_rec["gpu_tier"],
            "reboot_attempted": True,
            "error": diag_msg,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "exit_code": 0,
        }
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(lock_fd)
            except OSError:
                pass


def build_standby_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="platodoom standby",
        description="Deterministic, agent-free standby watchdog for small Colab accelerators (T4 / v5e-1).",
    )
    parser.add_argument("action", nargs="?", default="watchdog", help="Action: watchdog (default), status, or reset")
    parser.add_argument("-m", "--model", type=str, default=None, help="Small default model alias to keep warm")
    parser.add_argument("-g", "--gpu", type=str, default=None, help="Small hardware tier (T4 or v5e-1 only)")
    parser.add_argument("--check", "--check-only", dest="check_only", action="store_true", help="Probe health without attempting recovery")
    parser.add_argument("--force-reboot", "--reset", dest="force_reboot", action="store_true", help="Clear MANUAL_REBOOT_REQUIRED and permit 1 programmatic reboot")
    parser.add_argument("--timeout", type=float, default=3.0, help="Health probe timeout in seconds (<= 3.5s)")
    parser.add_argument("--workspace-dir", type=str, default=None, help="Optional workspace directory for config resolution")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON status")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for `platodoom watchdog` / `platodoom standby`. Always exits cleanly with `0`."""
    args_list = list(argv if argv is not None else sys.argv[1:])
    as_json = "--json" in args_list
    try:
        parser = build_standby_arg_parser()
        args, _unknown = parser.parse_known_args(args_list)
        action = str(args.action or "watchdog").strip().lower()

        if action == "reset":
            cleared = clear_standby_manual_reboot_flag()
            payload = {
                "ok": True,
                "action": "reset",
                "record": cleared,
                "exit_code": 0,
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print("Cleared standby MANUAL_REBOOT_REQUIRED flag.")
            return 0

        is_check_only = bool(args.check_only or action == "status")
        res = run_standby_watchdog_once(
            workspace_dir=args.workspace_dir,
            model_alias=args.model,
            gpu_tier=args.gpu,
            probe_timeout_sec=args.timeout,
            force_reboot=bool(args.force_reboot),
            check_only=is_check_only,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            state = res.get("watchdog_state")
            tier = res.get("gpu_tier", "T4")
            mod = res.get("model_alias", "")
            ep = res.get("endpoint_url") or "none"
            if state == "ONLINE":
                print(f"[standby · ONLINE] {mod} on {tier} -> {ep} ({res.get('elapsed_ms', 0.0)}ms)")
            elif state == MANUAL_REBOOT_REQUIRED:
                print(
                    f"[standby · MANUAL_REBOOT_REQUIRED] {mod} on {tier} is offline ({res.get('error')}). "
                    "Restart `platodoom` to reboot."
                )
            elif state == "POLICY_SKIPPED":
                print(f"[standby · POLICY_SKIPPED] {res.get('reason')}")
            else:
                print(f"[standby · {state}] status={res.get('status')}")
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        diag = f"Unhandled watchdog error ({type(exc).__name__}): {exc}"
        try:
            existing = read_standby_endpoint() or {}
            write_standby_endpoint(
                {
                    **existing,
                    "endpoint_url": "",
                    "status": "Offline",
                    "watchdog_state": MANUAL_REBOOT_REQUIRED,
                    "failure_state": MANUAL_REBOOT_REQUIRED,
                    "offline_reason": diag,
                    "last_error": diag,
                }
            )
        except Exception:
            pass
        if as_json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "status": "Offline",
                        "watchdog_state": MANUAL_REBOOT_REQUIRED,
                        "failure_state": MANUAL_REBOOT_REQUIRED,
                        "error": diag,
                        "exit_code": 0,
                    },
                    indent=2,
                )
            )
        else:
            print(f"[standby · MANUAL_REBOOT_REQUIRED] {diag}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
