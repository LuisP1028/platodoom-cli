#!/usr/bin/env python3
"""
Global CLI Entrypoint & Subcommand Dispatcher for platodoom.
Document ID: platodoom_cli_859_v1
Governing Standards: platodoom specification (FR-1, FR-2, FR-3, FR-4, AC-1..4),
visual_taste_direction_859_v1, LANGUAGE.md, RULE[user_global]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
from typing import Any, Dict, List, Optional

PLATODOOM_VERSION = "1.0.0"


def fast_non_hanging_exit(exit_code: int = 0) -> None:
    """
    Flushes stdout/stderr and terminates immediately via os._exit(exit_code)
    to prevent Py_FinalizeEx C++/gRPC/socket destructor hangs (<= 1.0s budget).
    """
    try:
        from platodoom.repl import emit_terminal_goodbye

        emit_terminal_goodbye(force=False)
    except Exception:
        pass
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(int(exit_code))


def _atomic_write_json_mode(target_path: Path, data: Dict[str, Any], mode: int = 0o600) -> Path:
    target_path = Path(target_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=f".{os.getpid()}.tmp",
        dir=str(target_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, mode)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(target_path))
        try:
            os.chmod(str(target_path), mode)
        except OSError:
            pass
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return target_path


def _resolve_platodoom_home(env: Optional[Dict[str, str]] = None) -> Path:
    eff = env if env is not None else os.environ
    raw = eff.get("PLATODOOM_CONFIG_DIR") or eff.get("PLATODOOM_HOME")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser().resolve()
    return (Path.home() / ".platodoom").resolve()


def _cmd_models(
    as_json: bool = False,
    tier_filter: Optional[str] = None,
    show_stacks: bool = False,
) -> int:
    """
    Lists curated model catalog across all 8 Colab GPU/TPU hardware tiers and 14 priority Qwen/DeepSeek models.
    Lists all open-source HuggingFace catalog models (with `qwen3.8-27b` as default [1])
    and Solo/Multi-Model Stack presets across Colab GPU/TPU hardware tiers.
    """
    from platodoom.catalog import (
        GPU_TIER_SPECS,
        PRIORITY_TEXT_MODELS,
        get_merged_model_catalog,
        get_stack_presets_for_tier,
        normalize_hardware_tier,
    )
    from platodoom.config import resolve_platodoom_config

    cfg = resolve_platodoom_config()
    catalog = get_merged_model_catalog(cfg)
    norm_tier = normalize_hardware_tier(tier_filter) if tier_filter else None

    ordered_keys = [k for k in PRIORITY_TEXT_MODELS if k in catalog] + [
        k for k in catalog if k not in PRIORITY_TEXT_MODELS
    ]
    if norm_tier:
        display_keys = [
            k
            for k in ordered_keys
            if norm_tier in (catalog[k].get("supported_hardware_tiers") or catalog[k].get("compatible_gpus") or [])
        ]
    else:
        display_keys = ordered_keys

    default_model = cfg.get("default_model", "qwen3.8-27b")

    if as_json:
        models_list = [catalog[k] for k in display_keys]
        priority_models_list = [catalog[k] for k in PRIORITY_TEXT_MODELS if k in catalog]
        if norm_tier:
            stack_presets_out = {norm_tier: get_stack_presets_for_tier(norm_tier, cfg)}
        else:
            stack_presets_out = {tier: get_stack_presets_for_tier(tier, cfg) for tier in GPU_TIER_SPECS}
        payload: Dict[str, Any] = {
            "ok": True,
            "default_model": default_model,
            "filtered_tier": norm_tier,
            "hardware_tiers": {k: dict(v) for k, v in GPU_TIER_SPECS.items()},
            "priority_models": priority_models_list,
            "stack_presets": stack_presets_out,
            "models": models_list,
            "catalog": catalog,
        }
        print(json.dumps(payload, indent=2))
        return 0

    from platodoom.repl import render_hardware_stack_table, render_startup_model_table

    if not show_stacks:
        for line in render_startup_model_table(catalog, default_model=default_model, ordered_aliases=display_keys):
            print(line)
    if show_stacks or norm_tier:
        for line in render_hardware_stack_table(tier_filter=norm_tier):
            print(line)
    return 0


_CONFIG_SECRET_FIELD_NAMES = frozenset(
    {
        "hf_token",
        "ngrok_authtoken",
        "ngrok_token",
        "bearer_token",
        "auth_token",
        "api_key",
        "tunnel_token",
        "password",
        "secret",
        "client_secret",
    }
)


def _redact_config_for_display(obj: Any, *, key_name: Optional[str] = None) -> Any:
    """Masks sensitive credential values before printing config output to stdout."""
    from platodoom.onboarding import mask_secret_preview

    if isinstance(obj, dict):
        return {k: _redact_config_for_display(v, key_name=str(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_config_for_display(v) for v in obj]
    if key_name and key_name.strip().lower() in _CONFIG_SECRET_FIELD_NAMES and obj is not None and str(obj).strip():
        return mask_secret_preview(str(obj))
    return obj


def _cmd_config(action: str, key_path: Optional[str], value_raw: Optional[str], as_json: bool = False) -> int:
    if action == "path":
        try:
            from platodoom.config import resolve_user_config_path

            cfg_file = resolve_user_config_path()
        except ImportError:
            cfg_file = _resolve_platodoom_home() / "config.json"
        if as_json:
            print(json.dumps({"ok": True, "path": str(cfg_file)}, indent=2))
        else:
            print(str(cfg_file))
        return 0

    try:
        from platodoom.config import get_user_config_value, resolve_platodoom_config, set_user_config_value

        if action == "set":
            if not key_path:
                print("ERROR: `platodoom config set <key> <value>` requires a key.", file=sys.stderr)
                return 1
            parsed_val: Any = value_raw
            if isinstance(value_raw, str):
                try:
                    parsed_val = json.loads(value_raw)
                except Exception:
                    parsed_val = value_raw
            updated = set_user_config_value(key_path, parsed_val)
            leaf_key = key_path.split(".")[-1]
            safe_val = _redact_config_for_display(parsed_val, key_name=leaf_key)
            safe_cfg = _redact_config_for_display(updated)
            if as_json:
                print(json.dumps({"ok": True, "key": key_path, "value": safe_val, "config": safe_cfg}, indent=2))
            else:
                print(f"Updated ~/.platodoom/config.json: {key_path} = {json.dumps(safe_val)}")
            return 0

        if action == "get" and key_path:
            val = get_user_config_value(key_path)
            leaf_key = key_path.split(".")[-1]
            safe_val = _redact_config_for_display(val, key_name=leaf_key)
            if as_json:
                print(json.dumps({"ok": True, "key": key_path, "value": safe_val}, indent=2))
            else:
                print(json.dumps(safe_val, indent=2) if isinstance(safe_val, (dict, list)) else str(safe_val))
            return 0

        full_cfg = _redact_config_for_display(resolve_platodoom_config())
        print(json.dumps(full_cfg, indent=2))
        return 0
    except ImportError:
        from platodoom.session import _resolve_config

        cfg_file = _resolve_platodoom_home() / "config.json"
        cfg = _resolve_config()
        if action == "set":
            if not key_path:
                return 1
            parsed_val = value_raw
            if isinstance(value_raw, str):
                try:
                    parsed_val = json.loads(value_raw)
                except Exception:
                    parsed_val = value_raw
            parts = [p for p in key_path.split(".") if p]
            cur = cfg
            for p in parts[:-1]:
                if not isinstance(cur.get(p), dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = parsed_val
            _atomic_write_json_mode(cfg_file, cfg, mode=0o600)
            if as_json:
                print(json.dumps({"ok": True, "key": key_path, "value": parsed_val, "config": cfg}, indent=2))
            else:
                print(f"Updated {cfg_file}: {key_path} = {json.dumps(parsed_val)}")
            return 0

        if action == "get" and key_path:
            cur: Any = cfg
            for p in [p for p in key_path.split(".") if p]:
                cur = cur.get(p) if isinstance(cur, dict) else None
            if as_json:
                print(json.dumps({"ok": True, "key": key_path, "value": cur}, indent=2))
            else:
                print(json.dumps(cur, indent=2) if isinstance(cur, (dict, list)) else str(cur))
            return 0

        print(json.dumps(cfg, indent=2))
        return 0


def _cmd_init(model_alias: Optional[str], access_mode: Optional[str], as_json: bool = False) -> int:
    cwd = Path.cwd().resolve()
    overrides: Dict[str, Any] = {
        "model": model_alias or "qwen3.8-27b",
        "gpu": "auto",
        "access": access_mode or "confirm-writes",
        "workspace_roots": ["."],
        "skill_paths": [".platodoom/skills", ".agents/skills"],
        "mcp_config_paths": [".platodoom/mcp.json"],
    }
    try:
        from platodoom.config import init_project_config, load_user_config, save_user_config

        user_cfg = load_user_config()
        user_cfg_path = save_user_config(user_cfg)
        proj_cfg_path = init_project_config(cwd, overrides=overrides)
    except ImportError:
        from platodoom.session import _resolve_config

        user_cfg_path = _resolve_platodoom_home() / "config.json"
        if not user_cfg_path.is_file():
            _atomic_write_json_mode(user_cfg_path, _resolve_config(), mode=0o600)
        proj_cfg_path = cwd / ".platodoom.json"
        _atomic_write_json_mode(proj_cfg_path, overrides, mode=0o644)

    payload = {
        "ok": True,
        "user_config_path": str(user_cfg_path),
        "project_config_path": str(proj_cfg_path),
        "project_config": overrides,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Initialized platodoom configuration:\n  User profile:    {user_cfg_path} (0600)\n  Project config:  {proj_cfg_path} (0644)")
    return 0


def _cmd_sessions(as_json: bool = False) -> int:
    """Lists saved conversations from `~/.platodoom/sessions/` (FR-4.1, FR-4.2)."""
    try:
        from platodoom.session import list_saved_sessions

        sessions = list_saved_sessions()
    except Exception:
        sessions = []
        sess_root = _resolve_platodoom_home() / "sessions"
        if sess_root.is_dir():
            for ep_file in sorted(sess_root.glob("*/endpoint.json")):
                try:
                    data = json.loads(ep_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        sessions.append(data)
                except Exception:
                    continue

    if as_json:
        print(json.dumps({"ok": True, "count": len(sessions), "sessions": sessions}, indent=2))
        return 0

    from platodoom.repl import format_sessions_table

    if sessions:
        print(format_sessions_table(sessions))
    else:
        print("No saved conversations found in ~/.platodoom/sessions/.")
    return 0


def _cmd_swarm(
    allocations: List[str],
    dry_run: bool = False,
    as_json: bool = False,
    models_agents: Optional[str] = None,
    gpu_tier: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    prompt: Optional[str] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    no_colab: bool = False,
) -> int:
    if models_agents:
        from platodoom.catalog import resolve_multi_model_agent_allocation
        from platodoom.config import resolve_platodoom_config
        from platodoom.swarm import run_single_gpu_multi_model_agents

        cfg = resolve_platodoom_config()
        eff_gpu = gpu_tier or "auto"
        if dry_run or (not endpoint_url and prompt is None and no_colab):
            plan = resolve_multi_model_agent_allocation(models_agents, gpu_tier=eff_gpu, config=cfg)
            payload = {
                "ok": True,
                "dry_run": bool(dry_run),
                "mode": "single_gpu_multi_model_agents",
                "gpu_tier": plan["hardware_tier"],
                "vram_gb": plan["vram_gb"],
                "configured_models": [m["alias"] for m in plan["all_models"]],
                "loaded_models": [m["alias"] for m in plan["active_models"]],
                "inactive_unloaded_models": [m["alias"] for m in plan["inactive_models"]],
                "agent_counts": plan["agent_counts"],
                "vram_allocated_by_model": plan["vram_allocated_by_model"],
                "total_agents": plan["total_agents"],
                "allocation": plan,
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"Single-GPU Multi-Model Plan (GPU 1: {plan['hardware_tier']} {plan['vram_gb']} GB, "
                    f"active={len(plan['active_models'])}, inactive={len(plan['inactive_models'])}, "
                    f"total_agents={plan['total_agents']}):"
                )
                for m in plan["all_models"]:
                    state_lbl = "LOADED" if m["agent_count"] > 0 else "UNLOADED (0 GB VRAM)"
                    print(
                        f"  - {m['spec']}: agents={m['agent_count']} [{state_lbl}] vram={m['vram_gb']} GB"
                    )
            return 0

        prompts_arg = {
            m["alias"]: [f"{prompt} (agent {i + 1}/{m['agent_count']})" for i in range(m["agent_count"])]
            if (prompt and m["agent_count"] > 1)
            else ([prompt] * m["agent_count"] if prompt else [f"Execute task for {m['alias']} agent {i + 1}." for i in range(m["agent_count"])])
            for m in resolve_multi_model_agent_allocation(models_agents, gpu_tier=eff_gpu, config=cfg)["active_models"]
        }
        res_multi = run_single_gpu_multi_model_agents(
            model_allocations=models_agents,
            prompts_by_model_or_agent=prompts_arg,
            gpu_tier=eff_gpu,
            endpoint_url=endpoint_url,
            workspace_dir=Path.cwd(),
            use_cache=use_cache,
            force_refresh=force_refresh,
            auto_provision=not no_colab and not endpoint_url,
            config=cfg,
        )
        if as_json:
            print(json.dumps(res_multi, indent=2))
        else:
            print(
                f"Executed {res_multi.get('total_agents', 0)} parallel agents across "
                f"{len(res_multi.get('loaded_models', []))} loaded models on GPU 1 ({res_multi.get('gpu_tier')}):"
            )
            for item in res_multi.get("agent_results", []):
                print(
                    f"  - [{item.get('model_alias')}#{item.get('agent_index')}] "
                    f"batch_size={item.get('observed_batch_size')} cache_hit={item.get('cache_hit')} "
                    f"ok={item.get('ok')}"
                )
        return 0 if res_multi.get("ok", False) else 1

    try:
        from platodoom.swarm import allocate_directory_swarm

        res = allocate_directory_swarm(allocations=allocations, dry_run=dry_run)
    except ImportError:
        from platodoom.session import _resolve_config

        cfg = _resolve_config()
        max_sessions = int((cfg.get("colab") or {}).get("max_concurrent_sessions", 3))
        parsed_items: List[Dict[str, Any]] = []
        for idx, spec in enumerate(allocations):
            parts = str(spec).split(":")
            dir_raw = parts[0] if len(parts) > 0 else "."
            agents = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            gpu = parts[2] if len(parts) > 2 and parts[2] else "T4"
            model = parts[3] if len(parts) > 3 and parts[3] else str(cfg.get("default_model", "qwen3.8-27b"))
            p_root = Path(dir_raw).expanduser().resolve()
            base = re.sub(r"[^a-zA-Z0-9]+", "-", p_root.name).strip("-").lower() or "repo"
            d_slug = f"{base}-{hashlib.sha1(str(p_root).encode('utf-8')).hexdigest()[:8]}"
            status = "allocated" if idx < max_sessions else "queued_session_limit"
            try:
                from curation.api_client import _compute_dynamic_max_num_seqs

                dyn_seqs = _compute_dynamic_max_num_seqs(
                    free_vram_gb=16.0 if gpu == "T4" else 24.0,
                    weights_gb=5.0,
                    model_alias=model,
                    requested_agents=agents,
                )
            except Exception:
                dyn_seqs = max(16, agents)
            item = {
                "directory_path": str(p_root),
                "project_root": str(p_root),
                "dir_slug": d_slug,
                "agent_count": agents,
                "gpu_tier": gpu,
                "model_alias": model,
                "max_num_seqs": max(dyn_seqs, agents),
                "status": status,
                "bearer_token": f"sk-plato-{secrets.token_hex(32)}",
            }
            if not dry_run:
                sdir = _resolve_platodoom_home() / "sessions" / d_slug
                _atomic_write_json_mode(sdir / "routing.json", item, mode=0o600)
            parsed_items.append(item)
        res = {
            "ok": True,
            "dry_run": dry_run,
            "max_concurrent_sessions": max_sessions,
            "requested_sessions": len(parsed_items),
            "overflow_warning": (
                f"Requested {len(parsed_items)} Colab GPU sessions exceeds max_concurrent_sessions={max_sessions}; excess directories queued."
                if len(parsed_items) > max_sessions
                else None
            ),
            "allocations": parsed_items,
        }

    if as_json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Swarm Allocation ({len(res.get('allocations', []))} directories):")
        for item in res.get("allocations", []):
            print(
                f"  - {item.get('dir_slug')}: dir={item.get('project_root')} "
                f"agents={item.get('agent_count')} gpu={item.get('gpu_tier')} "
                f"model={item.get('model_alias')} status={item.get('status')}"
            )
        if res.get("overflow_warning"):
            print(f"WARNING: {res['overflow_warning']}")
    return 0


def _cmd_cache_model(
    alias_or_hf_id: str,
    storage_uri: Optional[str] = None,
    local_source_dir: Optional[str] = None,
    dry_run: bool = False,
    as_json: bool = False,
) -> int:
    try:
        from platodoom.cache_model import cache_model_to_cloud

        res = cache_model_to_cloud(
            alias_or_hf_id=alias_or_hf_id,
            storage_uri=storage_uri,
            dry_run=dry_run,
            local_source_dir=local_source_dir,
        )
    except ImportError:
        from platodoom.session import _resolve_config, _resolve_model_profile

        cfg = _resolve_config()
        prof = _resolve_model_profile(alias_or_hf_id, cfg)
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(prof.get("alias") or alias_or_hf_id)).strip("-")
        dest_uri = storage_uri or (
            f"gs://{cfg['storage']['gcs_bucket']}/models/{slug}"
            if (cfg.get("storage") or {}).get("gcs_bucket")
            else f"/content/drive/MyDrive/platodoom_models/{slug}"
        )
        res = {
            "ok": True,
            "alias": prof.get("alias", alias_or_hf_id),
            "hf_repo_id": prof.get("hf_repo_id", alias_or_hf_id),
            "storage_uri": dest_uri,
            "dry_run": dry_run,
            "manifest_verified": True,
            "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    if as_json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Cached model '{alias_or_hf_id}' -> {res.get('storage_uri')} (verified={res.get('manifest_verified', True)})")
    return 0


def _cmd_auth(
    sub_action: Optional[str] = None,
    as_json: bool = False,
    non_interactive: bool = False,
) -> int:
    """Manages Google Colab browser login, HuggingFace token, and tunnel settings."""
    from platodoom.onboarding import (
        format_auth_status,
        get_auth_status,
        reset_onboarding_state,
        run_interactive_onboarding,
    )

    action = (sub_action or "").strip().lower()
    if action == "reset":
        res = reset_onboarding_state()
        if as_json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Reset onboarding status in {res['config_path']}. Run `platodoom auth` or launch `platodoom` to set up credentials.")
        return 0

    if action == "status" or as_json or non_interactive or not sys.stdin.isatty():
        status = get_auth_status()
        if as_json:
            print(json.dumps(status, indent=2))
        else:
            print(format_auth_status(status))
        return 0

    try:
        run_interactive_onboarding(force=True)
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except EOFError:
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="platodoom",
        description="platodoom: Standalone Open-Source Colab GPU AI Coding, Reasoning & Multi-Directory Swarm CLI",
    )
    parser.add_argument("-v", "--version", action="store_true", help="Print platodoom version and exit")
    parser.add_argument("-m", "--model", "--default-model", dest="model", type=str, default=None, help="Model alias or HuggingFace repo ID (14 priority Qwen/DeepSeek aliases or custom repo)")
    parser.add_argument("--non-interactive", action="store_true", help="Run non-interactively (for `platodoom init` / scripted invocations)")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Execute a non-interactive one-shot prompt and exit")
    parser.add_argument("-g", "--gpu", type=str, default=None, help="Target Colab GPU/TPU tier (auto, G4, H100, A100-80GB, A100, v6e-1, L4, T4, v5e-1)")
    parser.add_argument("--stack", type=str, default=None, help="Solo or co-resident multi-model stack preset (solo, recommended, best-balanced, max-quality, speed-stack, premium, heavy-agent, light-multi, ...)")
    parser.add_argument("--tier", type=str, default=None, help="Filter `platodoom models` output to a specific Colab hardware tier")
    parser.add_argument("--stacks", action="store_true", help="Display 8-tier Colab Solo (x=1) and Multi-Model Stack (x=2..3) presets")
    parser.add_argument("--storage-uri", type=str, default=None, help="Cloud weight cache URI (gs://<bucket>/<prefix> or gdrive://...)")
    parser.add_argument("--backend", type=str, default=None, help="Inference backend (colab, hf-api, custom-url)")
    parser.add_argument("--endpoint", "--endpoint-url", dest="endpoint", type=str, default=None, help="Explicit OpenAI-compatible /v1 endpoint URL")
    parser.add_argument("--no-colab", action="store_true", help="Disable automated Colab provisioning")
    parser.add_argument("--access", type=str, default=None, help="Operator access mode (read-only, confirm-writes, workspace-write, full-access)")
    parser.add_argument("--plan", action="store_true", help="Start session in Plan Mode (read-only architectural investigation)")
    parser.add_argument("--build", action="store_true", help="Start session in Build Mode (confirm-writes file modification)")
    parser.add_argument("--resume", nargs="?", const="__PICK__", default=None, help="Resume an existing conversation by index or session_slug, or prompt with an interactive selector")
    parser.add_argument("--yolo", "--full-access", dest="yolo", action="store_true", help="Shortcut for --access full-access")
    parser.add_argument("--attach", type=str, default=None, help="Attach to an existing warm session_slug or dir_slug")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    parser.add_argument("--allocate", action="append", default=[], help="Swarm allocation tuple <dir>:<agents>:<gpu>[:<model>]")
    parser.add_argument(
        "--models-agents",
        "--multi-model-agents",
        dest="models_agents",
        type=str,
        default=None,
        help="Single-GPU multi-model & multi-agent allocation CSV (e.g. 'qwen-9b@Q4:2,r1-7b@Q4:3,qwen-4b@Q4:0')",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable local-disk response cache lookup/write for this invocation")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass cached completions and overwrite local cache from live inference")
    parser.add_argument("--dry-run", action="store_true", help="Validate without provisioning cloud resources")
    parser.add_argument("--local-source-dir", type=str, default=None, help="Optional local model directory for cache-model")
    parser.add_argument("positional", nargs="*", help="Subcommand (auth, models, config, init, sessions, history, swarm, cache-model, standby, watchdog) and arguments")
    return parser


_PIPELINE_WATCHDOG_FLAGS = (
    "--pipeline",
    "--run-id",
    "--auto-resume",
    "--tmux-daemon",
    "--install-cron",
)


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    if raw_argv and raw_argv[0].lower() == "standby":
        from platodoom.standby import main as standby_main

        return standby_main(raw_argv[1:])

    if raw_argv and raw_argv[0].lower() == "watchdog":
        rest_args = raw_argv[1:]
        is_pipeline_mode = any(
            any(arg == flag or arg.startswith(f"{flag}=") for flag in _PIPELINE_WATCHDOG_FLAGS)
            for arg in rest_args
        )
        if is_pipeline_mode:
            filtered = [a for a in rest_args if a != "--pipeline"]
            from scripts.overnight_pipeline_watchdog import main as pipeline_watchdog_main

            return pipeline_watchdog_main(filtered)

        from platodoom.standby import main as standby_main

        return standby_main([a for a in rest_args if a != "--standby"])

    parser = build_arg_parser()
    norm_argv: List[str] = []
    idx = 0
    while idx < len(raw_argv):
        tok = raw_argv[idx]
        if tok == "--":
            norm_argv.extend(raw_argv[idx:])
            break
        if tok in ("-p", "--prompt") and idx + 1 < len(raw_argv):
            norm_argv.append(f"--prompt={raw_argv[idx + 1]}")
            idx += 2
            continue
        norm_argv.append(tok)
        idx += 1
    args = parser.parse_args(norm_argv)

    if args.version:
        if args.json:
            print(json.dumps({"name": "platodoom", "version": PLATODOOM_VERSION}))
        else:
            print(f"platodoom v{PLATODOOM_VERSION}")
        return 0

    pos = list(args.positional or [])
    subcmd = pos[0].lower() if pos else ""

    if subcmd == "auth":
        auth_action = pos[1].lower() if len(pos) > 1 else None
        return _cmd_auth(
            sub_action=auth_action,
            as_json=args.json,
            non_interactive=bool(args.non_interactive),
        )

    if subcmd == "models":
        return _cmd_models(as_json=args.json, tier_filter=args.tier, show_stacks=args.stacks)

    if subcmd in ("sessions", "history"):
        return _cmd_sessions(as_json=args.json)

    if subcmd == "config":
        raw_sub = pos[1].lower() if len(pos) > 1 else "list"
        if raw_sub == "set":
            key_arg = pos[2] if len(pos) > 2 else None
            val_arg = pos[3] if len(pos) > 3 else None
            return _cmd_config("set", key_arg, val_arg, as_json=args.json)
        if raw_sub in ("list", "all"):
            return _cmd_config("list", None, None, as_json=args.json)
        if raw_sub == "path":
            return _cmd_config("path", None, None, as_json=args.json)
        if raw_sub == "get":
            key_arg = pos[2] if len(pos) > 2 else None
            return _cmd_config("get", key_arg, None, as_json=args.json)
        return _cmd_config("get", pos[1] if len(pos) > 1 else None, None, as_json=args.json)

    if subcmd == "init":
        eff_access = "full-access" if args.yolo else args.access
        return _cmd_init(args.model, eff_access, as_json=args.json)

    if subcmd == "swarm":
        allocs = list(args.allocate or []) + pos[1:]
        return _cmd_swarm(
            allocs,
            dry_run=args.dry_run,
            as_json=args.json,
            models_agents=args.models_agents,
            gpu_tier=args.gpu,
            endpoint_url=args.endpoint,
            prompt=args.prompt,
            use_cache=not bool(args.no_cache),
            force_refresh=bool(args.force_refresh),
            no_colab=bool(args.no_colab),
        )

    if subcmd == "cache-model":
        target_model = pos[1] if len(pos) > 1 else (args.model or "qwen3.8-27b")
        return _cmd_cache_model(
            target_model,
            storage_uri=args.storage_uri,
            local_source_dir=args.local_source_dir,
            dry_run=args.dry_run,
            as_json=args.json,
        )

    # Interactive REPL or Non-Interactive One-Shot Mode (-p / --prompt)
    from platodoom.repl import (
        begin_interactive_terminal_session,
        emit_terminal_goodbye,
        prompt_startup_model_and_access,
        run_interactive_repl,
    )
    from platodoom.session import PlatoSession, _resolve_config

    cli_overrides: Dict[str, Any] = {}
    if args.model:
        cli_overrides["default_model"] = args.model
    if args.gpu:
        cli_overrides["default_gpu"] = args.gpu
    if args.stack:
        cli_overrides["default_stack"] = args.stack
    if args.backend:
        cli_overrides["default_backend"] = args.backend
    if args.yolo:
        cli_overrides["default_access"] = "full-access"
    elif args.plan:
        cli_overrides["default_access"] = "read-only"
    elif args.build:
        cli_overrides["default_access"] = "confirm-writes"
    elif args.access:
        cli_overrides["default_access"] = args.access

    cfg = _resolve_config(cli_overrides=cli_overrides)

    # Handle interactive resume picker
    target_resume: Optional[str | int] = None
    if args.resume == "__PICK__":
        try:
            from platodoom.session import list_saved_sessions

            saved = list_saved_sessions()
        except Exception:
            saved = []

        if not saved:
            if args.json:
                print(json.dumps({"ok": False, "error": "No saved conversations found"}, indent=2))
            else:
                print("No saved conversations found in ~/.platodoom/sessions/.")
            return 0

        from platodoom.repl import format_sessions_table

        if args.prompt is None and not args.json:
            begin_interactive_terminal_session(emit_banner=True)
        print(format_sessions_table(saved))
        try:
            r_pick = input(f"\nResume session [1-{len(saved)} or ID, Enter=[1]]: ").strip()
        except KeyboardInterrupt:
            emit_terminal_goodbye(force=True)
            return 130
        except EOFError:
            emit_terminal_goodbye(force=True)
            return 0
        target_resume = r_pick if r_pick else "1"
    elif args.resume:
        target_resume = args.resume

    # Resolve workflow_mode and access_mode
    workflow_mode: Optional[str] = None
    if args.plan:
        workflow_mode = "plan"
        selected_access = "read-only"
    elif args.build:
        workflow_mode = "build"
        selected_access = "full-access" if args.yolo else (args.access or "confirm-writes")
    elif args.yolo:
        selected_access = "full-access"
    elif args.access:
        selected_access = args.access
        if selected_access == "read-only":
            workflow_mode = "plan"
    else:
        selected_access = "full-access" if args.yolo else (args.access or str(cfg.get("default_access") or "confirm-writes"))

    selected_model = args.model or str(cfg.get("default_model") or "qwen3.8-27b")
    if workflow_mode is None:
        workflow_mode = "plan" if selected_access == "read-only" else "build"

    if args.prompt is None and not args.json:
        if (
            not args.non_interactive
            and not args.no_colab
            and not args.endpoint
            and sys.stdin.isatty()
        ):
            from platodoom.onboarding import (
                run_interactive_onboarding,
                should_run_first_run_onboarding,
            )

            if should_run_first_run_onboarding(cfg):
                begin_interactive_terminal_session(emit_banner=True)
                try:
                    run_interactive_onboarding(cfg, force=False)
                    cfg = _resolve_config(cli_overrides=cli_overrides)
                except KeyboardInterrupt:
                    emit_terminal_goodbye(force=True)
                    return 130
                except EOFError:
                    emit_terminal_goodbye(force=True)
                    return 0
        begin_interactive_terminal_session(emit_banner=False)

    session_kwargs: Dict[str, Any] = {
        "model_alias": selected_model,
        "access_mode": selected_access,
        "workspace_dir": Path.cwd(),
        "attach_slug": args.attach,
        "config": cfg,
        "endpoint_url": args.endpoint,
        "gpu_tier": args.gpu if (args.gpu and args.gpu.lower() != "auto") else None,
        "stack_preset": args.stack or cfg.get("default_stack"),
        "auto_attach_warm": bool(not args.no_colab and not args.endpoint),
        "sync_standby": True,
        "no_colab": bool(args.no_colab),
    }
    if workflow_mode:
        session_kwargs["workflow_mode"] = workflow_mode
    if target_resume:
        session_kwargs["resume_target"] = target_resume

    session = PlatoSession(**session_kwargs)
    if args.models_agents:
        session.configure_multi_model_agents(
            args.models_agents,
            gpu_tier=args.gpu if (args.gpu and args.gpu.lower() != "auto") else None,
            stack_preset=args.stack or cfg.get("default_stack"),
        )
    if (
        not args.no_colab
        and not args.endpoint
        and session.endpoint_url is None
        and str(cfg.get("default_backend", "colab")).lower() == "colab"
        and not (args.prompt and args.prompt.strip().startswith("/"))
    ):
        session.try_attach_warm_endpoint(allow_inplace_switch=True, probe_timeout_sec=0.65)

    # One-shot command execution
    if args.prompt is not None:
        if args.prompt.strip().startswith("/"):
            slash_res = session.handle_slash_command(args.prompt.strip())
            if args.json:
                print(json.dumps(slash_res, indent=2))
            else:
                print(str(slash_res.get("output") or json.dumps(slash_res)))
            return 0 if slash_res.get("ok", True) else 1

        stream_state = {"tokens": 0, "in_reasoning": False}

        def _cli_stream_cb(event_or_delta: Any) -> None:
            if isinstance(event_or_delta, str):
                token = event_or_delta
                if not token or "<tool_call>" in token or "<function=" in token:
                    return
                if not args.json:
                    if stream_state["in_reasoning"]:
                        sys.stdout.write("\n")
                        stream_state["in_reasoning"] = False
                    sys.stdout.write(token)
                    sys.stdout.flush()
                stream_state["tokens"] += len(token)
                return

            if not isinstance(event_or_delta, dict) or args.json:
                return

            ev_type = event_or_delta.get("type")
            if ev_type == "token_delta":
                token = str(event_or_delta.get("content") or "")
                if token and "<tool_call>" not in token and "<function=" not in token:
                    if stream_state["in_reasoning"]:
                        sys.stdout.write("\n")
                        stream_state["in_reasoning"] = False
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    stream_state["tokens"] += len(token)
            elif ev_type == "reasoning_delta":
                r_delta = str(event_or_delta.get("content") or "")
                if r_delta:
                    if not stream_state["in_reasoning"]:
                        sys.stdout.write("\n[Reasoning]: ")
                        stream_state["in_reasoning"] = True
                    sys.stdout.write(r_delta)
                    sys.stdout.flush()
            elif ev_type in ("thinking", "reasoning"):
                content = str(event_or_delta.get("content") or "").strip()
                if content:
                    sys.stdout.write(f"\n[Reasoning]: {content}\n")
                    sys.stdout.flush()
            elif ev_type == "tool_call":
                if stream_state["in_reasoning"]:
                    sys.stdout.write("\n")
                    stream_state["in_reasoning"] = False
                t_name = event_or_delta.get("name", "unknown")
                t_args = event_or_delta.get("arguments", {})
                sys.stdout.write(f"\n▶ [tool:{t_name}] {json.dumps(t_args, ensure_ascii=False)}\n")
                sys.stdout.flush()
            elif ev_type == "tool_result":
                t_name = event_or_delta.get("name", "unknown")
                t_out = str(event_or_delta.get("output") or "")
                if "PERMISSION_DENIED_READ_ONLY" in t_out:
                    from platodoom.repl import format_permission_denied_banner

                    sys.stdout.write("\n" + format_permission_denied_banner(tool_name=t_name, workflow_mode=getattr(session, "workflow_mode", "plan")) + "\n")
                    sys.stdout.flush()
                else:
                    lines_count = len(t_out.splitlines())
                    chars_count = len(t_out)
                    sys.stdout.write(f"  ✔ [result:{t_name}] {lines_count} lines ({chars_count:,} chars)\n")
                    sys.stdout.flush()

        turn_res = session.run_turn(
            user_prompt=args.prompt,
            stream_callback=_cli_stream_cb,
            use_cache=False if args.no_cache else None,
            force_refresh=bool(args.force_refresh),
        )
        if not args.json and stream_state["in_reasoning"]:
            sys.stdout.write("\n")
            sys.stdout.flush()
            stream_state["in_reasoning"] = False
        if args.json:
            print(json.dumps(turn_res, indent=2))
        else:
            if not turn_res.get("ok", True) and turn_res.get("error"):
                print(str(turn_res["error"]), file=sys.stderr)
            elif stream_state["tokens"] == 0:
                print(str(turn_res.get("response") or turn_res.get("output") or ""))
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()
        return 0 if turn_res.get("ok", True) else 1

    return run_interactive_repl(session)


if __name__ == "__main__":
    _code = 0
    try:
        _code = main()
    except KeyboardInterrupt:
        _code = 130
    except Exception as _exc:
        print(f"[platodoom error] {_exc}", file=sys.stderr)
        _code = 1
    fast_non_hanging_exit(_code)
