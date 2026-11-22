#!/usr/bin/env python3
"""Thin transport & telemetry adapter running the native `omp` harness against Colab/platodoom OpenAI endpoints.

Governing Spec: platodoom specification (Spec 873), LANGUAGE.md, RULE[user_global]
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curation.api_client import (
    normalize_colab_endpoint_url,
    probe_colab_qwen_endpoint,
    resolve_expected_model_identifiers,
    resolve_serving_bearer_token,
)
from platodoom.catalog import (
    OMP_COMPACT_CODING_TOOLS,
    get_model_catalog,
    introspect_model_protocol,
    resolve_model_context_profile,
    resolve_model_profile,
    resolve_omp_model_compat,
)
from platodoom.config import resolve_platodoom_config, resolve_platodoom_home
from scripts.provision_colab_serving import compute_dir_slug, resolve_project_root

MAX_ARGV_PROMPT_BYTES = 65536
OMP_DEFAULT_BUILTIN_TOOLS: Tuple[str, ...] = (
    "read",
    "edit",
    "write",
    "bash",
    "grep",
    "glob",
    "find",
    "eval",
    "task",
    "wait",
    "todo",
    "web_search",
    "lsp",
    "ast_edit",
)


def _sanitize_omp_mcp_part(value: str, fallback: str) -> str:
    sanitized = re.sub(r"[^a-z0-9_]+", "_", str(value or "").lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized if sanitized else fallback


def format_omp_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Mints the exact `mcp__<sanitized_server>_<normalized_tool>` key used by `omp`'s `createMCPToolName`."""
    raw_tool = str(tool_name or "").strip()
    if raw_tool.startswith("mcp__"):
        return raw_tool
    s_srv = _sanitize_omp_mcp_part(server_name, "server")
    s_tool = _sanitize_omp_mcp_part(raw_tool, "tool")
    prefix = f"{s_srv}_"
    if s_tool.startswith(prefix):
        s_tool = s_tool[len(prefix) :]
    minted = f"mcp__{s_srv}_{s_tool}"
    return minted[:64]


def _parse_csv_or_seq(val: Optional[str | Sequence[str]]) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        items = [x.strip() for x in val.split(",") if x.strip()]
    else:
        items = [str(x).strip() for x in val if str(x).strip()]
    deduped: List[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def resolve_scoped_mcp_allowlist(
    work_dir: Path,
    *,
    agent_name: Optional[str] = None,
    explicit_builtin_tools: Optional[str | Sequence[str]] = None,
    explicit_servers: Optional[str | Sequence[str]] = None,
    explicit_pinned_tools: Optional[str | Sequence[str]] = None,
    explicit_vision: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Resolves per-agent scoped built-in tools, MCP servers, pinned top-level tools, and vision flag from CLI, env, or wayfinder/<runId>/mcp-allowlist.json."""
    eff_env = env if env is not None else os.environ
    builtin_tools = _parse_csv_or_seq(explicit_builtin_tools) or _parse_csv_or_seq(
        eff_env.get("SANDCASTLE_OMP_BUILTIN_TOOLS")
    )
    servers = _parse_csv_or_seq(explicit_servers) or _parse_csv_or_seq(eff_env.get("SANDCASTLE_MCP_SERVERS"))
    raw_pinned = _parse_csv_or_seq(explicit_pinned_tools) or _parse_csv_or_seq(
        eff_env.get("SANDCASTLE_MCP_PINNED_TOOLS")
    )
    eff_agent = str(agent_name or eff_env.get("SANDCASTLE_AGENT_NAME") or "").strip()
    run_id = str(eff_env.get("SANDCASTLE_RUN_ID") or "").strip()

    vision: Optional[bool] = None
    env_vis = str(eff_env.get("SANDCASTLE_OMP_ENABLE_VISION") or "").strip().lower()
    if explicit_vision is not None:
        vision = bool(explicit_vision)
    elif env_vis in ("1", "true", "yes"):
        vision = True
    elif env_vis in ("0", "false", "no"):
        vision = False

    if (not builtin_tools or not servers and not raw_pinned or vision is None) and run_id:
        allowlist_path = work_dir / "wayfinder" / run_id / "mcp-allowlist.json"
        if allowlist_path.is_file():
            try:
                raw_json = json.loads(allowlist_path.read_text(encoding="utf-8"))
                if isinstance(raw_json, Mapping):
                    agent_entry = raw_json.get(eff_agent) if eff_agent else None
                    if not isinstance(agent_entry, Mapping) and isinstance(raw_json.get("agents"), Mapping):
                        agent_entry = raw_json["agents"].get(eff_agent)
                    if isinstance(agent_entry, Mapping):
                        if not builtin_tools:
                            builtin_tools = _parse_csv_or_seq(
                                agent_entry.get("builtinTools") or agent_entry.get("builtin_tools")
                            )
                        if not servers:
                            servers = _parse_csv_or_seq(agent_entry.get("servers") or agent_entry.get("mcp_servers"))
                        if not raw_pinned:
                            raw_pinned = _parse_csv_or_seq(
                                agent_entry.get("pinnedTools")
                                or agent_entry.get("pinned_tools")
                                or agent_entry.get("tools")
                            )
                        if vision is None and isinstance(agent_entry.get("vision"), bool):
                            vision = bool(agent_entry["vision"])
            except Exception:
                pass

    pinned_omp_names: List[str] = []
    default_srv = servers[0] if len(servers) == 1 else ""
    for item in raw_pinned:
        if item.startswith("mcp__"):
            minted = item
        elif ":" in item:
            srv_part, tool_part = item.split(":", 1)
            minted = format_omp_mcp_tool_name(srv_part, tool_part)
            if srv_part.strip() and srv_part.strip() not in servers:
                servers.append(srv_part.strip())
        elif "/" in item:
            srv_part, tool_part = item.split("/", 1)
            minted = format_omp_mcp_tool_name(srv_part, tool_part)
            if srv_part.strip() and srv_part.strip() not in servers:
                servers.append(srv_part.strip())
        elif default_srv:
            minted = format_omp_mcp_tool_name(default_srv, item)
        else:
            minted = item
        if minted and minted not in pinned_omp_names:
            pinned_omp_names.append(minted)

    return {
        "agent_name": eff_agent or None,
        "builtin_tools": builtin_tools,
        "servers": servers,
        "pinned_tools": pinned_omp_names,
        "vision": vision,
    }


def _discover_host_mcp_server_definitions(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Loads raw MCP server definitions across project and user-global MCP config files."""
    home = Path.home()
    candidates = [
        work_dir / ".platodoom" / "mcp.json",
        work_dir / ".gemini" / "mcp.json",
        work_dir / ".cursor" / "mcp.json",
        work_dir / ".claude" / "mcp.json",
        home / ".platodoom" / "mcp.json",
        home / ".gemini" / "config" / "mcp_config.json",
        home / ".gemini" / "antigravity" / "mcp_config.json",
        home / ".gemini" / "antigravity-ide" / "mcp_config.json",
        home / ".cursor" / "mcp.json",
        home / ".claude.json",
    ]
    catalog: Dict[str, Dict[str, Any]] = {}
    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(data, Mapping):
            continue
        block = data.get("mcpServers") or data.get("mcp_servers") or data.get("servers")
        if not isinstance(block, Mapping):
            continue
        for srv_name, srv_cfg in block.items():
            name_s = str(srv_name).strip()
            if not name_s or name_s in catalog or not isinstance(srv_cfg, Mapping):
                continue
            entry: Dict[str, Any] = {}
            if srv_cfg.get("command"):
                entry["command"] = str(srv_cfg["command"])
                if isinstance(srv_cfg.get("args"), list):
                    entry["args"] = [str(a) for a in srv_cfg["args"]]
                if isinstance(srv_cfg.get("env"), Mapping):
                    entry["env"] = {str(k): str(v) for k, v in srv_cfg["env"].items()}
            elif srv_cfg.get("url") or srv_cfg.get("serverUrl") or srv_cfg.get("httpUrl"):
                entry["url"] = str(srv_cfg.get("url") or srv_cfg.get("serverUrl") or srv_cfg.get("httpUrl"))
                if isinstance(srv_cfg.get("headers"), Mapping):
                    entry["headers"] = {str(k): str(v) for k, v in srv_cfg["headers"].items()}
            if entry:
                catalog[name_s] = entry
    return catalog


def materialize_worktree_mcp_json(
    work_dir: Path,
    allowed_servers: Sequence[str],
) -> Tuple[Optional[Path], bool, List[str]]:
    """Writes a scoped `<work_dir>/.mcp.json` containing only `allowed_servers`."""
    requested = [str(s).strip() for s in allowed_servers if str(s).strip()]
    if not requested:
        return None, False, []
    mcp_json_path = Path(work_dir).resolve() / ".mcp.json"
    preexisting = mcp_json_path.exists()
    host_defs = _discover_host_mcp_server_definitions(Path(work_dir).resolve())
    selected: Dict[str, Any] = {}
    for srv in requested:
        if srv in host_defs:
            selected[srv] = host_defs[srv]
    if not selected:
        return None, False, []
    payload = {"mcpServers": selected}
    mcp_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return mcp_json_path, not preexisting, list(selected.keys())

DEFAULT_VERIFY_TIMEOUT_SEC = 5.0
OMP_MODEL_YML_ALLOWED_KEYS = frozenset(
    {
        "id",
        "name",
        "api",
        "kind",
        "baseUrl",
        "reasoning",
        "thinking",
        "input",
        "imageInputDecoder",
        "imageCountBudget",
        "tokenizer",
        "supportsTools",
        "cost",
        "promptCache",
        "premiumMultiplier",
        "contextWindow",
        "maxContextWindow",
        "maxTokens",
        "omitMaxOutputTokens",
        "preferWebsockets",
        "headers",
        "compat",
    }
)


def _strip_precision_suffix(model_id: str) -> str:
    raw = str(model_id or "").strip()
    if "@" in raw and not Path(raw).expanduser().exists():
        return raw.split("@", 1)[0].strip()
    return raw


def resolve_omp_runtime_env(
    base_env: Optional[Mapping[str, str]] = None,
) -> Tuple[str, str, Dict[str, str]]:
    """Deterministically locates `omp` and `bun` across standard paths and fails fast if either is absent."""
    env = dict(base_env) if base_env is not None else os.environ.copy()
    home = Path.home()
    candidates: List[str] = []
    bun_install = str(env.get("BUN_INSTALL") or "").strip()
    if bun_install:
        candidates.append(str(Path(bun_install).expanduser() / "bin"))
    candidates.extend(
        [
            str(home / ".bun" / "bin"),
            "/opt/homebrew/bin",
            "/usr/local/bin",
            str(home / ".local" / "bin"),
        ]
    )
    existing_path = str(env.get("PATH") or "")
    for part in existing_path.split(os.pathsep):
        if part and part not in candidates:
            candidates.append(part)
    deduped_dirs: List[str] = []
    for d in candidates:
        if d and d not in deduped_dirs:
            deduped_dirs.append(d)
    search_path = os.pathsep.join(deduped_dirs)

    omp_bin = shutil.which("omp", path=search_path)
    if not omp_bin:
        raise FileNotFoundError(
            f"Missing required binary 'omp' for omp harness; searched PATH directories: {deduped_dirs}"
        )
    bun_bin = shutil.which("bun", path=search_path)
    if not bun_bin:
        raise FileNotFoundError(
            f"Missing required binary 'bun' (required by omp shebang) for omp harness; searched PATH directories: {deduped_dirs}"
        )

    final_dirs: List[str] = []
    for d in [str(Path(bun_bin).parent), str(Path(omp_bin).parent), *deduped_dirs]:
        if d and d not in final_dirs:
            final_dirs.append(d)
    env["PATH"] = os.pathsep.join(final_dirs)
    return omp_bin, bun_bin, env


def resolve_omp_agent_dir(explicit_agent_dir: Optional[Path | str] = None) -> Path:
    """Resolves the active `omp` agent directory without clobbering user state when unrequested."""
    if explicit_agent_dir is not None and str(explicit_agent_dir).strip():
        return Path(explicit_agent_dir).expanduser().resolve()
    env_dir = str(os.environ.get("PI_CODING_AGENT_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path.home() / ".omp" / "agent").resolve()


def _parse_models_yml_text(raw_text: str, models_path: Path, bun_bin: Optional[str] = None) -> Dict[str, Any]:
    """Parses `models.yml` with PyYAML when available, or Bun's built-in YAML/JSON parser as a zero-dep fallback."""
    if not raw_text.strip():
        return {}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw_text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Expected top-level mapping in {models_path}, got {type(loaded).__name__}")
        return loaded
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {models_path}: {exc}") from exc

    try:
        loaded_json = json.loads(raw_text)
        if isinstance(loaded_json, dict):
            return loaded_json
    except Exception:
        pass

    eff_bun = bun_bin or shutil.which("bun", path=os.pathsep.join([str(Path.home() / ".bun" / "bin"), "/opt/homebrew/bin", "/usr/local/bin", os.environ.get("PATH", "")]))
    if eff_bun:
        proc = subprocess.run(
            [
                eff_bun,
                "-e",
                (
                    "import fs from 'node:fs'; "
                    "const raw = fs.readFileSync(process.argv[1], 'utf8'); "
                    "const parsed = Bun.YAML.parse(raw); "
                    "process.stdout.write(JSON.stringify(parsed ?? {}));"
                ),
                str(models_path),
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc.returncode == 0:
            parsed = json.loads(proc.stdout or "{}")
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError(f"Expected top-level mapping in {models_path}, got {type(parsed).__name__}")
        raise RuntimeError(f"Failed to parse {models_path} via bun YAML parser: {proc.stderr.strip()}")
    raise RuntimeError(f"Unable to parse {models_path}: PyYAML not installed and bun binary unavailable")


def _dump_yaml_scalar(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    return json.dumps(str(val), ensure_ascii=False)


def _dump_yaml_value(val: Any, indent: int = 0) -> List[str]:
    prefix = "  " * indent
    if isinstance(val, Mapping):
        if not val:
            return [f"{prefix}{{}}"]
        lines: List[str] = []
        for k, v in val.items():
            key_s = str(k)
            if isinstance(v, Mapping):
                if not v:
                    lines.append(f"{prefix}{key_s}: {{}}")
                else:
                    lines.append(f"{prefix}{key_s}:")
                    lines.extend(_dump_yaml_value(v, indent + 1))
            elif isinstance(v, list):
                if not v:
                    lines.append(f"{prefix}{key_s}: []")
                else:
                    lines.append(f"{prefix}{key_s}:")
                    lines.extend(_dump_yaml_value(v, indent + 1))
            else:
                lines.append(f"{prefix}{key_s}: {_dump_yaml_scalar(v)}")
        return lines
    if isinstance(val, list):
        if not val:
            return [f"{prefix}[]"]
        lines = []
        for item in val:
            if isinstance(item, Mapping):
                if not item:
                    lines.append(f"{prefix}- {{}}")
                else:
                    items = list(item.items())
                    first_k, first_v = items[0]
                    if isinstance(first_v, (Mapping, list)) and first_v:
                        lines.append(f"{prefix}- {first_k}:")
                        lines.extend(_dump_yaml_value(first_v, indent + 2))
                    else:
                        lines.append(f"{prefix}- {first_k}: {_dump_yaml_scalar(first_v)}")
                    for rem_k, rem_v in items[1:]:
                        if isinstance(rem_v, Mapping):
                            if not rem_v:
                                lines.append(f"{prefix}  {rem_k}: {{}}")
                            else:
                                lines.append(f"{prefix}  {rem_k}:")
                                lines.extend(_dump_yaml_value(rem_v, indent + 2))
                        elif isinstance(rem_v, list):
                            if not rem_v:
                                lines.append(f"{prefix}  {rem_k}: []")
                            else:
                                lines.append(f"{prefix}  {rem_k}:")
                                lines.extend(_dump_yaml_value(rem_v, indent + 2))
                        else:
                            lines.append(f"{prefix}  {rem_k}: {_dump_yaml_scalar(rem_v)}")
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_value(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_dump_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_dump_yaml_scalar(val)}"]


def _serialize_models_yml(data: Mapping[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True, default_flow_style=False)
    except ImportError:
        return "\n".join(_dump_yaml_value(data)) + "\n"


def _read_json_file_strict(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.is_file():
        return None, None
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return None, f"{path} does not contain a JSON object"
        return parsed, None
    except Exception as exc:
        return None, f"Failed to parse {path}: {exc}"


def find_platodoom_cli_selection(
    project_root: Optional[Path | str] = None,
    *,
    allow_global: bool = False,
) -> Dict[str, Any]:
    """Reads the latest platodoom CLI/session selection scoped to the active directory unless `allow_global=True`."""
    home = resolve_platodoom_home()
    target_slug = compute_dir_slug(resolve_project_root(project_root)) if project_root is not None else None
    last_sel = home / "last_cli_selection.json"
    data, err = _read_json_file_strict(last_sel)
    if err:
        raise RuntimeError(err)
    if data and (data.get("active_model") or data.get("model") or data.get("endpoint_url")):
        status = str(data.get("status") or "").strip().lower()
        rec_slug = str(data.get("dir_slug") or "").strip()
        if status != "offline" and (allow_global or target_slug is None or not rec_slug or rec_slug == target_slug):
            return data

    sessions_dir = home / "sessions"
    if sessions_dir.is_dir():
        candidates: List[Tuple[float, Path]] = []
        try:
            for child in sessions_dir.iterdir():
                if not child.is_dir() or not child.name.startswith("platodoom-chat-"):
                    continue
                ep = child / "endpoint.json"
                if not ep.is_file():
                    ep = child / "session.json"
                if ep.is_file():
                    candidates.append((ep.stat().st_mtime, ep))
        except OSError:
            pass
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, ep_path in candidates:
            s_data, s_err = _read_json_file_strict(ep_path)
            if s_err or not s_data:
                continue
            status = str(s_data.get("status") or "").strip().lower()
            rec_slug = str(s_data.get("dir_slug") or "").strip()
            if status == "offline":
                continue
            if allow_global or target_slug is None or (rec_slug and rec_slug == target_slug):
                if s_data.get("active_model") or s_data.get("model") or s_data.get("endpoint_url"):
                    return s_data
    return {}


def resolve_omp_model_registration(
    model_id: str,
    *,
    base_url: str,
    live_model_record: Optional[Mapping[str, Any]] = None,
    registry_record: Optional[Mapping[str, Any]] = None,
    gpu_tier: Optional[str] = None,
    context_window_override: Optional[int] = None,
    max_tokens_override: Optional[int] = None,
    active_tools: Optional[Sequence[str]] = None,
    compact_tools: bool = False,
    vision_override: Optional[bool] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Derives dynamic `omp` `models.yml` registration metadata from `platodoom/catalog.py` and live `/v1/models`."""
    raw_id = str(model_id or "").strip()
    if not raw_id:
        raise ValueError("Cannot build omp model registration for an empty model_id")
    clean_id = _strip_precision_suffix(raw_id)
    norm_base = normalize_colab_endpoint_url(base_url)

    profile: Dict[str, Any] = {}
    canonical_alias = clean_id
    try:
        profile = resolve_model_profile(clean_id, config=config)
        canonical_alias = str(profile.get("alias") or profile.get("model_alias") or clean_id)
    except ValueError:
        profile = {}

    compat_info = resolve_omp_model_compat(canonical_alias)
    reasoning_flag = bool(compat_info.get("reasoning", False))
    if not profile and isinstance(live_model_record, Mapping) and isinstance(live_model_record.get("reasoning"), bool):
        reasoning_flag = bool(live_model_record.get("reasoning"))
        if reasoning_flag:
            compat_info["compat"].pop("supportsReasoningParams", None)
            compat_info["compat"].setdefault("thinkingFormat", "openai")

    resolved_ctx: Optional[int] = None
    ctx_source = "fallback"
    if isinstance(live_model_record, Mapping):
        for k in ("max_model_len", "context_length", "contextWindow"):
            v = live_model_record.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and int(v) > 0:
                resolved_ctx = int(v)
                ctx_source = f"live:{k}"
                break
    if resolved_ctx is None and isinstance(context_window_override, (int, float)) and int(context_window_override) > 0:
        resolved_ctx = int(context_window_override)
        ctx_source = "override"
    if resolved_ctx is None and isinstance(registry_record, Mapping):
        reg_ctx = registry_record.get("max_model_len")
        if isinstance(reg_ctx, (int, float)) and not isinstance(reg_ctx, bool) and int(reg_ctx) > 0:
            resolved_ctx = int(reg_ctx)
            ctx_source = "registry"
    if resolved_ctx is None:
        eff_tier = str(
            gpu_tier
            or (registry_record or {}).get("gpu_tier")
            or (registry_record or {}).get("gpu_type")
            or ""
        ).strip().upper()
        if eff_tier in ("A100", "A100-80GB", "H100", "G4") and profile.get("max_model_len_a100"):
            resolved_ctx = int(profile["max_model_len_a100"])
            ctx_source = "catalog_a100"
        elif profile.get("max_model_len"):
            resolved_ctx = int(profile["max_model_len"])
            ctx_source = "catalog"
    if resolved_ctx is None or resolved_ctx <= 0:
        resolved_ctx = 16384
        ctx_source = "conservative_fallback"
        if isinstance(live_model_record, Mapping):
            print(
                f"[omp-context warning] Live /v1/models omitted max_model_len for unknown model '{clean_id}'; "
                "falling back conservatively to 16384 tokens.",
                file=sys.stderr,
                flush=True,
            )
    resolved_ctx = max(4096, resolved_ctx)

    live_max_tokens: Optional[int] = None
    if isinstance(live_model_record, Mapping):
        limits = live_model_record.get("limits") if isinstance(live_model_record.get("limits"), Mapping) else {}
        for v in (
            live_model_record.get("max_output_tokens"),
            limits.get("max_output_tokens"),
            live_model_record.get("max_completion_tokens"),
            live_model_record.get("max_tokens"),
        ):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and int(v) > 0:
                live_max_tokens = int(v)
                break

    requested_out: Optional[int] = None
    if isinstance(max_tokens_override, (int, float)) and not isinstance(max_tokens_override, bool) and int(max_tokens_override) > 0:
        requested_out = int(max_tokens_override)
    elif live_max_tokens is not None:
        requested_out = live_max_tokens
    elif profile.get("max_output_tokens"):
        requested_out = int(profile["max_output_tokens"])

    ctx_profile = resolve_model_context_profile(
        canonical_alias or clean_id,
        context_window=resolved_ctx,
        active_tools=active_tools,
        compact_tools=compact_tools,
        requested_max_tokens=requested_out,
        config=config,
    )

    if isinstance(max_tokens_override, (int, float)) and not isinstance(max_tokens_override, bool) and int(max_tokens_override) > 0:
        resolved_max_tokens = int(max_tokens_override)
    else:
        resolved_max_tokens = int(ctx_profile["output_reserve_tokens"])
    resolved_max_tokens = min(max(1024, resolved_max_tokens), max(1024, resolved_ctx // 2))

    served_id = str((live_model_record or {}).get("id") or clean_id).strip() or clean_id
    display_name = str(profile.get("display_name") or canonical_alias or served_id)

    server_vision_known: Optional[bool] = None
    if isinstance(live_model_record, Mapping):
        if isinstance(live_model_record.get("vision"), bool):
            server_vision_known = bool(live_model_record["vision"])
        else:
            in_mods = live_model_record.get("input_modalities") or live_model_record.get("input")
            if isinstance(in_mods, (list, tuple)) and len(in_mods) > 0:
                server_vision_known = "image" in [str(x).lower() for x in in_mods]
            elif str(live_model_record.get("modality") or "").lower() in ("multimodal", "vision", "vlm"):
                server_vision_known = True

    live_vision = False
    env_vision = str(os.environ.get("SANDCASTLE_OMP_ENABLE_VISION") or "").strip().lower()
    explicit_agent_vision: Optional[bool] = None
    if vision_override is not None:
        explicit_agent_vision = bool(vision_override)
    elif env_vision in ("0", "false", "no"):
        explicit_agent_vision = False
    elif env_vision in ("1", "true", "yes"):
        explicit_agent_vision = True

    if explicit_agent_vision is True:
        if server_vision_known is False:
            raise RuntimeError(
                f"Agent requires Vision mode (vision=true), but live endpoint '{norm_base}' "
                f"reports vision=false (input_modalities={live_model_record.get('input_modalities') if isinstance(live_model_record, Mapping) else None}) "
                f"for model '{served_id}'. Toggle Vision ON via "
                f"`python scripts/provision_colab_serving.py --switch-model --model-alias={served_id} --vision --repo-dir=.` "
                f"or restart Colab with a `--vision` notebook."
            )
        live_vision = True
    elif explicit_agent_vision is False:
        live_vision = False
    elif server_vision_known is not None:
        live_vision = bool(server_vision_known)
    else:
        live_vision = False

    input_modalities = ["text", "image"] if live_vision else ["text"]

    return {
        "id": served_id,
        "name": display_name,
        "canonical_alias": canonical_alias,
        "api": "openai-completions",
        "baseUrl": norm_base,
        "reasoning": reasoning_flag,
        "input": input_modalities,
        "supportsTools": True,
        "contextWindow": resolved_ctx,
        "maxTokens": resolved_max_tokens,
        "headers": {"ngrok-skip-browser-warning": "true"},
        "compat": dict(compat_info["compat"]),
        "context_window_source": ctx_source,
    }


def sync_omp_models_yml(
    agent_dir: Path,
    base_url: str,
    model_reg: Mapping[str, Any],
    provider_name: str = "sandcastle",
    *,
    bun_bin: Optional[str] = None,
) -> Path:
    """Thread/multiprocess-safely upserts `providers[provider_name]` into `<agent_dir>/models.yml` under `fcntl.flock`."""
    target_dir = Path(agent_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    models_path = target_dir / "models.yml"
    lock_path = target_dir / "models.yml.lock"
    norm_base = normalize_colab_endpoint_url(base_url)

    sanitized_model = {k: v for k, v in model_reg.items() if k in OMP_MODEL_YML_ALLOWED_KEYS}
    if not sanitized_model.get("id"):
        raise ValueError("model_reg must include a non-empty 'id'")
    sanitized_model["baseUrl"] = norm_base

    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        existing_data: Dict[str, Any] = {}
        if models_path.is_file():
            existing_raw = models_path.read_text(encoding="utf-8")
            existing_data = _parse_models_yml_text(existing_raw, models_path, bun_bin=bun_bin)

        next_data = dict(existing_data)
        providers_raw = existing_data.get("providers")
        providers: Dict[str, Any] = dict(providers_raw) if isinstance(providers_raw, Mapping) else {}
        existing_prov_raw = providers.get(provider_name)
        prov: Dict[str, Any] = dict(existing_prov_raw) if isinstance(existing_prov_raw, Mapping) else {}
        prov["baseUrl"] = norm_base
        prov["apiKey"] = "SANDCASTLE_VLLM_API_KEY"
        prov["api"] = "openai-completions"
        prov_headers = dict(prov.get("headers") or {}) if isinstance(prov.get("headers"), Mapping) else {}
        prov_headers["ngrok-skip-browser-warning"] = "true"
        prov["headers"] = prov_headers

        existing_models = list(prov.get("models") or []) if isinstance(prov.get("models"), list) else []
        new_models: List[Dict[str, Any]] = []
        replaced = False
        target_id = str(sanitized_model["id"])
        for entry in existing_models:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("id") or "") == target_id:
                new_models.append(dict(sanitized_model))
                replaced = True
            else:
                new_models.append(dict(entry))
        if not replaced:
            new_models.append(dict(sanitized_model))
        prov["models"] = new_models
        providers[provider_name] = prov
        next_data["providers"] = providers

        if models_path.is_file() and next_data == existing_data:
            return models_path

        rendered = _serialize_models_yml(next_data)
        fd, tmp_name = tempfile.mkstemp(dir=str(target_dir), prefix=".models.yml.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_fh:
                tmp_fh.write(rendered)
                tmp_fh.flush()
                os.fsync(tmp_fh.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, str(models_path))
        except Exception:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return models_path
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _select_served_model_id(
    requested_model: str,
    served_models: Sequence[str],
    model_details: Mapping[str, Mapping[str, Any]],
) -> str:
    clean_req = _strip_precision_suffix(requested_model)
    for sm in served_models:
        if sm == clean_req:
            return sm
    accepted = resolve_expected_model_identifiers(clean_req)
    for sm in served_models:
        if sm.lower() in accepted:
            return sm
        root_val = str((model_details.get(sm) or {}).get("root") or "").strip().lower()
        if root_val and root_val in accepted:
            return sm
    return clean_req


def resolve_and_verify_omp_endpoint(
    cwd: Path | str,
    *,
    explicit_endpoint: str = "",
    explicit_model: str = "",
    explicit_api_key: str = "",
    verify_live: bool = True,
    verify_timeout_sec: float = DEFAULT_VERIFY_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Resolves endpoint URL, credentials, and model identity with strict precedence and bounded `/v1/models` check."""
    work_dir = Path(cwd).resolve()
    project_root = resolve_project_root(work_dir)
    dir_slug = compute_dir_slug(project_root)
    platodoom_home = resolve_platodoom_home()

    checked_sources: List[str] = []
    endpoint_url = ""
    registry_record: Dict[str, Any] = {}
    registry_tok = ""
    registry_model = ""
    gpu_tier = ""

    def _inspect_registry(reg_path: Path, label: str, *, require_slug_match: bool = False) -> Optional[Dict[str, Any]]:
        data, err = _read_json_file_strict(reg_path)
        if err:
            raise RuntimeError(err)
        if not data:
            checked_sources.append(f"{label} (missing)")
            return None
        status = str(data.get("status") or "").strip().lower()
        raw_url = str(data.get("endpoint_url") or "").strip()
        rec_slug = str(data.get("dir_slug") or "").strip()
        if require_slug_match and rec_slug and rec_slug != dir_slug and os.environ.get("PLATODOOM_ALLOW_GLOBAL_ENDPOINT") != "1":
            checked_sources.append(f"{label} (dir_slug mismatch: {rec_slug} != {dir_slug})")
            return None
        if status == "offline":
            checked_sources.append(f"{label} (status=Offline)")
            return None
        if not raw_url:
            checked_sources.append(f"{label} (empty endpoint_url)")
            return None
        checked_sources.append(f"{label} (selected)")
        return data

    # 1. Explicit argument
    if str(explicit_endpoint or "").strip():
        endpoint_url = normalize_colab_endpoint_url(str(explicit_endpoint).strip())
        checked_sources.append("--endpoint argument (selected)")
    else:
        checked_sources.append("--endpoint argument (unset)")

    # 2. Environment variables
    if not endpoint_url:
        for env_key in ("SANDCASTLE_VLLM_URL", "PLATODOOM_ENDPOINT_URL", "VLLM_API_BASE"):
            val = str(os.environ.get(env_key) or "").strip()
            if val:
                endpoint_url = normalize_colab_endpoint_url(val)
                checked_sources.append(f"env:{env_key} (selected)")
                break
            checked_sources.append(f"env:{env_key} (unset)")

    # Read project registry even if URL came from env, for token / context / gpu_tier metadata when matching
    proj_reg_path = project_root / ".sandcastle" / "colab_qwen_endpoint.json"
    proj_data, proj_err = _read_json_file_strict(proj_reg_path)
    if proj_err:
        raise RuntimeError(proj_err)
    if not endpoint_url:
        sel_proj = _inspect_registry(proj_reg_path, str(proj_reg_path))
        if sel_proj:
            registry_record = sel_proj
            endpoint_url = normalize_colab_endpoint_url(str(sel_proj["endpoint_url"]))
    elif proj_data and str(proj_data.get("status") or "").strip().lower() != "offline":
        proj_url_norm = normalize_colab_endpoint_url(str(proj_data.get("endpoint_url") or ""))
        if proj_url_norm and proj_url_norm == endpoint_url:
            registry_record = proj_data

    # 3. Scoped platodoom last_cli_selection & standby state
    last_sel_path = platodoom_home / "last_cli_selection.json"
    standby_path = platodoom_home / "standby" / "endpoint.json"
    if not endpoint_url:
        sel_cli = _inspect_registry(last_sel_path, str(last_sel_path), require_slug_match=True)
        if sel_cli:
            registry_record = sel_cli
            endpoint_url = normalize_colab_endpoint_url(str(sel_cli["endpoint_url"]))
    elif not registry_record:
        last_cli_data, last_cli_err = _read_json_file_strict(last_sel_path)
        if not last_cli_err and last_cli_data and str(last_cli_data.get("status") or "").strip().lower() != "offline":
            rec_slug = str(last_cli_data.get("dir_slug") or "").strip()
            if (not rec_slug or rec_slug == dir_slug or os.environ.get("PLATODOOM_ALLOW_GLOBAL_ENDPOINT") == "1"):
                if normalize_colab_endpoint_url(str(last_cli_data.get("endpoint_url") or "")) == endpoint_url:
                    registry_record = last_cli_data

    if not endpoint_url:
        sel_standby = _inspect_registry(standby_path, str(standby_path))
        if sel_standby:
            registry_record = sel_standby
            endpoint_url = normalize_colab_endpoint_url(str(sel_standby["endpoint_url"]))
    elif not registry_record:
        standby_data, standby_err = _read_json_file_strict(standby_path)
        if not standby_err and standby_data and str(standby_data.get("status") or "").strip().lower() != "offline":
            if normalize_colab_endpoint_url(str(standby_data.get("endpoint_url") or "")) == endpoint_url:
                registry_record = standby_data

    if not endpoint_url:
        raise RuntimeError(
            "No active Colab/platodoom OpenAI endpoint found. Checked sources: " + "; ".join(checked_sources)
        )

    if registry_record:
        registry_model = str(registry_record.get("active_model") or registry_record.get("model") or "").strip()
        gpu_tier = str(registry_record.get("gpu_tier") or registry_record.get("gpu_type") or "").strip()

    # Model identity resolution (NO static fallback to qwen-27b / qwen3.8-27b!)
    platodoom_cfg_model = ""
    proj_cfg_path = project_root / ".platodoom.json"
    proj_cfg, proj_cfg_err = _read_json_file_strict(proj_cfg_path)
    if proj_cfg_err:
        raise RuntimeError(proj_cfg_err)
    if proj_cfg:
        platodoom_cfg_model = str(proj_cfg.get("model") or proj_cfg.get("default_model") or "").strip()

    raw_model = (
        str(explicit_model or "").strip()
        or str(os.environ.get("PLATODOOM_MODEL") or "").strip()
        or registry_model
        or platodoom_cfg_model
    )
    if not raw_model:
        raise RuntimeError(
            "No model specified via --model, PLATODOOM_MODEL, .platodoom.json, or active project endpoint registry."
        )
    requested_model = _strip_precision_suffix(raw_model)

    # Validate catalog recognition (allow custom HF repo ids / catalog custom models if present)
    plato_cfg = resolve_platodoom_config(cwd=project_root)
    catalog = get_model_catalog(config=plato_cfg)
    is_known_catalog = False
    try:
        resolve_model_profile(requested_model, config=plato_cfg)
        is_known_catalog = True
    except ValueError:
        is_known_catalog = requested_model in catalog

    if not is_known_catalog and "/" not in requested_model and not verify_live:
        raise RuntimeError(
            f"Unrecognized model '{requested_model}' (not in platodoom catalog and not an HF repo identifier)."
        )

    api_key = resolve_serving_bearer_token(
        bearer_token=explicit_api_key or None,
        record=registry_record or None,
    )

    served_model_id = requested_model
    live_model_record: Optional[Dict[str, Any]] = None
    if verify_live:
        probe = probe_colab_qwen_endpoint(
            endpoint_url,
            api_key=None if api_key == "EMPTY" else api_key,
            bearer_token=None if api_key == "EMPTY" else api_key,
            expected_model=requested_model,
            timeout_sec=min(max(float(verify_timeout_sec), 0.2), 5.0),
            verify_inference=False,
        )
        if not probe.get("ok"):
            if probe.get("model_mismatch"):
                raise RuntimeError(
                    f"Model mismatch at {endpoint_url}: requested '{requested_model}', "
                    f"but endpoint serves {probe.get('served_models') or []}"
                )
            raise RuntimeError(
                f"Endpoint pre-flight verification failed for {endpoint_url}: {probe.get('error') or 'unreachable'}"
            )
        served_list = list(probe.get("served_models") or probe.get("models") or [])
        model_details = dict(probe.get("model_details") or {})
        served_model_id = _select_served_model_id(requested_model, served_list, model_details)
        live_model_record = model_details.get(served_model_id)

    return {
        "endpoint_url": endpoint_url,
        "api_key": api_key,
        "requested_model": requested_model,
        "served_model_id": served_model_id,
        "registry_record": registry_record,
        "live_model_record": live_model_record,
        "gpu_tier": gpu_tier,
        "checked_sources": checked_sources,
    }


def _emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _format_tool_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        content = result.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
        return json.dumps(result, ensure_ascii=False)
    if result is None:
        return ""
    return str(result)


def _extract_modified_path(tool_name: str, args: Any) -> Optional[str]:
    if tool_name not in ("write", "edit", "ast_edit", "write_file", "edit_file") or not isinstance(args, Mapping):
        return None
    for key in ("path", "file_path", "Path", "target"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_assistant_error(msg: Mapping[str, Any]) -> Optional[str]:
    stop_reason = str(msg.get("stopReason") or "").strip().lower()
    err_msg = str(msg.get("errorMessage") or "").strip()
    err_status = msg.get("errorStatus")
    if stop_reason == "error" or err_msg:
        base = err_msg or "omp assistant turn ended with stopReason=error"
        return f"HTTP {err_status}: {base}" if err_status else base
    return None


def _extract_assistant_text_and_thinking(msg: Mapping[str, Any]) -> Tuple[str, str]:
    text_parts: List[str] = []
    think_parts: List[str] = []
    for block in msg.get("content") or []:
        if not isinstance(block, Mapping):
            continue
        b_type = block.get("type")
        if b_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif b_type == "thinking" and isinstance(block.get("thinking"), str):
            think_parts.append(block["thinking"])
    return "".join(text_parts).strip(), "".join(think_parts)


_MD_FENCE_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
_MD_HEADING_RE = re.compile(r"^[ ]{0,3}(#{1,6})\s+(.*?)(?:\s+#+\s*)?$")
_REF_PATH_RE = re.compile(
    r"(?:wayfinder/[A-Za-z0-9_.\-/]+|handoff/[A-Za-z0-9_.\-/]+|\.sandcastle/[A-Za-z0-9_.\-/]+)"
)
_VLLM_MAX_LEN_RE = re.compile(r"maximum context length (?:is|of)\s+(\d+)\s+tokens", re.IGNORECASE)
_VLLM_INPUT_TOK_RE = re.compile(
    r"(?:contains(?:\s+at\s+least)?\s+(\d+)\s+(?:input\s+)?tokens|requested\s+(\d+)\s+tokens|(\d+)\s+input\s+tokens)",
    re.IGNORECASE,
)

_EPHEMERAL_OVERLAY_PATHS: set[str] = set()
_EPHEMERAL_OVERLAY_LOCK = threading.Lock()


def _cleanup_registered_overlays() -> None:
    with _EPHEMERAL_OVERLAY_LOCK:
        paths = list(_EPHEMERAL_OVERLAY_PATHS)
        _EPHEMERAL_OVERLAY_PATHS.clear()
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


atexit.register(_cleanup_registered_overlays)


def estimate_prompt_tokens(text: str, *, chars_per_token: float = 3.0) -> int:
    """Conservative upper-bound token count for mixed Markdown/code/JSON prompts."""
    if not text:
        return 0
    by_chars = math.ceil(len(text) / max(1.0, float(chars_per_token)))
    by_bytes = math.ceil(len(text.encode("utf-8", errors="replace")) / 3.2)
    return max(by_chars, by_bytes)


def extract_markdown_section_toc(text: str, *, max_depth: int = 3) -> List[Dict[str, Any]]:
    """Extracts fenced-code-block-safe Markdown headings with 1-indexed `[start_line, end_line]` ranges."""
    lines = text.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    start_idx = 0
    if lines[0].strip() == "---":
        for idx in range(1, total_lines):
            if lines[idx].strip() in ("---", "..."):
                start_idx = idx + 1
                break

    headings: List[Dict[str, Any]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for idx in range(start_idx, total_lines):
        line = lines[idx]
        m_fence = _MD_FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            trailing = m_fence.group(2).strip()
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                continue
            elif marker[0] == fence_char and len(marker) >= fence_len and not trailing:
                in_fence = False
                fence_char = ""
                fence_len = 0
                continue
        if in_fence:
            continue

        m_head = _MD_HEADING_RE.match(line)
        if m_head:
            level = len(m_head.group(1))
            if level <= max_depth:
                title = m_head.group(2).strip()
                if title:
                    headings.append(
                        {
                            "level": level,
                            "title": title,
                            "start_line": idx + 1,
                            "end_line": total_lines,
                        }
                    )

    for i, item in enumerate(headings):
        end_line = total_lines
        for nxt in headings[i + 1 :]:
            if nxt["level"] <= item["level"]:
                end_line = max(item["start_line"], nxt["start_line"] - 1)
                break
        item["end_line"] = end_line
        item["est_tokens"] = estimate_prompt_tokens("\n".join(lines[item["start_line"] - 1 : end_line]))

    return headings


def extract_referenced_paths(text: str, *, limit: int = 24) -> List[str]:
    """Extracts unique Sandcastle/wayfinder/handoff/spec file paths referenced in a prompt."""
    seen: set[str] = set()
    out: List[str] = []
    for match in _REF_PATH_RE.finditer(text):
        p = match.group(0).rstrip(".,;:)`'\"")
        if not p or p in seen or "{{" in p or "*" in p:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def parse_vllm_context_overflow_error(error_text: str) -> Optional[Dict[str, int]]:
    """Parses upstream vLLM/OpenAI HTTP 400 maximum-context-length errors."""
    raw = str(error_text or "")
    m_max = _VLLM_MAX_LEN_RE.search(raw)
    if not m_max:
        return None
    max_len = int(m_max.group(1))
    m_in = _VLLM_INPUT_TOK_RE.search(raw)
    if m_in:
        in_tok = int(next(g for g in m_in.groups() if g is not None))
    else:
        in_tok = max_len + 1
    return {"max_model_len": max_len, "input_tokens": in_tok}


def write_ephemeral_omp_config_overlay(
    compaction_cfg: Mapping[str, Any],
    *,
    extra_sections: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Writes a unique per-invocation `omp --config` YAML overlay without touching `~/.omp/agent/config.yml`."""
    payload: Dict[str, Any] = {"compaction": dict(compaction_cfg)}
    if extra_sections:
        for k, v in extra_sections.items():
            if isinstance(v, Mapping):
                payload[str(k)] = dict(v)
            else:
                payload[str(k)] = v
    rendered = _serialize_models_yml(payload)
    tmp_fh = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"omp-overlay-{os.getpid()}-",
        suffix=".yml",
        delete=False,
        encoding="utf-8",
    )
    try:
        tmp_fh.write(rendered)
        tmp_fh.flush()
        tmp_path = Path(tmp_fh.name)
    finally:
        tmp_fh.close()
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    with _EPHEMERAL_OVERLAY_LOCK:
        _EPHEMERAL_OVERLAY_PATHS.add(str(tmp_path))
    return tmp_path


def _unlink_ephemeral_overlay(overlay_path: Optional[Path]) -> None:
    if overlay_path is None:
        return
    p_str = str(overlay_path)
    with _EPHEMERAL_OVERLAY_LOCK:
        _EPHEMERAL_OVERLAY_PATHS.discard(p_str)
    try:
        os.unlink(p_str)
    except OSError:
        pass


def _extract_preserved_header_and_workflow(
    prompt_text: str,
    toc: Sequence[Mapping[str, Any]],
) -> Tuple[str, str, str]:
    """Extracts (opening role header, workflow/completion instructions, closing tail snippet) for pointer prompts."""
    lines = prompt_text.splitlines()
    total_lines = len(lines)
    header_end = 0
    idx = 0
    if lines and lines[0].strip() == "---":
        idx = 1
        while idx < total_lines and lines[idx].strip() not in ("---", "..."):
            idx += 1
        if idx < total_lines:
            idx += 1
        header_end = idx

    # Capture role heading / brief lines until first `## Input Context` or blank-bounded summary
    extra_lines: List[str] = []
    for j in range(header_end, min(total_lines, header_end + 12)):
        ln = lines[j]
        if re.match(r"^[ ]{0,3}##\s+Input\s+Context\b", ln, re.IGNORECASE):
            extra_lines.append(ln)
            break
        if len(ln) > 220:
            break
        extra_lines.append(ln)
    header_block = "\n".join(lines[:header_end] + extra_lines).strip()
    if len(header_block) > 700:
        header_block = header_block[:700].rstrip()

    # Preserve verbatim Workflow / Handoff / Deliverables section if present
    workflow_chunks: List[str] = []
    for item in toc:
        title_low = str(item.get("title") or "").strip().lower()
        if any(kw in title_low for kw in ("workflow", "handoff", "deliverable", "completion", "instructions")):
            if int(item.get("level", 2)) <= 2:
                s_l = max(1, int(item["start_line"]))
                e_l = min(total_lines, int(item["end_line"]))
                chunk = "\n".join(lines[s_l - 1 : e_l]).strip()
                if chunk and chunk not in workflow_chunks:
                    workflow_chunks.append(chunk)
    workflow_block = "\n\n".join(workflow_chunks).strip()

    # Capture compact closing tail snippet (preserves trailing user directives / markers)
    non_empty_tail = [ln.strip() for ln in lines[-6:] if ln.strip()]
    tail_block = ""
    if non_empty_tail:
        candidate_tail = "\n".join(non_empty_tail[-3:])
        if len(candidate_tail) > 260:
            candidate_tail = candidate_tail[-260:].lstrip()
        if candidate_tail and candidate_tail not in workflow_block and candidate_tail not in header_block:
            tail_block = candidate_tail

    return header_block, workflow_block, tail_block


def prepare_preflight_prompt(
    prompt_text: str,
    work_dir: Path,
    context_profile: Mapping[str, Any],
    *,
    budget_override_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Measures Turn-0 prompt tokens against `B_turn0` and deterministically offloads oversized context in <= 50ms."""
    t0 = time.perf_counter()
    chars_per_token = float(context_profile.get("chars_per_token") or 3.0)
    eff_budget = (
        int(budget_override_tokens)
        if (budget_override_tokens is not None and int(budget_override_tokens) > 0)
        else int(context_profile.get("max_turn0_prompt_tokens") or 2048)
    )
    est_tokens = estimate_prompt_tokens(prompt_text, chars_per_token=chars_per_token)
    if est_tokens <= eff_budget:
        return {
            "offloaded": False,
            "effective_prompt": prompt_text,
            "estimated_tokens": est_tokens,
            "turn0_budget_tokens": eff_budget,
            "compacted_tokens": est_tokens,
            "dossier_path": None,
            "dossier_rel_path": None,
            "section_count": 0,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Deterministically offload verbatim prompt into worktree-local gitignored transient dossier
    base_dir = Path(work_dir).resolve()
    tmp_dir = base_dir / ".sandcastle" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dossier_name = f"omp-context-{os.getpid()}-{uuid.uuid4().hex[:8]}.md"
    dossier_path = tmp_dir / dossier_name
    dossier_rel_path = f".sandcastle/tmp/{dossier_name}"

    fd, stage_name = tempfile.mkstemp(dir=str(tmp_dir), prefix=".omp-context-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(prompt_text)
        os.replace(stage_name, str(dossier_path))
    except Exception:
        try:
            if os.path.exists(stage_name):
                os.unlink(stage_name)
        except OSError:
            pass
        raise

    total_lines = len(prompt_text.splitlines())
    toc = extract_markdown_section_toc(prompt_text, max_depth=3)
    ref_paths = extract_referenced_paths(prompt_text, limit=18)
    header_block, workflow_block, tail_block = _extract_preserved_header_and_workflow(prompt_text, toc)

    target_cap = max(384, int(math.floor(eff_budget * 0.85)))
    live_ctx = int(context_profile.get("context_window") or 16384)

    def _render_pointer(max_toc: int, max_wf_chars: int, max_refs: int) -> str:
        parts: List[str] = []
        if header_block:
            parts.append(header_block)
        parts.append(
            "## Pre-Flight Offloaded Context Dossier\n"
            f"The full expanded Turn-0 prompt (~{est_tokens} est. tokens, {total_lines} lines) exceeded the safe "
            f"Turn-0 input budget ({eff_budget} tokens for live `max_model_len={live_ctx}`). "
            f"The complete verbatim context has been saved to `{dossier_rel_path}` (gitignored; do NOT commit it).\n"
            f"Read only the needed line ranges via inline path selectors `read(path=\"{dossier_rel_path}:<start>-<end>\")` "
            "or directly inspect referenced workspace files with `grep` / ranged `read`."
        )
        if toc and max_toc > 0:
            toc_lines = ["### Dossier Section Index", "| Section | Lines | Est. Tokens |", "| :--- | :--- | ---: |"]
            for h in toc[:max_toc]:
                indent = "  " * max(0, int(h["level"]) - 1)
                title_clean = str(h["title"]).replace("|", "\\|").replace("\n", " ").strip()[:84]
                toc_lines.append(
                    f"| {indent}H{h['level']} {title_clean} | `[L{h['start_line']}-L{h['end_line']}]` | ~{h['est_tokens']} |"
                )
            if len(toc) > max_toc:
                toc_lines.append(f"_+{len(toc) - max_toc} more sections in `{dossier_rel_path}`._")
            parts.append("\n".join(toc_lines))
        if ref_paths and max_refs > 0:
            parts.append(
                "### Referenced Workspace Files\n"
                + "\n".join(f"- `{p}`" for p in ref_paths[:max_refs])
            )
        if workflow_block and max_wf_chars > 0:
            wf_txt = workflow_block if len(workflow_block) <= max_wf_chars else (workflow_block[:max_wf_chars].rstrip() + "\n...(read full workflow from dossier)")
            parts.append(wf_txt)
        if tail_block:
            parts.append(f"### Closing Directive\n{tail_block}")
        return "\n\n".join(parts).strip() + "\n"

    pointer_prompt = _render_pointer(20, 2200, 14)
    if estimate_prompt_tokens(pointer_prompt, chars_per_token=chars_per_token) > target_cap:
        for toc_lim, wf_lim, ref_lim in (
            (12, 1400, 10),
            (8, 900, 8),
            (5, 500, 6),
            (3, 250, 4),
            (0, 0, 0),
        ):
            cand = _render_pointer(toc_lim, wf_lim, ref_lim)
            pointer_prompt = cand
            if estimate_prompt_tokens(cand, chars_per_token=chars_per_token) <= target_cap:
                break

    if estimate_prompt_tokens(pointer_prompt, chars_per_token=chars_per_token) > target_cap:
        max_chars = max(512, int(target_cap * chars_per_token) - 240)
        suffix = f"\n\n### Closing Directive\n{tail_block}\n" if tail_block else "\n"
        pointer_prompt = pointer_prompt[: max(256, max_chars - len(suffix))].rstrip() + suffix
        # Guard against dense multi-byte UTF-8 (e.g. CJK/emoji) where UTF-8 byte tokens exceed char ratio
        for _ in range(8):
            cur_tok = estimate_prompt_tokens(pointer_prompt, chars_per_token=chars_per_token)
            if cur_tok <= target_cap or len(pointer_prompt) <= 128:
                break
            ratio = (target_cap * 0.92) / float(max(1, cur_tok))
            next_len = max(128, int(len(pointer_prompt) * ratio))
            pointer_prompt = pointer_prompt[:next_len].rstrip() + "\n"

    compacted_tokens = estimate_prompt_tokens(pointer_prompt, chars_per_token=chars_per_token)
    return {
        "offloaded": True,
        "effective_prompt": pointer_prompt,
        "estimated_tokens": est_tokens,
        "turn0_budget_tokens": eff_budget,
        "compacted_tokens": compacted_tokens,
        "dossier_path": str(dossier_path),
        "dossier_rel_path": dossier_rel_path,
        "section_count": len(toc),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


SANDCASTLE_PROMISE_COMPLETE = "<promise>COMPLETE</promise>"
_YAML_FRONTMATTER_NAME_RE = re.compile(r"^---\s*\n[\s\S]*?^name:\s*([A-Za-z0-9_.-]+)\s*$", re.MULTILINE)
_PROMPT_RUN_ID_RE = re.compile(r"\bRun:\s*`?([A-Za-z0-9_-]+)`?")
_PROMPT_BRANCH_RE = re.compile(r"\bBranch:\s*`?([^`,\s\)]+)`?")
_WRITE_HANDOFF_PATH_RE = re.compile(
    r"(?:\bwrite\b|\band\b)\s+`?(handoff/[A-Za-z0-9_./-]+\.txt)`?",
    re.IGNORECASE,
)


def extract_sandcastle_completion_contract(
    prompt_text: str,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Extracts the Sandcastle handoff manifest & `<promise>COMPLETE</promise>` completion contract from a prompt.

    Returns `None` for non-Sandcastle interactive/ad-hoc prompts so ordinary `omp`/`platodoom` turns are untouched.
    """
    raw = str(prompt_text or "")
    has_promise_directive = SANDCASTLE_PROMISE_COMPLETE in raw
    eff_env = env if env is not None else os.environ

    agent_id = ""
    m_name = _YAML_FRONTMATTER_NAME_RE.search(raw[:1200])
    if m_name:
        agent_id = m_name.group(1).strip()
    if not agent_id:
        agent_id = str(eff_env.get("SANDCASTLE_AGENT_ID") or "").strip()

    run_id = ""
    m_run = _PROMPT_RUN_ID_RE.search(raw[:2400])
    if m_run:
        run_id = m_run.group(1).strip()
    if not run_id:
        run_id = str(eff_env.get("SANDCASTLE_RUN_ID") or "").strip()

    branch = ""
    m_branch = _PROMPT_BRANCH_RE.search(raw[:2400])
    if m_branch:
        branch = m_branch.group(1).strip()

    handoff_paths: List[str] = []
    seen_handoffs: set[str] = set()
    closing_lines: List[str] = []
    for line in raw.splitlines():
        low = line.lower()
        if "handoff/" in low or "complete</promise>" in low or "blackboard done" in low:
            closing_lines.append(line)
        if "write" not in low or "handoff/" not in low:
            continue
        for m_h in _WRITE_HANDOFF_PATH_RE.finditer(line):
            rel = m_h.group(1).strip().rstrip(".,;:)`'\"")
            if not rel or "prior-iteration/" in rel or "{{" in rel or "*" in rel:
                continue
            if rel not in seen_handoffs:
                seen_handoffs.add(rel)
                handoff_paths.append(rel)

    if not has_promise_directive and not (agent_id and run_id and handoff_paths):
        return None

    closing_blob = "\n".join(closing_lines)
    requires_blackboard_done = (
        ("blackboard done" in closing_blob.lower())
        and agent_id not in ("branch-merger", "merger")
        and (bool(str(eff_env.get("SANDCASTLE_BLACKBOARD_URL") or "").strip()) or "blackboard done" in raw.lower())
    )
    requires_commit = bool(re.search(r"\bcommit\b", closing_blob, re.IGNORECASE))

    return {
        "agent_id": agent_id or "sandcastle-agent",
        "run_id": run_id or None,
        "branch": branch or None,
        "handoff_paths": handoff_paths,
        "requires_blackboard_done": requires_blackboard_done,
        "requires_commit": requires_commit,
        "promise_token": SANDCASTLE_PROMISE_COMPLETE,
    }


def build_pinned_completion_system_addendum(contract: Mapping[str, Any]) -> str:
    """Builds a concise, compaction-proof `--append-system-prompt` completion contract for Sandcastle turns."""
    agent_id = str(contract.get("agent_id") or "sandcastle-agent")
    run_id = str(contract.get("run_id") or "").strip()
    handoff_paths = [str(p) for p in (contract.get("handoff_paths") or []) if str(p).strip()]
    promise_tok = str(contract.get("promise_token") or SANDCASTLE_PROMISE_COMPLETE)

    lines: List[str] = [
        "SANDCASTLE COMPLETION CONTRACT (PINNED — SURVIVES COMPACTION):",
        f"- Role: {agent_id}" + (f" | Run: {run_id}" if run_id else ""),
    ]
    if handoff_paths:
        paths_str = ", ".join(f"`{p}`" for p in handoff_paths)
        lines.append(f"- Required handoff file(s): {paths_str} (EXACT path; never use `HANDOFF.md`).")
    close_steps: List[str] = []
    if contract.get("requires_blackboard_done"):
        close_steps.append("`blackboard done`")
    if contract.get("requires_commit"):
        close_steps.append("`git add` + `git commit`")
    close_steps.append(f"end final reply with `{promise_tok}`")
    lines.append("- Close-out: " + ", ".join(close_steps) + ".")
    lines.append(
        f"- COMPACTION & ANTI-THRASH RULE: Keep goal, modified files, exact handoff path(s), and `{promise_tok}` in every summary. Once verified, stop rebuilding and close out immediately."
    )
    return "\n".join(lines)


def _check_worktree_uncommitted_changes(work_dir: Path) -> bool:
    """Returns True if `work_dir` is a git worktree with non-transient uncommitted changes."""
    if not (work_dir / ".git").exists():
        return False
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc.returncode != 0:
            return False
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            path_part = re.sub(r"^[MADRCU?!]{1,2}\s+", "", line).strip()
            if path_part.startswith(".sandcastle/tmp/") or path_part == ".sandcastle/tmp" or path_part.startswith("logs/"):
                continue
            return True
    except Exception:
        return False
    return False


def evaluate_sandcastle_completion(
    work_dir: Path,
    response_text: str,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Objectively evaluates whether the agent satisfied the Sandcastle handoff manifest & promise completion contract."""
    promise_tok = str(contract.get("promise_token") or SANDCASTLE_PROMISE_COMPLETE)
    has_promise = promise_tok in str(response_text or "")
    missing_handoffs: List[str] = []
    for rel in contract.get("handoff_paths") or []:
        rel_s = str(rel).strip()
        if not rel_s:
            continue
        fp = (work_dir / rel_s).resolve()
        if not fp.is_file() or fp.stat().st_size == 0:
            missing_handoffs.append(rel_s)
    uncommitted = bool(contract.get("requires_commit")) and _check_worktree_uncommitted_changes(work_dir)
    satisfied = has_promise and len(missing_handoffs) == 0
    return {
        "satisfied": satisfied,
        "has_promise": has_promise,
        "missing_handoffs": missing_handoffs,
        "uncommitted_changes": uncommitted,
    }


def build_completion_closeout_prompt(
    contract: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    """Builds a targeted, bounded close-out checklist prompt for the single `--resume`/`--continue` recovery turn."""
    promise_tok = str(contract.get("promise_token") or SANDCASTLE_PROMISE_COMPLETE)
    missing_handoffs = [str(p) for p in (evaluation.get("missing_handoffs") or []) if str(p).strip()]
    steps: List[str] = []
    step_num = 1
    if missing_handoffs:
        paths_str = ", ".join(f"`{p}`" for p in missing_handoffs)
        steps.append(
            f"{step_num}. Write the required handoff manifest file(s): {paths_str} listing every created/modified file (one path per line). Do NOT write `HANDOFF.md` instead."
        )
        step_num += 1
    if contract.get("requires_blackboard_done"):
        steps.append(f"{step_num}. Run `blackboard done` (ignore only if no Blackboard server is active).")
        step_num += 1
    if evaluation.get("uncommitted_changes") or (contract.get("requires_commit") and missing_handoffs):
        steps.append(f"{step_num}. Stage and commit your work (including the handoff file) with `git add` and `git commit`.")
        step_num += 1
    steps.append(f"{step_num}. End your final reply with the exact literal tag `{promise_tok}`.")
    return (
        "CLOSE-OUT CHECKLIST (YOUR IMPLEMENTATION/PLANNING WORK IS ALREADY COMPLETE — DO NOT RE-IMPLEMENT OR RE-VERIFY PASSING CODE):\n"
        "You stopped without completing the mandatory Sandcastle close-out contract.\n"
        "Execute ONLY these remaining steps right now:\n"
        + "\n".join(steps)
    )


def run_omp_turn(
    prompt: str,
    cwd: Path | str,
    endpoint_url: str,
    api_key: str,
    model_id: str,
    context_window: Optional[int] = None,
    max_tokens: Optional[int] = None,
    emit_ndjson: bool = False,
    stream_callback: Optional[Callable[[Any], None]] = None,
    reasoning_callback: Optional[Callable[[str], None]] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    *,
    no_session: bool = False,
    session_dir: Optional[Path | str] = None,
    continue_session: bool = False,
    resume_session: Optional[str] = None,
    tools: Optional[Sequence[str] | str] = None,
    thinking: Optional[str] = None,
    auto_approve: bool = False,
    approval_mode: Optional[str] = None,
    agent_dir: Optional[Path | str] = None,
    provider_name: str = "sandcastle",
    verify_endpoint: bool = True,
    verify_timeout_sec: float = DEFAULT_VERIFY_TIMEOUT_SEC,
    gpu_tier: Optional[str] = None,
    live_model_record: Optional[Mapping[str, Any]] = None,
    registry_record: Optional[Mapping[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    compact_tools: bool = False,
    builtin_tools: Optional[Sequence[str] | str] = None,
    mcp_servers: Optional[Sequence[str] | str] = None,
    mcp_pinned_tools: Optional[Sequence[str] | str] = None,
    vision: Optional[bool] = None,
    enable_preflight_offload: bool = True,
    enable_completion_closeout: bool = True,
) -> Dict[str, Any]:
    """Executes one agent turn through the native `omp` harness with model-aware context budgeting & compaction overlays."""
    norm_url = normalize_colab_endpoint_url(endpoint_url)
    if not norm_url:
        err = "Empty endpoint_url provided to run_omp_turn"
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err}})
        return {"ok": False, "exit_code": 1, "error": err, "response": "", "tool_events": [], "files_modified": []}

    raw_model = str(model_id or "").strip()
    if not raw_model:
        err = "Empty model_id provided to run_omp_turn"
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err}})
        return {"ok": False, "exit_code": 1, "error": err, "response": "", "tool_events": [], "files_modified": []}

    prompt_text = str(prompt or "")
    if not prompt_text.strip() and not continue_session and not resume_session:
        err = "Empty prompt provided to run_omp_turn"
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err}})
        return {"ok": False, "exit_code": 2, "error": err, "response": "", "tool_events": [], "files_modified": []}

    clean_model_id = _strip_precision_suffix(raw_model)

    try:
        omp_bin, bun_bin, env = resolve_omp_runtime_env()
    except FileNotFoundError as exc:
        err = str(exc)
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err}})
        return {"ok": False, "exit_code": 127, "error": err, "response": "", "tool_events": [], "files_modified": []}

    eff_agent_dir = resolve_omp_agent_dir(agent_dir)
    if agent_dir is not None:
        env["PI_CODING_AGENT_DIR"] = str(eff_agent_dir)

    eff_live_record = dict(live_model_record) if isinstance(live_model_record, Mapping) else None
    served_model_id = clean_model_id
    if verify_endpoint:
        probe = probe_colab_qwen_endpoint(
            norm_url,
            api_key=None if api_key in ("", "EMPTY") else api_key,
            bearer_token=None if api_key in ("", "EMPTY") else api_key,
            expected_model=clean_model_id,
            timeout_sec=min(max(float(verify_timeout_sec), 0.2), 5.0),
            verify_inference=False,
        )
        if not probe.get("ok"):
            if probe.get("model_mismatch"):
                err = (
                    f"Model mismatch at {norm_url}: requested '{clean_model_id}', "
                    f"but endpoint serves {probe.get('served_models') or []}"
                )
            else:
                err = f"Endpoint pre-flight check failed for {norm_url}: {probe.get('error') or 'unreachable'}"
            if emit_ndjson:
                _emit({"event": "result", "result": {"error": err}})
            return {"ok": False, "exit_code": 1, "error": err, "response": "", "tool_events": [], "files_modified": []}
        served_list = list(probe.get("served_models") or probe.get("models") or [])
        details_map = dict(probe.get("model_details") or {})
        served_model_id = _select_served_model_id(clean_model_id, served_list, details_map)
        if eff_live_record is None:
            eff_live_record = details_map.get(served_model_id)

    # Resolve explicit tool selection without silently stripping native omp agency (Spec 873 §3.2 & Spec 874 §4.4.3)
    eff_compact_tools = bool(compact_tools or os.environ.get("SANDCASTLE_OMP_COMPACT_TOOLS") == "1")
    eff_tools_list: Optional[List[str]] = None
    if tools is not None:
        if isinstance(tools, str):
            parsed_t = [t.strip() for t in tools.split(",") if t.strip()]
        else:
            parsed_t = [str(t).strip() for t in tools if str(t).strip()]
        if parsed_t:
            eff_tools_list = parsed_t
    elif eff_compact_tools:
        eff_tools_list = list(OMP_COMPACT_CODING_TOOLS)
        print(
            f"[omp-context] Explicit compact_tools profile active: {','.join(eff_tools_list)}",
            file=sys.stderr,
            flush=True,
        )

    work_dir = Path(cwd).resolve()
    mcp_allowlist = resolve_scoped_mcp_allowlist(
        work_dir,
        explicit_builtin_tools=builtin_tools,
        explicit_servers=mcp_servers,
        explicit_pinned_tools=mcp_pinned_tools,
        explicit_vision=vision,
        env=os.environ,
    )
    if eff_tools_list is None and mcp_allowlist["builtin_tools"]:
        eff_tools_list = list(mcp_allowlist["builtin_tools"])
        print(
            f"[omp-tools] Scoped built-in tool allowlist active: {','.join(eff_tools_list)}",
            file=sys.stderr,
            flush=True,
        )
    if mcp_allowlist["pinned_tools"]:
        base_tools = (
            list(eff_tools_list)
            if eff_tools_list is not None
            else (
                list(mcp_allowlist["builtin_tools"])
                if mcp_allowlist.get("builtin_tools")
                else (list(OMP_COMPACT_CODING_TOOLS) if eff_compact_tools else list(OMP_DEFAULT_BUILTIN_TOOLS))
            )
        )
        for pt in mcp_allowlist["pinned_tools"]:
            if pt not in base_tools:
                base_tools.append(pt)
        eff_tools_list = base_tools
        print(
            f"[omp-mcp] Scoped MCP allowlist active (servers={','.join(mcp_allowlist['servers'])}, "
            f"pinned={','.join(mcp_allowlist['pinned_tools'])})",
            file=sys.stderr,
            flush=True,
        )
    elif mcp_allowlist["servers"]:
        print(
            f"[omp-mcp] Scoped MCP servers active (xd:// lazy mode): {','.join(mcp_allowlist['servers'])}",
            file=sys.stderr,
            flush=True,
        )
    eff_config = config if config is not None else resolve_platodoom_config(cwd=resolve_project_root(work_dir))

    def _build_reg_and_profile(
        *,
        ctx_override: Optional[int],
        live_rec: Optional[Mapping[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        reg = resolve_omp_model_registration(
            served_model_id,
            base_url=norm_url,
            live_model_record=live_rec,
            registry_record=registry_record,
            gpu_tier=gpu_tier,
            context_window_override=ctx_override,
            max_tokens_override=max_tokens,
            active_tools=eff_tools_list,
            compact_tools=eff_compact_tools,
            vision_override=mcp_allowlist.get("vision"),
            config=eff_config,
        )
        prof = resolve_model_context_profile(
            reg.get("canonical_alias") or served_model_id,
            context_window=int(reg["contextWindow"]),
            active_tools=eff_tools_list,
            compact_tools=eff_compact_tools,
            requested_max_tokens=int(reg["maxTokens"]),
            config=eff_config,
        )
        return reg, prof

    try:
        model_reg, context_profile = _build_reg_and_profile(ctx_override=context_window, live_rec=eff_live_record)
    except RuntimeError as reg_err:
        err = str(reg_err)
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err}})
        return {
            "ok": False,
            "exit_code": 1,
            "error": err,
            "response": "",
            "tool_events": [],
            "files_modified": [],
        }

    if not context_profile.get("feasible", True):
        err = str(context_profile.get("budget_warning") or f"Unfeasible context budget for {served_model_id}")
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err}})
        return {
            "ok": False,
            "exit_code": 1,
            "error": err,
            "response": "",
            "tool_events": [],
            "files_modified": [],
            "context_profile": context_profile,
        }

    if context_profile.get("budget_warning"):
        warn_msg = str(context_profile["budget_warning"])
        print(f"[omp-context warning] {warn_msg}", file=sys.stderr, flush=True)
        if event_callback is not None:
            event_callback({"type": "context_budget_warning", "message": warn_msg, "context_profile": context_profile})

    def _run_preflight(
        active_profile: Mapping[str, Any],
        *,
        budget_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not enable_preflight_offload or not prompt_text.strip():
            est = estimate_prompt_tokens(prompt_text)
            return {
                "offloaded": False,
                "effective_prompt": prompt_text,
                "estimated_tokens": est,
                "turn0_budget_tokens": int(active_profile.get("max_turn0_prompt_tokens") or 2048),
                "compacted_tokens": est,
                "dossier_path": None,
                "dossier_rel_path": None,
                "section_count": 0,
                "elapsed_ms": 0.0,
            }
        pf = prepare_preflight_prompt(prompt_text, work_dir, active_profile, budget_override_tokens=budget_override)
        if pf["offloaded"]:
            pf_summary = (
                f"[omp-context preflight] Offloaded Turn-0 prompt ({pf['estimated_tokens']} est. tok > "
                f"budget {pf['turn0_budget_tokens']} tok for live max_model_len={active_profile['context_window']}) "
                f"to {pf['dossier_rel_path']} -> pointer prompt {pf['compacted_tokens']} est. tok ({pf['elapsed_ms']}ms)"
            )
            print(pf_summary, file=sys.stderr, flush=True)
            if event_callback is not None:
                event_callback({"type": "context_preflight", **{k: v for k, v in pf.items() if k != "effective_prompt"}})
            if emit_ndjson:
                _emit(
                    {
                        "event": "step_update",
                        "step_update": {
                            "step_index": 0,
                            "state": "ACTIVE",
                            "step_type": "context_preflight",
                            "thought": pf_summary,
                            "preflight": {k: v for k, v in pf.items() if k != "effective_prompt"},
                        },
                    }
                )
        return pf

    preflight_info = _run_preflight(context_profile)
    completion_contract = extract_sandcastle_completion_contract(prompt_text, env=env)
    pinned_completion_addendum = (
        build_pinned_completion_system_addendum(completion_contract)
        if completion_contract is not None
        else None
    )

    def _execute_omp_attempt(
        eff_prompt: str,
        active_reg: Mapping[str, Any],
        active_profile: Mapping[str, Any],
        *,
        resume_session_override: Optional[str] = None,
        continue_session_override: bool = False,
        step_index_offset: int = 0,
    ) -> Dict[str, Any]:
        sync_omp_models_yml(
            eff_agent_dir,
            norm_url,
            active_reg,
            provider_name=provider_name,
            bun_bin=bun_bin,
        )

        attempt_env = dict(env)
        attempt_env["SANDCASTLE_VLLM_API_KEY"] = api_key or "EMPTY"
        qualified_model = f"{provider_name}/{active_reg['id']}"
        attempt_env.setdefault("PI_SMOL_MODEL", qualified_model)
        attempt_env.setdefault("PI_SLOW_MODEL", qualified_model)
        attempt_env.setdefault("PI_PLAN_MODEL", qualified_model)

        overlay_path: Optional[Path] = None
        mcp_json_path: Optional[Path] = None
        cleanup_mcp_json = False
        try:
            overlay_extra: Dict[str, Any] = {}
            if mcp_allowlist["servers"]:
                # Wait for scoped MCP servers to finish connecting before omp validates `--tools`
                # (omp defaults to a 250ms startup window, which is shorter than npx/HTTP MCP boot).
                overlay_extra["mcp"] = {"startupTimeoutMs": 0}
                attempt_env["OMP_MCP_STARTUP_TIMEOUT_MS"] = "0"
                lower_srvs = {s.lower() for s in mcp_allowlist["servers"]}
                if any(tok in srv for srv in lower_srvs for tok in ("playwright", "puppeteer", "browser", "chrome")):
                    # Prevent omp's built-in eval browser prelude from filtering out explicit browser MCPs.
                    overlay_extra["browser"] = {"enabled": False}
                mcp_json_path, cleanup_mcp_json, _ = materialize_worktree_mcp_json(
                    work_dir,
                    mcp_allowlist["servers"],
                )
            overlay_path = write_ephemeral_omp_config_overlay(
                active_profile["compaction_overlay"],
                extra_sections=overlay_extra or None,
            )
            cmd: List[str] = [
                omp_bin,
                "-p",
                "--mode",
                "json",
                "--model",
                qualified_model,
                "--cwd",
                str(work_dir),
                "--config",
                str(overlay_path),
                "--no-judge",
                "--no-spotlight",
            ]
            addendum_parts: List[str] = []
            base_addendum = active_profile.get("append_system_prompt")
            if isinstance(base_addendum, str) and base_addendum.strip():
                addendum_parts.append(base_addendum.strip())
            if pinned_completion_addendum:
                addendum_parts.append(pinned_completion_addendum)
            if addendum_parts:
                cmd.extend(["--append-system-prompt", "\n\n".join(addendum_parts)])
            eff_resume = resume_session_override if resume_session_override is not None else resume_session
            eff_continue = bool(continue_session_override or continue_session)
            if no_session:
                cmd.extend(["--no-session", "--no-title"])
            if session_dir is not None and str(session_dir).strip():
                s_dir = Path(session_dir).expanduser().resolve()
                s_dir.mkdir(parents=True, exist_ok=True)
                cmd.extend(["--session-dir", str(s_dir)])
            if eff_resume:
                cmd.extend(["--resume", str(eff_resume)])
            elif eff_continue:
                cmd.append("--continue")
            if eff_tools_list is not None:
                cmd.extend(["--tools", ",".join(eff_tools_list)])
                if "lsp" not in eff_tools_list:
                    cmd.append("--no-lsp")
            if thinking:
                cmd.extend(["--thinking", str(thinking).strip()])
            if auto_approve:
                cmd.append("--auto-approve")
            if approval_mode:
                cmd.extend(["--approval-mode", str(approval_mode).strip()])

            proc = subprocess.Popen(
                cmd,
                cwd=str(work_dir),
                env=attempt_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            if proc.stdin is not None:
                def _feed_stdin() -> None:
                    try:
                        assert proc.stdin is not None
                        proc.stdin.write(eff_prompt)
                        proc.stdin.close()
                    except OSError:
                        pass

                threading.Thread(target=_feed_stdin, daemon=True).start()

            stderr_chunks: List[str] = []

            def _drain_stderr() -> None:
                if proc.stderr is None:
                    return
                try:
                    for s_line in proc.stderr:
                        stderr_chunks.append(s_line)
                except OSError:
                    pass

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            step_index = int(step_index_offset)
            omp_session_id: Optional[str] = None
            streamed_text_chunks: List[str] = []
            final_text = ""
            final_reasoning = ""
            tool_events: List[Dict[str, Any]] = []
            files_modified: List[str] = []
            stream_error_msg: Optional[str] = None
            non_json_stdout: List[str] = []

            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    non_json_stdout.append(line)
                    continue
                if not isinstance(ev, dict):
                    continue

                ev_type = ev.get("type")
                if ev_type == "session":
                    if ev.get("id"):
                        omp_session_id = str(ev["id"])
                elif ev_type == "message_update":
                    ame = ev.get("assistantMessageEvent") or {}
                    atype = ame.get("type")
                    delta = ame.get("delta")
                    if atype == "thinking_delta" and isinstance(delta, str) and delta:
                        final_reasoning += delta
                        if reasoning_callback is not None:
                            reasoning_callback(delta)
                        if event_callback is not None:
                            event_callback(
                                {
                                    "type": "reasoning_delta",
                                    "step": step_index,
                                    "delta": delta,
                                    "content": delta,
                                }
                            )
                        if emit_ndjson:
                            step_index += 1
                            _emit(
                                {
                                    "event": "step_update",
                                    "step_update": {
                                        "step_index": step_index,
                                        "state": "ACTIVE",
                                        "step_type": "agent_response",
                                        "thought": delta,
                                    },
                                }
                            )
                    elif atype == "text_delta" and isinstance(delta, str) and delta:
                        streamed_text_chunks.append(delta)
                        if stream_callback is not None:
                            stream_callback(delta)
                        if event_callback is not None:
                            event_callback(
                                {
                                    "type": "token_delta",
                                    "step": step_index,
                                    "delta": delta,
                                    "content": delta,
                                }
                            )
                        if emit_ndjson:
                            step_index += 1
                            _emit(
                                {
                                    "event": "step_update",
                                    "step_update": {
                                        "step_index": step_index,
                                        "state": "ACTIVE",
                                        "step_type": "agent_response",
                                        "text_delta": delta,
                                    },
                                }
                            )
                    elif atype == "error":
                        err_obj = ame.get("error") or ame.get("reason") or ame.get("message")
                        if isinstance(err_obj, Mapping):
                            err_msg_val = str(err_obj.get("errorMessage") or err_obj.get("message") or "").strip()
                        else:
                            err_msg_val = str(err_obj or "").strip()
                        if err_msg_val and not stream_error_msg:
                            stream_error_msg = err_msg_val
                elif ev_type == "error":
                    err_raw = ev.get("error") or ev.get("message")
                    if isinstance(err_raw, Mapping):
                        err_txt = str(err_raw.get("message") or err_raw.get("errorMessage") or "").strip()
                    else:
                        err_txt = str(err_raw or "").strip()
                    if err_txt and not stream_error_msg:
                        stream_error_msg = err_txt
                elif ev_type == "tool_execution_start":
                    t_name = str(ev.get("toolName") or "tool")
                    t_args = ev.get("args") or {}
                    tool_events.append({"tool": t_name, "arguments": t_args})
                    mod_path = _extract_modified_path(t_name, t_args)
                    if mod_path and mod_path not in files_modified:
                        files_modified.append(mod_path)
                    if event_callback is not None:
                        event_callback(
                            {
                                "type": "tool_call",
                                "step": step_index,
                                "name": t_name,
                                "arguments": t_args,
                            }
                        )
                    if emit_ndjson:
                        step_index += 1
                        _emit(
                            {
                                "event": "step_update",
                                "step_update": {
                                    "step_index": step_index,
                                    "state": "ACTIVE",
                                    "step_type": "tool",
                                    "tool_name": t_name,
                                    "tool_info": t_args,
                                },
                            }
                        )
                elif ev_type == "tool_execution_end":
                    t_name = str(ev.get("toolName") or "tool")
                    t_out = _format_tool_result_text(ev.get("result"))
                    is_err = bool(ev.get("isError", False))
                    if event_callback is not None:
                        event_callback(
                            {
                                "type": "tool_result",
                                "step": step_index,
                                "name": t_name,
                                "output": t_out,
                                "is_error": is_err,
                            }
                        )
                elif ev_type in ("message_end", "turn_end"):
                    msg = ev.get("message") or {}
                    if isinstance(msg, Mapping) and msg.get("role") == "assistant":
                        msg_err = _extract_assistant_error(msg)
                        if msg_err:
                            stream_error_msg = msg_err
                        joined_txt, joined_think = _extract_assistant_text_and_thinking(msg)
                        if joined_txt:
                            final_text = joined_txt
                        if joined_think and not final_reasoning:
                            final_reasoning = joined_think
                elif ev_type == "agent_end":
                    for msg in ev.get("messages") or []:
                        if isinstance(msg, Mapping) and msg.get("role") == "assistant":
                            msg_err = _extract_assistant_error(msg)
                            if msg_err and not stream_error_msg:
                                stream_error_msg = msg_err
                            joined_txt, joined_think = _extract_assistant_text_and_thinking(msg)
                            if joined_txt and not final_text:
                                final_text = joined_txt
                            if joined_think and not final_reasoning:
                                final_reasoning = joined_think

            rc = proc.wait()
            stderr_thread.join(timeout=2.0)
            stderr_txt = "".join(stderr_chunks).strip()

            if not final_text and streamed_text_chunks:
                final_text = "".join(streamed_text_chunks).strip()

            empty_without_tools = not final_text.strip() and not tool_events
            failed = (rc != 0) or bool(stream_error_msg) or empty_without_tools
            diag_parts = []
            if stream_error_msg:
                diag_parts.append(stream_error_msg)
            if stderr_txt:
                diag_parts.append(stderr_txt)
            if non_json_stdout:
                diag_parts.append("stdout: " + " | ".join(non_json_stdout[-5:]))
            if failed and not diag_parts:
                diag_parts.append(
                    f"omp exited with code {rc}" if rc != 0 else "omp completed with empty assistant response and no tool calls"
                )
            return {
                "failed": failed,
                "rc": rc,
                "error": " | ".join(diag_parts) if failed else None,
                "response": final_text,
                "reasoning": final_reasoning or None,
                "tool_events": tool_events,
                "files_modified": files_modified,
                "steps": step_index,
                "session_id": omp_session_id,
            }
        finally:
            _unlink_ephemeral_overlay(overlay_path)
            if cleanup_mcp_json and mcp_json_path is not None and mcp_json_path.is_file():
                try:
                    mcp_json_path.unlink()
                except OSError:
                    pass

    active_reg = model_reg
    active_profile = context_profile
    active_preflight = preflight_info
    recovery_attempted = False

    attempt_res = _execute_omp_attempt(active_preflight["effective_prompt"], active_reg, active_profile)

    # Strictly bounded Turn-0 recovery retry (max 1) when zero tool calls executed and server reports exact max_model_len
    if attempt_res["failed"] and not attempt_res["tool_events"] and enable_preflight_offload:
        overflow = parse_vllm_context_overflow_error(str(attempt_res["error"] or ""))
        if overflow is not None and overflow["max_model_len"] > 0:
            server_w = int(overflow["max_model_len"])
            reported_in = int(overflow["input_tokens"])
            updated_live = dict(eff_live_record or {})
            updated_live["max_model_len"] = server_w
            retry_reg, retry_profile = _build_reg_and_profile(ctx_override=server_w, live_rec=updated_live)
            if retry_profile.get("feasible", False):
                prev_prompt_tok = int(active_preflight.get("compacted_tokens") or estimate_prompt_tokens(active_preflight["effective_prompt"]))
                excess = max(512, reported_in - server_w + 512)
                tight_budget = max(
                    512,
                    min(int(retry_profile["max_turn0_prompt_tokens"]), max(512, prev_prompt_tok - excess)),
                )
                recovery_attempted = True
                rec_msg = (
                    f"[omp-context recovery] Turn-0 overflow detected (reported_input={reported_in} > "
                    f"server_max_model_len={server_w}); performing bounded retry (1/1) with tight budget={tight_budget} tok."
                )
                print(rec_msg, file=sys.stderr, flush=True)
                if event_callback is not None:
                    event_callback(
                        {
                            "type": "context_turn0_recovery",
                            "server_max_model_len": server_w,
                            "reported_input_tokens": reported_in,
                            "tight_budget_tokens": tight_budget,
                        }
                    )
                active_reg = retry_reg
                active_profile = retry_profile
                active_preflight = _run_preflight(active_profile, budget_override=tight_budget)
                attempt_res = _execute_omp_attempt(active_preflight["effective_prompt"], active_reg, active_profile)

    preflight_meta = {k: v for k, v in active_preflight.items() if k != "effective_prompt"}

    # Layer 2: Single bounded close-out recovery turn when a Sandcastle agent finishes work cleanly
    # but misses its required handoff manifest or `<promise>COMPLETE</promise>` (e.g. after compaction).
    completion_closeout_meta: Optional[Dict[str, Any]] = None
    if (
        not attempt_res["failed"]
        and completion_contract is not None
        and enable_completion_closeout
    ):
        eval_before = evaluate_sandcastle_completion(work_dir, attempt_res["response"], completion_contract)
        if not eval_before["satisfied"]:
            closeout_prompt = build_completion_closeout_prompt(completion_contract, eval_before)
            closeout_msg = (
                f"[omp-context closeout] Sandcastle agent '{completion_contract['agent_id']}' exited cleanly without "
                f"completing close-out contract (has_promise={eval_before['has_promise']}, "
                f"missing_handoffs={eval_before['missing_handoffs']}); executing single bounded close-out turn (1/1)."
            )
            print(closeout_msg, file=sys.stderr, flush=True)
            if event_callback is not None:
                event_callback(
                    {
                        "type": "context_completion_closeout",
                        "agent_id": completion_contract["agent_id"],
                        "has_promise_before": eval_before["has_promise"],
                        "missing_handoffs_before": list(eval_before["missing_handoffs"]),
                        "session_id": attempt_res.get("session_id"),
                    }
                )
            if emit_ndjson:
                _emit(
                    {
                        "event": "step_update",
                        "step_update": {
                            "step_index": int(attempt_res["steps"]) + 1,
                            "state": "ACTIVE",
                            "step_type": "context_completion_closeout",
                            "thought": closeout_msg,
                        },
                    }
                )
            use_resume = (not no_session) and bool(attempt_res.get("session_id"))
            use_continue = (not no_session) and not use_resume
            closeout_res = _execute_omp_attempt(
                closeout_prompt,
                active_reg,
                active_profile,
                resume_session_override=str(attempt_res["session_id"]) if use_resume else None,
                continue_session_override=use_continue,
                step_index_offset=int(attempt_res["steps"]) + 1,
            )
            # If session-ID resume failed immediately with zero tool calls (e.g. session not persisted),
            # fall back once to a stateless close-out turn in the same worktree without forging completion.
            if closeout_res["failed"] and use_resume and not closeout_res["tool_events"]:
                print(
                    f"[omp-context closeout] Session resume '{attempt_res.get('session_id')}' failed "
                    f"({closeout_res.get('error')}); retrying close-out turn statelessly in {work_dir}.",
                    file=sys.stderr,
                    flush=True,
                )
                closeout_res = _execute_omp_attempt(
                    closeout_prompt,
                    active_reg,
                    active_profile,
                    resume_session_override=None,
                    continue_session_override=False,
                    step_index_offset=int(attempt_res["steps"]) + 1,
                )

            merged_tools = list(attempt_res["tool_events"]) + list(closeout_res["tool_events"])
            merged_files = list(attempt_res["files_modified"])
            for fp in closeout_res["files_modified"]:
                if fp not in merged_files:
                    merged_files.append(fp)
            combined_response = (
                (attempt_res["response"].rstrip() + "\n\n" + closeout_res["response"].strip()).strip()
                if (attempt_res["response"].strip() and closeout_res["response"].strip())
                else (closeout_res["response"] or attempt_res["response"])
            )
            eval_after = evaluate_sandcastle_completion(work_dir, combined_response, completion_contract)
            completion_closeout_meta = {
                "attempted": True,
                "used_session_resume": use_resume,
                "has_promise_before": eval_before["has_promise"],
                "missing_handoffs_before": list(eval_before["missing_handoffs"]),
                "has_promise_after": eval_after["has_promise"],
                "missing_handoffs_after": list(eval_after["missing_handoffs"]),
                "satisfied_after": eval_after["satisfied"],
                "closeout_failed": bool(closeout_res["failed"]),
                "closeout_error": closeout_res.get("error"),
            }
            attempt_res = {
                **attempt_res,
                "failed": bool(closeout_res["failed"]),
                "rc": closeout_res["rc"] if closeout_res["failed"] else 0,
                "error": closeout_res["error"] if closeout_res["failed"] else None,
                "response": combined_response,
                "reasoning": closeout_res["reasoning"] or attempt_res["reasoning"],
                "tool_events": merged_tools,
                "files_modified": merged_files,
                "steps": max(int(attempt_res["steps"]), int(closeout_res["steps"])),
                "session_id": closeout_res["session_id"] or attempt_res["session_id"],
            }
        else:
            completion_closeout_meta = {
                "attempted": False,
                "has_promise_before": True,
                "missing_handoffs_before": [],
                "has_promise_after": True,
                "missing_handoffs_after": [],
                "satisfied_after": True,
                "closeout_failed": False,
                "closeout_error": None,
            }

    if attempt_res["failed"]:
        err_msg = str(attempt_res["error"] or "omp execution failed")
        parsed_ov = parse_vllm_context_overflow_error(err_msg)
        context_diag = {
            "server_max_model_len": parsed_ov["max_model_len"] if parsed_ov else active_reg["contextWindow"],
            "reported_input_tokens": parsed_ov["input_tokens"] if parsed_ov else None,
            "compaction_overlay": dict(active_profile["compaction_overlay"]),
            "session_id": attempt_res["session_id"],
            "dossier_path": active_preflight.get("dossier_path"),
            "recovery_attempted": recovery_attempted,
            "tool_turns_before_failure": len(attempt_res["tool_events"]),
        }
        if parsed_ov is not None:
            err_msg = (
                f"{err_msg} [context_diagnostics: server_max_model_len={context_diag['server_max_model_len']}, "
                f"reported_input_tokens={context_diag['reported_input_tokens']}, "
                f"compaction={json.dumps(context_diag['compaction_overlay'], separators=(',', ':'))}, "
                f"session_id={context_diag['session_id']}, recovery_attempted={recovery_attempted}]"
            )
        exit_code = int(attempt_res["rc"]) if int(attempt_res["rc"]) != 0 else 1
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err_msg}})
        return {
            "ok": False,
            "exit_code": exit_code,
            "error": err_msg,
            "response": attempt_res["response"],
            "reasoning": attempt_res["reasoning"],
            "tool_events": attempt_res["tool_events"],
            "files_modified": attempt_res["files_modified"],
            "steps": attempt_res["steps"],
            "model_id": clean_model_id,
            "served_model_id": active_reg["id"],
            "context_window": active_reg["contextWindow"],
            "max_tokens": active_reg["maxTokens"],
            "session_id": attempt_res["session_id"],
            "preflight": preflight_meta,
            "compaction_overlay": dict(active_profile["compaction_overlay"]),
            "context_diagnostics": context_diag,
            "recovery_attempted": recovery_attempted,
            "completion_contract": completion_contract,
            "completion_closeout": completion_closeout_meta,
        }

    if emit_ndjson:
        _emit({"event": "result", "result": {"response": attempt_res["response"]}})
    return {
        "ok": True,
        "exit_code": 0,
        "response": attempt_res["response"],
        "reasoning": attempt_res["reasoning"],
        "error": None,
        "tool_events": attempt_res["tool_events"],
        "files_modified": attempt_res["files_modified"],
        "steps": attempt_res["steps"],
        "model_id": clean_model_id,
        "served_model_id": active_reg["id"],
        "context_window": active_reg["contextWindow"],
        "max_tokens": active_reg["maxTokens"],
        "session_id": attempt_res["session_id"],
        "preflight": preflight_meta,
        "compaction_overlay": dict(active_profile["compaction_overlay"]),
        "recovery_attempted": recovery_attempted,
        "completion_contract": completion_contract,
        "completion_closeout": completion_closeout_meta,
    }


class _StrictOmpArgParser(argparse.ArgumentParser):
    """ArgumentParser that emits structured NDJSON errors when invoked in stream-json mode."""

    def __init__(self, *args: Any, emit_ndjson_on_error: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._emit_ndjson_on_error = emit_ndjson_on_error

    def error(self, message: str) -> None:
        if self._emit_ndjson_on_error:
            _emit({"event": "result", "result": {"error": f"[omp-adapter fatal] {message}"}})
        print(f"[omp-adapter fatal] {message}", file=sys.stderr, flush=True)
        raise SystemExit(2)


def _preprocess_prompt_flag_argv(argv: Sequence[str]) -> List[str]:
    """Converts `-p <val>` / `--prompt <val>` to `--prompt=<val>` before `argparse` so dash-prefixed values (e.g., `-x`, `--help`, `---`) are never misparsed as flags."""
    out: List[str] = []
    i = 0
    items = list(argv)
    while i < len(items):
        tok = items[i]
        if tok == "--":
            out.extend(items[i:])
            break
        if tok in ("-p", "--prompt") and i + 1 < len(items):
            out.append(f"--prompt={items[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def main(argv: Optional[List[str]] = None) -> int:
    def _signal_cleanup_handler(signum: int, _frame: Any) -> None:
        _cleanup_registered_overlays()
        raise SystemExit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _signal_cleanup_handler)
        except (ValueError, OSError):
            pass

    raw_argv = list(argv if argv is not None else sys.argv[1:])
    norm_argv = _preprocess_prompt_flag_argv(raw_argv)
    text_mode_requested = "--output-format=text" in norm_argv or (
        "--output-format" in norm_argv
        and norm_argv.index("--output-format") + 1 < len(norm_argv)
        and norm_argv[norm_argv.index("--output-format") + 1] == "text"
    )
    parser = _StrictOmpArgParser(
        description="Sandcastle native omp adapter",
        emit_ndjson_on_error=not text_mode_requested,
    )
    parser.add_argument("-p", "--prompt", default=None, help="Prompt text for omp")
    parser.add_argument("--prompt-file", default="", help="Path to UTF-8 file containing prompt text")
    parser.add_argument("--model", default="", help="Target model alias or ID")
    parser.add_argument("--endpoint", default="", help="Explicit OpenAI-compatible /v1 endpoint URL")
    parser.add_argument("--worktree", default="", help="Working directory override")
    parser.add_argument("--cwd", default="", help="Alias for --worktree")
    parser.add_argument("--output-format", default="stream-json", choices=["stream-json", "json", "text"])
    parser.add_argument("--max-turns", type=int, default=0, help="Compatibility argument")
    parser.add_argument("--tools", default="", help="Optional comma-separated omp tool override")
    parser.add_argument("--compact-tools", action="store_true", help="Explicitly enable lean coding tool subset for tight 16K runs")
    parser.add_argument("--mcp-servers", default="", help="Comma-separated scoped MCP servers to mount in <worktree>/.mcp.json")
    parser.add_argument("--mcp-pinned-tools", default="", help="Comma-separated MCP tool names to pin as top-level omp tools")
    parser.add_argument("--thinking", default="", help="Optional omp thinking level")
    parser.add_argument("--no-session", action="store_true", help="Run omp ephemerally without saving session")
    parser.add_argument("--session-dir", default="", help="Custom omp session directory")
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true", help="Continue omp session")
    parser.add_argument("-r", "--resume", default="", help="Resume omp session ID/path")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve omp tool calls")
    parser.add_argument("--approval-mode", default="", help="Override omp tools.approvalMode")
    parser.add_argument("--dangerously-skip-permissions", action="store_true", help="Maps to --auto-approve")
    parser.add_argument("--verify-timeout", type=float, default=DEFAULT_VERIFY_TIMEOUT_SEC, help="Endpoint probe timeout (<= 5s)")
    parser.add_argument("positional_prompt", nargs="*", help="Positional prompt tokens (optionally after --)")
    args = parser.parse_args(norm_argv)

    emit_ndjson = args.output_format in ("stream-json", "json")

    prompt_text = ""
    if args.prompt_file:
        p_file = Path(args.prompt_file).expanduser()
        if not p_file.is_file():
            err = f"--prompt-file does not exist: {p_file}"
            if emit_ndjson:
                _emit({"event": "result", "result": {"error": err}})
            print(err, file=sys.stderr, flush=True)
            return 2
        prompt_text = p_file.read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt_text = str(args.prompt)
    elif args.positional_prompt:
        prompt_text = " ".join(args.positional_prompt)
    elif not sys.stdin.isatty():
        prompt_text = sys.stdin.read()

    if not prompt_text.strip():
        err = "No prompt supplied via -p/--prompt, --prompt-file, positional arguments, or stdin"
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err}})
        print(err, file=sys.stderr, flush=True)
        return 2

    raw_cwd = args.worktree or args.cwd
    cwd = Path(raw_cwd).expanduser().resolve() if raw_cwd else Path.cwd().resolve()

    try:
        resolved = resolve_and_verify_omp_endpoint(
            cwd,
            explicit_endpoint=str(args.endpoint or "").strip(),
            explicit_model=str(args.model or "").strip(),
            verify_live=True,
            verify_timeout_sec=float(args.verify_timeout),
        )
    except Exception as exc:
        err = f"[omp-adapter fatal] {exc}"
        if emit_ndjson:
            _emit({"event": "result", "result": {"error": err}})
        print(err, file=sys.stderr, flush=True)
        return 1

    streamed_chars = [0]

    def _text_stream_cb(tok: str) -> None:
        if tok:
            streamed_chars[0] += len(tok)
            sys.stdout.write(tok)
            sys.stdout.flush()

    res = run_omp_turn(
        prompt=prompt_text,
        cwd=cwd,
        endpoint_url=resolved["endpoint_url"],
        api_key=resolved["api_key"],
        model_id=resolved["served_model_id"],
        emit_ndjson=emit_ndjson,
        stream_callback=_text_stream_cb if not emit_ndjson else None,
        no_session=bool(args.no_session),
        session_dir=args.session_dir or None,
        continue_session=bool(args.continue_session),
        resume_session=args.resume or None,
        tools=args.tools or None,
        compact_tools=bool(args.compact_tools),
        mcp_servers=args.mcp_servers or None,
        mcp_pinned_tools=args.mcp_pinned_tools or None,
        thinking=args.thinking or None,
        auto_approve=bool(args.auto_approve or args.dangerously_skip_permissions),
        approval_mode=args.approval_mode or None,
        verify_endpoint=False,
        gpu_tier=resolved.get("gpu_tier") or None,
        live_model_record=resolved.get("live_model_record"),
        registry_record=resolved.get("registry_record"),
    )
    if not emit_ndjson and res.get("ok"):
        if res.get("response"):
            if streamed_chars[0] == 0:
                sys.stdout.write(str(res["response"]))
            sys.stdout.write("\n")
            sys.stdout.flush()
    if not res.get("ok") and res.get("error"):
        print(res["error"], file=sys.stderr, flush=True)
    return int(res.get("exit_code", 0 if res.get("ok") else 1))


if __name__ == "__main__":
    raise SystemExit(main())
