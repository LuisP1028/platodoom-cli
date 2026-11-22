"""
5-Level Hierarchical Configuration Store, Session Slug Partitioning, and Mode-Aware Atomic Persistence for platodoom.
Document ID: platodoom_config_857_v1
Authoring Agent: storage-implementer
Governing Standards: refined_specification_857 (FR-1.2, FR-1.3, FR-2.1.2, FR-5.4, AC-1, AC-3, AC-4) & RULE[user_global]
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger("platodoom.config")

DEFAULT_USER_CONFIG_SCHEMA: Dict[str, Any] = {
    "version": 1,
    "default_model": "qwen3.8-27b",
    "default_gpu": "auto",
    "default_stack": None,
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
    "response_cache": {
        "enabled": True,
        "ttl_seconds": 604800,
        "cache_nondeterministic": False,
    },
    "auth": {
        "hf_token_env": "PLATODOOM_HF_TOKEN",
        "hf_token": None,
        "ngrok_authtoken": None,
        "tunnel_provider": "cloudflare",
        "google_token_path": "~/.mcp-colab/token.json",
        "google_credentials_path": "~/.mcp-colab/credentials.json",
    },
    "onboarding": {
        "completed": False,
        "skipped": False,
    },
    "models": {
        "custom": {},
    },
    "security": {
        "extra_denylist_globs": [],
        "redact_env_secrets": True,
    },
}

DEFAULT_PROJECT_CONFIG_TEMPLATE: Dict[str, Any] = {
    "model": "qwen3.8-27b",
    "gpu": "auto",
    "stack": None,
    "access": "workspace-write",
    "workspace_roots": ["."],
    "skill_paths": [".platodoom/skills", ".agents/skills"],
    "mcp_config_paths": [".platodoom/mcp.json"],
}

_FORBIDDEN_PROJECT_SECRET_KEYS = {
    "auth",
    "bearer_token",
    "auth_token",
    "api_key",
    "tunnel_token",
    "hf_token",
    "ngrok_authtoken",
    "ngrok_token",
    "secret",
    "password",
    "credentials",
    "google_token_path",
    "google_credentials_path",
}


def atomic_write_json(
    target_path: Path | str,
    data: Any,
    mode: Optional[int] = None,
) -> Path:
    """
    Write JSON payload atomically via same-directory temp file, fchmod, fsync, and os.replace.
    Automatically enforces 0o600 permissions for config.json, session.json, messages.json,
    and token-bearing endpoint registries (FR-4.1, FR-5.4).
    """
    resolved_target = Path(target_path).expanduser().resolve()
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    if (
        resolved_target.parent.name in (".platodoom", "sessions", "cache", "responses")
        or resolved_target.parent.parent.name in ("sessions", "responses", "cache")
    ):
        try:
            os.chmod(str(resolved_target.parent), 0o700)
        except OSError:
            pass
    serialized = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        sort_keys=isinstance(data, dict),
    ) + "\n"
    if mode is not None:
        effective_mode = int(mode)
    elif (
        resolved_target.name
        in (
            "config.json",
            "endpoint.json",
            "colab_endpoint.json",
            "colab_qwen_endpoint.json",
            "session.json",
            "messages.json",
            "transcript.jsonl",
            "audit.jsonl",
        )
        or (
            isinstance(data, dict)
            and (bool(data.get("bearer_token")) or bool(data.get("auth_token")))
        )
    ):
        effective_mode = 0o600
    else:
        effective_mode = 0o644

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{resolved_target.name}.",
        suffix=f".{os.getpid()}.{threading.get_ident()}.tmp",
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
    return resolved_target


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def resolve_platodoom_home(env: Optional[Dict[str, str]] = None) -> Path:
    """Resolves the user-global platodoom home directory (~/.platodoom or PLATODOOM_CONFIG_DIR / PLATODOOM_HOME)."""
    eff_env = env if env is not None else os.environ
    custom_dir = eff_env.get("PLATODOOM_CONFIG_DIR") or eff_env.get("PLATODOOM_HOME")
    if custom_dir and str(custom_dir).strip():
        return Path(str(custom_dir).strip()).expanduser().resolve()
    return (Path.home() / ".platodoom").resolve()


def resolve_user_config_path(env: Optional[Dict[str, str]] = None) -> Path:
    """Resolves ~/.platodoom/config.json dynamically without hardcoded user paths."""
    return resolve_platodoom_home(env=env) / "config.json"


def resolve_sessions_root_dir(env: Optional[Dict[str, str]] = None) -> Path:
    """Resolves ~/.platodoom/sessions/ directory and enforces 0o700 permissions when created."""
    home_dir = resolve_platodoom_home(env=env)
    sessions_dir = home_dir / "sessions"
    try:
        home_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(home_dir), 0o700)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(sessions_dir), 0o700)
    except OSError:
        pass
    return sessions_dir


def resolve_session_state_dir(
    session_or_dir_slug: str,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    """Resolves ~/.platodoom/sessions/<session_or_dir_slug>/ for isolated session or swarm state."""
    cleaned_slug = str(session_or_dir_slug or "").strip()
    if not cleaned_slug:
        raise ValueError("session_or_dir_slug must be non-empty.")
    return resolve_platodoom_home(env=env) / "sessions" / cleaned_slug


def resolve_response_cache_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    """Resolves ~/.platodoom/cache/responses/ on user local disk and enforces 0o700 permissions (Spec 867)."""
    eff_env = dict(env) if env is not None else None
    home_dir = resolve_platodoom_home(env=eff_env)
    cache_root = home_dir / "cache"
    responses_dir = cache_root / "responses"
    try:
        home_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(home_dir), 0o700)
        cache_root.mkdir(parents=True, exist_ok=True)
        os.chmod(str(cache_root), 0o700)
        responses_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(responses_dir), 0o700)
    except OSError:
        pass
    return responses_dir


def create_interactive_session_slug(session_id: Optional[str] = None) -> str:
    """Returns f'platodoom-chat-{session_id or secrets.token_hex(4)}' (FR-2.1.2)."""
    if session_id is not None and str(session_id).strip():
        cleaned = str(session_id).strip()
        if cleaned.startswith("platodoom-chat-"):
            return cleaned
        return f"platodoom-chat-{cleaned}"
    return f"platodoom-chat-{secrets.token_hex(4)}"


def parse_storage_uri(storage_uri: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Parses a storage URI ('gs://<bucket>[/<prefix>]', 'gdrive://...', '/content/drive/...', 'hf://...', or bare bucket name)
    into normalized storage configuration fields.
    """
    if not storage_uri or not str(storage_uri).strip():
        return {}
    raw = str(storage_uri).strip()
    if raw.startswith("gs://"):
        without_scheme = raw[len("gs://") :].strip("/")
        if not without_scheme:
            return {"backend": "gcs", "cache_backend": "gcs", "gcs_bucket": None}
        parts = without_scheme.split("/", 1)
        bucket = parts[0].strip()
        prefix = (parts[1].strip("/") + "/") if len(parts) > 1 and parts[1].strip("/") else "models/"
        return {
            "backend": "gcs",
            "cache_backend": "gcs",
            "gcs_bucket": bucket,
            "gcs_models_prefix": prefix,
            "storage_uri": raw,
        }
    if raw.startswith("gdrive://"):
        gdrive_path = raw[len("gdrive://") :].strip()
        if not gdrive_path.startswith("/"):
            gdrive_path = f"/content/drive/MyDrive/{gdrive_path}"
        return {
            "backend": "gdrive",
            "cache_backend": "gdrive",
            "gdrive_models_dir": gdrive_path,
            "storage_uri": raw,
        }
    if raw.startswith("/content/drive/") or "MyDrive" in raw:
        return {
            "backend": "gdrive",
            "cache_backend": "gdrive",
            "gdrive_models_dir": raw,
            "storage_uri": raw,
        }
    if raw.startswith("hf://") or raw == "hf_hub":
        repo_id = raw[len("hf://") :].strip("/") if raw.startswith("hf://") else None
        return {
            "backend": "hf_hub",
            "cache_backend": "hf_hub",
            "hf_repo_id": repo_id,
            "storage_uri": raw,
        }
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", raw):
        return {
            "backend": "gcs",
            "cache_backend": "gcs",
            "gcs_bucket": raw,
            "gcs_models_prefix": "models/",
            "storage_uri": f"gs://{raw}/models/",
        }
    return {"storage_uri": raw}


def load_user_config(
    config_path: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Loads ~/.platodoom/config.json merged over DEFAULT_USER_CONFIG_SCHEMA."""
    target = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else resolve_user_config_path(env=env)
    )
    base = copy.deepcopy(DEFAULT_USER_CONFIG_SCHEMA)
    if not target.is_file():
        return base
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return _deep_merge_dicts(base, loaded)
    except Exception as exc:
        logger.warning("Failed to parse user config at %s: %s", target, exc)
    return base


def save_user_config(
    config_data: Dict[str, Any],
    config_path: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    """Writes ~/.platodoom/config.json atomically with 0o600 permissions."""
    target = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else resolve_user_config_path(env=env)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(target.parent), 0o700)
    except OSError:
        pass
    merged = _deep_merge_dicts(DEFAULT_USER_CONFIG_SCHEMA, config_data or {})
    atomic_write_json(target, merged, mode=0o600)
    return target


def _coerce_config_value(raw_value: Any) -> Any:
    if not isinstance(raw_value, str):
        return raw_value
    stripped = raw_value.strip()
    lower = stripped.lower()
    if lower in ("null", "none"):
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(stripped)
        except Exception:
            pass
    if re.fullmatch(r"-?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            pass
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        try:
            return float(stripped)
        except ValueError:
            pass
    return raw_value


_CONFIG_KEY_ALIASES: Dict[str, str] = {
    "model": "default_model",
    "gpu": "default_gpu",
    "stack": "default_stack",
    "stack_preset": "default_stack",
    "access": "default_access",
    "backend": "default_backend",
    "gcs_bucket": "storage.gcs_bucket",
    "gcs_project_id": "storage.gcs_project_id",
    "gcs_project": "storage.gcs_project_id",
    "cache_backend": "storage.cache_backend",
    "max_concurrent_sessions": "colab.max_concurrent_sessions",
}


def set_user_config_value(
    key_path: str,
    value: Any,
    config_path: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Updates dotted key paths in ~/.platodoom/config.json atomically with 0o600 permissions."""
    cleaned_key = str(key_path or "").strip()
    if not cleaned_key:
        raise ValueError("key_path must be a non-empty string.")

    current = load_user_config(config_path=config_path, env=env)
    coerced = _coerce_config_value(value)

    if cleaned_key in ("storage_uri", "storage.storage_uri") and isinstance(coerced, str):
        parsed_uri = parse_storage_uri(coerced)
        current.setdefault("storage", {}).update(parsed_uri)
        save_user_config(current, config_path=config_path, env=env)
        return current

    canonical_key = _CONFIG_KEY_ALIASES.get(cleaned_key, cleaned_key)
    parts = [p.strip() for p in canonical_key.split(".") if p.strip()]
    node: Dict[str, Any] = current
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = coerced

    save_user_config(current, config_path=config_path, env=env)
    return current


def get_user_config_value(
    key_path: Optional[str] = None,
    config_path: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Any:
    """Reads full user config or a specific dotted key path from ~/.platodoom/config.json."""
    cfg = load_user_config(config_path=config_path, env=env)
    if key_path is None or not str(key_path).strip():
        return cfg
    cleaned_key = str(key_path).strip()
    canonical_key = _CONFIG_KEY_ALIASES.get(cleaned_key, cleaned_key)
    parts = [p.strip() for p in canonical_key.split(".") if p.strip()]
    curr: Any = cfg
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return None
    return curr


def _find_project_config_file(cwd: Optional[Path | str] = None) -> Optional[Path]:
    start_dir = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    if start_dir.is_file():
        start_dir = start_dir.parent

    curr = start_dir
    while True:
        cand = curr / ".platodoom.json"
        if cand.is_file():
            return cand
        if (curr / ".git").exists() or curr.parent == curr:
            break
        curr = curr.parent
    return None


def load_project_config(cwd: Optional[Path | str] = None) -> Dict[str, Any]:
    """Loads <workspace_root>/.platodoom.json while stripping any forbidden secret/credential keys."""
    cfg_file = _find_project_config_file(cwd=cwd)
    if cfg_file is None:
        return {}
    try:
        raw = json.loads(cfg_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        sanitized: Dict[str, Any] = {}
        for k, v in raw.items():
            if k.lower() in _FORBIDDEN_PROJECT_SECRET_KEYS:
                logger.warning("Ignoring secret key '%s' in project config %s", k, cfg_file)
                continue
            sanitized[k] = v
        sanitized["_project_config_path"] = str(cfg_file)
        return sanitized
    except Exception as exc:
        logger.warning("Failed to parse project config at %s: %s", cfg_file, exc)
        return {}


def init_project_config(
    workspace_dir: Path | str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """Writes git-safe <workspace_root>/.platodoom.json with 0o644 permissions, stripping any secret/token keys."""
    root = Path(workspace_dir).expanduser().resolve()
    if root.is_file():
        root = root.parent
    root.mkdir(parents=True, exist_ok=True)
    target = root / ".platodoom.json"

    payload = copy.deepcopy(DEFAULT_PROJECT_CONFIG_TEMPLATE)
    if overrides and isinstance(overrides, dict):
        for k, v in overrides.items():
            if k.lower() in _FORBIDDEN_PROJECT_SECRET_KEYS or k.startswith("_"):
                continue
            payload[k] = v

    atomic_write_json(target, payload, mode=0o644)
    return target


def resolve_platodoom_config(
    cli_overrides: Optional[Dict[str, Any]] = None,
    cwd: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Resolves 5-level priority hierarchy (FR-1.3):
    1. cli_overrides
    2. env vars (PLATODOOM_MODEL, PLATODOOM_GPU, PLATODOOM_STORAGE_URI, PLATODOOM_GCS_BUCKET,
       PLATODOOM_GCS_PROJECT, PLATODOOM_BACKEND, PLATODOOM_ACCESS, PLATODOOM_HF_TOKEN / HF_TOKEN,
       PLATODOOM_ENDPOINT_URL / SANDCASTLE_VLLM_URL, PLATODOOM_TUNNEL_TOKEN / SANDCASTLE_VLLM_API_KEY)
    3. <workspace_root>/.platodoom.json
    4. ~/.platodoom/config.json
    5. DEFAULT_USER_CONFIG_SCHEMA (open-source HF Hub defaults, gcs_bucket=None).
    Also parses storage_uri='gs://<bucket>/<prefix>' into storage.gcs_bucket and storage.gcs_models_prefix.
    """
    eff_env = env if env is not None else os.environ

    # Level 5 & Level 4: Built-in defaults + ~/.platodoom/config.json
    resolved = load_user_config(env=eff_env)

    # Level 3: Project configuration file (<workspace_root>/.platodoom.json)
    proj_cfg = load_project_config(cwd=cwd)
    if proj_cfg:
        if proj_cfg.get("model"):
            resolved["default_model"] = str(proj_cfg["model"])
        if proj_cfg.get("default_model"):
            resolved["default_model"] = str(proj_cfg["default_model"])
        if proj_cfg.get("gpu"):
            resolved["default_gpu"] = str(proj_cfg["gpu"])
        if proj_cfg.get("default_gpu"):
            resolved["default_gpu"] = str(proj_cfg["default_gpu"])
        if proj_cfg.get("stack") is not None:
            resolved["default_stack"] = str(proj_cfg["stack"]).strip() or None
        if proj_cfg.get("default_stack") is not None:
            resolved["default_stack"] = str(proj_cfg["default_stack"]).strip() or None
        if proj_cfg.get("access"):
            resolved["default_access"] = str(proj_cfg["access"])
        if proj_cfg.get("default_access"):
            resolved["default_access"] = str(proj_cfg["default_access"])
        if proj_cfg.get("backend"):
            resolved["default_backend"] = str(proj_cfg["backend"])
        if isinstance(proj_cfg.get("workspace_roots"), list):
            resolved["workspace_roots"] = list(proj_cfg["workspace_roots"])
        if isinstance(proj_cfg.get("skill_paths"), list):
            resolved["skill_paths"] = list(proj_cfg["skill_paths"])
        if isinstance(proj_cfg.get("mcp_config_paths"), list):
            resolved["mcp_config_paths"] = list(proj_cfg["mcp_config_paths"])
        if isinstance(proj_cfg.get("storage"), dict):
            resolved["storage"] = _deep_merge_dicts(resolved.get("storage", {}), proj_cfg["storage"])
        if isinstance(proj_cfg.get("models"), dict):
            resolved["models"] = _deep_merge_dicts(resolved.get("models", {}), proj_cfg["models"])
        if proj_cfg.get("_project_config_path"):
            resolved["project_config_path"] = proj_cfg["_project_config_path"]

    # Level 2: Environment variables
    env_model = eff_env.get("PLATODOOM_MODEL")
    if env_model and str(env_model).strip():
        resolved["default_model"] = str(env_model).strip()

    env_gpu = eff_env.get("PLATODOOM_GPU")
    if env_gpu and str(env_gpu).strip():
        resolved["default_gpu"] = str(env_gpu).strip()

    env_stack = eff_env.get("PLATODOOM_STACK")
    if env_stack and str(env_stack).strip():
        resolved["default_stack"] = str(env_stack).strip()

    env_access = eff_env.get("PLATODOOM_ACCESS")
    if env_access and str(env_access).strip():
        acc_val = str(env_access).strip().lower()
        resolved["default_access"] = "full-access" if acc_val in ("yolo", "--yolo") else acc_val

    env_backend = eff_env.get("PLATODOOM_BACKEND")
    if env_backend and str(env_backend).strip():
        resolved["default_backend"] = str(env_backend).strip()

    env_storage_uri = eff_env.get("PLATODOOM_STORAGE_URI")
    if env_storage_uri and str(env_storage_uri).strip():
        resolved["storage"].update(parse_storage_uri(env_storage_uri))

    env_gcs_bucket = eff_env.get("PLATODOOM_GCS_BUCKET") or eff_env.get("GCS_BUCKET_NAME")
    if env_gcs_bucket and str(env_gcs_bucket).strip():
        resolved["storage"]["gcs_bucket"] = str(env_gcs_bucket).strip()
        if resolved["storage"].get("cache_backend") == "hf_hub":
            resolved["storage"]["cache_backend"] = "gcs"

    env_gcs_project = eff_env.get("PLATODOOM_GCS_PROJECT") or eff_env.get("GCP_PROJECT_ID")
    if env_gcs_project and str(env_gcs_project).strip():
        resolved["storage"]["gcs_project_id"] = str(env_gcs_project).strip()

    env_max_sessions = eff_env.get("PLATODOOM_MAX_CONCURRENT_SESSIONS")
    if env_max_sessions and str(env_max_sessions).strip().isdigit():
        resolved["colab"]["max_concurrent_sessions"] = int(str(env_max_sessions).strip())

    hf_token_env_name = str(resolved.get("auth", {}).get("hf_token_env") or "PLATODOOM_HF_TOKEN")
    saved_hf_token = str(resolved.get("auth", {}).get("hf_token") or "").strip() or None
    hf_token = (
        eff_env.get(hf_token_env_name)
        or eff_env.get("PLATODOOM_HF_TOKEN")
        or eff_env.get("HF_TOKEN")
        or eff_env.get("HUGGING_FACE_HUB_TOKEN")
        or saved_hf_token
    )
    resolved["auth"]["hf_token"] = str(hf_token).strip() if hf_token and str(hf_token).strip() else None

    saved_ngrok_token = str(resolved.get("auth", {}).get("ngrok_authtoken") or "").strip() or None
    ngrok_token = eff_env.get("NGROK_AUTHTOKEN") or saved_ngrok_token
    resolved["auth"]["ngrok_authtoken"] = str(ngrok_token).strip() if ngrok_token and str(ngrok_token).strip() else None

    endpoint_url = (
        eff_env.get("PLATODOOM_ENDPOINT_URL")
        or eff_env.get("SANDCASTLE_VLLM_URL")
        or eff_env.get("VLLM_API_BASE")
    )
    if endpoint_url and str(endpoint_url).strip():
        resolved["endpoint_url"] = str(endpoint_url).strip()

    tunnel_token = (
        eff_env.get("PLATODOOM_TUNNEL_TOKEN")
        or eff_env.get("SANDCASTLE_VLLM_API_KEY")
        or eff_env.get("VLLM_API_KEY")
    )
    if tunnel_token and str(tunnel_token).strip():
        resolved["bearer_token"] = str(tunnel_token).strip()

    # Level 1: Explicit CLI / programmatic overrides
    if cli_overrides and isinstance(cli_overrides, dict):
        for k, v in cli_overrides.items():
            if v is None:
                continue
            if k in ("model", "default_model"):
                resolved["default_model"] = str(v).strip()
            elif k in ("gpu", "default_gpu", "gpu_tier"):
                resolved["default_gpu"] = str(v).strip()
            elif k in ("stack", "default_stack", "stack_preset"):
                resolved["default_stack"] = str(v).strip() or None
            elif k in ("access", "access_mode", "default_access"):
                acc = str(v).strip().lower()
                resolved["default_access"] = "full-access" if acc in ("yolo", "--yolo") else acc
            elif k == "yolo" and bool(v):
                resolved["default_access"] = "full-access"
            elif k in ("backend", "default_backend"):
                resolved["default_backend"] = str(v).strip()
            elif k == "storage_uri":
                resolved["storage"].update(parse_storage_uri(str(v)))
            elif k == "gcs_bucket":
                resolved["storage"]["gcs_bucket"] = str(v).strip()
                resolved["storage"]["cache_backend"] = "gcs"
            elif k in ("gcs_project_id", "gcs_project"):
                resolved["storage"]["gcs_project_id"] = str(v).strip()
            elif k in ("endpoint", "endpoint_url"):
                resolved["endpoint_url"] = str(v).strip()
            elif k in ("bearer_token", "tunnel_token", "api_key"):
                resolved["bearer_token"] = str(v).strip()
            elif isinstance(v, dict) and isinstance(resolved.get(k), dict):
                resolved[k] = _deep_merge_dicts(resolved[k], v)
            else:
                resolved[k] = v

    # Normalize hardware tier across all 8 Colab GPU/TPU tiers (or "auto")
    try:
        from platodoom.catalog import normalize_hardware_tier

        resolved["default_gpu"] = normalize_hardware_tier(resolved.get("default_gpu"), allow_auto=True)
    except ImportError:
        pass

    # Convenience top-level aliases
    resolved["model"] = resolved["default_model"]
    resolved["gpu"] = resolved["default_gpu"]
    resolved["stack"] = resolved.get("default_stack")
    resolved["access"] = resolved["default_access"]
    resolved["backend"] = resolved["default_backend"]
    resolved["user_config_path"] = str(resolve_user_config_path(env=eff_env))
    resolved["platodoom_home"] = str(resolve_platodoom_home(env=eff_env))
    return resolved
