"""
100% User Local-Disk Inference Prompt/Response Cache (`~/.platodoom/cache/responses/`).
Document ID: platodoom_response_cache_867_v1
Authoring Agent: architecture-implementer
Governing Standards: platodoom specification (§3.3, §4.3), architecture_and_dag_867_v1 (§3.2) & RULE[user_global]
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from platodoom.config import atomic_write_json, resolve_response_cache_dir
from platodoom.security import SECRET_ENDPOINT_KEYS, redact_env_secrets

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_BEARER_INLINE_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_EXTRA_SECRET_KEYS = frozenset(
    {
        *SECRET_ENDPOINT_KEYS,
        "password",
        "passwd",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "private_key",
    }
)


def _normalize_cache_key_token(cache_key: str) -> str:
    cleaned = str(cache_key or "").strip().lower()
    if not cleaned:
        raise ValueError("cache_key must be non-empty.")
    if _HEX64_RE.fullmatch(cleaned):
        return cleaned
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def redact_cache_payload(value: Any) -> Any:
    """Recursively scrubs bearer tokens, API keys, and .env secrets before local-disk cache persistence."""
    if isinstance(value, str):
        scrubbed = redact_env_secrets(value)
        scrubbed = _BEARER_INLINE_RE.sub("Bearer <REDACTED_SECRET>", scrubbed)
        return scrubbed
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k)
            if key_str.strip().lower() in _EXTRA_SECRET_KEYS:
                out[key_str] = "<REDACTED_SECRET>" if v is not None else None
            else:
                out[key_str] = redact_cache_payload(v)
        return out
    if isinstance(value, tuple):
        return [redact_cache_payload(item) for item in value]
    if isinstance(value, list):
        return [redact_cache_payload(item) for item in value]
    return copy.deepcopy(value)


def compute_response_cache_key(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    max_tokens: int = 2048,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop: Optional[Sequence[str]] = None,
    enable_thinking: Optional[bool] = None,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> str:
    """Computes a deterministic 64-character SHA-256 hex digest over canonical request parameters."""
    norm_model = str(model or "").strip().lower()
    if not norm_model:
        raise ValueError("model must be non-empty when computing response cache key.")

    norm_temp = round(float(temperature), 6) if temperature is not None else 0.0
    norm_top_p = round(float(top_p), 6) if top_p is not None else None
    norm_top_k = int(top_k) if top_k is not None else None
    norm_stop: Optional[List[str]] = [str(s) for s in stop] if stop else None
    norm_tools: Optional[List[Dict[str, Any]]] = [dict(t) for t in tools] if tools else None
    norm_messages: List[Dict[str, Any]] = []
    for m in messages or []:
        m_copy = dict(m)
        if m_copy.get("role") == "system" and isinstance(m_copy.get("content"), str):
            m_copy["content"] = re.sub(
                r"Session slug:\s*`[^`]+`",
                "Session slug: `<session>`",
                m_copy["content"],
            )
        norm_messages.append(m_copy)

    canonical_obj: Dict[str, Any] = {
        "schema_version": 1,
        "model": norm_model,
        "messages": norm_messages,
        "tools": norm_tools,
        "max_tokens": int(max_tokens),
        "temperature": norm_temp,
        "top_p": norm_top_p,
        "top_k": norm_top_k,
        "stop": norm_stop,
        "enable_thinking": bool(enable_thinking) if enable_thinking is not None else None,
        "chat_template_kwargs": dict(chat_template_kwargs) if chat_template_kwargs else None,
        "extra_params": dict(extra_params) if extra_params else None,
    }
    encoded = json.dumps(
        canonical_obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LocalResponseCache:
    """Crash-safe, secret-redacted local-disk prompt/response cache rooted at `~/.platodoom/cache/responses/`.

    Enforces `0o700` directory permissions and `0o600` atomic JSON entry files. Operates 100% on the user's
    local filesystem so cached completions survive Colab runtime disconnects or termination.
    """

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        *,
        enabled: bool = True,
        ttl_seconds: Optional[float] = 604800.0,
        cache_nondeterministic: bool = False,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.enabled: bool = bool(enabled)
        self.ttl_seconds: Optional[float] = float(ttl_seconds) if ttl_seconds is not None else None
        self.cache_nondeterministic: bool = bool(cache_nondeterministic)
        self._env: Optional[Dict[str, str]] = dict(env) if env is not None else None
        if cache_dir is not None:
            self.cache_dir: Path = Path(cache_dir).expanduser().resolve()
            if self.enabled:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    os.chmod(str(self.cache_dir), 0o700)
                except OSError:
                    pass
        else:
            if self.enabled:
                self.cache_dir = resolve_response_cache_dir(env=self._env)
            else:
                from platodoom.config import resolve_platodoom_home

                self.cache_dir = resolve_platodoom_home(env=self._env) / "cache" / "responses"

        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0
        self.writes: int = 0
        self.corrupt_evictions: int = 0

    def should_cache(self, *, temperature: Optional[float] = None) -> bool:
        if not self.enabled:
            return False
        if temperature is None or self.cache_nondeterministic:
            return True
        return float(temperature) <= 0.0

    def entry_path(self, cache_key: str) -> Path:
        norm_key = _normalize_cache_key_token(cache_key)
        return self.cache_dir / norm_key[:2] / f"{norm_key}.json"

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        norm_key = _normalize_cache_key_token(cache_key)
        target = self.cache_dir / norm_key[:2] / f"{norm_key}.json"
        if not target.is_file():
            with self._lock:
                self.misses += 1
            return None

        try:
            raw_text = target.read_text(encoding="utf-8")
            record = json.loads(raw_text)
        except Exception:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            with self._lock:
                self.corrupt_evictions += 1
                self.misses += 1
            return None

        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or record.get("cache_key") != norm_key
            or not isinstance(record.get("response"), dict)
            or not record["response"].get("choices")
        ):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            with self._lock:
                self.corrupt_evictions += 1
                self.misses += 1
            return None

        if self.ttl_seconds is not None:
            try:
                created_epoch = float(record.get("created_at_epoch", 0.0))
            except (TypeError, ValueError):
                created_epoch = 0.0
            if created_epoch <= 0.0 or (time.time() - created_epoch) > self.ttl_seconds:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
                with self._lock:
                    self.misses += 1
                return None

        with self._lock:
            self.hits += 1

        response_copy: Dict[str, Any] = copy.deepcopy(record["response"])
        response_copy["platodoom_cache_meta"] = {
            "cache_hit": True,
            "cache_key": norm_key,
            "cached_at": record.get("created_at"),
            "cache_path": str(target),
        }
        return response_copy

    def put(
        self,
        cache_key: str,
        response: Mapping[str, Any],
        *,
        request_fingerprint_meta: Optional[Mapping[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> Optional[Path]:
        if not self.enabled:
            return None
        if temperature is not None and not self.should_cache(temperature=temperature):
            return None
        if not isinstance(response, Mapping) or response.get("error") or not response.get("choices"):
            return None

        norm_key = _normalize_cache_key_token(cache_key)
        shard_dir = self.cache_dir / norm_key[:2]
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(str(self.cache_dir), 0o700)
            shard_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(str(shard_dir), 0o700)
        except OSError:
            pass

        clean_resp = redact_cache_payload(dict(response))
        if isinstance(clean_resp, dict):
            clean_resp.pop("platodoom_cache_meta", None)

        now_epoch = time.time()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "schema_version": 1,
            "cache_key": norm_key,
            "created_at": now_iso,
            "created_at_epoch": now_epoch,
            "request_meta": redact_cache_payload(dict(request_fingerprint_meta))
            if request_fingerprint_meta
            else {},
            "response": clean_resp,
        }
        target = shard_dir / f"{norm_key}.json"
        written = atomic_write_json(target, record, mode=0o600)
        try:
            os.chmod(str(shard_dir), 0o700)
        except OSError:
            pass
        with self._lock:
            self.writes += 1
        return written

    def invalidate(self, cache_key: str) -> bool:
        target = self.entry_path(cache_key)
        if target.exists():
            try:
                target.unlink()
                return True
            except OSError:
                return False
        return False

    def clear(self) -> int:
        removed = 0
        if not self.cache_dir.exists():
            return 0
        for json_file in self.cache_dir.rglob("*.json"):
            try:
                json_file.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "cache_dir": str(self.cache_dir),
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
                "corrupt_evictions": self.corrupt_evictions,
                "cache_hits": self.hits,
                "cache_misses": self.misses,
                "cache_entries_written": self.writes,
            }


__all__ = [
    "LocalResponseCache",
    "compute_response_cache_key",
    "redact_cache_payload",
]
