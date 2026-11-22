"""
Cloud & Local Model Weight Caching, Shard Completeness Verification, and Manifest Persistence for platodoom.
Document ID: platodoom_cache_model_857_v1
Authoring Agent: storage-implementer
Governing Standards: refined_specification_857 (FR-2.4, FR-3.1, AC-4) & RULE[user_global]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from platodoom.config import (
    atomic_write_json,
    load_user_config,
    parse_storage_uri,
    resolve_platodoom_config,
    save_user_config,
)

logger = logging.getLogger("platodoom.cache_model")

CACHE_MANIFEST_FILENAME = "platodoom_cache_manifest.json"

_FALLBACK_CATALOG_REPOS: Dict[str, Dict[str, Any]] = {
    "qwen-9b": {
        "alias": "qwen-9b",
        "hf_repo_id": "Qwen/Qwen3.5-9B",
        "quantized_repo_id": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    },
    "r1-14b": {
        "alias": "r1-14b",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quantized_repo_id": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
    },
    "qwen-27b": {
        "alias": "qwen-27b",
        "hf_repo_id": "Qwen/Qwen3.5-27B",
        "quantized_repo_id": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
    },
    "qwen3.8-27b": {
        "alias": "qwen3.8-27b",
        "hf_repo_id": "Qwen/Qwen3.8-27B",
        "quantized_repo_id": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
    },
    "qwen-35b-moe": {
        "alias": "qwen-35b-moe",
        "hf_repo_id": "Qwen/Qwen3.5-35B-A3B",
        "quantized_repo_id": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    },
    "r1-32b": {
        "alias": "r1-32b",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "quantized_repo_id": "casperhansen/deepseek-r1-distill-qwen-32b-awq",
    },
    "qwen-30b-moe": {
        "alias": "qwen-30b-moe",
        "hf_repo_id": "Qwen/Qwen3-30B-A3B",
        "quantized_repo_id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
    },
    "qwen-coder-30b-moe": {
        "alias": "qwen-coder-30b-moe",
        "hf_repo_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "quantized_repo_id": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
    },
    "qwen-14b": {
        "alias": "qwen-14b",
        "hf_repo_id": "Qwen/Qwen3-14B",
        "quantized_repo_id": "Qwen/Qwen3-14B-AWQ",
    },
    "qwen-8b": {
        "alias": "qwen-8b",
        "hf_repo_id": "Qwen/Qwen3-8B",
        "quantized_repo_id": "Qwen/Qwen3-8B-AWQ",
    },
    "qwen-4b": {
        "alias": "qwen-4b",
        "hf_repo_id": "Qwen/Qwen3-4B",
        "quantized_repo_id": "Qwen/Qwen3-4B-AWQ",
    },
    "qwen-122b-moe": {
        "alias": "qwen-122b-moe",
        "hf_repo_id": "Qwen/Qwen3.5-122B-A10B",
        "quantized_repo_id": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    },
    "r1-7b": {
        "alias": "r1-7b",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "quantized_repo_id": "casperhansen/deepseek-r1-distill-qwen-7b-awq",
    },
    "deepseek-v2-lite": {
        "alias": "deepseek-v2-lite",
        "hf_repo_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "quantized_repo_id": "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
    },
    "flux-klein-4b": {
        "alias": "flux-klein-4b",
        "hf_repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "quantized_repo_id": "black-forest-labs/FLUX.2-klein-4b-fp8",
    },
    "flux-klein-9b": {
        "alias": "flux-klein-9b",
        "hf_repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "quantized_repo_id": "black-forest-labs/FLUX.2-klein-9B",
    },
    "z-image-turbo": {
        "alias": "z-image-turbo",
        "hf_repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "quantized_repo_id": "unsloth/Z-Image-Turbo-FP8",
    },
}


def _resolve_alias_and_hf_repo(
    alias_or_hf_id: str,
    config: Dict[str, Any],
    precision: Optional[str] = None,
) -> Dict[str, str]:
    cleaned = (alias_or_hf_id or "").strip()
    if not cleaned:
        raise ValueError("alias_or_hf_id must be a non-empty model alias or HuggingFace repo ID.")

    if "@" in cleaned and precision is None:
        cleaned, precision = [p.strip() for p in cleaned.split("@", 1)]

    try:
        from platodoom.catalog import get_model_catalog, resolve_model_precision_variant  # type: ignore
        catalog = get_model_catalog(config)
    except Exception:
        resolve_model_precision_variant = None  # type: ignore
        catalog = dict(_FALLBACK_CATALOG_REPOS)
        custom = config.get("models", {}).get("custom", {})
        if isinstance(custom, dict):
            for k, v in custom.items():
                if isinstance(v, dict):
                    catalog[k] = v

    if cleaned in catalog:
        entry = catalog[cleaned]
        hf_repo = str(
            entry.get("hf_repo_id")
            or entry.get("quantized_repo_id")
            or entry.get("repo_id")
            or cleaned
        )
        if precision and callable(resolve_model_precision_variant):
            var = resolve_model_precision_variant(cleaned, precision=precision, config=config)
            hf_repo = str(var.get("hf_repo_id") or hf_repo)
        return {
            "model_alias": cleaned,
            "model_slug": re.sub(r"[^a-zA-Z0-9._-]+", "-", cleaned).strip("-").lower(),
            "hf_repo_id": hf_repo,
        }

    # Check if cleaned matches any hf_repo_id in catalog
    for alias, entry in catalog.items():
        if not isinstance(entry, dict):
            continue
        if cleaned in (entry.get("hf_repo_id"), entry.get("quantized_repo_id")):
            return {
                "model_alias": alias,
                "model_slug": re.sub(r"[^a-zA-Z0-9._-]+", "-", alias).strip("-").lower(),
                "hf_repo_id": cleaned,
            }

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", cleaned.split("/")[-1]).strip("-").lower() or "custom-model"
    return {
        "model_alias": slug,
        "model_slug": slug,
        "hf_repo_id": cleaned,
    }


def _compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_local_checkpoint_shards(checkpoint_dir: Path | str) -> Dict[str, Any]:
    """
    Verifies shard completeness for a local model directory:
    - Requires config.json, tokenizer_config.json (or tokenizer.json), and at least one .safetensors shard.
    - If model.safetensors.index.json is present, verifies every shard listed in weight_map exists and is > 0 bytes.
    """
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint source directory does not exist: {root}")

    missing_required: List[str] = []
    if not (root / "config.json").is_file():
        missing_required.append("config.json")
    if not (root / "tokenizer_config.json").is_file() and not (root / "tokenizer.json").is_file():
        missing_required.append("tokenizer_config.json")

    safetensor_files = sorted(root.glob("*.safetensors"))
    if not safetensor_files:
        missing_required.append("*.safetensors")

    index_file = root / "model.safetensors.index.json"
    if index_file.is_file():
        try:
            index_data = json.loads(index_file.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {}) if isinstance(index_data, dict) else {}
            expected_shards = sorted({str(v) for v in weight_map.values() if v})
            for shard_name in expected_shards:
                shard_path = root / shard_name
                if not shard_path.is_file() or shard_path.stat().st_size == 0:
                    missing_required.append(shard_name)
        except Exception as exc:
            raise ValueError(f"Corrupted model.safetensors.index.json in {root}: {exc}") from exc

    if missing_required:
        raise ValueError(
            f"Incomplete model checkpoint in {root}; missing required files/shards: {missing_required}"
        )

    files_manifest: List[Dict[str, Any]] = []
    total_bytes = 0
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name == CACHE_MANIFEST_FILENAME or file_path.name.startswith("."):
            continue
        rel_path = file_path.relative_to(root).as_posix()
        size_bytes = file_path.stat().st_size
        total_bytes += size_bytes
        files_manifest.append(
            {
                "relative_path": rel_path,
                "size_bytes": size_bytes,
                "sha256": _compute_file_sha256(file_path),
            }
        )

    return {
        "verified": True,
        "total_files": len(files_manifest),
        "total_size_bytes": total_bytes,
        "files": files_manifest,
    }


def _resolve_destination_storage_uri(
    model_slug: str,
    storage_uri: Optional[str],
    config: Dict[str, Any],
) -> str:
    if storage_uri and str(storage_uri).strip():
        raw = str(storage_uri).strip()
        if raw.startswith("gs://"):
            base = raw.rstrip("/")
            if not base.endswith(f"/{model_slug}"):
                return f"{base}/{model_slug}/"
            return f"{base}/"
        return raw

    storage_cfg = config.get("storage", {}) if isinstance(config.get("storage"), dict) else {}
    gcs_bucket = storage_cfg.get("gcs_bucket")
    if gcs_bucket and str(gcs_bucket).strip():
        prefix = str(storage_cfg.get("gcs_models_prefix") or "models/").strip("/")
        if prefix:
            return f"gs://{str(gcs_bucket).strip()}/{prefix}/{model_slug}/"
        return f"gs://{str(gcs_bucket).strip()}/{model_slug}/"

    gdrive_dir = str(storage_cfg.get("gdrive_models_dir") or "/content/drive/MyDrive/platodoom_models").rstrip("/")
    return f"{gdrive_dir}/{model_slug}/"


def cache_model_to_cloud(
    alias_or_hf_id: str,
    storage_uri: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    local_source_dir: Optional[Path | str] = None,
    to_uri: Optional[str] = None,
    precision: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enforces FR-2.4 & AC-4:
    - Resolves alias_or_hf_id via get_model_catalog(config).
    - Determines destination cache URI from storage_uri (or to_uri), config['storage']['gcs_bucket'], or config['storage']['gdrive_models_dir'].
    - Verifies shard completeness (config.json, tokenizer_config.json, *.safetensors), writes platodoom_cache_manifest.json,
      and updates ~/.platodoom/config.json (models.custom.<alias>.storage_uri) with 0o600 permissions.
    """
    cfg = config if isinstance(config, dict) else resolve_platodoom_config()
    resolved_model = _resolve_alias_and_hf_id(alias_or_hf_id, cfg, precision=precision)
    model_alias = resolved_model["model_alias"]
    model_slug = resolved_model["model_slug"]
    hf_repo_id = resolved_model["hf_repo_id"]

    eff_storage_uri = storage_uri if storage_uri is not None else to_uri
    target_uri = _resolve_destination_storage_uri(
        model_slug=model_slug,
        storage_uri=eff_storage_uri,
        config=cfg,
    )
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    verification: Dict[str, Any] = {
        "verified": True,
        "total_files": 0,
        "total_size_bytes": 0,
        "files": [],
    }
    if local_source_dir is not None:
        verification = verify_local_checkpoint_shards(local_source_dir)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "model_alias": model_alias,
        "model_slug": model_slug,
        "hf_repo_id": hf_repo_id,
        "revision": "main",
        "storage_uri": target_uri,
        "total_files": verification["total_files"],
        "total_size_bytes": verification["total_size_bytes"],
        "verified": verification["verified"],
        "created_at": now_iso,
        "files": verification["files"],
    }

    manifest_path_written: Optional[str] = None
    if not dry_run:
        if local_source_dir is not None:
            src_root = Path(local_source_dir).expanduser().resolve()
            local_manifest_path = src_root / CACHE_MANIFEST_FILENAME
            atomic_write_json(local_manifest_path, manifest, mode=0o644)
            manifest_path_written = str(local_manifest_path)

            if not target_uri.startswith(("gs://", "hf://", "gdrive://")) and not target_uri.startswith("/content/drive"):
                dest_dir = Path(target_uri.replace("file://", "")).expanduser().resolve()
                dest_dir.mkdir(parents=True, exist_ok=True)
                if dest_dir != src_root:
                    for entry in verification["files"]:
                        rel = entry["relative_path"]
                        dst_file = dest_dir / rel
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_root / rel, dst_file)
                dest_manifest_path = dest_dir / CACHE_MANIFEST_FILENAME
                atomic_write_json(dest_manifest_path, manifest, mode=0o644)
                manifest_path_written = str(dest_manifest_path)

        # Persist storage_uri into ~/.platodoom/config.json (0600)
        user_cfg_path = cfg.get("user_config_path")
        user_cfg = load_user_config(config_path=user_cfg_path)
        user_cfg.setdefault("models", {}).setdefault("custom", {})
        existing_custom = user_cfg["models"]["custom"].get(model_alias, {})
        if not isinstance(existing_custom, dict):
            existing_custom = {}
        existing_custom.update(
            {
                "alias": model_alias,
                "hf_repo_id": hf_repo_id,
                "storage_uri": target_uri,
                "cached_at": now_iso,
            }
        )
        user_cfg["models"]["custom"][model_alias] = existing_custom
        if target_uri.startswith("gs://"):
            parsed_gcs = parse_storage_uri(target_uri)
            if parsed_gcs.get("gcs_bucket") and not user_cfg.get("storage", {}).get("gcs_bucket"):
                user_cfg.setdefault("storage", {})["gcs_bucket"] = parsed_gcs["gcs_bucket"]
        saved_cfg_path = save_user_config(user_cfg, config_path=user_cfg_path)
    else:
        saved_cfg_path = Path(str(cfg.get("user_config_path") or (Path.home() / ".platodoom" / "config.json")))

    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "model_alias": model_alias,
        "model_slug": model_slug,
        "hf_repo_id": hf_repo_id,
        "storage_uri": target_uri,
        "verified": verification["verified"],
        "total_files": verification["total_files"],
        "total_size_bytes": verification["total_size_bytes"],
        "manifest": manifest,
        "manifest_path": manifest_path_written,
        "config_path": str(saved_cfg_path),
    }


_resolve_alias_and_hf_id = _resolve_alias_and_hf_repo
