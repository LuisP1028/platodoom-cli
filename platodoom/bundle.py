"""
Self-Bootstrapping Per-Directory Hermetic Runtime Bundler for platodoom.

Materializes `<workspace_dir>/.platodoom/` containing strictly the runtime files,
configuration, sessions, standby registry, and response cache required to run the
`platodoom` launcher, CLI, Colab provisioner, and interactive chat from any directory
as a standalone runtime bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from platodoom.config import DEFAULT_USER_CONFIG_SCHEMA, atomic_write_json

GLOBAL_TEMPLATE_RUNTIME_DIRS: Tuple[Path, ...] = (
    Path("/opt/homebrew/share/platodoom/runtime"),
    Path("/usr/local/share/platodoom/runtime"),
)

RUNTIME_BUNDLE_MANIFEST: Tuple[str, ...] = (
    "bin/platodoom.mjs",
    "requirements.txt",
    "scripts/postinstall.mjs",
    "platodoom/__init__.py",
    "platodoom/bundle.py",
    "platodoom/cache_model.py",
    "platodoom/catalog.py",
    "platodoom/cli.py",
    "platodoom/config.py",
    "platodoom/discovery.py",
    "platodoom/mcp_bridge.py",
    "platodoom/onboarding.py",
    "platodoom/repl.py",
    "platodoom/response_cache.py",
    "platodoom/security.py",
    "platodoom/session.py",
    "platodoom/standby.py",
    "platodoom/swarm.py",
    "platodoom/tools.py",
    "platodoom/assets/platodoom_utf8_blocks_truecolor.ansi",
    "curation/__init__.py",
    "curation/api_client.py",
    "curation/colab_server.py",
    "notebooks/serve_sandcastle_vllm.ipynb",
    "scripts/__init__.py",
    "scripts/omp_qwen_adapter.py",
    "scripts/overnight_pipeline_watchdog.py",
    "scripts/provision_colab_serving.py",
    "scripts/qwen_worktree_agent.py",
)

_SYNTHETIC_PACKAGE_INITS: Dict[str, str] = {
    "curation/__init__.py": '"""Minimal runtime curation package for platodoom."""\n',
    "scripts/__init__.py": '"""Minimal runtime scripts package for platodoom."""\n',
}

_EXECUTABLE_REL_PATHS: Tuple[str, ...] = (
    "bin/platodoom.mjs",
    "platodoom/cli.py",
)


def is_valid_runtime_source_root(root: Path | str) -> bool:
    """Returns True iff `root` contains the core platodoom runtime entrypoints."""
    p = Path(root).expanduser().resolve()
    required = (
        "bin/platodoom.mjs",
        "platodoom/__init__.py",
        "platodoom/cli.py",
        "platodoom/catalog.py",
        "curation/api_client.py",
        "curation/colab_server.py",
        "scripts/provision_colab_serving.py",
    )
    return all((p / rel).is_file() for rel in required)


def resolve_runtime_source_root(explicit_source_root: Optional[Path | str] = None) -> Path:
    """Deterministically locates the authoritative source root for the runtime bundle."""
    candidates: List[Path] = []
    if explicit_source_root is not None and str(explicit_source_root).strip():
        candidates.append(Path(str(explicit_source_root).strip()).expanduser().resolve())
    env_src = os.environ.get("PLATODOOM_BUNDLE_SOURCE_ROOT") or os.environ.get("PLATODOOM_RUNTIME_TEMPLATE")
    if env_src and env_src.strip():
        candidates.append(Path(env_src.strip()).expanduser().resolve())

    # Current package root (could be a dev checkout, global share template, or an active .platodoom/runtime)
    here_root = Path(__file__).resolve().parent.parent
    candidates.append(here_root)
    candidates.extend(GLOBAL_TEMPLATE_RUNTIME_DIRS)

    for cand in candidates:
        if is_valid_runtime_source_root(cand):
            return cand

    raise RuntimeError(
        "Could not locate a valid platodoom runtime bundle source root. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def compute_bundle_digest(source_root: Optional[Path | str] = None) -> str:
    """Computes a deterministic SHA-256 digest over all files in `RUNTIME_BUNDLE_MANIFEST`."""
    src = resolve_runtime_source_root(source_root)
    hasher = hashlib.sha256()
    for rel_path in RUNTIME_BUNDLE_MANIFEST:
        fp = src / rel_path
        if fp.is_file():
            data = fp.read_bytes()
        elif rel_path in _SYNTHETIC_PACKAGE_INITS:
            data = _SYNTHETIC_PACKAGE_INITS[rel_path].encode("utf-8")
        else:
            raise FileNotFoundError(
                f"Required platodoom runtime file '{rel_path}' is missing from source root '{src}'."
            )
        hasher.update(f"{rel_path}:{len(data)}\0".encode("utf-8"))
        hasher.update(data)
    return hasher.hexdigest()[:16]


def _copy_manifest_to_runtime_dir(source_root: Path, dest_runtime_dir: Path, *, force: bool = False) -> Dict[str, Any]:
    """Copies strictly `RUNTIME_BUNDLE_MANIFEST` from `source_root` into `dest_runtime_dir`."""
    src = source_root.resolve()
    dst = dest_runtime_dir.resolve()
    digest = compute_bundle_digest(src)
    version_file = dst / "VERSION"

    dst_digest: Optional[str] = None
    if not force and src != dst and version_file.is_file() and version_file.read_text(encoding="utf-8").strip() == digest:
        try:
            dst_digest = compute_bundle_digest(dst)
        except Exception:
            dst_digest = None

    if (
        not force
        and src != dst
        and dst_digest == digest
        and all((dst / rel).is_file() or rel in _SYNTHETIC_PACKAGE_INITS for rel in RUNTIME_BUNDLE_MANIFEST)
    ):
        return {
            "synced": False,
            "version": digest,
            "runtime_dir": str(dst),
            "files_copied": 0,
        }

    dst.mkdir(parents=True, exist_ok=True)
    files_copied = 0
    if src != dst:
        for rel_path in RUNTIME_BUNDLE_MANIFEST:
            src_fp = src / rel_path
            dst_fp = dst / rel_path
            dst_fp.parent.mkdir(parents=True, exist_ok=True)
            if src_fp.is_file():
                shutil.copy2(src_fp, dst_fp)
                files_copied += 1
            elif rel_path in _SYNTHETIC_PACKAGE_INITS:
                dst_fp.write_text(_SYNTHETIC_PACKAGE_INITS[rel_path], encoding="utf-8")
                files_copied += 1
            else:
                raise FileNotFoundError(
                    f"Required platodoom runtime file '{rel_path}' is missing from '{src}'."
                )

    for exec_rel in _EXECUTABLE_REL_PATHS:
        exec_fp = dst / exec_rel
        if exec_fp.is_file():
            try:
                cur_mode = stat.S_IMODE(exec_fp.stat().st_mode)
                os.chmod(exec_fp, cur_mode | 0o111)
            except OSError:
                pass

    version_file.write_text(f"{digest}\n", encoding="utf-8")
    return {
        "synced": True,
        "version": digest,
        "runtime_dir": str(dst),
        "files_copied": files_copied,
    }


def _ensure_git_exclude_for_platodoom(workspace_dir: Path) -> bool:
    """Adds `.platodoom/` to `<workspace_dir>/.git/info/exclude` if inside a git repository so git status stays clean."""
    git_dir = workspace_dir / ".git"
    if not git_dir.is_dir():
        return False
    exclude_file = git_dir / "info" / "exclude"
    try:
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_file.read_text(encoding="utf-8") if exclude_file.is_file() else ""
        lines = {line.strip() for line in existing.splitlines()}
        if ".platodoom/" in lines or ".platodoom" in lines:
            return False
        prefix = "" if (not existing or existing.endswith("\n")) else "\n"
        exclude_file.write_text(f"{existing}{prefix}.platodoom/\n", encoding="utf-8")
        return True
    except OSError:
        return False


def bootstrap_directory_platodoom_runtime(
    workspace_dir: Optional[Path | str] = None,
    *,
    source_root: Optional[Path | str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Bootstraps a self-contained `<workspace_dir>/.platodoom/` directory containing:
      - `<workspace_dir>/.platodoom/runtime/` (strictly the 26 runtime files in `RUNTIME_BUNDLE_MANIFEST`)
      - `<workspace_dir>/.platodoom/config.json` (0o600 permissions, default_model="qwen3.8-27b")
      - `<workspace_dir>/.platodoom/sessions/` (0o700 permissions)
      - `<workspace_dir>/.platodoom/cache/responses/` (0o700 permissions)
      - `<workspace_dir>/.platodoom/standby/` (0o700 permissions)
    """
    target_ws = Path(workspace_dir).expanduser().resolve() if workspace_dir is not None else Path.cwd().resolve()
    if target_ws.is_file():
        target_ws = target_ws.parent
    # If invoked from inside `<ws>/.platodoom` or `<ws>/.platodoom/runtime`, anchor at `<ws>`
    if target_ws.name == "runtime" and target_ws.parent.name == ".platodoom":
        target_ws = target_ws.parent.parent.resolve()
    elif target_ws.name == ".platodoom":
        target_ws = target_ws.parent.resolve()

    plato_dir = target_ws / ".platodoom"
    runtime_dir = plato_dir / "runtime"
    sessions_dir = plato_dir / "sessions"
    cache_dir = plato_dir / "cache"
    responses_dir = cache_dir / "responses"
    standby_dir = plato_dir / "standby"
    config_path = plato_dir / "config.json"

    for d_path in (plato_dir, sessions_dir, cache_dir, responses_dir, standby_dir):
        d_path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(d_path), 0o700)
        except OSError:
            pass

    src_root = resolve_runtime_source_root(source_root)
    sync_meta = _copy_manifest_to_runtime_dir(src_root, runtime_dir, force=force)

    config_created = False
    if not config_path.is_file():
        atomic_write_json(config_path, dict(DEFAULT_USER_CONFIG_SCHEMA), mode=0o600)
        config_created = True

    git_excluded = _ensure_git_exclude_for_platodoom(target_ws)

    return {
        "ok": True,
        "workspace_dir": str(target_ws),
        "platodoom_dir": str(plato_dir),
        "runtime_dir": str(runtime_dir),
        "config_path": str(config_path),
        "config_created": config_created,
        "sessions_dir": str(sessions_dir),
        "responses_cache_dir": str(responses_dir),
        "standby_dir": str(standby_dir),
        "version": sync_meta["version"],
        "synced": sync_meta["synced"],
        "files_copied": sync_meta["files_copied"],
        "manifest_count": len(RUNTIME_BUNDLE_MANIFEST),
        "git_excluded": git_excluded,
    }


def sync_global_platodoom_template(
    *,
    source_root: Optional[Path | str] = None,
    target_template_dir: Optional[Path | str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Installs/updates the golden minimal runtime template under `/opt/homebrew/share/platodoom/runtime`."""
    src_root = resolve_runtime_source_root(source_root)
    dst_dir = (
        Path(target_template_dir).expanduser().resolve()
        if target_template_dir is not None
        else GLOBAL_TEMPLATE_RUNTIME_DIRS[0]
    )
    res = _copy_manifest_to_runtime_dir(src_root, dst_dir, force=force)
    return {
        "ok": True,
        "source_root": str(src_root),
        "template_dir": str(dst_dir),
        **res,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap or sync self-contained platodoom runtime bundle")
    parser.add_argument("--workspace-dir", default="", help="Target directory in which to initialize .platodoom/")
    parser.add_argument("--source-root", default="", help="Source root containing platodoom runtime files")
    parser.add_argument("--install-global", action="store_true", help="Sync golden runtime template to /opt/homebrew/share/platodoom/runtime")
    parser.add_argument("--global-dir", default="", help="Override global runtime template directory")
    parser.add_argument("--force", action="store_true", help="Force re-copy of all manifest files")
    args = parser.parse_args(argv)

    src_arg: Optional[str] = args.source_root.strip() or None
    if args.install_global:
        g_dir: Optional[str] = args.global_dir.strip() or None
        out = sync_global_platodoom_template(
            source_root=src_arg,
            target_template_dir=g_dir,
            force=bool(args.force),
        )
        print(json.dumps(out, indent=2))
        return 0

    ws_arg: Optional[str] = args.workspace_dir.strip() or None
    out = bootstrap_directory_platodoom_runtime(
        workspace_dir=ws_arg,
        source_root=src_arg,
        force=bool(args.force),
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
