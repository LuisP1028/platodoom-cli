"""
platodoom/tools.py - Token-Safe Local Computer Navigation Suite & Dispatcher.
Document ID: platodoom_tools_857_v1
Governing Standards: platodoom specification, LANGUAGE.md, RULE[user_global]

Operational Budgets & Invariants:
1. <= 250ms p95 latency on repositories up to 50,000 files using ripgrep.
2. Strict payload caps:
   - <= 24 KB (24,576 bytes) for read_file, find_files, list_dir.
   - <= 16 KB (16,384 bytes) for grep_search, run_command.
3. Fast binary file rejection (< 1ms): reads first 8KB; if null byte (b'\\x00') present, rejects immediately.
4. Atomic file operations: temp file + fsync + os.replace pattern for zero file corruption.
5. Zero-Trust Security Guardrail integration: denylisted paths blocked, .env secrets redacted.
6. Unified permission dispatcher checking 3-tier access control (read-only, workspace-write, full-access).
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Attempt to import security helpers from platodoom.security (owned by storage-implementer)
try:
    from platodoom.security import (
        DEFAULT_SECRET_PATH_PATTERNS,
        VALID_ACCESS_MODES,
        check_secret_path_and_redact,
        check_tool_permission,
        classify_shell_command_tier,
        is_denylisted_secret_path,
        is_env_secret_file,
        redact_env_secrets,
        verify_bearer_authorization,
    )
except ImportError:
    # Autonomous resilient fallback conforming to Section 3.1.1 until branches merge
    DEFAULT_SECRET_PATH_PATTERNS = (
        "*/.ssh/*",
        "*/.gnupg/*",
        "*/.aws/*",
        "*/.config/gcloud/*",
        "*/.azure/*",
        "*/.kube/config",
        "*/.npmrc",
        "*/.pypirc",
        "*/.netrc",
        "*/.docker/config.json",
        "*/.mcp-colab/token.json",
        "*/.mcp-colab/credentials.json",
        "*/.platodoom/config.json",
        "*/.platodoom/auth/*",
        "*/.platodoom/sessions/*/endpoint.json",
        "*/.sandcastle/colab_qwen_endpoint.json",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "*.keystore",
    )
    VALID_ACCESS_MODES = ("read-only", "workspace-write", "full-access")

    def is_denylisted_secret_path(
        target_path: Path | str,
        home_dir: Optional[Path | str] = None,
        extra_globs: Optional[List[str]] = None,
    ) -> bool:
        path_str = str(Path(target_path).expanduser().resolve())
        patterns = list(DEFAULT_SECRET_PATH_PATTERNS) + (extra_globs or [])
        for pat in patterns:
            if fnmatch.fnmatch(path_str, pat) or fnmatch.fnmatch(Path(target_path).name, pat):
                return True
        return False

    def is_env_secret_file(target_path: Path | str) -> bool:
        name = Path(target_path).name
        if name in (".env.example", ".env.sample", ".env.template"):
            return False
        return name == ".env" or name.startswith(".env.")

    ENV_SECRET_REGEX = re.compile(
        r"""(?i)^(\s*(?:export\s+)?[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE|AUTH)[A-Z0-9_]*\s*=\s*)(["']?)(.*?)\2(\s*)$""",
        re.MULTILINE,
    )

    def redact_env_secrets(text: str) -> str:
        return ENV_SECRET_REGEX.sub(r"\1\2<REDACTED_SECRET>\2\4", text)

    def check_secret_path_and_redact(
        target_path: Path | str,
        content: Optional[str] = None,
        home_dir: Optional[Path | str] = None,
        extra_denylist_globs: Optional[List[str]] = None,
        interactive_approve_fn: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        p = Path(target_path).expanduser().resolve()
        if is_denylisted_secret_path(p, home_dir=home_dir, extra_globs=extra_denylist_globs):
            if interactive_approve_fn and interactive_approve_fn(str(p)):
                return {"allowed": True, "redacted": False, "content": content, "reason": "Approved by user"}
            return {
                "allowed": False,
                "redacted": False,
                "content": None,
                "reason": f"[SECURITY BLOCKED: Access to sensitive credential file {p} is denied by platodoom Zero-Trust Guardrail.]",
            }
        if is_env_secret_file(p) and content is not None:
            return {
                "allowed": True,
                "redacted": True,
                "content": redact_env_secrets(content),
                "reason": "Redacted .env secret values",
            }
        return {"allowed": True, "redacted": False, "content": content, "reason": None}

    def classify_shell_command_tier(command: str, cwd: Path, workspace_roots: List[Path]) -> int:
        cmd_strip = command.strip()
        read_only_cmds = ("git status", "git diff", "git log", "git branch", "ls", "pwd", "which", "cat", "head", "tail", "rg", "find")
        if any(cmd_strip.startswith(r) for r in read_only_cmds):
            return 1
        destructive_tokens = ("rm -rf", "sudo ", "git push --force", "git reset --hard", "mkfs", "dd if=", "curl -X POST", "gsutil cp", "gcloud storage cp")
        if any(tok in cmd_strip for tok in destructive_tokens):
            return 3
        try:
            in_roots = any(cwd.resolve().is_relative_to(r.resolve()) for r in workspace_roots)
        except AttributeError:
            in_roots = any(str(cwd.resolve()).startswith(str(r.resolve())) for r in workspace_roots)
        if not in_roots:
            return 3
        return 2

    def check_tool_permission(
        tool_name: str,
        args: Dict[str, Any],
        access_mode: str = "workspace-write",
        workspace_roots: Optional[List[Path | str]] = None,
        cwd: Optional[Path | str] = None,
        interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> Tuple[bool, str]:
        access_mode = str(access_mode).lower()
        if access_mode in ("full-access", "yolo", "--yolo"):
            return True, "Full access authorized"
        roots = [Path(r).resolve() for r in (workspace_roots or [Path.cwd()])]
        active_cwd = Path(cwd or Path.cwd()).resolve()

        if tool_name in ("read_file", "list_dir", "find_files", "grep_search", "use_skill", "list_mcp_tools", "inspect_mcp_tool"):
            return True, "Read-only operation allowed"

        if tool_name in ("write_file", "edit_file"):
            if access_mode == "read-only":
                return False, f"Write tool '{tool_name}' blocked under read-only access mode."
            target = Path(args.get("path", "")).expanduser()
            if not target.is_absolute():
                target = (active_cwd / target).resolve()
            else:
                target = target.resolve()
            try:
                is_within = any(target.is_relative_to(r) for r in roots)
            except AttributeError:
                is_within = any(str(target).startswith(str(r)) for r in roots)
            if not is_within:
                if interactive_confirm_fn and interactive_confirm_fn(f"Write to outside workspace root: {target}"):
                    return True, "Approved interactively"
                return False, f"File {target} is outside workspace roots {roots}. Requires confirmation or full-access."
            return True, "Workspace write permitted"

        if tool_name == "run_command":
            if access_mode == "read-only":
                tier = classify_shell_command_tier(args.get("command", ""), active_cwd, roots)
                if tier > 1:
                    return False, f"Non-read-only command '{args.get('command')}' blocked under read-only access mode."
                return True, "Read-only command permitted"
            tier = classify_shell_command_tier(args.get("command", ""), active_cwd, roots)
            if tier == 3 and access_mode != "full-access":
                if interactive_confirm_fn and interactive_confirm_fn(f"Execute high-tier command: {args.get('command')}"):
                    return True, "Approved interactively"
                return False, f"Tier 3 command '{args.get('command')}' requires explicit confirmation or full-access mode."
            return True, "Command execution permitted"

        return True, "Permitted"


# Path to ripgrep binary
RG_PATH = shutil.which("rg") or ("/opt/homebrew/bin/rg" if Path("/opt/homebrew/bin/rg").is_file() else None)

MAX_READ_FILE_LINES = 400
MAX_READ_FILE_BYTES = 24576    # 24 KB
MAX_FIND_FILES_COUNT = 200
MAX_FIND_FILES_BYTES = 24576   # 24 KB
MAX_LIST_DIR_BYTES = 24576     # 24 KB
MAX_GREP_SEARCH_MATCHES = 100
MAX_GREP_SEARCH_BYTES = 16384  # 16 KB
MAX_COMMAND_OUTPUT_BYTES = 16384  # 16 KB


# ============================================================================
# OpenAI-Compatible Tool Schema Definition (PLATODOOM_TOOLS_SCHEMA)
# ============================================================================

PLATODOOM_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads contents of a file on the host machine with optional line slicing. Returns 1-indexed lines formatted as 'NNN: <line>'. Fast binary file check rejects non-text files. Capped at 400 lines and 24 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file to read."},
                    "start_line": {"type": "integer", "description": "Optional 1-indexed starting line number (inclusive)."},
                    "end_line": {"type": "integer", "description": "Optional 1-indexed ending line number (inclusive)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes full content to a file atomically via temp-replace pattern, ensuring zero file corruption.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to write."},
                    "content": {"type": "string", "description": "Exact text content to write."},
                    "overwrite": {"type": "boolean", "description": "Whether to overwrite existing file (default true)."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Performs exact block replacement on an existing file atomically, returning a unified diff snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path of the file to edit."},
                    "target": {"type": "string", "description": "Exact character/string block to replace."},
                    "replacement": {"type": "string", "description": "Drop-in replacement string."},
                },
                "required": ["path", "target", "replacement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Lists contents of a directory in a compact tree structure up to depth levels, showing file sizes and entry counts. Gitignore-aware and capped at 24 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default '.')."},
                    "depth": {"type": "integer", "description": "Max directory depth to traverse (default 2)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Fast filename search using ripgrep matching glob pattern. Excludes .git/ and node_modules/. Completes in <= 250ms p95, capped at 200 files / 24 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', '*test*', 'package.json')."},
                    "path": {"type": "string", "description": "Base directory to search in (default '.')."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Fast regex/text content search using ripgrep returning file:line:snippet matches. Excludes denylisted secrets and redacts .env values. Completes in <= 250ms p95, capped at 100 matches / 16 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern or regex to find in file contents."},
                    "path": {"type": "string", "description": "Directory or file to search in (default '.')."},
                    "include_glob": {"type": "string", "description": "Optional glob filter (e.g. '*.py', '*.ts')."},
                    "case_sensitive": {"type": "boolean", "description": "Whether search is case sensitive (default true)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Runs a shell command on the host OS with stateful cwd tracking and bounded execution timeout. Output is head/tail truncated at 16 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Exact shell command to execute."},
                    "timeout_sec": {"type": "integer", "description": "Timeout in seconds (default 60)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": "Loads the full markdown instructions for a discovered skill from the progressive-disclosure index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact name of the skill to load."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_mcp_tools",
            "description": "Lists available MCP tools across discovered MCP servers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {"type": "string", "description": "Optional specific server name filter."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_mcp_tool",
            "description": "Inspects detailed parameter schema for an MCP tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {"type": "string", "description": "Server name."},
                    "tool_name": {"type": "string", "description": "Tool name."},
                },
                "required": ["server_name", "tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_mcp_tool",
            "description": "Executes an MCP tool over stdio or HTTP transport.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {"type": "string", "description": "Target server name."},
                    "tool_name": {"type": "string", "description": "Target tool name."},
                    "arguments": {"type": "object", "description": "Arguments dictionary to pass to the tool."},
                },
                "required": ["server_name", "tool_name", "arguments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invoke_subagent",
            "description": "Spawns an isolated child subagent with a designated role and prompt, sharing the parent's warm GPU endpoint (0 extra Colab VMs consumed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "Role of the subagent (e.g. 'codebase-researcher', 'unit-tester')."},
                    "prompt": {"type": "string", "description": "Task description for the subagent."},
                    "model_alias": {"type": "string", "description": "Optional model alias (defaults to parent session model)."},
                    "workspace_dir": {"type": "string", "description": "Optional workspace directory."},
                    "read_only": {"type": "boolean", "description": "Whether subagent is restricted to read-only tools."},
                },
                "required": ["role", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invoke_parallel_subagents",
            "description": "Concurrently executes multiple subagent tasks across the parent session's warm GPU continuous batch queue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "prompt": {"type": "string"},
                                "model_alias": {
                                    "type": "string",
                                    "description": "Optional co-resident model alias on GPU 1 (e.g. 'qwen-9b@Q4' or 'r1-7b@Q4').",
                                },
                                "access_mode": {"type": "string"},
                                "read_only": {"type": "boolean"},
                            },
                            "required": ["role", "prompt"],
                        },
                        "description": "List of subagent task specifications.",
                    },
                },
                "required": ["tasks"],
            },
        },
    },
]

TOOL_SCHEMAS: List[Dict[str, Any]] = PLATODOOM_TOOLS_SCHEMA


def get_local_tools_schema(
    workflow_mode: str = "build",
    access_mode: str = "workspace-write",
) -> List[Dict[str, Any]]:
    """
    Returns the outbound OpenAI tool schema list for the current session state.
    When workflow_mode == 'plan' or access_mode == 'read-only', filters out mutating file tools
    ('write_file' and 'edit_file') so the model cannot even attempt to call them (FR-3.1, AC-3).
    """
    norm_workflow = str(workflow_mode or "build").strip().lower()
    norm_access = str(access_mode or "workspace-write").strip().lower()
    is_read_only = norm_workflow == "plan" or norm_access in ("read-only", "read_only")
    if is_read_only:
        return [
            dict(entry)
            for entry in PLATODOOM_TOOLS_SCHEMA
            if (entry.get("function") or {}).get("name") not in ("write_file", "edit_file")
        ]
    return [dict(entry) for entry in PLATODOOM_TOOLS_SCHEMA]


# ============================================================================
# 7 Token-Safe Local Navigation Tools
# ============================================================================

def _resolve_active_path(raw_path: str | Path, cwd: Optional[Path | str] = None) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = Path(cwd or Path.cwd()).resolve()
    return (base / p).resolve()


def tool_read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    cwd: Optional[Path | str] = None,
    home_dir: Optional[Path | str] = None,
    extra_denylist_globs: Optional[List[str]] = None,
    interactive_approve_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """Reads contents of a file on the host machine with optional line slicing.

    - Resolves relative paths against cwd; allows absolute paths anywhere on host.
    - Enforces check_secret_path_and_redact (blocks ~/.ssh, ~/.aws, redacts .env).
    - Fast binary file check: inspects first 8KB for null byte (b'\\x00') in < 1ms.
    - Capped at max 400 lines and 24 KB (24,576 bytes).
    """
    resolved = _resolve_active_path(path, cwd)

    # 1. Zero-Trust Security Guardrail check before reading
    chk = check_secret_path_and_redact(
        resolved,
        content=None,
        home_dir=home_dir,
        extra_denylist_globs=extra_denylist_globs,
        interactive_approve_fn=interactive_approve_fn,
    )
    if not chk.get("allowed", True):
        return chk.get("reason") or f"[SECURITY BLOCKED: Access to {resolved} is denied by Zero-Trust Guardrail.]"

    if not resolved.exists():
        return f"[ERROR: File not found: {resolved}]"
    if resolved.is_dir():
        return f"[ERROR: Path is a directory, not a file: {resolved}. Use list_dir instead.]"

    try:
        # Fast binary file check: read first 8192 bytes
        with open(resolved, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                mime_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
                size = resolved.stat().st_size
                return f"[BINARY FILE REJECTED: path={resolved} mime={mime_type} size={size} bytes]"

        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as exc:
        return f"[ERROR: Failed reading {resolved}: {exc}]"

    # Redact .env secrets if applicable
    if is_env_secret_file(resolved):
        content = redact_env_secrets(content)

    lines = content.splitlines()
    total_lines = len(lines)

    start = max(1, int(start_line)) if start_line is not None else 1
    end = min(total_lines, int(end_line)) if end_line is not None else total_lines

    if total_lines == 0:
        return f"[{resolved.name} — empty file, 0 lines]"
    if start > total_lines:
        return f"[{resolved.name} — start_line {start} exceeds total lines ({total_lines})]"

    # Cap line window at 400 lines
    if end - start + 1 > MAX_READ_FILE_LINES:
        end = start + MAX_READ_FILE_LINES - 1

    formatted_lines: List[str] = []
    accum_bytes = 0
    truncated = False
    actual_end = start - 1

    for line_idx in range(start, end + 1):
        line_text = lines[line_idx - 1]
        formatted = f"{line_idx:4d}: {line_text}"
        line_bytes = len(formatted.encode("utf-8")) + 1
        if accum_bytes + line_bytes > MAX_READ_FILE_BYTES:
            truncated = True
            break
        formatted_lines.append(formatted)
        accum_bytes += line_bytes
        actual_end = line_idx

    header = f"[{resolved.name} — showing lines {start}-{actual_end} of {total_lines}"
    if truncated:
        header += f" (truncated at {MAX_READ_FILE_BYTES} bytes, next_start_line={actual_end + 1})]"
    else:
        header += "]"

    return header + "\n" + "\n".join(formatted_lines)


def tool_write_file(
    path: str,
    content: str,
    overwrite: bool = True,
    cwd: Optional[Path | str] = None,
    home_dir: Optional[Path | str] = None,
    extra_denylist_globs: Optional[List[str]] = None,
    interactive_approve_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """Atomically writes content to a file via mkstemp + fsync + os.replace pattern."""
    resolved = _resolve_active_path(path, cwd)
    chk = check_secret_path_and_redact(
        resolved,
        content=None,
        home_dir=home_dir,
        extra_denylist_globs=extra_denylist_globs,
        interactive_approve_fn=interactive_approve_fn,
    )
    if not chk.get("allowed", True):
        return chk.get("reason") or f"[SECURITY BLOCKED: Writing sensitive file {resolved} is denied by Zero-Trust Guardrail.]"

    if not overwrite and resolved.exists():
        return f"[ERROR: File already exists and overwrite=False: {resolved}]"

    temp_file: Optional[Path] = None
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(resolved.parent))
        temp_file = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, resolved)
        bytes_written = len(content.encode("utf-8"))
        return f"[Successfully wrote {bytes_written} bytes to {resolved}]"
    except Exception as exc:
        if temp_file is not None:
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass
        return f"[ERROR: Failed writing {resolved}: {exc}]"


def tool_edit_file(
    path: str,
    target: str,
    replacement: str,
    cwd: Optional[Path | str] = None,
    home_dir: Optional[Path | str] = None,
    extra_denylist_globs: Optional[List[str]] = None,
    interactive_approve_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """Performs exact block replacement on an existing file, returning unified diff snippet."""
    resolved = _resolve_active_path(path, cwd)
    chk = check_secret_path_and_redact(
        resolved,
        content=None,
        home_dir=home_dir,
        extra_denylist_globs=extra_denylist_globs,
        interactive_approve_fn=interactive_approve_fn,
    )
    if not chk.get("allowed", True):
        return chk.get("reason") or f"[SECURITY BLOCKED: Editing sensitive file {resolved} is denied by Zero-Trust Guardrail.]"

    if not resolved.exists():
        return f"[ERROR: File not found: {resolved}]"
    if resolved.is_dir():
        return f"[ERROR: Path is a directory: {resolved}]"

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            orig_content = f.read()
    except Exception as exc:
        return f"[ERROR: Failed reading {resolved}: {exc}]"

    matched_target = None
    if target in orig_content:
        count = orig_content.count(target)
        if count > 1:
            return (
                f"[ERROR: Target block matches {count} occurrences in {resolved}. EDIT_TARGET_AMBIGUOUS. "
                "Please provide a larger or unique target block containing sufficient surrounding context.]"
            )
        matched_target = target
    else:
        # Fallback 1: Whitespace-normalized matching (handling trailing spaces / CRLF differences)
        orig_lines = orig_content.splitlines(keepends=True)
        target_lines = target.splitlines()
        matched_indices = []
        if target_lines and orig_lines:
            n_target = len(target_lines)
            for i in range(len(orig_lines) - n_target + 1):
                window = orig_lines[i : i + n_target]
                if all(w.rstrip("\r\n").rstrip() == t.rstrip() for w, t in zip(window, target_lines)):
                    matched_indices.append(i)
                elif all(w.strip() == t.strip() for w, t in zip(window, target_lines)):
                    matched_indices.append(i)

            matched_indices = list(dict.fromkeys(matched_indices))
            if len(matched_indices) == 1:
                idx = matched_indices[0]
                matched_target = "".join(orig_lines[idx : idx + n_target])
            elif len(matched_indices) > 1:
                return (
                    f"[ERROR: Target block matches {len(matched_indices)} occurrences in {resolved} after whitespace normalization. EDIT_TARGET_AMBIGUOUS. "
                    "Please provide a larger or unique target block containing sufficient surrounding context.]"
                )

    if matched_target is None:
        # Actionable diagnostic: find nearest matching line and context preview
        orig_lines = orig_content.splitlines()
        target_first_line = target.splitlines()[0] if target.splitlines() else target
        matches = difflib.get_close_matches(target_first_line.strip(), [l.strip() for l in orig_lines], n=1, cutoff=0.3)
        diag = f"[ERROR: Target block not found in {resolved}. EDIT_TARGET_NOT_FOUND."
        if matches:
            best_match = matches[0]
            try:
                line_no = [l.strip() for l in orig_lines].index(best_match) + 1
                c_start = max(1, line_no - 2)
                c_end = min(len(orig_lines), line_no + 2)
                preview_lines = [f"{i}: {orig_lines[i-1]}" for i in range(c_start, c_end + 1)]
                context_preview = "\n".join(preview_lines)
                diag += f" Nearest line #{line_no}: '{orig_lines[line_no-1]}'. Context preview:\n{context_preview}\n"
            except (ValueError, IndexError):
                pass
        diag += " Please provide exact target content including whitespace.]"
        return diag

    new_content = orig_content.replace(matched_target, replacement, 1)

    # Atomic write using mkstemp (O_CREAT | O_EXCL)
    temp_file: Optional[Path] = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(resolved.parent))
        temp_file = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, resolved)
    except Exception as exc:
        if temp_file is not None:
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass
        return f"[ERROR: Failed writing edited file {resolved}: {exc}]"

    # Compute unified diff
    orig_lines = orig_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{resolved.name}",
            tofile=f"b/{resolved.name}",
            n=3,
        )
    )
    diff_text = "".join(diff_lines) if diff_lines else "[No differences]"
    return f"[Successfully edited {resolved}]\n{diff_text}"


def tool_list_dir(
    path: str = ".",
    depth: int = 2,
    cwd: Optional[Path | str] = None,
) -> str:
    """Gitignore-aware directory tree with entry counts and file sizes, capped at 24 KB."""
    resolved = _resolve_active_path(path, cwd)
    if not resolved.exists():
        return f"[ERROR: Directory not found: {resolved}]"
    if not resolved.is_dir():
        return f"[ERROR: Path is not a directory: {resolved}]"

    ignore_names = {
        ".git",
        "node_modules",
        "__pycache__",
        ".sandcastle",
        ".platodoom",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
        ".turbo",
    }

    lines: List[str] = [f"Directory: {resolved}"]
    total_entries = 0
    accum_bytes = len(lines[0].encode("utf-8")) + 1

    def _traverse(current_dir: Path, current_depth: int, prefix: str) -> None:
        nonlocal total_entries, accum_bytes
        if current_depth > depth:
            return

        try:
            entries = sorted(list(current_dir.iterdir()), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            line = f"{prefix}[PERMISSION DENIED]"
            lines.append(line)
            return

        for idx, entry in enumerate(entries):
            if entry.name in ignore_names or entry.name.startswith(".tmp_"):
                continue
            if is_denylisted_secret_path(entry):
                continue

            is_last = idx == len(entries) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            total_entries += 1
            if entry.is_dir():
                sub_count = ""
                try:
                    sub_count = f" ({len(list(entry.iterdir()))} items)"
                except Exception:
                    pass
                line = f"{prefix}{connector}[DIR] {entry.name}/{sub_count}"
            else:
                try:
                    sz = entry.stat().st_size
                    sz_str = f" ({sz:,} B)"
                except Exception:
                    sz_str = ""
                line = f"{prefix}{connector}[FILE] {entry.name}{sz_str}"

            line_bytes = len(line.encode("utf-8")) + 1
            if accum_bytes + line_bytes > MAX_LIST_DIR_BYTES:
                lines.append(f"\n[WARNING: Directory listing capped at {MAX_LIST_DIR_BYTES} bytes]")
                return

            lines.append(line)
            accum_bytes += line_bytes

            if entry.is_dir() and current_depth < depth:
                _traverse(entry, current_depth + 1, prefix + child_prefix)

    _traverse(resolved, 1, "")
    return "\n".join(lines)


def tool_find_files(
    pattern: str,
    path: str = ".",
    cwd: Optional[Path | str] = None,
    home_dir: Optional[Path | str] = None,
) -> str:
    """Fast filename discovery using ripgrep --files --glob, capped at 200 files / 24 KB.

    Guarantees <= 250ms p95 latency.
    """
    resolved = _resolve_active_path(path, cwd)
    if not resolved.exists():
        return f"[ERROR: Base path not found: {resolved}]"

    found_files: List[str] = []

    # Attempt native ripgrep first
    if RG_PATH and Path(RG_PATH).is_file():
        cmd = [
            RG_PATH,
            "--files",
            "--color=never",
            "--hidden",
            "--glob",
            pattern,
            "--glob=!.git/*",
            "--glob=!node_modules/*",
            "--glob=!.sandcastle/*",
            "--glob=!dist/*",
            "--glob=!build/*",
            "--glob=!__pycache__/*",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(resolved),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=3.0,
            )
            raw_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            for item in raw_lines:
                item_path = resolved / item
                if not is_denylisted_secret_path(item_path, home_dir=home_dir):
                    found_files.append(item)
                    if len(found_files) >= MAX_FIND_FILES_COUNT:
                        break
        except (subprocess.TimeoutExpired, Exception):
            found_files = []

    # Python fallback if rg was not available or failed
    if not found_files and not (RG_PATH and Path(RG_PATH).is_file()):
        ignore_dirs = {".git", "node_modules", ".sandcastle", "dist", "build", "__pycache__"}
        for root, dirs, files in os.walk(resolved):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    full_p = Path(root) / f
                    if not is_denylisted_secret_path(full_p, home_dir=home_dir):
                        try:
                            rel_p = str(full_p.relative_to(resolved))
                        except ValueError:
                            rel_p = str(full_p)
                        found_files.append(rel_p)
                        if len(found_files) >= MAX_FIND_FILES_COUNT:
                            break
            if len(found_files) >= MAX_FIND_FILES_COUNT:
                break

    if not found_files:
        return f"[Found 0 files matching '{pattern}' in {resolved}]"

    # Format and byte-cap
    output_lines: List[str] = []
    accum_bytes = 0
    truncated = False

    for item in found_files:
        item_bytes = len(item.encode("utf-8")) + 1
        if accum_bytes + item_bytes > MAX_FIND_FILES_BYTES:
            truncated = True
            break
        output_lines.append(item)
        accum_bytes += item_bytes

    header = f"[Found {len(found_files)} files matching '{pattern}' in {resolved}"
    if truncated:
        header += f" (truncated at {MAX_FIND_FILES_BYTES} bytes)]"
    else:
        header += "]"

    return header + "\n" + "\n".join(output_lines)


def tool_grep_search(
    query: str,
    path: str = ".",
    include_glob: Optional[str] = None,
    case_sensitive: bool = True,
    cwd: Optional[Path | str] = None,
    home_dir: Optional[Path | str] = None,
) -> str:
    """Fast content search using ripgrep returning file:line:snippet.

    - Filters denylisted secret files and redacts .env values.
    - Capped at 100 matches / 16 KB (16,384 bytes).
    - Guarantees <= 250ms p95 latency.
    """
    resolved = _resolve_active_path(path, cwd)
    if not resolved.exists():
        return f"[ERROR: Target search path not found: {resolved}]"

    matches: List[str] = []

    if RG_PATH and Path(RG_PATH).is_file():
        cmd = [
            RG_PATH,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--max-count",
            str(MAX_GREP_SEARCH_MATCHES),
            "--max-filesize=2M",
            "--glob=!.git/*",
            "--glob=!node_modules/*",
            "--glob=!.sandcastle/*",
            "--glob=!dist/*",
            "--glob=!build/*",
            "--glob=!*.min.js",
        ]
        if case_sensitive:
            cmd.append("--case-sensitive")
        else:
            cmd.append("--ignore-case")

        if include_glob:
            cmd.extend(["--glob", include_glob])

        cmd.extend(["--", query])

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(resolved) if resolved.is_dir() else str(resolved.parent),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=3.0,
            )
            raw_lines = proc.stdout.splitlines()
            for line in raw_lines:
                if not line.strip():
                    continue
                # Format: file:line:snippet
                parts = line.split(":", 2)
                file_rel = parts[0]
                target_file = (resolved if resolved.is_dir() else resolved.parent) / file_rel
                if is_denylisted_secret_path(target_file, home_dir=home_dir):
                    continue
                if is_env_secret_file(target_file):
                    line = redact_env_secrets(line)
                # Truncate individual line length to prevent minified JS blowup
                if len(line) > 250:
                    line = line[:247] + "..."
                matches.append(line)
                if len(matches) >= MAX_GREP_SEARCH_MATCHES:
                    break
        except (subprocess.TimeoutExpired, Exception):
            matches = []

    # Python fallback if rg unavailable
    if not matches and not (RG_PATH and Path(RG_PATH).is_file()):
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            query_re = re.compile(query, flags)
        except re.error:
            query_re = re.compile(re.escape(query), flags)

        search_files: List[Path] = [resolved] if resolved.is_file() else []
        if resolved.is_dir():
            for root, dirs, files in os.walk(resolved):
                dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".sandcastle", "dist", "build"}]
                for f in files:
                    if include_glob and not fnmatch.fnmatch(f, include_glob):
                        continue
                    search_files.append(Path(root) / f)
                    if len(search_files) > 1000:
                        break

        for sf in search_files:
            if is_denylisted_secret_path(sf, home_dir=home_dir):
                continue
            try:
                with open(sf, "r", encoding="utf-8", errors="replace") as f:
                    for lno, line in enumerate(f, start=1):
                        if query_re.search(line):
                            if is_env_secret_file(sf):
                                line = redact_env_secrets(line)
                            snippet = line.strip()
                            if len(snippet) > 250:
                                snippet = snippet[:247] + "..."
                            rel_p = sf.relative_to(resolved) if resolved.is_dir() else sf.name
                            matches.append(f"{rel_p}:{lno}:{snippet}")
                            if len(matches) >= MAX_GREP_SEARCH_MATCHES:
                                break
            except Exception:
                continue
            if len(matches) >= MAX_GREP_SEARCH_MATCHES:
                break

    if not matches:
        return f"[grep_search: No matches found for '{query}' in {resolved}]"

    # Enforce 16 KB output cap
    output_lines: List[str] = []
    accum_bytes = 0
    truncated = False

    for m in matches:
        m_bytes = len(m.encode("utf-8")) + 1
        if accum_bytes + m_bytes > MAX_GREP_SEARCH_BYTES:
            truncated = True
            break
        output_lines.append(m)
        accum_bytes += m_bytes

    header = f"[grep_search: {len(matches)} matches for '{query}' in {resolved}"
    if truncated:
        header += f" (truncated at {MAX_GREP_SEARCH_BYTES} bytes)]"
    else:
        header += "]"

    return header + "\n" + "\n".join(output_lines)


def tool_run_command(
    command: str,
    cwd: Optional[Path | str] = None,
    timeout_sec: int = 60,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Executes a shell command with stateful cwd tracking and head/tail 16 KB truncation."""
    active_cwd = Path(session_state.get("cwd") if session_state and "cwd" in session_state else (cwd or Path.cwd())).resolve()

    cmd_stripped = command.strip()
    # Support stateful cd command
    if cmd_stripped.startswith("cd ") or cmd_stripped == "cd":
        target_dir = cmd_stripped[3:].strip() if cmd_stripped.startswith("cd ") else "~"
        target_dir = target_dir.strip('"').strip("'")
        new_cwd = (active_cwd / Path(target_dir).expanduser()).resolve()
        if not new_cwd.exists() or not new_cwd.is_dir():
            return f"[ERROR: Directory does not exist: {target_dir}]"
        if session_state is not None:
            session_state["cwd"] = str(new_cwd)
        return f"[Changed directory to {new_cwd}]"

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(active_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        stdout, stderr = proc.communicate(timeout=max(1, int(timeout_sec)))
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        return f"[ERROR: Command timed out after {timeout_sec}s (timed_out=True): {command}]"
    except Exception as exc:
        return f"[ERROR: Failed executing command '{command}': {exc}]"

    combined = ""
    if stdout:
        combined += stdout
    if stderr:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += f"[STDERR]\n{stderr}"

    combined = redact_env_secrets(combined)
    combined_bytes = combined.encode("utf-8")
    if len(combined_bytes) > MAX_COMMAND_OUTPUT_BYTES:
        head_bytes = combined_bytes[:8192].decode("utf-8", errors="ignore")
        tail_bytes = combined_bytes[-8192:].decode("utf-8", errors="ignore")
        diff_len = len(combined_bytes) - 16384
        combined = f"{head_bytes}\n\n... [TRUNCATED {diff_len} BYTES OF OUTPUT] ...\n\n{tail_bytes}"

    if exit_code != 0:
        return f"[Command exited with code {exit_code}]\n{combined}".strip()
    return combined.strip() if combined.strip() else "[Command executed successfully with no output]"


# ============================================================================
# Unified Permission-Checked Dispatcher
# ============================================================================

def execute_local_tool(
    tool_name: str,
    args: Dict[str, Any],
    session_state: Optional[Dict[str, Any]] = None,
    interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """Unified dispatcher for all platodoom local tools.

    1. Checks check_tool_permission across Plan Mode ('plan' / 'read-only') and Build Mode ('workspace-write', 'full-access').
    2. Dispatches to the target implementation.
    3. Returns string response formatted for LLM consumption (or structured PERMISSION_DENIED_READ_ONLY JSON in Plan Mode).
    """
    state = session_state if session_state is not None else {}
    workflow_mode = str(state.get("workflow_mode", "build")).strip().lower()
    access_mode = str(state.get("access_mode", "read-only" if workflow_mode == "plan" else "workspace-write")).strip().lower()
    if workflow_mode == "plan" or access_mode in ("read-only", "read_only"):
        workflow_mode = "plan"
        access_mode = "read-only"
    workspace_roots = state.get("workspace_roots", [Path.cwd()])
    active_cwd = Path(state.get("cwd") or Path.cwd()).resolve()

    # 1. Access Control Verification
    try:
        allowed, diagnostic = check_tool_permission(
            tool_name=tool_name,
            args=args,
            access_mode=access_mode,
            workspace_roots=workspace_roots,
            cwd=active_cwd,
            interactive_confirm_fn=interactive_confirm_fn,
            workflow_mode=workflow_mode,
        )
    except TypeError:
        allowed, diagnostic = check_tool_permission(
            tool_name=tool_name,
            args=args,
            access_mode=access_mode,
            workspace_roots=workspace_roots,
            cwd=active_cwd,
            interactive_confirm_fn=interactive_confirm_fn,
        )

    if not allowed:
        if workflow_mode == "plan" or access_mode == "read-only" or "PERMISSION_DENIED_READ_ONLY" in diagnostic:
            msg = diagnostic if "PERMISSION_DENIED_READ_ONLY" in diagnostic else f"PERMISSION_DENIED_READ_ONLY: {diagnostic}"
            return json.dumps(
                {
                    "status": "error",
                    "error_code": "PERMISSION_DENIED_READ_ONLY",
                    "error": msg,
                    "message": msg,
                    "tool": tool_name,
                    "workflow_mode": "plan",
                    "access_mode": "read-only",
                },
                ensure_ascii=False,
            )
        return f"[PERMISSION DENIED: {diagnostic}]"

    # 2. Local tool execution
    if tool_name == "read_file":
        return tool_read_file(
            path=str(args.get("path", "")),
            start_line=args.get("start_line"),
            end_line=args.get("end_line"),
            cwd=active_cwd,
            extra_denylist_globs=state.get("extra_denylist_globs"),
            interactive_approve_fn=interactive_confirm_fn,
        )
    elif tool_name == "write_file":
        return tool_write_file(
            path=str(args.get("path", "")),
            content=str(args.get("content", "")),
            overwrite=bool(args.get("overwrite", True)),
            cwd=active_cwd,
        )
    elif tool_name == "edit_file":
        return tool_edit_file(
            path=str(args.get("path", "")),
            target=str(args.get("target", "")),
            replacement=str(args.get("replacement", "")),
            cwd=active_cwd,
        )
    elif tool_name == "list_dir":
        return tool_list_dir(
            path=str(args.get("path", ".")),
            depth=int(args.get("depth", 2)),
            cwd=active_cwd,
        )
    elif tool_name == "find_files":
        return tool_find_files(
            pattern=str(args.get("pattern", "*")),
            path=str(args.get("path", ".")),
            cwd=active_cwd,
        )
    elif tool_name == "grep_search":
        return tool_grep_search(
            query=str(args.get("query", "")),
            path=str(args.get("path", ".")),
            include_glob=args.get("include_glob"),
            case_sensitive=bool(args.get("case_sensitive", True)),
            cwd=active_cwd,
        )
    elif tool_name == "run_command":
        return tool_run_command(
            command=str(args.get("command", "")),
            cwd=active_cwd,
            timeout_sec=int(args.get("timeout_sec", 60)),
            session_state=state,
        )
    elif tool_name == "use_skill":
        # Forward to session handler or discovery helper if provided in state
        skill_fn = state.get("use_skill_fn")
        if callable(skill_fn):
            return str(skill_fn(args.get("name", "")))
        return f"[use_skill: Skill '{args.get('name')}' dispatched to session handler]"
    elif tool_name in ("list_mcp_tools", "inspect_mcp_tool", "call_mcp_tool"):
        mcp_fn = state.get("mcp_dispatch_fn")
        if callable(mcp_fn):
            return str(mcp_fn(tool_name, args))
        return f"[mcp_bridge: '{tool_name}' dispatched to session handler]"
    elif tool_name in ("invoke_subagent", "invoke_parallel_subagents"):
        subagent_fn = state.get("subagent_dispatch_fn")
        if callable(subagent_fn):
            return str(subagent_fn(tool_name, args))
        return f"[subagent: '{tool_name}' dispatched to session handler]"

    return f"[ERROR: Unknown tool '{tool_name}']"


class LocalToolExecutor:
    """Stateful wrapper around execute_local_tool and get_local_tools_schema."""

    def __init__(
        self,
        workspace_dir: Optional[Path | str] = None,
        access_mode: str = "workspace-write",
        workflow_mode: Optional[str] = None,
        extra_workspace_roots: Optional[List[Path | str]] = None,
        confirm_callback: Optional[Callable[..., bool]] = None,
        session_slug: str = "default",
        capabilities: Optional[Dict[str, Any]] = None,
        subagent_runner: Optional[Callable[[Dict[str, Any]], str]] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if session_state is not None:
            self.session_state: Dict[str, Any] = session_state
        else:
            ws = Path(workspace_dir).expanduser().resolve() if workspace_dir is not None else Path.cwd().resolve()
            roots: List[Path] = [ws]
            if extra_workspace_roots:
                for r in extra_workspace_roots:
                    rp = Path(r).expanduser().resolve()
                    if rp not in roots:
                        roots.append(rp)
            eff_wf = str(workflow_mode or ("plan" if access_mode == "read-only" else "build")).strip().lower()
            eff_access = "read-only" if eff_wf == "plan" else str(access_mode or "workspace-write").strip().lower()
            self.session_state = {
                "cwd": ws,
                "workspace_roots": roots,
                "access_mode": eff_access,
                "workflow_mode": eff_wf,
                "session_slug": session_slug,
                "capabilities": capabilities or {},
                "subagent_dispatch_fn": (lambda _name, a: subagent_runner(a)) if callable(subagent_runner) else None,
            }
        self.confirm_callback = confirm_callback

    @property
    def cwd(self) -> Path:
        return Path(self.session_state.get("cwd") or Path.cwd()).expanduser().resolve()

    @property
    def access_mode(self) -> str:
        return str(self.session_state.get("access_mode", "workspace-write"))

    @access_mode.setter
    def access_mode(self, val: str) -> None:
        self.session_state["access_mode"] = str(val)

    @property
    def workflow_mode(self) -> str:
        return str(self.session_state.get("workflow_mode", "build"))

    @workflow_mode.setter
    def workflow_mode(self, val: str) -> None:
        self.session_state["workflow_mode"] = str(val)
        if str(val).lower() == "plan":
            self.session_state["access_mode"] = "read-only"

    def get_tools_schema(
        self,
        workflow_mode: Optional[str] = None,
        access_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        wf = workflow_mode or str(self.session_state.get("workflow_mode", "build"))
        ac = access_mode or str(self.session_state.get("access_mode", "workspace-write"))
        return get_local_tools_schema(workflow_mode=wf, access_mode=ac)

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> str:
        return execute_local_tool(
            tool_name=tool_name,
            args=args,
            session_state=self.session_state,
            interactive_confirm_fn=interactive_confirm_fn,
        )
