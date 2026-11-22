#!/usr/bin/env python3
"""
Local Worktree Agent Tool-Loop Runner for qwen3.8-27b.
Document ID: qwen_worktree_agent_856_v1
Governing Standards: platodoom specification (FR-3.1, FR-4.3), LANGUAGE.md, RULE[user_global]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curation.api_client import (
    MAX_MODEL_LEN,
    UNIFIED_MODEL_ALIAS,
    _open_http_request,
    normalize_colab_endpoint_url,
    resolve_expected_model_identifiers,
    resolve_serving_bearer_token,
)
from scripts.provision_colab_serving import (
    compute_dir_slug,
    ensure_colab_serving_endpoint,
    probe_colab_qwen_endpoint,
    read_registered_endpoint,
    resolve_project_root,
)

_ORIG_WORKTREE_URLOPEN = urllib.request.urlopen


def _open_worktree_http_request(req: urllib.request.Request, timeout: float = 300.0) -> Any:
    if urllib.request.urlopen is not _ORIG_WORKTREE_URLOPEN:
        return urllib.request.urlopen(req, timeout=timeout)
    return _open_http_request(req, timeout=timeout)

TOOL_CALL_XML_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
QWEN_XML_FN_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
QWEN_XML_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
DEEPSEEK_TOOL_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?:function<｜tool▁sep｜>)?([^\n<]+)\n```json\n(.*?)\n```<｜tool▁call▁end｜>",
    re.DOTALL,
)
THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)

WORKSPACE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the isolated git worktree (including blackboard, git, pytest, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "CommandLine": {"type": "string", "description": "Exact command line string to execute."},
                    "WaitMsBeforeAsync": {"type": "integer", "default": 5000},
                    "toolSummary": {"type": "string", "description": "Brief summary of the command."},
                },
                "required": ["CommandLine"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file inside the worktree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Path": {"type": "string", "description": "File path to read."},
                    "StartLine": {"type": "integer"},
                    "EndLine": {"type": "integer"},
                },
                "required": ["Path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the worktree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Path": {"type": "string", "description": "Path of the file to write."},
                    "Content": {"type": "string", "description": "Full contents to write."},
                    "Overwrite": {"type": "boolean", "default": True},
                    "Append": {"type": "boolean", "default": False},
                },
                "required": ["Path", "Content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact target string in an existing file in the worktree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Path": {"type": "string", "description": "File path to modify."},
                    "Target": {"type": "string", "description": "Exact string to be replaced."},
                    "Replacement": {"type": "string", "description": "Replacement content."},
                },
                "required": ["Path", "Target", "Replacement"],
            },
        },
    },
]


def estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Heuristic token estimator (~3.5 chars per token for code/JSON ChatML)."""
    total_chars = 0
    for msg in messages:
        total_chars += len(str(msg.get("role", ""))) + 8
        total_chars += len(str(msg.get("content", "")))
        if msg.get("tool_calls"):
            total_chars += len(json.dumps(msg["tool_calls"]))
    return max(1, int(total_chars / 3.5))


def compact_messages_for_16k_window(
    messages: List[Dict[str, Any]],
    max_model_len: int = MAX_MODEL_LEN,
    reserve_completion_tokens: int = 4096,
) -> List[Dict[str, Any]]:
    """Deterministically compacts oldest middle <tool_response> payloads when approaching the 16,384-token budget,
    preserving Turn 0 (system + initial user task prompt) and the most recent turns (FR-4.3).
    """
    budget = max(2048, int(int(max_model_len) * 0.80) - int(reserve_completion_tokens))
    compacted = [dict(m) for m in messages]
    # Always strip completed <think>...</think> blocks from prior assistant turns (FR-3.4)
    for idx, msg in enumerate(compacted):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str) and "<think>" in msg["content"]:
            msg["content"] = THINK_BLOCK_RE.sub("", msg["content"]).strip()
    if estimate_message_tokens(compacted) <= budget:
        return compacted

    # Pass 1: Truncate oldest middle tool responses (keeping first 2 messages and last 4 messages intact)
    protect_tail = min(4, max(0, len(compacted) - 2))
    cutoff_idx = len(compacted) - protect_tail
    for idx in range(2, cutoff_idx):
        if estimate_message_tokens(compacted) <= budget:
            break
        msg = compacted[idx]
        content = str(msg.get("content", ""))
        if msg.get("role") == "tool" and len(content) > 320:
            head = content[:180]
            tail = content[-100:]
            msg["content"] = f"{head}\n...[compacted tool output for 16k window]...\n{tail}"

    # Pass 2: If still over budget, drop oldest middle turn pairs between index 2 and cutoff_idx
    while len(compacted) > 6 and estimate_message_tokens(compacted) > budget:
        del compacted[2]

    # Pass 3: If any remaining tool message is still huge, cap it so total stays within budget
    if estimate_message_tokens(compacted) > budget:
        for idx in range(2, len(compacted)):
            msg = compacted[idx]
            content = str(msg.get("content", ""))
            if len(content) > 600:
                msg["content"] = content[:350] + "\n...[truncated for 16k window]...\n" + content[-150:]

    return compacted


def parse_assistant_tool_calls(
    message: Dict[str, Any],
    *,
    forces_think_prefix: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts tool calls from OpenAI `tool_calls`, Qwen XML, `<tool_call>` JSON, and DeepSeek native markers.
    Returns `(cleaned_text, parsed_calls)` where malformed JSON produces an explicit `malformed_error` entry (Section 4.3).
    """
    content = str(message.get("content") or "")
    structured = message.get("tool_calls") or []
    try:
        from platodoom.catalog import parse_universal_tool_calls, split_thinking_and_content

        _, without_think = split_thinking_and_content(content, forces_think_prefix=forces_think_prefix)
        cleaned, parsed = parse_universal_tool_calls(
            without_think,
            openai_tool_calls=structured if isinstance(structured, list) and len(structured) > 0 else None,
        )
        normalized: List[Dict[str, Any]] = []
        for idx, tc in enumerate(parsed):
            normalized.append(
                {
                    "id": str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}_{idx}"),
                    "name": str(tc.get("name") or "run_command"),
                    "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {},
                    "malformed_error": tc.get("malformed_error"),
                }
            )
        return cleaned, normalized
    except ImportError:
        pass

    calls: List[Dict[str, Any]] = []

    if isinstance(structured, list) and len(structured) > 0:
        for idx, tc in enumerate(structured):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or tc.get("name") or "run_command")
            raw_args = fn.get("arguments", tc.get("arguments", "{}"))
            call_id = str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}_{idx}")
            if isinstance(raw_args, dict):
                calls.append({"id": call_id, "name": name, "arguments": raw_args, "malformed_error": None})
            else:
                try:
                    parsed_args = json.loads(str(raw_args)) if str(raw_args).strip() else {}
                    if not isinstance(parsed_args, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    calls.append({"id": call_id, "name": name, "arguments": parsed_args, "malformed_error": None})
                except Exception as exc:
                    calls.append(
                        {
                            "id": call_id,
                            "name": name,
                            "arguments": {},
                            "malformed_error": f"Malformed tool_call JSON arguments: {exc}. Raw: {raw_args}",
                        }
                    )

    if "<tool_call>" in content:
        for idx, match in enumerate(TOOL_CALL_XML_RE.finditer(content)):
            raw_block = match.group(1).strip()
            call_id = f"call_xml_{uuid.uuid4().hex[:8]}_{idx}"
            fn_match = QWEN_XML_FN_RE.search(raw_block)
            if fn_match:
                fn_name = fn_match.group(1).strip()
                fn_body = fn_match.group(2)
                params: Dict[str, Any] = {}
                for p_match in QWEN_XML_PARAM_RE.finditer(fn_body):
                    p_key = p_match.group(1).strip()
                    p_raw = p_match.group(2)
                    p_stripped = p_raw.strip()
                    if p_stripped in ("true", "false", "null") or (
                        p_stripped and (p_stripped[0] in "{[0123456789-" and not p_key.lower().endswith("path"))
                    ):
                        try:
                            params[p_key] = json.loads(p_stripped)
                            continue
                        except Exception:
                            pass
                    params[p_key] = p_raw
                calls.append({"id": call_id, "name": fn_name, "arguments": params, "malformed_error": None})
                continue
            try:
                parsed = json.loads(raw_block)
                if not isinstance(parsed, dict):
                    raise ValueError("Tool call payload must be a JSON object")
                name = str(parsed.get("name") or parsed.get("tool") or "run_command")
                args_val = parsed.get("arguments", parsed.get("args", {}))
                if isinstance(args_val, str):
                    args_val = json.loads(args_val)
                if not isinstance(args_val, dict):
                    raise ValueError("Tool arguments must be a dictionary")
                calls.append({"id": call_id, "name": name, "arguments": args_val, "malformed_error": None})
            except Exception as exc:
                calls.append(
                    {
                        "id": call_id,
                        "name": "malformed_tool_call",
                        "arguments": {},
                        "malformed_error": f"Malformed <tool_call> JSON: {exc}. Please emit valid JSON inside <tool_call> tags.",
                    }
                )
        content = TOOL_CALL_XML_RE.sub("", content).strip()

    if "<｜tool▁call▁begin｜>" in content:
        for idx, match in enumerate(DEEPSEEK_TOOL_CALL_RE.finditer(content)):
            fn_name = match.group(1).strip()
            raw_json = match.group(2).strip()
            call_id = f"call_ds_{uuid.uuid4().hex[:8]}_{idx}"
            try:
                parsed_args = json.loads(raw_json) if raw_json else {}
                if not isinstance(parsed_args, dict):
                    raise ValueError("Tool arguments must be a dictionary")
                calls.append({"id": call_id, "name": fn_name, "arguments": parsed_args, "malformed_error": None})
            except Exception as exc:
                calls.append(
                    {
                        "id": call_id,
                        "name": fn_name,
                        "arguments": {},
                        "malformed_error": f"Malformed DeepSeek tool_call JSON: {exc}",
                    }
                )
        content = DEEPSEEK_TOOL_CALL_RE.sub("", content).strip()
        content = content.replace("<｜tool▁calls▁begin｜>", "").replace("<｜tool▁calls▁end｜>", "").strip()

    if "<think>" in content:
        content = THINK_BLOCK_RE.sub("", content).strip()

    return content, calls


def resolve_worktree_path(worktree_dir: Path, raw_path: str) -> Path:
    """Resolves a tool file path inside the active agent worktree."""
    p = Path(raw_path)
    if not p.is_absolute():
        return (worktree_dir / p).resolve()
    try:
        wt_resolved = worktree_dir.resolve()
        if ".sandcastle/worktrees/" in str(wt_resolved):
            repo_root = resolve_project_root(wt_resolved)
            if str(p).startswith(str(repo_root) + "/") and not str(p).startswith(str(wt_resolved)):
                rel = p.relative_to(repo_root)
                if not str(rel).startswith(".sandcastle/"):
                    return (wt_resolved / rel).resolve()
    except Exception:
        pass
    return p


def _resolve_live_vllm_base_url(
    worktree_dir: Path | str,
    explicit_url: str = "",
    model_alias: Optional[str] = None,
) -> str:
    """Resolves the canonical project root's active Colab serving URL (FR-3.1, FR-4.3, Spec 872)."""
    if explicit_url and explicit_url.strip():
        return normalize_colab_endpoint_url(explicit_url.strip())

    env_url = (
        os.environ.get("PLATODOOM_ENDPOINT_URL")
        or os.environ.get("SANDCASTLE_VLLM_URL")
        or os.environ.get("VLLM_API_BASE")
        or ""
    )
    if env_url.strip():
        return normalize_colab_endpoint_url(env_url.strip())

    eff_model = str(model_alias or os.environ.get("PLATODOOM_MODEL") or "").strip() or None
    project_root = resolve_project_root(worktree_dir)
    existing = read_registered_endpoint(repo_dir=project_root, include_global=False)
    if existing and existing.get("endpoint_url"):
        cand_url = str(existing["endpoint_url"])
        tok = resolve_serving_bearer_token(record=existing)
        expected_mod = eff_model or str(existing.get("active_model") or existing.get("model") or "").strip() or None
        probe_kwargs: Dict[str, Any] = {
            "api_key": tok if tok != "EMPTY" else None,
            "timeout_sec": 5.0,
            "verify_inference": True,
        }
        if expected_mod:
            probe_kwargs["expected_model"] = expected_mod
        if tok != "EMPTY":
            probe_kwargs["bearer_token"] = tok
        try:
            import inspect

            psig = inspect.signature(probe_colab_qwen_endpoint)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in psig.parameters.values()):
                probe_kwargs = {k: v for k, v in probe_kwargs.items() if k in psig.parameters}
        except Exception:
            pass
        probe = probe_colab_qwen_endpoint(cand_url, **probe_kwargs)
        if probe["online"]:
            return normalize_colab_endpoint_url(cand_url)

    ensure_kwargs: Dict[str, Any] = {
        "repo_dir": project_root,
        "auto_provision": True,
        "include_global": False,
        "headless_launch": os.environ.get("SANDCASTLE_DISABLE_BROWSER") != "1",
    }
    if eff_model:
        ensure_kwargs["model_alias"] = eff_model
    ensured = ensure_colab_serving_endpoint(**ensure_kwargs)
    return normalize_colab_endpoint_url(str(ensured["endpoint_url"]))


def execute_worktree_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    worktree_dir: Path,
    env: Dict[str, str],
    command_timeout_sec: int = 120,
) -> str:
    """Executes one of the core workspace tools (run_command, read_file, write_file, edit_file) inside the worktree with Zero-Trust secret protection."""
    from platodoom.security import check_secret_path_and_redact, is_env_secret_file, redact_env_secrets

    if tool_name == "run_command":
        cmd = str(arguments.get("CommandLine") or arguments.get("command") or "").strip()
        if not cmd:
            return "ERROR: run_command requires a non-empty 'CommandLine' argument."
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(worktree_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=command_timeout_sec,
            )
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            out = redact_env_secrets(out.strip())
            if len(out) > 12000:
                out = out[:6000] + "\n...[command output truncated]...\n" + out[-4000:]
            return f"Exit code: {proc.returncode}\n{out}".strip()
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {command_timeout_sec}s: {cmd}"
        except Exception as exc:
            return f"ERROR executing command '{cmd}': {exc}"

    if tool_name == "read_file":
        raw_path = str(arguments.get("Path") or arguments.get("path") or "").strip()
        if not raw_path:
            return "ERROR: read_file requires 'Path'."
        target = resolve_worktree_path(worktree_dir, raw_path)
        chk = check_secret_path_and_redact(target, content=None)
        if not chk.get("allowed", True):
            return str(chk.get("reason") or f"[SECURITY BLOCKED: Access to {target} denied]")
        if not target.exists() or not target.is_file():
            return f"ERROR: File not found: {raw_path}"
        try:
            raw_content = target.read_text(encoding="utf-8", errors="replace")
            if is_env_secret_file(target):
                raw_content = redact_env_secrets(raw_content)
            lines = raw_content.splitlines()
            start_line = int(arguments.get("StartLine") or 1)
            end_line = int(arguments.get("EndLine") or min(len(lines), start_line + 799))
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            selected = [f"{i + 1}: {lines[i]}" for i in range(start_idx, end_idx)]
            body = "\n".join(selected)
            if len(body) > 46080:
                body = body[:46080] + "\n...[truncated at 46080 bytes]..."
            return f"Showing lines {start_idx + 1}-{end_idx} of {len(lines)}\n{body}"
        except Exception as exc:
            return f"ERROR reading file '{raw_path}': {exc}"

    if tool_name == "write_file":
        raw_path = str(arguments.get("Path") or arguments.get("path") or "").strip()
        content = str(arguments.get("Content") if "Content" in arguments else arguments.get("content", ""))
        target = resolve_worktree_path(worktree_dir, raw_path)
        chk = check_secret_path_and_redact(target, content=None)
        if not chk.get("allowed", True):
            return str(chk.get("reason") or f"[SECURITY BLOCKED: Writing {target} denied]")
        append = bool(arguments.get("Append", False))
        overwrite = bool(arguments.get("Overwrite", True))
        if not raw_path:
            return "ERROR: write_file requires 'Path'."
        target = resolve_worktree_path(worktree_dir, raw_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite and not append:
                return f"ERROR: File already exists and Overwrite=False: {raw_path}"
            mode = "a" if append else "w"
            with open(target, mode, encoding="utf-8") as fp:
                fp.write(content)
            return f"Successfully wrote {len(content)} chars to {raw_path}."
        except Exception as exc:
            return f"ERROR writing file '{raw_path}': {exc}"

    if tool_name == "edit_file":
        raw_path = str(arguments.get("Path") or arguments.get("path") or "").strip()
        target_str = str(arguments.get("Target") if "Target" in arguments else arguments.get("target", ""))
        replacement_str = str(
            arguments.get("Replacement") if "Replacement" in arguments else arguments.get("replacement", "")
        )
        if not raw_path or not target_str:
            return "ERROR: edit_file requires 'Path' and non-empty 'Target'."
        target = resolve_worktree_path(worktree_dir, raw_path)
        chk = check_secret_path_and_redact(target, content=None)
        if not chk.get("allowed", True):
            return str(chk.get("reason") or f"[SECURITY BLOCKED: Editing {target} denied]")
        if not target.exists() or not target.is_file():
            return f"ERROR: File not found: {raw_path}"
        try:
            original = target.read_text(encoding="utf-8")
            if target_str not in original:
                return f"ERROR: Target string not found in {raw_path}."
            updated = original.replace(target_str, replacement_str)
            target.write_text(updated, encoding="utf-8")
            return f"Successfully updated {raw_path}."
        except Exception as exc:
            return f"ERROR editing file '{raw_path}': {exc}"

    return f"ERROR: Unsupported tool '{tool_name}'. Available tools: run_command, read_file, write_file, edit_file."


def emit_stream_json_line(payload: Dict[str, Any], stream_out: Callable[[str], None]) -> None:
    """Emits a single NDJSON line matching the .mts parseStreamLine contract (FR-4.4)."""
    stream_out(json.dumps(payload, ensure_ascii=False))


def run_worktree_agent_loop(
    prompt: str,
    worktree_dir: Path | str,
    endpoint_url: str = "",
    model: str = UNIFIED_MODEL_ALIAS,
    max_turns: int = 24,
    env: Optional[Dict[str, str]] = None,
    stream_out: Optional[Callable[[str], None]] = None,
    enforce_blackboard_done: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    session_slug: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    auto_provision: bool = True,
) -> Dict[str, Any]:
    """Executes the multi-turn Qwen3.5-9B tool loop inside the agent's git worktree and streams NDJSON events."""
    from platodoom.config import (
        atomic_write_json,
        resolve_response_cache_dir,
        resolve_session_state_dir,
    )
    from platodoom.response_cache import (
        LocalResponseCache,
        compute_response_cache_key,
        redact_cache_payload,
    )

    wt_path = Path(worktree_dir).resolve()
    project_root = resolve_project_root(wt_path)
    exec_env = dict(os.environ)
    if env:
        exec_env.update(env)

    # Ensure .sandcastle/bin is prefixed to PATH
    shim_candidates = [
        wt_path / ".sandcastle" / "bin",
        project_root / ".sandcastle" / "bin",
        REPO_ROOT / ".sandcastle" / "bin",
    ]
    path_parts = [str(p) for p in shim_candidates if p.is_dir()]
    if exec_env.get("PATH"):
        path_parts.append(exec_env["PATH"])
    exec_env["PATH"] = ":".join(path_parts)

    if stream_out is None:
        stream_out = lambda line: print(line, flush=True)

    eff_agent_id = str(agent_id).strip() if (agent_id and str(agent_id).strip()) else exec_env.get("SANDCASTLE_AGENT_ID", "sandcastle-agent")
    agent_id = eff_agent_id
    dir_slug = compute_dir_slug(project_root)
    safe_agent_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(agent_id)).strip("-") or "agent"
    eff_session_slug = session_slug or f"{dir_slug}-{safe_agent_slug}"
    session_dir = resolve_session_state_dir(eff_session_slug, env=exec_env)
    session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(session_dir, 0o700)
    except OSError:
        pass
    cache_dir = resolve_response_cache_dir(env=exec_env)
    response_cache = LocalResponseCache(cache_dir=cache_dir, enabled=bool(use_cache))
    transcript_path = session_dir / "transcript.jsonl"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _append_local_transcript(event_obj: Dict[str, Any]) -> None:
        safe_event = redact_cache_payload(dict(event_obj))
        line_bytes = (json.dumps(safe_event, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(str(transcript_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line_bytes)
        finally:
            os.close(fd)
        try:
            os.chmod(transcript_path, 0o600)
        except OSError:
            pass

    def _persist_local_state(status_str: str = "running") -> None:
        safe_msgs = redact_cache_payload(messages)
        atomic_write_json(session_dir / "messages.json", safe_msgs, mode=0o600)
        atomic_write_json(
            session_dir / "session.json",
            {
                "session_slug": eff_session_slug,
                "agent_id": agent_id,
                "dir_slug": dir_slug,
                "project_root": str(project_root),
                "worktree_dir": str(wt_path),
                "active_model": model,
                "status": status_str,
                "steps": step_index,
                "message_count": len(messages),
                "started_at": started_at,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cache_stats": response_cache.stats(),
            },
            mode=0o600,
        )

    raw_ep = endpoint_url or base_url or ""
    base_url = normalize_colab_endpoint_url(raw_ep) if raw_ep else ""
    reg_rec = read_registered_endpoint(repo_dir=project_root, include_global=False) or {}
    api_key = (
        (str(api_key).strip() if api_key else "")
        or exec_env.get("PLATODOOM_TUNNEL_TOKEN")
        or exec_env.get("SANDCASTLE_VLLM_API_KEY")
        or exec_env.get("VLLM_API_KEY")
        or str(reg_rec.get("bearer_token") or reg_rec.get("auth_token") or "")
        or "EMPTY"
    )

    model_profile: Dict[str, Any] = {}
    protocol_profile: Dict[str, Any] = {}
    try:
        from platodoom.catalog import (
            compile_model_chat_messages,
            introspect_model_protocol,
            resolve_model_profile,
            split_thinking_and_content,
        )

        try:
            model_profile = resolve_model_profile(model)
        except Exception:
            model_profile = {}
        protocol_profile = introspect_model_protocol(model, allow_remote=False)
    except Exception:
        compile_model_chat_messages = None  # type: ignore[assignment]
        split_thinking_and_content = None  # type: ignore[assignment]

    eff_gpu_tier = str(reg_rec.get("gpu_type") or reg_rec.get("colab_gpu_type") or "").strip()
    if not eff_gpu_tier:
        for r_cand in (project_root / ".sandcastle" / "colab_gpu_routing.json", project_root / "colab_gpu_routing.json"):
            if r_cand.is_file():
                try:
                    r_json = json.loads(r_cand.read_text(encoding="utf-8"))
                    eff_gpu_tier = str(r_json.get("gpuType") or r_json.get("gpu_type") or "").strip()
                    if eff_gpu_tier:
                        break
                except Exception:
                    pass
    if model != UNIFIED_MODEL_ALIAS and model_profile:
        if eff_gpu_tier in ("A100", "A100-80GB", "H100", "G4", "v6e-1"):
            eff_max_model_len = int(model_profile.get("max_model_len_a100") or model_profile.get("max_model_len") or MAX_MODEL_LEN)
        else:
            eff_max_model_len = int(model_profile.get("max_model_len") or MAX_MODEL_LEN)
    else:
        eff_max_model_len = MAX_MODEL_LEN
    reg_max_model_len = reg_rec.get("max_model_len")
    if isinstance(reg_max_model_len, (int, float)) and int(reg_max_model_len) > 0:
        eff_max_model_len = min(eff_max_model_len, int(reg_max_model_len))

    served_model_id = model
    reg_norm_url = normalize_colab_endpoint_url(str(reg_rec.get("endpoint_url") or ""))
    reg_served = reg_rec.get("served_models")
    if base_url and reg_norm_url and base_url == reg_norm_url and isinstance(reg_served, list) and reg_served:
        served_list = [str(s) for s in reg_served if str(s).strip()]
        if served_list:
            if model in served_list:
                served_model_id = model
            else:
                accepted_ids = resolve_expected_model_identifiers(model)
                matched_id = next((s for s in served_list if s.strip().lower() in accepted_ids), None)
                if matched_id:
                    served_model_id = matched_id

    system_prompt = (
        f"You are **{agent_id}**, an autonomous coding agent in the Sandcastle multi-stage Blackboard fleet.\n"
        f"Working directory: {wt_path}\n"
        "Follow RULE[user_global] and LANGUAGE.md strictly. Use the provided tools (`run_command`, `read_file`, "
        "`write_file`, `edit_file`) and the `blackboard` CLI (`blackboard post-finding`, `blackboard post-request`, "
        "`blackboard claim`, `blackboard reply`, `blackboard read`, `blackboard done`) to complete your task and "
        "commit your required handoff manifest before finishing."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    step_index = 0
    final_response_text = ""
    blackboard_done_verified = False

    _append_local_transcript({"type": "user", "role": "user", "content": prompt, "model": model, "ts": time.time()})
    _persist_local_state(status_str="running")

    for _turn in range(1, max_turns + 1):
        messages = compact_messages_for_16k_window(messages, max_model_len=eff_max_model_len)
        if callable(compile_model_chat_messages) and protocol_profile:
            caller_thinking: Optional[bool] = False if protocol_profile.get("supports_enable_thinking_kwarg") else None
            compiled_msgs, compiled_sampling = compile_model_chat_messages(
                messages=messages,
                tools=WORKSPACE_TOOLS_SCHEMA,
                protocol_profile=protocol_profile,
                enable_thinking=caller_thinking,
                output_format="openai_http",
            )
            req_sampling: Dict[str, Any] = {}
            for sk in ("temperature", "top_p", "top_k", "stop", "chat_template_kwargs"):
                if sk in compiled_sampling and compiled_sampling[sk] is not None:
                    req_sampling[sk] = compiled_sampling[sk]
            eff_temp = float(req_sampling.get("temperature", 0.0))
            eff_max_tokens = min(4096, int(compiled_sampling.get("max_tokens") or 4096))
        else:
            compiled_msgs = messages
            req_sampling = {"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
            eff_temp = 0.0
            eff_max_tokens = 4096

        req_payload: Dict[str, Any] = {
            "model": served_model_id,
            "messages": compiled_msgs,
            "tools": WORKSPACE_TOOLS_SCHEMA,
            "max_tokens": eff_max_tokens,
            "stream": False,
            **req_sampling,
        }

        ctk_val = req_sampling.get("chat_template_kwargs") if isinstance(req_sampling.get("chat_template_kwargs"), dict) else None
        extra_cache_params: Dict[str, Any] = {}
        if ctk_val is not None:
            extra_cache_params["chat_template_kwargs"] = ctk_val
        stop_val = req_sampling.get("stop")
        norm_stop = [stop_val] if isinstance(stop_val, str) else ([str(s) for s in stop_val] if isinstance(stop_val, (list, tuple)) else None)

        cache_key = compute_response_cache_key(
            model=model,
            messages=compiled_msgs,
            tools=WORKSPACE_TOOLS_SCHEMA,
            temperature=eff_temp,
            top_p=req_sampling.get("top_p"),
            top_k=req_sampling.get("top_k"),
            stop=norm_stop,
            enable_thinking=ctk_val.get("enable_thinking") if ctk_val else None,
            chat_template_kwargs=ctk_val,
            max_tokens=eff_max_tokens,
            extra_params=extra_cache_params or None,
        )
        resp_data: Optional[Dict[str, Any]] = None
        cache_hit = False
        if use_cache and not force_refresh:
            cached_resp = response_cache.get(cache_key)
            if cached_resp is not None:
                resp_data = cached_resp
                cache_hit = True

        # Bounded transient tunnel reconnect retry (FR-4.3: max_retries = 2)
        if resp_data is None:
            if not base_url:
                raise RuntimeError("No endpoint_url configured for worktree agent loop on cache miss.")
            if base_url == reg_norm_url and isinstance(reg_served, list) and reg_served:
                accepted_ids = resolve_expected_model_identifiers(model)
                if not any(str(s).strip().lower() in accepted_ids for s in reg_served):
                    allow_reprov_mismatch = (
                        bool(auto_provision)
                        and exec_env.get("PLATODOOM_NO_COLAB") != "1"
                        and exec_env.get("SANDCASTLE_NO_COLAB") != "1"
                    )
                    if allow_reprov_mismatch:
                        refreshed = ensure_colab_serving_endpoint(
                            repo_dir=project_root,
                            auto_provision=True,
                            include_global=False,
                            headless_launch=exec_env.get("SANDCASTLE_DISABLE_BROWSER") != "1",
                            model_alias=model,
                        )
                        if refreshed and refreshed.get("endpoint_url"):
                            base_url = normalize_colab_endpoint_url(str(refreshed["endpoint_url"]))
                            ref_served = refreshed.get("served_models") or []
                            matched_ref = next((str(s) for s in ref_served if str(s).strip().lower() in accepted_ids), None)
                            if matched_ref:
                                served_model_id = matched_ref
                            req_payload["model"] = served_model_id
                    else:
                        raise RuntimeError(
                            f"Endpoint {base_url} is serving {reg_served} instead of requested model '{model}'."
                        )

            max_tunnel_retries = 2
            for attempt in range(max_tunnel_retries + 1):
                req = urllib.request.Request(
                    f"{base_url}/chat/completions",
                    data=json.dumps(req_payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "ngrok-skip-browser-warning": "true",
                    },
                    method="POST",
                )
                try:
                    with _open_worktree_http_request(req, timeout=300) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                    is_retryable_http = isinstance(exc, urllib.error.HTTPError) and exc.code in (502, 503, 504)
                    is_network_err = not isinstance(exc, urllib.error.HTTPError)
                    allow_reprovision = (
                        bool(auto_provision)
                        and exec_env.get("PLATODOOM_NO_COLAB") != "1"
                        and exec_env.get("SANDCASTLE_NO_COLAB") != "1"
                    )
                    if allow_reprovision and attempt < max_tunnel_retries and (is_retryable_http or is_network_err):
                        time.sleep(0.2 * (attempt + 1))
                        try:
                            refreshed = ensure_colab_serving_endpoint(
                                repo_dir=project_root,
                                auto_provision=True,
                                include_global=False,
                                headless_launch=exec_env.get("SANDCASTLE_DISABLE_BROWSER") != "1",
                                model_alias=model,
                            )
                            if refreshed and refreshed.get("endpoint_url"):
                                base_url = normalize_colab_endpoint_url(str(refreshed["endpoint_url"]))
                                continue
                        except Exception:
                            pass
                    if isinstance(exc, urllib.error.HTTPError):
                        try:
                            err_body = exc.read().decode("utf-8", errors="replace").strip()
                        except Exception:
                            err_body = "<unreadable body>"
                        msg_summary = [
                            f"{m.get('role')}(len={len(str(m.get('content') or ''))},tc={len(m.get('tool_calls') or [])})"
                            for m in compiled_msgs
                        ]
                        exc.msg = (
                            f"{exc.msg} | body={err_body} | "
                            f"model={served_model_id} max_tokens={eff_max_tokens} est_tokens={estimate_message_tokens(compiled_msgs)} "
                            f"msgs={msg_summary}"
                        )
                        raise exc
                    raise

            if resp_data is not None and use_cache and resp_data.get("choices"):
                response_cache.put(
                    cache_key,
                    resp_data,
                    request_fingerprint_meta={"model": model, "temperature": eff_temp},
                    temperature=eff_temp,
                )

        if resp_data is None:
            raise RuntimeError(f"No response payload received from {base_url}/chat/completions")

        choices = resp_data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices returned from {base_url}/chat/completions")

        assistant_msg = choices[0].get("message") or {}
        reasoning_text = str(assistant_msg.get("reasoning_content") or assistant_msg.get("reasoning") or "").strip() or None
        forces_think = bool(protocol_profile.get("forces_think_prefix", False))
        if not reasoning_text and callable(split_thinking_and_content):
            think_extracted, _ = split_thinking_and_content(
                str(assistant_msg.get("content") or ""),
                forces_think_prefix=forces_think,
            )
            if think_extracted:
                reasoning_text = think_extracted
        cleaned_text, tool_calls = parse_assistant_tool_calls(assistant_msg, forces_think_prefix=forces_think)

        if cleaned_text:
            step_index += 1
            final_response_text = cleaned_text
            emit_stream_json_line(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": step_index,
                        "state": "ACTIVE",
                        "step_type": "agent_response",
                        "text_delta": cleaned_text,
                        "reasoning_content": reasoning_text,
                        "cache_hit": cache_hit,
                    },
                },
                stream_out,
            )

        if tool_calls:
            assistant_entry = {
                "role": "assistant",
                "content": cleaned_text,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"], ensure_ascii=False)
                                if isinstance(tc.get("arguments"), (dict, list))
                                else str(tc.get("arguments") or "{}")
                            ),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_entry)
            _append_local_transcript(
                {
                    "type": "assistant",
                    **assistant_entry,
                    "reasoning_content": reasoning_text,
                    "cache_hit": cache_hit,
                    "ts": time.time(),
                }
            )
            for tc in tool_calls:
                step_index += 1
                emit_stream_json_line(
                    {
                        "event": "step_update",
                        "step_update": {
                            "step_index": step_index,
                            "state": "ACTIVE",
                            "step_type": "tool",
                            "tool_name": tc["name"],
                            "tool_info": tc["arguments"],
                        },
                    },
                    stream_out,
                )
                if tc.get("malformed_error"):
                    tool_output = f"ERROR: {tc['malformed_error']}"
                else:
                    tool_output = execute_worktree_tool(
                        tool_name=tc["name"],
                        arguments=tc["arguments"],
                        worktree_dir=wt_path,
                        env=exec_env,
                    )
                    cmd_str = str(tc["arguments"].get("CommandLine", ""))
                    if (
                        tc["name"] == "run_command"
                        and "blackboard" in cmd_str
                        and "done" in cmd_str
                        and "Exit code: 0" in tool_output
                    ):
                        blackboard_done_verified = True

                tool_entry = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": tool_output,
                }
                messages.append(tool_entry)
                _append_local_transcript({"type": "tool", **tool_entry, "ts": time.time()})
            _persist_local_state(status_str="running")
            continue

        # Model stopped issuing tool calls — enforce Blackboard `done` Obligation Gate if applicable (Section 4.4)
        messages.append({"role": "assistant", "content": cleaned_text})
        _append_local_transcript(
            {
                "type": "assistant",
                "role": "assistant",
                "content": cleaned_text,
                "reasoning_content": reasoning_text,
                "cache_hit": cache_hit,
                "ts": time.time(),
            }
        )
        _persist_local_state(status_str="running")
        has_blackboard = bool(
            enforce_blackboard_done
            and exec_env.get("SANDCASTLE_BLACKBOARD_URL")
            and exec_env.get("SANDCASTLE_AGENT_ID")
            and exec_env.get("SANDCASTLE_AGENT_ID") not in ("branch-merger", "merger")
        )
        if has_blackboard and not blackboard_done_verified:
            bb_out = execute_worktree_tool(
                tool_name="run_command",
                arguments={"CommandLine": "blackboard done", "toolSummary": "Blackboard obligation gate check"},
                worktree_dir=wt_path,
                env=exec_env,
            )
            if "Exit code: 0" in bb_out:
                blackboard_done_verified = True
                break
            gate_msg = {
                "role": "user",
                "content": (
                    "Blackboard Obligation Gate rejected `blackboard done`:\n"
                    f"{bb_out}\n"
                    "Please read/reply to any unread messages or open requests targeting you, commit your handoff manifest, and call `blackboard done`."
                ),
            }
            messages.append(gate_msg)
            _append_local_transcript({"type": "user", **gate_msg, "ts": time.time()})
            _persist_local_state(status_str="running")
            continue
        break

    _persist_local_state(status_str="completed")
    emit_stream_json_line(
        {
            "event": "result",
            "result": {
                "response": final_response_text or f"Completed {agent_id} execution.",
            },
        },
        stream_out,
    )
    return {
        "ok": True,
        "status": "completed",
        "agent_id": agent_id,
        "dir_slug": dir_slug,
        "session_slug": eff_session_slug,
        "session_dir": str(session_dir),
        "project_root": str(project_root),
        "steps": step_index,
        "response": final_response_text,
        "final_message": final_response_text,
        "blackboard_done": blackboard_done_verified,
        "cache_stats": response_cache.stats(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    from scripts.omp_qwen_adapter import main as _omp_main

    return _omp_main(argv)


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
