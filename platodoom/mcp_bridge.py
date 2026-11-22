from __future__ import annotations

import ipaddress
import json
import os
import select
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from platodoom.discovery import discover_workspace_capabilities

MAX_MCP_OUTPUT_BYTES: int = 24576  # 24 KB payload cap (FR-4.3)

_SENSITIVE_MCP_ENV_KEYS = frozenset(
    {
        "HF_TOKEN",
        "PLATODOOM_HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NGROK_AUTHTOKEN",
        "PLATODOOM_TUNNEL_TOKEN",
        "SANDCASTLE_VLLM_API_KEY",
        "VLLM_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    }
)

_BLOCKED_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata",
        "fd00:ec2::254",
    }
)


def _validate_mcp_http_url(raw_url: str) -> Optional[str]:
    """Validates MCP HTTP URL scheme and blocks cloud metadata SSRF hosts."""
    parsed = urllib.parse.urlparse(str(raw_url or "").strip())
    if parsed.scheme.lower() not in ("http", "https"):
        return f"Blocked unsafe MCP URL scheme '{parsed.scheme or '<empty>'}'; only http:// and https:// are permitted."
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "MCP HTTP URL is missing a valid hostname."
    if host in _BLOCKED_METADATA_HOSTS:
        return f"Blocked access to metadata host '{host}'."
    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_link_local:
            return f"Blocked link-local IP address '{host}' in MCP HTTP URL."
    except ValueError:
        pass
    return None


def _build_sanitized_mcp_env(srv_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Scrubs host API tokens/secrets before spawning a stdio MCP child process."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in _SENSITIVE_MCP_ENV_KEYS
    }
    for k, v in (srv_cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    return env


def _check_project_stdio_trust(
    srv_cfg: Dict[str, Any],
    *,
    require_project_trust: bool = False,
    interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """Requires explicit trust before spawning stdio commands defined inside a project repository."""
    if not require_project_trust:
        return None
    if str(srv_cfg.get("scope") or "").lower() != "project":
        return None
    if str(os.environ.get("PLATODOOM_TRUST_PROJECT_MCP") or "").strip().lower() in ("1", "true", "yes"):
        return None
    cmd = str(srv_cfg.get("command") or "")
    args = " ".join(str(a) for a in (srv_cfg.get("args") or []))
    src = str(srv_cfg.get("source_path") or "project config")
    if interactive_confirm_fn is not None:
        try:
            if bool(
                interactive_confirm_fn(
                    f"Allow project MCP server '{srv_cfg.get('name')}' ({src}) to execute `{cmd} {args}`.strip()?"
                )
            ):
                return None
        except Exception:
            pass
    return (
        f"[SECURITY BLOCKED: Untrusted project-scoped stdio MCP server '{srv_cfg.get('name')}' "
        f"({src}) requires explicit confirmation or PLATODOOM_TRUST_PROJECT_MCP=1 before executing '{cmd}'.]"
    )


def _cap_mcp_payload(data: Any, max_bytes: int = MAX_MCP_OUTPUT_BYTES) -> Any:
    try:
        serialized = json.dumps(data, ensure_ascii=False)
    except Exception:
        serialized = str(data)
    raw_bytes = serialized.encode("utf-8")
    if len(raw_bytes) <= max_bytes:
        return data
    truncated_str = raw_bytes[: max_bytes - 256].decode("utf-8", errors="ignore")
    capped_summary = {
        "truncated": True,
        "max_bytes": max_bytes,
        "preview": truncated_str + f"\n...[MCP OUTPUT CAPPED AT {max_bytes} BYTES]...",
    }
    if isinstance(data, dict) and "ok" in data:
        return {
            "ok": bool(data.get("ok")),
            "server": data.get("server"),
            "tool": data.get("tool"),
            "transport": data.get("transport"),
            "truncated": True,
            "max_bytes": max_bytes,
            "result": capped_summary,
        }
    return capped_summary


def _resolve_server_config(
    server_name: str,
    capabilities: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    caps = capabilities if isinstance(capabilities, dict) else discover_workspace_capabilities(Path.cwd())

    def _find_in(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        servers = manifest.get("mcp_servers") or {}
        if server_name in servers:
            return servers[server_name]
        s_lower = (server_name or "").strip().lower()
        for k, v in servers.items():
            if k.lower() == s_lower:
                return v
        return None

    found = _find_in(caps)
    if found is None and not caps.get("include_global", False):
        ws_dir = caps.get("workspace_dir") or Path.cwd()
        home_dir = caps.get("home_dir")
        global_caps = discover_workspace_capabilities(ws_dir, home_dir=home_dir, include_global=True)
        found = _find_in(global_caps)
    return found


def _load_cached_schemas_for_server(srv_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    schema_dir_str = srv_cfg.get("schema_dir")
    if not schema_dir_str:
        return []
    schema_dir = Path(schema_dir_str)
    if not schema_dir.exists() or not schema_dir.is_dir():
        return []

    tools: List[Dict[str, Any]] = []
    try:
        for item in sorted(schema_dir.iterdir(), key=lambda p: p.name):
            if not item.is_file() or item.suffix.lower() != ".json":
                continue
            try:
                raw = json.loads(item.read_text(encoding="utf-8", errors="replace"))
                if isinstance(raw, dict):
                    t_name = str(raw.get("name") or item.stem)
                    t_desc = str(raw.get("description") or "")
                    input_schema = (
                        raw.get("inputSchema")
                        or raw.get("parameters")
                        or {"type": "object", "properties": {}}
                    )
                    tools.append(
                        {
                            "name": t_name,
                            "description": t_desc,
                            "inputSchema": input_schema,
                        }
                    )
            except Exception:
                continue
    except OSError:
        pass
    return tools


def _execute_stdio_jsonrpc(
    srv_cfg: Dict[str, Any],
    method: str,
    params: Dict[str, Any],
    timeout_sec: float = 15.0,
    *,
    require_project_trust: bool = False,
    interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Any]:
    """
    Spawns an MCP stdio server and executes the JSON-RPC 2.0 handshake:
      1. initialize (id=1)
      2. notifications/initialized
      3. <method> (id=2, e.g. tools/list or tools/call)
    Reads newline-delimited JSON-RPC responses up to timeout_sec and terminates cleanly.
    """
    command = srv_cfg.get("command")
    if not command:
        return {
            "ok": False,
            "error": f"MCP server '{srv_cfg.get('name')}' has no stdio 'command' configured.",
        }

    trust_err = _check_project_stdio_trust(
        srv_cfg,
        require_project_trust=require_project_trust,
        interactive_confirm_fn=interactive_confirm_fn,
    )
    if trust_err:
        return {
            "ok": False,
            "error": trust_err,
            "server": srv_cfg.get("name"),
            "transport": "stdio",
        }

    args = [str(a) for a in (srv_cfg.get("args") or [])]
    env = _build_sanitized_mcp_env(srv_cfg)

    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "platodoom", "version": "1.0.1"},
        },
    }
    initialized_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    target_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": method,
        "params": params,
    }

    proc: Optional[subprocess.Popen[str]] = None
    try:
        proc = subprocess.Popen(
            [str(command), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None

        # Send initialize + notifications/initialized + target request
        payload_lines = (
            json.dumps(init_req) + "\n"
            + json.dumps(initialized_notif) + "\n"
            + json.dumps(target_req) + "\n"
        )
        proc.stdin.write(payload_lines)
        proc.stdin.flush()

        deadline = time.monotonic() + max(0.5, float(timeout_sec))
        fd = proc.stdout.fileno()
        buffer = ""
        init_result: Optional[Dict[str, Any]] = None

        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            ready, _, _ = select.select([fd], [], [], min(0.25, remaining))
            if not ready:
                if proc.poll() is not None and not buffer:
                    break
                continue
            chunk = os.read(fd, 65536).decode("utf-8", errors="replace")
            if not chunk:
                if proc.poll() is not None:
                    break
                continue
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id")
                if msg_id == 1:
                    init_result = msg.get("result")
                elif msg_id == 2:
                    if "error" in msg and msg["error"]:
                        return {
                            "ok": False,
                            "error": msg["error"],
                            "server": srv_cfg.get("name"),
                            "transport": "stdio",
                        }
                    return {
                        "ok": True,
                        "result": _cap_mcp_payload(msg.get("result")),
                        "initialize": init_result,
                        "server": srv_cfg.get("name"),
                        "transport": "stdio",
                    }

        stderr_out = ""
        if proc.stderr:
            try:
                r_err, _, _ = select.select([proc.stderr.fileno()], [], [], 0.05)
                if r_err:
                    stderr_out = os.read(proc.stderr.fileno(), 4096).decode("utf-8", errors="replace").strip()
            except Exception:
                pass
        return {
            "ok": False,
            "error": f"Timed out or EOF waiting for JSON-RPC response from stdio MCP server '{srv_cfg.get('name')}'. {stderr_out}".strip(),
            "server": srv_cfg.get("name"),
            "transport": "stdio",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Stdio MCP execution failed for '{srv_cfg.get('name')}': {exc}",
            "server": srv_cfg.get("name"),
            "transport": "stdio",
        }
    finally:
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def _execute_http_jsonrpc(
    srv_cfg: Dict[str, Any],
    method: str,
    params: Dict[str, Any],
    timeout_sec: float = 15.0,
) -> Dict[str, Any]:
    """
    Executes a JSON-RPC 2.0 request over HTTP/SSE transport (`url` / `serverUrl`).
    Supports both application/json responses and text/event-stream (`data: {...}`) responses.
    """
    url = srv_cfg.get("url")
    if not url:
        return {
            "ok": False,
            "error": f"MCP server '{srv_cfg.get('name')}' has no HTTP 'url' configured.",
        }
    url_err = _validate_mcp_http_url(str(url))
    if url_err:
        return {
            "ok": False,
            "error": url_err,
            "server": srv_cfg.get("name"),
            "transport": "http",
        }

    req_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    for k, v in (srv_cfg.get("headers") or {}).items():
        headers[str(k)] = str(v)

    body_bytes = json.dumps(req_payload).encode("utf-8")
    req = urllib.request.Request(str(url), data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_sec))) as resp:
            raw_text = resp.read(MAX_MCP_OUTPUT_BYTES * 2).decode("utf-8", errors="replace").strip()
            content_type = resp.headers.get("Content-Type", "")

            parsed_msg: Optional[Dict[str, Any]] = None
            if "text/event-stream" in content_type or raw_text.startswith("event:") or raw_text.startswith("data:"):
                for line in raw_text.splitlines():
                    s = line.strip()
                    if s.startswith("data:"):
                        data_str = s[len("data:") :].strip()
                        if data_str.startswith("{"):
                            try:
                                cand = json.loads(data_str)
                                if isinstance(cand, dict):
                                    parsed_msg = cand
                            except Exception:
                                pass
            if parsed_msg is None:
                parsed_msg = json.loads(raw_text)

            if isinstance(parsed_msg, dict):
                if "error" in parsed_msg and parsed_msg["error"]:
                    return {
                        "ok": False,
                        "error": parsed_msg["error"],
                        "server": srv_cfg.get("name"),
                        "transport": "http",
                    }
                result_val = parsed_msg.get("result", parsed_msg)
                return {
                    "ok": True,
                    "result": _cap_mcp_payload(result_val),
                    "server": srv_cfg.get("name"),
                    "transport": "http",
                }
            return {
                "ok": True,
                "result": _cap_mcp_payload(parsed_msg),
                "server": srv_cfg.get("name"),
                "transport": "http",
            }
    except urllib.error.HTTPError as http_err:
        err_body = ""
        try:
            err_body = http_err.read(2048).decode("utf-8", errors="replace")
        except Exception:
            pass
        return {
            "ok": False,
            "status_code": http_err.code,
            "error": f"HTTP {http_err.code} from MCP server '{srv_cfg.get('name')}': {err_body or http_err.reason}",
            "server": srv_cfg.get("name"),
            "transport": "http",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"HTTP MCP request failed for '{srv_cfg.get('name')}': {exc}",
            "server": srv_cfg.get("name"),
            "transport": "http",
        }


def list_mcp_tools(
    server_name: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 10.0,
    *,
    discovery_manifest: Optional[Dict[str, Any]] = None,
    require_project_trust: bool = False,
    interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Any]:
    """
    Lists available MCP tools across all discovered servers or for a specific server_name (FR-4.3, AC-2).
    Uses local schema cache when available (< 5ms) or live JSON-RPC `tools/list`.
    """
    eff_caps = capabilities if isinstance(capabilities, dict) else discovery_manifest
    caps = eff_caps if isinstance(eff_caps, dict) else discover_workspace_capabilities(Path.cwd())
    servers = caps.get("mcp_servers") or {}

    if not server_name:
        summary: Dict[str, Any] = {}
        for s_name, s_cfg in servers.items():
            cached = _load_cached_schemas_for_server(s_cfg)
            summary[s_name] = {
                "ecosystem": s_cfg.get("ecosystem"),
                "scope": s_cfg.get("scope"),
                "transport": s_cfg.get("transport"),
                "tool_count": len(cached) if cached else len(s_cfg.get("cached_tools") or []),
                "tools": [
                    {"name": t["name"], "description": t["description"][:120]}
                    for t in cached
                ]
                if cached
                else [{"name": t_n, "description": ""} for t_n in (s_cfg.get("cached_tools") or [])],
            }
        return {
            "ok": True,
            "server_count": len(summary),
            "servers": _cap_mcp_payload(summary),
        }

    srv_cfg = _resolve_server_config(server_name, caps)
    if srv_cfg is None:
        return {
            "ok": False,
            "server": server_name,
            "error": f"MCP server '{server_name}' not found. Discovered servers: {', '.join(sorted(servers.keys()))}",
            "tools": [],
        }

    # 1. Fast path: local schema cache directory if present (< 5ms without spawning subprocess)
    cached_tools = _load_cached_schemas_for_server(srv_cfg)
    if cached_tools:
        return {
            "ok": True,
            "server": srv_cfg["name"],
            "transport": "schema_cache",
            "tools": _cap_mcp_payload(cached_tools),
        }

    # 2. Live transport query (stdio or HTTP)
    if srv_cfg.get("url") and not srv_cfg.get("command"):
        rpc_res = _execute_http_jsonrpc(srv_cfg, "tools/list", {}, timeout_sec=timeout_sec)
    elif srv_cfg.get("command"):
        rpc_res = _execute_stdio_jsonrpc(
            srv_cfg,
            "tools/list",
            {},
            timeout_sec=timeout_sec,
            require_project_trust=require_project_trust,
            interactive_confirm_fn=interactive_confirm_fn,
        )
    else:
        rpc_res = {"ok": False, "error": "No live command or URL configured"}

    if rpc_res.get("ok"):
        res_obj = rpc_res.get("result")
        tools_list = res_obj.get("tools", res_obj) if isinstance(res_obj, dict) else res_obj
        return {
            "ok": True,
            "server": srv_cfg["name"],
            "transport": rpc_res.get("transport", srv_cfg.get("transport")),
            "tools": _cap_mcp_payload(tools_list),
        }

    return {
        "ok": False,
        "server": srv_cfg["name"],
        "error": rpc_res.get("error", "Failed to list MCP tools"),
        "tools": [],
    }


def inspect_mcp_tool(
    server_name: str,
    tool_name: str,
    capabilities: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 10.0,
    *,
    discovery_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns the full inputSchema and description for a specific MCP tool on `server_name` (FR-4.3).
    """
    eff_caps = capabilities if isinstance(capabilities, dict) else discovery_manifest
    caps = eff_caps if isinstance(eff_caps, dict) else discover_workspace_capabilities(Path.cwd())
    srv_cfg = _resolve_server_config(server_name, caps)
    if srv_cfg is None:
        return {
            "ok": False,
            "server": server_name,
            "tool": tool_name,
            "error": f"MCP server '{server_name}' not found.",
        }

    # Check cached schema files first
    cached_tools = _load_cached_schemas_for_server(srv_cfg)
    t_lower = (tool_name or "").strip().lower()
    for t in cached_tools:
        if str(t.get("name", "")).lower() == t_lower:
            return {
                "ok": True,
                "server": srv_cfg["name"],
                "tool": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {}),
            }

    # Query live tools/list
    listed = list_mcp_tools(server_name=srv_cfg["name"], capabilities=caps, timeout_sec=timeout_sec)
    if listed.get("ok") and isinstance(listed.get("tools"), list):
        for t in listed["tools"]:
            if isinstance(t, dict) and str(t.get("name", "")).lower() == t_lower:
                return {
                    "ok": True,
                    "server": srv_cfg["name"],
                    "tool": t.get("name", tool_name),
                    "description": t.get("description", ""),
                    "inputSchema": t.get("inputSchema") or t.get("parameters") or {},
                }

    return {
        "ok": False,
        "server": srv_cfg["name"],
        "tool": tool_name,
        "error": f"Tool '{tool_name}' not found on MCP server '{srv_cfg['name']}'.",
    }


def call_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 30.0,
    *,
    discovery_manifest: Optional[Dict[str, Any]] = None,
    require_project_trust: bool = False,
    interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Any]:
    """
    Executes an MCP tool (`tools/call`) on `server_name` over `stdio` JSON-RPC 2.0 or `HTTP/SSE`
    transport and returns a payload strictly capped at 24 KB (FR-4.3, AC-2).
    """
    eff_caps = capabilities if isinstance(capabilities, dict) else discovery_manifest
    caps = eff_caps if isinstance(eff_caps, dict) else discover_workspace_capabilities(Path.cwd())
    srv_cfg = _resolve_server_config(server_name, caps)
    if srv_cfg is None:
        servers = caps.get("mcp_servers") or {}
        return {
            "ok": False,
            "server": server_name,
            "tool": tool_name,
            "error": f"MCP server '{server_name}' not found. Available servers: {', '.join(sorted(servers.keys()))}",
        }

    call_params = {
        "name": tool_name,
        "arguments": arguments if isinstance(arguments, dict) else {},
    }

    if srv_cfg.get("url") and not srv_cfg.get("command"):
        res = _execute_http_jsonrpc(srv_cfg, "tools/call", call_params, timeout_sec=timeout_sec)
    elif srv_cfg.get("command"):
        res = _execute_stdio_jsonrpc(
            srv_cfg,
            "tools/call",
            call_params,
            timeout_sec=timeout_sec,
            require_project_trust=require_project_trust,
            interactive_confirm_fn=interactive_confirm_fn,
        )
    else:
        return {
            "ok": False,
            "server": srv_cfg["name"],
            "tool": tool_name,
            "error": f"MCP server '{srv_cfg['name']}' is schema-only (no executable command or HTTP URL configured).",
        }

    res["tool"] = tool_name
    return _cap_mcp_payload(res)
