from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Shallow YAML Frontmatter & Markdown Skill Parser (FR-4.1, FR-4.2)
# ---------------------------------------------------------------------------

_FRONTMATTER_KV_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$")


def parse_skill_markdown(
    skill_path_or_text: Path | str,
    shallow: bool = False,
) -> Dict[str, Any]:
    """
    Parses a SKILL.md / skill.md file (or raw markdown text) and returns:
      {
        "name": str,
        "description": str,
        "metadata": dict,
        "body": str (empty when shallow=True),
      }
    Supports single-line YAML, multi-line block scalars (`>-` / `|`), and nested metadata (`hermes`).
    """
    default_name = "skill"
    raw_head = ""
    full_text = ""

    is_file = False
    if isinstance(skill_path_or_text, Path):
        is_file = True
        file_path = skill_path_or_text
    elif isinstance(skill_path_or_text, str) and "\n" not in skill_path_or_text and len(skill_path_or_text) < 1024:
        candidate = Path(skill_path_or_text)
        if candidate.exists() and candidate.is_file():
            is_file = True
            file_path = candidate

    if is_file:
        default_name = (
            file_path.parent.name
            if file_path.name.lower() in ("skill.md", "instructions.md", "readme.md")
            else file_path.stem
        )
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                if shallow:
                    lines: List[str] = []
                    first = fh.readline()
                    lines.append(first)
                    if first.strip() == "---":
                        for _ in range(35):
                            nxt = fh.readline()
                            if not nxt:
                                break
                            lines.append(nxt)
                            if nxt.strip() == "---":
                                break
                    else:
                        for _ in range(12):
                            nxt = fh.readline()
                            if not nxt:
                                break
                            lines.append(nxt)
                    raw_head = "".join(lines)
                else:
                    full_text = fh.read()
                    raw_head = full_text[:4096]
        except Exception:
            return {"name": default_name, "description": "", "metadata": {}, "body": ""}
    else:
        full_text = str(skill_path_or_text)
        raw_head = full_text[:4096]

    name = default_name
    description = ""
    metadata: Dict[str, Any] = {}
    body = full_text

    if raw_head.startswith("---"):
        parts = (full_text if not shallow else raw_head).split("---", 2)
        if len(parts) >= 3:
            yaml_block = parts[1]
            if not shallow:
                body = parts[2].lstrip("\r\n")
            try:
                parsed_yaml = yaml.safe_load(yaml_block)
                if isinstance(parsed_yaml, dict):
                    metadata = parsed_yaml
                    raw_name = parsed_yaml.get("name")
                    if raw_name:
                        name = str(raw_name).strip()
                    raw_desc = parsed_yaml.get("description")
                    if raw_desc:
                        description = " ".join(str(raw_desc).split()).strip()
            except Exception:
                # Fallback regex line parser if YAML has unquoted colons
                for line in yaml_block.splitlines():
                    m = _FRONTMATTER_KV_RE.match(line.strip())
                    if not m:
                        continue
                    k, v = m.group(1).lower(), m.group(2).strip().strip("\"'")
                    if k == "name" and v and v not in (">-", "|"):
                        name = v
                    elif k == "description" and v and v not in (">-", "|"):
                        description = v

    if not description:
        # Extract first non-empty, non-heading paragraph line as fallback description
        source_for_desc = body if not shallow else raw_head
        in_fm = False
        for idx, line in enumerate(source_for_desc.splitlines()):
            s = line.strip()
            if idx == 0 and s == "---":
                in_fm = True
                continue
            if in_fm:
                if s == "---":
                    in_fm = False
                continue
            if not s or s.startswith("#") or s.startswith("```"):
                continue
            description = " ".join(s.split())[:200]
            break

    return {
        "name": str(name).strip() or default_name,
        "description": str(description).strip(),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "body": "" if shallow else body,
    }


# ---------------------------------------------------------------------------
# Multi-Ecosystem Auto-Discovery Engine (FR-4.1, FR-4.2, FR-4.3, AC-2)
# ---------------------------------------------------------------------------

def _find_git_root(start_dir: Path) -> Optional[Path]:
    curr = start_dir.resolve()
    for candidate in (curr, *curr.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _scan_skills_in_dir(
    base_dir: Path,
    ecosystem: str,
    scope: str,
    skills_out: Dict[str, Dict[str, Any]],
    commands_out: Dict[str, Dict[str, Any]],
    is_command_dir: bool = False,
    allow_nested_category: bool = False,
    shadowed_paths_out: Optional[List[str]] = None,
) -> None:
    if not base_dir.exists() or not base_dir.is_dir():
        return

    try:
        entries = sorted(base_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return

    def _register_skill(norm_key: str, record: Dict[str, Any]) -> None:
        if norm_key not in skills_out:
            record.setdefault("shadowed_paths", [])
            skills_out[norm_key] = record
        else:
            existing = skills_out[norm_key]
            sp = record.get("file_path") or record.get("path")
            if sp and sp != existing.get("file_path"):
                existing.setdefault("shadowed_paths", [])
                if sp not in existing["shadowed_paths"]:
                    existing["shadowed_paths"].append(sp)
                if shadowed_paths_out is not None and sp not in shadowed_paths_out:
                    shadowed_paths_out.append(sp)

    for entry in entries:
        name_head = entry.name
        if name_head.startswith("."):
            continue

        if is_command_dir and entry.is_file() and entry.suffix.lower() in (".md", ".mdc"):
            parsed = parse_skill_markdown(entry, shallow=True)
            cmd_name = parsed["name"] if parsed["name"] != "skill" else entry.stem
            norm_key = cmd_name.strip().lower()
            resolved_p = str(entry.resolve())
            record = {
                "name": cmd_name,
                "ecosystem": ecosystem,
                "scope": scope,
                "description": parsed["description"] or f"Slash command /{cmd_name} ({ecosystem})",
                "file_path": resolved_p,
                "path": resolved_p,
                "shadowed_paths": [],
            }
            if norm_key not in commands_out:
                commands_out[norm_key] = record
            _register_skill(norm_key, record)
            continue

        if entry.is_file() and entry.name.lower() in ("skill.md",) or (
            entry.is_file() and entry.suffix.lower() == ".mdc"
        ):
            parsed = parse_skill_markdown(entry, shallow=True)
            skill_name = parsed["name"]
            norm_key = skill_name.strip().lower()
            resolved_p = str(entry.resolve())
            _register_skill(
                norm_key,
                {
                    "name": skill_name,
                    "ecosystem": ecosystem,
                    "scope": scope,
                    "description": parsed["description"],
                    "file_path": resolved_p,
                    "path": resolved_p,
                    "shadowed_paths": [],
                },
            )
            continue

        if not entry.is_dir():
            continue

        # Direct child skill folder: <base_dir>/<skill_dir>/SKILL.md or skill.md
        skill_file: Optional[Path] = None
        for cand_name in ("SKILL.md", "skill.md"):
            cand = entry / cand_name
            if cand.exists() and cand.is_file():
                skill_file = cand
                break

        if skill_file is not None:
            parsed = parse_skill_markdown(skill_file, shallow=True)
            skill_name = parsed["name"]
            norm_key = skill_name.strip().lower()
            resolved_p = str(skill_file.resolve())
            _register_skill(
                norm_key,
                {
                    "name": skill_name,
                    "ecosystem": ecosystem,
                    "scope": scope,
                    "description": parsed["description"],
                    "file_path": resolved_p,
                    "path": resolved_p,
                    "shadowed_paths": [],
                },
            )
            continue

        # Nested category folder (e.g., Hermes ~/.hermes/skills/<category>/<skill>/SKILL.md)
        if allow_nested_category:
            try:
                sub_entries = sorted(entry.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            for sub in sub_entries:
                if sub.name.startswith(".") or not sub.is_dir():
                    continue
                sub_skill_file: Optional[Path] = None
                for cand_name in ("SKILL.md", "skill.md"):
                    cand = sub / cand_name
                    if cand.exists() and cand.is_file():
                        sub_skill_file = cand
                        break
                if sub_skill_file is not None:
                    parsed = parse_skill_markdown(sub_skill_file, shallow=True)
                    skill_name = parsed["name"]
                    norm_key = skill_name.strip().lower()
                    resolved_p = str(sub_skill_file.resolve())
                    _register_skill(
                        norm_key,
                        {
                            "name": skill_name,
                            "ecosystem": ecosystem,
                            "scope": scope,
                            "category": entry.name,
                            "description": parsed["description"],
                            "file_path": resolved_p,
                            "path": resolved_p,
                            "shadowed_paths": [],
                        },
                    )


def _scan_mcp_json_file(
    cfg_path: Path,
    ecosystem: str,
    scope: str,
    mcp_servers_out: Dict[str, Dict[str, Any]],
) -> None:
    if not cfg_path.exists() or not cfg_path.is_file():
        return
    try:
        raw_text = cfg_path.read_text(encoding="utf-8", errors="replace")
        if cfg_path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(raw_text)
        else:
            data = json.loads(raw_text)
    except Exception:
        return

    if not isinstance(data, dict):
        return

    servers_block = (
        data.get("mcpServers")
        or data.get("mcp_servers")
        or data.get("servers")
        or (data.get("mcp") or {}).get("servers")
    )
    if not isinstance(servers_block, dict):
        return

    for srv_name, srv_cfg in servers_block.items():
        if not isinstance(srv_cfg, dict):
            continue
        norm_name = str(srv_name).strip()
        if not norm_name or norm_name in mcp_servers_out:
            continue
        url = srv_cfg.get("url") or srv_cfg.get("serverUrl") or srv_cfg.get("httpUrl")
        command = srv_cfg.get("command")
        args = srv_cfg.get("args") if isinstance(srv_cfg.get("args"), list) else []
        env_map = srv_cfg.get("env") if isinstance(srv_cfg.get("env"), dict) else {}
        headers = srv_cfg.get("headers") if isinstance(srv_cfg.get("headers"), dict) else {}
        transport = "http" if url and not command else "stdio"

        mcp_servers_out[norm_name] = {
            "name": norm_name,
            "ecosystem": ecosystem,
            "scope": scope,
            "transport": transport,
            "command": command,
            "args": [str(a) for a in args],
            "env": {str(k): str(v) for k, v in env_map.items()},
            "url": str(url) if url else None,
            "headers": {str(k): str(v) for k, v in headers.items()},
            "source_path": str(cfg_path.resolve()),
            "schema_dir": None,
            "cached_tools": [],
        }


def _scan_antigravity_mcp_schema_dir(
    mcp_root: Path,
    scope: str,
    mcp_servers_out: Dict[str, Dict[str, Any]],
) -> None:
    if not mcp_root.exists() or not mcp_root.is_dir():
        return
    try:
        server_dirs = sorted(mcp_root.iterdir(), key=lambda p: p.name)
    except OSError:
        return

    for srv_dir in server_dirs:
        if srv_dir.name.startswith(".") or not srv_dir.is_dir():
            continue
        srv_name = srv_dir.name
        tool_names: List[str] = []
        try:
            for item in sorted(srv_dir.iterdir(), key=lambda p: p.name):
                if item.is_file() and item.suffix.lower() == ".json":
                    tool_names.append(item.stem)
        except OSError:
            pass

        if srv_name in mcp_servers_out:
            mcp_servers_out[srv_name]["schema_dir"] = str(srv_dir.resolve())
            if tool_names and not mcp_servers_out[srv_name].get("cached_tools"):
                mcp_servers_out[srv_name]["cached_tools"] = tool_names
        elif tool_names:
            mcp_servers_out[srv_name] = {
                "name": srv_name,
                "ecosystem": "gemini",
                "scope": scope,
                "transport": "schema_cache",
                "command": None,
                "args": [],
                "env": {},
                "url": None,
                "headers": {},
                "source_path": str(srv_dir.resolve()),
                "schema_dir": str(srv_dir.resolve()),
                "cached_tools": tool_names,
            }


def discover_workspace_capabilities(
    workspace_dir: Path | str,
    home_dir: Optional[Path | str] = None,
    extra_skill_paths: Optional[List[str]] = None,
    extra_mcp_paths: Optional[List[str]] = None,
    include_global: bool = False,
    *,
    extra_skill_dirs: Optional[List[str]] = None,
    extra_mcp_configs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Discovers, deduplicates, and normalizes project (and optionally user-global) instructions,
    Skills, slash commands, and MCP server configurations in <= 300ms.
    By default (`include_global=False`), only workspace/project-scoped definitions are loaded so
    unrelated global directories (`~/.hermes`, `~/.cursor`, `~/.gemini`) never bloat session prompts.
    """
    t0 = time.perf_counter()
    ws_path = Path(workspace_dir).expanduser().resolve()
    home_path = Path(home_dir).expanduser().resolve() if home_dir else Path.home().resolve()
    git_root = _find_git_root(ws_path) or ws_path

    # Ancestor chain from workspace_dir up to git_root
    ancestor_dirs: List[Path] = [ws_path]
    curr = ws_path
    while curr != git_root and curr.parent != curr:
        curr = curr.parent
        ancestor_dirs.append(curr)
        if curr == git_root:
            break

    # 1. Discover Instruction Files (project-local first, then user-global)
    instructions: List[Dict[str, Any]] = []
    seen_instruction_paths: set[str] = set()

    project_instruction_relpaths: List[Tuple[str, str]] = [
        ("LANGUAGE.md", "universal"),
        ("AGENTS.md", "universal"),
        ("CLAUDE.md", "claude"),
        ("GEMINI.md", "gemini"),
        (".platodoom/instructions.md", "platodoom"),
        (".cursorrules", "cursor"),
        (".github/copilot-instructions.md", "github"),
    ]

    for anc in ancestor_dirs:
        for rel_file, eco in project_instruction_relpaths:
            cand = anc / rel_file
            if cand.exists() and cand.is_file():
                resolved_str = str(cand.resolve())
                if resolved_str not in seen_instruction_paths:
                    seen_instruction_paths.add(resolved_str)
                    parsed = parse_skill_markdown(cand, shallow=True)
                    instructions.append(
                        {
                            "name": cand.name,
                            "ecosystem": eco,
                            "scope": "project",
                            "path": resolved_str,
                            "file_path": resolved_str,
                            "summary": parsed["description"] or f"Project instruction file ({cand.name})",
                        }
                    )
        cursor_rules_dir = anc / ".cursor" / "rules"
        if cursor_rules_dir.exists() and cursor_rules_dir.is_dir():
            try:
                for mdc_file in sorted(cursor_rules_dir.iterdir(), key=lambda p: p.name):
                    if mdc_file.is_file() and mdc_file.suffix.lower() in (".mdc", ".md"):
                        resolved_str = str(mdc_file.resolve())
                        if resolved_str not in seen_instruction_paths:
                            seen_instruction_paths.add(resolved_str)
                            parsed = parse_skill_markdown(mdc_file, shallow=True)
                            instructions.append(
                                {
                                    "name": mdc_file.name,
                                    "ecosystem": "cursor",
                                    "scope": "project",
                                    "path": resolved_str,
                                    "file_path": resolved_str,
                                    "summary": parsed["description"] or f"Cursor rule ({mdc_file.name})",
                                }
                            )
            except OSError:
                pass

    if include_global:
        global_instruction_paths: List[Tuple[Path, str]] = [
            (home_path / ".platodoom" / "instructions.md", "platodoom"),
            (home_path / ".claude" / "CLAUDE.md", "claude"),
            (home_path / ".gemini" / "GEMINI.md", "gemini"),
            (home_path / ".hermes" / "SOUL.md", "hermes"),
        ]
        for g_cand, eco in global_instruction_paths:
            if g_cand.exists() and g_cand.is_file():
                resolved_str = str(g_cand.resolve())
                if resolved_str not in seen_instruction_paths:
                    seen_instruction_paths.add(resolved_str)
                    parsed = parse_skill_markdown(g_cand, shallow=True)
                    instructions.append(
                        {
                            "name": g_cand.name,
                            "ecosystem": eco,
                            "scope": "user",
                            "path": resolved_str,
                            "file_path": resolved_str,
                            "summary": parsed["description"] or f"Global {eco} instructions ({g_cand.name})",
                        }
                    )

    # 2. Discover Skills & Slash Commands (project-local first so project overrides global!)
    skills_by_key: Dict[str, Dict[str, Any]] = {}
    commands_by_key: Dict[str, Dict[str, Any]] = {}
    shadowed_paths: List[str] = []

    eff_extra_skills = list(extra_skill_paths or []) + list(extra_skill_dirs or [])
    if eff_extra_skills:
        for extra_p in eff_extra_skills:
            ep = Path(extra_p).expanduser()
            if not ep.is_absolute():
                ep = ws_path / ep
            _scan_skills_in_dir(
                ep,
                "platodoom",
                "project",
                skills_by_key,
                commands_by_key,
                allow_nested_category=True,
                shadowed_paths_out=shadowed_paths,
            )

    project_skill_locations: List[Tuple[str, str, bool, bool]] = [
        (".platodoom/skills", "platodoom", False, True),
        (".platodoom/commands", "platodoom", True, False),
        (".agents/skills", "agents", False, True),
        (".claude/skills", "claude", False, True),
        (".claude/commands", "claude", True, False),
        (".gemini/skills", "gemini", False, True),
        (".codex/skills", "codex", False, True),
        (".hermes/skills", "hermes", False, True),
        (".cursor/skills", "cursor", False, True),
    ]
    for anc in ancestor_dirs:
        for rel_dir, eco, is_cmd, allow_nested in project_skill_locations:
            _scan_skills_in_dir(
                anc / rel_dir,
                ecosystem=eco,
                scope="project",
                skills_out=skills_by_key,
                commands_out=commands_by_key,
                is_command_dir=is_cmd,
                allow_nested_category=allow_nested,
                shadowed_paths_out=shadowed_paths,
            )

    if include_global:
        global_skill_locations: List[Tuple[Path, str, bool, bool]] = [
            (home_path / ".platodoom" / "skills", "platodoom", False, True),
            (home_path / ".platodoom" / "commands", "platodoom", True, False),
            (home_path / ".agents" / "skills", "agents", False, True),
            (home_path / ".claude" / "skills", "claude", False, True),
            (home_path / ".claude" / "commands", "claude", True, False),
            (home_path / ".gemini" / "antigravity-cli" / "skills", "gemini", False, False),
            (home_path / ".gemini" / "antigravity-cli" / "builtin" / "skills", "gemini", False, False),
            (home_path / ".gemini" / "skills", "gemini", False, False),
            (home_path / ".codex" / "skills", "codex", False, True),
            (home_path / ".hermes" / "skills", "hermes", False, True),
            (home_path / ".cursor" / "skills", "cursor", False, False),
            (home_path / ".cursor" / "skills-cursor", "cursor", False, False),
        ]
        for g_dir, eco, is_cmd, allow_nested in global_skill_locations:
            _scan_skills_in_dir(
                g_dir,
                ecosystem=eco,
                scope="user",
                skills_out=skills_by_key,
                commands_out=commands_by_key,
                is_command_dir=is_cmd,
                allow_nested_category=allow_nested,
                shadowed_paths_out=shadowed_paths,
            )

    # Preserve canonical skill names as keys in the returned dict
    skills_dict: Dict[str, Dict[str, Any]] = {}
    for _, s_entry in skills_by_key.items():
        skills_dict[s_entry["name"]] = s_entry

    commands_dict: Dict[str, Dict[str, Any]] = {}
    for _, c_entry in commands_by_key.items():
        commands_dict[c_entry["name"]] = c_entry

    # 3. Discover MCP Server Configurations (project-local first, then user-global)
    mcp_servers: Dict[str, Dict[str, Any]] = {}

    eff_extra_mcp = list(extra_mcp_paths or []) + list(extra_mcp_configs or [])
    if eff_extra_mcp:
        for extra_m in eff_extra_mcp:
            mp = Path(extra_m).expanduser()
            if not mp.is_absolute():
                mp = ws_path / mp
            _scan_mcp_json_file(mp, "platodoom", "project", mcp_servers)

    for anc in ancestor_dirs:
        _scan_mcp_json_file(anc / ".platodoom" / "mcp.json", "platodoom", "project", mcp_servers)
        _scan_mcp_json_file(anc / ".cursor" / "mcp.json", "cursor", "project", mcp_servers)
        _scan_mcp_json_file(anc / ".claude" / "mcp.json", "claude", "project", mcp_servers)
        _scan_mcp_json_file(anc / ".gemini" / "mcp.json", "gemini", "project", mcp_servers)
        _scan_mcp_json_file(anc / ".codex" / "mcp.json", "codex", "project", mcp_servers)
        _scan_mcp_json_file(anc / ".agents" / "mcp.json", "agents", "project", mcp_servers)
        _scan_mcp_json_file(anc / ".hermes" / "config.yaml", "hermes", "project", mcp_servers)

    if include_global:
        global_mcp_files: List[Tuple[Path, str]] = [
            (home_path / ".platodoom" / "mcp.json", "platodoom"),
            (home_path / ".claude.json", "claude"),
            (home_path / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json", "claude"),
            (home_path / ".config" / "Claude" / "claude_desktop_config.json", "claude"),
            (home_path / ".cursor" / "mcp.json", "cursor"),
            (home_path / ".gemini" / "config" / "mcp_config.json", "gemini"),
            (home_path / ".gemini" / "settings.json", "gemini"),
            (home_path / ".codex" / "mcp.json", "codex"),
            (home_path / ".hermes" / "config.yaml", "hermes"),
        ]
        for g_mcp_path, eco in global_mcp_files:
            _scan_mcp_json_file(g_mcp_path, eco, "user", mcp_servers)

        _scan_antigravity_mcp_schema_dir(home_path / ".gemini" / "antigravity-cli" / "mcp", "user", mcp_servers)

    discovered_ecosystems = sorted(
        {
            *(i.get("ecosystem") for i in instructions if i.get("ecosystem")),
            *(s.get("ecosystem") for s in skills_dict.values() if s.get("ecosystem")),
            *(m.get("ecosystem") for m in mcp_servers.values() if m.get("ecosystem")),
        }
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "workspace_dir": str(ws_path),
        "git_root": str(git_root),
        "home_dir": str(home_path),
        "include_global": bool(include_global),
        "instructions": instructions,
        "skills": skills_dict,
        "commands": commands_dict,
        "mcp_servers": mcp_servers,
        "shadowed_paths": shadowed_paths,
        "ecosystems": discovered_ecosystems,
        "elapsed_ms": elapsed_ms,
    }


def _collect_git_and_tree_snapshot(workspace_dir: Path) -> Tuple[str, str, List[str]]:
    branch = "detached"
    dirty = "clean"
    try:
        br = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=0.5,
        )
        if br.returncode == 0 and br.stdout.strip():
            branch = br.stdout.strip()
        st = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=0.5,
        )
        if st.returncode == 0:
            dirty = "dirty" if st.stdout.strip() else "clean"
    except Exception:
        pass

    ignored_names = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache"}
    top_entries: List[str] = []
    try:
        for item in sorted(workspace_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name))[:35]:
            if item.name in ignored_names:
                continue
            suffix = "/" if item.is_dir() else ""
            top_entries.append(f"{item.name}{suffix}")
    except OSError:
        pass

    return branch, dirty, top_entries


def build_zero_turn_workspace_compass(
    workspace_dir: Path | str,
    capabilities: Optional[Dict[str, Any]] = None,
    max_tokens: int = 450,
    *,
    include_global: Optional[bool] = None,
    discovery_manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Constructs the compact startup system context snapshot (`Zero-Turn Workspace Compass`)
    guaranteed <= max_tokens (default 450 tokens so baseline system prompt stays < 800 tokens).
    Filters out user-global entries unless `include_global` is explicitly enabled.
    """
    ws_path = Path(workspace_dir).expanduser().resolve()
    eff_caps = capabilities if isinstance(capabilities, dict) else discovery_manifest
    req_global = bool(include_global) if include_global is not None else False
    caps = eff_caps if isinstance(eff_caps, dict) else discover_workspace_capabilities(ws_path, include_global=req_global)
    allow_global = bool(include_global) if include_global is not None else bool(caps.get("include_global", False))
    branch, dirty, top_entries = _collect_git_and_tree_snapshot(ws_path)

    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    shell_info = os.environ.get("SHELL", "/bin/zsh")

    lines: List[str] = [
        "<workspace_compass>",
        "# Zero-Turn Workspace Compass",
        f"- **Host OS & Shell:** `{os_info}` | `{shell_info}`",
        f"- **Working Directory (`cwd`):** `{ws_path}`",
        f"- **Git Root & Status:** `{caps.get('git_root', ws_path)}` (branch: `{branch}`, status: `{dirty}`)",
        f"- **Top-Level Workspace Entries ({len(top_entries)}):** {', '.join(f'`{e}`' for e in top_entries) if top_entries else '(empty)'}",
    ]

    raw_instructions = caps.get("instructions") or []
    instructions = [
        inst for inst in raw_instructions if allow_global or inst.get("scope") == "project"
    ]
    if instructions:
        lines.append("\n## Discovered Instruction Files")
        for inst in instructions[:12]:
            lines.append(
                f"- `{inst.get('name')}` [{inst.get('ecosystem')}/{inst.get('scope')}]: {str(inst.get('summary', ''))[:100]}"
            )

    raw_mcp_servers = caps.get("mcp_servers") or {}
    mcp_servers = {
        srv_name: srv
        for srv_name, srv in raw_mcp_servers.items()
        if allow_global or srv.get("scope") == "project"
    }
    if mcp_servers:
        lines.append(
            "\n## Discovered MCP Servers (<available_mcp_servers> — Use `list_mcp_tools`, `inspect_mcp_tool`, `call_mcp_tool`)"
        )
        for srv_name, srv in list(mcp_servers.items())[:25]:
            tools_preview = srv.get("cached_tools") or []
            preview_str = f" ({len(tools_preview)} tools: {', '.join(tools_preview[:4])}{'...' if len(tools_preview) > 4 else ''})" if tools_preview else ""
            lines.append(f"- `{srv_name}` [{srv.get('ecosystem')}/{srv.get('transport')}]{preview_str}")
        lines.append("</available_mcp_servers>")

    raw_skills = caps.get("skills") or {}
    skills = {
        s_name: s_info
        for s_name, s_info in raw_skills.items()
        if allow_global or s_info.get("scope") == "project"
    }
    if skills:
        lines.append(
            f"\n## Progressive-Disclosure Skill Index (<available_skills> — {len(skills)} Discovered — Call `use_skill(name)` for Full Instructions)"
        )
        # Adaptive description length so total Compass never exceeds max_tokens (~4 chars/token)
        max_chars = max(400, int(max_tokens * 3.6))
        desc_cap = 55 if len(skills) > 50 else 90
        for s_name, s_info in skills.items():
            desc = str(s_info.get("description") or "").replace("\n", " ").strip()
            if len(desc) > desc_cap:
                desc = desc[: desc_cap - 3].rstrip() + "..."
            entry_line = f"- `{s_name}` ({s_info.get('ecosystem')}/{s_info.get('scope')}): {desc}"
            tentative = "\n".join(lines + [entry_line])
            if len(tentative) > max_chars - 120:
                remaining = len(skills) - sum(1 for l in lines if l.startswith("- `") and " (" in l)
                if remaining > 0:
                    lines.append(f"- *(+{remaining} more skills available via `/skills` or `use_skill(name)`)*")
                break
            lines.append(entry_line)
        lines.append("</available_skills>")

    lines.append("</workspace_compass>")
    compass = "\n".join(lines)
    hard_char_limit = int(max_tokens * 3.8)
    if len(compass) > hard_char_limit:
        compass = compass[: max(0, hard_char_limit - 84)].rsplit("\n", 1)[0] + "\n...[Compass capped for token budget]...\n</workspace_compass>"
    return compass


def use_skill(
    name: str,
    capabilities: Optional[Dict[str, Any]] = None,
    workspace_dir: Optional[Path | str] = None,
    *,
    home_dir: Optional[Path | str] = None,
    discovery_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    On-demand Progressive-Disclosure skill loader (`use_skill(name)`).
    Checks workspace capabilities first, and if not found in a workspace-scoped manifest,
    performs an on-demand global lookup (`include_global=True`) so explicitly requested
    skills resolve cleanly without injecting global skill bloat into baseline prompts.
    """
    eff_caps = capabilities if isinstance(capabilities, dict) else discovery_manifest
    caps = eff_caps if isinstance(eff_caps, dict) else discover_workspace_capabilities(
        workspace_dir or Path.cwd(),
        home_dir=home_dir,
        include_global=False,
    )

    query = (name or "").strip().lstrip("/")

    def _lookup_in_manifest(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        s_map = manifest.get("skills") or {}
        c_map = manifest.get("commands") or {}
        if query in s_map:
            return s_map[query]
        if query in c_map:
            return c_map[query]
        q_lower = query.lower()
        for k, v in s_map.items():
            if k.lower() == q_lower or str(v.get("name", "")).lower() == q_lower:
                return v
        for k, v in c_map.items():
            if k.lower() == q_lower or str(v.get("name", "")).lower() == q_lower:
                return v
        return None

    matched = _lookup_in_manifest(caps)
    if matched is None and not caps.get("include_global", False):
        resolved_ws = workspace_dir or caps.get("workspace_dir") or Path.cwd()
        resolved_home = home_dir or caps.get("home_dir")
        global_caps = discover_workspace_capabilities(
            resolved_ws,
            home_dir=resolved_home,
            include_global=True,
        )
        matched = _lookup_in_manifest(global_caps)

    skills = caps.get("skills") or {}

    if matched is None:
        available = sorted(list(skills.keys()))[:30]
        return {
            "ok": False,
            "name": query,
            "error": f"Skill '{query}' not found. Available skills ({len(skills)}): {', '.join(available)}",
            "content": "",
        }

    file_path = Path(matched.get("file_path") or matched.get("path") or "")
    if not file_path.exists():
        return {
            "ok": False,
            "name": matched.get("name", query),
            "error": f"Skill file does not exist on disk: {file_path}",
            "content": "",
        }

    skill_dir = file_path.parent.resolve()
    helper_dirs: Dict[str, str] = {}
    for helper_name in ("scripts", "references", "examples", "resources"):
        h_path = skill_dir / helper_name
        if h_path.exists():
            helper_dirs[helper_name] = str(h_path.resolve())

    full_content = file_path.read_text(encoding="utf-8", errors="replace")
    parsed_full = parse_skill_markdown(full_content, shallow=False)

    # Resolve relative references like `(scripts/...` or `scripts/...` inside markdown body
    def _resolve_rel_helper(m: re.Match[str]) -> str:
        prefix = m.group(1)
        rel_sub = m.group(2)
        abs_target = (skill_dir / rel_sub).resolve()
        return f"{prefix}{abs_target}"

    resolved_content = re.sub(
        r"(?P<prefix>[\(`\"'\s]|^)(?P<rel>(?:scripts|references|examples|resources)/[A-Za-z0-9_./-]+)",
        _resolve_rel_helper,
        full_content,
    )

    if len(resolved_content.encode("utf-8")) > 24576:
        resolved_content = (
            resolved_content.encode("utf-8")[:24576].decode("utf-8", errors="ignore")
            + "\n...[Skill instructions truncated at 24 KB]..."
        )

    return {
        "ok": True,
        "name": matched.get("name", query),
        "ecosystem": matched.get("ecosystem", "unknown"),
        "scope": matched.get("scope", "user"),
        "path": str(file_path.resolve()),
        "file_path": str(file_path.resolve()),
        "skill_dir": str(skill_dir),
        "helper_dirs": helper_dirs,
        "description": matched.get("description", ""),
        "body": parsed_full.get("body", ""),
        "content": resolved_content,
    }
