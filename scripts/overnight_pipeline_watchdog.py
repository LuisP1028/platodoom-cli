#!/usr/bin/env python3
"""
Overnight Cron & tmux Pipeline Watchdog, Auto-Healer, and Smart Resume Supervisor.
Document ID: overnight_pipeline_watchdog_860_v1
Governing Standards: platodoom specification, LANGUAGE.md, RULE[user_global]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as _xml_escape


DEFAULT_PIPELINE_TMUX_SESSION = "sandcastle-pipeline"
DEFAULT_WATCHDOG_TMUX_SESSION = "sandcastle-watchdog"
DEFAULT_PIPELINE_SCRIPT = ".sandcastle/multi-stage-blackboard-pipeline-performance.mts"
DEFAULT_CHECK_INTERVAL_SECONDS = 3600
DEFAULT_STALL_TIMEOUT_SECONDS = 2700
CRON_TAG_PREFIX = "# sandcastle-overnight-watchdog"

STAGE_WORKER_MANIFESTS: Dict[int, List[Tuple[str, str]]] = {
    1: [
        ("codebase-mapper", "codebase-mapper.txt"),
        ("api-schema-mapper", "api-schema-mapper.txt"),
        ("mcp-provisioner", "mcp-provisioner.txt"),
        ("library-installer", "libs-installed.txt"),
        ("function-mapper", "function-mapper.txt"),
    ],
    2: [
        ("planning-coordinator", "wayfinder-read-and-plan.txt"),
        ("architecture-planner", "architecture-planner.txt"),
        ("storage-planner", "storage-planner.txt"),
        ("stack-planner", "stack-planner.txt"),
        ("optimization-planner", "optimization-planner.txt"),
    ],
    3: [
        ("architecture-implementer", "architecture-implementer.txt"),
        ("storage-implementer", "storage-implementer.txt"),
        ("stack-implementer", "stack-implementer.txt"),
        ("optimization-implementer", "optimization-implementer.txt"),
    ],
    4: [
        ("architecture-tester", "architecture-tester.txt"),
        ("storage-tester", "storage-tester.txt"),
        ("stack-tester", "stack-tester.txt"),
        ("optimization-tester", "optimization-tester.txt"),
    ],
    5: [
        ("branch-merger", "merged.txt"),
    ],
}

SLASH_COMMAND_SANITIZE_RE = re.compile(
    r"(^|[\s([{:;,-])(/(?:plan|goal|compact|schedule|browser|grill-me|teamwork|learn|boost|help|model|access|status|skills|mcp|exit|shutdown|clear|new|resume|history|build)\b)",
    re.IGNORECASE | re.MULTILINE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_repo_dir(repo_dir: Optional[str | Path] = None) -> Path:
    if repo_dir:
        return Path(repo_dir).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def compute_repo_slug(repo_dir: Path) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", repo_dir.name).strip("-").lower() or "repo"
    digest = hashlib.sha1(str(repo_dir).encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def state_file_path(repo_dir: Path) -> Path:
    return repo_dir / ".sandcastle" / "logs" / "overnight-watchdog-state.json"


def lock_file_path(repo_dir: Path) -> Path:
    return repo_dir / ".sandcastle" / "logs" / "overnight-watchdog.lock"


def watchdog_log_path(repo_dir: Path) -> Path:
    return repo_dir / ".sandcastle" / "logs" / "overnight-watchdog.log"


def append_watchdog_log(repo_dir: Path, event: str, details: Dict[str, Any]) -> None:
    log_file = watchdog_log_path(repo_dir)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": utc_now_iso(),
        "event": event,
        **details,
    }
    line = f"[{record['timestamp']}] [{event}] {json.dumps(record, sort_keys=True)}\n"
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(line)


def load_watchdog_state(repo_dir: Path) -> Dict[str, Any]:
    sf = state_file_path(repo_dir)
    if sf.is_file():
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_watchdog_state(repo_dir: Path, state: Dict[str, Any]) -> Path:
    sf = state_file_path(repo_dir)
    sf.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now_iso()
    tmp = sf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(sf))
    return sf


def _git_branch_exists(repo_dir: Path, branch: str) -> bool:
    res = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=str(repo_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def _git_file_non_empty(repo_dir: Path, branch: str, rel_path: str) -> bool:
    wt_dir = repo_dir / ".sandcastle" / "worktrees" / branch.replace("/", "-")
    disk_file = wt_dir / rel_path
    if disk_file.is_file():
        try:
            if disk_file.read_text(encoding="utf-8", errors="replace").strip():
                return True
        except Exception:
            pass
    if branch not in ("main", "HEAD") and not _git_branch_exists(repo_dir, branch):
        return False
    res = subprocess.run(
        ["git", "show", f"{branch}:{rel_path}"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    return res.returncode == 0 and len(res.stdout.strip()) > 0


def _git_read_file(repo_dir: Path, branch: str, rel_path: str) -> str:
    wt_dir = repo_dir / ".sandcastle" / "worktrees" / branch.replace("/", "-")
    disk_file = wt_dir / rel_path
    if disk_file.is_file():
        try:
            content = disk_file.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                return content
        except Exception:
            pass
    res = subprocess.run(
        ["git", "show", f"{branch}:{rel_path}"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    return res.stdout if res.returncode == 0 else ""


def discover_latest_run_id(repo_dir: Path) -> Optional[str]:
    state = load_watchdog_state(repo_dir)
    if state.get("run_id"):
        return str(state["run_id"])
    logs_dir = repo_dir / ".sandcastle" / "logs"
    if not logs_dir.is_dir():
        return None
    run_id_re = re.compile(r"^(\d{8}T\d{6}-\d{3}-[a-z0-9]{4})-")
    candidates: List[Tuple[float, str]] = []
    for entry in logs_dir.iterdir():
        m = run_id_re.match(entry.name)
        if m:
            try:
                candidates.append((entry.stat().st_mtime, m.group(1)))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def inspect_run_progress(repo_dir: Path, run_id: str) -> Dict[str, Any]:
    plan_branch = f"agent/plan-{run_id}"
    already_merged = _git_file_non_empty(repo_dir, "main", f"handoff/{run_id}/merged.txt")
    if already_merged:
        merged_settled: List[str] = []
        for stage_num in sorted(STAGE_WORKER_MANIFESTS.keys()):
            for worker_name, manifest_name in STAGE_WORKER_MANIFESTS[stage_num]:
                manifest_rel = f"handoff/{run_id}/{manifest_name}"
                if _git_file_non_empty(repo_dir, "main", manifest_rel) and worker_name not in merged_settled:
                    merged_settled.append(worker_name)
        if "branch-merger" not in merged_settled:
            merged_settled.append("branch-merger")
        return {
            "run_id": run_id,
            "plan_branch": plan_branch,
            "plan_branch_exists": _git_branch_exists(repo_dir, plan_branch),
            "already_merged": True,
            "current_stage": 5,
            "status": "COMPLETED",
            "reason": f"handoff/{run_id}/merged.txt exists on main",
            "settled_workers": merged_settled,
            "pending_workers": [],
        }

    plan_exists = _git_branch_exists(repo_dir, plan_branch)
    current_stage = 1
    reason = "Stage 1 in progress"

    if plan_exists:
        if _git_file_non_empty(repo_dir, plan_branch, f"handoff/{run_id}/validated.txt"):
            verdict_raw = _git_read_file(repo_dir, plan_branch, f"handoff/{run_id}/validated.txt")
            failed = bool(re.search(r"\bSTATUS\s*:\s*FAIL\b|^FAIL\b", verdict_raw, re.I | re.M))
            if not failed:
                current_stage = 5
                reason = "Stage 4 validation passed; ready for Stage 5 branch-merger"
            else:
                current_stage = 1
                reason = "Stage 4 validation failed; remediation iteration at Stage 1"
        elif _git_file_non_empty(repo_dir, plan_branch, f"handoff/{run_id}/verification-blueprint-ready.txt"):
            current_stage = 4
            reason = "Stage 3 complete; at Stage 4 testing & validation"
        elif (
            _git_file_non_empty(repo_dir, plan_branch, f"handoff/{run_id}/implementation-blueprint-ready.txt")
            or _git_file_non_empty(repo_dir, plan_branch, f"handoff/{run_id}/impl-blueprint-ready.txt")
        ):
            current_stage = 3
            reason = "Stage 2 complete; at Stage 3 implementation"
        elif (
            _git_file_non_empty(repo_dir, plan_branch, f"handoff/{run_id}/refined-spec-ready.txt")
            and _git_file_non_empty(repo_dir, plan_branch, f"handoff/{run_id}/function-mapper.txt")
        ):
            current_stage = 2
            reason = "Stage 1 complete; at Stage 2 architecture planning"

    settled_workers: List[str] = []
    pending_workers: List[str] = []
    for worker_name, manifest_name in STAGE_WORKER_MANIFESTS.get(current_stage, []):
        manifest_rel = f"handoff/{run_id}/{manifest_name}"
        worker_branch = "main" if worker_name == "branch-merger" else f"agent/{worker_name}-{run_id}"
        if _git_file_non_empty(repo_dir, worker_branch, manifest_rel) or (
            plan_exists and _git_file_non_empty(repo_dir, plan_branch, manifest_rel)
        ):
            settled_workers.append(worker_name)
        else:
            pending_workers.append(worker_name)

    return {
        "run_id": run_id,
        "plan_branch": plan_branch,
        "plan_branch_exists": plan_exists,
        "already_merged": False,
        "current_stage": current_stage,
        "status": "IN_PROGRESS",
        "reason": reason,
        "settled_workers": settled_workers,
        "pending_workers": pending_workers,
    }


def latest_log_activity(repo_dir: Path, run_id: str) -> Dict[str, Any]:
    logs_dir = repo_dir / ".sandcastle" / "logs"
    if not logs_dir.is_dir():
        return {"latest_log_file": None, "age_seconds": None, "mtime_iso": None}
    newest_mtime: Optional[float] = None
    newest_file: Optional[Path] = None
    for path in logs_dir.glob(f"{run_id}-*.log"):
        if path.name.endswith("-watchdog-healer.log"):
            continue
        try:
            st = path.stat()
            if newest_mtime is None or st.st_mtime > newest_mtime:
                newest_mtime = st.st_mtime
                newest_file = path
        except OSError:
            continue
    if newest_mtime is None or newest_file is None:
        return {"latest_log_file": None, "age_seconds": None, "mtime_iso": None}
    age = max(0.0, time.time() - newest_mtime)
    mtime_iso = datetime.fromtimestamp(newest_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "latest_log_file": str(newest_file.relative_to(repo_dir)),
        "age_seconds": round(age, 1),
        "mtime_iso": mtime_iso,
    }


def is_tmux_session_alive(session_name: str) -> bool:
    if not shutil.which("tmux"):
        return False
    res = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def is_pipeline_process_running(repo_dir: Path, run_id: Optional[str], seed_file: Optional[str], tmux_session: str) -> Dict[str, Any]:
    tmux_alive = is_tmux_session_alive(tmux_session)
    pane_command = ""
    pane_dead = False
    if tmux_alive:
        res = subprocess.run(
            ["tmux", "list-panes", "-t", tmux_session, "-F", "#{pane_dead}|#{pane_current_command}|#{pane_pid}"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().splitlines()[0]
            parts = first_line.split("|")
            pane_dead = parts[0] == "1"
            pane_command = parts[1] if len(parts) > 1 else ""

    pgrep_patterns = ["multi-stage-blackboard-pipeline"]
    matched_pids: List[int] = []
    res_ps = subprocess.run(["ps", "-Ao", "pid,command"], capture_output=True, text=True)
    if res_ps.returncode == 0:
        for line in res_ps.stdout.splitlines()[1:]:
            line_str = line.strip()
            if not line_str:
                continue
            if any(pat in line_str for pat in pgrep_patterns):
                if (run_id and run_id in line_str) or (seed_file and seed_file in line_str) or str(repo_dir) in line_str:
                    pid_part = line_str.split(None, 1)[0]
                    if pid_part.isdigit() and int(pid_part) != os.getpid():
                        matched_pids.append(int(pid_part))

    process_alive = len(matched_pids) > 0 or (tmux_alive and not pane_dead and pane_command not in ("", "zsh", "bash", "sh"))
    return {
        "tmux_session": tmux_session,
        "tmux_session_exists": tmux_alive,
        "tmux_pane_dead": pane_dead,
        "tmux_pane_command": pane_command,
        "matched_pids": matched_pids,
        "pipeline_alive": process_alive,
    }


def sanitize_slash_commands_in_text(text: str) -> str:
    if not text or "/" not in text:
        return text
    return SLASH_COMMAND_SANITIZE_RE.sub(r"\1`\2`", text)


def perform_deterministic_repairs(repo_dir: Path, run_id: str) -> List[str]:
    actions: List[str] = []

    # 1. Remove stale git index/config locks if no git process is active
    ps_git = subprocess.run(["pgrep", "-f", "git "], capture_output=True, text=True)
    git_running = ps_git.returncode == 0 and bool(ps_git.stdout.strip())
    if not git_running:
        git_dir = repo_dir / ".git"
        lock_candidates = list(git_dir.glob("*.lock")) + list(git_dir.glob("worktrees/*/*.lock"))
        for lf in lock_candidates:
            try:
                lf.unlink()
                actions.append(f"Removed stale git lock: {lf.relative_to(repo_dir)}")
            except OSError:
                pass

    subprocess.run(["git", "worktree", "prune"], cwd=str(repo_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Sanitize any raw slash commands in existing Blackboard JSON files for run_id
    logs_dir = repo_dir / ".sandcastle" / "logs"
    if logs_dir.is_dir():
        for bb_file in logs_dir.glob(f"{run_id}*blackboard*.json"):
            try:
                raw_json = bb_file.read_text(encoding="utf-8")
                parsed = json.loads(raw_json)
                modified = False
                for entry in parsed.get("entries", []):
                    if isinstance(entry, dict) and isinstance(entry.get("content"), str):
                        cleaned = sanitize_slash_commands_in_text(entry["content"])
                        if cleaned != entry["content"]:
                            entry["content"] = cleaned
                            modified = True
                if modified:
                    bb_file.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
                    actions.append(f"Sanitized unbackticked slash commands in {bb_file.name}")
            except Exception:
                continue

    # 3. Salvage uncommitted worktree manifests or clean broken merge state in worker worktrees
    wt_root = repo_dir / ".sandcastle" / "worktrees"
    if wt_root.is_dir():
        for stage_num, worker_list in STAGE_WORKER_MANIFESTS.items():
            for worker_name, manifest_name in worker_list:
                if worker_name == "branch-merger":
                    continue
                branch = f"agent/{worker_name}-{run_id}"
                wt_dir = wt_root / branch.replace("/", "-")
                if not wt_dir.is_dir():
                    continue
                subprocess.run(["git", "merge", "--abort"], cwd=str(wt_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                manifest_disk = wt_dir / "handoff" / run_id / manifest_name
                if manifest_disk.is_file() and manifest_disk.read_text(encoding="utf-8", errors="replace").strip():
                    if not _git_file_non_empty(repo_dir, branch, f"handoff/{run_id}/{manifest_name}"):
                        subprocess.run(["git", "add", "-A"], cwd=str(wt_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        commit_res = subprocess.run(
                            ["git", "commit", "-m", f"chore(watchdog-salvage): commit completed manifest for {worker_name}"],
                            cwd=str(wt_dir),
                            capture_output=True,
                            text=True,
                        )
                        if commit_res.returncode == 0:
                            actions.append(f"Salvaged and committed completed worktree manifest for {worker_name} on {branch}")

    return actions


def collect_recent_failure_excerpt(repo_dir: Path, run_id: str, tmux_session: str) -> str:
    snippets: List[str] = []
    if is_tmux_session_alive(tmux_session):
        pane_res = subprocess.run(
            ["tmux", "capture-pane", "-pt", f"{tmux_session}:0.0", "-S", "-80"],
            capture_output=True,
            text=True,
        )
        if pane_res.returncode == 0 and pane_res.stdout.strip():
            snippets.append(f"=== tmux pane ({tmux_session}) ===\n" + pane_res.stdout.strip()[-3000:])

    logs_dir = repo_dir / ".sandcastle" / "logs"
    if logs_dir.is_dir():
        log_files = sorted(
            [p for p in logs_dir.glob(f"{run_id}-*.log") if not p.name.endswith("-watchdog-healer.log")],
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        for lf in log_files[:3]:
            try:
                lines = lf.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                snippets.append(f"=== {lf.name} (last {len(lines)} lines) ===\n" + "\n".join(lines)[-2500:])
            except Exception:
                continue
    return "\n\n".join(snippets)


def invoke_ai_healer(
    repo_dir: Path,
    run_id: str,
    seed_file: str,
    progress: Dict[str, Any],
    failure_excerpt: str,
    timeout_seconds: int = 240,
) -> Dict[str, Any]:
    agy_bin = shutil.which("agy") or str(Path.home() / ".local" / "bin" / "agy")
    prompt_file = repo_dir / ".sandcastle" / "overnight-watchdog.md"
    healer_log = repo_dir / ".sandcastle" / "logs" / f"{run_id}-watchdog-healer.log"
    healer_log.parent.mkdir(parents=True, exist_ok=True)

    if not Path(agy_bin).is_file():
        return {"invoked": False, "reason": "agy binary not found"}

    skill_instructions = prompt_file.read_text(encoding="utf-8") if prompt_file.is_file() else ""
    healer_prompt = (
        f"{skill_instructions}\n\n"
        f"## Active Crash Context\n"
        f"- Repository: `{repo_dir}`\n"
        f"- Seed Specification: `{seed_file}`\n"
        f"- Run ID: `{run_id}`\n"
        f"- Current Stage: `Stage {progress.get('current_stage')}` ({progress.get('reason')})\n"
        f"- Already Settled Workers in Stage: `{progress.get('settled_workers')}`\n"
        f"- Pending/Failed Workers in Stage: `{progress.get('pending_workers')}`\n\n"
        f"## Recent Log Tail\n```\n{failure_excerpt[-5000:]}\n```\n\n"
        f"Perform only Step 2 and Step 3 repairs (do NOT start a new pipeline process yourself; the watchdog supervisor will immediately reboot the pipeline in tmux after you exit). Summarize your diagnosis and any fix applied."
    )

    try:
        proc = subprocess.run(
            [agy_bin, "-p", healer_prompt, "--mode", "agent", "--yolo"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        with open(healer_log, "a", encoding="utf-8") as handle:
            handle.write(f"\n=== Healer invocation at {utc_now_iso()} (exit={proc.returncode}) ===\n")
            handle.write(proc.stdout or "")
            if proc.stderr:
                handle.write("\n--- STDERR ---\n" + proc.stderr)
        return {
            "invoked": True,
            "exit_code": proc.returncode,
            "healer_log": str(healer_log.relative_to(repo_dir)),
        }
    except subprocess.TimeoutExpired:
        return {"invoked": True, "exit_code": 124, "reason": "healer timed out"}
    except Exception as exc:
        return {"invoked": False, "reason": str(exc)}


def start_pipeline_in_tmux(
    repo_dir: Path,
    seed_file: str,
    run_id: str,
    pipeline_script: str = DEFAULT_PIPELINE_SCRIPT,
    tmux_session: str = DEFAULT_PIPELINE_TMUX_SESSION,
    kickstart: str = "blackboard",
) -> Dict[str, Any]:
    if not shutil.which("tmux"):
        raise RuntimeError("`tmux` is required for overnight pipeline supervision but was not found on PATH.")

    logs_dir = repo_dir / ".sandcastle" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmux_log = logs_dir / f"{run_id}-tmux-pipeline.log"

    if is_tmux_session_alive(tmux_session):
        subprocess.run(["tmux", "kill-session", "-t", tmux_session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd_str = (
        f"cd {shlex.quote(str(repo_dir))} && "
        f"npx tsx {shlex.quote(pipeline_script)} {shlex.quote(seed_file)} "
        f"--kickstart={shlex.quote(kickstart)} "
        f"--run-id={shlex.quote(run_id)} "
        f"--auto-resume "
        f"2>&1 | tee -a {shlex.quote(str(tmux_log))}"
    )

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", tmux_session, cmd_str],
        cwd=str(repo_dir),
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", tmux_session, "remain-on-exit", "on"],
        cwd=str(repo_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    state = load_watchdog_state(repo_dir)
    state.update(
        {
            "repo_dir": str(repo_dir),
            "seed_file": seed_file,
            "run_id": run_id,
            "pipeline_script": pipeline_script,
            "tmux_session": tmux_session,
            "kickstart": kickstart,
            "last_started_at": utc_now_iso(),
            "restart_count": int(state.get("restart_count", 0)) + 1,
            "status": "RUNNING",
        }
    )
    save_watchdog_state(repo_dir, state)
    append_watchdog_log(
        repo_dir,
        "PIPELINE_LAUNCHED_IN_TMUX",
        {
            "run_id": run_id,
            "seed_file": seed_file,
            "tmux_session": tmux_session,
            "tmux_log": str(tmux_log.relative_to(repo_dir)),
            "restart_count": state["restart_count"],
        },
    )
    return {
        "ok": True,
        "tmux_session": tmux_session,
        "run_id": run_id,
        "seed_file": seed_file,
        "tmux_log": str(tmux_log),
        "command": cmd_str,
    }


def start_watchdog_daemon_in_tmux(
    repo_dir: Path,
    seed_file: str,
    run_id: str,
    interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
    stall_timeout_seconds: int = DEFAULT_STALL_TIMEOUT_SECONDS,
    tmux_watchdog_session: str = DEFAULT_WATCHDOG_TMUX_SESSION,
    enable_ai_healer: bool = True,
) -> Dict[str, Any]:
    if is_tmux_session_alive(tmux_watchdog_session):
        subprocess.run(["tmux", "kill-session", "-t", tmux_watchdog_session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    script_abs = Path(__file__).resolve()
    healer_flag = "--enable-ai-healer" if enable_ai_healer else "--no-ai-healer"
    cmd_str = (
        f"cd {shlex.quote(str(repo_dir))} && "
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script_abs))} "
        f"--daemon "
        f"--seed {shlex.quote(seed_file)} "
        f"--run-id {shlex.quote(run_id)} "
        f"--interval-seconds {int(interval_seconds)} "
        f"--stall-timeout-seconds {int(stall_timeout_seconds)} "
        f"{healer_flag}"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", tmux_watchdog_session, cmd_str],
        cwd=str(repo_dir),
        check=True,
    )
    return {
        "ok": True,
        "tmux_watchdog_session": tmux_watchdog_session,
        "interval_seconds": interval_seconds,
        "stall_timeout_seconds": stall_timeout_seconds,
    }


def _cron_tag_for_repo(repo_dir: Path) -> str:
    return f"{CRON_TAG_PREFIX}:{compute_repo_slug(repo_dir)}"


def _launchd_plist_path(repo_dir: Path) -> Path:
    slug = compute_repo_slug(repo_dir)
    return Path.home() / "Library" / "LaunchAgents" / f"com.sandcastle.overnight-watchdog.{slug}.plist"


def _read_current_crontab() -> str:
    try:
        res = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=2.0)
        if res.returncode != 0:
            return ""
        return res.stdout
    except Exception:
        return ""


def is_cron_installed(repo_dir: Path) -> Tuple[bool, Optional[str]]:
    tag = _cron_tag_for_repo(repo_dir)
    for line in _read_current_crontab().splitlines():
        if tag in line or (CRON_TAG_PREFIX in line and str(repo_dir) in line):
            return True, line.strip()
    plist_path = _launchd_plist_path(repo_dir)
    local_cron_file = repo_dir / ".sandcastle" / "logs" / "overnight-watchdog.crontab"
    if plist_path.is_file():
        if local_cron_file.is_file():
            return True, local_cron_file.read_text(encoding="utf-8").strip()
        return True, f"launchd:{plist_path}"
    return False, None


_SAFE_CRON_SCHEDULE_RE = re.compile(r"^[0-9*/,\-]+(?:\s+[0-9*/,\-]+){4}$")


def install_hourly_cron(
    repo_dir: Path,
    seed_file: str,
    run_id: str,
    cron_schedule: str = "0 * * * *",
) -> Dict[str, Any]:
    cleaned_schedule = str(cron_schedule or "").strip()
    if not _SAFE_CRON_SCHEDULE_RE.fullmatch(cleaned_schedule):
        raise ValueError(
            f"Invalid cron_schedule {cron_schedule!r}; expected a 5-field cron expression without newlines or shell metacharacters."
        )
    tag = _cron_tag_for_repo(repo_dir)
    script_abs = Path(__file__).resolve()
    cron_log = repo_dir / ".sandcastle" / "logs" / "overnight-watchdog-cron.log"
    cron_log.parent.mkdir(parents=True, exist_ok=True)

    path_env = os.environ.get("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin").replace("\n", "").replace("\r", "")
    if "/opt/homebrew/bin" not in path_env:
        path_env = f"/opt/homebrew/bin:/usr/local/bin:{path_env}"
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in path_env:
        path_env = f"{local_bin}:{path_env}"

    cron_line = (
        f"{cleaned_schedule} PATH={shlex.quote(path_env)} "
        f"cd {shlex.quote(str(repo_dir))} && "
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script_abs))} "
        f"--check-now --seed {shlex.quote(seed_file)} --run-id {shlex.quote(run_id)} "
        f">> {shlex.quote(str(cron_log))} 2>&1 {tag}"
    )

    local_cron_file = repo_dir / ".sandcastle" / "logs" / "overnight-watchdog.crontab"
    local_cron_file.write_text(cron_line + "\n", encoding="utf-8")

    existing_lines = [
        ln
        for ln in _read_current_crontab().splitlines()
        if tag not in ln and not (CRON_TAG_PREFIX in ln and str(repo_dir) in ln)
    ]
    existing_lines.append(cron_line)
    new_crontab = "\n".join([ln for ln in existing_lines if ln.strip()]) + "\n"

    mechanism = "crontab"
    try:
        proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True, timeout=2.0)
        if proc.returncode != 0:
            mechanism = "launchd"
    except subprocess.TimeoutExpired:
        mechanism = "launchd"
    except Exception:
        mechanism = "launchd"

    # Also install native macOS LaunchAgent (hourly StartInterval=3600) so headless macOS never blocks on TCC crontab modal
    plist_path = _launchd_plist_path(repo_dir)
    try:
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        label = f"com.sandcastle.overnight-watchdog.{compute_repo_slug(repo_dir)}"
        plist_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_xml_escape(label)}</string>
    <key>WorkingDirectory</key>
    <string>{_xml_escape(str(repo_dir))}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{_xml_escape(path_env)}</string>
    </dict>
    <key>ProgramArguments</key>
    <array>
        <string>{_xml_escape(sys.executable)}</string>
        <string>{_xml_escape(str(script_abs))}</string>
        <string>--check-now</string>
        <string>--seed</string>
        <string>{_xml_escape(str(seed_file))}</string>
        <string>--run-id</string>
        <string>{_xml_escape(str(run_id))}</string>
    </array>
    <key>StartInterval</key>
    <integer>3600</integer>
    <key>StandardOutPath</key>
    <string>{_xml_escape(str(cron_log))}</string>
    <key>StandardErrorPath</key>
    <string>{_xml_escape(str(cron_log))}</string>
</dict>
</plist>
"""
        plist_path.write_text(plist_xml, encoding="utf-8")
        subprocess.run(["launchctl", "unload", str(plist_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3.0)
        subprocess.run(["launchctl", "load", str(plist_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3.0)
    except Exception:
        pass

    append_watchdog_log(
        repo_dir,
        "CRON_INSTALLED",
        {
            "mechanism": mechanism,
            "schedule": cron_schedule,
            "cron_line": cron_line,
            "launchd_plist": str(plist_path),
            "run_id": run_id,
            "seed_file": seed_file,
        },
    )
    return {
        "ok": True,
        "installed": True,
        "mechanism": mechanism,
        "schedule": cron_schedule,
        "cron_line": cron_line,
        "launchd_plist": str(plist_path),
    }


def uninstall_hourly_cron(repo_dir: Path) -> Dict[str, Any]:
    tag = _cron_tag_for_repo(repo_dir)
    removed = False
    current = _read_current_crontab().splitlines()
    filtered = [
        ln
        for ln in current
        if tag not in ln and not (CRON_TAG_PREFIX in ln and str(repo_dir) in ln)
    ]
    if len(filtered) != len(current):
        try:
            if filtered:
                new_crontab = "\n".join([ln for ln in filtered if ln.strip()]) + "\n"
                subprocess.run(["crontab", "-"], input=new_crontab, text=True, check=False, timeout=2.0)
            else:
                subprocess.run(["crontab", "-r"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0)
            removed = True
        except Exception:
            pass

    plist_path = _launchd_plist_path(repo_dir)
    if plist_path.is_file():
        try:
            subprocess.run(["launchctl", "unload", str(plist_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3.0)
        except Exception:
            pass
        try:
            plist_path.unlink()
            removed = True
        except OSError:
            pass

    append_watchdog_log(repo_dir, "CRON_UNINSTALLED", {"tag": tag, "removed": removed})
    return {"ok": True, "removed": removed}


def get_watchdog_status(
    repo_dir: Path,
    seed_file: Optional[str] = None,
    run_id: Optional[str] = None,
    tmux_session: str = DEFAULT_PIPELINE_TMUX_SESSION,
    tmux_watchdog_session: str = DEFAULT_WATCHDOG_TMUX_SESSION,
) -> Dict[str, Any]:
    state = load_watchdog_state(repo_dir)
    eff_run_id = run_id or state.get("run_id") or discover_latest_run_id(repo_dir)
    eff_seed = seed_file or state.get("seed_file") or "platodoom specification"

    progress = (
        inspect_run_progress(repo_dir, eff_run_id)
        if eff_run_id
        else {
            "run_id": None,
            "already_merged": False,
            "current_stage": 1,
            "status": "NO_RUN_ID",
            "settled_workers": [],
            "pending_workers": [],
        }
    )
    proc_info = is_pipeline_process_running(repo_dir, eff_run_id, eff_seed, tmux_session)
    log_info = latest_log_activity(repo_dir, eff_run_id) if eff_run_id else {"latest_log_file": None, "age_seconds": None}
    cron_installed, cron_line = is_cron_installed(repo_dir)

    return {
        "ok": True,
        "timestamp": utc_now_iso(),
        "repo_dir": str(repo_dir),
        "seed_file": eff_seed,
        "run_id": eff_run_id,
        "progress": progress,
        "pipeline_process": proc_info,
        "tmux_pipeline_alive": proc_info["pipeline_alive"],
        "tmux_watchdog_alive": is_tmux_session_alive(tmux_watchdog_session),
        "latest_log": log_info,
        "cron_installed": cron_installed,
        "cron_line": cron_line,
        "restart_count": int(state.get("restart_count", 0)),
    }


def run_health_check_and_heal(
    repo_dir: Path,
    seed_file: Optional[str] = None,
    run_id: Optional[str] = None,
    pipeline_script: str = DEFAULT_PIPELINE_SCRIPT,
    tmux_session: str = DEFAULT_PIPELINE_TMUX_SESSION,
    stall_timeout_seconds: int = DEFAULT_STALL_TIMEOUT_SECONDS,
    enable_ai_healer: bool = True,
    auto_uninstall_cron_on_complete: bool = False,
) -> Dict[str, Any]:
    lock_path = lock_file_path(repo_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {
            "ok": True,
            "action": "SKIPPED_LOCK_HELD",
            "reason": "Another watchdog health check is currently running.",
        }

    try:
        state = load_watchdog_state(repo_dir)
        eff_run_id = run_id or state.get("run_id") or discover_latest_run_id(repo_dir)
        eff_seed = seed_file or state.get("seed_file") or "platodoom specification"
        eff_script = state.get("pipeline_script") or pipeline_script

        if not eff_run_id:
            return {
                "ok": False,
                "action": "NO_ACTIVE_RUN_ID",
                "reason": "No runId provided or discovered in .sandcastle/logs/",
            }

        progress = inspect_run_progress(repo_dir, eff_run_id)
        if progress["already_merged"]:
            state["status"] = "COMPLETED"
            state["completed_at"] = utc_now_iso()
            save_watchdog_state(repo_dir, state)
            if auto_uninstall_cron_on_complete:
                uninstall_hourly_cron(repo_dir)
            append_watchdog_log(
                repo_dir,
                "PIPELINE_COMPLETED",
                {"run_id": eff_run_id, "seed_file": eff_seed},
            )
            return {
                "ok": True,
                "action": "COMPLETED",
                "run_id": eff_run_id,
                "seed_file": eff_seed,
                "progress": progress,
            }

        proc_info = is_pipeline_process_running(repo_dir, eff_run_id, eff_seed, tmux_session)
        log_info = latest_log_activity(repo_dir, eff_run_id)
        age_sec = log_info.get("age_seconds")
        is_stalled = (
            proc_info["pipeline_alive"]
            and age_sec is not None
            and float(age_sec) > float(stall_timeout_seconds)
        )

        if proc_info["pipeline_alive"] and not is_stalled:
            append_watchdog_log(
                repo_dir,
                "HEALTHY",
                {
                    "run_id": eff_run_id,
                    "current_stage": progress["current_stage"],
                    "settled_workers": progress["settled_workers"],
                    "pending_workers": progress["pending_workers"],
                    "latest_log_age_seconds": age_sec,
                    "matched_pids": proc_info["matched_pids"],
                },
            )
            return {
                "ok": True,
                "action": "HEALTHY",
                "run_id": eff_run_id,
                "seed_file": eff_seed,
                "progress": progress,
                "pipeline_process": proc_info,
                "latest_log": log_info,
            }

        failure_reason = (
            f"Pipeline stalled (no log activity for {age_sec}s > {stall_timeout_seconds}s)"
            if is_stalled
            else "Pipeline process is not running and Stage 5 merge is not complete"
        )
        append_watchdog_log(
            repo_dir,
            "CRASH_OR_STALL_DETECTED",
            {
                "run_id": eff_run_id,
                "seed_file": eff_seed,
                "failure_reason": failure_reason,
                "current_stage": progress["current_stage"],
                "settled_workers": progress["settled_workers"],
                "pending_workers": progress["pending_workers"],
            },
        )

        # If stalled, terminate wedged pipeline PIDs before repairing and restarting
        if is_stalled:
            for pid in proc_info.get("matched_pids", []):
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass

        excerpt = collect_recent_failure_excerpt(repo_dir, eff_run_id, tmux_session)
        deterministic_actions = perform_deterministic_repairs(repo_dir, eff_run_id)
        updated_progress = inspect_run_progress(repo_dir, eff_run_id)

        healer_result: Dict[str, Any] = {"invoked": False}
        if enable_ai_healer:
            healer_result = invoke_ai_healer(
                repo_dir=repo_dir,
                run_id=eff_run_id,
                seed_file=eff_seed,
                progress=updated_progress,
                failure_excerpt=excerpt,
            )

        launch_info = start_pipeline_in_tmux(
            repo_dir=repo_dir,
            seed_file=eff_seed,
            run_id=eff_run_id,
            pipeline_script=eff_script,
            tmux_session=tmux_session,
        )
        append_watchdog_log(
            repo_dir,
            "PIPELINE_HEALED_AND_REBOOTED",
            {
                "run_id": eff_run_id,
                "seed_file": eff_seed,
                "resumed_stage": updated_progress["current_stage"],
                "settled_workers": updated_progress["settled_workers"],
                "pending_workers": updated_progress["pending_workers"],
                "deterministic_actions": deterministic_actions,
                "healer_result": healer_result,
            },
        )
        return {
            "ok": True,
            "action": "HEALED_AND_REBOOTED",
            "run_id": eff_run_id,
            "seed_file": eff_seed,
            "failure_reason": failure_reason,
            "deterministic_actions": deterministic_actions,
            "healer_result": healer_result,
            "progress": updated_progress,
            "launch": launch_info,
        }
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fd.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overnight_pipeline_watchdog",
        description="Overnight Cron & tmux Watchdog Supervisor for the Sandcastle Blackboard Pipeline",
    )
    parser.add_argument("--repo-dir", type=str, default=None, help="Path to repository root")
    parser.add_argument("--seed", type=str, default=None, help="Seed specification file (e.g., platodoom specification)")
    parser.add_argument("--run-id", type=str, default=None, help="Existing pipeline runId to monitor/resume")
    parser.add_argument("--pipeline-script", type=str, default=DEFAULT_PIPELINE_SCRIPT, help="Pipeline .mts script path")
    parser.add_argument("--tmux-session", type=str, default=DEFAULT_PIPELINE_TMUX_SESSION, help="tmux session name for pipeline")
    parser.add_argument("--watchdog-tmux-session", type=str, default=DEFAULT_WATCHDOG_TMUX_SESSION, help="tmux session name for watchdog daemon")
    parser.add_argument("--start", action="store_true", help="Start or resume the pipeline in tmux and configure overnight supervision")
    parser.add_argument("--check-now", action="store_true", help="Run one health check and auto-heal cycle immediately")
    parser.add_argument("--daemon", action="store_true", help="Run continuous periodic health check loop")
    parser.add_argument("--status", action="store_true", help="Show current pipeline, stage, tmux, and cron status")
    parser.add_argument("--stop", action="store_true", help="Stop the watchdog and pipeline tmux sessions")
    parser.add_argument("--install-cron", action="store_true", help="Install idempotent hourly crontab entry")
    parser.add_argument("--uninstall-cron", action="store_true", help="Remove managed crontab entry")
    parser.add_argument("--cron-schedule", type=str, default="0 * * * *", help="Cron schedule expression (default: '0 * * * *' hourly)")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_CHECK_INTERVAL_SECONDS, help="Daemon check interval in seconds (default: 3600)")
    parser.add_argument("--stall-timeout-seconds", type=int, default=DEFAULT_STALL_TIMEOUT_SECONDS, help="Log inactivity timeout before declaring stall (default: 2700)")
    parser.add_argument("--enable-ai-healer", dest="enable_ai_healer", action="store_true", default=True, help="Invoke agy healer prompt on crash")
    parser.add_argument("--no-ai-healer", dest="enable_ai_healer", action="store_false", help="Disable agy AI healer invocation (deterministic repairs only)")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_dir = resolve_repo_dir(args.repo_dir)
    state = load_watchdog_state(repo_dir)

    seed_file = args.seed or state.get("seed_file") or "platodoom specification"
    run_id = args.run_id or state.get("run_id") or discover_latest_run_id(repo_dir)

    if args.uninstall_cron and not (args.start or args.check_now or args.daemon or args.status or args.stop):
        res = uninstall_hourly_cron(repo_dir)
        print(json.dumps(res, indent=2) if args.json else f"Uninstalled watchdog cron (removed={res['removed']})")
        return 0

    if args.stop:
        for sess in (args.watchdog_tmux_session, args.tmux_session):
            if is_tmux_session_alive(sess):
                subprocess.run(["tmux", "kill-session", "-t", sess], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if args.uninstall_cron:
            uninstall_hourly_cron(repo_dir)
        payload = {"ok": True, "stopped_sessions": [args.watchdog_tmux_session, args.tmux_session]}
        print(json.dumps(payload, indent=2) if args.json else f"Stopped tmux sessions: {', '.join(payload['stopped_sessions'])}")
        return 0

    if args.install_cron and not args.start:
        if not run_id:
            print("ERROR: --run-id is required (or must exist in .sandcastle/logs) to install cron.", file=sys.stderr)
            return 1
        res = install_hourly_cron(repo_dir, seed_file, run_id, cron_schedule=args.cron_schedule)
        print(json.dumps(res, indent=2) if args.json else f"Installed hourly cron: {res['cron_line']}")
        if not (args.check_now or args.daemon or args.status):
            return 0

    if args.start:
        if not run_id:
            print("ERROR: --run-id is required when starting watchdog resume.", file=sys.stderr)
            return 1
        repairs = perform_deterministic_repairs(repo_dir, run_id)
        proc_info = is_pipeline_process_running(repo_dir, run_id, seed_file, args.tmux_session)
        if not proc_info["pipeline_alive"]:
            launch_res = start_pipeline_in_tmux(
                repo_dir=repo_dir,
                seed_file=seed_file,
                run_id=run_id,
                pipeline_script=args.pipeline_script,
                tmux_session=args.tmux_session,
            )
        else:
            launch_res = {"ok": True, "already_running": True, "tmux_session": args.tmux_session}

        daemon_res = start_watchdog_daemon_in_tmux(
            repo_dir=repo_dir,
            seed_file=seed_file,
            run_id=run_id,
            interval_seconds=args.interval_seconds,
            stall_timeout_seconds=args.stall_timeout_seconds,
            tmux_watchdog_session=args.watchdog_tmux_session,
            enable_ai_healer=args.enable_ai_healer,
        )
        cron_res = None
        if args.install_cron:
            cron_res = install_hourly_cron(repo_dir, seed_file, run_id, cron_schedule=args.cron_schedule)

        status_res = get_watchdog_status(
            repo_dir=repo_dir,
            seed_file=seed_file,
            run_id=run_id,
            tmux_session=args.tmux_session,
            tmux_watchdog_session=args.watchdog_tmux_session,
        )
        out = {
            "ok": True,
            "action": "STARTED",
            "repairs_applied": repairs,
            "pipeline_tmux": launch_res,
            "watchdog_tmux": daemon_res,
            "cron": cron_res,
            "status": status_res,
        }
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            prog = status_res["progress"]
            print(
                f"[overnight-watchdog] Started supervision for {seed_file} (runId={run_id}):\n"
                f"  - Pipeline tmux session : {args.tmux_session} (alive={status_res['tmux_pipeline_alive']})\n"
                f"  - Watchdog tmux session : {args.watchdog_tmux_session} (alive={status_res['tmux_watchdog_alive']}, interval={args.interval_seconds}s)\n"
                f"  - Hourly crontab active : {status_res['cron_installed']}\n"
                f"  - Current stage         : Stage {prog['current_stage']} ({prog['reason']})\n"
                f"  - Settled in stage      : {', '.join(prog['settled_workers']) or 'none'}\n"
                f"  - Remaining in stage    : {', '.join(prog['pending_workers']) or 'none'}"
            )
        return 0

    if args.check_now:
        res = run_health_check_and_heal(
            repo_dir=repo_dir,
            seed_file=seed_file,
            run_id=run_id,
            pipeline_script=args.pipeline_script,
            tmux_session=args.tmux_session,
            stall_timeout_seconds=args.stall_timeout_seconds,
            enable_ai_healer=args.enable_ai_healer,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"[overnight-watchdog] check-now -> action={res.get('action')} runId={res.get('run_id')}")
        return 0 if res.get("ok", False) else 1

    if args.daemon:
        append_watchdog_log(
            repo_dir,
            "WATCHDOG_DAEMON_STARTED",
            {
                "run_id": run_id,
                "seed_file": seed_file,
                "interval_seconds": args.interval_seconds,
                "stall_timeout_seconds": args.stall_timeout_seconds,
            },
        )
        while True:
            res = run_health_check_and_heal(
                repo_dir=repo_dir,
                seed_file=seed_file,
                run_id=run_id,
                pipeline_script=args.pipeline_script,
                tmux_session=args.tmux_session,
                stall_timeout_seconds=args.stall_timeout_seconds,
                enable_ai_healer=args.enable_ai_healer,
            )
            print(f"[{utc_now_iso()}] [watchdog-daemon] action={res.get('action')} run_id={res.get('run_id')}", flush=True)
            if res.get("action") == "COMPLETED":
                break
            time.sleep(max(5, int(args.interval_seconds)))
        return 0

    # Default to --status
    status_res = get_watchdog_status(
        repo_dir=repo_dir,
        seed_file=seed_file,
        run_id=run_id,
        tmux_session=args.tmux_session,
        tmux_watchdog_session=args.watchdog_tmux_session,
    )
    if args.json:
        print(json.dumps(status_res, indent=2))
    else:
        prog = status_res["progress"]
        print(
            f"Overnight Pipeline Watchdog Status:\n"
            f"  Repo Directory        : {status_res['repo_dir']}\n"
            f"  Seed Specification    : {status_res['seed_file']}\n"
            f"  Active Run ID         : {status_res['run_id']}\n"
            f"  Current Stage         : Stage {prog.get('current_stage')} ({prog.get('status')} — {prog.get('reason')})\n"
            f"  Settled Stage Workers : {', '.join(prog.get('settled_workers', [])) or 'none'}\n"
            f"  Pending Stage Workers : {', '.join(prog.get('pending_workers', [])) or 'none'}\n"
            f"  Pipeline tmux Alive   : {status_res['tmux_pipeline_alive']} (session={args.tmux_session}, pids={status_res['pipeline_process']['matched_pids']})\n"
            f"  Watchdog tmux Alive   : {status_res['tmux_watchdog_alive']} (session={args.watchdog_tmux_session})\n"
            f"  Hourly Cron Installed : {status_res['cron_installed']}\n"
            f"  Latest Log Activity   : {status_res['latest_log'].get('latest_log_file')} ({status_res['latest_log'].get('age_seconds')}s ago)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
