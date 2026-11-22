"""
Zero-Trust Security Guardrail, Secret Denylist, .env Redaction, and 3-Tier Access Control for platodoom.
Document ID: platodoom_security_857_v1
Authoring Agent: storage-implementer
Governing Standards: refined_specification_857 (FR-5.1, FR-5.2, FR-5.4, AC-3, AC-4) & RULE[user_global]
"""

from __future__ import annotations

import fnmatch
import hmac
import os
import re
import secrets
import shlex
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

DEFAULT_SECRET_PATH_PATTERNS: Tuple[str, ...] = (
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
    "*/.mcp-colab/*",
    "*/.mcp-colab/token.json",
    "*/.mcp-colab/credentials.json",
    "*/.platodoom/config.json",
    "*/.platodoom/auth/*",
    "*/.platodoom/sessions/*/endpoint.json",
    "*/.sandcastle/colab_qwen_endpoint.json",
    "*/colab_qwen_endpoint.json",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
)

SECRET_ENDPOINT_KEYS: Tuple[str, ...] = (
    "bearer_token",
    "auth_token",
    "api_key",
    "tunnel_token",
    "hf_token",
    "ngrok_authtoken",
    "ngrok_token",
    "headers",
    "authorization",
)

VALID_ACCESS_MODES: Tuple[str, ...] = (
    "read-only",
    "workspace-write",
    "confirm-writes",
    "full-access",
)

VALID_WORKFLOW_MODES: Tuple[str, ...] = (
    "plan",
    "build",
)

PLAN_MODE_FORBIDDEN_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file"})

FORBIDDEN_PLAN_OPERATORS: Tuple[str, ...] = (
    "&&",
    "||",
    ";",
    "|",
    ">",
    ">>",
    "<",
    "$(",
    "`",
    "\n",
)

PLAN_MODE_FORBIDDEN_BINARIES: frozenset[str] = frozenset(
    {
        "rm",
        "rmdir",
        "unlink",
        "mv",
        "cp",
        "mkdir",
        "touch",
        "chmod",
        "chown",
        "ln",
        "truncate",
        "dd",
        "mkfs",
        "tee",
        "patch",
        "install",
        "perl",
        "ruby",
        "python",
        "python3",
        "node",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "tsx",
        "pip",
        "pip3",
        "pytest",
        "make",
        "cmake",
        "cargo",
        "go",
        "sh",
        "bash",
        "zsh",
        "fish",
        "curl",
        "wget",
        "tar",
        "unzip",
        "zip",
    }
)

PLAN_MODE_ALLOWED_BINARIES: frozenset[str] = frozenset(
    {
        "ls",
        "pwd",
        "which",
        "whoami",
        "uname",
        "cat",
        "head",
        "tail",
        "wc",
        "rg",
        "grep",
        "egrep",
        "fgrep",
        "find",
        "stat",
        "file",
        "du",
        "df",
        "git",
        "true",
        "echo",
        "date",
        "sed",
        "awk",
    }
)

PLAN_MODE_ALLOWED_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "status",
        "diff",
        "log",
        "show",
        "rev-parse",
        "ls-files",
        "remote",
        "branch",
    }
)

_SAFE_ENV_TEMPLATE_FILENAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.defaults",
    ".env.dist",
}

_SECRET_BASENAMES = {
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "authorized_keys",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "colab_qwen_endpoint.json",
    "secring.gpg",
    "credentials.json",
    "token.json",
}

_SECRET_EXTENSIONS = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".keystore",
}

_ENV_SECRET_LINE_RE = re.compile(
    r"(?i)^((?:[^\n:]*:\d+:)?\s*(?:export\s+)?[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|AUTH|BEARER|COOKIE)[A-Z0-9_]*\s*=\s*)(.+)$"
)

_INLINE_TOKEN_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-plato-[0-9a-fA-F]{12,64}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\.[0-9A-Za-z_-]{16,}"),
    re.compile(r"\b2[A-Za-z0-9]{18,}_[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
)

_SAFE_READONLY_COMMANDS = {
    "git status",
    "git diff",
    "git log",
    "git branch",
    "git show",
    "git rev-parse",
    "git remote",
    "ls",
    "pwd",
    "which",
    "whoami",
    "uname",
    "cat",
    "head",
    "tail",
    "wc",
    "rg",
    "find",
    "stat",
    "file",
    "echo",
    "true",
}

_TIER3_DANGEROUS_PATTERNS: Tuple[str, ...] = (
    "rm -rf /",
    "rm -rf ~",
    "rm -rf *",
    "rm -rf .",
    "sudo ",
    "su -",
    "chmod 777",
    "chown ",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "killall ",
    "pkill -9",
    "git push --force",
    "git push -f",
    "git reset --hard",
    "git clean -fd",
    "curl -x post",
    "curl --data",
    "curl -d ",
    "curl -t ",
    "curl --upload-file",
    "wget --post",
    "nc ",
    "netcat ",
    "gsutil cp",
    "gsutil rsync",
    "gcloud storage cp",
    "aws s3 cp",
    "scp ",
    "rsync ",
)


def generate_ephemeral_tunnel_token() -> str:
    """Returns 'sk-plato-' + secrets.token_hex(32) (256-bit cryptographically random secret, FR-5.4)."""
    return f"sk-plato-{secrets.token_hex(32)}"


def verify_bearer_authorization(
    auth_header: Optional[str],
    expected_token: Optional[str],
) -> bool:
    """
    Constant-time comparison via hmac.compare_digest (FR-5.4, AC-3).
    If expected_token is None or empty or 'EMPTY', returns True (legacy local-only fallback).
    Otherwise requires auth_header == f'Bearer {expected_token}'.
    """
    if expected_token is None:
        return True
    cleaned_expected = str(expected_token).strip()
    if not cleaned_expected or cleaned_expected == "EMPTY":
        return True
    if not auth_header or not isinstance(auth_header, str):
        return False
    stripped_header = auth_header.strip()
    if not stripped_header.lower().startswith("bearer "):
        return False
    provided_token = stripped_header[7:].strip()
    if not provided_token:
        return False
    return hmac.compare_digest(
        provided_token.encode("utf-8"),
        cleaned_expected.encode("utf-8"),
    )


def strip_secrets_for_remote_sync(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a shallow copy of record with 'bearer_token', 'auth_token', 'api_key', and 'tunnel_token'
    set to None and 'auth_required': True before any optional remote GCS upload (FR-5.4).
    """
    if not isinstance(record, dict):
        return {}
    sanitized = dict(record)
    for key in SECRET_ENDPOINT_KEYS:
        if key in sanitized or key == "bearer_token":
            sanitized[key] = None
    sanitized["auth_required"] = True
    return sanitized


sanitize_endpoint_record_for_cloud_sync = strip_secrets_for_remote_sync


def _normalize_candidate_paths(
    target_path: Path | str,
    home_dir: Optional[Path | str] = None,
) -> Tuple[List[str], str]:
    raw_str = str(target_path).strip()
    effective_home = Path(home_dir).expanduser().resolve() if home_dir is not None else Path.home().resolve()
    if raw_str.startswith("~/") or raw_str == "~":
        expanded = effective_home / raw_str[2:] if raw_str.startswith("~/") else effective_home
    else:
        expanded = Path(raw_str).expanduser()

    candidates: List[str] = []
    for p in (Path(raw_str), expanded):
        posix_raw = p.as_posix()
        if not posix_raw.startswith("/"):
            posix_raw = "/" + posix_raw
        candidates.append(posix_raw)
        try:
            resolved_posix = p.resolve().as_posix()
            if resolved_posix not in candidates:
                candidates.append(resolved_posix)
        except OSError:
            pass

    basename = expanded.name
    return candidates, basename


def is_denylisted_secret_path(
    target_path: Path | str,
    home_dir: Optional[Path | str] = None,
    extra_globs: Optional[List[str]] = None,
) -> bool:
    """
    Returns True if target_path matches DEFAULT_SECRET_PATH_PATTERNS or extra_globs
    (note: .env / .env.* files are handled via redact_env_secrets or blocked if raw read is requested).
    """
    if not target_path or not str(target_path).strip():
        return False

    candidates, basename = _normalize_candidate_paths(target_path, home_dir=home_dir)
    lower_base = basename.lower()

    if lower_base in _SECRET_BASENAMES:
        return True
    if (
        lower_base.startswith(("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"))
        or "service-account" in lower_base
        or "service_account" in lower_base
        or (lower_base.endswith(".json") and "credentials" in lower_base)
    ):
        return True

    _, ext = os.path.splitext(lower_base)
    if ext in _SECRET_EXTENSIONS:
        return True

    effective_home = Path(home_dir).expanduser().resolve() if home_dir is not None else Path.home().resolve()
    patterns: List[str] = list(DEFAULT_SECRET_PATH_PATTERNS)
    if extra_globs:
        for g in extra_globs:
            if not g or not str(g).strip():
                continue
            g_str = str(g).strip().replace("\\", "/")
            if g_str.startswith("~/"):
                patterns.append(f"{effective_home.as_posix()}/{g_str[2:]}")
                patterns.append(f"*/{g_str[2:]}")
            else:
                patterns.append(g_str)
                if not g_str.startswith("*"):
                    patterns.append(f"*/{g_str.lstrip('/')}")

    for cand in candidates:
        cand_lower = cand.lower()
        if (
            "/.ssh/" in cand_lower
            or cand_lower.endswith("/.ssh")
            or "/.gnupg/" in cand_lower
            or cand_lower.endswith("/.gnupg")
            or "/.aws/" in cand_lower
            or cand_lower.endswith("/.aws")
            or "/.config/gcloud/" in cand_lower
            or cand_lower.endswith("/.config/gcloud")
            or "/.azure/" in cand_lower
            or cand_lower.endswith("/.azure")
            or "/.mcp-colab/" in cand_lower
            or cand_lower.endswith("/.mcp-colab")
            or "/.platodoom/auth/" in cand_lower
            or cand_lower.endswith("/.platodoom/config.json")
            or cand_lower.endswith("/.kube/config")
            or cand_lower.endswith("/.docker/config.json")
        ):
            return True

        for pat in patterns:
            pat_lower = pat.lower()
            if fnmatch.fnmatch(cand_lower, pat_lower) or fnmatch.fnmatch(lower_base, pat_lower):
                return True

    return False


def is_env_secret_file(target_path: Path | str) -> bool:
    """Returns True if filename is '.env' or starts with '.env.' (excluding '.env.example', '.env.sample', '.env.template')."""
    if not target_path:
        return False
    name = Path(str(target_path)).name.lower()
    if name in _SAFE_ENV_TEMPLATE_FILENAMES:
        return False
    return name == ".env" or name.startswith(".env.")


def redact_env_secrets(text: str) -> str:
    """Redacts values of *_KEY, *_TOKEN, *_SECRET, *_PASSWORD, *_CREDENTIAL, *_PRIVATE, *_AUTH assignments as '<REDACTED_SECRET>'."""
    if not text:
        return text

    redacted_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        line_ending = ""
        if line.endswith("\r\n"):
            line_ending = "\r\n"
            body = line[:-2]
        elif line.endswith("\n"):
            line_ending = "\n"
            body = line[:-1]
        else:
            body = line

        stripped = body.lstrip()
        if not stripped.startswith("#"):
            match = _ENV_SECRET_LINE_RE.match(body)
            if match:
                prefix = match.group(1)
                raw_val = match.group(2).strip()
                if raw_val and raw_val not in ("", '""', "''", "null", "None", "<REDACTED_SECRET>"):
                    body = f"{prefix}<REDACTED_SECRET>"
                    redacted_lines.append(body + line_ending)
                    continue

        for tok_re in _INLINE_TOKEN_PATTERNS:
            body = tok_re.sub("<REDACTED_SECRET>", body)

        redacted_lines.append(body + line_ending)

    return "".join(redacted_lines)


def check_secret_path_and_redact(
    target_path: Path | str,
    content: Optional[str] = None,
    home_dir: Optional[Path | str] = None,
    extra_denylist_globs: Optional[List[str]] = None,
    interactive_approve_fn: Optional[Callable[[str], bool]] = None,
    extra_globs: Optional[List[str]] = None,
    redact_env: bool = True,
    allow_secret_files: bool = False,
) -> Dict[str, Any]:
    """
    Enforces FR-5.1 & AC-3:
    - If target_path is in is_denylisted_secret_path(...):
      calls interactive_approve_fn(str(target_path)) if provided (or allows if allow_secret_files=True); if False/None, returns:
      {'allowed': False, 'blocked': True, 'error_code': 'SECRET_PATH_BLOCKED', 'redacted': False, 'content': None,
       'reason': f'[SECURITY BLOCKED: Access to sensitive credential file {target_path} is denied by platodoom Zero-Trust Guardrail.]'}
    - If target_path is an .env secret file and content is provided:
      redacts secret values via redact_env_secrets(content) and returns:
      {'allowed': True, 'blocked': False, 'error_code': None, 'redacted': True, 'content': redacted_text, 'reason': 'Redacted .env secret values'}
    - Otherwise returns {'allowed': True, 'blocked': False, 'error_code': None, 'redacted': False, 'content': content, 'reason': None}.
    """
    effective_globs = extra_denylist_globs if extra_denylist_globs is not None else extra_globs
    if is_denylisted_secret_path(target_path, home_dir=home_dir, extra_globs=effective_globs):
        if allow_secret_files:
            return {
                "allowed": True,
                "blocked": False,
                "error_code": None,
                "redacted": False,
                "content": redact_env_secrets(content) if (content is not None and redact_env) else content,
                "reason": "Allowed via allow_secret_files=True",
            }
        if interactive_approve_fn is not None:
            try:
                if bool(interactive_approve_fn(str(target_path))):
                    return {
                        "allowed": True,
                        "blocked": False,
                        "error_code": None,
                        "redacted": False,
                        "content": redact_env_secrets(content) if (content is not None and redact_env) else content,
                        "reason": "Approved interactively by operator",
                    }
            except Exception:
                pass
        return {
            "allowed": False,
            "blocked": True,
            "error_code": "SECRET_PATH_BLOCKED",
            "redacted": False,
            "content": None,
            "reason": (
                f"[SECURITY BLOCKED: Access to sensitive credential file {target_path} "
                "is denied by platodoom Zero-Trust Guardrail.]"
            ),
        }

    if redact_env and is_env_secret_file(target_path):
        if content is not None:
            return {
                "allowed": True,
                "blocked": False,
                "error_code": None,
                "redacted": True,
                "content": redact_env_secrets(content),
                "reason": "Redacted .env secret values",
            }
        return {
            "allowed": True,
            "blocked": False,
            "error_code": None,
            "redacted": True,
            "content": None,
            "reason": "Redacted .env secret values",
        }

    return {
        "allowed": True,
        "blocked": False,
        "error_code": None,
        "redacted": False,
        "content": content,
        "reason": None,
    }


def _is_path_within_roots(target_path: Path, workspace_roots: List[Path]) -> bool:
    if not workspace_roots:
        return True
    try:
        resolved_target = target_path.expanduser().resolve()
    except OSError:
        resolved_target = target_path.expanduser()
    for root in workspace_roots:
        try:
            resolved_root = root.expanduser().resolve()
        except OSError:
            resolved_root = root.expanduser()
        if resolved_target == resolved_root or resolved_root in resolved_target.parents:
            return True
    return False


def _command_references_secret_path(
    tokens: List[str],
    cwd: Optional[Path | str] = None,
) -> Optional[str]:
    """Returns the offending token if any shell argument references a denylisted secret path or .env file."""
    base_dir = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    for tok in tokens:
        raw = str(tok or "").strip()
        if not raw:
            continue
        candidates_to_check: List[str] = [raw]
        if "=" in raw and not raw.startswith("="):
            rhs = raw.split("=", 1)[1].strip()
            if rhs:
                candidates_to_check.append(rhs)
        for cand in candidates_to_check:
            if not cand or cand.startswith("-"):
                continue
            p = Path(cand).expanduser()
            resolved_cand = p if p.is_absolute() else (base_dir / p)
            if is_denylisted_secret_path(cand) or is_denylisted_secret_path(resolved_cand):
                return cand
            if is_env_secret_file(cand) or is_env_secret_file(resolved_cand):
                return cand
    return None


def validate_plan_mode_command(command: str, cwd: Optional[Path | str] = None) -> Tuple[bool, str]:
    """
    Strictly validates a shell command for Plan Mode (workflow_mode == 'plan' or access_mode == 'read-only').
    Blocks any file modification, deletion, redirection, pipe, command chain, interpreter execution,
    or mutating git subcommand, returning (False, 'PERMISSION_DENIED_READ_ONLY: ...').
    """
    raw = (command or "").strip()
    if not raw:
        return False, "PERMISSION_DENIED_READ_ONLY: Empty shell command is not permitted in Plan Mode."

    for op in FORBIDDEN_PLAN_OPERATORS:
        if op in raw:
            op_disp = "\\n" if op == "\n" else op
            return (
                False,
                f"PERMISSION_DENIED_READ_ONLY: Shell operator '{op_disp}' is blocked in Plan Mode (`/plan`). Switch to `/build` to execute chained, piped, or redirecting commands.",
            )

    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError as exc:
        return False, f"PERMISSION_DENIED_READ_ONLY: Malformed shell quoting in Plan Mode: {exc}"

    # Strip leading FOO=bar environment variable assignments
    cmd_tokens = list(tokens)
    while cmd_tokens and "=" in cmd_tokens[0] and not cmd_tokens[0].startswith("-"):
        lhs = cmd_tokens[0].split("=", 1)[0]
        if lhs and lhs.replace("_", "").isalnum():
            cmd_tokens.pop(0)
        else:
            break

    if not cmd_tokens:
        return False, "PERMISSION_DENIED_READ_ONLY: Empty command after environment variable prefix in Plan Mode."

    binary = Path(cmd_tokens[0]).name.lower()
    args_tokens = cmd_tokens[1:]

    if binary in PLAN_MODE_FORBIDDEN_BINARIES or binary not in PLAN_MODE_ALLOWED_BINARIES:
        return (
            False,
            f"PERMISSION_DENIED_READ_ONLY: Command '{binary}' is blocked in Plan Mode (`/plan`). Only read-only inspection commands (ls, pwd, rg, cat, head, tail, find, git status/diff/log) are permitted.",
        )

    if binary == "sed":
        if any(t == "-i" or t.startswith("-i") or t == "--in-place" or t.startswith("--in-place=") for t in args_tokens):
            return False, "PERMISSION_DENIED_READ_ONLY: 'sed -i' in-place file mutation is blocked in Plan Mode."

    if binary == "awk":
        if any(t in ("-i", "inplace") or t.startswith("-i") for t in args_tokens):
            return False, "PERMISSION_DENIED_READ_ONLY: 'awk -i inplace' file mutation is blocked in Plan Mode."

    if binary == "find":
        if any(t in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls") for t in args_tokens):
            return False, "PERMISSION_DENIED_READ_ONLY: Mutating 'find' flags (-delete/-exec) are blocked in Plan Mode."

    if binary == "git":
        # Find the git subcommand (skipping leading global flags like -C <dir> or --no-pager)
        sub_idx = -1
        idx = 0
        while idx < len(args_tokens):
            tok = args_tokens[idx]
            if tok in ("-C", "-c", "--git-dir", "--work-tree"):
                idx += 2
                continue
            if tok.startswith("-"):
                idx += 1
                continue
            sub_idx = idx
            break
        if sub_idx < 0:
            return False, "PERMISSION_DENIED_READ_ONLY: 'git' requires a read-only subcommand (status, diff, log, show, branch) in Plan Mode."
        subcmd = args_tokens[sub_idx].lower()
        sub_args = args_tokens[sub_idx + 1 :]
        if subcmd not in PLAN_MODE_ALLOWED_GIT_SUBCOMMANDS:
            return (
                False,
                f"PERMISSION_DENIED_READ_ONLY: 'git {subcmd}' is blocked in Plan Mode (`/plan`). Allowed git subcommands: status, diff, log, show, rev-parse, ls-files, remote, branch.",
            )
        if subcmd == "remote":
            if any(a not in ("-v", "--verbose") for a in sub_args):
                return False, "PERMISSION_DENIED_READ_ONLY: Mutating 'git remote' subcommands are blocked in Plan Mode."
        if subcmd == "branch":
            if any(a in ("-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy") for a in sub_args):
                return False, "PERMISSION_DENIED_READ_ONLY: Mutating 'git branch' flags are blocked in Plan Mode."

    secret_tok = _command_references_secret_path(args_tokens, cwd=cwd)
    if secret_tok is not None:
        return (
            False,
            f"SECRET_PATH_BLOCKED: Shell access to sensitive path '{secret_tok}' is denied by platodoom Zero-Trust Guardrail.",
        )

    return True, "Read-only command permitted in Plan Mode."


def classify_shell_command_tier(
    command: str,
    cwd: Optional[Path | str] = None,
    workspace_roots: Optional[List[Path]] = None,
) -> int:
    """
    Classifies shell command into:
    - Tier 1 (1): Safe read-only commands validated by validate_plan_mode_command()
    - Tier 3 (3): Destructive, privilege-escalation, force-push, or outbound upload commands ('rm -rf', 'sudo ', 'git push --force', 'git reset --hard', 'mkfs', 'dd if=', 'curl -X POST', 'gsutil cp', 'gcloud storage cp') or when cwd is outside workspace_roots
    - Tier 2 (2): Standard workspace build/test/dev commands inside workspace_roots.
    """
    cleaned = (command or "").strip()
    if not cleaned:
        return 2

    lower_cmd = cleaned.lower()
    for pattern in _TIER3_DANGEROUS_PATTERNS:
        if pattern in lower_cmd:
            return 3
    if re.search(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\b", lower_cmd):
        return 3

    resolved_cwd = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    roots = [Path(r).expanduser().resolve() for r in workspace_roots] if workspace_roots else []
    if roots and not _is_path_within_roots(resolved_cwd, roots):
        return 3

    try:
        parsed_tokens = shlex.split(cleaned, posix=True)
    except ValueError:
        parsed_tokens = cleaned.split()
    if _command_references_secret_path(parsed_tokens, cwd=resolved_cwd) is not None:
        return 3

    ok_plan, _ = validate_plan_mode_command(cleaned, cwd=resolved_cwd)
    if ok_plan:
        return 1

    return 2


def check_tool_permission(
    tool_name: str,
    args: Optional[Mapping[str, Any]] = None,
    access_mode: str = "workspace-write",
    workspace_roots: Optional[List[Path | str] | Path | str] = None,
    cwd: Optional[Path | str] = None,
    interactive_confirm_fn: Optional[Callable[[str], bool]] = None,
    tool_args: Optional[Mapping[str, Any]] = None,
    workspace_root: Optional[Path | str] = None,
    workspace_dir: Optional[Path | str] = None,
    confirm_callback: Optional[Callable[[str], bool]] = None,
    audit_logger: Optional[Any] = None,
    extra_denylist_globs: Optional[List[str]] = None,
    allow_secret_files: bool = False,
    workflow_mode: str = "build",
) -> Tuple[bool, str]:
    """
    Enforces FR-3.1, FR-3.2, FR-5.2 & AC-3/AC-4 across Plan Mode ('plan' / 'read-only')
    and Build Mode ('workspace-write' / 'confirm-writes' / 'full-access'):
    Returns (allowed: bool, diagnostic_message: str).
    Diagnostic messages include canonical error codes ('SECRET_PATH_BLOCKED',
    'PERMISSION_DENIED_READ_ONLY', 'PERMISSION_DENIED_OUTSIDE_WORKSPACE') when denied.
    """
    norm_workflow = (workflow_mode or "build").strip().lower()
    norm_mode = (access_mode or "workspace-write").strip().lower()
    requires_write_confirm = norm_mode in ("confirm-writes", "confirm_writes")
    if norm_mode in ("yolo", "--yolo", "full"):
        norm_mode = "full-access"
    elif requires_write_confirm:
        norm_mode = "workspace-write"
    elif norm_mode == "read_only":
        norm_mode = "read-only"

    if norm_workflow == "plan":
        norm_mode = "read-only"

    if norm_mode not in VALID_ACCESS_MODES:
        return (
            False,
            f"PERMISSION_DENIED_INVALID_MODE: Invalid access_mode '{access_mode}'. Must be one of {VALID_ACCESS_MODES}.",
        )

    eff_confirm_fn = interactive_confirm_fn if interactive_confirm_fn is not None else confirm_callback
    eff_cwd_input = cwd if cwd is not None else workspace_dir
    resolved_cwd = Path(eff_cwd_input).expanduser().resolve() if eff_cwd_input is not None else Path.cwd().resolve()
    eff_roots_input = (
        workspace_roots
        if workspace_roots is not None
        else (workspace_root if workspace_root is not None else workspace_dir)
    )
    if isinstance(eff_roots_input, (Path, str)):
        roots: List[Path] = [Path(eff_roots_input).expanduser().resolve()]
    elif eff_roots_input:
        roots = [Path(r).expanduser().resolve() for r in eff_roots_input]
    else:
        roots = [resolved_cwd]
    effective_args = args if args is not None else tool_args
    safe_args = dict(effective_args) if isinstance(effective_args, Mapping) else {}

    def _resolve_arg_path(raw_p: Optional[str]) -> Optional[Path]:
        if not raw_p or not str(raw_p).strip():
            return None
        p = Path(str(raw_p).strip()).expanduser()
        if not p.is_absolute():
            p = resolved_cwd / p
        try:
            return p.resolve()
        except OSError:
            return p

    # 1. Read-only navigation, discovery, and inspection tools (Tier 1)
    if tool_name in (
        "read_file",
        "list_dir",
        "find_files",
        "grep_search",
        "use_skill",
        "list_mcp_tools",
        "inspect_mcp_tool",
    ):
        target_p = _resolve_arg_path(safe_args.get("path"))
        if (
            target_p is not None
            and not allow_secret_files
            and is_denylisted_secret_path(target_p, extra_globs=extra_denylist_globs)
        ):
            if eff_confirm_fn is not None:
                try:
                    if bool(eff_confirm_fn(f"Allow read access to protected path '{target_p}'?")):
                        return True, f"Allowed read access to '{target_p}' via operator confirmation."
                except Exception:
                    pass
            return (
                False,
                f"[SECURITY BLOCKED: SECRET_PATH_BLOCKED — Access to sensitive credential file {safe_args.get('path')} is denied by platodoom Zero-Trust Guardrail.]",
            )
        return True, f"Allowed Tier 1 tool '{tool_name}' under '{norm_mode}' mode."

    # 2. Local file mutation tools (write_file, edit_file)
    if tool_name in ("write_file", "edit_file"):
        if norm_mode == "read-only" or norm_workflow == "plan":
            return (
                False,
                f"[PERMISSION DENIED: PERMISSION_DENIED_READ_ONLY — Tool '{tool_name}' is disabled in Plan Mode ('read-only' access mode). Run '/build' or '/access workspace-write' to enable file modifications.]",
            )

        target_p = _resolve_arg_path(safe_args.get("path"))
        if target_p is None:
            return False, f"Tool '{tool_name}' requires a non-empty 'path' argument."

        if not allow_secret_files and is_denylisted_secret_path(target_p, extra_globs=extra_denylist_globs):
            if eff_confirm_fn is not None:
                try:
                    if bool(eff_confirm_fn(f"Allow modifying protected file '{target_p}'?")):
                        return True, f"Allowed modifying '{target_p}' via operator confirmation."
                except Exception:
                    pass
            return (
                False,
                f"[SECURITY BLOCKED: SECRET_PATH_BLOCKED — Modifying sensitive credential file {target_p} is denied by platodoom Zero-Trust Guardrail.]",
            )

        if norm_mode == "workspace-write":
            if _is_path_within_roots(target_p, roots):
                if requires_write_confirm and eff_confirm_fn is not None:
                    try:
                        if bool(eff_confirm_fn(f"Allow '{tool_name}' on workspace file '{target_p}'?")):
                            return True, f"Allowed workspace write to '{target_p}' via operator confirmation."
                    except Exception:
                        pass
                    return (
                        False,
                        f"[PERMISSION DENIED: PERMISSION_DENIED_USER_DECLINED — Operator declined '{tool_name}' on '{target_p}'.]",
                    )
                return True, f"Allowed Tier 2 workspace write to '{target_p}'."
            if eff_confirm_fn is not None:
                try:
                    if bool(
                        eff_confirm_fn(
                            f"Allow '{tool_name}' on path '{target_p}' outside workspace roots {[str(r) for r in roots]}?"
                        )
                    ):
                        return True, f"Allowed out-of-workspace write to '{target_p}' via operator confirmation."
                except Exception:
                    pass
            return (
                False,
                f"[PERMISSION DENIED: PERMISSION_DENIED_OUTSIDE_WORKSPACE — Target path '{target_p}' is outside registered workspace roots {[str(r) for r in roots]} in 'workspace-write' mode. Add the directory with '/dirs {target_p.parent}' or switch to '/access full-access'.]",
            )

        return True, f"Allowed Tier 3 write to '{target_p}' in 'full-access' mode."

    # 3. Shell command execution (run_command)
    if tool_name == "run_command":
        cmd_str = str(safe_args.get("command") or "").strip()
        cmd_cwd = _resolve_arg_path(safe_args.get("cwd")) or resolved_cwd

        if norm_mode == "read-only" or norm_workflow == "plan":
            ok_plan, plan_reason = validate_plan_mode_command(cmd_str, cwd=cmd_cwd)
            if not ok_plan:
                return (
                    False,
                    f"[PERMISSION DENIED: PERMISSION_DENIED_READ_ONLY — {plan_reason}]",
                )
            tier = classify_shell_command_tier(cmd_str, cwd=cmd_cwd, workspace_roots=roots)
            if tier == 1:
                return True, f"Allowed Tier 1 read-only shell command '{cmd_str}'."
            return (
                False,
                f"[PERMISSION DENIED: PERMISSION_DENIED_READ_ONLY — Shell command '{cmd_str}' (Tier {tier}) is blocked in Plan Mode ('read-only' access mode).]",
            )

        tier = classify_shell_command_tier(cmd_str, cwd=cmd_cwd, workspace_roots=roots)
        if norm_mode == "workspace-write":
            if tier in (1, 2) and _is_path_within_roots(cmd_cwd, roots):
                return True, f"Allowed Tier {tier} command inside workspace ('{cmd_cwd}')."
            if eff_confirm_fn is not None:
                try:
                    if bool(
                        eff_confirm_fn(
                            f"Allow Tier {tier} shell command '{cmd_str}' in '{cmd_cwd}'?"
                        )
                    ):
                        return True, f"Allowed Tier {tier} command via operator confirmation."
                except Exception:
                    pass
            err_tag = (
                "PERMISSION_DENIED_OUTSIDE_WORKSPACE"
                if not _is_path_within_roots(cmd_cwd, roots)
                else "PERMISSION_DENIED_TIER3_COMMAND"
            )
            return (
                False,
                f"[PERMISSION DENIED: {err_tag} — Command '{cmd_str}' (Tier {tier}, cwd='{cmd_cwd}') requires 'full-access' mode or explicit operator confirmation.]",
            )

        return True, f"Allowed Tier {tier} command in 'full-access' mode."

    # 4. Subagent & MCP execution tools
    if tool_name in ("invoke_subagent", "invoke_parallel_subagents", "call_mcp_tool"):
        if (norm_mode == "read-only" or norm_workflow == "plan") and tool_name == "call_mcp_tool":
            mcp_tool = str(safe_args.get("tool_name") or "").lower()
            if any(
                mcp_tool.startswith(prefix)
                for prefix in ("write_", "create_", "delete_", "update_", "drop_", "execute_")
            ):
                return (
                    False,
                    f"[PERMISSION DENIED: PERMISSION_DENIED_READ_ONLY — Mutating MCP tool '{mcp_tool}' is blocked in 'read-only' mode.]",
                )
        return True, f"Allowed tool '{tool_name}' under '{norm_mode}' mode."

    return True, f"Allowed tool '{tool_name}'."
