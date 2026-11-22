"""
User-Agnostic, Skippable First-Run Onboarding & Credential Management for platodoom.

Provides a seamless single-browser Google Colab sign-in plus clean terminal prompts
for optional HuggingFace and ngrok tokens (defaulting to free, zero-account Cloudflare
Quick Tunnels). Every step is skippable on first launch and revisitable anytime via
`platodoom auth` or `/auth` inside the REPL.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import webbrowser
from typing import Any, Callable, Dict, Mapping, Optional

from platodoom.config import (
    load_user_config,
    resolve_platodoom_config,
    resolve_user_config_path,
    save_user_config,
)

HF_TOKENS_URL = "https://huggingface.co/settings/tokens"
NGROK_TOKEN_URL = "https://dashboard.ngrok.com/get-started/your-authtoken"


def mask_secret_preview(secret: Optional[str]) -> str:
    """Returns a non-sensitive redacted preview of a token for terminal/JSON status display."""
    raw = str(secret or "").strip()
    if not raw:
        return "<not set>"
    if len(raw) <= 8:
        return "***"
    return f"{raw[:4]}...{raw[-4:]}"


def _resolve_colab_auth_profile_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    eff_env = env if env is not None else os.environ
    home_raw = eff_env.get("HOME")
    base_home = Path(home_raw).expanduser() if (home_raw and str(home_raw).strip()) else Path.home()
    return base_home / ".mcp-colab" / "chrome_profiles" / "colab_auth_mock"


def _has_colab_browser_session(env: Optional[Mapping[str, str]] = None) -> bool:
    profile_dir = _resolve_colab_auth_profile_dir(env=env)
    try:
        return (profile_dir / "Default" / "Cookies").is_file()
    except OSError:
        return False


def _has_google_drive_oauth_token(
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    eff_env = env if env is not None else os.environ
    home_raw = eff_env.get("HOME")
    base_home = Path(home_raw).expanduser() if (home_raw and str(home_raw).strip()) else Path.home()
    raw_tok = str(((config or {}).get("auth") or {}).get("google_token_path") or "~/.mcp-colab/token.json").strip()
    if raw_tok.startswith("~/"):
        tok_path = base_home / raw_tok[2:]
    else:
        tok_path = Path(raw_tok).expanduser()
    try:
        return tok_path.is_file()
    except OSError:
        return False


def get_auth_status(
    config: Optional[Dict[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """
    Inspects active authentication & onboarding state without ever returning raw secrets.
    """
    eff_env = dict(env) if env is not None else dict(os.environ)
    user_cfg = load_user_config(env=eff_env)
    resolved = dict(config) if isinstance(config, dict) else resolve_platodoom_config(env=eff_env)

    auth_user = user_cfg.get("auth") if isinstance(user_cfg.get("auth"), dict) else {}
    auth_resolved = resolved.get("auth") if isinstance(resolved.get("auth"), dict) else {}
    onboarding_cfg = user_cfg.get("onboarding") if isinstance(user_cfg.get("onboarding"), dict) else {}

    hf_env_name = str(auth_user.get("hf_token_env") or "PLATODOOM_HF_TOKEN")
    hf_from_env = str(
        eff_env.get(hf_env_name)
        or eff_env.get("PLATODOOM_HF_TOKEN")
        or eff_env.get("HF_TOKEN")
        or eff_env.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    hf_from_cfg = str(auth_user.get("hf_token") or "").strip()
    hf_effective = hf_from_env or hf_from_cfg or str(auth_resolved.get("hf_token") or "").strip()
    hf_source: Optional[str] = "env" if hf_from_env else ("config" if hf_from_cfg else None)

    ngrok_from_env = str(eff_env.get("NGROK_AUTHTOKEN") or "").strip()
    ngrok_from_cfg = str(auth_user.get("ngrok_authtoken") or "").strip()
    ngrok_effective = ngrok_from_env or ngrok_from_cfg or str(auth_resolved.get("ngrok_authtoken") or "").strip()
    ngrok_source: Optional[str] = "env" if ngrok_from_env else ("config" if ngrok_from_cfg else None)

    saved_tunnel = str(auth_user.get("tunnel_provider") or "").strip().lower()
    if ngrok_effective and saved_tunnel != "cloudflare":
        tunnel_provider = "ngrok"
    elif saved_tunnel in ("cloudflare", "ngrok"):
        tunnel_provider = saved_tunnel if (saved_tunnel == "cloudflare" or ngrok_effective) else "cloudflare"
    else:
        tunnel_provider = "ngrok" if ngrok_effective else "cloudflare"

    colab_logged_in = _has_colab_browser_session(env=eff_env)
    drive_oauth_configured = _has_google_drive_oauth_token(config=user_cfg, env=eff_env)
    onboarding_completed = bool(onboarding_cfg.get("completed", False))
    onboarding_skipped = bool(onboarding_cfg.get("skipped", False))

    return {
        "ok": True,
        "colab_logged_in": colab_logged_in,
        "colab_profile_path": str(_resolve_colab_auth_profile_dir(env=eff_env)),
        "drive_oauth_configured": drive_oauth_configured,
        "hf_token_configured": bool(hf_effective),
        "hf_token_source": hf_source,
        "hf_token_preview": mask_secret_preview(hf_effective),
        "ngrok_token_configured": bool(ngrok_effective),
        "ngrok_token_source": ngrok_source,
        "ngrok_token_preview": mask_secret_preview(ngrok_effective),
        "tunnel_provider": tunnel_provider,
        "onboarding_completed": onboarding_completed,
        "onboarding_skipped": onboarding_skipped,
        "needs_onboarding": not (onboarding_completed or onboarding_skipped or colab_logged_in),
        "config_path": str(resolve_user_config_path(env=eff_env)),
    }


def format_auth_status(status: Mapping[str, Any]) -> str:
    """Formats a clean, human-readable authentication summary with strictly masked secrets."""
    colab_mark = "✓ Signed in" if status.get("colab_logged_in") else "✗ Not signed in"
    hf_mark = (
        f"✓ Configured ({status.get('hf_token_preview')}, via {status.get('hf_token_source') or 'config'})"
        if status.get("hf_token_configured")
        else "○ Not set (optional — needed for gated HF models / higher rate limits)"
    )
    if status.get("ngrok_token_configured") and status.get("tunnel_provider") == "ngrok":
        tunnel_mark = f"✓ ngrok ({status.get('ngrok_token_preview')}, via {status.get('ngrok_token_source') or 'config'})"
    else:
        tunnel_mark = "✓ Cloudflare Quick Tunnel (free default, no account required)"

    lines = [
        "platodoom Authentication & Setup Status:",
        f"  Google Colab browser session : {colab_mark}",
        f"  HuggingFace API token        : {hf_mark}",
        f"  Tunnel provider              : {tunnel_mark}",
        f"  Config file (0600)           : {status.get('config_path')}",
        "",
        "Run `platodoom auth` or `/auth` anytime to sign in, add/update tokens, or switch tunnel settings.",
    ]
    return "\n".join(lines)


def should_run_first_run_onboarding(
    config: Optional[Dict[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    is_tty: Optional[bool] = None,
) -> bool:
    """
    Returns True iff this is an interactive TTY session where the user has neither
    completed/skipped onboarding nor signed into Google Colab yet.
    """
    eff_env = env if env is not None else os.environ
    if str(eff_env.get("PLATODOOM_SKIP_ONBOARDING") or "").strip().lower() in ("1", "true", "yes"):
        return False
    tty_ok = bool(is_tty) if is_tty is not None else bool(getattr(sys.stdin, "isatty", lambda: False)())
    if not tty_ok:
        return False
    status = get_auth_status(config=config, env=eff_env)
    return bool(status["needs_onboarding"])


def _default_colab_login() -> Dict[str, Any]:
    from scripts.provision_colab_serving import interactive_colab_login

    return interactive_colab_login()


def _default_open_url(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def reset_onboarding_state(*, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Resets `onboarding.completed` and `onboarding.skipped` in `<platodoom_home>/config.json`."""
    eff_env = dict(env) if env is not None else None
    cfg = load_user_config(env=eff_env)
    cfg["onboarding"] = {"completed": False, "skipped": False}
    cfg_path = save_user_config(cfg, env=eff_env)
    return {
        "ok": True,
        "config_path": str(cfg_path),
        "onboarding": cfg["onboarding"],
    }


def run_interactive_onboarding(
    config: Optional[Dict[str, Any]] = None,
    *,
    force: bool = False,
    input_fn: Optional[Callable[[str], str]] = None,
    output_fn: Callable[[str], None] = print,
    open_url_fn: Optional[Callable[[str], Any]] = None,
    colab_login_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """
    Runs the 3-step skippable onboarding & credential setup wizard:
      1. Single Google Colab browser login (saves cookies to colab_auth_mock)
      2. Optional HuggingFace API token prompt (with [o] to open browser token page)
      3. Optional ngrok authtoken prompt (defaults to free Cloudflare Quick Tunnel)

    Persists saved secrets exclusively to `<platodoom_home>/config.json` (`0600`)
    and marks `onboarding.completed` / `onboarding.skipped` so subsequent launches never nag.
    """
    eff_env_dict = dict(env) if env is not None else dict(os.environ)
    using_default_stdin = input_fn is None
    tty_ok = True if not using_default_stdin else bool(getattr(sys.stdin, "isatty", lambda: False)())

    if not tty_ok:
        status = get_auth_status(config=config, env=eff_env_dict)
        return {
            **status,
            "ran": False,
            "skipped": False,
            "reason": "non-interactive stdin",
        }

    initial_status = get_auth_status(config=config, env=eff_env_dict)
    if not force and not initial_status["needs_onboarding"]:
        return {
            **initial_status,
            "ran": False,
            "skipped": bool(initial_status.get("onboarding_skipped")),
            "reason": "already_configured_or_skipped",
        }

    read_line = input_fn or input
    open_url = open_url_fn or _default_open_url
    do_colab_login = colab_login_fn or _default_colab_login

    def _safe_prompt(prompt_text: str) -> str:
        raw = read_line(prompt_text)
        if raw is not None and "\x03" in raw:
            raise KeyboardInterrupt
        if raw is not None and "\x04" in raw:
            raise EOFError
        return str(raw or "").strip()

    user_cfg = load_user_config(env=eff_env_dict)
    user_cfg.setdefault("auth", {})
    user_cfg.setdefault("onboarding", {})

    output_fn("")
    output_fn("platodoom setup — all steps are optional. Press [s] to skip now; run `platodoom auth` or `/auth` anytime.")
    output_fn("")

    configured_anything = False
    skipped_all = False
    colab_step_ok = bool(initial_status.get("colab_logged_in"))

    # ------------------------------------------------------------------
    # Step 1/3: Single Google Colab Browser Sign-In
    # ------------------------------------------------------------------
    output_fn("[Step 1/3] Google Colab Browser Sign-In (powers free & Pro GPU sessions)")
    if initial_status.get("colab_logged_in"):
        output_fn("  Current status: ✓ Signed in")
        step1_choice = _safe_prompt("  [Enter] Keep current session | [r] Re-sign in via browser | [q] Skip remaining setup: ").lower()
        should_login = step1_choice in ("r", "relogin", "login", "y", "yes")
        if step1_choice in ("q", "quit", "skip-all"):
            skipped_all = True
    else:
        output_fn("  Current status: Not signed in")
        step1_choice = _safe_prompt("  [Enter] Open browser to sign into Google Colab | [s] Skip for now | [q] Skip all setup: ").lower()
        if step1_choice in ("q", "quit", "skip-all"):
            skipped_all = True
            should_login = False
        elif step1_choice in ("s", "skip", "n", "no"):
            should_login = False
            output_fn("  Skipped Google Colab sign-in. Run `platodoom auth` or `/auth` whenever you're ready.")
        else:
            should_login = True

    if should_login and not skipped_all:
        output_fn("  Opening your browser to Google Colab... Finish signing in and wait for Colab to load.")
        try:
            login_res = do_colab_login()
            if isinstance(login_res, dict) and login_res.get("ok"):
                colab_step_ok = True
                configured_anything = True
                output_fn("  ✓ Google Colab browser session saved.")
            else:
                err_msg = (login_res or {}).get("error", "Sign-in did not complete") if isinstance(login_res, dict) else str(login_res)
                output_fn(f"  ! Google Colab sign-in not completed: {err_msg}")
                output_fn("    You can retry anytime with `platodoom auth` or `/auth`.")
        except Exception as exc:
            output_fn(f"  ! Could not launch browser sign-in ({exc}).")
            output_fn("    You can retry anytime with `platodoom auth` or `/auth`.")

    # ------------------------------------------------------------------
    # Step 2/3: Optional HuggingFace API Token
    # ------------------------------------------------------------------
    if not skipped_all:
        output_fn("")
        output_fn("[Step 2/3] HuggingFace API Token (optional — for gated models & faster downloads)")
        if initial_status.get("hf_token_configured"):
            output_fn(f"  Current status: ✓ Configured ({initial_status.get('hf_token_preview')})")
        else:
            output_fn("  Current status: Not set")

        while True:
            hf_input = _safe_prompt(
                "  Paste HuggingFace token | [o] open huggingface.co/settings/tokens | [c] clear | [Enter] skip: "
            )
            low_hf = hf_input.lower()
            if low_hf in ("q", "quit", "skip-all"):
                skipped_all = True
                break
            if low_hf in ("o", "open"):
                opened = open_url(HF_TOKENS_URL)
                if opened:
                    output_fn(f"  Opened {HF_TOKENS_URL} in your browser. Paste the token below when ready.")
                else:
                    output_fn(f"  Open this link in your browser: {HF_TOKENS_URL}")
                continue
            if low_hf in ("", "s", "skip"):
                output_fn("  Kept existing HuggingFace token setting." if initial_status.get("hf_token_configured") else "  Skipped HuggingFace token.")
                break
            if low_hf in ("c", "clear", "remove"):
                user_cfg["auth"]["hf_token"] = None
                if env is None:
                    os.environ.pop("PLATODOOM_HF_TOKEN", None)
                eff_env_dict.pop("PLATODOOM_HF_TOKEN", None)
                configured_anything = True
                output_fn("  Cleared saved HuggingFace token from config.json.")
                break

            user_cfg["auth"]["hf_token"] = hf_input
            if env is None:
                os.environ["PLATODOOM_HF_TOKEN"] = hf_input
                if not os.environ.get("HF_TOKEN"):
                    os.environ["HF_TOKEN"] = hf_input
            eff_env_dict["PLATODOOM_HF_TOKEN"] = hf_input
            configured_anything = True
            output_fn(f"  ✓ Saved HuggingFace token ({mask_secret_preview(hf_input)}) to config.json (0600).")
            break

    # ------------------------------------------------------------------
    # Step 3/3: Optional Tunnel Provider (Cloudflare default vs ngrok)
    # ------------------------------------------------------------------
    if not skipped_all:
        output_fn("")
        output_fn("[Step 3/3] Tunnel Provider (optional — free Cloudflare Quick Tunnel works out of the box)")
        if initial_status.get("ngrok_token_configured"):
            output_fn(f"  Current status: ✓ ngrok configured ({initial_status.get('ngrok_token_preview')})")
        else:
            output_fn("  Current status: ✓ Cloudflare Quick Tunnel (free default, zero config)")

        while True:
            ng_input = _safe_prompt(
                "  Paste ngrok authtoken | [o] open ngrok dashboard | [c] use Cloudflare default | [Enter] keep default: "
            )
            low_ng = ng_input.lower()
            if low_ng in ("q", "quit", "skip-all"):
                break
            if low_ng in ("o", "open"):
                opened = open_url(NGROK_TOKEN_URL)
                if opened:
                    output_fn(f"  Opened {NGROK_TOKEN_URL} in your browser. Paste your authtoken below when ready.")
                else:
                    output_fn(f"  Open this link in your browser: {NGROK_TOKEN_URL}")
                continue
            if low_ng in ("", "s", "skip"):
                if not user_cfg["auth"].get("tunnel_provider"):
                    user_cfg["auth"]["tunnel_provider"] = "ngrok" if initial_status.get("ngrok_token_configured") else "cloudflare"
                output_fn(
                    "  Kept existing tunnel configuration."
                    if initial_status.get("ngrok_token_configured")
                    else "  Using free Cloudflare Quick Tunnel (no token needed)."
                )
                break
            if low_ng in ("c", "clear", "cloudflare"):
                user_cfg["auth"]["ngrok_authtoken"] = None
                user_cfg["auth"]["tunnel_provider"] = "cloudflare"
                if env is None:
                    os.environ.pop("NGROK_AUTHTOKEN", None)
                eff_env_dict.pop("NGROK_AUTHTOKEN", None)
                configured_anything = True
                output_fn("  ✓ Switched to free Cloudflare Quick Tunnel default.")
                break

            user_cfg["auth"]["ngrok_authtoken"] = ng_input
            user_cfg["auth"]["tunnel_provider"] = "ngrok"
            if env is None:
                os.environ["NGROK_AUTHTOKEN"] = ng_input
            eff_env_dict["NGROK_AUTHTOKEN"] = ng_input
            configured_anything = True
            output_fn(f"  ✓ Saved ngrok authtoken ({mask_secret_preview(ng_input)}) to config.json (0600).")
            break

    # Persist onboarding completion/skip marker with 0o600 permissions
    if configured_anything or colab_step_ok:
        user_cfg["onboarding"]["completed"] = True
        user_cfg["onboarding"]["skipped"] = False
    else:
        user_cfg["onboarding"]["completed"] = False
        user_cfg["onboarding"]["skipped"] = True

    save_env = dict(env) if env is not None else None
    cfg_path = save_user_config(user_cfg, env=save_env)

    final_status = get_auth_status(env=eff_env_dict)
    output_fn("")
    output_fn(
        "Setup saved! You can revisit or update credentials anytime via `platodoom auth` or `/auth`."
        if (configured_anything or colab_step_ok)
        else "Setup skipped for now. Run `platodoom auth` or `/auth` anytime to configure Google Colab or API tokens."
    )
    output_fn("")

    return {
        **final_status,
        "ran": True,
        "configured_anything": configured_anything,
        "skipped": bool(user_cfg["onboarding"].get("skipped", False)),
        "config_path": str(cfg_path),
    }
