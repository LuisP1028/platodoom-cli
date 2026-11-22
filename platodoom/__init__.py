"""
platodoom: Standalone Open-Source Colab GPU AI Coding, Reasoning & Multi-Directory Swarm Package.
Document ID: platodoom_init_857_v1
Governing Standards: platodoom specification (FR-1..FR-7), LANGUAGE.md, RULE[user_global]
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.0.1"

_LAZY_EXPORT_MAP = {
    # Session & Subagent Engine
    "PlatoSession": ("platodoom.session", "PlatoSession"),
    "VALID_ACCESS_MODES": ("platodoom.session", "VALID_ACCESS_MODES"),
    "VALID_WORKFLOW_MODES": ("platodoom.session", "VALID_WORKFLOW_MODES"),
    "list_saved_sessions": ("platodoom.session", "list_saved_sessions"),
    "load_saved_session_payload": ("platodoom.session", "load_saved_session_payload"),
    "delete_saved_session": ("platodoom.session", "delete_saved_session"),
    "reconstruct_messages_from_transcript": ("platodoom.session", "reconstruct_messages_from_transcript"),
    "estimate_messages_tokens": ("platodoom.session", "estimate_messages_tokens"),
    "invoke_subagent": ("platodoom.session", "invoke_subagent"),
    "invoke_parallel_subagents": ("platodoom.session", "invoke_parallel_subagents"),
    # REPL & Provisioning Telemetry
    "PROVISIONING_STAGE_LABELS": ("platodoom.repl", "PROVISIONING_STAGE_LABELS"),
    "TERMINAL_GOODBYE_MESSAGE": ("platodoom.repl", "TERMINAL_GOODBYE_MESSAGE"),
    "begin_interactive_terminal_session": ("platodoom.repl", "begin_interactive_terminal_session"),
    "emit_terminal_goodbye": ("platodoom.repl", "emit_terminal_goodbye"),
    "format_provisioning_stage_banner": ("platodoom.repl", "format_provisioning_stage_banner"),
    "get_raw_terminal_size": ("platodoom.repl", "get_raw_terminal_size"),
    "prompt_startup_model_and_access": ("platodoom.repl", "prompt_startup_model_and_access"),
    "render_startup_ascii_banner": ("platodoom.repl", "render_startup_ascii_banner"),
    "run_interactive_repl": ("platodoom.repl", "run_interactive_repl"),
    # CLI
    "fast_non_hanging_exit": ("platodoom.cli", "fast_non_hanging_exit"),
    "main": ("platodoom.cli", "main"),
    # Configuration & Local Response Cache (Partition A / Spec 867)
    "resolve_platodoom_config": ("platodoom.config", "resolve_platodoom_config"),
    "resolve_sessions_root_dir": ("platodoom.config", "resolve_sessions_root_dir"),
    "resolve_response_cache_dir": ("platodoom.config", "resolve_response_cache_dir"),
    "set_user_config_value": ("platodoom.config", "set_user_config_value"),
    "get_user_config_value": ("platodoom.config", "get_user_config_value"),
    "create_interactive_session_slug": ("platodoom.config", "create_interactive_session_slug"),
    "LocalResponseCache": ("platodoom.response_cache", "LocalResponseCache"),
    "compute_response_cache_key": ("platodoom.response_cache", "compute_response_cache_key"),
    "redact_cache_payload": ("platodoom.response_cache", "redact_cache_payload"),
    # Security (Partition A)
    "generate_ephemeral_tunnel_token": ("platodoom.security", "generate_ephemeral_tunnel_token"),
    "verify_bearer_authorization": ("platodoom.security", "verify_bearer_authorization"),
    "strip_secrets_for_remote_sync": ("platodoom.security", "strip_secrets_for_remote_sync"),
    "check_secret_path_and_redact": ("platodoom.security", "check_secret_path_and_redact"),
    "check_tool_permission": ("platodoom.security", "check_tool_permission"),
    # Catalog, 8-Tier Hardware & Co-Resident Stack Engine (Partition B / Spec 865)
    "ALL_HARDWARE_TIERS": ("platodoom.catalog", "ALL_HARDWARE_TIERS"),
    "PRIORITY_TEXT_MODELS": ("platodoom.catalog", "PRIORITY_TEXT_MODELS"),
    "PRIMARY_MENU_MODELS": ("platodoom.catalog", "PRIMARY_MENU_MODELS"),
    "PRECISION_TIERS": ("platodoom.catalog", "PRECISION_TIERS"),
    "MIN_KV_CACHE_HEADROOM_GB": ("platodoom.catalog", "MIN_KV_CACHE_HEADROOM_GB"),
    "GPU_TIER_SPECS": ("platodoom.catalog", "GPU_TIER_SPECS"),
    "BUILTIN_MODEL_CATALOG": ("platodoom.catalog", "BUILTIN_MODEL_CATALOG"),
    "HARDWARE_STACK_PRESETS": ("platodoom.catalog", "HARDWARE_STACK_PRESETS"),
    "normalize_hardware_tier": ("platodoom.catalog", "normalize_hardware_tier"),
    "normalize_precision_code": ("platodoom.catalog", "normalize_precision_code"),
    "get_hardware_tier_specs": ("platodoom.catalog", "get_hardware_tier_specs"),
    "get_model_catalog": ("platodoom.catalog", "get_model_catalog"),
    "resolve_model_profile": ("platodoom.catalog", "resolve_model_profile"),
    "resolve_model_precision_variant": ("platodoom.catalog", "resolve_model_precision_variant"),
    "route_gpu_for_model_alias": ("platodoom.catalog", "route_gpu_for_model_alias"),
    "verify_vram_compatibility": ("platodoom.catalog", "verify_vram_compatibility"),
    "validate_model_gpu_compatibility": ("platodoom.catalog", "validate_model_gpu_compatibility"),
    "select_optimal_gpu_tier": ("platodoom.catalog", "select_optimal_gpu_tier"),
    "verify_stack_vram_compatibility": ("platodoom.catalog", "verify_stack_vram_compatibility"),
    "parse_multi_model_agent_spec": ("platodoom.catalog", "parse_multi_model_agent_spec"),
    "resolve_multi_model_agent_allocation": ("platodoom.catalog", "resolve_multi_model_agent_allocation"),
    "get_stack_presets_for_tier": ("platodoom.catalog", "get_stack_presets_for_tier"),
    "resolve_hardware_stack": ("platodoom.catalog", "resolve_hardware_stack"),
    "export_colab_catalog_payload": ("platodoom.catalog", "export_colab_catalog_payload"),
    "introspect_model_protocol": ("platodoom.catalog", "introspect_model_protocol"),
    "classify_model_family_neutral": ("platodoom.catalog", "classify_model_family_neutral"),
    "clear_protocol_introspection_cache": ("platodoom.catalog", "clear_protocol_introspection_cache"),
    "compile_model_chat_messages": ("platodoom.catalog", "compile_model_chat_messages"),
    "parse_universal_tool_calls": ("platodoom.catalog", "parse_universal_tool_calls"),
    "compact_conversation_messages": ("platodoom.catalog", "compact_conversation_messages"),
    # Multi-Ecosystem Discovery & MCP Bridge (Partition B)
    "discover_workspace_capabilities": ("platodoom.discovery", "discover_workspace_capabilities"),
    "build_zero_turn_workspace_compass": ("platodoom.discovery", "build_zero_turn_workspace_compass"),
    "use_skill": ("platodoom.discovery", "use_skill"),
    "list_mcp_tools": ("platodoom.mcp_bridge", "list_mcp_tools"),
    "inspect_mcp_tool": ("platodoom.mcp_bridge", "inspect_mcp_tool"),
    "call_mcp_tool": ("platodoom.mcp_bridge", "call_mcp_tool"),
    # Local Navigation Tools (Partition C)
    "PLATODOOM_TOOLS_SCHEMA": ("platodoom.tools", "PLATODOOM_TOOLS_SCHEMA"),
    "execute_local_tool": ("platodoom.tools", "execute_local_tool"),
    # Swarm & Cloud Weight Cache (Partition A / Spec 867)
    "allocate_directory_swarm": ("platodoom.swarm", "allocate_directory_swarm"),
    "run_single_gpu_multi_model_agents": ("platodoom.swarm", "run_single_gpu_multi_model_agents"),
    "list_active_sessions": ("platodoom.swarm", "list_active_sessions"),
    "cache_model_to_cloud": ("platodoom.cache_model", "cache_model_to_cloud"),
    # Warm-GPU Discovery & Deterministic Standby Watchdog (Spec 866)
    "SMALL_STANDBY_TIERS": ("platodoom.standby", "SMALL_STANDBY_TIERS"),
    "FORBIDDEN_STANDBY_TIERS": ("platodoom.standby", "FORBIDDEN_STANDBY_TIERS"),
    "MANUAL_REBOOT_REQUIRED": ("platodoom.standby", "MANUAL_REBOOT_REQUIRED"),
    "resolve_standby_dir": ("platodoom.standby", "resolve_standby_dir"),
    "resolve_standby_endpoint_path": ("platodoom.standby", "resolve_standby_endpoint_path"),
    "read_standby_endpoint": ("platodoom.standby", "read_standby_endpoint"),
    "write_standby_endpoint": ("platodoom.standby", "write_standby_endpoint"),
    "is_small_standby_tier": ("platodoom.standby", "is_small_standby_tier"),
    "resolve_standby_target_spec": ("platodoom.standby", "resolve_standby_target_spec"),
    "sync_standby_from_live_small_endpoint": ("platodoom.standby", "sync_standby_from_live_small_endpoint"),
    "run_standby_watchdog_once": ("platodoom.standby", "run_standby_watchdog_once"),
    "discover_warm_colab_endpoints": ("scripts.provision_colab_serving", "discover_warm_colab_endpoints"),
    # User-Agnostic First-Run Onboarding & Credential Management
    "get_auth_status": ("platodoom.onboarding", "get_auth_status"),
    "format_auth_status": ("platodoom.onboarding", "format_auth_status"),
    "should_run_first_run_onboarding": ("platodoom.onboarding", "should_run_first_run_onboarding"),
    "run_interactive_onboarding": ("platodoom.onboarding", "run_interactive_onboarding"),
    "reset_onboarding_state": ("platodoom.onboarding", "reset_onboarding_state"),
    # Self-Bootstrapping Per-Directory Hermetic Runtime Bundle
    "RUNTIME_BUNDLE_MANIFEST": ("platodoom.bundle", "RUNTIME_BUNDLE_MANIFEST"),
    "compute_bundle_digest": ("platodoom.bundle", "compute_bundle_digest"),
    "bootstrap_directory_platodoom_runtime": ("platodoom.bundle", "bootstrap_directory_platodoom_runtime"),
    "sync_global_platodoom_template": ("platodoom.bundle", "sync_global_platodoom_template"),
}

__all__ = ["__version__", *list(_LAZY_EXPORT_MAP.keys())]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORT_MAP:
        mod_name, attr_name = _LAZY_EXPORT_MAP[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name)
    raise AttributeError(f"module 'platodoom' has no attribute {name!r}")
