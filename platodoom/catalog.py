from __future__ import annotations

import copy
import json
import math
import os
import re
import struct
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Authoritative Empirical Chat Templates & Metadata Snapshots (FR-3.1, FR-3.3)
# ---------------------------------------------------------------------------

QWEN35_CHAT_TEMPLATE_JINJA: str = (
    "{%- set image_count = namespace(value=0) %}\n"
    "{%- set video_count = namespace(value=0) %}\n"
    "{%- macro render_content(content, do_vision_count, is_system_content=false) %}\n"
    "    {%- if content is string %}\n"
    "        {{- content }}\n"
    "    {%- elif content is iterable and content is not mapping %}\n"
    "        {%- for item in content %}\n"
    "            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}\n"
    "                {%- if is_system_content %}\n"
    "                    {{- raise_exception('System message cannot contain images.') }}\n"
    "                {%- endif %}\n"
    "                {%- if do_vision_count %}\n"
    "                    {%- set image_count.value = image_count.value + 1 %}\n"
    "                {%- endif %}\n"
    "                {%- if add_vision_id %}\n"
    "                    {{- 'Picture ' ~ image_count.value ~ ': ' }}\n"
    "                {%- endif %}\n"
    "                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}\n"
    "            {%- elif 'video' in item or item.type == 'video' %}\n"
    "                {%- if is_system_content %}\n"
    "                    {{- raise_exception('System message cannot contain videos.') }}\n"
    "                {%- endif %}\n"
    "                {%- if do_vision_count %}\n"
    "                    {%- set video_count.value = video_count.value + 1 %}\n"
    "                {%- endif %}\n"
    "                {%- if add_vision_id %}\n"
    "                    {{- 'Video ' ~ video_count.value ~ ': ' }}\n"
    "                {%- endif %}\n"
    "                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}\n"
    "            {%- elif 'text' in item %}\n"
    "                {{- item.text }}\n"
    "            {%- else %}\n"
    "                {{- raise_exception('Unexpected item type in content.') }}\n"
    "            {%- endif %}\n"
    "        {%- endfor %}\n"
    "    {%- elif content is none or content is undefined %}\n"
    "        {{- '' }}\n"
    "    {%- else %}\n"
    "        {{- raise_exception('Unexpected content type.') }}\n"
    "    {%- endif %}\n"
    "{%- endmacro %}\n"
    "{%- if not messages %}\n"
    "    {{- raise_exception('No messages provided.') }}\n"
    "{%- endif %}\n"
    "{%- if tools and tools is iterable and tools is not mapping %}\n"
    "    {{- '<|im_start|>system\\n' }}\n"
    "    {{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}\n"
    "    {%- for tool in tools %}\n"
    "        {{- \"\\n\" }}\n"
    "        {{- tool | tojson }}\n"
    "    {%- endfor %}\n"
    "    {{- \"\\n</tools>\" }}\n"
    "    {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}\n"
    "    {%- if messages[0].role == 'system' %}\n"
    "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n"
    "        {%- if content %}\n"
    "            {{- '\\n\\n' + content }}\n"
    "        {%- endif %}\n"
    "    {%- endif %}\n"
    "    {{- '<|im_end|>\\n' }}\n"
    "{%- else %}\n"
    "    {%- if messages[0].role == 'system' %}\n"
    "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n"
    "        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
    "{%- for message in messages[::-1] %}\n"
    "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
    "    {%- if ns.multi_step_tool and message.role == \"user\" %}\n"
    "        {%- set content = render_content(message.content, false)|trim %}\n"
    "        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}\n"
    "            {%- set ns.multi_step_tool = false %}\n"
    "            {%- set ns.last_query_index = index %}\n"
    "        {%- endif %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if ns.multi_step_tool %}\n"
    "    {{- raise_exception('No user query found in messages.') }}\n"
    "{%- endif %}\n"
    "{%- for message in messages %}\n"
    "    {%- set content = render_content(message.content, true)|trim %}\n"
    "    {%- if message.role == \"system\" %}\n"
    "        {%- if not loop.first %}\n"
    "            {{- raise_exception('System message must be at the beginning.') }}\n"
    "        {%- endif %}\n"
    "    {%- elif message.role == \"user\" %}\n"
    "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
    "    {%- elif message.role == \"assistant\" %}\n"
    "        {%- set reasoning_content = '' %}\n"
    "        {%- if message.reasoning_content is string %}\n"
    "            {%- set reasoning_content = message.reasoning_content %}\n"
    "        {%- else %}\n"
    "            {%- if '</think>' in content %}\n"
    "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
    "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
    "            {%- endif %}\n"
    "        {%- endif %}\n"
    "        {%- set reasoning_content = reasoning_content|trim %}\n"
    "        {%- if loop.index0 > ns.last_query_index %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n"
    "        {%- else %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "        {%- endif %}\n"
    "        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}\n"
    "            {%- for tool_call in message.tool_calls %}\n"
    "                {%- if tool_call.function is defined %}\n"
    "                    {%- set tool_call = tool_call.function %}\n"
    "                {%- endif %}\n"
    "                {%- if loop.first %}\n"
    "                    {%- if content|trim %}\n"
    "                        {{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n"
    "                    {%- else %}\n"
    "                        {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n"
    "                    {%- endif %}\n"
    "                {%- else %}\n"
    "                    {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n"
    "                {%- endif %}\n"
    "                {%- if tool_call.arguments is defined %}\n"
    "                    {%- for args_name, args_value in tool_call.arguments|items %}\n"
    "                        {{- '<parameter=' + args_name + '>\\n' }}\n"
    "                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}\n"
    "                        {{- args_value }}\n"
    "                        {{- '\\n</parameter>\\n' }}\n"
    "                    {%- endfor %}\n"
    "                {%- endif %}\n"
    "                {{- '</function>\\n</tool_call>' }}\n"
    "            {%- endfor %}\n"
    "        {%- endif %}\n"
    "        {{- '<|im_end|>\\n' }}\n"
    "    {%- elif message.role == \"tool\" %}\n"
    "        {%- if loop.previtem and loop.previtem.role != \"tool\" %}\n"
    "            {{- '<|im_start|>user' }}\n"
    "        {%- endif %}\n"
    "        {{- '\\n<tool_response>\\n' }}\n"
    "        {{- content }}\n"
    "        {{- '\\n</tool_response>' }}\n"
    "        {%- if not loop.last and loop.nextitem.role != \"tool\" %}\n"
    "            {{- '<|im_end|>\\n' }}\n"
    "        {%- elif loop.last %}\n"
    "            {{- '<|im_end|>\\n' }}\n"
    "        {%- endif %}\n"
    "    {%- else %}\n"
    "        {{- raise_exception('Unexpected message role.') }}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
)

DEEPSEEK_R1_DISTILL_CHAT_TEMPLATE_JINJA: str = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}"
    "{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}"
    "{{bos_token}}{{ns.system_prompt}}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}"
    "{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}"
    "{%- for tool in message['tool_calls']%}"
    "{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}"
    "{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}"
    "{%- endfor %}{%- endif %}"
    "{%- if message['role'] == 'assistant' and message['content'] is not none %}"
    "{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}"
    "{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}"
    "{%- endif %}"
    "{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}"
    "{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}"
    "{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}"
    "{%- endif %}"
    "{%- endfor -%}"
    "{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}"
    "{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}"
)

DEEPSEEK_V2_LITE_CHAT_TEMPLATE_JINJA: str = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}"
    "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'Assistant:' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)

# ---------------------------------------------------------------------------
# Authoritative 8-Tier Hardware, 14-Model Priority Catalog & Stack Presets
# (Spec 857, Spec 859, Spec 862, Spec 865)
# ---------------------------------------------------------------------------

PRIMARY_MENU_MODELS: Tuple[str, ...] = (
    "qwen3.8-27b",
    "qwen-9b",
    "r1-14b",
    "qwen-27b",
    "qwen-35b-moe",
    "r1-32b",
)

PRIORITY_TEXT_MODELS: Tuple[str, ...] = (
    "qwen3.8-27b",
    "qwen-9b",
    "r1-14b",
    "qwen-27b",
    "qwen-35b-moe",
    "r1-32b",
    "qwen-30b-moe",
    "qwen-coder-30b-moe",
    "qwen-14b",
    "qwen-8b",
    "qwen-4b",
    "qwen-122b-moe",
    "r1-7b",
    "deepseek-v2-lite",
)

ALL_HARDWARE_TIERS: Tuple[str, ...] = (
    "G4",
    "H100",
    "A100-80GB",
    "A100",
    "v6e-1",
    "L4",
    "T4",
    "v5e-1",
)

PRECISION_TIERS: Tuple[str, ...] = (
    "BF16",
    "FP8",
    "INT8",
    "Q8",
    "Q6",
    "Q5",
    "Q4",
)

MIN_KV_CACHE_HEADROOM_GB: float = 4.0

GPU_TIER_SPECS: Dict[str, Dict[str, Any]] = {
    "v5e-1": {
        "tier": "v5e-1",
        "accelerator_type": "TPU",
        "memory_type": "HBM",
        "vram_gb": 16.0,
        "nominal_vram_gb": 16.0,
        "min_kv_headroom_gb": 4.0,
        "max_stack_weights_gb": 12.0,
        "colab_accelerator": "TPU",
        "colab_gpu_type": None,
        "colab_tpu_type": "v5e-1",
        "system_ram_gb": 47.0,
        "rank": 1,
        "max_num_seqs": 4,
    },
    "T4": {
        "tier": "T4",
        "accelerator_type": "GPU",
        "memory_type": "VRAM",
        "vram_gb": 15.8,
        "nominal_vram_gb": 16.0,
        "min_kv_headroom_gb": 4.0,
        "max_stack_weights_gb": 11.8,
        "colab_accelerator": "GPU",
        "colab_gpu_type": "T4",
        "colab_tpu_type": None,
        "system_ram_gb": 25.5,
        "rank": 2,
        "max_num_seqs": 32,
    },
    "L4": {
        "tier": "L4",
        "accelerator_type": "GPU",
        "memory_type": "VRAM",
        "vram_gb": 23.5,
        "nominal_vram_gb": 24.0,
        "min_kv_headroom_gb": 4.0,
        "max_stack_weights_gb": 19.5,
        "colab_accelerator": "GPU",
        "colab_gpu_type": "L4",
        "colab_tpu_type": None,
        "system_ram_gb": 53.0,
        "rank": 3,
        "max_num_seqs": 64,
    },
    "v6e-1": {
        "tier": "v6e-1",
        "accelerator_type": "TPU",
        "memory_type": "HBM",
        "vram_gb": 32.0,
        "nominal_vram_gb": 32.0,
        "min_kv_headroom_gb": 4.0,
        "max_stack_weights_gb": 28.0,
        "colab_accelerator": "TPU",
        "colab_gpu_type": None,
        "colab_tpu_type": "v6e-1",
        "system_ram_gb": 64.0,
        "rank": 4,
        "max_num_seqs": 8,
    },
    "A100": {
        "tier": "A100",
        "accelerator_type": "GPU",
        "memory_type": "VRAM",
        "vram_gb": 40.0,
        "nominal_vram_gb": 40.0,
        "min_kv_headroom_gb": 4.0,
        "max_stack_weights_gb": 36.0,
        "colab_accelerator": "GPU",
        "colab_gpu_type": "A100",
        "colab_tpu_type": None,
        "system_ram_gb": 83.5,
        "rank": 5,
        "max_num_seqs": 96,
    },
    "A100-80GB": {
        "tier": "A100-80GB",
        "accelerator_type": "GPU",
        "memory_type": "VRAM",
        "vram_gb": 79.2,
        "nominal_vram_gb": 80.0,
        "min_kv_headroom_gb": 4.0,
        "max_stack_weights_gb": 75.2,
        "colab_accelerator": "GPU",
        "colab_gpu_type": "A100",
        "colab_tpu_type": None,
        "system_ram_gb": 167.0,
        "rank": 6,
        "max_num_seqs": 128,
    },
    "H100": {
        "tier": "H100",
        "accelerator_type": "GPU",
        "memory_type": "VRAM",
        "vram_gb": 79.2,
        "nominal_vram_gb": 80.0,
        "min_kv_headroom_gb": 4.0,
        "max_stack_weights_gb": 75.2,
        "colab_accelerator": "GPU",
        "colab_gpu_type": "H100",
        "colab_tpu_type": None,
        "system_ram_gb": 167.0,
        "rank": 7,
        "max_num_seqs": 160,
    },
    "G4": {
        "tier": "G4",
        "accelerator_type": "GPU",
        "memory_type": "VRAM",
        "vram_gb": 95.0,
        "nominal_vram_gb": 96.0,
        "min_kv_headroom_gb": 4.0,
        "max_stack_weights_gb": 91.0,
        "colab_accelerator": "GPU",
        "colab_gpu_type": "G4",
        "colab_tpu_type": None,
        "system_ram_gb": 180.0,
        "rank": 8,
        "max_num_seqs": 256,
    },
}

_HARDWARE_TIER_ALIASES: Dict[str, str] = {
    "v5e-1": "v5e-1",
    "v5e": "v5e-1",
    "tpu-v5e": "v5e-1",
    "tpu-v5e-1": "v5e-1",
    "v5litepod-1": "v5e-1",
    "t4": "T4",
    "t4-16gb": "T4",
    "l4": "L4",
    "l4-24gb": "L4",
    "v6e-1": "v6e-1",
    "v6e": "v6e-1",
    "trillium": "v6e-1",
    "tpu-v6e": "v6e-1",
    "tpu-v6e-1": "v6e-1",
    "a100": "A100",
    "a100-40": "A100",
    "a100_40": "A100",
    "a100-40gb": "A100",
    "a100_40gb": "A100",
    "a100-sxm4-40gb": "A100",
    "a100-80": "A100-80GB",
    "a100_80": "A100-80GB",
    "a100-80gb": "A100-80GB",
    "a100_80gb": "A100-80GB",
    "a100-sxm4-80gb": "A100-80GB",
    "h100": "H100",
    "h100-80gb": "H100",
    "h100_80gb": "H100",
    "g4": "G4",
    "g4-96gb": "G4",
    "rtx-pro-6000": "G4",
    "blackwell-96gb": "G4",
}

_PRECISION_CODE_ALIASES: Dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "16bit": "BF16",
    "fp8": "FP8",
    "float8": "FP8",
    "int8": "INT8",
    "8bit": "Q8",
    "q8": "Q8",
    "q8_0": "Q8",
    "bnb_8bit": "Q8",
    "q6": "Q6",
    "q6_k": "Q6",
    "q5": "Q5",
    "q5_k_m": "Q5",
    "q4": "Q4",
    "q4_k_m": "Q4",
    "int4": "Q4",
    "4bit": "Q4",
    "awq": "Q4",
    "awq_4bit": "Q4",
    "gptq": "Q4",
    "gptq_4bit": "Q4",
    "compressed-tensors": "Q4",
    "compressed_tensors": "Q4",
    "bitsandbytes": "Q4",
    "nf4": "Q4",
    "nf4_4bit": "Q4",
}


def normalize_hardware_tier(raw_tier: Optional[str], *, allow_auto: bool = True) -> str:
    """Normalizes any user/config hardware tier label into one of the 8 canonical Colab tiers or 'auto'.

    Raises ValueError immediately on unrecognized tier strings (Fail-Fast per RULE[user_global]).
    """
    cleaned = str(raw_tier or "").strip()
    if not cleaned or cleaned.lower() == "auto":
        if allow_auto:
            return "auto"
        raise ValueError(
            f"Hardware tier cannot be 'auto' or empty in this context. "
            f"Valid tiers: {list(ALL_HARDWARE_TIERS)}"
        )
    if cleaned in GPU_TIER_SPECS:
        return cleaned
    mapped = _HARDWARE_TIER_ALIASES.get(cleaned.lower())
    if mapped is not None:
        return mapped
    valid_list = [*ALL_HARDWARE_TIERS, *(["auto"] if allow_auto else [])]
    raise ValueError(
        f"Unsupported Colab hardware tier '{raw_tier}'. Valid tiers: {valid_list}"
    )


def normalize_precision_code(raw_precision: Optional[str]) -> str:
    """Normalizes precision labels ('BF16', 'bfloat16', 'FP8', 'INT8', 'Q8', 'Q6', 'Q5', 'Q4', 'awq_4bit', etc.)
    into canonical codes ('BF16', 'FP8', 'INT8', 'Q8', 'Q6', 'Q5', 'Q4').
    """
    cleaned = str(raw_precision or "").strip()
    if not cleaned:
        raise ValueError(f"Precision cannot be empty. Valid precisions: {list(PRECISION_TIERS)}")
    upper = cleaned.upper()
    if upper in PRECISION_TIERS:
        return upper
    mapped = _PRECISION_CODE_ALIASES.get(cleaned.lower())
    if mapped is not None:
        return mapped
    raise ValueError(
        f"Unsupported precision '{raw_precision}'. Valid precisions: {list(PRECISION_TIERS)}"
    )


def get_hardware_tier_specs() -> Dict[str, Dict[str, Any]]:
    """Returns a deep copy of the authoritative 8-tier Colab GPU/TPU capacity matrix."""
    return {tier: copy.deepcopy(GPU_TIER_SPECS[tier]) for tier in ALL_HARDWARE_TIERS}


def _build_precision_variants(
    *,
    base_repo: str,
    q4_repo: str,
    q4_method: str,
    weights_by_tier: Dict[str, float],
    fp8_repo: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    bits_map = {"BF16": 16, "FP8": 8, "INT8": 8, "Q8": 8, "Q6": 6, "Q5": 5, "Q4": 4}
    method_map = {
        "BF16": "bfloat16",
        "FP8": "fp8" if fp8_repo else "bnb_8bit",
        "INT8": "bnb_8bit",
        "Q8": "bnb_8bit",
        "Q6": q4_method,
        "Q5": q4_method,
        "Q4": q4_method,
    }
    repo_map = {
        "BF16": base_repo,
        "FP8": fp8_repo or base_repo,
        "INT8": base_repo,
        "Q8": base_repo,
        "Q6": q4_repo,
        "Q5": q4_repo,
        "Q4": q4_repo,
    }
    variants: Dict[str, Dict[str, Any]] = {}
    for code in PRECISION_TIERS:
        w_gb = round(float(weights_by_tier[code]), 1)
        variants[code] = {
            "precision": code,
            "weights_gb": w_gb,
            "min_vram_gb": round(w_gb + MIN_KV_CACHE_HEADROOM_GB, 1),
            "hf_repo_id": repo_map[code],
            "quant_method": method_map[code],
            "bits": bits_map[code],
        }
    return variants


def _compute_supported_hardware_tiers(variants: Dict[str, Dict[str, Any]]) -> List[str]:
    min_required = min(float(v["min_vram_gb"]) for v in variants.values())
    return [
        tier
        for tier in ALL_HARDWARE_TIERS
        if float(GPU_TIER_SPECS[tier]["vram_gb"]) >= min_required
    ]



_VARIANTS_QWEN_9B = _build_precision_variants(
    base_repo="Qwen/Qwen3.5-9B",
    q4_repo="cyankiwi/Qwen3.5-9B-AWQ-4bit",
    q4_method="compressed-tensors",
    weights_by_tier={"BF16": 18.0, "FP8": 9.2, "INT8": 9.2, "Q8": 9.6, "Q6": 7.4, "Q5": 6.4, "Q4": 5.0},
)
_VARIANTS_R1_14B = _build_precision_variants(
    base_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    q4_repo="casperhansen/deepseek-r1-distill-qwen-14b-awq",
    q4_method="awq_4bit",
    weights_by_tier={"BF16": 29.6, "FP8": 15.1, "INT8": 15.1, "Q8": 15.7, "Q6": 12.1, "Q5": 10.4, "Q4": 8.2},
)
_VARIANTS_QWEN_27B = _build_precision_variants(
    base_repo="Qwen/Qwen3.5-27B",
    q4_repo="cyankiwi/Qwen3.5-27B-AWQ-4bit",
    q4_method="compressed-tensors",
    fp8_repo="Qwen/Qwen3.5-27B-FP8",
    weights_by_tier={"BF16": 54.0, "FP8": 27.5, "INT8": 27.5, "Q8": 28.6, "Q6": 22.2, "Q5": 19.0, "Q4": 15.0},
)
_VARIANTS_QWEN38_27B = _build_precision_variants(
    base_repo="Qwen/Qwen3.8-27B",
    q4_repo="cyankiwi/Qwen3.8-27B-AWQ-INT4",
    q4_method="compressed-tensors",
    fp8_repo="Qwen/Qwen3.8-27B-FP8",
    weights_by_tier={"BF16": 54.0, "FP8": 27.5, "INT8": 27.5, "Q8": 28.6, "Q6": 22.2, "Q5": 19.0, "Q4": 15.0},
)
_VARIANTS_QWEN_35B_MOE = _build_precision_variants(
    base_repo="Qwen/Qwen3.5-35B-A3B",
    q4_repo="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    q4_method="gptq_4bit",
    fp8_repo="Qwen/Qwen3.5-35B-A3B-FP8",
    weights_by_tier={"BF16": 70.0, "FP8": 35.5, "INT8": 35.5, "Q8": 37.0, "Q6": 28.8, "Q5": 24.5, "Q4": 19.0},
)
_VARIANTS_R1_32B = _build_precision_variants(
    base_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    q4_repo="casperhansen/deepseek-r1-distill-qwen-32b-awq",
    q4_method="awq_4bit",
    weights_by_tier={"BF16": 65.5, "FP8": 33.5, "INT8": 33.5, "Q8": 34.8, "Q6": 27.0, "Q5": 23.0, "Q4": 18.0},
)
# DeepSeek-R1-Distill-Qwen-32B (64-layer dense reasoning with 8,192-token <think> traces)
# requires >=30.0 GB solo memory (starts at v6e-1 32GB HBM & A100 40GB per Spec 865 §3.2).
_VARIANTS_R1_32B["Q4"]["min_vram_gb"] = 30.0
_VARIANTS_R1_32B["Q5"]["min_vram_gb"] = 30.0
_VARIANTS_QWEN_30B_MOE = _build_precision_variants(
    base_repo="Qwen/Qwen3-30B-A3B",
    q4_repo="Qwen/Qwen3-30B-A3B-GPTQ-Int4",
    q4_method="gptq_4bit",
    fp8_repo="Qwen/Qwen3-30B-A3B-FP8",
    weights_by_tier={"BF16": 61.0, "FP8": 31.0, "INT8": 31.0, "Q8": 32.4, "Q6": 25.0, "Q5": 21.4, "Q4": 16.8},
)
_VARIANTS_QWEN_CODER_30B_MOE = _build_precision_variants(
    base_repo="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    q4_repo="cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
    q4_method="compressed-tensors",
    fp8_repo="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    weights_by_tier={"BF16": 61.0, "FP8": 31.0, "INT8": 31.0, "Q8": 32.4, "Q6": 25.0, "Q5": 21.4, "Q4": 16.8},
)
_VARIANTS_QWEN_14B = _build_precision_variants(
    base_repo="Qwen/Qwen3-14B",
    q4_repo="Qwen/Qwen3-14B-AWQ",
    q4_method="awq_4bit",
    fp8_repo="Qwen/Qwen3-14B-FP8",
    weights_by_tier={"BF16": 29.6, "FP8": 15.1, "INT8": 15.1, "Q8": 15.7, "Q6": 12.1, "Q5": 10.4, "Q4": 8.2},
)
_VARIANTS_QWEN_8B = _build_precision_variants(
    base_repo="Qwen/Qwen3-8B",
    q4_repo="Qwen/Qwen3-8B-AWQ",
    q4_method="awq_4bit",
    fp8_repo="Qwen/Qwen3-8B-FP8",
    weights_by_tier={"BF16": 16.4, "FP8": 8.4, "INT8": 8.4, "Q8": 8.7, "Q6": 6.7, "Q5": 5.8, "Q4": 4.7},
)
_VARIANTS_QWEN_4B = _build_precision_variants(
    base_repo="Qwen/Qwen3-4B",
    q4_repo="Qwen/Qwen3-4B-AWQ",
    q4_method="awq_4bit",
    fp8_repo="Qwen/Qwen3-4B-FP8",
    weights_by_tier={"BF16": 8.0, "FP8": 4.1, "INT8": 4.1, "Q8": 4.3, "Q6": 3.3, "Q5": 2.8, "Q4": 2.4},
)
_VARIANTS_QWEN_122B_MOE = _build_precision_variants(
    base_repo="Qwen/Qwen3.5-122B-A10B",
    q4_repo="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    q4_method="gptq_4bit",
    fp8_repo="Qwen/Qwen3.5-122B-A10B-FP8",
    weights_by_tier={"BF16": 244.0, "FP8": 124.0, "INT8": 124.0, "Q8": 129.0, "Q6": 100.0, "Q5": 85.5, "Q4": 66.0},
)
_VARIANTS_R1_7B = _build_precision_variants(
    base_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    q4_repo="casperhansen/deepseek-r1-distill-qwen-7b-awq",
    q4_method="awq_4bit",
    weights_by_tier={"BF16": 15.2, "FP8": 7.8, "INT8": 7.8, "Q8": 8.1, "Q6": 6.3, "Q5": 5.4, "Q4": 4.4},
)
_VARIANTS_DEEPSEEK_V2_LITE = _build_precision_variants(
    base_repo="deepseek-ai/DeepSeek-V2-Lite-Chat",
    q4_repo="TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
    q4_method="awq_4bit",
    weights_by_tier={"BF16": 31.5, "FP8": 16.0, "INT8": 16.0, "Q8": 16.7, "Q6": 12.9, "Q5": 11.0, "Q4": 8.8},
)

BUILTIN_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "qwen-9b": {
        "alias": "qwen-9b",
        "family": "qwen",
        "total_params_b": 9.0,
        "active_params_b": 9.0,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-9B",
        "fallback_hf_repo_id": "Qwen/Qwen3.5-9B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "quantized_repo_ids": {
            "awq_4bit": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
            "compressed-tensors": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
            "gptq_4bit": "mssfj/Qwen3.5-9B-GPTQ-INT4",
            "quanttrio_awq_4bit": "QuantTrio/Qwen3.5-9B-AWQ",
            "bfloat16": "Qwen/Qwen3.5-9B",
            "Q4": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
            "FP8": "Qwen/Qwen3.5-9B",
            "Q8": "Qwen/Qwen3.5-9B",
            "BF16": "Qwen/Qwen3.5-9B",
        },
        "precision_variants": _VARIANTS_QWEN_9B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_9B),
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "T4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "T4",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_9B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_9B),
        "compatible_gpu_tiers": ["T4", "L4", "A100"],
        "min_vram_gb": 9.0,
        "stop_token_ids": [248046, 248044],
        "default_precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "Q4",
            "v6e-1": "bfloat16",
            "A100": "bfloat16",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "Q4",
            "v6e-1": "bfloat16",
            "A100": "bfloat16",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "max_num_seqs": 32,
        "storage_uri": None,
        "description": "Fast default reasoning & coding model (Qwen3.5-9B, fits T4/L4/v5e-1 in 4-bit Q4 or v6e-1/A100+ in BF16).",
    },
    "r1-14b": {
        "alias": "r1-14b",
        "family": "deepseek",
        "total_params_b": 14.8,
        "active_params_b": 14.8,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "fallback_hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
        "quantized_repo_ids": {
            "awq_4bit": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
            "nf4_4bit": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "bfloat16": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "Q4": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
            "Q8": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "BF16": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        },
        "precision_variants": _VARIANTS_R1_14B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_R1_14B),
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "template_family": "deepseek_r1_distill",
        "preferred_gpu": "L4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "L4",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_R1_14B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_R1_14B),
        "compatible_gpu_tiers": ["T4", "L4", "A100"],
        "min_vram_gb": 12.2,
        "stop_token_ids": [151643, 151645],
        "default_precision_by_gpu": {
            "v5e-1": "awq_4bit",
            "T4": "awq_4bit",
            "L4": "Q8",
            "v6e-1": "Q8",
            "A100": "bfloat16",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "precision_by_gpu": {
            "v5e-1": "awq_4bit",
            "T4": "awq_4bit",
            "L4": "Q8",
            "v6e-1": "Q8",
            "A100": "bfloat16",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "max_num_seqs": 32,
        "storage_uri": None,
        "description": "Medium DeepSeek-R1 distilled reasoning specialist (14B dense, fits T4/L4 in AWQ Q4 and A100+ in BF16).",
    },
    "qwen-27b": {
        "alias": "qwen-27b",
        "family": "qwen",
        "total_params_b": 27.0,
        "active_params_b": 27.0,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-27B",
        "fallback_hf_repo_id": "Qwen/Qwen3.5-27B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
        "quantized_repo_ids": {
            "awq_4bit": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
            "compressed-tensors": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
            "qwen36_awq_4bit": "Alennnndy/Qwen3.6-27B-AWQ-4bit",
            "fp8": "Qwen/Qwen3.5-27B-FP8",
            "bfloat16": "Qwen/Qwen3.5-27B",
            "Q4": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
            "FP8": "Qwen/Qwen3.5-27B-FP8",
            "BF16": "Qwen/Qwen3.5-27B",
        },
        "precision_variants": _VARIANTS_QWEN_27B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_27B),
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "A100",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "A100",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_27B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_27B),
        "compatible_gpu_tiers": ["L4", "A100"],
        "min_vram_gb": 19.0,
        "stop_token_ids": [248046, 248044],
        "default_precision_by_gpu": {
            "L4": "awq_4bit",
            "v6e-1": "awq_4bit",
            "A100": "awq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "precision_by_gpu": {
            "L4": "awq_4bit",
            "v6e-1": "awq_4bit",
            "A100": "awq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "max_num_seqs": 48,
        "storage_uri": None,
        "description": "Daily workhorse 27B dense coding & tool model (Qwen3.5/3.6-27B, fits L4/v6e-1/A100 in Q4 and 80GB+ in BF16).",
    },
    "qwen3.8-27b": {
        "alias": "qwen3.8-27b",
        "family": "qwen",
        "total_params_b": 27.0,
        "active_params_b": 27.0,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.8-27B",
        "fallback_hf_repo_id": "Qwen/Qwen3.8-27B",
        "quantized_hf_repo_id": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
        "quantized_repo_ids": {
            "awq_4bit": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
            "compressed-tensors": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
            "fp8": "Qwen/Qwen3.8-27B-FP8",
            "bfloat16": "Qwen/Qwen3.8-27B",
            "Q4": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
            "FP8": "Qwen/Qwen3.8-27B-FP8",
            "BF16": "Qwen/Qwen3.8-27B",
        },
        "precision_variants": _VARIANTS_QWEN38_27B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN38_27B),
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "A100",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "A100",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN38_27B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN38_27B),
        "compatible_gpu_tiers": ["L4", "A100"],
        "min_vram_gb": 19.0,
        "stop_token_ids": [248046, 248044],
        "default_precision_by_gpu": {
            "L4": "awq_4bit",
            "v6e-1": "awq_4bit",
            "A100": "awq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "precision_by_gpu": {
            "L4": "awq_4bit",
            "v6e-1": "awq_4bit",
            "A100": "awq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "max_num_seqs": 48,
        "storage_uri": None,
        "description": "Default platodoom 27B dense coding, reasoning & vision model (Qwen3.8-27B, fits L4/v6e-1/A100 in Q4 and 80GB+ in BF16).",
    },
    "qwen-35b-moe": {
        "alias": "qwen-35b-moe",
        "family": "qwen",
        "total_params_b": 35.0,
        "active_params_b": 3.0,
        "is_moe": True,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-35B-A3B",
        "fallback_hf_repo_id": "Qwen/Qwen3.5-35B-A3B",
        "quantized_hf_repo_id": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        "quantized_repo_ids": {
            "gptq_4bit": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
            "awq_4bit": "QuantTrio/Qwen3.6-35B-A3B-AWQ",
            "fp8": "Qwen/Qwen3.5-35B-A3B-FP8",
            "bfloat16": "Qwen/Qwen3.5-35B-A3B",
            "Q4": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
            "FP8": "Qwen/Qwen3.5-35B-A3B-FP8",
            "BF16": "Qwen/Qwen3.5-35B-A3B",
        },
        "precision_variants": _VARIANTS_QWEN_35B_MOE,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_35B_MOE),
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "model_type": "qwen3_5_moe",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "A100",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "A100",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_35B_MOE),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_35B_MOE),
        "compatible_gpu_tiers": ["L4", "A100"],
        "min_vram_gb": 23.0,
        "stop_token_ids": [248046, 248044],
        "default_precision_by_gpu": {
            "L4": "gptq_4bit",
            "v6e-1": "gptq_4bit",
            "A100": "gptq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "precision_by_gpu": {
            "L4": "gptq_4bit",
            "v6e-1": "gptq_4bit",
            "A100": "gptq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "max_num_seqs": 64,
        "storage_uri": None,
        "description": "High-throughput 35B MoE model (~3B active parameters/token, fits L4/v6e-1/A100 in Q4 and 80GB+ in BF16).",
    },
    "r1-32b": {
        "alias": "r1-32b",
        "family": "deepseek",
        "total_params_b": 32.8,
        "active_params_b": 32.8,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "fallback_hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-32b-awq",
        "quantized_repo_ids": {
            "awq_4bit": "casperhansen/deepseek-r1-distill-qwen-32b-awq",
            "bfloat16": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "Q4": "casperhansen/deepseek-r1-distill-qwen-32b-awq",
            "BF16": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        },
        "precision_variants": _VARIANTS_R1_32B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_R1_32B),
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "template_family": "deepseek_r1_distill",
        "preferred_gpu": "A100",
        "min_gpu_tier": "A100",
        "recommended_gpu_tier": "A100",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_R1_32B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_R1_32B),
        "compatible_gpu_tiers": ["A100"],
        "min_vram_gb": 30.0,
        "stop_token_ids": [151643, 151645],
        "default_precision_by_gpu": {
            "v6e-1": "awq_4bit",
            "A100": "awq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "precision_by_gpu": {
            "v6e-1": "awq_4bit",
            "A100": "awq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "max_num_seqs": 32,
        "max_output_tokens": 8192,
        "storage_uri": None,
        "description": "Deep reasoning 32B dense specialist (DeepSeek-R1-Distill-Qwen-32B, fits v6e-1/A100 in Q4 and 80GB+ in BF16).",
    },
    "qwen-30b-moe": {
        "alias": "qwen-30b-moe",
        "family": "qwen",
        "total_params_b": 30.5,
        "active_params_b": 3.3,
        "is_moe": True,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-30B-A3B",
        "fallback_hf_repo_id": "Qwen/Qwen3-30B-A3B",
        "quantized_hf_repo_id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        "quantized_repo_ids": {
            "gptq_4bit": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
            "awq_4bit": "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ",
            "stelterlab_awq_4bit": "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ",
            "fp8": "Qwen/Qwen3-30B-A3B-FP8",
            "bfloat16": "Qwen/Qwen3-30B-A3B",
            "Q4": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
            "FP8": "Qwen/Qwen3-30B-A3B-FP8",
            "BF16": "Qwen/Qwen3-30B-A3B",
        },
        "precision_variants": _VARIANTS_QWEN_30B_MOE,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_30B_MOE),
        "architecture": "Qwen3MoeForCausalLM",
        "model_type": "qwen3_moe",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "A100",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "A100",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_30B_MOE),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_30B_MOE),
        "compatible_gpu_tiers": ["L4", "v6e-1", "A100", "A100-80GB", "H100", "G4"],
        "min_vram_gb": 20.8,
        "stop_token_ids": [151645, 151643],
        "default_precision_by_gpu": {
            "L4": "Q4",
            "v6e-1": "Q4",
            "A100": "Q4",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "precision_by_gpu": {
            "L4": "Q4",
            "v6e-1": "Q4",
            "A100": "Q4",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "max_num_seqs": 64,
        "storage_uri": None,
        "description": "Fast Qwen3-30B-A3B MoE general & agent workhorse (3.3B active/token, fits L4/v6e-1/A100 in Q4 and 80GB+ in BF16).",
    },
    "qwen-coder-30b-moe": {
        "alias": "qwen-coder-30b-moe",
        "family": "qwen",
        "total_params_b": 30.5,
        "active_params_b": 3.3,
        "is_moe": True,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "fallback_hf_repo_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "quantized_hf_repo_id": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
        "quantized_repo_ids": {
            "awq_4bit": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "compressed-tensors": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "fp8": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            "bfloat16": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "Q4": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "FP8": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            "BF16": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        },
        "precision_variants": _VARIANTS_QWEN_CODER_30B_MOE,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_CODER_30B_MOE),
        "architecture": "Qwen3MoeForCausalLM",
        "model_type": "qwen3_moe",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "A100",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "A100",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_CODER_30B_MOE),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_CODER_30B_MOE),
        "compatible_gpu_tiers": ["L4", "v6e-1", "A100", "A100-80GB", "H100", "G4"],
        "min_vram_gb": 20.8,
        "stop_token_ids": [151645, 151643],
        "default_precision_by_gpu": {
            "L4": "Q4",
            "v6e-1": "Q4",
            "A100": "Q4",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "precision_by_gpu": {
            "L4": "Q4",
            "v6e-1": "Q4",
            "A100": "Q4",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "max_num_seqs": 64,
        "storage_uri": None,
        "description": "Dedicated Qwen3-Coder-30B-A3B MoE coding specialist (fits L4/v6e-1/A100 in Q4 and 80GB+ in BF16).",
    },
    "qwen-14b": {
        "alias": "qwen-14b",
        "family": "qwen",
        "total_params_b": 14.8,
        "active_params_b": 14.8,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-14B",
        "fallback_hf_repo_id": "Qwen/Qwen3-14B",
        "quantized_hf_repo_id": "Qwen/Qwen3-14B-AWQ",
        "quantized_repo_ids": {
            "awq_4bit": "Qwen/Qwen3-14B-AWQ",
            "fp8": "Qwen/Qwen3-14B-FP8",
            "bfloat16": "Qwen/Qwen3-14B",
            "Q4": "Qwen/Qwen3-14B-AWQ",
            "FP8": "Qwen/Qwen3-14B-FP8",
            "BF16": "Qwen/Qwen3-14B",
        },
        "precision_variants": _VARIANTS_QWEN_14B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_14B),
        "architecture": "Qwen3ForCausalLM",
        "model_type": "qwen3",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "L4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "L4",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_14B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_14B),
        "compatible_gpu_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_14B),
        "min_vram_gb": 12.2,
        "stop_token_ids": [151645, 151643],
        "default_precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "Q8",
            "v6e-1": "Q8",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "Q8",
            "v6e-1": "Q8",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "max_num_seqs": 48,
        "storage_uri": None,
        "description": "Balanced Qwen3-14B dense coding & tool specialist (fits T4/v5e-1 in Q4, L4/v6e-1 in Q8, A100+ in BF16).",
    },
    "qwen-8b": {
        "alias": "qwen-8b",
        "family": "qwen",
        "total_params_b": 8.2,
        "active_params_b": 8.2,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-8B",
        "fallback_hf_repo_id": "Qwen/Qwen3-8B",
        "quantized_hf_repo_id": "Qwen/Qwen3-8B-AWQ",
        "quantized_repo_ids": {
            "awq_4bit": "Qwen/Qwen3-8B-AWQ",
            "fp8": "Qwen/Qwen3-8B-FP8",
            "bfloat16": "Qwen/Qwen3-8B",
            "Q4": "Qwen/Qwen3-8B-AWQ",
            "FP8": "Qwen/Qwen3-8B-FP8",
            "BF16": "Qwen/Qwen3-8B",
        },
        "precision_variants": _VARIANTS_QWEN_8B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_8B),
        "architecture": "Qwen3ForCausalLM",
        "model_type": "qwen3",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "T4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "T4",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_8B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_8B),
        "compatible_gpu_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_8B),
        "min_vram_gb": 8.7,
        "stop_token_ids": [151645, 151643],
        "default_precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "BF16",
            "v6e-1": "BF16",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "BF16",
            "v6e-1": "BF16",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "max_num_seqs": 48,
        "storage_uri": None,
        "description": "Compact Qwen3-8B dense specialist for T4/v5e-1 solo or co-resident multi-model stacks.",
    },
    "qwen-4b": {
        "alias": "qwen-4b",
        "family": "qwen",
        "total_params_b": 4.0,
        "active_params_b": 4.0,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3-4B",
        "fallback_hf_repo_id": "Qwen/Qwen3-4B",
        "quantized_hf_repo_id": "Qwen/Qwen3-4B-AWQ",
        "quantized_repo_ids": {
            "awq_4bit": "Qwen/Qwen3-4B-AWQ",
            "fp8": "Qwen/Qwen3-4B-FP8",
            "bfloat16": "Qwen/Qwen3-4B",
            "Q4": "Qwen/Qwen3-4B-AWQ",
            "FP8": "Qwen/Qwen3-4B-FP8",
            "BF16": "Qwen/Qwen3-4B",
        },
        "precision_variants": _VARIANTS_QWEN_4B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_4B),
        "architecture": "Qwen3ForCausalLM",
        "model_type": "qwen3",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "T4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "T4",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_4B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_4B),
        "compatible_gpu_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_4B),
        "min_vram_gb": 6.4,
        "stop_token_ids": [151645, 151643],
        "default_precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "BF16",
            "v6e-1": "BF16",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "BF16",
            "v6e-1": "BF16",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "max_num_seqs": 64,
        "storage_uri": None,
        "description": "Ultra-light Qwen3-4B fast router/verifier model (2.4 GB in Q4, ideal for 2-3 model co-resident stacks).",
    },
    "qwen-122b-moe": {
        "alias": "qwen-122b-moe",
        "family": "qwen",
        "total_params_b": 122.0,
        "active_params_b": 10.0,
        "is_moe": True,
        "source": "hf_hub",
        "hf_repo_id": "Qwen/Qwen3.5-122B-A10B",
        "fallback_hf_repo_id": "Qwen/Qwen3.5-122B-A10B",
        "quantized_hf_repo_id": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        "quantized_repo_ids": {
            "gptq_4bit": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
            "fp8": "Qwen/Qwen3.5-122B-A10B-FP8",
            "bfloat16": "Qwen/Qwen3.5-122B-A10B",
            "Q4": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
            "FP8": "Qwen/Qwen3.5-122B-A10B-FP8",
            "BF16": "Qwen/Qwen3.5-122B-A10B",
        },
        "precision_variants": _VARIANTS_QWEN_122B_MOE,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_122B_MOE),
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "model_type": "qwen3_5_moe",
        "template_family": "qwen3_chatml",
        "preferred_gpu": "G4",
        "min_gpu_tier": "A100-80GB",
        "recommended_gpu_tier": "G4",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_122B_MOE),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_QWEN_122B_MOE),
        "compatible_gpu_tiers": _compute_supported_hardware_tiers(_VARIANTS_QWEN_122B_MOE),
        "min_vram_gb": 70.0,
        "empirical_lm_weights_gib": 67.65,
        "checkpoint_disk_gib": 73.45,
        "stop_token_ids": [248046, 248044],
        "default_precision_by_gpu": {
            "A100-80GB": "Q4",
            "H100": "Q4",
            "G4": "Q4",
        },
        "precision_by_gpu": {
            "A100-80GB": "Q4",
            "H100": "Q4",
            "G4": "Q4",
        },
        "max_model_len": 32768,
        "max_model_len_a100": 32768,
        "max_num_seqs": 32,
        "storage_uri": None,
        "description": "Flagship Qwen3.5-122B-A10B MoE frontier model (67.65 GiB LM weights in Q4; prefers G4 96GB, fits A100-80GB & H100 in bounded solo mode).",
    },
    "r1-7b": {
        "alias": "r1-7b",
        "family": "deepseek",
        "total_params_b": 7.6,
        "active_params_b": 7.6,
        "is_moe": False,
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "fallback_hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "quantized_hf_repo_id": "casperhansen/deepseek-r1-distill-qwen-7b-awq",
        "quantized_repo_ids": {
            "awq_4bit": "casperhansen/deepseek-r1-distill-qwen-7b-awq",
            "bfloat16": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Q4": "casperhansen/deepseek-r1-distill-qwen-7b-awq",
            "BF16": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        },
        "precision_variants": _VARIANTS_R1_7B,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_R1_7B),
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "template_family": "deepseek_r1_distill",
        "preferred_gpu": "T4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "T4",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_R1_7B),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_R1_7B),
        "compatible_gpu_tiers": _compute_supported_hardware_tiers(_VARIANTS_R1_7B),
        "min_vram_gb": 8.4,
        "stop_token_ids": [151643, 151645],
        "default_precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "BF16",
            "v6e-1": "BF16",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "BF16",
            "v6e-1": "BF16",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "max_num_seqs": 48,
        "storage_uri": None,
        "description": "Compact DeepSeek-R1-Distill-Qwen-7B reasoning specialist (4.4 GB in Q4, ideal co-resident reasoning slot).",
    },
    "deepseek-v2-lite": {
        "alias": "deepseek-v2-lite",
        "family": "deepseek",
        "total_params_b": 15.7,
        "active_params_b": 2.4,
        "is_moe": True,
        "source": "hf_hub",
        "hf_repo_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "fallback_hf_repo_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "quantized_hf_repo_id": "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
        "quantized_repo_ids": {
            "awq_4bit": "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
            "bfloat16": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "Q4": "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
            "BF16": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        },
        "precision_variants": _VARIANTS_DEEPSEEK_V2_LITE,
        "supported_hardware_tiers": _compute_supported_hardware_tiers(_VARIANTS_DEEPSEEK_V2_LITE),
        "architecture": "DeepseekV2ForCausalLM",
        "model_type": "deepseek_v2",
        "template_family": "deepseek_v2_chat",
        "preferred_gpu": "L4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "L4",
        "supported_gpus": _compute_supported_hardware_tiers(_VARIANTS_DEEPSEEK_V2_LITE),
        "compatible_gpus": _compute_supported_hardware_tiers(_VARIANTS_DEEPSEEK_V2_LITE),
        "compatible_gpu_tiers": _compute_supported_hardware_tiers(_VARIANTS_DEEPSEEK_V2_LITE),
        "min_vram_gb": 12.8,
        "stop_token_ids": [100001],
        "default_precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "Q4",
            "v6e-1": "Q4",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "precision_by_gpu": {
            "v5e-1": "Q4",
            "T4": "Q4",
            "L4": "Q4",
            "v6e-1": "Q4",
            "A100": "BF16",
            "A100-80GB": "BF16",
            "H100": "BF16",
            "G4": "BF16",
        },
        "max_model_len": 16384,
        "max_model_len_a100": 32768,
        "max_num_seqs": 64,
        "storage_uri": None,
        "description": "Fast DeepSeek-V2-Lite-Chat MoE model (15.7B total / 2.4B active, 8.8 GB in Q4).",
    },
    "flux-klein-4b": {
        "alias": "flux-klein-4b",
        "source": "hf_hub",
        "hf_repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "quantized_hf_repo_id": "black-forest-labs/FLUX.2-klein-4b-fp8",
        "quantized_repo_ids": {
            "fp8": "black-forest-labs/FLUX.2-klein-4b-fp8",
            "gguf": "unsloth/FLUX.2-klein-4B-GGUF",
            "bfloat16": "black-forest-labs/FLUX.2-klein-base-4B",
        },
        "architecture": "FluxPipeline",
        "model_type": "diffusion_image",
        "template_family": "image_generation",
        "preferred_gpu": "T4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "T4",
        "supported_gpus": ["T4", "L4", "A100", "A100-80GB", "H100", "G4"],
        "compatible_gpus": ["T4", "L4", "A100", "A100-80GB", "H100", "G4"],
        "compatible_gpu_tiers": ["T4", "L4", "A100"],
        "min_vram_gb": 12.0,
        "stop_token_ids": [],
        "default_precision_by_gpu": {"T4": "fp8", "L4": "fp8", "A100": "bfloat16", "A100-80GB": "bfloat16", "H100": "bfloat16", "G4": "bfloat16"},
        "precision_by_gpu": {"T4": "fp8", "L4": "fp8", "A100": "bfloat16", "A100-80GB": "bfloat16", "H100": "bfloat16", "G4": "bfloat16"},
        "max_model_len": 512,
        "max_num_seqs": 1,
        "storage_uri": None,
        "modality": "image",
        "description": "Optional auxiliary image generation model (FLUX.2 Klein 4B).",
    },
    "flux-klein-9b": {
        "alias": "flux-klein-9b",
        "source": "hf_hub",
        "hf_repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "quantized_hf_repo_id": "unsloth/FLUX.2-klein-9B-GGUF",
        "quantized_repo_ids": {
            "gguf": "unsloth/FLUX.2-klein-9B-GGUF",
            "bfloat16": "black-forest-labs/FLUX.2-klein-9B",
        },
        "architecture": "FluxPipeline",
        "model_type": "diffusion_image",
        "template_family": "image_generation",
        "preferred_gpu": "L4",
        "min_gpu_tier": "L4",
        "recommended_gpu_tier": "L4",
        "supported_gpus": ["L4", "A100", "A100-80GB", "H100", "G4"],
        "compatible_gpus": ["L4", "A100", "A100-80GB", "H100", "G4"],
        "compatible_gpu_tiers": ["L4", "A100"],
        "min_vram_gb": 18.0,
        "stop_token_ids": [],
        "default_precision_by_gpu": {"L4": "gguf", "A100": "bfloat16", "A100-80GB": "bfloat16", "H100": "bfloat16", "G4": "bfloat16"},
        "precision_by_gpu": {"L4": "gguf", "A100": "bfloat16", "A100-80GB": "bfloat16", "H100": "bfloat16", "G4": "bfloat16"},
        "max_model_len": 512,
        "max_num_seqs": 1,
        "storage_uri": None,
        "modality": "image",
        "description": "Optional auxiliary image generation model (FLUX.2 Klein 9B).",
    },
    "z-image-turbo": {
        "alias": "z-image-turbo",
        "source": "hf_hub",
        "hf_repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "quantized_hf_repo_id": "unsloth/Z-Image-Turbo-FP8",
        "quantized_repo_ids": {
            "fp8": "unsloth/Z-Image-Turbo-FP8",
            "bfloat16": "Tongyi-MAI/Z-Image-Turbo",
        },
        "architecture": "ZImagePipeline",
        "model_type": "diffusion_image",
        "template_family": "image_generation",
        "preferred_gpu": "T4",
        "min_gpu_tier": "T4",
        "recommended_gpu_tier": "T4",
        "supported_gpus": ["T4", "L4", "A100", "A100-80GB", "H100", "G4"],
        "compatible_gpus": ["T4", "L4", "A100", "A100-80GB", "H100", "G4"],
        "compatible_gpu_tiers": ["T4", "L4", "A100"],
        "min_vram_gb": 12.0,
        "stop_token_ids": [],
        "default_precision_by_gpu": {"T4": "fp8", "L4": "fp8", "A100": "bfloat16", "A100-80GB": "bfloat16", "H100": "bfloat16", "G4": "bfloat16"},
        "precision_by_gpu": {"T4": "fp8", "L4": "fp8", "A100": "bfloat16", "A100-80GB": "bfloat16", "H100": "bfloat16", "G4": "bfloat16"},
        "max_model_len": 512,
        "max_num_seqs": 1,
        "storage_uri": None,
        "modality": "image",
        "description": "Optional fast auxiliary image generation model (Z-Image-Turbo).",
    },
}

ALL_MODEL_CATALOG: Dict[str, Dict[str, Any]] = BUILTIN_MODEL_CATALOG

HARDWARE_STACK_PRESETS: Dict[str, Dict[str, Any]] = {
    "G4": {
        "tier": "G4",
        "default_solo": {"alias": "qwen-35b-moe", "precision": "BF16"},
        "default_stack": "premium",
        "solo_options": [
            {"alias": "qwen3.8-27b", "precision": "BF16"},
            {"alias": "qwen-35b-moe", "precision": "BF16"},
            {"alias": "r1-32b", "precision": "BF16"},
            {"alias": "qwen-30b-moe", "precision": "BF16"},
            {"alias": "qwen-coder-30b-moe", "precision": "BF16"},
            {"alias": "qwen-27b", "precision": "BF16"},
            {"alias": "qwen-122b-moe", "precision": "Q4"},
        ],
        "presets": {
            "premium": {
                "preset_id": "premium",
                "label": "Premium Stack (x=3)",
                "description": "Qwen3.5/3.6-35B-A3B (FP8) + Qwen3-Coder-30B-A3B (Q8) + DeepSeek-R1-Distill-Qwen-14B (Q8)",
                "models": [
                    {"alias": "qwen-35b-moe", "precision": "FP8", "role": "planner"},
                    {"alias": "qwen-coder-30b-moe", "precision": "Q8", "role": "coder"},
                    {"alias": "r1-14b", "precision": "Q8", "role": "reasoner"},
                ],
            },
            "heavy-agent": {
                "preset_id": "heavy-agent",
                "label": "Heavy Agent Stack (x=3)",
                "description": "Qwen3.5/3.6-35B-A3B (Q4) + Qwen3-30B-A3B (Q4) + DeepSeek-V2-Lite (Q8)",
                "models": [
                    {"alias": "qwen-35b-moe", "precision": "Q4", "role": "planner"},
                    {"alias": "qwen-30b-moe", "precision": "Q4", "role": "executor"},
                    {"alias": "deepseek-v2-lite", "precision": "Q8", "role": "router"},
                ],
            },
        },
    },
    "H100": {
        "tier": "H100",
        "default_solo": {"alias": "qwen-35b-moe", "precision": "BF16"},
        "default_stack": "best-balanced",
        "solo_options": [
            {"alias": "qwen3.8-27b", "precision": "BF16"},
            {"alias": "qwen-35b-moe", "precision": "BF16"},
            {"alias": "qwen-30b-moe", "precision": "BF16"},
            {"alias": "qwen-coder-30b-moe", "precision": "BF16"},
            {"alias": "qwen-27b", "precision": "BF16"},
            {"alias": "r1-32b", "precision": "BF16"},
            {"alias": "qwen-122b-moe", "precision": "Q4"},
        ],
        "presets": {
            "best-balanced": {
                "preset_id": "best-balanced",
                "label": "Best Balanced Stack (x=3)",
                "description": "Qwen3.5/3.6-35B-A3B (Q4) + Qwen3-Coder-30B-A3B (Q4) + DeepSeek-R1-Distill-Qwen-7B (Q8)",
                "models": [
                    {"alias": "qwen-35b-moe", "precision": "Q4", "role": "planner"},
                    {"alias": "qwen-coder-30b-moe", "precision": "Q4", "role": "coder"},
                    {"alias": "r1-7b", "precision": "Q8", "role": "verifier"},
                ],
            },
            "max-quality": {
                "preset_id": "max-quality",
                "label": "Max Quality Stack (x=2)",
                "description": "Qwen3.5/3.6-35B-A3B (FP8) + DeepSeek-R1-Distill-Qwen-32B (Q4)",
                "models": [
                    {"alias": "qwen-35b-moe", "precision": "FP8", "role": "primary"},
                    {"alias": "r1-32b", "precision": "Q4", "role": "reasoner"},
                ],
            },
            "speed-stack": {
                "preset_id": "speed-stack",
                "label": "Speed Stack (x=3)",
                "description": "Qwen3-30B-A3B (Q4) + DeepSeek-V2-Lite (Q4) + Qwen3.5-9B (Q8)",
                "models": [
                    {"alias": "qwen-30b-moe", "precision": "Q4", "role": "executor"},
                    {"alias": "deepseek-v2-lite", "precision": "Q4", "role": "router"},
                    {"alias": "qwen-9b", "precision": "Q8", "role": "fast-coder"},
                ],
            },
        },
    },
    "A100-80GB": {
        "tier": "A100-80GB",
        "default_solo": {"alias": "qwen-35b-moe", "precision": "BF16"},
        "default_stack": "best-balanced",
        "solo_options": [
            {"alias": "qwen3.8-27b", "precision": "BF16"},
            {"alias": "qwen-35b-moe", "precision": "BF16"},
            {"alias": "qwen-30b-moe", "precision": "BF16"},
            {"alias": "qwen-coder-30b-moe", "precision": "BF16"},
            {"alias": "qwen-27b", "precision": "BF16"},
            {"alias": "r1-32b", "precision": "BF16"},
            {"alias": "qwen-122b-moe", "precision": "Q4"},
        ],
        "presets": {
            "best-balanced": {
                "preset_id": "best-balanced",
                "label": "Best Balanced Stack (x=3)",
                "description": "Qwen3.5/3.6-35B-A3B (Q4) + Qwen3-Coder-30B-A3B (Q4) + DeepSeek-R1-Distill-Qwen-7B (Q8)",
                "models": [
                    {"alias": "qwen-35b-moe", "precision": "Q4", "role": "planner"},
                    {"alias": "qwen-coder-30b-moe", "precision": "Q4", "role": "coder"},
                    {"alias": "r1-7b", "precision": "Q8", "role": "verifier"},
                ],
            },
            "max-quality": {
                "preset_id": "max-quality",
                "label": "Max Quality Stack (x=2)",
                "description": "Qwen3.5/3.6-35B-A3B (Q8) + DeepSeek-R1-Distill-Qwen-32B (Q4)",
                "models": [
                    {"alias": "qwen-35b-moe", "precision": "Q8", "role": "primary"},
                    {"alias": "r1-32b", "precision": "Q4", "role": "reasoner"},
                ],
            },
            "speed-stack": {
                "preset_id": "speed-stack",
                "label": "Speed Stack (x=3)",
                "description": "Qwen3-30B-A3B (Q4) + DeepSeek-V2-Lite (Q4) + Qwen3.5-9B (Q8)",
                "models": [
                    {"alias": "qwen-30b-moe", "precision": "Q4", "role": "executor"},
                    {"alias": "deepseek-v2-lite", "precision": "Q4", "role": "router"},
                    {"alias": "qwen-9b", "precision": "Q8", "role": "fast-coder"},
                ],
            },
        },
    },
    "A100": {
        "tier": "A100",
        "default_solo": {"alias": "qwen-30b-moe", "precision": "Q4"},
        "default_stack": "recommended",
        "solo_options": [
            {"alias": "qwen3.8-27b", "precision": "Q4"},
            {"alias": "qwen-30b-moe", "precision": "Q4"},
            {"alias": "qwen-35b-moe", "precision": "Q4"},
            {"alias": "qwen-coder-30b-moe", "precision": "Q4"},
            {"alias": "qwen-27b", "precision": "Q4"},
            {"alias": "r1-32b", "precision": "Q4"},
            {"alias": "qwen-14b", "precision": "Q8"},
        ],
        "presets": {
            "recommended": {
                "preset_id": "recommended",
                "label": "Recommended Stack (x=2)",
                "description": "Qwen3-30B-A3B (Q4) + DeepSeek-R1-Distill-Qwen-14B (Q4)",
                "models": [
                    {"alias": "qwen-30b-moe", "precision": "Q4", "role": "primary"},
                    {"alias": "r1-14b", "precision": "Q4", "role": "reasoner"},
                ],
            },
            "alternative": {
                "preset_id": "alternative",
                "label": "Alternative Stack (x=2)",
                "description": "Qwen3.5/3.6-27B (Q4) + DeepSeek-V2-Lite (Q4)",
                "models": [
                    {"alias": "qwen-27b", "precision": "Q4", "role": "primary"},
                    {"alias": "deepseek-v2-lite", "precision": "Q4", "role": "router"},
                ],
            },
            "light": {
                "preset_id": "light",
                "label": "Light Stack (x=3)",
                "description": "Qwen3-14B (Q6) + Qwen3.5-9B (Q8) + DeepSeek-R1-Distill-Qwen-7B (Q4)",
                "models": [
                    {"alias": "qwen-14b", "precision": "Q6", "role": "coder"},
                    {"alias": "qwen-9b", "precision": "Q8", "role": "executor"},
                    {"alias": "r1-7b", "precision": "Q4", "role": "verifier"},
                ],
            },
        },
    },
    "v6e-1": {
        "tier": "v6e-1",
        "default_solo": {"alias": "qwen-30b-moe", "precision": "Q4"},
        "default_stack": "recommended",
        "solo_options": [
            {"alias": "qwen3.8-27b", "precision": "Q4"},
            {"alias": "qwen-30b-moe", "precision": "Q4"},
            {"alias": "qwen-35b-moe", "precision": "Q4"},
            {"alias": "qwen-coder-30b-moe", "precision": "Q4"},
            {"alias": "qwen-27b", "precision": "Q4"},
            {"alias": "r1-32b", "precision": "Q4"},
            {"alias": "qwen-14b", "precision": "Q8"},
        ],
        "presets": {
            "recommended": {
                "preset_id": "recommended",
                "label": "Recommended TPU Stack (x=2)",
                "description": "Qwen3-30B-A3B (Q4) + DeepSeek-R1-Distill-Qwen-7B (Q4)",
                "models": [
                    {"alias": "qwen-30b-moe", "precision": "Q4", "role": "primary"},
                    {"alias": "r1-7b", "precision": "Q4", "role": "reasoner"},
                ],
            },
            "balanced": {
                "preset_id": "balanced",
                "label": "Balanced TPU Stack (x=2)",
                "description": "Qwen3.5/3.6-27B (Q4) + DeepSeek-V2-Lite (Q4)",
                "models": [
                    {"alias": "qwen-27b", "precision": "Q4", "role": "primary"},
                    {"alias": "deepseek-v2-lite", "precision": "Q4", "role": "router"},
                ],
            },
            "multi": {
                "preset_id": "multi",
                "label": "Multi TPU Stack (x=3)",
                "description": "Qwen3-14B (Q5) + DeepSeek-R1-Distill-Qwen-14B (Q4) + Qwen3-4B (Q4)",
                "models": [
                    {"alias": "qwen-14b", "precision": "Q5", "role": "coder"},
                    {"alias": "r1-14b", "precision": "Q4", "role": "reasoner"},
                    {"alias": "qwen-4b", "precision": "Q4", "role": "router"},
                ],
            },
        },
    },
    "L4": {
        "tier": "L4",
        "default_solo": {"alias": "qwen-30b-moe", "precision": "Q4"},
        "default_stack": "recommended",
        "solo_options": [
            {"alias": "qwen3.8-27b", "precision": "Q4"},
            {"alias": "qwen-30b-moe", "precision": "Q4"},
            {"alias": "qwen-coder-30b-moe", "precision": "Q4"},
            {"alias": "qwen-27b", "precision": "Q4"},
            {"alias": "qwen-35b-moe", "precision": "Q4"},
            {"alias": "qwen-14b", "precision": "Q8"},
            {"alias": "r1-14b", "precision": "Q5"},
        ],
        "presets": {
            "recommended": {
                "preset_id": "recommended",
                "label": "Recommended Stack (x=2)",
                "description": "Qwen3-14B (Q4) + DeepSeek-R1-Distill-Qwen-7B (Q5)",
                "models": [
                    {"alias": "qwen-14b", "precision": "Q4", "role": "coder"},
                    {"alias": "r1-7b", "precision": "Q5", "role": "reasoner"},
                ],
            },
            "light-multi": {
                "preset_id": "light-multi",
                "label": "Light Multi Stack (x=3)",
                "description": "Qwen3.5-9B (Q6) + DeepSeek-V2-Lite (Q4) + Qwen3-4B (Q4)",
                "models": [
                    {"alias": "qwen-9b", "precision": "Q6", "role": "coder"},
                    {"alias": "deepseek-v2-lite", "precision": "Q4", "role": "executor"},
                    {"alias": "qwen-4b", "precision": "Q4", "role": "router"},
                ],
            },
        },
    },
    "T4": {
        "tier": "T4",
        "default_solo": {"alias": "qwen-9b", "precision": "Q4"},
        "default_stack": "recommended-pair",
        "solo_options": [
            {"alias": "qwen-9b", "precision": "Q4"},
            {"alias": "qwen-14b", "precision": "Q4"},
            {"alias": "r1-14b", "precision": "Q4"},
            {"alias": "qwen-8b", "precision": "Q4"},
            {"alias": "r1-7b", "precision": "Q4"},
            {"alias": "deepseek-v2-lite", "precision": "Q4"},
            {"alias": "qwen-4b", "precision": "Q4"},
        ],
        "presets": {
            "recommended-pair": {
                "preset_id": "recommended-pair",
                "label": "Recommended Pair (x=2)",
                "description": "Qwen3.5-9B (Q4) + DeepSeek-R1-Distill-Qwen-7B (Q4)",
                "models": [
                    {"alias": "qwen-9b", "precision": "Q4", "role": "coder"},
                    {"alias": "r1-7b", "precision": "Q4", "role": "reasoner"},
                ],
            },
            "coder-pair": {
                "preset_id": "coder-pair",
                "label": "Coder Pair (x=2)",
                "description": "Qwen3-14B (Q4) + Qwen3-4B (Q4)",
                "models": [
                    {"alias": "qwen-14b", "precision": "Q4", "role": "coder"},
                    {"alias": "qwen-4b", "precision": "Q4", "role": "router"},
                ],
            },
            "deepseek-pair": {
                "preset_id": "deepseek-pair",
                "label": "DeepSeek Pair (x=2)",
                "description": "DeepSeek-V2-Lite (Q4) + Qwen3-4B (Q4)",
                "models": [
                    {"alias": "deepseek-v2-lite", "precision": "Q4", "role": "executor"},
                    {"alias": "qwen-4b", "precision": "Q4", "role": "router"},
                ],
            },
            "light-multi": {
                "preset_id": "light-multi",
                "label": "Light Multi Stack (x=3)",
                "description": "Qwen3.5-9B (Q4) + DeepSeek-R1-Distill-Qwen-7B (Q4) + Qwen3-4B (Q4)",
                "models": [
                    {"alias": "qwen-9b", "precision": "Q4", "role": "coder"},
                    {"alias": "r1-7b", "precision": "Q4", "role": "reasoner"},
                    {"alias": "qwen-4b", "precision": "Q4", "role": "router"},
                ],
            },
        },
    },
    "v5e-1": {
        "tier": "v5e-1",
        "default_solo": {"alias": "qwen-9b", "precision": "Q4"},
        "default_stack": "recommended-pair",
        "solo_options": [
            {"alias": "qwen-9b", "precision": "Q4"},
            {"alias": "qwen-14b", "precision": "Q4"},
            {"alias": "r1-14b", "precision": "Q4"},
            {"alias": "qwen-8b", "precision": "Q4"},
            {"alias": "r1-7b", "precision": "Q4"},
            {"alias": "deepseek-v2-lite", "precision": "Q4"},
            {"alias": "qwen-4b", "precision": "Q4"},
        ],
        "presets": {
            "recommended-pair": {
                "preset_id": "recommended-pair",
                "label": "Recommended TPU Pair (x=2)",
                "description": "Qwen3.5-9B (Q4) + DeepSeek-R1-Distill-Qwen-7B (Q4)",
                "models": [
                    {"alias": "qwen-9b", "precision": "Q4", "role": "coder"},
                    {"alias": "r1-7b", "precision": "Q4", "role": "reasoner"},
                ],
            },
            "coder-pair": {
                "preset_id": "coder-pair",
                "label": "Coder TPU Pair (x=2)",
                "description": "Qwen3-14B (Q4) + Qwen3-4B (Q4)",
                "models": [
                    {"alias": "qwen-14b", "precision": "Q4", "role": "coder"},
                    {"alias": "qwen-4b", "precision": "Q4", "role": "router"},
                ],
            },
            "deepseek-pair": {
                "preset_id": "deepseek-pair",
                "label": "DeepSeek TPU Pair (x=2)",
                "description": "DeepSeek-V2-Lite (Q4) + Qwen3-4B (Q4)",
                "models": [
                    {"alias": "deepseek-v2-lite", "precision": "Q4", "role": "executor"},
                    {"alias": "qwen-4b", "precision": "Q4", "role": "router"},
                ],
            },
            "light-multi": {
                "preset_id": "light-multi",
                "label": "Light Multi TPU Stack (x=3)",
                "description": "Qwen3.5-9B (Q4) + DeepSeek-R1-Distill-Qwen-7B (Q4) + Qwen3-4B (Q4)",
                "models": [
                    {"alias": "qwen-9b", "precision": "Q4", "role": "coder"},
                    {"alias": "r1-7b", "precision": "Q4", "role": "reasoner"},
                    {"alias": "qwen-4b", "precision": "Q4", "role": "router"},
                ],
            },
        },
    },
}

# Bundled empirical metadata snapshots for offline <1ms protocol introspection (FR-3.3, AC-6)
BUNDLED_MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen3.5-9B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vocab_size": 248320,
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attn_layers": 8,
            "linear_attn_layers": 24,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
        },
        "generation_config": {
            "bos_token_id": 248044,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3.5-27B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vocab_size": 248320,
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attn_layers": 16,
            "linear_attn_layers": 48,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
        },
        "generation_config": {
            "bos_token_id": 248044,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3.6-27B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vocab_size": 248320,
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attn_layers": 16,
            "linear_attn_layers": 48,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
        },
        "generation_config": {
            "bos_token_id": 248044,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3.8-27B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vocab_size": 248320,
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attn_layers": 16,
            "linear_attn_layers": 48,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
        },
        "generation_config": {
            "bos_token_id": 248044,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "vocab_size": 248320,
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attn_layers": 10,
            "linear_attn_layers": 30,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
        },
        "generation_config": {
            "bos_token_id": 248044,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3.6-35B-A3B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "vocab_size": 248320,
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attn_layers": 10,
            "linear_attn_layers": 30,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
        },
        "generation_config": {
            "bos_token_id": 248044,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "tokenizer_config": {
            "tokenizer_class": "LlamaTokenizerFast",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
            "chat_template": DEEPSEEK_R1_DISTILL_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "vocab_size": 152064,
            "bos_token_id": 151646,
            "eos_token_id": 151643,
            "num_hidden_layers": 48,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "generation_config": {
            "bos_token_id": 151646,
            "eos_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "tokenizer_config": {
            "tokenizer_class": "LlamaTokenizerFast",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
            "chat_template": DEEPSEEK_R1_DISTILL_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "vocab_size": 152064,
            "bos_token_id": 151646,
            "eos_token_id": 151643,
            "num_hidden_layers": 64,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "generation_config": {
            "bos_token_id": 151646,
            "eos_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "tokenizer_config": {
            "tokenizer_class": "LlamaTokenizerFast",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
            "chat_template": DEEPSEEK_R1_DISTILL_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "vocab_size": 152064,
            "bos_token_id": 151646,
            "eos_token_id": 151643,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "head_dim": 128,
        },
        "generation_config": {
            "bos_token_id": 151646,
            "eos_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    },
    "Qwen/Qwen3-8B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": 151936,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "generation_config": {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3-14B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": 151936,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "generation_config": {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3-4B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": 151936,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "generation_config": {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3-30B-A3B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "vocab_size": 151936,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
        },
        "generation_config": {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "vocab_size": 151936,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
        },
        "generation_config": {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen3.5-122B-A10B": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": QWEN35_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "vocab_size": 248320,
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attn_layers": 12,
            "linear_attn_layers": 36,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 64,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
        },
        "generation_config": {
            "bos_token_id": 248044,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        },
        "model_config": {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "vocab_size": 152064,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
        },
        "generation_config": {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": False,
            "temperature": 0.0,
        },
    },
    "Qwen/Qwen2.5-Coder-14B-Instruct": {
        "tokenizer_config": {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        },
        "model_config": {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "vocab_size": 152064,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
        },
        "generation_config": {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": False,
            "temperature": 0.0,
        },
    },
    "deepseek-ai/DeepSeek-V2-Lite": {
        "tokenizer_config": {
            "tokenizer_class": "LlamaTokenizerFast",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
            "chat_template": DEEPSEEK_V2_LITE_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v2",
            "vocab_size": 102400,
            "bos_token_id": 100000,
            "eos_token_id": 100001,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        },
        "generation_config": {
            "bos_token_id": 100000,
            "eos_token_id": 100001,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    },
    "deepseek-ai/DeepSeek-V2-Lite-Chat": {
        "tokenizer_config": {
            "tokenizer_class": "LlamaTokenizerFast",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
            "chat_template": DEEPSEEK_V2_LITE_CHAT_TEMPLATE_JINJA,
        },
        "model_config": {
            "architectures": ["DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v2",
            "vocab_size": 102400,
            "bos_token_id": 100000,
            "eos_token_id": 100001,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        },
        "generation_config": {
            "bos_token_id": 100000,
            "eos_token_id": 100001,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    },
    "black-forest-labs/FLUX.2-klein-base-4B": {
        "tokenizer_config": {},
        "model_config": {
            "_class_name": "FluxPipeline",
            "architectures": ["FluxPipeline"],
            "model_type": "diffusion_image",
            "modality": "image",
        },
        "generation_config": {},
    },
    "black-forest-labs/FLUX.2-klein-9B": {
        "tokenizer_config": {},
        "model_config": {
            "_class_name": "FluxPipeline",
            "architectures": ["FluxPipeline"],
            "model_type": "diffusion_image",
            "modality": "image",
        },
        "generation_config": {},
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "tokenizer_config": {},
        "model_config": {
            "_class_name": "ZImagePipeline",
            "architectures": ["ZImagePipeline"],
            "model_type": "diffusion_image",
            "modality": "image",
        },
        "generation_config": {},
    },
}

# Map quantized repo IDs and aliases to canonical bundled metadata keys
_REPO_METADATA_ALIASES: Dict[str, str] = {
    "qwen-9b": "Qwen/Qwen3.5-9B",
    "cyankiwi/Qwen3.5-9B-AWQ-4bit": "Qwen/Qwen3.5-9B",
    "mssfj/Qwen3.5-9B-GPTQ-INT4": "Qwen/Qwen3.5-9B",
    "QuantTrio/Qwen3.5-9B-AWQ": "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-GPTQ-Int4": "Qwen/Qwen3.5-9B",
    "r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "casperhansen/deepseek-r1-distill-qwen-7b-awq": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "r1-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "casperhansen/deepseek-r1-distill-qwen-14b-awq": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "qwen-27b": "Qwen/Qwen3.5-27B",
    "cyankiwi/Qwen3.5-27B-AWQ-4bit": "Qwen/Qwen3.5-27B",
    "Alennnndy/Qwen3.6-27B-AWQ-4bit": "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.5-27B-FP8": "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.6-27B-FP8": "Qwen/Qwen3.6-27B",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen-27b-38": "Qwen/Qwen3.8-27B",
    "cyankiwi/Qwen3.8-27B-AWQ-INT4": "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.8-27B-FP8": "Qwen/Qwen3.8-27B",
    "qwen-35b-moe": "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-35B-A3B-FP8": "Qwen/Qwen3.5-35B-A3B",
    "QuantTrio/Qwen3.6-35B-A3B-AWQ": "Qwen/Qwen3.6-35B-A3B",
    "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4": "Qwen/Qwen3.6-35B-A3B",
    "r1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "casperhansen/deepseek-r1-distill-qwen-32b-awq": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "qwen-4b": "Qwen/Qwen3-4B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-AWQ": "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-FP8": "Qwen/Qwen3-4B",
    "qwen-8b": "Qwen/Qwen3-8B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-AWQ": "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-FP8": "Qwen/Qwen3-8B",
    "qwen-14b": "Qwen/Qwen3-14B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-AWQ": "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-FP8": "Qwen/Qwen3-14B",
    "qwen-30b-moe": "Qwen/Qwen3-30B-A3B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4": "Qwen/Qwen3-30B-A3B",
    "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ": "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-AWQ": "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-FP8": "Qwen/Qwen3-30B-A3B",
    "qwen-coder-30b-moe": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen3-coder-30b-a3b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen-122b-moe": "Qwen/Qwen3.5-122B-A10B",
    "qwen3.5-122b-a10b": "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4": "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-122B-A10B-FP8": "Qwen/Qwen3.5-122B-A10B",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "sandcastle-qwen2.5-coder-14b-unified": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-v2-lite-chat": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "casperhansen/deepseek-v2-lite-chat-awq": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "flux-klein-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "black-forest-labs/FLUX.2-klein-4b-fp8": "black-forest-labs/FLUX.2-klein-base-4B",
    "unsloth/FLUX.2-klein-4B-GGUF": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
    "unsloth/FLUX.2-klein-9B-GGUF": "black-forest-labs/FLUX.2-klein-9B",
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",
    "unsloth/Z-Image-Turbo-FP8": "Tongyi-MAI/Z-Image-Turbo",
}

# Explicit quantization_config metadata for pre-quantized HF checkpoints (Spec 868 §3.1 & Spec 869 §4.1)
_REPO_QUANTIZATION_CONFIG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "cyankiwi/Qwen3.5-9B-AWQ-4bit": {"quant_method": "compressed-tensors", "bits": 4, "group_size": 32, "format": "pack-quantized"},
    "mssfj/Qwen3.5-9B-GPTQ-INT4": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    "QuantTrio/Qwen3.5-9B-AWQ": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "Qwen/Qwen3.5-9B-GPTQ-Int4": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ": {"quant_method": "compressed-tensors", "bits": 4, "group_size": 128, "format": "pack-quantized"},
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    "casperhansen/deepseek-r1-distill-qwen-7b-awq": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "casperhansen/deepseek-r1-distill-qwen-14b-awq": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "casperhansen/deepseek-r1-distill-qwen-32b-awq": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "casperhansen/deepseek-v2-lite-chat-awq": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "cyankiwi/Qwen3.5-27B-AWQ-4bit": {"quant_method": "compressed-tensors", "bits": 4, "group_size": 32, "format": "pack-quantized"},
    "Alennnndy/Qwen3.6-27B-AWQ-4bit": {"quant_method": "compressed-tensors", "bits": 4, "group_size": 32, "format": "pack-quantized"},
    "cyankiwi/Qwen3.8-27B-AWQ-INT4": {"quant_method": "compressed-tensors", "bits": 4, "group_size": 32, "format": "pack-quantized"},
    "QuantTrio/Qwen3.6-35B-A3B-AWQ": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "Qwen/Qwen3-4B-AWQ": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "Qwen/Qwen3-8B-AWQ": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "Qwen/Qwen3-14B-AWQ": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "Qwen/Qwen3-30B-A3B-AWQ": {"quant_method": "awq", "bits": 4, "group_size": 128},
    "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit": {"quant_method": "compressed-tensors", "bits": 4, "group_size": 32, "format": "pack-quantized"},
    "Qwen/Qwen3.5-27B-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3.6-27B-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3.8-27B-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3.5-35B-A3B-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3-4B-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3-8B-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3-14B-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3-30B-A3B-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": {"quant_method": "fp8", "bits": 8},
    "Qwen/Qwen3.5-122B-A10B-FP8": {"quant_method": "fp8", "bits": 8},
    "black-forest-labs/FLUX.2-klein-4b-fp8": {"quant_method": "fp8", "bits": 8},
    "unsloth/Z-Image-Turbo-FP8": {"quant_method": "fp8", "bits": 8},
    "unsloth/FLUX.2-klein-4B-GGUF": {"quant_method": "gguf", "bits": 4},
    "unsloth/FLUX.2-klein-9B-GGUF": {"quant_method": "gguf", "bits": 4},
}

# Pre-quantized checkpoints calibrated/serialized with torch_dtype="float16" (Spec 869 §4.1)
# Passing `--dtype bfloat16` to vLLM on these checkpoints triggers a dtype mismatch failure.
_REPO_TORCH_DTYPE_OVERRIDES: Dict[str, str] = {
    "Qwen/Qwen3-4B-AWQ": "float16",
    "Qwen/Qwen3-8B-AWQ": "float16",
    "Qwen/Qwen3-14B-AWQ": "float16",
    "QuantTrio/Qwen3.6-35B-A3B-AWQ": "float16",
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4": "float16",
    "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ": "float16",
}

_PROTOCOL_CACHE_MAXSIZE = 256
_PROTOCOL_INTROSPECTION_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}
_PROTOCOL_CACHE_LOCK = threading.Lock()


def clear_protocol_introspection_cache() -> None:
    """Clears the memoized model protocol introspection cache."""
    with _PROTOCOL_CACHE_LOCK:
        _PROTOCOL_INTROSPECTION_CACHE.clear()


_DEFAULT_HF_HUB_CACHE_ROOT: Path = Path.home() / ".cache" / "huggingface" / "hub"


def _find_local_hf_snapshot_dir(hf_repo_id: str) -> Optional[Path]:
    """Resolves the active snapshot directory for `hf_repo_id` in the local HuggingFace Hub cache with 1 stat."""
    clean_repo = str(hf_repo_id or "").strip().strip("/")
    if not clean_repo or "/" not in clean_repo or clean_repo.startswith((".", "/")):
        return None

    cache_roots: List[Path] = []
    env_hub_cache = os.environ.get("HF_HUB_CACHE")
    if env_hub_cache and env_hub_cache.strip():
        cache_roots.append(Path(env_hub_cache).expanduser())
    env_hf_home = os.environ.get("HF_HOME")
    if env_hf_home and env_hf_home.strip():
        cache_roots.append(Path(env_hf_home).expanduser() / "hub")
    cache_roots.append(_DEFAULT_HF_HUB_CACHE_ROOT)

    cache_dir_name = "models--" + clean_repo.replace("/", "--")
    seen_roots: set[str] = set()
    for root in cache_roots:
        root_key = str(root)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        model_cache_dir = root / cache_dir_name
        if not model_cache_dir.is_dir():
            continue
        snapshots_dir = model_cache_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue

        ref_main = model_cache_dir / "refs" / "main"
        if ref_main.is_file():
            try:
                rev = ref_main.read_text(encoding="utf-8").strip()
                if rev:
                    ref_snap = snapshots_dir / rev
                    if ref_snap.is_dir():
                        return ref_snap
            except Exception:
                pass

        try:
            snap_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if snap_dirs:
                snap_dirs.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
                return snap_dirs[0]
        except Exception:
            continue
    return None


def _find_local_hf_cached_file(hf_repo_id: str, filename: str) -> Optional[Path]:
    """Resolves `filename` from local HuggingFace Hub snapshot cache with zero network I/O (< 0.2 ms)."""
    snap_dir = _find_local_hf_snapshot_dir(hf_repo_id)
    if snap_dir is not None:
        cand = snap_dir / filename
        if cand.is_file():
            return cand
        # Fallback across sibling snapshots if the newest snapshot lacks `filename`
        parent = snap_dir.parent
        try:
            for sib in sorted((p for p in parent.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime_ns, reverse=True):
                sib_cand = sib / filename
                if sib_cand.is_file():
                    return sib_cand
        except Exception:
            pass
    return None


def _normalize_eos_id_list(raw_eos: Any) -> List[int]:
    if isinstance(raw_eos, int) and not isinstance(raw_eos, bool):
        return [int(raw_eos)]
    if isinstance(raw_eos, (list, tuple, set)):
        out: List[int] = []
        for item in raw_eos:
            if isinstance(item, int) and not isinstance(item, bool) and int(item) not in out:
                out.append(int(item))
        return out
    return []


def classify_model_family_neutral(
    identifier: str,
    model_config: Optional[Dict[str, Any]] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classifies model architecture family, loader kind, vocabulary size, and stop tokens
    deterministically from checkpoint metadata when present, or neutral family heuristics offline,
    NEVER defaulting unknown or non-Qwen3.5 models to Qwen3.5 or 248,320-token assumptions.
    """
    ident = str(identifier or "").strip()
    ident_lower = ident.lower()
    mod_cfg = model_config if isinstance(model_config, dict) else {}
    text_cfg = mod_cfg.get("text_config") if isinstance(mod_cfg.get("text_config"), dict) else {}
    tok_cfg = tokenizer_config if isinstance(tokenizer_config, dict) else {}

    raw_archs = mod_cfg.get("architectures") or []
    archs = [str(a).strip() for a in raw_archs if str(a).strip()] if isinstance(raw_archs, (list, tuple)) else []
    if not archs and mod_cfg.get("architecture"):
        archs = [str(mod_cfg["architecture"]).strip()]
    arch_set = set(archs)

    mt = str(mod_cfg.get("model_type") or "").strip()
    tmt = str(text_cfg.get("model_type") or mt).strip()
    mt_lower = mt.lower()
    tmt_lower = tmt.lower()

    bos_str = _extract_token_str(tok_cfg.get("bos_token")) or ""
    eos_str = _extract_token_str(tok_cfg.get("eos_token"))
    pad_str = _extract_token_str(tok_cfg.get("pad_token"))
    chat_tmpl = str(tok_cfg.get("chat_template") or "")

    has_config_arch = bool(mt_lower or tmt_lower or archs)
    is_r1_ident = (
        "deepseek-r1" in ident_lower
        or "r1-distill" in ident_lower
        or bool(re.search(r"(?:^|[/_\-\s])r1-(?:1\.5b|7b|8b|14b|32b|70b)(?:$|[/_\-\s])", ident_lower))
    )
    is_r1_tok = (
        ("begin▁of▁sentence" in bos_str and ("<｜User｜>" in chat_tmpl or "<｜Assistant｜>" in chat_tmpl or mt_lower in {"qwen2", "llama"}))
        or "<｜User｜>" in chat_tmpl
        or "<｜Assistant｜>" in chat_tmpl
    )

    family = "causal_lm"
    if (
        mt_lower == "diffusion_image"
        or mod_cfg.get("modality") == "image"
        or str(mod_cfg.get("_class_name") or "").endswith("Pipeline")
        or any(a.endswith("Pipeline") for a in archs)
        or any(tok in ident_lower for tok in ("flux-klein", "flux.2-klein", "z-image-turbo"))
    ):
        family = "diffusion_image"
    elif has_config_arch:
        if (
            mt_lower in {"qwen3_5_moe", "qwen3_5_moe_text"}
            or tmt_lower in {"qwen3_5_moe", "qwen3_5_moe_text"}
            or "Qwen3_5MoeForConditionalGeneration" in arch_set
        ):
            family = "qwen3_5_moe"
        elif (
            mt_lower in {"qwen3_5", "qwen3_5_text"}
            or tmt_lower in {"qwen3_5", "qwen3_5_text"}
            or "Qwen3_5ForConditionalGeneration" in arch_set
        ):
            family = "qwen3_5"
        elif any(a.endswith(("ForConditionalGeneration", "ForVision2Seq", "ImageTextToText")) for a in archs):
            family = "multimodal_vlm"
        elif mt_lower == "qwen3_moe" or "Qwen3MoeForCausalLM" in arch_set:
            family = "qwen3_moe"
        elif mt_lower == "qwen3" or "Qwen3ForCausalLM" in arch_set:
            family = "qwen3"
        elif (mt_lower in {"qwen2", "llama"} or "Qwen2ForCausalLM" in arch_set or "LlamaForCausalLM" in arch_set) and (
            is_r1_tok or is_r1_ident
        ):
            family = "deepseek_r1"
        elif mt_lower == "qwen2" or "Qwen2ForCausalLM" in arch_set:
            family = "qwen2"
        elif mt_lower == "deepseek_v2" or "DeepseekV2ForCausalLM" in arch_set:
            family = "deepseek_v2"
        elif mt_lower == "deepseek_v3" or "DeepseekV3ForCausalLM" in arch_set:
            family = "deepseek_v3"
        elif mt_lower == "llama" or "LlamaForCausalLM" in arch_set:
            family = "llama"
        elif mt_lower in {"mistral", "mixtral"} or "MistralForCausalLM" in arch_set or "MixtralForCausalLM" in arch_set:
            family = "mistral"
        else:
            family = "causal_lm"
    else:
        has_q35_token = any(
            tok in ident_lower
            for tok in (
                "qwen3.5",
                "qwen3_5",
                "qwen3.6",
                "qwen3_6",
                "qwen3.8",
                "qwen3_8",
                "qwen-9b",
                "qwen-27b",
                "qwen3.8-27b",
                "qwen-35b-moe",
                "qwen-122b-moe",
                "122b-a10b",
            )
        )
        has_moe_token = any(tok in ident_lower for tok in ("moe", "a3b", "a10b", "35b", "122b", "a22b"))
        has_qwen3_token = (
            bool(re.search(r"qwen[-_]?3(?![._]?[568])", ident_lower))
            or ident_lower in ("qwen-4b", "qwen-8b", "qwen-14b", "qwen-30b-moe", "qwen-coder-30b-moe")
            or any(tok in ident_lower for tok in ("qwen3-coder", "qwen-coder-30b"))
        )
        has_qwen2_token = any(tok in ident_lower for tok in ("qwen2", "qwen-2", "qwq"))

        if has_q35_token and has_moe_token:
            family = "qwen3_5_moe"
        elif has_q35_token:
            family = "qwen3_5"
        elif is_r1_ident or is_r1_tok:
            family = "deepseek_r1"
        elif any(tok in ident_lower for tok in ("deepseek-v2", "deepseek_v2", "deepseek-coder-v2")):
            family = "deepseek_v2"
        elif any(tok in ident_lower for tok in ("deepseek-v3", "deepseek_v3")):
            family = "deepseek_v3"
        elif has_qwen3_token and has_moe_token:
            family = "qwen3_moe"
        elif has_qwen3_token:
            family = "qwen3"
        elif has_qwen2_token:
            family = "qwen2"
        elif any(tok in ident_lower for tok in ("-vl", "llava", "vision-instruct")):
            family = "multimodal_vlm"
        elif "llama" in ident_lower:
            family = "llama"
        elif any(tok in ident_lower for tok in ("mistral", "mixtral", "codestral")):
            family = "mistral"
        else:
            family = "causal_lm"

    explicit_causal_lm = bool(archs) and all(a.endswith("ForCausalLM") for a in archs)
    is_llama_r1 = family == "deepseek_r1" and (mt_lower == "llama" or "LlamaForCausalLM" in arch_set or "llama" in ident_lower)

    family_defaults: Dict[str, Dict[str, Any]] = {
        "diffusion_image": {
            "architecture": str(mod_cfg.get("_class_name") or (archs[0] if archs else "DiffusionPipeline")),
            "model_type": "diffusion_image",
            "loader_kind": "diffusion_pipeline",
            "loader_class_name": str(mod_cfg.get("_class_name") or (archs[0] if archs else "DiffusionPipeline")),
            "is_qwen35_family": False,
            "template_family": "image_generation",
            "vocab_size": None,
            "eos_token_ids": [],
            "eos_token": None,
            "bos_token": None,
            "pad_token": None,
        },
        "qwen3_5": {
            "architecture": "Qwen3_5ForConditionalGeneration",
            "model_type": "qwen3_5",
            "loader_kind": "causal_lm" if explicit_causal_lm else "conditional_generation",
            "loader_class_name": "AutoModelForCausalLM" if explicit_causal_lm else "Qwen3_5ForConditionalGeneration",
            "is_qwen35_family": True,
            "template_family": "qwen3_chatml",
            "vocab_size": 248320,
            "eos_token_ids": [248046, 248044],
            "eos_token": "<|im_end|>",
            "bos_token": None,
            "pad_token": "<|endoftext|>",
        },
        "qwen3_5_moe": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "model_type": "qwen3_5_moe",
            "loader_kind": "causal_lm" if explicit_causal_lm else "conditional_generation",
            "loader_class_name": "AutoModelForCausalLM" if explicit_causal_lm else "Qwen3_5MoeForConditionalGeneration",
            "is_qwen35_family": True,
            "template_family": "qwen3_chatml",
            "vocab_size": 248320,
            "eos_token_ids": [248046, 248044],
            "eos_token": "<|im_end|>",
            "bos_token": None,
            "pad_token": "<|endoftext|>",
        },
        "multimodal_vlm": {
            "architecture": "AutoModelForImageTextToText",
            "model_type": mt or "multimodal_vlm",
            "loader_kind": "conditional_generation",
            "loader_class_name": "AutoModelForImageTextToText",
            "is_qwen35_family": False,
            "template_family": "generic_chat",
            "vocab_size": None,
            "eos_token_ids": [],
            "eos_token": None,
            "bos_token": None,
            "pad_token": None,
        },
        "qwen3": {
            "architecture": "Qwen3ForCausalLM",
            "model_type": "qwen3",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "qwen3_chatml",
            "vocab_size": 151936,
            "eos_token_ids": [151645, 151643],
            "eos_token": "<|im_end|>",
            "bos_token": None,
            "pad_token": "<|endoftext|>",
        },
        "qwen3_moe": {
            "architecture": "Qwen3MoeForCausalLM",
            "model_type": "qwen3_moe",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "qwen3_chatml",
            "vocab_size": 151936,
            "eos_token_ids": [151645, 151643],
            "eos_token": "<|im_end|>",
            "bos_token": None,
            "pad_token": "<|endoftext|>",
        },
        "qwen2": {
            "architecture": "Qwen2ForCausalLM",
            "model_type": "qwen2",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "qwen2_chatml",
            "vocab_size": 152064,
            "eos_token_ids": [151645, 151643],
            "eos_token": "<|im_end|>",
            "bos_token": None,
            "pad_token": "<|endoftext|>",
        },
        "deepseek_r1": {
            "architecture": "LlamaForCausalLM" if is_llama_r1 else "Qwen2ForCausalLM",
            "model_type": "llama" if is_llama_r1 else "qwen2",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "deepseek_r1_distill",
            "vocab_size": 128256 if is_llama_r1 else 152064,
            "eos_token_ids": [128001] if is_llama_r1 else [151643],
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
        },
        "deepseek_v2": {
            "architecture": "DeepseekV2ForCausalLM",
            "model_type": "deepseek_v2",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "deepseek_v2_chat",
            "vocab_size": 102400,
            "eos_token_ids": [100001],
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
        },
        "deepseek_v3": {
            "architecture": "DeepseekV3ForCausalLM",
            "model_type": "deepseek_v3",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "deepseek_v2_chat",
            "vocab_size": 129280,
            "eos_token_ids": [1],
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
        },
        "llama": {
            "architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "generic_chat",
            "vocab_size": 128256,
            "eos_token_ids": [128009, 128001],
            "eos_token": "<|eot_id|>",
            "bos_token": "<|begin_of_text|>",
            "pad_token": None,
        },
        "mistral": {
            "architecture": "MistralForCausalLM",
            "model_type": "mistral",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "generic_chat",
            "vocab_size": 32768,
            "eos_token_ids": [2],
            "eos_token": "</s>",
            "bos_token": "<s>",
            "pad_token": None,
        },
        "causal_lm": {
            "architecture": "AutoModelForCausalLM",
            "model_type": mt or "causal_lm",
            "loader_kind": "causal_lm",
            "loader_class_name": "AutoModelForCausalLM",
            "is_qwen35_family": False,
            "template_family": "generic_chat",
            "vocab_size": None,
            "eos_token_ids": [],
            "eos_token": None,
            "bos_token": None,
            "pad_token": None,
        },
    }

    spec = family_defaults[family]
    cfg_vocab = text_cfg.get("vocab_size", mod_cfg.get("vocab_size"))
    resolved_vocab = int(cfg_vocab) if isinstance(cfg_vocab, int) and cfg_vocab > 0 else spec["vocab_size"]
    cfg_eos = _normalize_eos_id_list(text_cfg.get("eos_token_id", mod_cfg.get("eos_token_id")))
    if cfg_eos:
        resolved_eos_ids = cfg_eos
    else:
        resolved_eos_ids = list(spec["eos_token_ids"])
    if isinstance(resolved_vocab, int) and resolved_vocab > 0:
        resolved_eos_ids = [tid for tid in resolved_eos_ids if isinstance(tid, int) and 0 <= tid < resolved_vocab]

    return {
        "architecture_family": family,
        "architecture": archs[0] if archs else spec["architecture"],
        "model_type": mt or spec["model_type"],
        "loader_kind": spec["loader_kind"],
        "loader_class_name": spec["loader_class_name"],
        "is_qwen35_family": bool(spec["is_qwen35_family"]),
        "template_family": spec["template_family"],
        "vocab_size": resolved_vocab,
        "eos_token_ids": resolved_eos_ids,
        "stop_token_ids": list(resolved_eos_ids),
        "eos_token": eos_str if eos_str is not None else spec["eos_token"],
        "bos_token": bos_str if bos_str else spec["bos_token"],
        "pad_token": pad_str if pad_str is not None else spec["pad_token"],
    }


def _load_local_or_cached_metadata(identifier: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    """Loads `(model_config, tokenizer_config, generation_config, chat_template_jinja)` from
    local directory -> local HF snapshot cache -> BUNDLED_MODEL_METADATA (< 1 ms, zero network).
    """
    target = str(identifier or "").strip()
    canonical_key = _REPO_METADATA_ALIASES.get(target, _REPO_METADATA_ALIASES.get(target.lower(), target))
    mod_cfg: Dict[str, Any] = {}
    tok_cfg: Dict[str, Any] = {}
    gen_cfg: Dict[str, Any] = {}
    jinja_text = ""

    local_path = Path(target).expanduser() if target else Path("")
    if target and local_path.is_dir():
        for fname, dest in (
            ("config.json", mod_cfg),
            ("model_index.json", mod_cfg),
            ("tokenizer_config.json", tok_cfg),
            ("generation_config.json", gen_cfg),
        ):
            fpath = local_path / fname
            if fpath.is_file():
                try:
                    loaded = json.loads(fpath.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        dest.update(loaded)
                except Exception:
                    pass
        jinja_file = local_path / "chat_template.jinja"
        if jinja_file.is_file():
            try:
                jinja_text = jinja_file.read_text(encoding="utf-8")
            except Exception:
                jinja_text = ""
        if mod_cfg or tok_cfg:
            return mod_cfg, tok_cfg, gen_cfg, jinja_text

    for repo_candidate in dict.fromkeys([target, canonical_key]):
        if "/" in repo_candidate and not repo_candidate.startswith((".", "/")):
            cfg_snap = _find_local_hf_cached_file(repo_candidate, "config.json") or _find_local_hf_cached_file(repo_candidate, "model_index.json")
            tok_snap = _find_local_hf_cached_file(repo_candidate, "tokenizer_config.json")
            gen_snap = _find_local_hf_cached_file(repo_candidate, "generation_config.json")
            jinja_snap = _find_local_hf_cached_file(repo_candidate, "chat_template.jinja")
            if cfg_snap is not None:
                try:
                    loaded = json.loads(cfg_snap.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        mod_cfg.update(loaded)
                except Exception:
                    pass
            if tok_snap is not None:
                try:
                    loaded = json.loads(tok_snap.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        tok_cfg.update(loaded)
                except Exception:
                    pass
            if gen_snap is not None:
                try:
                    loaded = json.loads(gen_snap.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        gen_cfg.update(loaded)
                except Exception:
                    pass
            if jinja_snap is not None:
                try:
                    jinja_text = jinja_snap.read_text(encoding="utf-8")
                except Exception:
                    jinja_text = ""
            if mod_cfg or tok_cfg:
                break

    if not mod_cfg and not tok_cfg:
        if canonical_key in BUNDLED_MODEL_METADATA:
            bundled = BUNDLED_MODEL_METADATA[canonical_key]
            mod_cfg = copy.deepcopy(bundled.get("model_config", {}))
            tok_cfg = copy.deepcopy(bundled.get("tokenizer_config", {}))
            gen_cfg = copy.deepcopy(bundled.get("generation_config", {}))
        else:
            for b_key, bundled in BUNDLED_MODEL_METADATA.items():
                if b_key.lower() == canonical_key.lower():
                    mod_cfg = copy.deepcopy(bundled.get("model_config", {}))
                    tok_cfg = copy.deepcopy(bundled.get("tokenizer_config", {}))
                    gen_cfg = copy.deepcopy(bundled.get("generation_config", {}))
                    break

    # Preserve pre-quantized checkpoint quantization_config when target is a quantized repo ID (Spec 868 §3.1)
    if isinstance(mod_cfg, dict) and "quantization_config" not in mod_cfg:
        q_override = _REPO_QUANTIZATION_CONFIG_OVERRIDES.get(target)
        if q_override is None:
            for q_key, q_val in _REPO_QUANTIZATION_CONFIG_OVERRIDES.items():
                if q_key.lower() == target.lower():
                    q_override = q_val
                    break
        if q_override is None:
            t_low = target.lower()
            if "gptq" in t_low:
                q_override = {"quant_method": "gptq", "bits": 4, "group_size": 128}
            elif "awq" in t_low:
                q_override = {"quant_method": "awq", "bits": 4, "group_size": 128}
            elif "-fp8" in t_low or "_fp8" in t_low or t_low.endswith("fp8"):
                q_override = {"quant_method": "fp8", "bits": 8}
            elif "gguf" in t_low:
                q_override = {"quant_method": "gguf", "bits": 4}
        if q_override is not None:
            mod_cfg["quantization_config"] = copy.deepcopy(q_override)

    if isinstance(mod_cfg, dict) and "torch_dtype" not in mod_cfg:
        dt_override = _REPO_TORCH_DTYPE_OVERRIDES.get(target)
        if dt_override is None:
            for d_key, d_val in _REPO_TORCH_DTYPE_OVERRIDES.items():
                if d_key.lower() == target.lower():
                    dt_override = d_val
                    break
        if dt_override is not None:
            mod_cfg["torch_dtype"] = dt_override

    return mod_cfg, tok_cfg, gen_cfg, jinja_text


def _enrich_catalog_profile_metadata(alias: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Populates neutral architecture_family, loader_kind, loader_class_name, is_qwen35_family, and vocab_size."""
    if profile.get("modality") == "image" or profile.get("model_type") == "diffusion_image":
        profile.setdefault("architecture_family", "diffusion_image")
        profile.setdefault("loader_kind", "diffusion_pipeline")
        profile.setdefault("loader_class_name", str(profile.get("architecture") or "DiffusionPipeline"))
        profile.setdefault("is_qwen35_family", False)
        profile.setdefault("vocab_size", None)
        return profile

    repo_or_alias = str(profile.get("hf_repo_id") or alias)
    mod_cfg, tok_cfg, _, _ = _load_local_or_cached_metadata(repo_or_alias)
    synth_mod = dict(mod_cfg)
    if profile.get("model_type") and not synth_mod.get("model_type"):
        synth_mod["model_type"] = profile["model_type"]
    if profile.get("architecture") and not synth_mod.get("architectures"):
        synth_mod["architectures"] = [profile["architecture"]]
    family_info = classify_model_family_neutral(
        repo_or_alias,
        model_config=synth_mod or None,
        tokenizer_config=tok_cfg or None,
    )
    profile.setdefault("architecture_family", family_info["architecture_family"])
    profile.setdefault("loader_kind", family_info["loader_kind"])
    profile.setdefault("loader_class_name", family_info["loader_class_name"])
    profile.setdefault("is_qwen35_family", family_info["is_qwen35_family"])
    profile.setdefault("vocab_size", family_info["vocab_size"])
    return profile


def get_model_catalog(config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Returns the complete model catalog merging the 5 built-in daily-driver models,
    compatibility aliases, auxiliary image models, and custom user models from config.
    """
    catalog: Dict[str, Dict[str, Any]] = {
        alias: _enrich_catalog_profile_metadata(alias, copy.deepcopy(prof))
        for alias, prof in BUILTIN_MODEL_CATALOG.items()
    }
    if config and isinstance(config, dict):
        models_section = config.get("models")
        if isinstance(models_section, dict):
            custom_models = models_section.get("custom")
            if isinstance(custom_models, dict):
                for alias, entry in custom_models.items():
                    if not isinstance(entry, dict):
                        continue
                    explicit_base = entry.get("base_alias")
                    if explicit_base and str(explicit_base) in catalog:
                        base_profile = copy.deepcopy(catalog[str(explicit_base)])
                        base_profile.update(entry)
                        base_profile["alias"] = str(alias)
                    else:
                        repo_id = str(entry.get("hf_repo_id") or entry.get("local_dir") or alias)
                        mod_cfg, tok_cfg, _, _ = _load_local_or_cached_metadata(repo_id)
                        synth_mod = dict(mod_cfg)
                        if entry.get("model_type") and not synth_mod.get("model_type"):
                            synth_mod["model_type"] = entry["model_type"]
                        if entry.get("architecture") and not synth_mod.get("architectures"):
                            synth_mod["architectures"] = [entry["architecture"]]
                        family_info = classify_model_family_neutral(
                            repo_id,
                            model_config=synth_mod or None,
                            tokenizer_config=tok_cfg or None,
                        )
                        repo_lower = f"{alias} {repo_id}".lower()
                        is_large = any(sz in repo_lower for sz in ("27b", "30b", "32b", "35b", "70b", "72b"))
                        is_med = any(sz in repo_lower for sz in ("14b", "16b"))
                        preferred_gpu = "A100" if is_large else ("L4" if is_med else "T4")
                        min_vram = 34.0 if is_large else (20.0 if is_med else 13.5)
                        supported_gpus = (
                            ["A100", "A100-80GB"]
                            if is_large
                            else (["L4", "A100", "A100-80GB"] if is_med else ["T4", "L4", "A100", "A100-80GB"])
                        )
                        base_profile = {
                            "alias": str(alias),
                            "source": "custom",
                            "hf_repo_id": str(entry.get("hf_repo_id") or alias),
                            "fallback_hf_repo_id": str(entry.get("hf_repo_id") or alias),
                            "quantized_hf_repo_id": str(
                                entry.get("quantized_hf_repo_id") or entry.get("hf_repo_id") or alias
                            ),
                            "quantized_repo_ids": {"default": str(entry.get("hf_repo_id") or alias)},
                            "architecture": family_info["architecture"],
                            "model_type": family_info["model_type"],
                            "architecture_family": family_info["architecture_family"],
                            "loader_kind": family_info["loader_kind"],
                            "loader_class_name": family_info["loader_class_name"],
                            "is_qwen35_family": family_info["is_qwen35_family"],
                            "vocab_size": family_info["vocab_size"],
                            "template_family": family_info["template_family"],
                            "preferred_gpu": preferred_gpu,
                            "min_gpu_tier": "L4" if is_large else "T4",
                            "recommended_gpu_tier": preferred_gpu,
                            "supported_gpus": supported_gpus,
                            "compatible_gpus": supported_gpus,
                            "compatible_gpu_tiers": ["L4", "A100"] if is_large else ["T4", "L4", "A100"],
                            "min_vram_gb": min_vram,
                            "stop_token_ids": list(family_info["eos_token_ids"]),
                            "default_precision_by_gpu": {
                                "T4": "nf4_4bit",
                                "L4": "awq_4bit",
                                "A100": "awq_4bit",
                                "A100-80GB": "bfloat16",
                            },
                            "precision_by_gpu": {
                                "T4": "nf4_4bit",
                                "L4": "awq_4bit",
                                "A100": "awq_4bit",
                                "A100-80GB": "bfloat16",
                            },
                            "max_model_len": 32768 if is_large else 16384,
                            "max_model_len_a100": 32768,
                            "max_num_seqs": 4,
                            "storage_uri": None,
                            "description": f"Custom model profile for {alias}",
                        }
                        base_profile.update(entry)
                        base_profile["alias"] = str(alias)

                    base_profile.setdefault("hf_repo_id", str(entry.get("hf_repo_id") or alias))
                    base_profile.setdefault(
                        "quantized_hf_repo_id",
                        entry.get("quantized_hf_repo_id", base_profile["hf_repo_id"]),
                    )
                    base_profile.setdefault("compatible_gpus", base_profile.get("supported_gpus", ["T4", "L4", "A100"]))
                    base_profile.setdefault(
                        "compatible_gpu_tiers",
                        base_profile.get("compatible_gpu_tiers", ["T4", "L4", "A100"]),
                    )
                    base_profile.setdefault("precision_by_gpu", base_profile.get("default_precision_by_gpu", {}))
                    catalog[str(alias)] = _enrich_catalog_profile_metadata(str(alias), base_profile)
    return catalog


def get_merged_model_catalog(config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Alias for get_model_catalog(config) returning the merged built-in + custom model catalog."""
    return get_model_catalog(config)


get_Merged_model_catalog = get_merged_model_catalog


def resolve_model_profile(
    alias_or_hf_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolves a built-in alias, custom alias, local checkpoint directory, or HuggingFace repo ID
    into a normalized, architecture-neutral catalog profile.
    """
    catalog = get_model_catalog(config)
    query = (alias_or_hf_id or "qwen-9b").strip()
    if "@" in query and not Path(query).expanduser().exists():
        query = query.split("@", 1)[0].strip()
    if query in catalog:
        return catalog[query]

    query_lower = query.lower()
    for alias, profile in catalog.items():
        if alias.lower() == query_lower:
            return profile
        if str(profile.get("hf_repo_id", "")).lower() == query_lower:
            return profile
        if str(profile.get("quantized_hf_repo_id", "")).lower() == query_lower:
            return profile
        for q_repo in (profile.get("quantized_repo_ids") or {}).values():
            if str(q_repo).lower() == query_lower:
                return profile
        for v_entry in (profile.get("precision_variants") or {}).values():
            if isinstance(v_entry, dict) and str(v_entry.get("hf_repo_id", "")).lower() == query_lower:
                return profile

    # Metadata-driven neutral profile construction for local checkpoints, cached repos, or arbitrary HF IDs
    mod_cfg, tok_cfg, _, _ = _load_local_or_cached_metadata(query)
    family_info = classify_model_family_neutral(
        query,
        model_config=mod_cfg or None,
        tokenizer_config=tok_cfg or None,
    )
    quant_cfg = mod_cfg.get("quantization_config") if isinstance(mod_cfg.get("quantization_config"), dict) else None

    is_xl = any(sz in query_lower for sz in ("122b", "110b", "72b", "70b"))
    is_large = any(sz in query_lower for sz in ("27b", "30b", "32b", "35b"))
    is_med = any(sz in query_lower for sz in ("14b", "16b"))
    preferred_gpu = "H100" if is_xl else ("A100" if is_large else ("L4" if is_med else "T4"))
    min_vram = 70.0 if is_xl else (22.0 if is_large else (12.2 if is_med else 9.0))
    supported_gpus = (
        ["G4", "H100", "A100-80GB"]
        if is_xl
        else (
            ["G4", "H100", "A100-80GB", "A100", "v6e-1", "L4"]
            if is_large
            else list(ALL_HARDWARE_TIERS)
        )
    )
    compatible_gpu_tiers = ["A100-80GB", "H100", "G4"] if is_xl else (["L4", "A100"] if is_large else ["T4", "L4", "A100"])
    return {
        "alias": query,
        "source": "local" if Path(query).expanduser().is_dir() else "hf_hub",
        "hf_repo_id": query,
        "fallback_hf_repo_id": query,
        "quantized_hf_repo_id": query,
        "quantized_repo_ids": {"default": query},
        "architecture": family_info["architecture"],
        "model_type": family_info["model_type"],
        "architecture_family": family_info["architecture_family"],
        "loader_kind": family_info["loader_kind"],
        "loader_class_name": family_info["loader_class_name"],
        "is_qwen35_family": family_info["is_qwen35_family"],
        "vocab_size": family_info["vocab_size"],
        "quantization_config": quant_cfg,
        "template_family": family_info["template_family"],
        "preferred_gpu": preferred_gpu,
        "min_gpu_tier": "A100-80GB" if is_xl else ("L4" if is_large else "T4"),
        "recommended_gpu_tier": preferred_gpu,
        "supported_gpus": supported_gpus,
        "compatible_gpus": supported_gpus,
        "supported_hardware_tiers": supported_gpus,
        "compatible_gpu_tiers": compatible_gpu_tiers,
        "min_vram_gb": min_vram,
        "stop_token_ids": list(family_info["eos_token_ids"]),
        "default_precision_by_gpu": {
            "v5e-1": "nf4_4bit",
            "T4": "nf4_4bit",
            "L4": "awq_4bit",
            "v6e-1": "awq_4bit",
            "A100": "awq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "precision_by_gpu": {
            "v5e-1": "nf4_4bit",
            "T4": "nf4_4bit",
            "L4": "awq_4bit",
            "v6e-1": "awq_4bit",
            "A100": "awq_4bit",
            "A100-80GB": "bfloat16",
            "H100": "bfloat16",
            "G4": "bfloat16",
        },
        "max_model_len": 32768 if (is_large or is_xl) else 16384,
        "max_model_len_a100": 32768,
        "max_num_seqs": 4,
        "storage_uri": None,
        "description": f"Custom HuggingFace model profile for {query}",
    }


def resolve_model_precision_variant(
    model_alias: str,
    precision: Optional[str] = None,
    *,
    gpu_tier: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolves a model alias + optional precision (or hardware tier default) into concrete
    precision metadata (`precision`, `legacy_precision`, `weights_gb`, `min_vram_gb`, `hf_repo_id`,
    `quant_method`, `source`, `storage_uri`).
    """
    raw_alias = str(model_alias or "qwen-9b").strip()
    inline_prec: Optional[str] = None
    if "@" in raw_alias and not Path(raw_alias).expanduser().exists():
        raw_alias, inline_prec = raw_alias.split("@", 1)
        raw_alias = raw_alias.strip()
        inline_prec = inline_prec.strip() or None

    profile = resolve_model_profile(raw_alias, config=config)
    canonical_tier: Optional[str] = None
    if gpu_tier and str(gpu_tier).strip().lower() != "auto":
        canonical_tier = normalize_hardware_tier(gpu_tier, allow_auto=False)

    precision_map = profile.get("default_precision_by_gpu") or profile.get("precision_by_gpu") or {}
    eff_raw_prec = precision or inline_prec
    if not eff_raw_prec and canonical_tier:
        eff_raw_prec = precision_map.get(canonical_tier)
    if not eff_raw_prec:
        eff_raw_prec = precision_map.get(profile.get("preferred_gpu", "T4")) or "Q4"

    norm_code = normalize_precision_code(eff_raw_prec)
    variants = profile.get("precision_variants") or {}
    variant = variants.get(norm_code)

    if isinstance(variant, dict):
        weights_gb = float(variant["weights_gb"])
        min_vram_gb = float(variant["min_vram_gb"])
        hf_repo_id = str(variant["hf_repo_id"])
        quant_method = str(variant["quant_method"])
        bits = int(variant.get("bits", 4))
    else:
        base_min = float(profile.get("min_vram_gb", 13.5))
        base_w = max(2.0, round(base_min - MIN_KV_CACHE_HEADROOM_GB, 1))
        scale = {"BF16": 3.4, "FP8": 1.8, "INT8": 1.8, "Q8": 1.9, "Q6": 1.45, "Q5": 1.25, "Q4": 1.0}[norm_code]
        weights_gb = round(base_w * scale, 1)
        min_vram_gb = round(weights_gb + MIN_KV_CACHE_HEADROOM_GB, 1)
        quant_repos = profile.get("quantized_repo_ids") or {}
        hf_repo_id = str(
            quant_repos.get(str(eff_raw_prec))
            or quant_repos.get(norm_code)
            or (profile.get("hf_repo_id") if norm_code == "BF16" else profile.get("quantized_hf_repo_id"))
            or profile.get("hf_repo_id")
            or raw_alias
        )
        quant_method = "bfloat16" if norm_code == "BF16" else ("bnb_8bit" if norm_code in ("Q8", "INT8") else "nf4_4bit")
        bits = {"BF16": 16, "FP8": 8, "INT8": 8, "Q8": 8, "Q6": 6, "Q5": 5, "Q4": 4}[norm_code]

    legacy_prec = str(eff_raw_prec) if str(eff_raw_prec) in ("nf4_4bit", "awq_4bit", "gptq_4bit", "bfloat16", "fp8") else norm_code
    return {
        "alias": profile["alias"],
        "family": profile.get("family", "qwen"),
        "precision": norm_code,
        "legacy_precision": legacy_prec,
        "weights_gb": weights_gb,
        "min_vram_gb": min_vram_gb,
        "hf_repo_id": hf_repo_id,
        "base_hf_repo_id": profile.get("hf_repo_id", hf_repo_id),
        "quant_method": quant_method,
        "bits": bits,
        "source": profile.get("source", "hf_hub"),
        "storage_uri": profile.get("storage_uri"),
        "template_family": profile.get("template_family", "qwen3_chatml"),
    }


def route_gpu_for_model_alias(
    model_alias: str,
    agent_count: int = 1,
    gpu_override: Optional[str] = None,
    requested_gpu: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    precision: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Determines the target Colab GPU/TPU tier, precision, max_model_len, and batch concurrency
    for a given model_alias and concurrent agent_count across all 8 Colab accelerators.
    """
    raw_alias = str(model_alias or "qwen-9b").strip()
    inline_prec: Optional[str] = None
    if "@" in raw_alias and not Path(raw_alias).expanduser().exists():
        raw_alias, inline_prec = raw_alias.split("@", 1)
        raw_alias = raw_alias.strip()
        inline_prec = inline_prec.strip() or None
    eff_precision = precision or inline_prec

    profile = resolve_model_profile(raw_alias, config=config)
    preferred_gpu = normalize_hardware_tier(profile.get("preferred_gpu", "T4"), allow_auto=False)
    supported = profile.get("supported_hardware_tiers") or profile.get("supported_gpus") or list(ALL_HARDWARE_TIERS)

    eff_override = gpu_override or requested_gpu
    effective_gpu = preferred_gpu
    if eff_override and str(eff_override).strip().lower() != "auto":
        effective_gpu = normalize_hardware_tier(str(eff_override), allow_auto=False)
    elif eff_precision:
        var_check = resolve_model_precision_variant(profile["alias"], precision=eff_precision, config=config)
        req_mem = float(var_check["min_vram_gb"])
        if float(GPU_TIER_SPECS[effective_gpu]["vram_gb"]) < req_mem:
            ascending_tiers = sorted(ALL_HARDWARE_TIERS, key=lambda t: (float(GPU_TIER_SPECS[t]["vram_gb"]), int(GPU_TIER_SPECS[t]["rank"])))
            gpu_candidates = [t for t in ascending_tiers if GPU_TIER_SPECS[t]["accelerator_type"] == "GPU" and float(GPU_TIER_SPECS[t]["vram_gb"]) >= req_mem]
            if gpu_candidates:
                effective_gpu = gpu_candidates[0]

    # Upgrade T4 -> L4 or A100 when high concurrent agent counts are requested without explicit override
    if not (eff_override and str(eff_override).strip().lower() != "auto"):
        if effective_gpu == "T4" and agent_count >= 6:
            effective_gpu = "L4" if "L4" in supported else "A100"
        elif effective_gpu in ("T4", "L4") and agent_count >= 10:
            effective_gpu = "A100"

    tier_spec = GPU_TIER_SPECS[effective_gpu]
    target_vram_gb = float(tier_spec["vram_gb"])

    precision_map = profile.get("default_precision_by_gpu") or profile.get("precision_by_gpu") or {}
    selected_precision = eff_precision or precision_map.get(effective_gpu, "nf4_4bit")
    variant_info = resolve_model_precision_variant(
        profile["alias"],
        precision=selected_precision,
        gpu_tier=effective_gpu,
        config=config,
    )
    quant_repos = profile.get("quantized_repo_ids") or {}
    resolved_hf_repo = (
        quant_repos.get(str(selected_precision))
        or variant_info["hf_repo_id"]
        or profile.get("quantized_hf_repo_id")
        or profile["hf_repo_id"]
    )
    weights_gb = float(variant_info["weights_gb"])
    free_kv_gb = round(target_vram_gb - weights_gb, 2)

    max_model_len = (
        profile.get("max_model_len_a100", 32768)
        if effective_gpu in ("A100", "A100-80GB", "H100", "G4", "v6e-1")
        else profile.get("max_model_len", 16384)
    )
    base_seqs = int(profile.get("max_num_seqs", 4))
    tier_max_seqs = int(tier_spec.get("max_num_seqs", 12))
    max_num_seqs = max(base_seqs, min(max(12, tier_max_seqs), max(1, int(agent_count))))

    return {
        "model_alias": profile["alias"],
        "hf_repo_id": profile["hf_repo_id"],
        "resolved_checkpoint_repo_id": resolved_hf_repo,
        "gpu_tier": effective_gpu,
        "recommended_gpu": effective_gpu,
        "accelerator_type": tier_spec["accelerator_type"],
        "memory_type": tier_spec["memory_type"],
        "target_vram_gb": target_vram_gb,
        "nominal_vram_gb": float(tier_spec["nominal_vram_gb"]),
        "weights_gb": weights_gb,
        "free_kv_gb": free_kv_gb,
        "min_vram_gb": float(variant_info["min_vram_gb"]),
        "precision": selected_precision,
        "normalized_precision": variant_info["precision"],
        "max_model_len": int(max_model_len),
        "max_num_seqs": max_num_seqs,
        "agent_count": max(1, int(agent_count)),
        "storage_uri": profile.get("storage_uri"),
        "template_family": profile.get("template_family", "qwen3_chatml"),
    }


def _infer_hardware_tier_from_memory(avail_gb: float) -> str:
    for cand_tier in ("G4", "A100-80GB", "A100", "v6e-1", "L4"):
        tier_vram = float(GPU_TIER_SPECS[cand_tier]["vram_gb"])
        if avail_gb >= (tier_vram - MIN_KV_CACHE_HEADROOM_GB - 2.0):
            return cand_tier
    return "T4"


def verify_vram_compatibility(
    model_alias: str,
    available_vram_gb: float | str = 16.0,
    config: Optional[Dict[str, Any]] = None,
    *,
    active_gpu_tier: Optional[str] = None,
    agent_count: int = 1,
    precision: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verifies whether available_vram_gb (or any of the 8 Colab GPU/TPU hardware tiers)
    satisfies the model profile's minimum physical memory floor (weights_gb + 4.0 GB KV headroom).
    """
    raw_alias = str(model_alias or "qwen-9b").strip()
    inline_prec: Optional[str] = None
    if "@" in raw_alias and not Path(raw_alias).expanduser().exists():
        raw_alias, inline_prec = raw_alias.split("@", 1)
        raw_alias = raw_alias.strip()
        inline_prec = inline_prec.strip() or None
    explicit_precision = precision or inline_prec

    catalog = get_model_catalog(config)
    profile = catalog.get(raw_alias) or resolve_model_profile(raw_alias, config=config)
    base_min_vram_gb = float(profile.get("min_vram_gb", 13.5))
    preferred_gpu = str(
        profile.get("preferred_gpu")
        or ("H100" if base_min_vram_gb >= 70.0 else ("A100" if base_min_vram_gb >= 19.0 else ("L4" if base_min_vram_gb >= 12.0 else "T4")))
    )

    if isinstance(available_vram_gb, str):
        canonical_tier = normalize_hardware_tier(available_vram_gb, allow_auto=False)
        active_gpu_tier = active_gpu_tier or canonical_tier
        avail = float(GPU_TIER_SPECS[canonical_tier]["vram_gb"])
    else:
        avail = float(available_vram_gb)

    if active_gpu_tier:
        try:
            inferred_gpu = normalize_hardware_tier(active_gpu_tier, allow_auto=False)
        except ValueError:
            inferred_gpu = active_gpu_tier.strip().upper()
    else:
        inferred_gpu = _infer_hardware_tier_from_memory(avail)

    precision_map = profile.get("default_precision_by_gpu") or profile.get("precision_by_gpu") or {}
    if explicit_precision:
        selected_precision = explicit_precision
        variant_info = resolve_model_precision_variant(
            profile["alias"],
            precision=explicit_precision,
            gpu_tier=inferred_gpu if inferred_gpu in GPU_TIER_SPECS else None,
            config=config,
        )
        effective_min_vram = float(variant_info["min_vram_gb"])
        weights_gb = float(variant_info["weights_gb"])
    else:
        selected_precision = precision_map.get(inferred_gpu) or next(iter(precision_map.values()), "nf4_4bit")
        variant_info = resolve_model_precision_variant(
            profile["alias"],
            precision=selected_precision,
            gpu_tier=inferred_gpu if inferred_gpu in GPU_TIER_SPECS else None,
            config=config,
        )
        effective_min_vram = float(variant_info["min_vram_gb"])
        weights_gb = float(variant_info["weights_gb"])

        # If default tier precision exceeds available memory (e.g. BF16 on partial VRAM), step down to highest fitting precision
        variants = profile.get("precision_variants") or {}
        if avail < effective_min_vram and variants:
            for cand_code in PRECISION_TIERS:
                cand_var = variants.get(cand_code)
                if isinstance(cand_var, dict) and avail >= float(cand_var["min_vram_gb"]):
                    variant_info = resolve_model_precision_variant(
                        profile["alias"],
                        precision=cand_code,
                        gpu_tier=inferred_gpu if inferred_gpu in GPU_TIER_SPECS else None,
                        config=config,
                    )
                    effective_min_vram = float(variant_info["min_vram_gb"])
                    weights_gb = float(variant_info["weights_gb"])
                    if cand_code == "Q4" and selected_precision in ("nf4_4bit", "awq_4bit", "gptq_4bit"):
                        pass
                    else:
                        selected_precision = cand_code
                    break
            else:
                # None fit; report the Q4 floor so diagnostic states true minimum requirement
                q4_var = variants.get("Q4")
                if isinstance(q4_var, dict):
                    effective_min_vram = float(q4_var["min_vram_gb"])
                    weights_gb = float(q4_var["weights_gb"])

    quant_repos = profile.get("quantized_repo_ids") or {}
    quantized_hf_repo_id = str(
        profile.get("quantized_hf_repo_id")
        or variant_info.get("hf_repo_id")
        or profile.get("hf_repo_id")
    )
    resolved_checkpoint_repo_id = str(
        quant_repos.get(str(selected_precision))
        or variant_info.get("hf_repo_id")
        or quantized_hf_repo_id
    )
    free_kv_gb = round(avail - weights_gb, 2)
    tier_spec = GPU_TIER_SPECS.get(inferred_gpu, GPU_TIER_SPECS["T4"])

    if avail >= effective_min_vram and free_kv_gb >= MIN_KV_CACHE_HEADROOM_GB - 1e-6:
        return {
            "compatible": True,
            "action": "hot_swap_or_load",
            "active_model": profile["alias"],
            "requested_alias": profile["alias"],
            "requested_model": profile["alias"],
            "available_vram_gb": avail,
            "required_min_vram_gb": effective_min_vram,
            "weights_gb": weights_gb,
            "free_kv_gb": free_kv_gb,
            "inferred_gpu_tier": inferred_gpu,
            "recommended_gpu": inferred_gpu,
            "preferred_gpu": preferred_gpu,
            "accelerator_type": tier_spec["accelerator_type"],
            "memory_type": tier_spec["memory_type"],
            "selected_precision": selected_precision,
            "normalized_precision": variant_info["precision"],
            "quantized_hf_repo_id": quantized_hf_repo_id,
            "resolved_checkpoint_repo_id": resolved_checkpoint_repo_id,
            "agent_count": max(1, int(agent_count)),
            "fallback_alias": None,
            "diagnostic": (
                f"VRAM check passed: {avail:.1f} GB available >= {effective_min_vram:.1f} GB required "
                f"for {profile['alias']} ({selected_precision}, {weights_gb:.1f} GB weights + "
                f"{free_kv_gb:.1f} GB free KV on {inferred_gpu})."
            ),
        }

    # Select largest compatible fallback model that fits safely in available memory
    if avail >= 23.5:
        fallback_alias = "qwen-30b-moe"
    elif avail >= 14.5:
        fallback_alias = "r1-14b" if avail >= 20.0 else "qwen-9b"
    else:
        fallback_alias = "qwen-4b" if avail >= 6.5 else "qwen-9b"

    return {
        "compatible": False,
        "action": "provision_higher_tier_or_quantize",
        "active_model": None,
        "requested_alias": profile["alias"],
        "requested_model": profile["alias"],
        "available_vram_gb": avail,
        "required_min_vram_gb": effective_min_vram,
        "weights_gb": weights_gb,
        "free_kv_gb": free_kv_gb,
        "inferred_gpu_tier": inferred_gpu,
        "recommended_gpu": preferred_gpu,
        "preferred_gpu": preferred_gpu,
        "accelerator_type": tier_spec["accelerator_type"],
        "memory_type": tier_spec["memory_type"],
        "selected_precision": selected_precision,
        "normalized_precision": variant_info["precision"],
        "quantized_hf_repo_id": quantized_hf_repo_id,
        "resolved_checkpoint_repo_id": resolved_checkpoint_repo_id,
        "agent_count": max(1, int(agent_count)),
        "fallback_alias": fallback_alias,
        "diagnostic": (
            f"Physical GPU VRAM {avail:.1f} GB ({inferred_gpu}) is below the {effective_min_vram:.1f} GB minimum floor "
            f"required for '{profile['alias']}' ({selected_precision}, {weights_gb:.1f} GB weights + 4.0 GB KV). "
            f"Halting weight load to prevent CUDA OOM. "
            f"Recommended GPU tier: '{preferred_gpu}', fallback: '{fallback_alias}' or '--backend=hf-api'."
        ),
    }


def validate_model_gpu_compatibility(
    model_alias: str,
    gpu_tier: str = "T4",
    custom_models: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[bool, str]:
    """
    Validates whether `model_alias` can run on `gpu_tier` across all 8 Colab GPU/TPU tiers.
    Returns `(compatible: bool, diagnostic: str)`.
    """
    eff_cfg = config if isinstance(config, dict) else ({"models": {"custom": custom_models}} if custom_models else None)
    norm_gpu = normalize_hardware_tier(gpu_tier or "T4", allow_auto=False)
    res = verify_vram_compatibility(
        model_alias=model_alias,
        available_vram_gb=norm_gpu,
        config=eff_cfg,
        active_gpu_tier=norm_gpu,
        **kwargs,
    )
    return bool(res.get("compatible")), str(res.get("diagnostic") or "")


def select_optimal_gpu_tier(
    model_alias: str,
    requested_gpu: str = "auto",
    custom_models: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Selects the optimal Colab hardware tier across all 8 GPU/TPU tiers for `model_alias`.
    Raises `ValueError` if an explicit `requested_gpu` is unrecognized or incompatible.
    """
    eff_cfg = config if isinstance(config, dict) else ({"models": {"custom": custom_models}} if custom_models else None)
    profile = resolve_model_profile(model_alias, eff_cfg)
    norm_gpu = normalize_hardware_tier(requested_gpu, allow_auto=True)
    if norm_gpu == "auto":
        return str(profile.get("recommended_gpu_tier") or profile.get("preferred_gpu") or "T4")

    ok, diag = validate_model_gpu_compatibility(model_alias, norm_gpu, custom_models=custom_models, config=eff_cfg)
    if not ok:
        raise ValueError(diag)
    return norm_gpu


def _parse_stack_slot_spec(
    item: str | Dict[str, Any],
    *,
    gpu_tier: Optional[str] = None,
    slot_index: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(item, str):
        cleaned = item.strip()
        if not cleaned:
            raise ValueError("Empty model specification in stack.")
        role = "primary" if slot_index == 0 else f"slot_{slot_index + 1}"
        resolved = resolve_model_precision_variant(cleaned, gpu_tier=gpu_tier, config=config)
        resolved["role"] = role
        return resolved
    if isinstance(item, dict):
        raw_alias = str(item.get("alias") or item.get("model") or item.get("model_alias") or "").strip()
        if not raw_alias:
            raise ValueError(f"Stack model dict missing 'alias': {item}")
        raw_prec = item.get("precision")
        role = str(item.get("role") or ("primary" if slot_index == 0 else f"slot_{slot_index + 1}"))
        resolved = resolve_model_precision_variant(
            raw_alias,
            precision=str(raw_prec) if raw_prec else None,
            gpu_tier=gpu_tier,
            config=config,
        )
        if item.get("hf_repo_id"):
            resolved["hf_repo_id"] = str(item["hf_repo_id"])
        resolved["role"] = role
        return resolved
    raise ValueError(f"Unsupported stack model specification type: {type(item).__name__}")


def _coerce_nonnegative_agent_count(raw_count: Any, context: str = "agent_count") -> int:
    if isinstance(raw_count, bool) or raw_count is None:
        raise ValueError(f"Invalid {context} '{raw_count}': must be an integer >= 0.")
    if isinstance(raw_count, int):
        val = raw_count
    elif isinstance(raw_count, str):
        stripped = raw_count.strip()
        if not stripped or not re.fullmatch(r"[+-]?\d+", stripped):
            raise ValueError(f"Invalid {context} '{raw_count}': must be an integer >= 0.")
        val = int(stripped)
    else:
        raise ValueError(f"Invalid {context} '{raw_count}' ({type(raw_count).__name__}): must be an integer >= 0.")
    if val < 0:
        raise ValueError(f"Negative {context} n_i={val} is invalid: agent_count must satisfy n_i >= 0.")
    return val


def _split_inline_model_and_agent_count(
    raw_token: str,
    *,
    default_agents: int = 1,
) -> Tuple[str, int]:
    cleaned = str(raw_token or "").strip()
    if not cleaned:
        raise ValueError("Empty model specification in multi-model agent allocation.")
    # Support '<model_spec>=<count>' or '<model_spec>:<count>' without breaking URI schemes (gs://, hf://, gdrive://)
    eq_match = re.match(r"^(.+?)=([^=]+)$", cleaned)
    if eq_match:
        lhs, rhs = eq_match.group(1).strip(), eq_match.group(2).strip()
        if not lhs:
            raise ValueError(f"Missing model alias in '{cleaned}'.")
        return lhs, _coerce_nonnegative_agent_count(rhs, context=f"agent count for '{lhs}'")

    # Check trailing ':<token>' where it is not a scheme '://'
    if ":" in cleaned:
        lhs, rhs = cleaned.rsplit(":", 1)
        lhs_s, rhs_s = lhs.strip(), rhs.strip()
        if lhs_s and not lhs_s.endswith("/") and "://" not in f"{lhs_s[-2:] if len(lhs_s) >= 2 else lhs_s}:{rhs_s}":
            # If rhs starts with '//' it was a bare scheme like 'hf://repo' with no trailing ':count'
            if not rhs_s.startswith("//"):
                return lhs_s, _coerce_nonnegative_agent_count(rhs_s, context=f"agent count for '{lhs_s}'")

    return cleaned, _coerce_nonnegative_agent_count(default_agents, context=f"default_agents for '{cleaned}'")


def parse_multi_model_agent_spec(
    model_allocations: Union[
        str,
        Mapping[str, Any],
        Sequence[Union[str, Tuple[str, Any], Mapping[str, Any]]],
    ],
    *,
    default_agents: int = 1,
    gpu_tier: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Parses `(model_spec, n_i)` allocations from CSV string, mapping, or sequence of dicts/tuples/strings.

    Enforces fail-fast validation (`n_i >= 0`, valid model profiles, non-empty specs).
    """
    if model_allocations is None:
        raise ValueError("model_allocations cannot be None.")

    raw_entries: List[Dict[str, Any]] = []
    if isinstance(model_allocations, str):
        parts = [p.strip() for p in model_allocations.split(",") if p.strip()]
        if not parts:
            raise ValueError("model_allocations string must specify at least 1 model.")
        for part in parts:
            m_spec, count = _split_inline_model_and_agent_count(part, default_agents=default_agents)
            raw_entries.append({"raw_spec": m_spec, "precision": None, "agent_count": count, "hf_repo_id": None})
    elif isinstance(model_allocations, Mapping):
        if not model_allocations:
            raise ValueError("model_allocations mapping must specify at least 1 model.")
        for key, val in model_allocations.items():
            m_spec = str(key or "").strip()
            if not m_spec:
                raise ValueError("Empty model key in model_allocations mapping.")
            if isinstance(val, Mapping):
                raw_cnt = (
                    val["agent_count"]
                    if "agent_count" in val
                    else (val["agents"] if "agents" in val else (val["count"] if "count" in val else default_agents))
                )
                count = _coerce_nonnegative_agent_count(raw_cnt, context=f"agent count for '{m_spec}'")
                raw_entries.append(
                    {
                        "raw_spec": m_spec,
                        "precision": str(val["precision"]).strip() if val.get("precision") else None,
                        "agent_count": count,
                        "hf_repo_id": str(val["hf_repo_id"]).strip() if val.get("hf_repo_id") else None,
                    }
                )
            else:
                count = _coerce_nonnegative_agent_count(val, context=f"agent count for '{m_spec}'")
                raw_entries.append({"raw_spec": m_spec, "precision": None, "agent_count": count, "hf_repo_id": None})
    elif isinstance(model_allocations, Sequence):
        items_list = list(model_allocations)
        if not items_list:
            raise ValueError("model_allocations sequence must specify at least 1 model.")
        for entry in items_list:
            if isinstance(entry, str):
                m_spec, count = _split_inline_model_and_agent_count(entry, default_agents=default_agents)
                raw_entries.append({"raw_spec": m_spec, "precision": None, "agent_count": count, "hf_repo_id": None})
            elif isinstance(entry, tuple) and len(entry) == 2:
                m_spec = str(entry[0] or "").strip()
                if not m_spec:
                    raise ValueError(f"Empty model alias in tuple entry: {entry}")
                count = _coerce_nonnegative_agent_count(entry[1], context=f"agent count for '{m_spec}'")
                raw_entries.append({"raw_spec": m_spec, "precision": None, "agent_count": count, "hf_repo_id": None})
            elif isinstance(entry, Mapping):
                m_spec = str(
                    entry.get("model")
                    or entry.get("alias")
                    or entry.get("model_alias")
                    or entry.get("model_spec")
                    or ""
                ).strip()
                if not m_spec:
                    raise ValueError(f"Allocation dict missing 'model'/'alias': {entry}")
                if ":" in m_spec and "agent_count" not in entry and "agents" not in entry and "count" not in entry:
                    m_spec, count = _split_inline_model_and_agent_count(m_spec, default_agents=default_agents)
                else:
                    raw_cnt = (
                        entry["agent_count"]
                        if "agent_count" in entry
                        else (
                            entry["agents"]
                            if "agents" in entry
                            else (entry["count"] if "count" in entry else default_agents)
                        )
                    )
                    count = _coerce_nonnegative_agent_count(raw_cnt, context=f"agent count for '{m_spec}'")
                raw_entries.append(
                    {
                        "raw_spec": m_spec,
                        "precision": str(entry["precision"]).strip() if entry.get("precision") else None,
                        "agent_count": count,
                        "hf_repo_id": str(entry["hf_repo_id"]).strip() if entry.get("hf_repo_id") else None,
                    }
                )
            else:
                raise ValueError(f"Unsupported model allocation entry type: {type(entry).__name__}")
    else:
        raise ValueError(f"Unsupported model_allocations type: {type(model_allocations).__name__}")

    parsed_list: List[Dict[str, Any]] = []
    for item in raw_entries:
        raw_spec = item["raw_spec"]
        inline_prec: Optional[str] = None
        base_alias_raw = raw_spec
        if "@" in raw_spec and not Path(raw_spec).expanduser().exists():
            base_alias_raw, inline_prec = raw_spec.split("@", 1)
            base_alias_raw = base_alias_raw.strip()
            inline_prec = inline_prec.strip() or None
        if not base_alias_raw:
            raise ValueError(f"Missing model alias in specification '{raw_spec}'.")
        explicit_prec = item["precision"] or inline_prec
        resolved = resolve_model_precision_variant(
            base_alias_raw,
            precision=explicit_prec,
            gpu_tier=gpu_tier if (gpu_tier and str(gpu_tier).strip().lower() != "auto") else None,
            config=config,
        )
        if item.get("hf_repo_id"):
            resolved["hf_repo_id"] = str(item["hf_repo_id"])
        parsed_list.append(
            {
                "raw_spec": raw_spec,
                "alias": resolved["alias"],
                "explicit_precision": normalize_precision_code(explicit_prec) if explicit_prec else None,
                "precision": resolved["precision"],
                "model_spec": f"{resolved['alias']}@{resolved['precision']}",
                "hf_repo_id": resolved["hf_repo_id"],
                "weights_gb": float(resolved["weights_gb"]),
                "min_vram_gb": float(resolved["min_vram_gb"]),
                "quant_method": resolved["quant_method"],
                "template_family": resolved.get("template_family", "qwen3_chatml"),
                "source": resolved.get("source", "hf_hub"),
                "storage_uri": resolved.get("storage_uri"),
                "agent_count": int(item["agent_count"]),
            }
        )
    return parsed_list


def _lookup_agent_count_for_slot(
    slot: Mapping[str, Any],
    raw_entry: Any,
    agent_counts: Optional[Mapping[str, Any]],
) -> int:
    if isinstance(raw_entry, Mapping):
        for key_name in ("agent_count", "agents", "count"):
            if key_name in raw_entry and raw_entry[key_name] is not None:
                return _coerce_nonnegative_agent_count(raw_entry[key_name], context=f"agent count for '{slot.get('alias')}'")
    if isinstance(raw_entry, str) and ":" in raw_entry:
        try:
            _, inline_cnt = _split_inline_model_and_agent_count(raw_entry, default_agents=1)
            return inline_cnt
        except ValueError:
            raise

    if agent_counts is not None:
        alias = str(slot.get("alias") or "")
        prec = str(slot.get("precision") or "")
        spec_key = f"{alias}@{prec}"
        repo_id = str(slot.get("hf_repo_id") or "")
        raw_str = str(raw_entry).strip() if isinstance(raw_entry, str) else ""
        # Case-exact first, then case-insensitive fallback
        for candidate in (spec_key, alias, raw_str, repo_id):
            if candidate and candidate in agent_counts:
                return _coerce_nonnegative_agent_count(agent_counts[candidate], context=f"agent count for '{candidate}'")
        lower_map = {str(k).strip().lower(): v for k, v in agent_counts.items()}
        for candidate in (spec_key, alias, raw_str, repo_id):
            if candidate and candidate.lower() in lower_map:
                return _coerce_nonnegative_agent_count(
                    lower_map[candidate.lower()],
                    context=f"agent count for '{candidate}'",
                )
    return 1


def verify_stack_vram_compatibility(
    models: Sequence[str | Dict[str, Any]],
    gpu_tier_or_vram: str | float,
    *,
    min_kv_headroom_gb: float = MIN_KV_CACHE_HEADROOM_GB,
    agent_counts: Optional[Mapping[str, int]] = None,
    custom_models: Optional[Mapping[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verifies whether co-resident models fit simultaneously on `gpu_tier_or_vram`
    while preserving at least `min_kv_headroom_gb` (default 4.0 GB) of free memory for KV caches.

    When `agent_counts` (or per-entry `agent_count` / `agents`) is supplied (Spec 867):
      - Models with `n_i == 0` are partitioned into `inactive_models` (`loaded=False`, `vram_allocated_gb=0.0`,
        `vram_bytes=0`) and excluded from active weight summation and `1 <= x_active <= 3` cardinality limits.
      - Active models (`n_i > 0`) are deduplicated so each active model's weights are counted once,
        and KV headroom scales with total active agents `max(min_kv_headroom_gb, 1.50 + 0.35 * total_agents)`.
    """
    eff_cfg = (
        config
        if isinstance(config, dict)
        else ({"models": {"custom": dict(custom_models)}} if custom_models else None)
    )
    items = list(models or [])
    has_agent_counts = agent_counts is not None or any(
        (isinstance(it, Mapping) and ("agent_count" in it or "agents" in it or "count" in it))
        or (isinstance(it, str) and ":" in it and not it.strip().startswith(("gs://", "hf://", "gdrive://", "http://", "https://")))
        for it in items
    )

    if not has_agent_counts:
        if not items or len(items) > 3:
            return {
                "compatible": False,
                "code": "invalid_stack_cardinality",
                "model_count": len(items),
                "gpu_tier": str(gpu_tier_or_vram),
                "available_memory_gb": 0.0,
                "nominal_memory_gb": 0.0,
                "estimated_weights_gb": 0.0,
                "free_kv_gb": 0.0,
                "nominal_free_kv_gb": 0.0,
                "min_kv_headroom_gb": float(min_kv_headroom_gb),
                "models": [],
                "inactive_models": [],
                "diagnostic": f"Stack model count x={len(items)} is outside the supported co-resident range 1 <= x <= 3.",
            }

    if isinstance(gpu_tier_or_vram, str):
        canonical_tier = normalize_hardware_tier(gpu_tier_or_vram, allow_auto=False)
        tier_spec = GPU_TIER_SPECS[canonical_tier]
        avail_mem = float(tier_spec["vram_gb"])
        nominal_mem = float(tier_spec["nominal_vram_gb"])
        memory_type = str(tier_spec["memory_type"])
        accelerator_type = str(tier_spec["accelerator_type"])
    else:
        avail_mem = float(gpu_tier_or_vram)
        canonical_tier = _infer_hardware_tier_from_memory(avail_mem)
        tier_spec = GPU_TIER_SPECS[canonical_tier]
        nominal_mem = avail_mem
        memory_type = str(tier_spec["memory_type"])
        accelerator_type = str(tier_spec["accelerator_type"])

    if not has_agent_counts:
        resolved_slots: List[Dict[str, Any]] = [
            _parse_stack_slot_spec(entry, gpu_tier=canonical_tier, slot_index=idx, config=eff_cfg)
            for idx, entry in enumerate(items)
        ]
        total_weights_gb = round(sum(float(slot["weights_gb"]) for slot in resolved_slots), 2)
        free_kv_gb = round(avail_mem - total_weights_gb, 2)
        nominal_free_kv_gb = round(nominal_mem - total_weights_gb, 2)
        floor_gb = float(min_kv_headroom_gb)
        compatible = free_kv_gb >= (floor_gb - 1e-6)

        slot_summary = " + ".join(
            f"{s['alias']}@{s['precision']} ({s['weights_gb']:.1f} GB)" for s in resolved_slots
        )
        if compatible:
            diag = (
                f"Stack memory check passed on {canonical_tier} ({avail_mem:.1f} GB {memory_type}): "
                f"x={len(resolved_slots)} [{slot_summary}] = {total_weights_gb:.1f} GB weights, "
                f"leaving {free_kv_gb:.1f} GB free {memory_type} (>= {floor_gb:.1f} GB KV floor)."
            )
        else:
            diag = (
                f"Stack memory check failed on {canonical_tier} ({avail_mem:.1f} GB {memory_type}): "
                f"x={len(resolved_slots)} [{slot_summary}] requires {total_weights_gb:.1f} GB weights + "
                f"{floor_gb:.1f} GB KV floor = {total_weights_gb + floor_gb:.1f} GB, leaving only "
                f"{free_kv_gb:.1f} GB free (< {floor_gb:.1f} GB minimum floor required)."
            )

        return {
            "compatible": compatible,
            "code": "ok" if compatible else "insufficient_vram",
            "model_count": len(resolved_slots),
            "gpu_tier": canonical_tier,
            "accelerator_type": accelerator_type,
            "memory_type": memory_type,
            "available_memory_gb": avail_mem,
            "nominal_memory_gb": nominal_mem,
            "estimated_weights_gb": total_weights_gb,
            "free_kv_gb": free_kv_gb,
            "nominal_free_kv_gb": nominal_free_kv_gb,
            "min_kv_headroom_gb": floor_gb,
            "models": resolved_slots,
            "inactive_models": [],
            "diagnostic": diag,
        }

    if not items:
        return {
            "compatible": True,
            "code": "ok",
            "model_count": 0,
            "gpu_tier": canonical_tier,
            "accelerator_type": accelerator_type,
            "memory_type": memory_type,
            "available_memory_gb": avail_mem,
            "nominal_memory_gb": nominal_mem,
            "estimated_weights_gb": 0.0,
            "free_kv_gb": avail_mem,
            "nominal_free_kv_gb": nominal_mem,
            "min_kv_headroom_gb": 0.0,
            "models": [],
            "inactive_models": [],
            "agent_counts": {},
            "total_agents": 0,
            "diagnostic": f"0 models configured on {canonical_tier}: 0.0 GB weights allocated.",
        }

    dedup_by_alias: Dict[str, Dict[str, Any]] = {}
    ordered_aliases: List[str] = []
    for idx, raw_entry in enumerate(items):
        cleaned_entry: str | Dict[str, Any] = raw_entry
        if isinstance(raw_entry, str) and ":" in raw_entry:
            m_part, _ = _split_inline_model_and_agent_count(raw_entry, default_agents=1)
            cleaned_entry = m_part
        slot = _parse_stack_slot_spec(cleaned_entry, gpu_tier=canonical_tier, slot_index=idx, config=eff_cfg)
        cnt = _lookup_agent_count_for_slot(slot, raw_entry, agent_counts)
        alias = str(slot["alias"])
        if alias in dedup_by_alias:
            existing = dedup_by_alias[alias]
            if existing["precision"] != slot["precision"]:
                raise ValueError(
                    f"Conflicting precisions for model '{alias}' on GPU 1: "
                    f"'{existing['precision']}' vs '{slot['precision']}'."
                )
            existing["agent_count"] = int(existing["agent_count"]) + int(cnt)
        else:
            slot_copy = dict(slot)
            slot_copy["agent_count"] = int(cnt)
            slot_copy["catalog_weights_gb"] = float(slot["weights_gb"])
            dedup_by_alias[alias] = slot_copy
            ordered_aliases.append(alias)

    active_slots: List[Dict[str, Any]] = []
    inactive_slots: List[Dict[str, Any]] = []
    resolved_counts: Dict[str, int] = {}
    for alias in ordered_aliases:
        entry = dedup_by_alias[alias]
        cnt = int(entry["agent_count"])
        resolved_counts[alias] = cnt
        if cnt > 0:
            role = "primary" if len(active_slots) == 0 else f"slot_{len(active_slots) + 1}"
            entry["role"] = role
            entry["active"] = True
            entry["loaded"] = True
            entry["vram_allocated_gb"] = float(entry["catalog_weights_gb"])
            entry["vram_bytes"] = int(round(float(entry["catalog_weights_gb"]) * (1024 ** 3)))
            active_slots.append(entry)
        else:
            entry["role"] = "inactive"
            entry["active"] = False
            entry["loaded"] = False
            entry["vram_allocated_gb"] = 0.0
            entry["vram_bytes"] = 0
            inactive_slots.append(entry)

    if len(active_slots) > 3:
        return {
            "compatible": False,
            "code": "invalid_stack_cardinality",
            "model_count": len(active_slots),
            "configured_model_count": len(ordered_aliases),
            "gpu_tier": canonical_tier,
            "accelerator_type": accelerator_type,
            "memory_type": memory_type,
            "available_memory_gb": avail_mem,
            "nominal_memory_gb": nominal_mem,
            "estimated_weights_gb": 0.0,
            "free_kv_gb": 0.0,
            "nominal_free_kv_gb": 0.0,
            "min_kv_headroom_gb": float(min_kv_headroom_gb),
            "models": [],
            "inactive_models": inactive_slots,
            "agent_counts": resolved_counts,
            "total_agents": sum(resolved_counts.values()),
            "diagnostic": (
                f"Active model count x_active={len(active_slots)} exceeds the supported co-resident "
                f"ceiling 0 <= x_active <= 3."
            ),
        }

    total_agents = sum(int(s["agent_count"]) for s in active_slots)
    if len(active_slots) == 0:
        return {
            "compatible": True,
            "code": "ok",
            "model_count": 0,
            "configured_model_count": len(ordered_aliases),
            "gpu_tier": canonical_tier,
            "accelerator_type": accelerator_type,
            "memory_type": memory_type,
            "available_memory_gb": avail_mem,
            "nominal_memory_gb": nominal_mem,
            "estimated_weights_gb": 0.0,
            "free_kv_gb": avail_mem,
            "nominal_free_kv_gb": nominal_mem,
            "min_kv_headroom_gb": 0.0,
            "models": [],
            "inactive_models": inactive_slots,
            "agent_counts": resolved_counts,
            "total_agents": 0,
            "diagnostic": (
                f"All {len(inactive_slots)} configured model(s) have n_i=0 active agents; "
                f"0 models loaded into {canonical_tier} VRAM (0.0 GB used, {avail_mem:.1f} GB free)."
            ),
        }

    total_weights_gb = round(sum(float(slot["weights_gb"]) for slot in active_slots), 2)
    free_kv_gb = round(avail_mem - total_weights_gb, 2)
    nominal_free_kv_gb = round(nominal_mem - total_weights_gb, 2)
    floor_gb = round(max(float(min_kv_headroom_gb), 1.50 + 0.35 * total_agents), 2)
    compatible = free_kv_gb >= (floor_gb - 1e-6)

    slot_summary = " + ".join(
        f"{s['alias']}@{s['precision']} (n={s['agent_count']}, {s['weights_gb']:.1f} GB)" for s in active_slots
    )
    inactive_note = (
        f"; skipped inactive (0 GB): {[s['alias'] for s in inactive_slots]}" if inactive_slots else ""
    )
    if compatible:
        diag = (
            f"Active stack memory check passed on {canonical_tier} ({avail_mem:.1f} GB {memory_type}): "
            f"x_active={len(active_slots)} [{slot_summary}] = {total_weights_gb:.1f} GB weights, "
            f"leaving {free_kv_gb:.1f} GB free {memory_type} (>= {floor_gb:.1f} GB KV floor for {total_agents} agents)"
            f"{inactive_note}."
        )
    else:
        diag = (
            f"Active stack memory check failed on {canonical_tier} ({avail_mem:.1f} GB {memory_type}): "
            f"x_active={len(active_slots)} [{slot_summary}] requires {total_weights_gb:.1f} GB weights + "
            f"{floor_gb:.1f} GB KV floor = {total_weights_gb + floor_gb:.1f} GB, leaving only "
            f"{free_kv_gb:.1f} GB free (< {floor_gb:.1f} GB required){inactive_note}."
        )

    return {
        "compatible": compatible,
        "code": "ok" if compatible else "insufficient_vram",
        "model_count": len(active_slots),
        "configured_model_count": len(ordered_aliases),
        "gpu_tier": canonical_tier,
        "accelerator_type": accelerator_type,
        "memory_type": memory_type,
        "available_memory_gb": avail_mem,
        "nominal_memory_gb": nominal_mem,
        "estimated_weights_gb": total_weights_gb,
        "free_kv_gb": free_kv_gb,
        "nominal_free_kv_gb": nominal_free_kv_gb,
        "min_kv_headroom_gb": floor_gb,
        "models": active_slots,
        "inactive_models": inactive_slots,
        "agent_counts": resolved_counts,
        "total_agents": total_agents,
        "diagnostic": diag,
    }


def resolve_multi_model_agent_allocation(
    model_allocations: Union[
        str,
        Mapping[str, Any],
        Sequence[Union[str, Tuple[str, Any], Mapping[str, Any]]],
    ],
    gpu_tier: str = "auto",
    *,
    hardware_tier: Optional[str] = None,
    custom_models: Optional[Mapping[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    min_kv_headroom_gb: float = MIN_KV_CACHE_HEADROOM_GB,
    per_agent_kv_gb: float = 0.35,
    base_runtime_overhead_gb: float = 1.50,
    prefer_cuda_gpu: bool = True,
) -> Dict[str, Any]:
    """Single Source of Truth (Spec 867) resolver for `n` parallel agents across `x` models on GPU 1.

    - Partitions configured models into `active_models` (`n_i > 0`, deduplicated so each active model's
      weights are loaded into GPU 1's VRAM **once**) and `inactive_models` (`n_i == 0`, unloaded,
      `0.0 GB` VRAM, `0` bytes, `0.0` compute share).
    - Sizes GPU 1 from the 8-tier Spec 865 hardware catalog (`GPU_TIER_SPECS`) when `gpu_tier == 'auto'`
      or validates explicit `gpu_tier`, failing fast with `ValueError` on VRAM overflow or `x_active > 3`.
    """
    eff_cfg = (
        config
        if isinstance(config, dict)
        else ({"models": {"custom": dict(custom_models)}} if custom_models else None)
    )
    eff_tier_arg = hardware_tier if (hardware_tier is not None and str(hardware_tier).strip()) else gpu_tier
    norm_tier_req = normalize_hardware_tier(eff_tier_arg or "auto", allow_auto=True)
    initial_parsed = parse_multi_model_agent_spec(
        model_allocations,
        default_agents=1,
        gpu_tier=None if norm_tier_req == "auto" else norm_tier_req,
        config=eff_cfg,
    )

    def _evaluate_on_tier(cand_tier: str, *, force_q4_auto: bool = False) -> Dict[str, Any]:
        tier_spec = GPU_TIER_SPECS[cand_tier]
        avail_mem = float(tier_spec["vram_gb"])
        nominal_mem = float(tier_spec["nominal_vram_gb"])
        cand_rank = int(tier_spec["rank"])
        dedup: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        for item in initial_parsed:
            resolved = resolve_model_precision_variant(
                item["alias"],
                precision=item["explicit_precision"],
                gpu_tier=cand_tier,
                config=eff_cfg,
            )
            # When auto-sizing across tiers without explicit precision, keep compact Q4 if default BF16/Q8 would overflow
            if norm_tier_req == "auto" and item["explicit_precision"] is None and int(item["agent_count"]) > 0:
                if force_q4_auto or (float(resolved["weights_gb"]) + float(min_kv_headroom_gb) > avail_mem):
                    q4_fallback = resolve_model_precision_variant(
                        item["alias"],
                        precision="Q4",
                        gpu_tier=cand_tier,
                        config=eff_cfg,
                    )
                    if float(q4_fallback["weights_gb"]) < float(resolved["weights_gb"]):
                        resolved = q4_fallback
            if item.get("hf_repo_id") and item["explicit_precision"] is not None:
                resolved["hf_repo_id"] = item["hf_repo_id"]

            alias = str(resolved["alias"])
            prec = str(resolved["precision"])
            cnt = int(item["agent_count"])
            prof = resolve_model_profile(alias, config=eff_cfg)
            if alias in dedup:
                prev = dedup[alias]
                if prev["precision"] != prec:
                    raise ValueError(
                        f"Conflicting precisions for model '{alias}' on GPU 1: "
                        f"'{prev['precision']}' vs '{prec}'."
                    )
                prev["agent_count"] = int(prev["agent_count"]) + cnt
                if item["explicit_precision"] is None:
                    prev["_has_auto_precision"] = True
            else:
                dedup[alias] = {
                    "model_spec": f"{alias}@{prec}",
                    "alias": alias,
                    "precision": prec,
                    "hf_repo_id": str(resolved["hf_repo_id"]),
                    "quant_method": str(resolved["quant_method"]),
                    "template_family": str(resolved.get("template_family", "qwen3_chatml")),
                    "source": str(resolved.get("source", "hf_hub")),
                    "storage_uri": resolved.get("storage_uri"),
                    "agent_count": cnt,
                    "catalog_weights_gb": float(resolved["weights_gb"]),
                    "min_vram_gb": max(
                        float(prof.get("min_vram_gb", 0.0) or 0.0),
                        float(resolved.get("min_vram_gb", 0.0) or 0.0),
                    ),
                    "min_gpu_tier": prof.get("min_gpu_tier"),
                    "preferred_gpu": prof.get("preferred_gpu"),
                    "supported_hardware_tiers": prof.get("supported_hardware_tiers") or prof.get("supported_gpus"),
                    "modality": str(prof.get("modality", "text") or "text"),
                    "_has_auto_precision": item["explicit_precision"] is None,
                }
                order.append(alias)

        total_agents = sum(int(dedup[a]["agent_count"]) for a in order if int(dedup[a]["agent_count"]) > 0)
        active_list: List[Dict[str, Any]] = []
        inactive_list: List[Dict[str, Any]] = []
        all_list: List[Dict[str, Any]] = []
        for alias in order:
            base = dedup[alias]
            cnt = int(base["agent_count"])
            if cnt > 0:
                active_list.append(base)
            else:
                inactive_list.append(base)
            all_list.append(base)

        if len(active_list) > 3:
            raise ValueError(
                f"invalid_stack_cardinality: Active model count x_active={len(active_list)} exceeds "
                f"supported single-GPU co-resident ceiling 0 <= x_active <= 3."
            )

        tier_violations: List[str] = []
        if len(active_list) == 0:
            req_kv = 0.0
            total_w = 0.0
            free_kv = avail_mem
            safe_batch_cap = int(tier_spec.get("max_num_seqs", 4))
            compatible = True
        else:
            total_w = round(sum(float(m["catalog_weights_gb"]) for m in active_list), 2)
            req_kv = round(
                max(
                    float(min_kv_headroom_gb),
                    float(base_runtime_overhead_gb) + float(per_agent_kv_gb) * total_agents,
                ),
                2,
            )
            free_kv = round(avail_mem - total_w, 2)
            if (
                norm_tier_req == "auto"
                and not force_q4_auto
                and free_kv < (req_kv - 1e-6)
                and any(bool(m.get("_has_auto_precision")) for m in active_list)
            ):
                return _evaluate_on_tier(cand_tier, force_q4_auto=True)

            for m in active_list:
                sup_tiers = m.get("supported_hardware_tiers")
                in_explicit_supported = False
                if isinstance(sup_tiers, (list, tuple, set)) and sup_tiers:
                    norm_sup = {str(t).strip() for t in sup_tiers if str(t).strip()}
                    if cand_tier not in norm_sup:
                        tier_violations.append(
                            f"{m['alias']} does not support tier {cand_tier} (supported: {sorted(norm_sup)})"
                        )
                    else:
                        in_explicit_supported = True
                if not in_explicit_supported:
                    min_gpu = str(m.get("min_gpu_tier") or "").strip()
                    if min_gpu in GPU_TIER_SPECS and cand_rank < int(GPU_TIER_SPECS[min_gpu]["rank"]):
                        tier_violations.append(
                            f"{m['alias']} requires minimum GPU tier {min_gpu} (above {cand_tier})"
                        )
                req_min_vram = float(m.get("min_vram_gb", 0.0) or 0.0)
                if avail_mem < req_min_vram - 1e-6:
                    tier_violations.append(
                        f"{m['alias']} requires >= {req_min_vram:.1f} GB VRAM ({cand_tier} has {avail_mem:.1f} GB)"
                    )

            vram_fits = free_kv >= (req_kv - 1e-6)
            compatible = vram_fits and (len(tier_violations) == 0)
            kv_seq_cap = max(
                1,
                int(max(0.5, free_kv - float(base_runtime_overhead_gb)) // max(0.1, float(per_agent_kv_gb))),
            )
            max_cap = 256 if str(tier_spec.get("accelerator_type")) == "GPU" else 32
            safe_batch_cap = min(max_cap, max(int(tier_spec.get("max_num_seqs", 4)), kv_seq_cap))

        enriched_active: List[Dict[str, Any]] = []
        enriched_inactive: List[Dict[str, Any]] = []
        enriched_all: List[Dict[str, Any]] = []
        active_idx = 0
        for base in all_list:
            cnt = int(base["agent_count"])
            entry_out = {k: v for k, v in base.items() if not k.startswith("_")}
            if cnt > 0:
                active_idx += 1
                w_gb = float(base["catalog_weights_gb"])
                slot_max_batch = min(safe_batch_cap, max(int(tier_spec.get("max_num_seqs", 4)), cnt))
                entry_out.update(
                    {
                        "spec": base["model_spec"],
                        "active": True,
                        "loaded": True,
                        "weights_gb": w_gb,
                        "vram_gb": w_gb,
                        "vram_allocated_gb": w_gb,
                        "vram_bytes": int(round(w_gb * (1024 ** 3))),
                        "compute_share": round(cnt / total_agents, 4) if total_agents > 0 else 0.0,
                        "max_batch_size": slot_max_batch,
                        "target_cohort_size": min(cnt, slot_max_batch),
                        "role": "primary" if active_idx == 1 else f"slot_{active_idx}",
                    }
                )
                enriched_active.append(entry_out)
            else:
                entry_out.update(
                    {
                        "spec": base["model_spec"],
                        "active": False,
                        "loaded": False,
                        "weights_gb": 0.0,
                        "vram_gb": 0.0,
                        "vram_allocated_gb": 0.0,
                        "vram_bytes": 0,
                        "compute_share": 0.0,
                        "max_batch_size": 0,
                        "target_cohort_size": 0,
                        "role": "inactive",
                    }
                )
                enriched_inactive.append(entry_out)
            enriched_all.append(entry_out)

        max_cohort = max((int(m["agent_count"]) for m in enriched_active), default=0)
        cohort_fits_tier_seqs = int(tier_spec.get("max_num_seqs", 4)) >= max_cohort or safe_batch_cap >= max_cohort

        active_desc = (
            " + ".join(
                f"{m['alias']}@{m['precision']} (n={m['agent_count']}, {m['vram_allocated_gb']:.1f} GB)"
                for m in enriched_active
            )
            if enriched_active
            else "none (0 active models)"
        )
        inactive_aliases = [m["alias"] for m in enriched_inactive]
        if compatible:
            diag = (
                f"GPU 1 allocation sized on {cand_tier} ({avail_mem:.1f} GB {tier_spec['memory_type']}): "
                f"active={len(enriched_active)}/{len(enriched_all)} [{active_desc}] = {total_w:.1f} GB weights + "
                f"{req_kv:.1f} GB KV budget ({free_kv:.1f} GB free) for {total_agents} parallel agents; "
                f"unloaded n_i=0 models (0 GB VRAM): {inactive_aliases}."
            )
        else:
            if not vram_fits:
                deficit = round((total_w + req_kv) - avail_mem, 2)
                cap_msg = (
                    f"Insufficient GPU 1 memory on {cand_tier} ({avail_mem:.1f} GB usable {tier_spec['memory_type']}): "
                    f"active models [{active_desc}] require {total_w:.1f} GB weights + {req_kv:.1f} GB KV headroom "
                    f"= {total_w + req_kv:.1f} GB (deficit {deficit:.1f} GB)."
                )
            else:
                cap_msg = ""
            diag = " ".join(part for part in ("; ".join(tier_violations) + "." if tier_violations else "", cap_msg) if part)

        return {
            "compatible": compatible,
            "cohort_fits_tier_seqs": cohort_fits_tier_seqs,
            "gpu_tier": cand_tier,
            "hardware_tier": cand_tier,
            "accelerator_type": str(tier_spec["accelerator_type"]),
            "memory_type": str(tier_spec["memory_type"]),
            "available_memory_gb": avail_mem,
            "vram_gb": avail_mem,
            "nominal_memory_gb": nominal_mem,
            "configured_model_count": len(enriched_all),
            "active_model_count": len(enriched_active),
            "inactive_model_count": len(enriched_inactive),
            "stack_mode": (
                "inactive"
                if len(enriched_active) == 0
                else ("solo" if len(enriched_active) == 1 else "stack")
            ),
            "total_agents": total_agents,
            "estimated_weights_gb": total_w,
            "total_weights_gb": total_w,
            "required_kv_headroom_gb": req_kv,
            "estimated_peak_vram_gb": round(total_w + (req_kv if len(enriched_active) > 0 else 0.0), 2),
            "free_kv_gb": free_kv,
            "nominal_free_kv_gb": round(nominal_mem - total_w, 2),
            "agent_counts": {m["alias"]: int(m["agent_count"]) for m in enriched_all},
            "agent_counts_by_spec": {m["model_spec"]: int(m["agent_count"]) for m in enriched_all},
            "loaded_models": [m["alias"] for m in enriched_active],
            "inactive_unloaded_models": inactive_aliases,
            "vram_allocated_by_model": {m["alias"]: float(m["vram_allocated_gb"]) for m in enriched_all},
            "active_models": enriched_active,
            "inactive_models": enriched_inactive,
            "all_models": enriched_all,
            "safe_parallel_batch_cap": safe_batch_cap,
            "diagnostic": diag,
        }

    if norm_tier_req != "auto":
        res = _evaluate_on_tier(norm_tier_req)
        if not res["compatible"]:
            raise ValueError(res["diagnostic"])
        res.pop("cohort_fits_tier_seqs", None)
        return res

    # Auto-select smallest cost-ranked hardware tier from Spec 865 GPU_TIER_SPECS
    ordered_tiers = sorted(
        ALL_HARDWARE_TIERS,
        key=lambda t: (int(GPU_TIER_SPECS[t]["rank"]), float(GPU_TIER_SPECS[t]["vram_gb"])),
    )
    candidate_tiers = [
        t for t in ordered_tiers if (not prefer_cuda_gpu or GPU_TIER_SPECS[t]["accelerator_type"] == "GPU")
    ]
    if not candidate_tiers:
        candidate_tiers = ordered_tiers

    compatible_evals: List[Dict[str, Any]] = []
    last_eval: Optional[Dict[str, Any]] = None
    for cand in candidate_tiers:
        ev = _evaluate_on_tier(cand)
        last_eval = ev
        if ev["compatible"]:
            compatible_evals.append(ev)

    if not compatible_evals:
        err_msg = last_eval["diagnostic"] if last_eval else "No compatible hardware tier found."
        raise ValueError(
            f"Multi-model agent allocation exceeds maximum single-accelerator capacity across all Spec 865 tiers: {err_msg}"
        )

    # Enforce preferred_gpu floor for active models without explicit precision override (e.g. r1-14b -> L4, qwen-27b/qwen-coder-30b-moe -> A100, qwen-122b-moe -> H100)
    auto_pref_ranks: List[int] = []
    active_aliases_set = {str(m["alias"]) for m in compatible_evals[0]["active_models"]}
    for item in initial_parsed:
        if item["explicit_precision"] is None and item["alias"] in active_aliases_set:
            p_tier = str(resolve_model_profile(item["alias"], config=eff_cfg).get("preferred_gpu") or "").strip()
            if p_tier in GPU_TIER_SPECS:
                auto_pref_ranks.append(int(GPU_TIER_SPECS[p_tier]["rank"]))
    pref_floor_rank = max(auto_pref_ranks, default=0)
    pref_evals = [
        ev for ev in compatible_evals if int(GPU_TIER_SPECS[ev["gpu_tier"]]["rank"]) >= pref_floor_rank
    ]
    pool_evals = pref_evals if pref_evals else compatible_evals

    # Prefer smallest compatible tier whose tier max_num_seqs >= max single-model agent count when possible
    preferred_matches = [
        ev
        for ev in pool_evals
        if int(GPU_TIER_SPECS[ev["gpu_tier"]].get("max_num_seqs", 4))
        >= max((int(m["agent_count"]) for m in ev["active_models"]), default=0)
    ]
    chosen = preferred_matches[0] if preferred_matches else pool_evals[-1]
    chosen.pop("cohort_fits_tier_seqs", None)
    return chosen


def get_stack_presets_for_tier(
    gpu_tier: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Returns enriched Solo (`x=1`) options and Multi-Model Stack (`x=2..3`) presets for `gpu_tier`."""
    canonical_tier = normalize_hardware_tier(gpu_tier, allow_auto=False)
    tier_spec = copy.deepcopy(GPU_TIER_SPECS[canonical_tier])
    raw_tier_presets = HARDWARE_STACK_PRESETS[canonical_tier]

    solo_enriched: List[Dict[str, Any]] = []
    for opt in raw_tier_presets["solo_options"]:
        check = verify_stack_vram_compatibility([opt], canonical_tier, config=config)
        slot = check["models"][0]
        solo_enriched.append(
            {
                "alias": slot["alias"],
                "precision": slot["precision"],
                "hf_repo_id": slot["hf_repo_id"],
                "weights_gb": slot["weights_gb"],
                "free_kv_gb": check["free_kv_gb"],
                "nominal_free_kv_gb": check["nominal_free_kv_gb"],
                "compatible": check["compatible"],
            }
        )

    presets_enriched: Dict[str, Dict[str, Any]] = {}
    for preset_id, p_meta in raw_tier_presets["presets"].items():
        check = verify_stack_vram_compatibility(p_meta["models"], canonical_tier, config=config)
        presets_enriched[preset_id] = {
            "preset_id": preset_id,
            "label": p_meta["label"],
            "description": p_meta["description"],
            "model_count": check["model_count"],
            "estimated_weights_gb": check["estimated_weights_gb"],
            "total_weights_gb": check["estimated_weights_gb"],
            "free_kv_gb": check["free_kv_gb"],
            "nominal_free_kv_gb": check["nominal_free_kv_gb"],
            "compatible": check["compatible"],
            "models": check["models"],
        }

    return {
        "tier": canonical_tier,
        "hardware": tier_spec,
        "default_solo": copy.deepcopy(raw_tier_presets["default_solo"]),
        "default_stack": raw_tier_presets["default_stack"],
        "solo_options": solo_enriched,
        "presets": presets_enriched,
    }


def resolve_hardware_stack(
    gpu_tier: str,
    *,
    mode: Optional[str] = None,
    model_alias: Optional[str] = None,
    precision: Optional[str] = None,
    stack_preset: Optional[str] = None,
    preset: Optional[str] = None,
    preset_id: Optional[str] = None,
    custom_models: Optional[Sequence[str | Dict[str, Any]]] = None,
    stack_models: Optional[Sequence[str | Dict[str, Any]]] = None,
    models: Optional[Sequence[str | Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolves either a Solo (`x = 1`) model selection or a Co-Resident Multi-Model Stack (`x = 2..3`)
    on `gpu_tier`.

    Fails fast with `ValueError` if `x < 1`, `x > 3`, an unknown preset is requested, or
    `free_kv_gb < 4.0`.
    """
    if custom_models is None:
        if stack_models is not None:
            custom_models = stack_models
        elif models is not None:
            custom_models = models
    if stack_preset is None:
        stack_preset = preset if preset is not None else preset_id
    canonical_tier = normalize_hardware_tier(gpu_tier, allow_auto=False)
    tier_presets = HARDWARE_STACK_PRESETS[canonical_tier]

    eff_mode = (mode or "").strip().lower() or None
    eff_preset = (stack_preset or "").strip() or None

    raw_slots: List[str | Dict[str, Any]] = []
    resolved_preset_id: Optional[str] = None

    if custom_models is not None:
        raw_slots = list(custom_models)
        eff_mode = "solo" if len(raw_slots) == 1 else "stack"
    elif eff_preset and eff_preset.lower() != "solo":
        if "," in eff_preset:
            raw_slots = [part.strip() for part in eff_preset.split(",") if part.strip()]
            eff_mode = "solo" if len(raw_slots) == 1 else "stack"
        else:
            preset_key = eff_preset.lower()
            presets_map = tier_presets["presets"]
            if preset_key == "default":
                preset_key = str(tier_presets["default_stack"])
            if preset_key in presets_map:
                resolved_preset_id = preset_key
                raw_slots = copy.deepcopy(presets_map[preset_key]["models"])
                eff_mode = "stack"
            elif "@" in eff_preset or eff_preset in BUILTIN_MODEL_CATALOG:
                raw_slots = [eff_preset]
                eff_mode = "solo"
            else:
                valid_presets = ["solo", *presets_map.keys()]
                raise ValueError(
                    f"Unknown stack preset '{eff_preset}' for hardware tier '{canonical_tier}'. "
                    f"Valid presets on {canonical_tier}: {valid_presets}"
                )
    elif eff_mode == "stack":
        resolved_preset_id = str(tier_presets["default_stack"])
        raw_slots = copy.deepcopy(tier_presets["presets"][resolved_preset_id]["models"])
    else:
        eff_mode = "solo"
        resolved_preset_id = "solo"
        if model_alias:
            if precision:
                raw_slots = [{"alias": model_alias, "precision": precision, "role": "primary"}]
            else:
                # Prefer matching tier solo_options precision if configured
                matched_solo = next(
                    (
                        opt
                        for opt in tier_presets["solo_options"]
                        if opt["alias"].lower() == str(model_alias).strip().lower()
                    ),
                    None,
                )
                if matched_solo and "@" not in str(model_alias):
                    raw_slots = [{"alias": matched_solo["alias"], "precision": matched_solo["precision"], "role": "primary"}]
                else:
                    raw_slots = [model_alias]
        else:
            default_solo = tier_presets["default_solo"]
            raw_slots = [
                {
                    "alias": default_solo["alias"],
                    "precision": precision or default_solo["precision"],
                    "role": "primary",
                }
            ]

    if not (1 <= len(raw_slots) <= 3):
        raise ValueError(
            f"Invalid model stack cardinality x={len(raw_slots)}: must satisfy 1 <= x <= 3."
        )

    check = verify_stack_vram_compatibility(raw_slots, canonical_tier, config=config)
    if not check["compatible"]:
        raise ValueError(check["diagnostic"])

    tier_spec = GPU_TIER_SPECS[canonical_tier]
    primary_slot = check["models"][0]
    return {
        "gpu_tier": canonical_tier,
        "accelerator_type": tier_spec["accelerator_type"],
        "memory_type": tier_spec["memory_type"],
        "stack_mode": "solo" if check["model_count"] == 1 else "stack",
        "stack_preset": resolved_preset_id,
        "model_count": check["model_count"],
        "primary_model": primary_slot["alias"],
        "primary_precision": primary_slot["precision"],
        "primary_hf_repo_id": primary_slot["hf_repo_id"],
        "available_memory_gb": check["available_memory_gb"],
        "nominal_memory_gb": check["nominal_memory_gb"],
        "estimated_weights_gb": check["estimated_weights_gb"],
        "total_weights_gb": check["estimated_weights_gb"],
        "free_kv_gb": check["free_kv_gb"],
        "nominal_free_kv_gb": check["nominal_free_kv_gb"],
        "min_kv_headroom_gb": check["min_kv_headroom_gb"],
        "models": check["models"],
        "diagnostic": check["diagnostic"],
    }


QUANT_METHOD_REQUIRED_PACKAGES: Dict[str, List[str]] = {
    "gptq": [
        "vllm==0.33.0",
        "fastsafetensors",
    ],
    "gptq_4bit": [
        "vllm==0.33.0",
        "fastsafetensors",
    ],
    "awq": [
        "vllm==0.33.0",
        "fastsafetensors",
    ],
    "awq_4bit": [
        "vllm==0.33.0",
        "fastsafetensors",
    ],
    "nf4": ["vllm==0.33.0", "bitsandbytes>=0.43.0"],
    "nf4_4bit": ["vllm==0.33.0", "bitsandbytes>=0.43.0"],
    "bitsandbytes": ["vllm==0.33.0", "bitsandbytes>=0.43.0"],
    "bnb_8bit": ["vllm==0.33.0", "bitsandbytes>=0.43.0"],
    "fp8": ["vllm==0.33.0", "compressed-tensors==0.17.0", "fastsafetensors"],
    "compressed-tensors": ["vllm==0.33.0", "compressed-tensors==0.17.0", "fastsafetensors"],
    "gguf": ["huggingface_hub[hf_transfer]", "hf_transfer"],
    "diffusion": ["diffusers", "accelerate>=0.34.0"],
    "diffusion_pipeline": ["diffusers", "accelerate>=0.34.0"],
    "bfloat16": ["vllm==0.33.0", "fastsafetensors", "accelerate>=0.34.0", "transformers>=4.49.0"],
    "float16": ["vllm==0.33.0", "fastsafetensors", "accelerate>=0.34.0", "transformers>=4.49.0"],
}

COLAB_SERVING_REQUIRED_PACKAGES: List[str] = [
    "uv",
    "vllm==0.33.0",
    "fastsafetensors",
    "huggingface_hub[hf_transfer]",
    "hf_transfer",
    "google-cloud-storage",
    "bitsandbytes>=0.43.0",
    "compressed-tensors==0.17.0",
    "accelerate>=0.34.0",
    "transformers>=4.49.0",
    "diffusers",
    "sentencepiece",
    "protobuf",
    "requests",
    "pyngrok",
]

CUDA_ONLY_PACKAGES: Tuple[str, ...] = (
    "vllm",
    "sglang",
    "fastsafetensors",
    "flash-linear-attention",
    "bitsandbytes",
    "gptqmodel",
    "autoawq",
    "auto-gptq",
    "ninja",
    "defuser",
    "tokenicer",
    "device-smi",
    "logbar",
    "pypcre",
)

TPU_SAFE_SERVING_PACKAGES: List[str] = [
    "huggingface_hub[hf_transfer]",
    "hf_transfer",
    "google-cloud-storage",
    "accelerate>=0.34.0",
    "transformers>=4.49.0",
    "optimum",
    "compressed-tensors",
    "diffusers",
    "sentencepiece",
    "protobuf",
    "requests",
    "pyngrok",
]


def resolve_compute_dtype_for_hardware(
    gpu_tier: Optional[str] = None,
    compute_capability: Optional[Tuple[int, int]] = None,
) -> str:
    """Selects compute datatype (`'float16'` vs `'bfloat16'`) based on GPU microarchitecture or hardware tier (Spec 868 §4.4).

    - Turing / pre-Ampere GPUs (`sm_75`, e.g. NVIDIA T4) lack native hardware BF16 tensor-core instructions
      and MUST use `'float16'`.
    - Ampere / Ada / Hopper / Blackwell GPUs (`sm_80+`: L4, A100, A100-80GB, H100, G4) and Cloud TPUs (`v5e-1`, `v6e-1`)
      support native BF16 and use `'bfloat16'`.
    """
    if compute_capability is not None:
        try:
            major = int(compute_capability[0])
            if major < 8:
                return "float16"
            return "bfloat16"
        except (IndexError, TypeError, ValueError):
            pass

    if gpu_tier and str(gpu_tier).strip().lower() != "auto":
        try:
            canonical = normalize_hardware_tier(gpu_tier, allow_auto=False)
        except ValueError:
            canonical = str(gpu_tier).strip().upper()
        if canonical == "T4":
            return "float16"
        return "bfloat16"

    return "bfloat16"


def resolve_required_runtime_packages(
    model_alias_or_stack: Any = None,
    *,
    precision: Optional[str] = None,
    gpu_tier: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Resolves the complete closure of Python packages required in `Cell 1` to download, dequantize,
    and load the selected model(s) and quantization format(s) on `gpu_tier` (Spec 868 §4.1).
    """
    is_tpu = False
    canonical_tier: Optional[str] = None
    if gpu_tier and str(gpu_tier).strip().lower() != "auto":
        try:
            canonical_tier = normalize_hardware_tier(gpu_tier, allow_auto=False)
            is_tpu = GPU_TIER_SPECS[canonical_tier].get("accelerator_type") == "TPU"
        except ValueError:
            is_tpu = str(gpu_tier).strip().lower() in ("v5e-1", "v6e-1", "tpu")

    if is_tpu:
        return list(TPU_SAFE_SERVING_PACKAGES)

    if model_alias_or_stack is None:
        return list(COLAB_SERVING_REQUIRED_PACKAGES)

    items: List[Any]
    if isinstance(model_alias_or_stack, (str, Mapping)):
        items = [model_alias_or_stack]
    elif isinstance(model_alias_or_stack, Sequence):
        items = list(model_alias_or_stack)
    else:
        items = [str(model_alias_or_stack)]

    required: List[str] = [
        "uv",
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
        "google-cloud-storage",
        "accelerate>=0.34.0",
        "transformers>=4.49.0",
        "sentencepiece",
        "protobuf",
        "requests",
        "pyngrok",
    ]

    def _add_pkg(pkg: str) -> None:
        if pkg not in required:
            required.append(pkg)

    catalog = get_model_catalog(config)
    for item in items:
        raw_alias = str(item.get("alias") or item.get("model") or "") if isinstance(item, Mapping) else str(item or "")
        item_prec = (item.get("precision") if isinstance(item, Mapping) else None) or precision
        if "@" in raw_alias and not Path(raw_alias).expanduser().exists():
            raw_alias, inline_p = raw_alias.split("@", 1)
            item_prec = item_prec or inline_p.strip()
        raw_alias = raw_alias.strip()
        prof = catalog.get(raw_alias) or resolve_model_profile(raw_alias, config=config)

        is_image = prof.get("modality") == "image" or prof.get("model_type") == "diffusion_image"
        if is_image:
            _add_pkg("diffusers")
        else:
            _add_pkg("vllm==0.33.0")
            _add_pkg("fastsafetensors")

        try:
            var_info = resolve_model_precision_variant(
                prof["alias"],
                precision=item_prec,
                gpu_tier=canonical_tier,
                config=config,
            )
            q_method = str(var_info.get("quant_method") or "").lower()
            repo_id_low = str(var_info.get("hf_repo_id") or "").lower()
        except Exception:
            q_method = ""
            repo_id_low = str(prof.get("quantized_hf_repo_id") or prof.get("hf_repo_id") or "").lower()

        for q_key, pkgs in QUANT_METHOD_REQUIRED_PACKAGES.items():
            if q_key == q_method or (q_key in ("gptq", "awq", "fp8", "compressed-tensors") and q_key in repo_id_low):
                for p in pkgs:
                    _add_pkg(p)
        if not is_image:
            _add_pkg("bitsandbytes>=0.43.0")

    return required


_NON_LM_TENSOR_PREFIXES: Tuple[str, ...] = (
    "visual.",
    "model.visual.",
    "vision_tower.",
    "model.vision_tower.",
    "multi_modal_projector.",
    "mm_projector.",
    "mtp.",
    "model.mtp.",
)


def _is_non_lm_tensor_name(tensor_name: str) -> bool:
    t_low = str(tensor_name or "").strip().lower()
    return any(t_low.startswith(pfx) for pfx in _NON_LM_TENSOR_PREFIXES)


def _read_safetensors_header_tensor_bytes(shard_path: Path) -> Optional[Tuple[int, int]]:
    """Reads ONLY the 8-byte uint64 length + JSON header of a `.safetensors` file (zero weight I/O).

    Returns `(total_tensor_bytes, non_lm_tensor_bytes)` or `None` if header is unreadable.
    """
    try:
        with shard_path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                return None
            (hdr_len,) = struct.unpack("<Q", raw_len)
            if hdr_len <= 0 or hdr_len > 64 * 1024 * 1024:
                return None
            hdr_obj = json.loads(fh.read(hdr_len).decode("utf-8"))
        if not isinstance(hdr_obj, Mapping):
            return None
        total_b = 0
        non_lm_b = 0
        for t_name, meta in hdr_obj.items():
            if t_name == "__metadata__" or not isinstance(meta, Mapping):
                continue
            offsets = meta.get("data_offsets")
            if isinstance(offsets, Sequence) and len(offsets) == 2:
                span = max(0, int(offsets[1]) - int(offsets[0]))
                total_b += span
                if _is_non_lm_tensor_name(str(t_name)):
                    non_lm_b += span
        return (total_b, non_lm_b) if total_b > 0 else None
    except Exception:
        return None


def inspect_local_checkpoint_footprint_gib(
    checkpoint_dir: Union[str, Path, None],
    *,
    language_model_only: bool = False,
    quant_method: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Computes exact on-disk and post-load GPU weight footprint from local safetensors headers in <= 50 ms.

    Performs zero network I/O and zero tensor-body reads (Spec 871 §3.1 & §4.3).
    When `language_model_only=True`, subtracts multimodal vision encoder (`visual.*`, `vision_tower.*`)
    and unused speculative MTP (`mtp.*`) tensors so hybrid VLM checkpoints (e.g. `73.45 GiB` on disk ->
    `67.65 GiB` resident language-model weights on `Qwen3.5-122B-A10B-GPTQ-Int4`) are sized accurately.
    """
    if not checkpoint_dir:
        return None
    try:
        ckpt_path = Path(str(checkpoint_dir)).expanduser()
        if not ckpt_path.exists() or not ckpt_path.is_dir():
            return None
    except Exception:
        return None

    detected_quant = str(quant_method or "").strip().lower() or None
    cfg_file = ckpt_path / "config.json"
    if cfg_file.is_file():
        try:
            cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(cfg_data, Mapping) and not detected_quant:
                q_cfg = cfg_data.get("quantization_config")
                if isinstance(q_cfg, Mapping) and q_cfg.get("quant_method"):
                    detected_quant = str(q_cfg["quant_method"]).strip().lower()
        except Exception:
            pass

    index_file = ckpt_path / "model.safetensors.index.json"
    total_disk_bytes = 0
    total_tensor_bytes = 0
    non_lm_bytes = 0
    shard_count = 0

    if index_file.is_file():
        try:
            idx_data = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            idx_data = {}
        weight_map = idx_data.get("weight_map") if isinstance(idx_data, Mapping) else None
        meta_block = idx_data.get("metadata") if isinstance(idx_data, Mapping) else None
        meta_total_size = int((meta_block or {}).get("total_size") or 0) if isinstance(meta_block, Mapping) else 0

        unique_shards: List[str] = []
        if isinstance(weight_map, Mapping) and weight_map:
            seen_shards: set[str] = set()
            for sh in weight_map.values():
                sh_s = str(sh)
                if sh_s not in seen_shards:
                    seen_shards.add(sh_s)
                    unique_shards.append(sh_s)

        existing_shards = [ckpt_path / sh for sh in unique_shards if (ckpt_path / sh).is_file()]
        if existing_shards:
            shard_count = len(existing_shards)
            for sp in existing_shards:
                f_sz = sp.stat().st_size
                total_disk_bytes += f_sz
                parsed = _read_safetensors_header_tensor_bytes(sp)
                if parsed is not None:
                    total_tensor_bytes += parsed[0]
                    non_lm_bytes += parsed[1]
                else:
                    total_tensor_bytes += f_sz
        elif meta_total_size > 0:
            shard_count = max(1, len(unique_shards))
            total_disk_bytes = meta_total_size
            total_tensor_bytes = meta_total_size
            if language_model_only and isinstance(weight_map, Mapping) and weight_map:
                # If only index metadata is present, identify shards solely holding non-LM tensors
                shard_all_non_lm: Dict[str, bool] = {}
                for t_name, sh_name in weight_map.items():
                    sh_s = str(sh_name)
                    is_nlm = _is_non_lm_tensor_name(str(t_name))
                    shard_all_non_lm[sh_s] = shard_all_non_lm.get(sh_s, True) and is_nlm
                if any(shard_all_non_lm.values()) and len(shard_all_non_lm) > 1:
                    nlm_frac = sum(1 for v in shard_all_non_lm.values() if v) / float(len(shard_all_non_lm))
                    non_lm_bytes = int(meta_total_size * nlm_frac)

    if total_tensor_bytes <= 0:
        st_files = sorted(p for p in ckpt_path.glob("*.safetensors") if p.is_file())
        if not st_files:
            return None
        shard_count = len(st_files)
        for sp in st_files:
            f_sz = sp.stat().st_size
            total_disk_bytes += f_sz
            parsed = _read_safetensors_header_tensor_bytes(sp)
            if parsed is not None:
                total_tensor_bytes += parsed[0]
                non_lm_bytes += parsed[1]
            else:
                total_tensor_bytes += f_sz

    if total_tensor_bytes <= 0:
        return None

    excluded_bytes = non_lm_bytes if language_model_only else 0
    lm_bytes = max(1, total_tensor_bytes - excluded_bytes)
    lm_gib = lm_bytes / float(1024 ** 3)
    # Account for Marlin / GPTQ / AWQ scale alignment & repacking overhead (~2.5% on packed 4-bit kernels: 66.0 GiB -> 67.65 GiB)
    repack_factor = 1.025 if detected_quant in ("gptq", "auto_gptq", "awq", "compressed-tensors") else 1.005
    est_gpu_gib = lm_gib * repack_factor

    return {
        "disk_checkpoint_bytes": int(total_disk_bytes),
        "disk_checkpoint_gib": round(total_disk_bytes / float(1024 ** 3), 4),
        "total_tensor_bytes": int(total_tensor_bytes),
        "total_tensor_gib": round(total_tensor_bytes / float(1024 ** 3), 4),
        "lm_tensor_bytes": int(lm_bytes),
        "lm_tensor_gib": round(lm_gib, 4),
        "active_tensor_gib": round(lm_gib, 4),
        "excluded_non_lm_bytes": int(excluded_bytes),
        "excluded_non_lm_gib": round(excluded_bytes / float(1024 ** 3), 4),
        "excluded_multimodal_or_mtp_gib": round(excluded_bytes / float(1024 ** 3), 4),
        "estimated_gpu_weights_gib": round(est_gpu_gib, 4),
        "shard_count": int(shard_count),
        "quant_method": detected_quant,
        "language_model_only": bool(language_model_only),
    }


def estimate_model_kv_and_state_footprint(
    model_alias_or_config: Any,
    *,
    max_model_len: int = 16384,
    kv_cache_dtype: str = "auto",
    mamba_ssm_cache_dtype: str = "auto",
    enable_prefix_caching: bool = True,
) -> Dict[str, Any]:
    """Computes exact physical KV cache bytes/token, Gated-DeltaNet (GDN) / MLA state footprint,
    and vLLM V1 `mamba_cache_mode='align'` unified page size from `config.json` / `BUNDLED_MODEL_METADATA`
    (Spec 869 §4.1 & Spec 871 §3.2).
    """
    mod_cfg: Dict[str, Any] = {}
    if isinstance(model_alias_or_config, Mapping):
        if "model_config" in model_alias_or_config and isinstance(model_alias_or_config["model_config"], Mapping):
            mod_cfg = dict(model_alias_or_config["model_config"])
        elif "num_hidden_layers" in model_alias_or_config or "kv_lora_rank" in model_alias_or_config or "full_attn_layers" in model_alias_or_config:
            mod_cfg = dict(model_alias_or_config)
        else:
            ident = str(model_alias_or_config.get("hf_repo_id") or model_alias_or_config.get("alias") or "")
            if ident:
                mod_cfg, _, _, _ = _load_local_or_cached_metadata(ident)
    else:
        ident = str(model_alias_or_config or "").strip()
        if "@" in ident and not Path(ident).expanduser().exists():
            ident = ident.split("@", 1)[0].strip()
        if ident:
            if ident in BUILTIN_MODEL_CATALOG:
                ident = str(BUILTIN_MODEL_CATALOG[ident].get("hf_repo_id") or ident)
            mod_cfg, _, _, _ = _load_local_or_cached_metadata(ident)

    text_cfg = mod_cfg.get("text_config") if isinstance(mod_cfg.get("text_config"), Mapping) else mod_cfg
    mt = str(text_cfg.get("model_type") or mod_cfg.get("model_type") or "").lower()

    is_mla = bool(text_cfg.get("kv_lora_rank")) or mt in ("deepseek_v2", "deepseek_v3")
    n_linear = int(text_cfg.get("linear_attn_layers", 0) or 0)
    layer_types = text_cfg.get("layer_types")
    if n_linear == 0 and isinstance(layer_types, Sequence) and not isinstance(layer_types, (str, bytes)):
        n_linear = sum(1 for lt in layer_types if "linear" in str(lt).lower() or "mamba" in str(lt).lower())
    is_hybrid_gdn = n_linear > 0 or mt.startswith("qwen3_5")

    kv_dtype_norm = str(kv_cache_dtype or "auto").strip().lower()
    # DeepSeek MLA does not support FP8 KV cache in vLLM; clamp element size to 2 bytes on MLA
    elem_bytes = 1.0 if (kv_dtype_norm.startswith("fp8") and not is_mla) else 2.0

    n_layers = int(text_cfg.get("num_hidden_layers", 32) or 32)
    n_full = n_layers
    aligned_block_tokens = 16
    unified_aligned_block_bytes = 0.0

    if is_mla:
        kv_lora_rank = int(text_cfg.get("kv_lora_rank", 512) or 512)
        qk_rope_dim = int(text_cfg.get("qk_rope_head_dim", 64) or 64)
        kv_bytes_per_token = float(n_layers * (kv_lora_rank + qk_rope_dim) * 2.0)
        gdn_bytes_per_seq = 0.0
        aligned_block_tokens = 64
        unified_aligned_block_bytes = float(aligned_block_tokens) * kv_bytes_per_token
    elif is_hybrid_gdn:
        n_full = int(text_cfg.get("full_attn_layers", max(1, n_layers // 4)) or max(1, n_layers // 4))
        if n_linear <= 0:
            n_linear = max(0, n_layers - n_full)
        n_kv_heads = int(text_cfg.get("num_key_value_heads", 4) or 4)
        head_dim = int(text_cfg.get("head_dim", 256) or 256)
        attn_layer_kv_bytes_per_token = float(2 * n_kv_heads * head_dim * elem_bytes)
        kv_bytes_per_token = float(n_full) * attn_layer_kv_bytes_per_token

        lin_k_heads = int(text_cfg.get("linear_num_key_heads", 16) or 16)
        lin_v_heads = int(text_cfg.get("linear_num_value_heads", 32) or 32)
        lin_k_dim = int(text_cfg.get("linear_key_head_dim", 128) or 128)
        lin_v_dim = int(text_cfg.get("linear_value_head_dim", 128) or 128)
        conv_k = int(text_cfg.get("linear_conv_kernel_dim", 4) or 4)

        ssm_dtype_norm = str(mamba_ssm_cache_dtype or "auto").strip().lower()
        # In vLLM V1, `--mamba-ssm-cache-dtype auto` stores GDN recurrent state in float32 (4 bytes)
        ssm_elem_bytes = 4.0 if ssm_dtype_norm in ("auto", "float32", "fp32") else 2.0
        recurrent_mat = float(lin_v_heads * lin_k_dim * lin_v_dim) * ssm_elem_bytes
        conv_state = float((2 * lin_k_heads * lin_k_dim + lin_v_heads * lin_v_dim) * max(1, conv_k - 1)) * 2.0
        mamba_page_bytes_per_layer = recurrent_mat + conv_state
        gdn_bytes_per_seq = float(n_linear) * mamba_page_bytes_per_layer

        if enable_prefix_caching and attn_layer_kv_bytes_per_token > 0.0:
            raw_tokens = math.ceil(mamba_page_bytes_per_layer / attn_layer_kv_bytes_per_token)
            aligned_block_tokens = int(math.ceil(raw_tokens / 64.0) * 64)
            aligned_page_bytes = float(aligned_block_tokens) * attn_layer_kv_bytes_per_token
            unified_aligned_block_bytes = float(n_full + n_linear) * aligned_page_bytes
        else:
            aligned_block_tokens = 16
            unified_aligned_block_bytes = float(aligned_block_tokens) * kv_bytes_per_token + gdn_bytes_per_seq
    else:
        n_kv_heads = int(text_cfg.get("num_key_value_heads", 8) or 8)
        hidden_sz = int(text_cfg.get("hidden_size", 4096) or 4096)
        n_attn_heads = int(text_cfg.get("num_attention_heads", 32) or 32)
        head_dim = int(text_cfg.get("head_dim") or max(64, hidden_sz // max(1, n_attn_heads)))
        kv_bytes_per_token = float(2 * n_layers * n_kv_heads * head_dim * elem_bytes)
        gdn_bytes_per_seq = 0.0
        aligned_block_tokens = 16
        unified_aligned_block_bytes = float(aligned_block_tokens) * kv_bytes_per_token

    eff_len = max(256, int(max_model_len or 16384))
    kv_gb_per_seq = (kv_bytes_per_token * eff_len) / float(1024 ** 3)
    gdn_gb_per_seq = gdn_bytes_per_seq / float(1024 ** 3)
    unified_aligned_block_gib = unified_aligned_block_bytes / float(1024 ** 3)
    return {
        "kv_bytes_per_token": round(kv_bytes_per_token, 2),
        "kv_gb_per_seq_at_max_len": round(kv_gb_per_seq, 6),
        "gdn_bytes_per_seq": round(gdn_bytes_per_seq, 2),
        "gdn_gb_per_seq": round(gdn_gb_per_seq, 6),
        "total_gb_per_seq_at_max_len": round(kv_gb_per_seq + gdn_gb_per_seq, 6),
        "aligned_block_tokens": int(aligned_block_tokens),
        "unified_aligned_block_bytes": int(round(unified_aligned_block_bytes)),
        "unified_aligned_block_gib": round(unified_aligned_block_gib, 6),
        "num_hidden_layers": int(n_layers),
        "full_attn_layers": int(n_full),
        "linear_attn_layers": int(n_linear),
        "is_hybrid_gdn": bool(is_hybrid_gdn),
        "is_mla": bool(is_mla),
    }


def compute_dynamic_max_num_seqs(
    free_vram_gb: float,
    weights_gb: float = 0.0,
    *,
    model_alias: str = "",
    max_model_len: int = 16384,
    kv_cache_dtype: str = "auto",
    requested_agents: int = 1,
    active_slot_count: int = 1,
    tier_max_seqs: Optional[int] = None,
) -> int:
    """Computes dynamic continuous-batching `max_num_seqs` (`1..256` on CUDA) from physical KV + GDN pool headroom.

    Strictly respects post-weight KV/aligned-block pool capacity without forcing an unconditional 16-sequence
    floor when tight post-weight headroom cannot physically hold 16 concurrent sequences (Spec 871 §3.2).
    """
    slot_cnt = max(1, int(active_slot_count or 1))
    eff_vram = max(0.0, float(free_vram_gb))
    eff_w = max(0.0, float(weights_gb))
    # Support callers passing either total GPU `free_vram_gb` (with `weights_gb > 0`) or already-netted KV headroom (`weights_gb == 0`)
    if eff_w > 0.0:
        if slot_cnt == 1:
            os_margin = 4.0 if eff_vram >= 70.0 else max(1.4, 0.06 * eff_vram)
            u_safe = min(0.94, max(0.10, (eff_vram - os_margin) / max(1.0, eff_vram)))
            w_ratio = eff_w / max(1.0, eff_vram)
            util_cap = min(u_safe, 0.90 + 0.04 * min(1.0, max(0.0, (w_ratio - 0.60) / 0.25)))
            usable_slot_vram = eff_vram * util_cap
            post_weight = usable_slot_vram - eff_w
            runtime_overhead_gb = 2.26 if post_weight < 9.0 else max(1.4, min(3.2, 0.045 * eff_vram))
        else:
            usable_slot_vram = eff_vram * ((0.92 - 0.025 * (slot_cnt - 1)) / slot_cnt)
            runtime_overhead_gb = max(1.2, 0.05 * eff_vram / slot_cnt)
        kv_pool_gb = usable_slot_vram - eff_w - runtime_overhead_gb
    else:
        kv_pool_gb = eff_vram * 0.90

    max_cap = min(256, max(1, int(tier_max_seqs))) if tier_max_seqs is not None else 256
    if kv_pool_gb <= 0.0:
        return 1

    fp = estimate_model_kv_and_state_footprint(
        model_alias or "qwen-9b",
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
    )
    working_tokens = min(max(512, int(max_model_len or 16384)), 2048)
    working_kv_gb = (float(fp["kv_bytes_per_token"]) * working_tokens) / float(1024 ** 3)
    per_seq_working_gb = max(0.02, working_kv_gb + float(fp["gdn_gb_per_seq"]))

    if fp.get("is_hybrid_gdn") and float(fp.get("unified_aligned_block_gib", 0.0)) > 0.0:
        # In vLLM V1 mamba_cache_mode='align', reserve ~3 aligned blocks per active sequence to prevent preemption OOM
        aligned_blk_gib = float(fp["unified_aligned_block_gib"])
        per_seq_working_gb = max(per_seq_working_gb, 3.0 * aligned_blk_gib)

    raw_seqs = max(1, int(kv_pool_gb // per_seq_working_gb))
    if raw_seqs < 16:
        return max(1, min(max_cap, raw_seqs))
    req_floor = min(max_cap, max(16, int(requested_agents or 1)))
    if raw_seqs >= req_floor:
        return min(max_cap, raw_seqs)
    return min(max_cap, raw_seqs)


def resolve_vllm_model_parser_spec(
    model_alias_or_arch: str,
    *,
    tokenizer_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolves exact `vllm serve` `--reasoning-parser`, `--tool-call-parser`, `--enable-auto-tool-choice`,
    hybrid Mamba/GDN flags, and `enable_thinking` support for each model family in `[01]`–`[14]` (Spec 869 §3.3).
    """
    raw = str(model_alias_or_arch or "").strip()
    if "@" in raw and not Path(raw).expanduser().exists():
        raw = raw.split("@", 1)[0].strip()
    low = raw.lower()

    prof = BUILTIN_MODEL_CATALOG.get(raw) or BUILTIN_MODEL_CATALOG.get(low)
    model_type = str((prof or {}).get("model_type") or "").lower()
    alias = str((prof or {}).get("alias") or low)

    if not model_type:
        mod_cfg, tok_cfg, _, _ = _load_local_or_cached_metadata(raw)
        fam = classify_model_family_neutral(
            raw,
            model_config=mod_cfg or None,
            tokenizer_config=dict(tokenizer_meta) if isinstance(tokenizer_meta, Mapping) else (tok_cfg or None),
        )
        model_type = str(fam.get("architecture_family") or fam.get("model_type") or "").lower()

    is_qwen3_coder = (
        alias == "qwen-coder-30b-moe"
        or "qwen3-coder" in low
        or "qwen-coder-30b" in low
    )
    is_qwen35 = (
        model_type in ("qwen3_5", "qwen3_5_moe")
        or alias in ("qwen-9b", "qwen-27b", "qwen3.8-27b", "qwen-27b-38", "qwen-35b-moe", "qwen-122b-moe")
        or any(t in low for t in ("qwen3.5", "qwen3_5", "qwen3.6", "qwen3_6", "qwen3.8", "qwen3_8"))
    )
    is_r1 = (
        model_type == "deepseek_r1"
        or alias in ("r1-7b", "r1-14b", "r1-32b")
        or "deepseek-r1" in low
        or "r1-distill" in low
    )
    is_deepseek_v2 = (
        model_type in ("deepseek_v2", "deepseek_v3")
        or alias == "deepseek-v2-lite"
        or "deepseek-v2" in low
    )
    is_qwen3 = (
        not is_qwen35
        and not is_qwen3_coder
        and (
            model_type in ("qwen3", "qwen3_moe")
            or alias in ("qwen-4b", "qwen-8b", "qwen-14b", "qwen-30b-moe")
            or bool(re.search(r"qwen[-_]?3(?![._]?[568])", low))
        )
    )

    if is_qwen35:
        return {
            "family": "qwen3_5",
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "enable_auto_tool_choice": True,
            "mamba_ssm_cache_dtype": "auto",
            "language_model_only": True,
            "supports_enable_thinking": True,
            "client_prompt_tool_fallback": False,
        }
    if is_qwen3_coder:
        # [08] Qwen3-Coder-30B-A3B-Instruct has NO `<think>` block in chat_template — NEVER pass `--reasoning-parser`!
        return {
            "family": "qwen3_coder",
            "reasoning_parser": None,
            "tool_call_parser": "qwen3_coder",
            "enable_auto_tool_choice": True,
            "mamba_ssm_cache_dtype": None,
            "language_model_only": False,
            "supports_enable_thinking": False,
            "client_prompt_tool_fallback": False,
        }
    if is_qwen3:
        return {
            "family": "qwen3",
            "reasoning_parser": "qwen3",
            "tool_call_parser": "hermes",
            "enable_auto_tool_choice": True,
            "mamba_ssm_cache_dtype": None,
            "language_model_only": False,
            "supports_enable_thinking": True,
            "client_prompt_tool_fallback": False,
        }
    if is_r1:
        return {
            "family": "deepseek_r1_distill",
            "reasoning_parser": "deepseek_r1",
            "tool_call_parser": None,
            "enable_auto_tool_choice": False,
            "mamba_ssm_cache_dtype": None,
            "language_model_only": False,
            "supports_enable_thinking": False,
            "client_prompt_tool_fallback": True,
        }
    if is_deepseek_v2:
        return {
            "family": "deepseek_v2_lite",
            "reasoning_parser": None,
            "tool_call_parser": None,
            "enable_auto_tool_choice": False,
            "mamba_ssm_cache_dtype": None,
            "language_model_only": False,
            "supports_enable_thinking": False,
            "client_prompt_tool_fallback": True,
        }
    return {
        "family": model_type or "generic_causal_lm",
        "reasoning_parser": None,
        "tool_call_parser": None,
        "enable_auto_tool_choice": False,
        "mamba_ssm_cache_dtype": None,
        "language_model_only": False,
        "supports_enable_thinking": False,
        "client_prompt_tool_fallback": True,
    }


def resolve_omp_model_compat(
    model_alias_or_arch: str,
    *,
    tokenizer_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolves `omp` (`models.yml`) reasoning flag and OpenAI-completions `compat` fields from catalog/protocol metadata."""
    parser_spec = resolve_vllm_model_parser_spec(model_alias_or_arch, tokenizer_meta=tokenizer_meta)
    family = str(parser_spec.get("family") or "").strip().lower()
    has_reasoning = bool(parser_spec.get("reasoning_parser") is not None)
    supports_enable_thinking = bool(parser_spec.get("supports_enable_thinking", False))

    if not has_reasoning and family not in (
        "qwen3_5",
        "qwen3_coder",
        "qwen3",
        "deepseek_r1_distill",
        "deepseek_v2_lite",
    ):
        try:
            proto = introspect_model_protocol(str(model_alias_or_arch or "").strip(), allow_remote=False)
            if proto.get("forces_think_prefix") or proto.get("default_enable_thinking"):
                has_reasoning = True
                if proto.get("supports_enable_thinking_kwarg"):
                    supports_enable_thinking = True
        except Exception:
            pass

    compat: Dict[str, Any] = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "maxTokensField": "max_tokens",
        "reasoningContentField": "reasoning_content",
        "streamMarkupHealingPattern": "qwen" if family in ("qwen3_5", "qwen3", "qwen3_coder") else "thinking",
    }
    if has_reasoning:
        if supports_enable_thinking or family in ("qwen3_5", "qwen3"):
            compat["thinkingFormat"] = "qwen-chat-template"
        elif family == "deepseek_r1_distill":
            compat["thinkingFormat"] = "openai"
    else:
        compat["supportsReasoningParams"] = False

    return {
        "family": family,
        "reasoning": bool(has_reasoning),
        "supportsTools": True,
        "compat": compat,
    }


OMP_DEFAULT_SYSTEM_OVERHEAD_TOKENS: int = 2300
OMP_FULL_TOOLSET_OVERHEAD_TOKENS: int = 5200
OMP_PER_TOOL_SCHEMA_TOKENS: int = 460
OMP_COMPACT_CODING_TOOLS: Tuple[str, ...] = (
    "read",
    "write",
    "edit",
    "bash",
    "grep",
    "glob",
    "todo",
)

_CLASS_C_LARGE_REASONING_MODELS = frozenset(
    {
        "qwen-27b",
        "qwen-27b-36",
        "qwen3.8-27b",
        "qwen-27b-38",
        "qwen-35b-moe",
        "qwen-30b-moe",
        "qwen-122b-moe",
    }
)
_CLASS_B_NON_REASONING_MODELS = frozenset(
    {
        "deepseek-v2-lite",
        "qwen-coder-30b-moe",
    }
)


def resolve_model_context_profile(
    model_alias_or_id: str,
    *,
    context_window: int,
    active_tools: Optional[Sequence[str]] = None,
    compact_tools: bool = False,
    requested_max_tokens: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolves model-family context-engineering budgets, compaction overlay, and feasibility gates (Spec 874 §3 & §4.3-4.4).

    Covers all 14 `PRIORITY_TEXT_MODELS` (Classes A/B/C/D) plus custom HF repositories.
    """
    raw_id = str(model_alias_or_id or "").strip()
    clean_id = raw_id.split("@", 1)[0].strip() if ("@" in raw_id and not Path(raw_id).expanduser().exists()) else raw_id

    profile: Dict[str, Any] = {}
    canonical_alias = clean_id
    if clean_id:
        try:
            profile = resolve_model_profile(clean_id, config=config)
            canonical_alias = str(profile.get("alias") or profile.get("model_alias") or clean_id)
        except ValueError:
            profile = {}

    compat_info = resolve_omp_model_compat(canonical_alias or clean_id)
    reasoning = bool(compat_info.get("reasoning", False))

    eff_window = max(1024, int(context_window or 16384))
    low_id = f"{canonical_alias} {clean_id}".lower()

    if canonical_alias == "r1-32b" or "r1-distill-qwen-32b" in low_id:
        profile_class = "class_d_heavy_reasoning"
        reasoning = True
    elif canonical_alias in _CLASS_B_NON_REASONING_MODELS or not reasoning:
        profile_class = "class_b_non_reasoning"
        reasoning = False
    elif canonical_alias in _CLASS_C_LARGE_REASONING_MODELS or eff_window >= 24576:
        profile_class = "class_c_large_reasoning"
    else:
        profile_class = "class_a_compact_reasoning"

    if eff_window <= 16384:
        recommended_max_write_lines = 150
    elif eff_window <= 32768:
        recommended_max_write_lines = 220
    else:
        recommended_max_write_lines = 500

    append_system_prompt: Optional[str] = None
    if eff_window <= 32768:
        append_system_prompt = (
            f"Context discipline ({eff_window}-token window): "
            "1) Use targeted ranged reads via inline path selectors (`read` with `path=\"file:start-end\"`, never bare full-file reads of large docs) and `grep`. "
            "2) Keep bash outputs bounded (head/tail/filtered) so verbose logs never overflow context. "
            f"3) Split large file writes (> {recommended_max_write_lines} lines) into an initial write plus incremental edits to avoid stop=length truncation."
        )

    addendum_tokens = math.ceil(len(append_system_prompt) / 3.0) if append_system_prompt else 0

    if active_tools is not None:
        norm_tools = [str(t).strip() for t in active_tools if str(t).strip()]
        tool_overhead = min(OMP_FULL_TOOLSET_OVERHEAD_TOKENS, len(norm_tools) * OMP_PER_TOOL_SCHEMA_TOKENS)
    elif compact_tools:
        tool_overhead = min(OMP_FULL_TOOLSET_OVERHEAD_TOKENS, len(OMP_COMPACT_CODING_TOOLS) * OMP_PER_TOOL_SCHEMA_TOKENS)
    else:
        tool_overhead = OMP_FULL_TOOLSET_OVERHEAD_TOKENS

    omp_overhead_tokens = OMP_DEFAULT_SYSTEM_OVERHEAD_TOKENS + tool_overhead + addendum_tokens

    # Window-tiered target output reserve (R_out) and class Turn-0 prompt ceiling
    if profile_class == "class_d_heavy_reasoning":
        if eff_window >= 65536:
            baseline_r_out, class_cap = 12288, 16000
        elif eff_window >= 24576:
            baseline_r_out, class_cap = 8192, 5500
        else:
            baseline_r_out, class_cap = 8192, 2048
    elif profile_class == "class_b_non_reasoning":
        if eff_window >= 65536:
            baseline_r_out, class_cap = 8192, 20000
        elif eff_window >= 24576:
            baseline_r_out, class_cap = 4096, 8000
        else:
            baseline_r_out, class_cap = 3072, 3500
    else:
        if eff_window >= 65536:
            baseline_r_out, class_cap = 8192, 18000
        elif eff_window >= 24576:
            baseline_r_out, class_cap = 6144, 6500
        else:
            baseline_r_out, class_cap = 4096, 2500

    target_r_out = (
        int(requested_max_tokens)
        if (isinstance(requested_max_tokens, (int, float)) and not isinstance(requested_max_tokens, bool) and int(requested_max_tokens) > 0)
        else baseline_r_out
    )

    # First-principles feasibility & proportional maxTokens clamp gate (Spec 874 §3.2 & §4.3.4)
    feasible = True
    budget_warning: Optional[str] = None
    window_out_cap = (4096 if reasoning else 3072) if eff_window <= 16384 else target_r_out
    net_raw = eff_window - omp_overhead_tokens - target_r_out
    if net_raw < 4096 or (eff_window <= 16384 and target_r_out > window_out_cap):
        raw_clamped = max(2048, min(target_r_out, window_out_cap, eff_window - omp_overhead_tokens - 4096))
        clamped_r_out = max(2048, (raw_clamped // 256) * 256)
        net_after_clamp = eff_window - omp_overhead_tokens - clamped_r_out
        if net_after_clamp >= 4096:
            eff_r_out = clamped_r_out
            budget_warning = (
                f"Proportional maxTokens clamp for '{canonical_alias}' at context_window={eff_window}: "
                f"reduced output reserve {target_r_out} -> {eff_r_out} "
                f"(omp_overhead={omp_overhead_tokens}, net_working={net_after_clamp}) to preserve >= 4096 working tokens."
            )
        else:
            net_min_floor = eff_window - omp_overhead_tokens - 2048
            eff_r_out = 2048
            if net_min_floor >= 2048:
                budget_warning = (
                    f"Tight context window warning for '{canonical_alias}' at context_window={eff_window}: "
                    f"clamped output reserve {target_r_out} -> 2048 "
                    f"(omp_overhead={omp_overhead_tokens}, net_working={net_min_floor})."
                )
            else:
                feasible = False
                budget_warning = (
                    f"Unfeasible context budget for '{canonical_alias}': "
                    f"context_window={eff_window} - omp_overhead={omp_overhead_tokens} - min_output_reserve=2048 "
                    f"= {net_min_floor} (< 2048 minimum viable working window)."
                )
    else:
        eff_r_out = target_r_out

    net_working = eff_window - omp_overhead_tokens - eff_r_out
    turn0_alpha = 0.50
    if feasible:
        raw_turn0 = int(math.floor(net_working * turn0_alpha))
        safe_ceiling = max(1024, net_working - 1024)
        max_turn0 = max(1024, min(class_cap, raw_turn0, safe_ceiling))
    else:
        max_turn0 = 0

    # Dynamic native `omp` compaction overlay calibrated to live window tier (Spec 874 §4.4)
    if eff_window <= 16384:
        comp_threshold = 72
        comp_reserve = 4096 if reasoning else 3584
        comp_keep_recent = min(2500, max(1024, net_working - 1024)) if feasible else 1024
        comp_v2_budget = 4096
        comp_async = False
    elif eff_window <= 32768:
        comp_threshold = 78
        comp_reserve = 6144 if profile_class == "class_d_heavy_reasoning" else 5120
        comp_keep_recent = 6000
        comp_v2_budget = 10240
        comp_async = False
    else:
        comp_threshold = 82
        comp_reserve = 8192
        comp_keep_recent = 14000
        comp_v2_budget = 24576
        comp_async = True

    return {
        "canonical_alias": canonical_alias,
        "profile_class": profile_class,
        "reasoning": reasoning,
        "context_window": eff_window,
        "chars_per_token": 3.0,
        "omp_overhead_tokens": omp_overhead_tokens,
        "output_reserve_tokens": eff_r_out,
        "baseline_output_tokens": baseline_r_out,
        "net_working_budget_tokens": net_working,
        "turn0_alpha": turn0_alpha,
        "max_turn0_prompt_tokens": max_turn0,
        "feasible": feasible,
        "budget_warning": budget_warning,
        "compaction_overlay": {
            "enabled": True,
            "thresholdPercent": comp_threshold,
            "reserveTokens": comp_reserve,
            "keepRecentTokens": comp_keep_recent,
            "v2RetainedMessageBudget": comp_v2_budget,
            "asyncEnabled": comp_async,
        },
        "append_system_prompt": append_system_prompt,
        "recommended_max_write_lines": recommended_max_write_lines,
    }


def resolve_download_ignore_patterns(
    model_alias_or_repo: str = "",
    *,
    precision: Optional[str] = None,
    quant_method: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Returns `snapshot_download(ignore_patterns=...)` that never strips weight files required by the selected model format (Spec 868 §4.5).

    Specifically, when the model or checkpoint uses GGUF weights (e.g. `flux-klein-9b` -> `unsloth/FLUX.2-klein-9B-GGUF`),
    `'*.gguf'` is omitted from `ignore_patterns`.
    """
    raw = str(model_alias_or_repo or "").strip()
    inline_prec: Optional[str] = None
    if "@" in raw and not Path(raw).expanduser().exists():
        raw, inline_prec = raw.split("@", 1)
        raw = raw.strip()
        inline_prec = inline_prec.strip() or None
    eff_prec = (precision or inline_prec or "").strip()
    eff_qmethod = (quant_method or "").strip().lower()

    is_gguf = "gguf" in raw.lower() or "gguf" in eff_prec.lower() or "gguf" in eff_qmethod
    if not is_gguf and raw:
        catalog = get_model_catalog(config)
        prof = catalog.get(raw)
        if prof is not None:
            q_repo = str(prof.get("quantized_hf_repo_id") or "")
            q_map = prof.get("quantized_repo_ids") or {}
            def_precs = prof.get("default_precision_by_gpu") or {}
            if "gguf" in q_repo.lower() and (not eff_prec or eff_prec.upper() in ("Q4", "GGUF") or "gguf" in def_precs.values()):
                if eff_prec.upper() != "BF16" and eff_prec.lower() != "bfloat16":
                    is_gguf = True
            elif eff_prec and "gguf" in str(q_map.get(eff_prec, "")).lower():
                is_gguf = True

    if is_gguf:
        return ["*.bin", "original/*", "*.msgpack", "*.h5", "consolidated*"]
    return ["*.gguf", "*.bin", "original/*", "*.msgpack", "*.h5", "consolidated*"]


def export_colab_catalog_payload(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Exports a compact JSON-serializable snapshot of all 8 hardware tiers, 14+3 model profiles,
    precision variants, compute dtypes, required packages, and pre-validated stack presets for Colab Cell 1-6 synthesis.
    """
    catalog = get_model_catalog(config)
    models_export: Dict[str, Any] = {}
    for alias, prof in catalog.items():
        models_export[alias] = {
            "alias": alias,
            "family": prof.get("family", "qwen"),
            "modality": prof.get("modality", "text"),
            "source": prof.get("source", "hf_hub"),
            "storage_uri": prof.get("storage_uri"),
            "hf_repo_id": prof.get("hf_repo_id"),
            "quantized_hf_repo_id": prof.get("quantized_hf_repo_id"),
            "quantized_repo_ids": copy.deepcopy(prof.get("quantized_repo_ids") or {}),
            "precision_variants": copy.deepcopy(prof.get("precision_variants") or {}),
            "default_precision_by_gpu": copy.deepcopy(prof.get("default_precision_by_gpu") or {}),
            "min_vram_gb": float(prof.get("min_vram_gb", 12.0)),
            "preferred_gpu": prof.get("preferred_gpu", "T4"),
            "supported_hardware_tiers": list(prof.get("supported_hardware_tiers") or prof.get("supported_gpus") or []),
            "architecture": prof.get("architecture"),
            "model_type": prof.get("model_type"),
            "loader_kind": prof.get("loader_kind", "causal_lm"),
            "loader_class_name": prof.get("loader_class_name", "AutoModelForCausalLM"),
            "is_qwen35_family": bool(prof.get("is_qwen35_family", False)),
            "template_family": prof.get("template_family"),
            "stop_token_ids": list(prof.get("stop_token_ids") or []),
            "max_model_len": int(prof.get("max_model_len", 16384)),
            "max_num_seqs": int(prof.get("max_num_seqs", 32)),
            "vllm_parser_spec": (
                resolve_vllm_model_parser_spec(alias)
                if prof.get("modality", "text") != "image"
                else None
            ),
        }
    tier_specs = get_hardware_tier_specs()
    compute_dtype_by_gpu = {
        tier: resolve_compute_dtype_for_hardware(tier) for tier in ALL_HARDWARE_TIERS
    }
    return {
        "all_hardware_tiers": list(ALL_HARDWARE_TIERS),
        "priority_text_models": list(PRIORITY_TEXT_MODELS),
        "precision_tiers": list(PRECISION_TIERS),
        "min_kv_cache_headroom_gb": MIN_KV_CACHE_HEADROOM_GB,
        "gpu_tier_specs": tier_specs,
        "hardware_tiers": copy.deepcopy(tier_specs),
        "compute_dtype_by_gpu": compute_dtype_by_gpu,
        "quant_method_required_packages": copy.deepcopy(QUANT_METHOD_REQUIRED_PACKAGES),
        "colab_serving_required_packages": list(COLAB_SERVING_REQUIRED_PACKAGES),
        "tpu_safe_serving_packages": list(TPU_SAFE_SERVING_PACKAGES),
        "models": models_export,
        "hardware_stack_presets": copy.deepcopy(HARDWARE_STACK_PRESETS),
    }


def verify_cross_stage_model_contract(
    model_alias: str,
    gpu_tier: Optional[str] = None,
    *,
    precision: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verifies end-to-end cross-stage producer-consumer closure (Spec 868 §3 & §4) for `model_alias` on `gpu_tier`:
      - Stage 1 (Catalog SSOT): resolves `hf_repo_id`, `quantized_hf_repo_id`, `precision`, `quant_method`, `min_vram_gb`.
      - Stage 2 (Cell 1 Runtime Dependencies): verifies every package required by `quant_method`, `model_type`, and `modality`
        is present in `resolve_required_runtime_packages(...)` and TPU CUDA guards are respected.
      - Stage 3 (Cell 2 Hardware & Compute Dtype): verifies `compute_dtype` (`float16` on `T4` / `sm_75` vs `bfloat16` on `sm_80+`).
      - Stage 4 (Cell 3 Download & Filter): verifies `resolved_checkpoint_repo_id` and `ignore_patterns` (never stripping `.gguf` for GGUF models).
      - Stage 5 (Cell 4 Loader & Modality Dispatch): verifies `loader_kind`, `loader_class_name`, `modality`, and checkpoint `quantization_config`.
    """
    profile = resolve_model_profile(model_alias, config=config)
    canonical_alias = str(profile["alias"])
    modality = str(profile.get("modality", "text"))
    eff_tier = normalize_hardware_tier(
        gpu_tier or str(profile.get("preferred_gpu") or "T4"),
        allow_auto=False,
    )
    tier_spec = GPU_TIER_SPECS[eff_tier]
    accelerator_type = str(tier_spec["accelerator_type"])
    compute_dtype = resolve_compute_dtype_for_hardware(eff_tier)

    def_prec_map = profile.get("default_precision_by_gpu") or profile.get("precision_by_gpu") or {}
    raw_prec = precision or def_prec_map.get(eff_tier) or next(iter(def_prec_map.values()), "Q4")

    if modality == "image":
        q_repos = profile.get("quantized_repo_ids") or {}
        if str(raw_prec).lower() in ("bfloat16", "bf16"):
            resolved_repo = str(profile["hf_repo_id"])
            quant_method = "bfloat16"
        else:
            resolved_repo = str(q_repos.get(str(raw_prec)) or profile.get("quantized_hf_repo_id") or profile["hf_repo_id"])
            quant_method = "gguf" if "gguf" in resolved_repo.lower() or str(raw_prec).lower() == "gguf" else "fp8"
        min_vram_gb = float(profile.get("min_vram_gb", 12.0))
        weights_gb = round(max(2.0, min_vram_gb - MIN_KV_CACHE_HEADROOM_GB), 1)
        norm_prec = "BF16" if quant_method == "bfloat16" else ("Q4" if quant_method == "gguf" else "FP8")
    else:
        var_info = resolve_model_precision_variant(
            canonical_alias,
            precision=raw_prec,
            gpu_tier=eff_tier,
            config=config,
        )
        norm_prec = str(var_info["precision"])
        quant_method = str(var_info["quant_method"])
        resolved_repo = str(var_info["hf_repo_id"])
        min_vram_gb = float(var_info["min_vram_gb"])
        weights_gb = float(var_info["weights_gb"])

    mod_cfg, _, _, _ = _load_local_or_cached_metadata(resolved_repo)
    ckpt_quant_cfg = mod_cfg.get("quantization_config") if isinstance(mod_cfg, dict) else None

    required_pkgs = resolve_required_runtime_packages(
        canonical_alias,
        precision=raw_prec,
        gpu_tier=eff_tier,
        config=config,
    )
    ignore_patterns = resolve_download_ignore_patterns(
        canonical_alias,
        precision=raw_prec,
        quant_method=quant_method,
        config=config,
    )

    is_gguf_repo = "gguf" in resolved_repo.lower() or quant_method == "gguf"
    filter_safe = ("*.gguf" not in ignore_patterns) if is_gguf_repo else ("*.gguf" in ignore_patterns)
    tpu_safe = (
        not any(any( p.startswith(c_pkg) for c_pkg in CUDA_ONLY_PACKAGES) for p in required_pkgs)
        if accelerator_type == "TPU"
        else True
    )

    avail_vram_gb = float(tier_spec["vram_gb"])
    vram_compatible = avail_vram_gb >= (min_vram_gb - 1e-6)

    return {
        "valid": bool(filter_safe and tpu_safe),
        "model_alias": canonical_alias,
        "modality": modality,
        "gpu_tier": eff_tier,
        "accelerator_type": accelerator_type,
        "compute_dtype": compute_dtype,
        "requested_precision": str(raw_prec),
        "normalized_precision": norm_prec,
        "quant_method": quant_method,
        "base_hf_repo_id": str(profile["hf_repo_id"]),
        "quantized_hf_repo_id": str(profile.get("quantized_hf_repo_id") or profile["hf_repo_id"]),
        "resolved_checkpoint_repo_id": resolved_repo,
        "checkpoint_quantization_config": ckpt_quant_cfg,
        "required_packages": required_pkgs,
        "ignore_patterns": ignore_patterns,
        "filter_safe": filter_safe,
        "tpu_safe": tpu_safe,
        "loader_kind": str(profile.get("loader_kind", "causal_lm")),
        "loader_class_name": str(profile.get("loader_class_name", "AutoModelForCausalLM")),
        "is_qwen35_family": bool(profile.get("is_qwen35_family", False)),
        "available_vram_gb": avail_vram_gb,
        "required_min_vram_gb": min_vram_gb,
        "weights_gb": weights_gb,
        "vram_compatible": vram_compatible,
    }


def verify_all_catalog_cross_stage_contracts(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Audits every catalog model across its preferred and supported hardware tiers (Spec 868 §5)."""
    catalog = get_model_catalog(config)
    results: Dict[str, Dict[str, Any]] = {}
    violations: List[str] = []
    all_valid = True
    for alias, prof in catalog.items():
        preferred_tier = str(prof.get("preferred_gpu") or "T4")
        contract = verify_cross_stage_model_contract(alias, preferred_tier, config=config)
        results[alias] = contract
        if not contract["valid"] or not contract["vram_compatible"]:
            all_valid = False
            violations.append(f"{alias}@{preferred_tier}: valid={contract['valid']}, vram_compatible={contract['vram_compatible']}")
    return {
        "ok": all_valid,
        "all_valid": all_valid,
        "model_count": len(results),
        "models_checked": len(results),
        "tiers_checked": list(ALL_HARDWARE_TIERS),
        "violations": violations,
        "contracts": results,
    }


def _extract_token_str(token_entry: Any) -> Optional[str]:
    if token_entry is None:
        return None
    if isinstance(token_entry, str):
        return token_entry
    if isinstance(token_entry, dict):
        return token_entry.get("content")
    return str(token_entry)


def introspect_model_protocol(
    hf_repo_id_or_local_dir: str,
    token: Optional[str] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    chat_template_override: Optional[str] = None,
    *,
    local_dir: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
    allow_remote: bool = False,
    allow_network: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Deterministically inspects tokenizer_config.json, chat_template.jinja, config.json, and
    generation_config.json to produce an architecture-neutral ModelProtocolProfile (FR-3.3, AC-6, Spec 862).
    """
    if allow_network is not None:
        allow_remote = bool(allow_network)
    target = (hf_repo_id_or_local_dir or "Qwen/Qwen3.5-9B").strip()
    custom_entry: Dict[str, Any] = {}
    if isinstance(config, dict):
        custom_map = (config.get("models") or {}).get("custom") or {}
        if isinstance(custom_map, dict) and target in custom_map and isinstance(custom_map[target], dict):
            custom_entry = custom_map[target]

    effective_repo = str(custom_entry.get("hf_repo_id") or target)
    canonical_key = _REPO_METADATA_ALIASES.get(
        effective_repo,
        _REPO_METADATA_ALIASES.get(effective_repo.lower(), effective_repo),
    )

    eff_local_arg = local_dir or custom_entry.get("local_dir")
    local_path = Path(eff_local_arg).expanduser() if eff_local_arg else Path(target).expanduser()
    is_local_dir = False if (not eff_local_arg and target in BUILTIN_MODEL_CATALOG) else local_path.is_dir()
    local_mtime_ns = 0
    if is_local_dir:
        for fname in ("config.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
            fp = local_path / fname
            if fp.is_file():
                try:
                    local_mtime_ns = max(local_mtime_ns, fp.stat().st_mtime_ns)
                except OSError:
                    pass

    can_cache = (
        tokenizer_config is None
        and model_config is None
        and generation_config is None
        and chat_template_override is None
        and not custom_entry
    )
    hf_cache_env_sig = f"{os.environ.get('HF_HUB_CACHE', '')}|{os.environ.get('HF_HOME', '')}"
    cache_key = (target, str(eff_local_arg or ""), bool(allow_remote), local_mtime_ns, hf_cache_env_sig)
    if can_cache:
        with _PROTOCOL_CACHE_LOCK:
            cached_prof = _PROTOCOL_INTROSPECTION_CACHE.get(cache_key)
            if cached_prof is not None:
                return copy.deepcopy(cached_prof)

    tok_cfg: Dict[str, Any] = dict(tokenizer_config) if isinstance(tokenizer_config, dict) else {}
    mod_cfg: Dict[str, Any] = dict(model_config) if isinstance(model_config, dict) else {}
    gen_cfg: Dict[str, Any] = dict(generation_config) if isinstance(generation_config, dict) else {}
    jinja_text: str = chat_template_override or ""

    # Tier 1: Local directory on disk
    if is_local_dir:
        if not tok_cfg:
            tok_file = local_path / "tokenizer_config.json"
            if tok_file.is_file():
                try:
                    tok_cfg = json.loads(tok_file.read_text(encoding="utf-8"))
                except Exception:
                    tok_cfg = {}
        if not mod_cfg:
            cfg_file = local_path / "config.json"
            if cfg_file.is_file():
                try:
                    mod_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                except Exception:
                    mod_cfg = {}
        if not gen_cfg:
            gen_file = local_path / "generation_config.json"
            if gen_file.is_file():
                try:
                    gen_cfg = json.loads(gen_file.read_text(encoding="utf-8"))
                except Exception:
                    gen_cfg = {}
        jinja_file = local_path / "chat_template.jinja"
        if not jinja_text and jinja_file.is_file():
            try:
                jinja_text = jinja_file.read_text(encoding="utf-8")
            except Exception:
                jinja_text = ""

    # Tier 2: Local HuggingFace cache snapshot (single snapshot dir stat per repo candidate, zero network)
    if not (tok_cfg and mod_cfg) and not is_local_dir:
        for repo_cand in dict.fromkeys([canonical_key, effective_repo]):
            if "/" in repo_cand and not repo_cand.startswith((".", "/")):
                snap_dir = _find_local_hf_snapshot_dir(repo_cand)
                if snap_dir is None:
                    continue
                if not tok_cfg:
                    tok_snap = snap_dir / "tokenizer_config.json"
                    if tok_snap.is_file():
                        try:
                            tok_cfg = json.loads(tok_snap.read_text(encoding="utf-8"))
                        except Exception:
                            tok_cfg = {}
                if not mod_cfg:
                    cfg_snap = snap_dir / "config.json"
                    if cfg_snap.is_file():
                        try:
                            mod_cfg = json.loads(cfg_snap.read_text(encoding="utf-8"))
                        except Exception:
                            mod_cfg = {}
                if not gen_cfg:
                    gen_snap = snap_dir / "generation_config.json"
                    if gen_snap.is_file():
                        try:
                            gen_cfg = json.loads(gen_snap.read_text(encoding="utf-8"))
                        except Exception:
                            gen_cfg = {}
                if not jinja_text:
                    jinja_snap = snap_dir / "chat_template.jinja"
                    if jinja_snap.is_file():
                        try:
                            jinja_text = jinja_snap.read_text(encoding="utf-8")
                        except Exception:
                            jinja_text = ""
                if tok_cfg or mod_cfg:
                    break

    # Tier 3: Bundled ground-truth metadata snapshot (< 0.1 ms offline fast-path)
    bundled_entry = BUNDLED_MODEL_METADATA.get(canonical_key)
    if bundled_entry is None:
        for b_key, b_val in BUNDLED_MODEL_METADATA.items():
            if b_key.lower() == canonical_key.lower():
                bundled_entry = b_val
                break
    if bundled_entry is not None:
        if not tok_cfg:
            tok_cfg = dict(bundled_entry.get("tokenizer_config", {}))
        if not mod_cfg:
            mod_cfg = dict(bundled_entry.get("model_config", {}))
        if not gen_cfg:
            gen_cfg = dict(bundled_entry.get("generation_config", {}))

    # Tier 4: Optional remote HuggingFace Hub download only when explicitly allowed
    if allow_remote and not tok_cfg and not mod_cfg and "/" in effective_repo and not is_local_dir:
        try:
            import huggingface_hub

            eff_token = token or os.environ.get("PLATODOOM_HF_TOKEN") or os.environ.get("HF_TOKEN")
            for fname, dest in (
                ("tokenizer_config.json", tok_cfg),
                ("config.json", mod_cfg),
                ("generation_config.json", gen_cfg),
            ):
                try:
                    dl_path = huggingface_hub.hf_hub_download(
                        repo_id=effective_repo,
                        filename=fname,
                        token=eff_token,
                        etag_timeout=5,
                    )
                    dest.update(json.loads(Path(dl_path).read_text(encoding="utf-8")))
                except Exception:
                    pass
            if not jinja_text and not tok_cfg.get("chat_template"):
                try:
                    dl_jinja = huggingface_hub.hf_hub_download(
                        repo_id=effective_repo,
                        filename="chat_template.jinja",
                        token=eff_token,
                        etag_timeout=5,
                    )
                    jinja_text = Path(dl_jinja).read_text(encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            pass

    if custom_entry:
        if custom_entry.get("model_type") and not mod_cfg.get("model_type"):
            mod_cfg["model_type"] = custom_entry["model_type"]
        if custom_entry.get("architecture") and not mod_cfg.get("architectures"):
            mod_cfg["architectures"] = [custom_entry["architecture"]]

    effective_tok_cfg = dict(tok_cfg)
    if jinja_text and not effective_tok_cfg.get("chat_template"):
        effective_tok_cfg["chat_template"] = jinja_text

    family_info = classify_model_family_neutral(
        canonical_key,
        model_config=mod_cfg or None,
        tokenizer_config=effective_tok_cfg or None,
    )
    architecture_family = str(family_info["architecture_family"])
    architecture = str(family_info["architecture"])
    model_type = str(family_info["model_type"])
    loader_kind = str(family_info["loader_kind"])
    loader_class_name = str(family_info["loader_class_name"])
    is_qwen35_family = bool(family_info["is_qwen35_family"])
    vocab_size = family_info["vocab_size"]
    quantization_config = (
        copy.deepcopy(mod_cfg["quantization_config"])
        if isinstance(mod_cfg.get("quantization_config"), dict)
        else None
    )

    chat_template = jinja_text or str(tok_cfg.get("chat_template") or "")
    bos_token = _extract_token_str(tok_cfg.get("bos_token")) or family_info["bos_token"]
    eos_token = _extract_token_str(tok_cfg.get("eos_token")) or family_info["eos_token"]
    pad_token = _extract_token_str(tok_cfg.get("pad_token")) or family_info["pad_token"]
    template_family = str(custom_entry.get("template_family") or family_info["template_family"])
    target_lower = f"{target} {canonical_key}".lower()

    raw_eos = gen_cfg.get("eos_token_id") if gen_cfg.get("eos_token_id") is not None else mod_cfg.get("eos_token_id")

    if template_family == "deepseek_r1_distill":
        if not chat_template:
            chat_template = DEEPSEEK_R1_DISTILL_CHAT_TEMPLATE_JINJA
        if not bos_token:
            bos_token = "<｜begin▁of▁sentence｜>"
        if not eos_token:
            eos_token = "<｜end▁of▁sentence｜>"

        native_jinja_tools = bool(
            ("if tools" in chat_template or "for tool in tools" in chat_template)
            and "<tools>" in chat_template
        )
        requires_dict_tool_arguments = "arguments|items" in chat_template
        serialize_tools_in_system_prompt = not native_jinja_tools
        # DeepSeek-R1-Distill template ends with `{% if add_generation_prompt and not ns.is_tool %}<｜Assistant｜><think>\n{% endif %}`
        # so role=="tool" leaves ns.is_tool=True and omits `<｜Assistant｜><think>\n` unless serialized as user `<tool_response>`!
        serialize_tool_role_as_user_xml = (
            "not ns.is_tool" in chat_template
            or (
                'role == "tool"' not in chat_template
                and "role == 'tool'" not in chat_template
                and "['role'] == 'tool'" not in chat_template
            )
        )
        supports_enable_thinking_kwarg = "enable_thinking" in chat_template
        forces_think_prefix = ("<think>" in chat_template and not supports_enable_thinking_kwarg) or True
        if isinstance(raw_eos, int):
            eos_token_ids = [raw_eos]
        elif isinstance(raw_eos, list) and raw_eos:
            eos_token_ids = [int(x) for x in raw_eos]
        else:
            eos_token_ids = list(family_info["eos_token_ids"])

        temp = float(gen_cfg.get("temperature", 0.6))
        if temp <= 0.0:
            temp = 0.6
        top_p = float(gen_cfg.get("top_p", 0.95))
        recommended_sampling: Dict[str, Any] = {
            "do_sample": True,
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": 8192,
            "thinking": {"do_sample": True, "temperature": temp, "top_p": top_p, "max_tokens": 8192},
            "non_thinking": {"do_sample": True, "temperature": temp, "top_p": top_p, "max_tokens": 8192},
        }
        emitted_tool_formats = [
            "qwen_json_tag",
            "deepseek_native_tool_call",
            "qwen_xml_function",
            "openai_tool_calls",
        ]
    elif template_family in ("qwen3_chatml", "qwen2_chatml"):
        if not chat_template and template_family == "qwen3_chatml":
            chat_template = QWEN35_CHAT_TEMPLATE_JINJA
        if not eos_token:
            eos_token = "<|im_end|>"
        if not pad_token:
            pad_token = "<|endoftext|>"

        native_jinja_tools = bool(
            ("tools" in chat_template)
            and ("<tools>" in chat_template or "tool_call" in chat_template)
        )
        requires_dict_tool_arguments = ("arguments|items" in chat_template) or (template_family == "qwen3_chatml")
        serialize_tools_in_system_prompt = not native_jinja_tools
        supports_tool_role_in_jinja = (
            'message.role == "tool"' in chat_template
            or "message.role == 'tool'" in chat_template
            or 'message["role"] == "tool"' in chat_template
            or "message['role'] == 'tool'" in chat_template
        )
        serialize_tool_role_as_user_xml = not supports_tool_role_in_jinja if chat_template else False
        supports_enable_thinking_kwarg = (
            ("enable_thinking" in chat_template) or (template_family == "qwen3_chatml")
        )
        forces_think_prefix = "<think>" in chat_template and not supports_enable_thinking_kwarg

        if isinstance(raw_eos, list) and raw_eos:
            eos_token_ids = [int(x) for x in raw_eos]
        elif isinstance(raw_eos, int):
            eos_token_ids = [raw_eos]
        else:
            eos_token_ids = list(family_info["eos_token_ids"])

        default_temp = 1.0 if ("35b" in target_lower or "3.6" in target_lower) else (0.2 if template_family == "qwen2_chatml" else 0.6)
        think_temp = float(gen_cfg.get("temperature", default_temp))
        top_p = float(gen_cfg.get("top_p", 0.8 if template_family == "qwen2_chatml" else 0.95))
        top_k = int(gen_cfg.get("top_k", 20))
        recommended_sampling = {
            "do_sample": bool(gen_cfg.get("do_sample", True)),
            "temperature": think_temp,
            "top_p": top_p,
            "top_k": top_k,
            "thinking": {
                "do_sample": True,
                "temperature": think_temp,
                "top_p": top_p,
                "top_k": top_k,
            },
            "non_thinking": {
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
            },
        }
        emitted_tool_formats = [
            "qwen_xml_function",
            "qwen_json_tag",
            "openai_tool_calls",
        ]
    else:
        # Neutral DeepSeek-V2/V3, Llama, Mistral, or unknown standard causal_lm family:
        # zero Qwen3.5 template, architecture, or 248046/248044 token assumptions.
        native_jinja_tools = bool(
            ("tools" in chat_template)
            and ("<tools>" in chat_template or "tool_call" in chat_template)
        )
        requires_dict_tool_arguments = "arguments|items" in chat_template
        serialize_tools_in_system_prompt = not native_jinja_tools
        supports_tool_role_in_jinja = any(
            marker in chat_template
            for marker in (
                'message.role == "tool"',
                "message.role == 'tool'",
                'message["role"] == "tool"',
                "message['role'] == 'tool'",
                'role == "tool"',
                "role == 'tool'",
            )
        )
        serialize_tool_role_as_user_xml = not supports_tool_role_in_jinja
        supports_enable_thinking_kwarg = "enable_thinking" in chat_template
        forces_think_prefix = "<think>" in chat_template and not supports_enable_thinking_kwarg

        if isinstance(raw_eos, list) and raw_eos:
            eos_token_ids = [int(x) for x in raw_eos]
        elif isinstance(raw_eos, int):
            eos_token_ids = [raw_eos]
        else:
            eos_token_ids = list(family_info["eos_token_ids"])

        temp = float(gen_cfg.get("temperature", 0.3))
        top_p = float(gen_cfg.get("top_p", 0.95))
        top_k = int(gen_cfg.get("top_k", 20))
        recommended_sampling = {
            "do_sample": bool(gen_cfg.get("do_sample", temp > 0.0)),
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "thinking": {"do_sample": True, "temperature": max(temp, 0.6), "top_p": top_p, "top_k": top_k},
            "non_thinking": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 1},
        }
        emitted_tool_formats = [
            "qwen_json_tag",
            "openai_tool_calls",
        ]

    if isinstance(vocab_size, int) and vocab_size > 0:
        eos_token_ids = [tid for tid in eos_token_ids if isinstance(tid, int) and 0 <= tid < vocab_size]

    raw_pad_id = gen_cfg.get("pad_token_id") if gen_cfg.get("pad_token_id") is not None else mod_cfg.get("pad_token_id")
    pad_token_id: Optional[int] = None
    if isinstance(raw_pad_id, int) and not isinstance(raw_pad_id, bool):
        if raw_pad_id >= 0 and (not isinstance(vocab_size, int) or vocab_size <= 0 or raw_pad_id < vocab_size):
            pad_token_id = raw_pad_id
    if pad_token_id is None and eos_token_ids:
        pad_token_id = eos_token_ids[0]

    default_enable_thinking = bool(template_family == "deepseek_r1_distill" or forces_think_prefix)

    supports_system_role = (
        "system" in chat_template
        or template_family in ("qwen3_chatml", "qwen2_chatml", "deepseek_r1_distill", "deepseek_v2_chat", "generic_chat")
    )
    default_tok_cls = "Qwen2Tokenizer" if architecture_family.startswith("qwen") else "PreTrainedTokenizerFast"

    result = {
        "model_alias": target if target in BUILTIN_MODEL_CATALOG else canonical_key,
        "model_id": canonical_key,
        "hf_repo_id": canonical_key,
        "architecture": architecture,
        "model_type": model_type,
        "architecture_family": architecture_family,
        "loader_kind": loader_kind,
        "loader_class_name": loader_class_name,
        "is_qwen35_family": is_qwen35_family,
        "vocab_size": vocab_size,
        "quantization_config": quantization_config,
        "tokenizer_class": tok_cfg.get("tokenizer_class", default_tok_cls),
        "template_family": template_family,
        "native_jinja_tools": native_jinja_tools,
        "requires_dict_tool_arguments": requires_dict_tool_arguments,
        "serialize_tools_in_system_prompt": serialize_tools_in_system_prompt,
        "serialize_tool_role_as_user_xml": serialize_tool_role_as_user_xml,
        "supports_tool_role": not serialize_tool_role_as_user_xml,
        "supports_system_role": supports_system_role,
        "supports_enable_thinking_kwarg": supports_enable_thinking_kwarg,
        "forces_think_prefix": forces_think_prefix,
        "default_enable_thinking": default_enable_thinking,
        "tool_result_serialization": (
            "user_turn_tool_response_xml" if serialize_tool_role_as_user_xml else "native_tool_role"
        ),
        "emitted_tool_formats": emitted_tool_formats,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "pad_token_id": pad_token_id,
        "eos_token_ids": eos_token_ids,
        "stop_token_ids": eos_token_ids,
        "stop_strings": [eos_token] if eos_token else [],
        "recommended_sampling": recommended_sampling,
        "sampling_defaults": {
            "temperature": recommended_sampling.get("temperature", 0.6),
            "top_p": recommended_sampling.get("top_p", 0.95),
            "top_k": recommended_sampling.get("top_k", 20),
            "eos_token_id": eos_token_ids,
        },
        "chat_template": chat_template,
    }
    if can_cache:
        with _PROTOCOL_CACHE_LOCK:
            _PROTOCOL_INTROSPECTION_CACHE[cache_key] = copy.deepcopy(result)
    return result


# ---------------------------------------------------------------------------
# Thinking Trace & Multi-Turn Prompt Compilation (FR-3.3, FR-3.4, AC-6)
# ---------------------------------------------------------------------------

THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)


def strip_prior_turn_thinking(text: Optional[str]) -> str:
    """
    Strips completed <think>...</think> blocks (or implicit leading reasoning before </think>)
    from prior assistant turns so reasoning traces do not bloat subsequent turn prompts (FR-3.4).
    """
    if not text:
        return ""
    cleaned = THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    return cleaned.strip()


def split_thinking_and_content(
    raw_text: str,
    forces_think_prefix: bool = False,
) -> Tuple[Optional[str], str]:
    """
    Extracts `<think>...</think>` (or implicit leading thinking before `</think>` when the
    generation prompt already ended with `<think>\\n`) into `(reasoning_content, clean_content)`.
    """
    if not raw_text:
        return None, ""

    explicit_matches = re.findall(r"<think>([\s\S]*?)</think>", raw_text, flags=re.IGNORECASE)
    if explicit_matches:
        reasoning = "\n\n".join(m.strip() for m in explicit_matches if m.strip())
        remainder = THINK_BLOCK_RE.sub("", raw_text).strip()
        return (reasoning or None), remainder

    if "</think>" in raw_text:
        think_part, rest = raw_text.split("</think>", 1)
        think_part = re.sub(r"^\s*<think>\s*", "", think_part, flags=re.IGNORECASE).strip()
        return (think_part or None), rest.strip()

    if raw_text.lstrip().startswith("<think>"):
        # Unclosed <think> block (e.g. truncated output or streaming in-progress)
        think_part = re.sub(r"^\s*<think>\s*", "", raw_text.lstrip(), flags=re.IGNORECASE).strip()
        return (think_part or None), ""

    return None, raw_text


def _normalize_tool_arguments_dict(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        stripped = raw_args.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except Exception:
            return {"raw": raw_args}
    return {}


def compile_model_chat_messages(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    protocol_profile: Optional[Dict[str, Any]] = None,
    enable_thinking: Optional[bool] = None,
    *,
    protocol: Optional[Dict[str, Any]] = None,
    custom_sampling: Optional[Dict[str, Any]] = None,
    output_format: str = "openai_http",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compiles OpenAI-format multi-turn messages + tools into a normalized message list
    and request kwargs that render deterministically without Jinja crashes or dropped tool
    outputs across both Qwen3.5/3.6 ChatML and DeepSeek-R1-Distill-Qwen models (FR-3.3, AC-6).

    - `output_format="openai_http"` (default for vLLM / OpenAI HTTP servers): emits standard OpenAI
      `tool_calls[].function.arguments` as a JSON-encoded `str` and omits extra top-level `arguments`/`name`
      keys so Pydantic request validation never fails with `HTTP 422`.
    - `output_format="hf_jinja"`: emits `dict` `arguments` + top-level `arguments`/`name` for direct local
      HuggingFace Jinja `QWEN35_CHAT_TEMPLATE_JINJA` rendering (`tool_call.arguments|items`).
    """
    # Support positional caller passing `(messages, protocol_dict, tools=...)` as well as `(messages, tools_list, protocol_profile)`
    if (
        isinstance(tools, dict)
        and protocol_profile is None
        and protocol is None
        and ("template_family" in tools or "model_id" in tools or "hf_repo_id" in tools)
    ):
        protocol_profile = tools
        tools = None

    profile = protocol_profile or protocol or introspect_model_protocol("qwen-9b")
    family = profile.get("template_family", "qwen3_chatml")
    use_jinja_dict_args = str(output_format or "openai_http").strip().lower() == "hf_jinja"
    inject_tools_in_prompt = bool(profile.get("serialize_tools_in_system_prompt", False))
    tool_as_user_xml = bool(profile.get("serialize_tool_role_as_user_xml", False))

    normalized: List[Dict[str, Any]] = []
    total_msgs = len(messages)

    for idx, msg in enumerate(messages):
        role = str(msg.get("role", "user"))
        content = msg.get("content")
        is_prior_assistant = role == "assistant" and idx < total_msgs - 1

        if role == "assistant" and isinstance(content, str):
            content = strip_prior_turn_thinking(content) if is_prior_assistant else strip_prior_turn_thinking(content)
            # Also strip duplicate BOS tokens if accidentally included inside content
            bos_tok = profile.get("bos_token")
            if bos_tok and isinstance(bos_tok, str) and bos_tok in content:
                content = content.replace(bos_tok, "").strip()

        if role == "assistant" and msg.get("tool_calls"):
            raw_calls = msg.get("tool_calls") or []
            if family == "deepseek_r1_distill" or inject_tools_in_prompt:
                # Defect 2 Remediation: serialize assistant tool_calls directly into content
                # as <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call> and omit raw tool_calls key.
                serialized_calls: List[str] = []
                for tc in raw_calls:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    fn = fn if isinstance(fn, dict) else {}
                    name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else None) or "unknown"
                    args_dict = _normalize_tool_arguments_dict(
                        fn.get("arguments", tc.get("arguments", {}) if isinstance(tc, dict) else {})
                    )
                    call_json = json.dumps({"name": name, "arguments": args_dict}, ensure_ascii=False)
                    serialized_calls.append(f"<tool_call>\n{call_json}\n</tool_call>")
                combined_content = "\n".join(filter(None, [content or "", *serialized_calls])).strip()
                normalized.append({"role": "assistant", "content": combined_content})
            else:
                norm_calls: List[Dict[str, Any]] = []
                for tc_idx, tc in enumerate(raw_calls):
                    tc_copy = dict(tc) if isinstance(tc, dict) else {}
                    fn_copy = dict(tc_copy.get("function") or {})
                    name = fn_copy.get("name") or tc_copy.get("name") or "unknown"
                    raw_args = fn_copy.get("arguments", tc_copy.get("arguments", {}))
                    args_dict = _normalize_tool_arguments_dict(raw_args)
                    call_id = str(tc_copy.get("id") or f"call_{idx}_{tc_idx}")
                    if use_jinja_dict_args:
                        fn_copy["name"] = name
                        fn_copy["arguments"] = args_dict
                        tc_copy["id"] = call_id
                        tc_copy["type"] = "function"
                        tc_copy["name"] = name
                        tc_copy["function"] = fn_copy
                        tc_copy["arguments"] = args_dict
                        norm_calls.append(tc_copy)
                    else:
                        if isinstance(raw_args, str):
                            try:
                                json.loads(raw_args)
                                args_json = raw_args
                            except Exception:
                                args_json = json.dumps(args_dict, ensure_ascii=False)
                        else:
                            args_json = json.dumps(args_dict, ensure_ascii=False)
                        norm_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": str(name),
                                    "arguments": args_json,
                                },
                            }
                        )
                normalized.append(
                    {
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": norm_calls,
                    }
                )
        elif role == "tool" and tool_as_user_xml:
            # Defect 3 Remediation for DeepSeek-R1-Distill: convert role=="tool" into role=="user"
            # wrapped in <tool_response>...</tool_response> (merging consecutive tool responses)
            # so ns.is_tool is False and `<｜Assistant｜><think>\n` is always emitted!
            tool_name = msg.get("name") or msg.get("tool_call_id") or ""
            header = f"[tool={tool_name}]\n" if tool_name else ""
            wrapped = f"<tool_response>\n{header}{content if content is not None else ''}\n</tool_response>"
            if (
                normalized
                and normalized[-1].get("role") == "user"
                and "<tool_response>" in str(normalized[-1].get("content", ""))
            ):
                normalized[-1]["content"] = f"{normalized[-1]['content']}\n{wrapped}"
            else:
                normalized.append({"role": "user", "content": wrapped})
        else:
            entry = dict(msg)
            clean_c = content if content is not None else ""
            bos_tok = profile.get("bos_token")
            if bos_tok and isinstance(bos_tok, str) and isinstance(clean_c, str) and bos_tok in clean_c:
                clean_c = clean_c.replace(bos_tok, "").strip()
            entry["content"] = clean_c
            if role == "assistant" and "reasoning_content" in entry and is_prior_assistant:
                entry.pop("reasoning_content", None)
            normalized.append(entry)

    # Consolidate multiple system messages into a single leading system message (and prepend
    # `### System Instructions` header on deepseek_r1_distill when multiple system blocks are merged)
    system_contents: List[str] = []
    non_system_msgs: List[Dict[str, Any]] = []
    for m in normalized:
        if m.get("role") == "system":
            c_txt = str(m.get("content") or "").strip()
            if c_txt:
                system_contents.append(c_txt)
        else:
            non_system_msgs.append(m)

    if system_contents:
        merged_sys = "\n\n".join(system_contents)
        if len(system_contents) > 1 and "### System Instructions" not in merged_sys:
            merged_sys = f"### System Instructions\n{merged_sys}"
        normalized = [{"role": "system", "content": merged_sys}, *non_system_msgs]
    else:
        normalized = non_system_msgs

    # Defect 1 Remediation for DeepSeek-R1-Distill: deterministically inject <tools> block into system prompt
    if tools and inject_tools_in_prompt:
        tools_json_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        tool_prompt_block = (
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_json_lines}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        )
        if normalized and normalized[0].get("role") == "system":
            if "<tools>" not in str(normalized[0].get("content", "")):
                normalized[0]["content"] = str(normalized[0].get("content", "")).rstrip() + tool_prompt_block
        else:
            normalized.insert(0, {"role": "system", "content": tool_prompt_block.strip()})

    # Fold system message if a template lacks system role support
    if not profile.get("supports_system_role", True) and normalized and normalized[0].get("role") == "system":
        sys_text = str(normalized.pop(0).get("content", "")).strip()
        if sys_text:
            if normalized and normalized[0].get("role") == "user":
                normalized[0]["content"] = f"{sys_text}\n\n{normalized[0].get('content', '')}"
            else:
                normalized.insert(0, {"role": "user", "content": sys_text})

    # Ensure Qwen3.5/3.6 Jinja invariant: at least one real user query (not purely <tool_response>) exists
    has_real_user_query = any(
        m.get("role") == "user"
        and not (
            str(m.get("content", "")).strip().startswith("<tool_response>")
            and str(m.get("content", "")).strip().endswith("</tool_response>")
        )
        for m in normalized
    )
    if not has_real_user_query and normalized:
        insert_idx = 1 if normalized[0].get("role") == "system" else 0
        normalized.insert(insert_idx, {"role": "user", "content": "Continue with the task."})

    # Build sampling & template kwargs envelope (strictly bounded by vocab_size, zero cross-vocab IDs)
    rec_sampling = profile.get("recommended_sampling") or {}
    raw_profile_stops = profile.get("eos_token_ids")
    vocab_bound = profile.get("vocab_size")
    resolved_stops: List[int] = []
    if isinstance(raw_profile_stops, list):
        for tid in raw_profile_stops:
            if isinstance(tid, int) and not isinstance(tid, bool) and tid >= 0:
                if not isinstance(vocab_bound, int) or vocab_bound <= 0 or tid < vocab_bound:
                    resolved_stops.append(tid)
    extra_kwargs: Dict[str, Any] = {
        "stop_token_ids": resolved_stops,
    }

    default_thinking = bool(
        profile.get("default_enable_thinking", family == "deepseek_r1_distill" or profile.get("forces_think_prefix", False))
    )
    effective_thinking = default_thinking if enable_thinking is None else bool(enable_thinking)

    if profile.get("forces_think_prefix") or family == "deepseek_r1_distill":
        if enable_thinking is not None:
            extra_kwargs["chat_template_kwargs"] = {"enable_thinking": effective_thinking}
        # Never force greedy temperature=0.0 on DeepSeek-R1-Distill when <think> is active
        temp = float(rec_sampling.get("temperature", 0.6))
        if temp <= 0.0:
            temp = 0.6
        extra_kwargs["temperature"] = temp
        extra_kwargs["top_p"] = float(rec_sampling.get("top_p", 0.95))
        extra_kwargs["do_sample"] = True
        extra_kwargs["max_tokens"] = int(rec_sampling.get("max_tokens", 8192))
    elif profile.get("supports_enable_thinking_kwarg") or enable_thinking is not None:
        extra_kwargs["chat_template_kwargs"] = {"enable_thinking": effective_thinking}
        if effective_thinking:
            think_cfg = rec_sampling.get("thinking") or rec_sampling
            extra_kwargs["temperature"] = float(think_cfg.get("temperature", 0.6))
            extra_kwargs["top_p"] = float(think_cfg.get("top_p", 0.95))
            extra_kwargs["top_k"] = int(think_cfg.get("top_k", 20))
            extra_kwargs["do_sample"] = True
        else:
            non_think_cfg = rec_sampling.get("non_thinking") or {}
            extra_kwargs["temperature"] = float(non_think_cfg.get("temperature", 0.0))
            extra_kwargs["top_p"] = float(non_think_cfg.get("top_p", 1.0))
            extra_kwargs["do_sample"] = False

    if isinstance(custom_sampling, dict) and custom_sampling:
        for k, v in custom_sampling.items():
            if v is not None:
                extra_kwargs[k] = v

    return normalized, extra_kwargs


# ---------------------------------------------------------------------------
# Universal 4-Format Tool-Call Response Parser (FR-3.3, FR-4.4, AC-6)
# ---------------------------------------------------------------------------

QWEN_XML_TOOL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*([\s\S]*?)\s*</function>\s*</tool_call>",
    re.IGNORECASE,
)
QWEN_XML_PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\n?([\s\S]*?)\n?</parameter>",
    re.IGNORECASE,
)
JSON_TOOL_TAG_RE = re.compile(
    r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>",
    re.IGNORECASE,
)
DEEPSEEK_TOOL_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?:function<｜tool▁sep｜>)?([^\n<]+)\n(?:```(?:json)?\n)?([\s\S]*?)(?:\n```)?<｜tool▁call▁end｜>"
)
DEEPSEEK_TOOL_WRAPPER_RE = re.compile(r"</?｜tool▁calls▁(?:begin|end)｜>")
FENCED_TOOL_BLOCK_RE = re.compile(
    r"```(?:tool_call|json)\s*\n(\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:[\s\S]*?\})\s*\n```",
    re.IGNORECASE,
)


def _coerce_xml_param_value(raw_val: str) -> Any:
    trimmed = raw_val.strip()
    if not trimmed:
        return ""
    if trimmed in ("true", "false", "null") or (
        (trimmed[0] in "[{\"" or trimmed[0].isdigit() or (trimmed[0] == "-" and len(trimmed) > 1 and trimmed[1].isdigit()))
    ):
        try:
            return json.loads(trimmed)
        except Exception:
            pass
    return raw_val.strip("\n")


def parse_universal_tool_calls(
    text: str,
    openai_tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parses tool calls across all 4 model output syntaxes into canonical dictionaries:
      {
        "id": "call_...",
        "type": "function",
        "name": "<fn_name>",
        "arguments": {...},
        "raw_arguments": "<json_str>",
        "function": {"name": "<fn_name>", "arguments": "<json_str>"},
        "malformed_error": Optional[str],
      }
    Returns (cleaned_text_without_tool_tags_or_think, normalized_tool_calls).
    """
    _, cleaned = split_thinking_and_content(text or "")
    parsed_calls: List[Dict[str, Any]] = []

    def _build_entry(
        name: str,
        args_dict: Dict[str, Any],
        call_id: Optional[str] = None,
        malformed_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        cid = call_id or f"call_{uuid.uuid4().hex[:8]}_{len(parsed_calls)}"
        raw_json = json.dumps(args_dict, ensure_ascii=False)
        return {
            "id": cid,
            "type": "function",
            "name": name,
            "arguments": args_dict,
            "raw_arguments": raw_json,
            "function": {
                "name": name,
                "arguments": raw_json,
            },
            "malformed_error": malformed_error,
        }

    # Format 1: OpenAI native tool_calls array
    if openai_tool_calls:
        for tc in openai_tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(fn.get("name") or tc.get("name") or "").strip()
            if not name:
                continue
            raw_args = fn.get("arguments", tc.get("arguments", {}))
            malformed: Optional[str] = None
            if isinstance(raw_args, dict):
                args_dict = raw_args
            elif isinstance(raw_args, str):
                try:
                    loaded = json.loads(raw_args) if raw_args.strip() else {}
                    args_dict = loaded if isinstance(loaded, dict) else {"value": loaded}
                except Exception as exc:
                    args_dict = {"raw": raw_args}
                    malformed = f"Malformed JSON in tool arguments for '{name}': {exc}"
            else:
                args_dict = {}
            parsed_calls.append(_build_entry(name, args_dict, call_id=tc.get("id"), malformed_error=malformed))

    # Format 2: Qwen3.5 / Qwen3.6 native XML `<tool_call><function=NAME><parameter=KEY>VAL</parameter></function></tool_call>`
    def _replace_qwen_xml(match: re.Match[str]) -> str:
        fn_name = match.group(1).strip()
        body = match.group(2)
        args_dict: Dict[str, Any] = {}
        for p_match in QWEN_XML_PARAM_RE.finditer(body):
            p_key = p_match.group(1).strip()
            p_val = _coerce_xml_param_value(p_match.group(2))
            args_dict[p_key] = p_val
        parsed_calls.append(_build_entry(fn_name, args_dict))
        return ""

    cleaned = QWEN_XML_TOOL_RE.sub(_replace_qwen_xml, cleaned)

    # Format 3: Qwen / Hermes / R1 prompt-injected JSON `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
    def _replace_json_tag(match: re.Match[str]) -> str:
        raw_json = match.group(1).strip()
        try:
            payload = json.loads(raw_json)
            if isinstance(payload, dict) and payload.get("name"):
                name = str(payload["name"]).strip()
                raw_args = payload.get("arguments", payload.get("parameters", {}))
                args_dict = _normalize_tool_arguments_dict(raw_args)
                parsed_calls.append(_build_entry(name, args_dict))
                return ""
        except Exception as exc:
            name_m = re.search(r'"name"\s*:\s*"([^"]+)"', raw_json)
            fn_name = name_m.group(1) if name_m else "malformed_tool_call"
            parsed_calls.append(
                _build_entry(
                    fn_name,
                    {"raw": raw_json},
                    malformed_error=f"Malformed JSON inside <tool_call>: {exc}",
                )
            )
            return ""
        return match.group(0)

    cleaned = JSON_TOOL_TAG_RE.sub(_replace_json_tag, cleaned)

    # Format 4A: DeepSeek native `<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\n{...}\n```<｜tool▁call▁end｜>`
    def _replace_deepseek_tag(match: re.Match[str]) -> str:
        fn_name = match.group(1).strip()
        raw_args = match.group(2).strip()
        try:
            loaded = json.loads(raw_args) if raw_args else {}
            args_dict = loaded if isinstance(loaded, dict) else {"value": loaded}
            parsed_calls.append(_build_entry(fn_name, args_dict))
        except Exception as exc:
            parsed_calls.append(
                _build_entry(
                    fn_name,
                    {"raw": raw_args},
                    malformed_error=f"Malformed JSON in DeepSeek tool call '{fn_name}': {exc}",
                )
            )
        return ""

    cleaned = DEEPSEEK_TOOL_CALL_RE.sub(_replace_deepseek_tag, cleaned)
    cleaned = DEEPSEEK_TOOL_WRAPPER_RE.sub("", cleaned)

    # Format 4B: Fenced ```tool_call / ```json blocks with {"name": ..., "arguments": ...}
    if not parsed_calls:
        def _replace_fenced(match: re.Match[str]) -> str:
            raw_json = match.group(1).strip()
            try:
                payload = json.loads(raw_json)
                if isinstance(payload, dict) and payload.get("name") and "arguments" in payload:
                    name = str(payload["name"]).strip()
                    args_dict = _normalize_tool_arguments_dict(payload.get("arguments"))
                    parsed_calls.append(_build_entry(name, args_dict))
                    return ""
            except Exception:
                pass
            return match.group(0)

        cleaned = FENCED_TOOL_BLOCK_RE.sub(_replace_fenced, cleaned)

    return cleaned.strip(), parsed_calls


# ---------------------------------------------------------------------------
# Context Window Compaction at 80% Watermark (FR-3.4)
# ---------------------------------------------------------------------------

def _estimate_message_tokens(msg: Dict[str, Any]) -> int:
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning_content") or "")
    tool_calls = msg.get("tool_calls")
    tc_chars = len(json.dumps(tool_calls, ensure_ascii=False)) if tool_calls else 0
    return max(1, (len(content) + len(reasoning) + tc_chars) // 4 + 4)


def estimate_messages_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Estimates total token count across a conversation message array and optional tools schema."""
    total = sum(_estimate_message_tokens(m) for m in (messages or []) if isinstance(m, dict))
    if tools:
        try:
            total += max(1, len(json.dumps(tools, ensure_ascii=False)) // 4)
        except Exception:
            pass
    return total


def compact_conversation_messages(
    messages: List[Dict[str, Any]],
    max_model_len: int = 16384,
    threshold_ratio: float = 0.80,
    keep_recent_turns: int = 6,
) -> List[Dict[str, Any]]:
    """
    1. Strips completed <think>...</think> blocks from all prior assistant messages.
    2. When estimated prompt tokens exceed `threshold_ratio * max_model_len` (default 80%),
       deterministically compacts older tool outputs and intermediate assistant turns while
       preserving the system prompt (Zero-Turn Workspace Compass) and the most recent turns (FR-3.4).
    """
    if not messages:
        return []

    budget_tokens = max(512, int(max_model_len * threshold_ratio))
    compacted: List[Dict[str, Any]] = []
    last_idx = len(messages) - 1

    for idx, msg in enumerate(messages):
        m_copy = copy.deepcopy(msg)
        if m_copy.get("role") == "assistant" and isinstance(m_copy.get("content"), str) and idx < last_idx:
            m_copy["content"] = strip_prior_turn_thinking(m_copy["content"])
            m_copy.pop("reasoning_content", None)
        compacted.append(m_copy)

    def _total_tokens(msgs: List[Dict[str, Any]]) -> int:
        return sum(_estimate_message_tokens(m) for m in msgs)

    if _total_tokens(compacted) <= budget_tokens:
        return compacted

    # Pass 1: Truncate older tool outputs (preserving the last `keep_recent_turns` messages intact)
    protected_tail = min(max(1, int(keep_recent_turns)), len(compacted))
    cutoff_idx = max(1, len(compacted) - protected_tail)

    for idx in range(1, cutoff_idx):
        if _total_tokens(compacted) <= budget_tokens:
            break
        role = compacted[idx].get("role")
        content = str(compacted[idx].get("content") or "")
        is_tool_msg = role == "tool" or (role == "user" and "<tool_response>" in content)
        if is_tool_msg and len(content) > 320:
            head = content[:180].rstrip()
            tail = content[-80:].lstrip()
            compacted[idx]["content"] = (
                f"{head}\n...[COMPACTED {len(content)} CHARS OF HISTORICAL TOOL OUTPUT FOR CONTEXT BUDGET]...\n{tail}"
            )

    # Pass 2: Truncate older verbose assistant/user messages if still above budget
    for idx in range(1, cutoff_idx):
        if _total_tokens(compacted) <= budget_tokens:
            break
        content = str(compacted[idx].get("content") or "")
        if len(content) > 400:
            compacted[idx]["content"] = (
                content[:240].rstrip()
                + f"\n...[COMPACTED {len(content)} CHARS OF HISTORICAL TURN]..."
            )

    # Pass 3: Summarize oldest middle turns into a compact summary message if still over budget
    if _total_tokens(compacted) > budget_tokens and len(compacted) > protected_tail + 1:
        has_system = compacted[0].get("role") == "system"
        head_msgs = [compacted[0]] if has_system else []
        start_mid = 1 if has_system else 0
        tail_keep = compacted[-protected_tail:]
        middle_msgs = compacted[start_mid:-protected_tail]
        summary_lines = [
            f"[COMPACTED HISTORY SUMMARY ({len(middle_msgs)} earlier turns compacted at 80% of {max_model_len} window)]"
        ]
        for m in middle_msgs[:8]:
            r = m.get("role", "user")
            snippet = str(m.get("content") or "").replace("\n", " ")[:80]
            summary_lines.append(f"- {r}: {snippet}")
        summary_msg = {"role": "user", "content": "\n".join(summary_lines)}
        compacted = head_msgs + [summary_msg] + tail_keep

    return compacted
