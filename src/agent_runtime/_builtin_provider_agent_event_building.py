from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from ._runtime_lifecycle import AgentEvent


def _raw_event_payload(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _render_tool_call_display_message(tool_name: str, payload: str) -> str:
    if payload:
        return f"{tool_name}({payload})"
    return tool_name


def _agent_message(line: str, text: str) -> AgentEvent:
    return AgentEvent(
        type="agent_message",
        display_message=text,
        raw_provider_output=line,
    )


def _agent_tool_call(line: str, tool_name: str, payload: str) -> AgentEvent:
    return AgentEvent(
        type="agent_tool_call",
        display_message=_render_tool_call_display_message(tool_name, payload),
        raw_provider_output=line,
    )


def _other(line: str, descriptor: str) -> AgentEvent:
    return AgentEvent(
        type="other",
        display_message=descriptor,
        raw_provider_output=line,
    )


def _turn_summary(line: str, summary: str) -> AgentEvent:
    return AgentEvent(
        type="turn_summary",
        display_message=summary,
        raw_provider_output=line,
    )


def _raw_text_fallback(line: str) -> AgentEvent:
    return _other(line, line.strip())


def _summary_field_join(*fields: str | None) -> str:
    populated = [field for field in fields if field]
    if populated:
        return " | ".join(populated)
    return "turn_summary"


def _format_duration_ms(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"duration_ms={value}"
    return None


def _format_cost_usd(value: object) -> str | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"cost_usd={value}"
    return None


def _format_token_count(label: str, value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{label}={value}"
    return None


def _build_claude_assistant_event(
    line: str,
    event: dict[str, object],
) -> AgentEvent | None:
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    tool_blocks: list[dict[str, object]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        elif block_type == "tool_use":
            tool_blocks.append(cast(dict[str, object], block))
    if text_parts:
        return _agent_message(line, "\n\n".join(text_parts))
    if not tool_blocks:
        return None
    first_tool = tool_blocks[0]
    tool_name = first_tool.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        tool_name = "tool_use"
    payload_value: object = (
        first_tool.get("input")
        if len(tool_blocks) == 1 and first_tool.get("input") is not None
        else tool_blocks
    )
    return _agent_tool_call(
        line,
        tool_name,
        _raw_event_payload(payload_value),
    )


def _build_claude_result_event(line: str, event: dict[str, object]) -> AgentEvent:
    subtype = event.get("subtype")
    stop_reason = (
        f"stop_reason={subtype}" if isinstance(subtype, str) and subtype else None
    )
    return _turn_summary(
        line,
        _summary_field_join(
            stop_reason,
            _format_duration_ms(event.get("duration_ms")),
            _format_cost_usd(event.get("total_cost_usd")),
        ),
    )


def _build_claude_object_event(line: str, event: dict[str, object]) -> AgentEvent:
    event_type = event.get("type")
    if event_type == "assistant":
        assistant_event = _build_claude_assistant_event(line, event)
        if assistant_event is not None:
            return assistant_event
    if event_type == "system":
        descriptor = _render_claude_system_display_message(event)
        if descriptor is not None:
            return _other(line, descriptor)
    if event_type == "result":
        return _build_claude_result_event(line, event)
    descriptor = event_type if isinstance(event_type, str) and event_type else "other"
    return _other(line, descriptor)


def build_claude_agent_event(line: str) -> AgentEvent:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return _raw_text_fallback(line)
    if not isinstance(event, dict):
        return _other(line, "non_object")
    return _build_claude_object_event(line, event)


def claude_built_in_provider_agent_event_builder() -> Callable[[str], AgentEvent]:
    return build_claude_agent_event


def _render_claude_system_display_message(event: dict[str, object]) -> str | None:
    subtype = event.get("subtype")
    if not isinstance(subtype, str) or not subtype:
        return None
    if subtype == "system.init":
        cwd = event.get("cwd")
        if isinstance(cwd, str) and cwd:
            return f"{subtype} cwd={cwd}"
    if subtype == "system.thinking_tokens":
        estimated_tokens = event.get("estimated_tokens")
        if isinstance(estimated_tokens, int) and not isinstance(estimated_tokens, bool):
            return f"{subtype} tokens={estimated_tokens}"
    return subtype


def _codex_tool_payload(item: dict[str, object]) -> str:
    for key in ("arguments", "input", "payload"):
        value = item.get(key)
        if value is not None:
            return _raw_event_payload(value)
    return _raw_event_payload(item)


def build_codex_agent_event(line: str) -> AgentEvent:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return _raw_text_fallback(line)
    if not isinstance(event, dict):
        return _other(line, "non_object")
    event_type = event.get("type")
    if event_type in {"item.completed", "item.started"}:
        item = event.get("item")
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "agent_message":
                content = item.get("text")
                if content is None:
                    content = item.get("content") or ""
                if isinstance(content, str):
                    return _agent_message(line, content)
            if isinstance(item_type, str):
                tool_name = item.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    tool_name = item_type
                return _agent_tool_call(line, tool_name, _codex_tool_payload(item))
    if event_type == "turn.completed":
        usage = event.get("usage")
        usage_dict = usage if isinstance(usage, dict) else {}
        return _turn_summary(
            line,
            _summary_field_join(
                _format_token_count("input_tokens", usage_dict.get("input_tokens")),
                _format_token_count("cached_tokens", usage_dict.get("cached_tokens")),
                _format_token_count("output_tokens", usage_dict.get("output_tokens")),
            ),
        )
    descriptor = event_type if isinstance(event_type, str) and event_type else "other"
    return _other(line, descriptor)


def codex_built_in_provider_agent_event_builder() -> Callable[[str], AgentEvent]:
    return build_codex_agent_event


def build_opencode_agent_event(line: str) -> AgentEvent:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return _raw_text_fallback(line)
    if not isinstance(event, dict):
        return _other(line, "non_object")
    if event.get("type") == "text":
        part = event.get("part")
        if isinstance(part, dict):
            part_type = part.get("type")
            if part_type == "text":
                time = part.get("time")
                if isinstance(time, dict) and time.get("end") is not None:
                    text = part.get("text")
                    if isinstance(text, str):
                        stripped = text.strip()
                        if stripped:
                            return _agent_message(line, stripped)
            if part_type == "tool":
                tool_name = part.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    tool_name = "tool"
                payload_value = (
                    part.get("input")
                    if part.get("input") is not None
                    else part.get("text", part)
                )
                return _agent_tool_call(
                    line,
                    tool_name,
                    _raw_event_payload(payload_value),
                )
    if event.get("type") == "step_finish":
        step = event.get("step")
        step_dict = step if isinstance(step, dict) else {}
        tokens = step_dict.get("tokens")
        token_dict = tokens if isinstance(tokens, dict) else {}
        cache = token_dict.get("cache")
        cache_dict = cache if isinstance(cache, dict) else {}
        return _turn_summary(
            line,
            _summary_field_join(
                _format_token_count("input", token_dict.get("input")),
                _format_token_count("output", token_dict.get("output")),
                _format_token_count("reasoning", token_dict.get("reasoning")),
                _format_token_count("cache_read", cache_dict.get("read")),
                _format_token_count("cache_write", cache_dict.get("write")),
                _format_cost_usd(step_dict.get("cost")),
            ),
        )
    if event.get("type") == "session.status":
        status = event.get("status")
        descriptor = "session.status"
        if isinstance(status, dict):
            status_type = status.get("type")
            if isinstance(status_type, str):
                descriptor = status_type
        return _other(line, descriptor)
    event_type = event.get("type")
    descriptor = event_type if isinstance(event_type, str) and event_type else "other"
    return _other(line, descriptor)
