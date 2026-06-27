from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Literal, get_args, get_type_hints

import pytest

import agent_runtime as runtime
from agent_runtime._builtin_provider_stream_interpretation import (
    claude_built_in_provider_stream_interpretation,
    codex_built_in_provider_stream_interpretation,
    opencode_built_in_provider_stream_interpretation,
)
import agent_runtime._provider_invocation as provider_invocation_runtime
from tests.runtime_client_execution_harness import RuntimeClientExecutionHarness


def _codex_message_output_line(text: str) -> str:
    return (
        json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
        )
        + "\n"
    )


def _run_completed_codex_ephemeral(monkeypatch, tmp_path: Path):
    line = _codex_message_output_line("hello")
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(line,)
        )
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(monkeypatch, tmp_path)
    return asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=runtime.ProviderSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )


def test_resolved_provider_is_credential_free_triple() -> None:
    selected = runtime.ResolvedProvider(service="claude", model="haiku", effort="low")

    assert selected.service == "claude"
    assert selected.model == "haiku"
    assert selected.effort == "low"
    # Credential-free: no auth attribute can leak ProviderAuth.
    assert not hasattr(selected, "auth")
    with pytest.raises(FrozenInstanceError):
        setattr(selected, "service", "codex")


def test_agent_event_collapses_to_three_fields() -> None:
    assert {field.name for field in fields(runtime.AgentEvent)} == {
        "type",
        "display_message",
        "raw_provider_output",
    }
    event = runtime.AgentEvent(
        type="agent_message",
        display_message="hello",
        raw_provider_output="raw",
    )
    with pytest.raises(FrozenInstanceError):
        setattr(event, "display_message", "changed")


def test_agent_event_type_includes_turn_summary_public_vocabulary() -> None:
    type_hint = get_type_hints(runtime.AgentEvent, globalns={"Literal": Literal})[
        "type"
    ]

    assert get_args(type_hint) == (
        "agent_message",
        "agent_tool_call",
        "turn_summary",
        "other",
    )


def _claude_message_line(text: str) -> str:
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
        )
        + "\n"
    )


def _claude_tool_line(name: str, tool_input: dict[str, object]) -> str:
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": name, "input": tool_input}]
                },
            }
        )
        + "\n"
    )


def _codex_message_line(text: str) -> str:
    return (
        json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
        )
        + "\n"
    )


def _opencode_tool_line(name: str, tool_input: dict[str, object]) -> str:
    return (
        json.dumps(
            {
                "type": "text",
                "part": {"type": "tool", "name": name, "input": tool_input},
            }
        )
        + "\n"
    )


def _built_in_provider_event(service_name: str, line: str) -> runtime.AgentEvent:
    if service_name == "claude":
        return claude_built_in_provider_stream_interpretation().build_agent_event(line)
    if service_name == "codex":
        return codex_built_in_provider_stream_interpretation().build_agent_event(line)
    if service_name == "opencode":
        return opencode_built_in_provider_stream_interpretation().build_agent_event(
            line
        )
    raise AssertionError(f"unexpected service {service_name!r}")


def test_message_lines_render_display_message_text_for_each_provider() -> None:
    for service, line in (
        ("claude", _claude_message_line("hi from claude")),
        ("codex", _codex_message_line("hi from codex")),
    ):
        event = _built_in_provider_event(service, line)
        assert event.type == "agent_message"
        assert event.display_message == f"hi from {service}"
        assert event.raw_provider_output == line


def test_tool_call_lines_render_tool_identity_and_arguments() -> None:
    claude_tool = _claude_tool_line("Read", {"path": "README.md"})
    event = _built_in_provider_event("claude", claude_tool)
    assert event.type == "agent_tool_call"
    assert "Read" in event.display_message
    assert '{"path":"README.md"}' in event.display_message
    assert event.raw_provider_output == claude_tool

    opencode_tool = _opencode_tool_line("Grep", {"q": "needle"})
    event = _built_in_provider_event("opencode", opencode_tool)
    assert event.type == "agent_tool_call"
    assert "Grep" in event.display_message
    assert '{"q":"needle"}' in event.display_message


def test_claude_tool_only_lines_preserve_tool_payload_shape() -> None:
    line = (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {"path": "a.md"}},
                        {
                            "type": "tool_use",
                            "name": "Write",
                            "input": {"path": "b.md"},
                        },
                    ]
                },
            }
        )
        + "\n"
    )

    event = _built_in_provider_event("claude", line)

    assert event.type == "agent_tool_call"
    assert event.display_message == (
        'Read([{"type":"tool_use","name":"Read","input":{"path":"a.md"}},'
        '{"type":"tool_use","name":"Write","input":{"path":"b.md"}}])'
    )
    assert event.raw_provider_output == line


def test_other_lines_render_neutral_descriptor_as_display_message() -> None:
    line = '{"type":"thread.started","thread_id":"t-1"}\n'
    event = _built_in_provider_event("codex", line)
    assert event.type == "other"
    assert event.display_message == "thread.started"
    assert event.raw_provider_output == line


def test_completed_ephemeral_run_is_completed_kind_with_run_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcome = _run_completed_codex_ephemeral(monkeypatch, tmp_path)

    assert isinstance(outcome.kind, runtime.Completed)
    result = outcome.result
    assert result.output == "hello"
    assert result.continuation is None  # ephemeral
    assert result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    # No finished-run log on the outcome.
    assert not hasattr(outcome, "invocation_records")
    assert not hasattr(outcome, "account_label")
    assert not hasattr(result, "runtime_metadata")
    assert not hasattr(result, "session_namespace")
    assert not hasattr(result, "used_fallback")
    assert not hasattr(result, "selected_service_path")
    assert not hasattr(result, "metadata")
    assert not hasattr(outcome, "used_fallback")
    assert not hasattr(outcome, "selected_service_path")
    assert not hasattr(outcome, "usage_limit_scope")


def test_run_result_has_unified_field_set() -> None:
    assert {field.name for field in fields(runtime.RunResult)} == {
        "output",
        "usage",
        "continuation",
        "selected",
    }


def test_runtime_outcome_is_kind_plus_result_only() -> None:
    assert {field.name for field in fields(runtime.RuntimeOutcome)} == {
        "kind",
        "result",
    }
    outcome = runtime.RuntimeOutcome(
        kind=runtime.UsageLimited(reset_time=None),
        result=runtime.RunResult(
            output="",
            usage=None,
            continuation=None,
            selected=runtime.ResolvedProvider(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
        ),
    )
    assert not hasattr(outcome, "usage_limit_scope")


def test_outcome_kind_variants_carry_only_their_own_data() -> None:
    # reset_time lives only on UsageLimited.
    assert {f.name for f in fields(runtime.UsageLimited)} == {"reset_time"}
    assert {f.name for f in fields(runtime.ProviderUnavailable)} == {
        "reason",
        "detail",
    }
    for variant in (runtime.Completed, runtime.Cancelled, runtime.TimedOut):
        assert fields(variant) == ()
