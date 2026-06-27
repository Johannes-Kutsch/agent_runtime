from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Literal, get_args, get_type_hints

import pytest

import agent_runtime as runtime
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
