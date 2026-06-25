from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime._builtin_provider_rendering as builtin_provider_rendering_runtime
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.runtime as prompt_runtime
from tests.runtime_client_execution_harness import RuntimeClientExecutionHarness
from agent_runtime.errors import (
    AgentCancelledError,
    AgentCredentialFailureError,
    HardAgentError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    RuntimeConfigurationError,
    TransientAgentError,
)
from agent_runtime.session import RunKind
from agent_runtime.types import ProviderSelection as InternalStageSelection


def _codex_executable() -> str:
    return "codex.cmd" if os.name == "nt" else "codex"


@dataclasses.dataclass
class _DeterministicTimeoutWatchdog:
    timeout_seconds: int
    timeout_check_numbers: tuple[int, ...]
    check_count: int = 0

    def start_monitoring(self) -> None:
        return None

    def reset_timer(self) -> None:
        return None

    def check_timeout(self) -> None:
        self.check_count += 1
        if self.check_count in self.timeout_check_numbers:
            raise runtime.AgentTimeoutError(
                "Idle timeout: no Agent Event within configured window"
            )

    def stop_monitoring(self) -> None:
        return None


def _install_deterministic_timeout_watchdog(
    monkeypatch: pytest.MonkeyPatch,
    *,
    timeout_check_numbers: tuple[int, ...],
) -> None:
    monkeypatch.setattr(
        prompt_runtime._live_runtime_output_timeout_context_module,
        "_IdleTimeoutWatchdog",
        lambda timeout_seconds: _DeterministicTimeoutWatchdog(
            timeout_seconds=timeout_seconds,
            timeout_check_numbers=timeout_check_numbers,
        ),
    )


def _normalize_tool_policy_profile(
    tool_policy: runtime.ToolPolicy | contracts_runtime.ToolPolicyProfile,
) -> contracts_runtime.ToolPolicyProfile:
    return (
        tool_policy.profile
        if isinstance(tool_policy, runtime.ToolPolicy)
        else tool_policy
    )


def _opencode_tool_access(
    tool_policy: runtime.ToolPolicy | contracts_runtime.ToolPolicyProfile,
    invocation_dir: Path,
) -> contracts_runtime.ToolAccess:
    return (
        contracts_runtime.ToolAccess(
            kind="none",
            workspace=None,
            tool_policy=tool_policy,
        )
        if _normalize_tool_policy_profile(tool_policy)
        == runtime.ToolPolicy.NONE.profile
        else contracts_runtime.ToolAccess.workspace_backed(
            invocation_dir,
            tool_policy=tool_policy,
        )
    )


def _codex_assistant_output_line(text: str) -> str:
    return (
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": text,
                },
            }
        )
        + "\n"
    )


def _claude_assistant_output_line(
    contents: str | tuple[str, ...] | list[str],
) -> str:
    if isinstance(contents, str):
        blocks = [{"type": "text", "text": contents}]
    else:
        blocks = [{"type": "text", "text": text} for text in contents]
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": blocks,
                },
            }
        )
        + "\n"
    )


def _claude_result_output_line(text: str) -> str:
    return json.dumps({"type": "result", "result": text}) + "\n"


def _claude_tool_output_line(name: str, payload: object) -> str:
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": name, "input": payload}],
                },
            }
        )
        + "\n"
    )


def _opencode_text_output_line(text: str, *, session_id: str = "sess_123") -> str:
    return (
        json.dumps(
            {
                "type": "text",
                "timestamp": 1,
                "sessionID": session_id,
                "part": {
                    "type": "text",
                    "text": text,
                    "time": {"start": 1, "end": 2},
                },
            }
        )
        + "\n"
    )


def _opencode_tool_output_line(
    *,
    name: str,
    payload: object,
    session_id: str = "sess_123",
) -> str:
    return (
        json.dumps(
            {
                "type": "text",
                "timestamp": 1,
                "sessionID": session_id,
                "part": {
                    "type": "tool",
                    "name": name,
                    "input": payload,
                    "time": {"start": 1, "end": 2},
                },
            }
        )
        + "\n"
    )


def _opencode_idle_output_line(*, session_id: str = "sess_123") -> str:
    return (
        json.dumps(
            {
                "type": "session.status",
                "timestamp": 2,
                "sessionID": session_id,
                "status": {"type": "idle"},
            }
        )
        + "\n"
    )


def test_runtime_client_ephemeral_run_emits_typed_agent_message_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[runtime.AgentEvent] = []

    def on_live_output(event: runtime.AgentEvent) -> None:
        observed.append(event)

    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        )
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(harness.recorded_requests) == 1
    assert outcome.result.output == "hello\nworld"
    assert len(observed) == 2
    assert observed[0].type == "agent_message"
    assert observed[0].display_message == "hello"
    assert observed[1].type == "agent_message"
    assert observed[1].display_message == "world"


def test_runtime_client_ephemeral_run_event_carries_raw_provider_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[runtime.AgentEvent] = []

    def on_live_output(event: runtime.AgentEvent) -> None:
        observed.append(event)

    hello_line = _codex_assistant_output_line("hello")
    world_line = _codex_assistant_output_line("world")
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(hello_line, world_line),
        )
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(harness.recorded_requests) == 1
    assert outcome.result.output == "hello\nworld"
    assert len(observed) == 2
    assert observed[0].raw_provider_output == hello_line
    assert observed[1].raw_provider_output == world_line
    assert (
        "".join(event.raw_provider_output for event in observed)
        == hello_line + world_line
    )


def test_runtime_client_ephemeral_run_emits_other_agent_event_for_codex_life_sign(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[runtime.AgentEvent] = []

    def on_live_output(event: runtime.AgentEvent) -> None:
        observed.append(event)

    thread_started_line = '{"type":"thread.started","thread_id":"thread-123"}\n'
    message_line = _codex_assistant_output_line("hello")
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(thread_started_line, message_line),
        )
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(harness.recorded_requests) == 1
    assert outcome.result.output == "hello"
    assert [event.type for event in observed] == ["other", "agent_message"]
    assert observed[0].display_message == "thread.started"
    assert observed[0].raw_provider_output == thread_started_line
    assert observed[1].display_message == "hello"
    assert "".join(event.raw_provider_output for event in observed) == (
        thread_started_line + message_line
    )


def test_runtime_client_ephemeral_run_emits_tool_call_and_other_agent_events_for_opencode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[runtime.AgentEvent] = []

    def on_live_output(event: runtime.AgentEvent) -> None:
        observed.append(event)

    tool_line = _opencode_tool_output_line(
        name="Read",
        payload={"path": "README.md"},
    )
    text_line = _opencode_text_output_line("assistant output")
    idle_line = _opencode_idle_output_line()
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(tool_line, text_line, idle_line),
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(harness.recorded_requests) == 1
    assert outcome.result.output == "assistant output"
    assert [event.type for event in observed] == [
        "agent_tool_call",
        "agent_message",
        "other",
    ]
    assert observed[0].display_message == 'Read({"path":"README.md"})'
    assert observed[0].raw_provider_output == tool_line
    assert observed[1].display_message == "assistant output"
    assert observed[2].display_message == "idle"
    assert "".join(event.raw_provider_output for event in observed) == (
        tool_line + text_line + idle_line
    )


def test_runtime_client_runs_claude_new_session_with_runtime_state_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "intermediate"}],
                            "usage": {
                                "input_tokens": 5,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                            },
                        },
                    }
                )
                + "\n",
                json.dumps({"type": "result", "result": "final output"}) + "\n",
            ),
        ),
    )
    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    provider_state_dir_relpath = "implementer/main/claude/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    result = outcome.result
    assert result.output == "final output"
    assert result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert result.usage == runtime.ProviderUsage(
        input_tokens=5,
        output_tokens=None,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cost_usd=None,
        duration_seconds=None,
    )
    assert result.continuation is not None
    assert result.continuation.provider_resume_state == {
        "run_kind": "resume",
        "provider_session_id": "session-uuid",
        "exact_transcript_match": False,
    }
    assert result.continuation.tool_access == contracts_runtime.ToolAccess.no_tools()
    assert len(harness.recorded_requests) == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "session-uuid"
    assert provider_state_dir.is_dir()


def test_runtime_client_new_session_without_runtime_state_dir_returns_meaningful_continuation_without_state_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_generated_provider_session_id(
        monkeypatch,
        "prepared-session-id",
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "provider-session-777",
                        "part": {
                            "type": "text",
                            "text": "hello from opencode",
                            "time": {"end": True},
                        },
                    }
                )
                + "\n",
                json.dumps({"type": "session.status", "status": {"type": "idle"}})
                + "\n",
            ),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="opencode",
                    model="glm-5.2",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(opencode_api_key="opencode-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome.result.output == "hello from opencode"
    session_result = cast(prompt_runtime.RunResult, outcome.result)
    assert session_result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "provider-session-777",
            "provider_state": {"session_id": "provider-session-777"},
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_new_session_still_validates_provider_selection_credentials_invocation_dir_and_tool_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all()

    with pytest.raises(RuntimeConfigurationError):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="unsupported",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(AgentCredentialFailureError):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(RuntimeConfigurationError):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="missing",
                        model="ignored",
                        effort="low",
                    ),
                    provider_auth=runtime.ProviderAuth(
                        opencode_api_key="root-only-key"
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    with pytest.raises(TypeError, match="requires an `invocation_dir` value"):
        prompt_runtime.NewSessionRunRequest(
            prompt="already rendered prompt",
            provider_selection=InternalStageSelection(
                service="opencode",
                model="glm-5.2",
                effort="medium",
            ),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            "NewSessionRunRequest requires an explicit `tool_policy` value."
        ),
    ):
        prompt_runtime.NewSessionRunRequest(
            prompt="already rendered prompt",
            invocation_dir=tmp_path,
            provider_selection=InternalStageSelection(
                service="opencode",
                model="glm-5.2",
                effort="medium",
            ),
        )

    assert len(harness.recorded_requests) == 0


def test_runtime_client_new_session_validation_failure_cleans_runtime_managed_state_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    managed_state_dir = tmp_path / "runtime-managed-state"
    cleaned_up = False

    class _TrackingTemporaryDirectory:
        def __init__(self, *, prefix: str) -> None:
            del prefix
            managed_state_dir.mkdir()
            self.name = str(managed_state_dir)

        def cleanup(self) -> None:
            nonlocal cleaned_up
            cleaned_up = True

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.tempfile,
        "TemporaryDirectory",
        _TrackingTemporaryDirectory,
    )

    with pytest.raises(RuntimeConfigurationError):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="unsupported",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert cleaned_up is True


def test_runtime_client_runs_claude_new_session_and_returns_portable_continuation_for_resumption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "first output"}) + "\n",
            ),
        ),
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "continued output"}) + "\n",
            ),
        ),
    )

    first_outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=harness.prepare_runtime_state_dir(tmp_path),
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert first_outcome.result is not None
    assert isinstance(first_outcome.result, prompt_runtime.RunResult)
    continuation = first_outcome.result.continuation
    assert continuation is not None
    assert continuation.provider_resume_state == {
        "run_kind": "resume",
        "provider_session_id": "session-uuid",
        "exact_transcript_match": False,
    }

    second_outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert isinstance(second_outcome.kind, prompt_runtime.Completed)
    assert second_outcome.result.output == "continued output"
    assert second_outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert second_outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "session-uuid",
            "exact_transcript_match": False,
        },
    )
    assert harness.recorded_request_count == 2
    resumed_request = harness.recorded_request(1)
    assert resumed_request.provider_session_id == "session-uuid"


def test_runtime_client_runs_claude_new_session_through_in_memory_provider_invocation_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    adapter = harness._adapter
    harness.prepare_result(
        provider_invocation_runtime.ProviderInvocationResult(
            output="final output",
            usage=runtime.ProviderUsage(
                input_tokens=5,
                output_tokens=2,
            ),
            stdout_lines=(
                '{"type":"assistant","message":{"content":[{"type":"text","text":"thinking"}]}}\n',
            ),
            provider_session_id="observed-session",
        )
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    request = harness.start_session_run_request(
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        provider_selection=InternalStageSelection(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        provider_auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )
    outcome = prompt_runtime._run_builtin_session_outcome(
        lambda: prompt_runtime._builtin_runtime_client_module._run_builtin_new_session(
            request,
            provider_invocation_adapter=adapter,
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "final output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=5,
        output_tokens=2,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "observed-session",
            "exact_transcript_match": False,
        },
    )
    assert harness.recorded_request_count == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "session-uuid"


def test_runtime_client_runs_opencode_new_session_through_in_memory_provider_invocation_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_generated_provider_session_id(
        monkeypatch,
        "prepared-session-id",
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_result(
        provider_invocation_runtime.ProviderInvocationResult(
            output="final output",
            usage=runtime.ProviderUsage(
                input_tokens=7,
                output_tokens=3,
            ),
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "part": {
                            "type": "text",
                            "time": {"end": True},
                            "text": "final output",
                        },
                    }
                )
                + "\n",
                json.dumps({"type": "session.status", "status": {"type": "idle"}})
                + "\n",
            ),
            provider_session_id="observed-session-id",
        )
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    request = harness.start_session_run_request(
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        provider_selection=InternalStageSelection(
            service="opencode",
            model="glm-5.2",
            effort="medium",
        ),
        provider_auth=runtime.ProviderAuth(opencode_api_key="opencode-key"),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )
    outcome = prompt_runtime._run_builtin_session_outcome(
        lambda: prompt_runtime._builtin_runtime_client_module._run_builtin_new_session(
            request,
            provider_invocation_adapter=harness.provider_invocation_adapter,
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "final output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=7,
        output_tokens=3,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "observed-session-id",
            "provider_state": {"session_id": "observed-session-id"},
            "exact_transcript_match": False,
        },
    )
    assert harness.recorded_request_count == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "prepared-session-id"
    provider_state_dir = runtime_state_dir / "implementer" / "main" / "opencode"
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "observed-session-id"
    )


def test_runtime_client_uses_observed_opencode_new_session_id_over_adapter_and_prepared_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_generated_provider_session_id(
        monkeypatch,
        "prepared-session-id",
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="final output",
            usage=runtime.ProviderUsage(
                input_tokens=7,
                output_tokens=3,
            ),
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "observed-session-id",
                        "part": {
                            "type": "text",
                            "time": {"end": True},
                            "text": "final output",
                        },
                    }
                )
                + "\n",
                json.dumps({"type": "session.status", "status": {"type": "idle"}})
                + "\n",
            ),
            provider_session_id="adapter-session-id",
        ),
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="opencode",
                    model="glm-5.2",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(opencode_api_key="opencode-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "observed-session-id",
            "provider_state": {"session_id": "observed-session-id"},
            "exact_transcript_match": False,
        },
    )
    assert harness.recorded_request().provider_session_id == "prepared-session-id"
    provider_state_dir = runtime_state_dir / "implementer" / "main" / "opencode"
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "observed-session-id"
    )


def test_runtime_client_ephemeral_run_calls_live_output_observer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        )
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert harness.recorded_request_count == 1
    assert outcome.result.output == "hello\nworld"
    assert observed == ["hello", "world"]


def test_runtime_client_ephemeral_run_forwards_live_output_observer_exceptions_as_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)
        raise runtime.UsageLimitError(service_name="codex")

    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        )
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
    )

    with pytest.raises(runtime.UsageLimitError):
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                harness.ephemeral_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    provider_auth=runtime.ProviderAuth(
                        claude_code_oauth_token="oauth-token"
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert observed == ["hello"]


def test_runtime_client_ephemeral_run_propagates_live_output_observer_timeout_exceptions_as_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []
    observer_failure = runtime.AgentTimeoutError("observer timeout")

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)
        raise observer_failure

    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        )
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
    )

    with pytest.raises(runtime.AgentTimeoutError, match="observer timeout") as excinfo:
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                harness.ephemeral_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    provider_auth=runtime.ProviderAuth(
                        claude_code_oauth_token="oauth-token"
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert excinfo.value is observer_failure
    assert observed == ["hello"]


def test_runtime_client_start_session_run_calls_live_output_observer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(monkeypatch, tmp_path)

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert harness.recorded_request_count == 1
    assert outcome.result.output == "hello\nworld"
    assert observed == ["hello", "world"]


def test_runtime_client_start_session_run_observes_current_codex_turns_when_reusing_deduplicated_rollout_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    RuntimeClientExecutionHarness.install_local_codex_host_auth(monkeypatch, tmp_path)
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("current invocation output"),
                json.dumps({"type": "turn.completed"}) + "\n",
            ),
        ),
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    provider_state_dir = harness.prepare_provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    harness.prepare_codex_rollout_state(provider_state_dir, "thread-123", "thread-123")

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(harness.recorded_requests) == 1
    assert harness.recorded_request().run_kind is RunKind.RESUME
    assert outcome.result.output == "current invocation output"
    assert observed == ["current invocation output"]


def test_runtime_client_start_session_run_forwards_live_output_observer_exceptions_as_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)
        raise runtime.UsageLimitError(service_name="codex")

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(monkeypatch, tmp_path)

    with pytest.raises(runtime.UsageLimitError):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert observed == ["hello"]


def test_runtime_client_start_session_run_propagates_live_output_observer_timeout_exceptions_as_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []
    observer_failure = runtime.AgentTimeoutError("observer timeout")

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)
        raise observer_failure

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(monkeypatch, tmp_path)

    with pytest.raises(runtime.AgentTimeoutError, match="observer timeout") as excinfo:
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert excinfo.value is observer_failure
    assert observed == ["hello"]


def test_runtime_client_start_session_run_propagates_live_output_observer_cancellation_exceptions_as_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []
    observer_failure = AgentCancelledError()

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)
        raise observer_failure

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    RuntimeClientExecutionHarness.install_local_codex_host_auth(monkeypatch, tmp_path)

    with pytest.raises(AgentCancelledError) as excinfo:
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert excinfo.value is observer_failure
    assert observed == ["hello"]


def test_runtime_client_resumed_session_run_calls_live_output_observer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )

    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={"provider_session_id": "resume-session"},
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
                on_live_output=on_live_output,
            )
        )
    )

    assert len(harness.recorded_requests) == 1
    assert outcome.result.output == "hello\nworld"
    assert observed == ["hello", "world"]


def test_runtime_client_resumed_session_run_forwards_live_output_observer_exceptions_as_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)
        raise ProviderUnavailableError(
            service_name="codex",
            message="observer failure",
            reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
        )

    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )

    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={"provider_session_id": "resume-session"},
    )

    with pytest.raises(ProviderUnavailableError):
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                RuntimeClientExecutionHarness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    continuation=continuation,
                    on_live_output=on_live_output,
                )
            )
        )

    assert observed == ["hello"]


def test_runtime_client_resumed_session_run_propagates_live_output_observer_timeout_exceptions_as_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []
    observer_failure = runtime.AgentTimeoutError("observer timeout")

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)
        raise observer_failure

    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )

    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={"provider_session_id": "resume-session"},
    )

    with pytest.raises(runtime.AgentTimeoutError, match="observer timeout") as excinfo:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                RuntimeClientExecutionHarness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    continuation=continuation,
                    on_live_output=on_live_output,
                )
            )
        )

    assert excinfo.value is observer_failure
    assert observed == ["hello"]


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        (
            ProviderUnavailableReason.SERVICE_NOT_AVAILABLE,
            "No configured service candidates are currently available.",
        ),
        (
            ProviderUnavailableReason.TRANSIENT_API_ERROR,
            "temporary provider failure",
        ),
    ],
)
def test_runtime_client_new_session_maps_provider_unavailable_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: ProviderUnavailableReason,
    detail: str,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE,
            detail=detail,
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=harness.prepare_runtime_state_dir(tmp_path),
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome.kind == prompt_runtime.ProviderUnavailable(
        reason=reason,
        detail=detail,
    )
    assert outcome.result.output == ""
    assert outcome.result.usage is None
    assert outcome.result.continuation is None
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude",
        model="sonnet",
        effort="medium",
    )


def test_runtime_client_ephemeral_run_calls_live_output_observer_for_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(_claude_assistant_output_line(("  hello ", "world")),),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert outcome.result.output == "hello\n\nworld"
    assert observed == [
        "hello\n\nworld",
    ]


def test_runtime_client_new_session_run_calls_live_output_observer_for_resumed_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _claude_assistant_output_line("intermediate"),
                _claude_result_output_line("final output"),
            ),
        ),
    )
    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    provider_state_dir = harness.prepare_provider_state_dir(
        runtime_state_dir,
        service="claude",
    )
    (provider_state_dir / "session-state.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(harness.recorded_requests) == 1
    assert outcome.result.output == "final output"
    assert observed == [
        "intermediate",
    ]


def test_runtime_client_ephemeral_run_emits_claude_tool_call_and_other_agent_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[runtime.AgentEvent] = []

    def on_live_output(event: runtime.AgentEvent) -> None:
        observed.append(event)

    tool_line = _claude_tool_output_line("Read", {"path": "README.md"})
    result_line = _claude_result_output_line("final output")
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(tool_line, result_line),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert outcome.result.output == "final output"
    assert [event.type for event in observed] == ["agent_tool_call", "other"]
    assert observed[0].display_message == 'Read({"path":"README.md"})'
    assert observed[0].raw_provider_output == tool_line
    assert observed[1].display_message == "result"
    assert observed[1].raw_provider_output == result_line
    assert "".join(event.raw_provider_output for event in observed) == (
        tool_line + result_line
    )


@pytest.mark.parametrize("run_mode", ("ephemeral", "new_session", "resumed_session"))
def test_runtime_client_claude_live_runtime_output_matches_final_parser_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_mode: str,
) -> None:
    observed: list[runtime.AgentEvent] = []

    def on_live_output(event: runtime.AgentEvent) -> None:
        observed.append(event)

    assistant_line = _claude_assistant_output_line("intermediate")
    tool_line = _claude_tool_output_line("Read", {"path": "README.md"})
    result_line = _claude_result_output_line("final output")
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(assistant_line, tool_line, result_line),
        ),
    )

    client = runtime.RuntimeClient()
    if run_mode == "ephemeral":
        outcome = asyncio.run(
            client.run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                        InternalStageSelection(
                            service="claude",
                            model="sonnet",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                    ),
                    on_live_output=on_live_output,
                    tool_policy=runtime.ToolPolicy.NONE,
                )
            )
        )
    elif run_mode == "new_session":
        monkeypatch.setattr(
            prompt_runtime._builtin_runtime_client_module,
            "_new_provider_session_id",
            lambda: "session-uuid",
        )
        outcome = asyncio.run(
            client.run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=harness.prepare_runtime_state_dir(tmp_path),
                    provider_selection=InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    provider_auth=runtime.ProviderAuth(
                        claude_code_oauth_token="oauth-token"
                    ),
                    on_live_output=on_live_output,
                    tool_policy=runtime.ToolPolicy.NONE,
                )
            )
        )
    else:
        outcome = asyncio.run(
            client.run_resumed_session(
                harness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    provider_auth=runtime.ProviderAuth(
                        claude_code_oauth_token="oauth-token"
                    ),
                    continuation=harness.claude_continuation(
                        provider_session_id="session-uuid"
                    ),
                    on_live_output=on_live_output,
                )
            )
        )

    assert outcome.result.output == "final output"
    assert [event.type for event in observed] == [
        "agent_message",
        "agent_tool_call",
        "other",
    ]
    assert [event.display_message for event in observed] == [
        "intermediate",
        'Read({"path":"README.md"})',
        "result",
    ]
    assert [event.raw_provider_output for event in observed] == [
        assistant_line,
        tool_line,
        result_line,
    ]
    assert len(harness.recorded_requests) == 1


def test_runtime_client_new_session_run_propagates_claude_live_output_observer_failure_for_resumed_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(_claude_assistant_output_line("intermediate"),),
        ),
    )
    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    provider_state_dir = harness.prepare_provider_state_dir(
        runtime_state_dir,
        service="claude",
    )
    (provider_state_dir / "session-state.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )

    def on_live_output(_turn: runtime.AgentEvent) -> None:
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    provider_selection=InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    provider_auth=runtime.ProviderAuth(
                        claude_code_oauth_token="oauth-token"
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert len(harness.recorded_requests) == 1


def test_runtime_client_new_opencode_session_calls_live_runtime_output_observer_once_per_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "provider-session-777",
                        "part": {
                            "type": "text",
                            "text": " hello from opencode ",
                            "time": {"end": True},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "provider-session-777",
                        "part": {
                            "type": "text",
                            "text": "second turn",
                            "time": {"end": True},
                        },
                    }
                )
                + "\n",
                json.dumps({"type": "session.status", "status": {"type": "idle"}})
                + "\n",
            ),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="opencode",
                    model="glm-5.2",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(opencode_api_key="opencode-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert outcome.result.output == "hello from opencode\n\nsecond turn"
    assert observed == [
        "hello from opencode",
        "second turn",
    ]


@pytest.mark.parametrize("run_mode", ("ephemeral", "new_session", "resumed_session"))
def test_runtime_client_opencode_live_runtime_output_matches_final_parser_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_mode: str,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1,
                        "sessionID": "sess_123",
                        "part": {
                            "id": "part_1",
                            "sessionID": "sess_123",
                            "messageID": "msg_1",
                            "type": "text",
                            "text": "hello",
                            "time": {"start": 1, "end": 2},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 2,
                        "sessionID": "sess_123",
                        "part": {
                            "id": "part_2",
                            "sessionID": "sess_123",
                            "messageID": "msg_1",
                            "type": "text",
                            "text": "second",
                            "time": {"start": 2, "end": 3},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "session.status",
                        "timestamp": 3,
                        "sessionID": "sess_123",
                        "status": {"type": "idle"},
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 4,
                        "sessionID": "sess_123",
                        "part": {
                            "id": "part_3",
                            "sessionID": "sess_123",
                            "messageID": "msg_1",
                            "type": "text",
                            "text": "should be ignored",
                            "time": {"start": 4, "end": 5},
                        },
                    }
                )
                + "\n",
            ),
        ),
    )

    client = runtime.RuntimeClient()
    if run_mode == "ephemeral":
        outcome = asyncio.run(
            client.run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                        InternalStageSelection(
                            service="opencode",
                            model="kimi-k2.6",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(opencode_api_key="go-key"),
                    ),
                    on_live_output=on_live_output,
                    tool_policy=runtime.ToolPolicy.NONE,
                )
            )
        )
    elif run_mode == "new_session":
        outcome = asyncio.run(
            client.run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )
    else:
        outcome = asyncio.run(
            client.run_resumed_session(
                harness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                    continuation=harness.opencode_continuation(
                        model="kimi-k2.6",
                        provider_session_id="sess_123",
                    ),
                    on_live_output=on_live_output,
                )
            )
        )

    assert outcome.result.output == "hello\n\nsecond"
    assert observed == ["hello", "second"]
    assert len(harness.recorded_requests) == 1


def test_runtime_client_opencode_live_runtime_output_stops_after_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1,
                        "sessionID": "sess_123",
                        "part": {
                            "id": "part_1",
                            "sessionID": "sess_123",
                            "messageID": "msg_1",
                            "type": "text",
                            "text": "hello",
                            "time": {"start": 1, "end": 2},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 2,
                        "sessionID": "sess_123",
                        "error": {
                            "name": "InternalServerError",
                            "data": {
                                "message": "provider failed",
                                "statusCode": 503,
                                "isRetryable": True,
                            },
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 3,
                        "sessionID": "sess_123",
                        "part": {
                            "id": "part_2",
                            "sessionID": "sess_123",
                            "messageID": "msg_2",
                            "type": "text",
                            "text": "should not be observed",
                            "time": {"start": 3, "end": 4},
                        },
                    }
                )
                + "\n",
            ),
        ),
    )

    with pytest.raises(TransientAgentError):
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                        InternalStageSelection(
                            service="opencode",
                            model="kimi-k2.6",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(opencode_api_key="go-key"),
                    ),
                    on_live_output=on_live_output,
                    tool_policy=runtime.ToolPolicy.NONE,
                )
            )
        )

    assert observed == ["hello"]


def test_runtime_client_new_opencode_session_observes_live_runtime_output_before_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1,
                        "part": {
                            "id": "part_1",
                            "messageID": "msg_1",
                            "type": "text",
                            "text": "hello before session id",
                            "time": {"start": 1, "end": 2},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 2,
                        "sessionID": "sess_123",
                        "part": {
                            "id": "part_2",
                            "sessionID": "sess_123",
                            "messageID": "msg_1",
                            "type": "text",
                            "text": "hello after session id",
                            "time": {"start": 2, "end": 3},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "session.status",
                        "timestamp": 3,
                        "sessionID": "sess_123",
                        "status": {"type": "idle"},
                    }
                )
                + "\n",
            ),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    session_result = cast(prompt_runtime.RunResult, outcome.result)
    assert session_result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="kimi-k2.6",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "sess_123",
            "provider_state": {"session_id": "sess_123"},
            "exact_transcript_match": False,
        },
    )

    assert outcome.result.output == "hello before session id\n\nhello after session id"
    assert observed == [
        "hello before session id",
        "hello after session id",
    ]


def test_runtime_client_runs_claude_resumed_session_through_built_in_provider_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="continued output",
            usage=runtime.ProviderUsage(
                input_tokens=7,
                output_tokens=2,
            ),
            provider_session_id="observed-session",
        ),
    )

    continuation = harness.claude_continuation()

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=7,
        output_tokens=2,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "observed-session",
            "exact_transcript_match": False,
        },
    )
    assert harness.recorded_request_count == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.command


def test_runtime_client_runs_claude_resumed_session_from_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "intermediate"}],
                            "usage": {
                                "input_tokens": 7,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 1,
                            },
                        },
                    }
                )
                + "\n",
                json.dumps({"type": "result", "result": "continued output"}) + "\n",
            ),
        ),
    )

    continuation = adapter.claude_continuation()
    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            adapter.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=7,
        output_tokens=None,
        cache_read_input_tokens=1,
        cache_creation_input_tokens=0,
        cost_usd=None,
        duration_seconds=None,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "claude-session-123",
            "exact_transcript_match": False,
        },
    )
    assert adapter.recorded_request_count == 1
    recorded_request = adapter.recorded_request()
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.command


@pytest.mark.parametrize(
    "tool_access",
    [
        contracts_runtime.ToolAccess.no_tools(),
        contracts_runtime.ToolAccess.workspace_backed(
            Path("."), tool_policy=runtime.ToolPolicy.INSPECT_ONLY
        ),
        contracts_runtime.ToolAccess.workspace_backed(
            Path("."), tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION
        ),
        contracts_runtime.ToolAccess.workspace_backed(
            Path("."), tool_policy=runtime.ToolPolicy.UNRESTRICTED
        ),
    ],
)
def test_runtime_client_runs_codex_resumed_session_through_built_in_provider_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_access: contracts_runtime.ToolAccess,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="continued output",
            usage=runtime.ProviderUsage(
                input_tokens=7,
                output_tokens=2,
            ),
            stdout_lines=(
                '{"type":"thread.started","thread_id":"observed-thread"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"continued output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
            provider_session_id="ignored-provider-session-id",
        ),
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    provider_state_dir = harness.prepare_provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    harness.prepare_codex_rollout_state(provider_state_dir, "recovered-thread")
    continuation = harness.codex_continuation(
        tool_access=(
            tool_access
            if tool_access.kind == "none"
            else contracts_runtime.ToolAccess.workspace_backed(
                tmp_path, tool_policy=tool_access.tool_policy
            )
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=7,
        output_tokens=2,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=(
            tool_access
            if tool_access.kind == "none"
            else contracts_runtime.ToolAccess.workspace_backed(
                tmp_path, tool_policy=tool_access.tool_policy
            )
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "observed-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert isinstance(outcome.result, prompt_runtime.RunResult)
    assert harness.recorded_request_count == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "selected-thread"
    assert recorded_request.command.startswith(
        f"{_codex_executable()} exec resume selected-thread -m gpt-5.4 "
    )
    assert (provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
    )


def test_runtime_client_resumes_codex_session_from_completed_new_session_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="initial output",
            usage=runtime.ProviderUsage(
                input_tokens=5,
                output_tokens=2,
            ),
            stdout_lines=(
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"initial output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
            provider_session_id="thread-123",
        ),
        provider_invocation_runtime.ProviderInvocationResult(
            output="continued output",
            usage=runtime.ProviderUsage(
                input_tokens=3,
                output_tokens=1,
            ),
            stdout_lines=(
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"continued output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
            provider_session_id=None,
        ),
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    new_outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(new_outcome.kind, prompt_runtime.Completed)
    assert new_outcome.result.output == "initial output"
    assert new_outcome.result.usage == runtime.ProviderUsage(
        input_tokens=5,
        output_tokens=2,
    )
    assert new_outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert new_outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )

    assert isinstance(new_outcome.result, prompt_runtime.RunResult)
    continuation = new_outcome.result.continuation
    assert continuation is not None

    resumed_outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
            )
        )
    )

    assert isinstance(resumed_outcome.kind, prompt_runtime.Completed)
    assert resumed_outcome.result.output == "continued output"
    assert resumed_outcome.result.usage == runtime.ProviderUsage(
        input_tokens=3,
        output_tokens=1,
    )
    assert resumed_outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert resumed_outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )

    assert harness.recorded_request().command.startswith(
        f"{_codex_executable()} exec -m gpt-5.4"
    )
    assert harness.recorded_request(1).command == (
        f"{_codex_executable()} exec resume thread-123 -m gpt-5.4 -c model_reasoning_effort=medium -c approval_policy=never "
        "--sandbox read-only --json"
    )


def test_runtime_client_runs_codex_resumed_session_from_continuation_without_portable_state_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )

    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="continued output",
            usage=runtime.ProviderUsage(
                input_tokens=3,
                output_tokens=2,
            ),
            stdout_lines=(
                '{"type":"thread.started","thread_id":"selected-thread"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"continued output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
            provider_session_id="selected-thread",
        ),
    )

    continuation = harness.codex_continuation(provider_state_dir_relpath=None)

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=3,
        output_tokens=2,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "selected-thread",
            "exact_transcript_match": False,
        },
    )
    assert harness.recorded_request().environment == {"TZ": "UTC"}
    assert harness.recorded_request().command == (
        f"{_codex_executable()} exec resume selected-thread -m gpt-5.4 -c model_reasoning_effort=medium -c approval_policy=never "
        "--sandbox read-only --json"
    )


def test_runtime_client_preserves_tool_policy_in_resumed_session_usage_limited_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail="Usage limit reached (reset_time=None)",
            usage=runtime.ProviderUsage(
                input_tokens=1,
                output_tokens=1,
            ),
            stdout_lines=(
                '{"type":"assistant","message":{"content":[{"type":"text","text":"continued"}]}}\n',
            ),
            provider_session_id="usage-session-1",
        ),
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail="Usage limit reached (reset_time=None)",
            usage=runtime.ProviderUsage(
                input_tokens=1,
                output_tokens=1,
            ),
            stdout_lines=(
                '{"type":"assistant","message":{"content":[{"type":"text","text":"continued"}]}}\n',
            ),
            provider_session_id="usage-session-2",
        ),
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    provider_state_dir = harness.prepare_provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    harness.prepare_codex_rollout_state(provider_state_dir, "recovered-thread")
    continuation = harness.codex_continuation(
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            tmp_path,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
    )

    first = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
            )
        )
    )
    assert isinstance(first.kind, prompt_runtime.UsageLimited)
    assert first.kind.reset_time is None
    assert first.result.output == ""
    assert first.result.usage == runtime.ProviderUsage(
        input_tokens=1,
        output_tokens=1,
    )
    assert first.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert first.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            tmp_path,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "usage-session-1",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert first.result.continuation is not None
    assert (
        first.result.continuation.tool_access.tool_policy
        == runtime.ToolPolicy.NO_FILE_MUTATION
    )

    second = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=first.result.continuation,
            )
        )
    )
    assert isinstance(second.kind, prompt_runtime.UsageLimited)
    assert second.kind.reset_time is None
    assert second.result.output == ""
    assert second.result.usage == runtime.ProviderUsage(
        input_tokens=1,
        output_tokens=1,
    )
    assert second.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert second.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            tmp_path,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "usage-session-2",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert second.result.continuation is not None
    assert (
        second.result.continuation.tool_access.tool_policy
        == runtime.ToolPolicy.NO_FILE_MUTATION
    )


def test_runtime_client_does_not_store_provider_credentials_in_codex_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )

    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="initial output",
            usage=runtime.ProviderUsage(input_tokens=1, output_tokens=1),
            stdout_lines=(
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"initial output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )
    assert len(adapter.recorded_requests) == 1

    assert isinstance(outcome.result, prompt_runtime.RunResult)
    continuation = outcome.result.continuation
    assert continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert "provider_auth" not in continuation.provider_resume_state
    assert "auth" not in continuation.provider_resume_state
    assert "provider_secret" not in continuation.provider_resume_state
    assert continuation.provider_resume_state["provider_session_id"] == "thread-123"

    assert not any(
        key.startswith("codex") for key in continuation.provider_resume_state.keys()
    )


def test_runtime_client_returns_started_usage_limited_outcome_from_in_memory_provider_invocation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    adapter = provider_invocation_runtime.InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            provider_invocation_runtime.ProviderInvocationFailure(
                kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
                detail="Usage limit reached (reset_time=None)",
                usage=runtime.ProviderUsage(
                    input_tokens=3,
                    output_tokens=1,
                ),
                stdout_lines=(
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"thinking"}]}}\n',
                ),
                provider_session_id="observed-session",
            )
        ]
    )

    outcome = prompt_runtime._run_builtin_session_outcome(
        lambda: prompt_runtime._builtin_runtime_client_module._run_builtin_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            ),
            provider_invocation_adapter=adapter,
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time is None
    assert outcome.result.output == ""
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=3,
        output_tokens=1,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="", effort=""
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "observed-session",
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_keeps_claude_continuation_when_provider_invocation_failure_only_reports_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail="Usage limit reached (reset_time=None)",
            provider_session_id="observed-session",
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "observed-session",
            "exact_transcript_match": False,
        },
    )


@pytest.mark.parametrize(
    ("entrypoint", "failure", "expected_kind"),
    [
        (
            "new",
            provider_invocation_runtime.ProviderInvocationFailure(
                kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
                detail="Usage limit reached (reset_time=None)",
                usage=runtime.ProviderUsage(
                    input_tokens=3,
                    output_tokens=1,
                ),
                stdout_lines=(),
                provider_session_id=None,
            ),
            prompt_runtime.UsageLimited(reset_time=None),
        ),
        (
            "resumed",
            provider_invocation_runtime.ProviderInvocationFailure(
                kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
                detail="Usage limit reached (reset_time=None)",
                usage=runtime.ProviderUsage(
                    input_tokens=3,
                    output_tokens=1,
                ),
                stdout_lines=(),
                provider_session_id=None,
            ),
            prompt_runtime.UsageLimited(reset_time=None),
        ),
        (
            "new",
            provider_invocation_runtime.ProviderInvocationFailure(
                kind=provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE,
                detail="temporary provider failure",
                stdout_lines=(),
                provider_session_id=None,
            ),
            prompt_runtime.ProviderUnavailable(
                reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
                detail="temporary provider failure",
            ),
        ),
        (
            "resumed",
            provider_invocation_runtime.ProviderInvocationFailure(
                kind=provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE,
                detail="temporary provider failure",
                stdout_lines=(),
                provider_session_id=None,
            ),
            prompt_runtime.ProviderUnavailable(
                reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
                detail="temporary provider failure",
            ),
        ),
    ],
)
def test_runtime_client_omits_codex_continuation_for_pre_start_session_backed_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    failure: provider_invocation_runtime.ProviderInvocationFailure,
    expected_kind: prompt_runtime.UsageLimited | prompt_runtime.ProviderUnavailable,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        failure,
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    provider_state_dir = harness.prepare_provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    harness.prepare_codex_rollout_state(provider_state_dir, "recovered-thread")
    if entrypoint == "new":
        outcome = asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    provider_selection=InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )
        expected_recorded_provider_session_id = "recovered-thread"
    else:
        continuation = harness.codex_continuation()

        outcome = asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                harness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                )
            )
        )
        expected_recorded_provider_session_id = "selected-thread"

    assert outcome.kind == expected_kind
    assert outcome.result.output == ""
    assert outcome.result.usage == failure.usage
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.continuation is None
    assert harness.recorded_request_count == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.provider_session_id == expected_recorded_provider_session_id


def test_runtime_client_runs_claude_resumed_session_with_generated_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "generated-session-id",
    )
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "generated output"}) + "\n",
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir_relpath = "implementer/main/claude/"
    continuation = adapter.claude_continuation(
        provider_session_id=None,
        provider_state_dir_relpath=provider_state_dir_relpath,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            adapter.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "generated output"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "generated-session-id",
            "exact_transcript_match": False,
        },
    )
    assert adapter.recorded_request_count == 1
    recorded_request = adapter.recorded_request()
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "generated-session-id"
    assert recorded_request.command


def test_runtime_client_runs_claude_resumed_session_with_generated_provider_session_id_without_runtime_state_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "generated-session-id",
    )
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "generated output"}) + "\n",
            ),
        ),
    )

    continuation = adapter.claude_continuation(provider_session_id=None)

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            adapter.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "generated output"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "generated-session-id",
            "exact_transcript_match": False,
        },
    )
    assert adapter.recorded_request_count == 1
    recorded_request = adapter.recorded_request()
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "generated-session-id"
    assert recorded_request.command


@pytest.mark.parametrize("create_state_dir", [False, True])
def test_runtime_client_runs_claude_resumed_session_fresh_when_provider_state_is_not_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    create_state_dir: bool,
) -> None:
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "fresh output"}) + "\n",
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir_relpath = "implementer/main/claude/"
    if create_state_dir:
        (runtime_state_dir / provider_state_dir_relpath).mkdir(parents=True)

    continuation = adapter.claude_continuation(
        provider_state_dir_relpath=provider_state_dir_relpath,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            adapter.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "fresh output"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "claude-session-123",
            "exact_transcript_match": False,
        },
    )
    assert adapter.recorded_request_count == 1
    recorded_request = adapter.recorded_request()
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.command


def test_runtime_client_new_session_requires_claude_auth_when_runtime_state_is_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "should not run"}) + "\n",
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/claude" / "nested"
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    (provider_state_dir / "transcript.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    provider_selection=InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    session_namespace="main",
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert str(exc_info.value) == "Missing Claude Code OAuth token."
    assert exc_info.value.service_name == "claude"
    assert adapter.recorded_requests == []


def test_runtime_client_returns_started_usage_limited_outcome_for_claude_new_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "thinking"}],
                                "usage": {
                                    "input_tokens": 3,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0,
                                },
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 429,
                            "result": "usage limited",
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            RuntimeClientExecutionHarness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time is None
    result = outcome.result
    assert result.output == ""
    assert result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert result.usage == runtime.ProviderUsage(
        input_tokens=3,
        output_tokens=None,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cost_usd=None,
        duration_seconds=None,
    )
    assert result.continuation is not None
    assert result.continuation.provider_resume_state == {
        "run_kind": "resume",
        "provider_session_id": "session-uuid",
        "exact_transcript_match": False,
    }
    assert result.continuation.tool_access == contracts_runtime.ToolAccess.no_tools()


def test_runtime_client_omits_continuation_for_pre_start_claude_new_session_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "is_error": True,
                            "api_error_status": 429,
                            "result": "usage limited",
                        }
                    )
                    + "\n",
                ]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: _ClaudeProcess(),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            RuntimeClientExecutionHarness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=RuntimeClientExecutionHarness.prepare_runtime_state_dir(
                    tmp_path
                ),
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time is None
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation is None


def test_runtime_client_runs_codex_new_session_with_runtime_state_and_host_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"continued output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
        ),
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.workspace_backed(
                    tmp_path,
                    tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
                ),
            )
        )
    )

    provider_state_dir_relpath = "implementer/main/codex/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            tmp_path,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": False,
        },
    )
    assert len(harness.recorded_requests) == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id is None
    assert recorded_request.command.startswith(f"{_codex_executable()} exec -m gpt-5.4")
    assert (provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
    )


def test_runtime_client_rejects_codex_resumed_session_without_usable_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all()

    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                harness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    continuation=harness.codex_continuation(
                        provider_session_id="   ",
                        provider_state_dir_relpath=None,
                    ),
                )
            )
        )

    assert str(exc_info.value) == (
        "Codex continuation is missing `provider_session_id`."
    )
    assert harness.recorded_request_count == 0


def test_runtime_client_rejects_resumed_session_with_non_object_portable_continuation_resume_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                harness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=harness.prepare_runtime_state_dir(tmp_path),
                    continuation=prompt_runtime.Continuation(
                        selected_service="codex",
                        selected_model="gpt-5.4",
                        selected_effort="medium",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state=["resume"],
                    ),
                ),
            )
        )

    assert str(exc_info.value) == (
        "Continuation provider_resume_state must be a JSON object."
    )


def test_runtime_client_rejects_resumed_session_with_malformed_continuation_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                harness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    continuation=prompt_runtime.Continuation(serialized="{not-json"),
                ),
            )
        )

    assert str(exc_info.value) == "Continuation data is not valid JSON."


def test_runtime_client_rejects_new_session_for_unsupported_session_backed_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all()
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_PORTABLE_CONTINUATION_PROVIDERS",
        frozenset({"claude"}),
    )

    with pytest.raises(RuntimeConfigurationError, match="Portable continuation"):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                harness.start_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=harness.prepare_runtime_state_dir(tmp_path),
                    provider_selection=InternalStageSelection(
                        service="opencode",
                        model="deepseek-v4-flash",
                        effort="medium",
                    ),
                    provider_auth=runtime.ProviderAuth(opencode_api_key="api-key"),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )
    assert harness.recorded_requests == []


def test_runtime_client_rejects_resumed_session_for_unsupported_session_backed_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all()
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_PORTABLE_CONTINUATION_PROVIDERS",
        frozenset({"claude"}),
    )

    with pytest.raises(RuntimeConfigurationError, match="Portable continuation"):
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                harness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=harness.prepare_runtime_state_dir(tmp_path),
                    continuation=harness.opencode_continuation(
                        model="deepseek-v4-flash",
                        provider_session_id="restored-session",
                        exact_transcript_match=False,
                    ),
                ),
            )
        )
    assert harness.recorded_request_count == 0


@pytest.mark.parametrize("entrypoint", ["new", "resumed"])
def test_runtime_client_requires_host_codex_auth_for_session_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all()
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: tmp_path / "missing-home",
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        if entrypoint == "new":
            asyncio.run(
                runtime.RuntimeClient().run_new_session(
                    harness.start_session_run_request(
                        invocation_dir=tmp_path,
                        runtime_state_dir=runtime_state_dir,
                        provider_selection=InternalStageSelection(
                            service="codex",
                            model="gpt-5.4",
                            effort="medium",
                        ),
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                    )
                )
            )
        else:
            asyncio.run(
                runtime.RuntimeClient().run_resumed_session(
                    harness.resume_session_run_request(
                        invocation_dir=tmp_path,
                        runtime_state_dir=runtime_state_dir,
                        continuation=harness.codex_continuation(),
                    )
                )
            )

    assert str(exc_info.value) == (
        "Codex authentication missing: run `codex login` on the host."
    )
    assert exc_info.value.service_name == "codex"
    assert harness.recorded_request_count == 0


def test_runtime_client_treats_nested_claude_provider_state_as_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "continued output"}) + "\n",
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/claude" / "nested"
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    (provider_state_dir / "transcript.json").write_text("{}", encoding="utf-8")

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome.result is not None
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "session-uuid"
    assert recorded_request.command


@pytest.mark.parametrize(
    (
        "service_name",
        "stage",
        "auth",
        "prepared_invocation",
        "expected_output",
        "expected_usage",
    ),
    [
        pytest.param(
            "claude",
            InternalStageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
            provider_invocation_runtime.ProviderInvocationPreparedStream(
                stdout_lines=(
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "intermediate"}],
                                "usage": {
                                    "input_tokens": 5,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0,
                                },
                            },
                        }
                    )
                    + "\n",
                    json.dumps({"type": "result", "result": "final output"}) + "\n",
                ),
            ),
            "final output",
            runtime.ProviderUsage(
                input_tokens=5,
                output_tokens=None,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=None,
                duration_seconds=None,
            ),
            id="claude",
        ),
        pytest.param(
            "codex",
            InternalStageSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            None,
            provider_invocation_runtime.ProviderInvocationPreparedStream(
                stdout_lines=(
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "agent_message", "text": "codex output"},
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {
                                "input_tokens": 3,
                                "cached_tokens": 1,
                                "output_tokens": 2,
                            },
                        }
                    )
                    + "\n",
                ),
            ),
            "codex output",
            runtime.ProviderUsage(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=1,
            ),
            id="codex",
        ),
        pytest.param(
            "opencode",
            InternalStageSelection(
                service="opencode",
                model="kimi-k2.6",
                effort="medium",
            ),
            runtime.ProviderAuth(opencode_api_key="go-key"),
            provider_invocation_runtime.ProviderInvocationPreparedStream(
                stdout_lines=(
                    json.dumps(
                        {
                            "type": "text",
                            "timestamp": 1,
                            "sessionID": "sess_123",
                            "part": {
                                "id": "part_1",
                                "sessionID": "sess_123",
                                "messageID": "msg_1",
                                "type": "text",
                                "text": "first assistant turn",
                                "time": {"start": 1, "end": 2},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "text",
                            "timestamp": 2,
                            "sessionID": "sess_123",
                            "part": {
                                "id": "part_2",
                                "sessionID": "sess_123",
                                "messageID": "msg_1",
                                "type": "text",
                                "text": "second assistant turn",
                                "time": {"start": 2, "end": 3},
                            },
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "session.status",
                            "timestamp": 3,
                            "sessionID": "sess_123",
                            "status": {"type": "idle"},
                        }
                    )
                    + "\n",
                ),
            ),
            "first assistant turn\n\nsecond assistant turn",
            None,
            id="opencode",
        ),
    ],
)
def test_runtime_client_runs_ephemeral_built_in_provider_through_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service_name: str,
    stage: InternalStageSelection,
    auth: runtime.ProviderAuth | None,
    prepared_invocation: provider_invocation_runtime.ProviderInvocationPreparedStream,
    expected_output: str,
    expected_usage: runtime.ProviderUsage | None,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(prepared_invocation)
    if service_name == "codex":
        RuntimeClientExecutionHarness.install_local_codex_host_auth(
            monkeypatch,
            tmp_path,
        )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=stage,
                provider_auth=auth,
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    result = outcome.result
    assert result.output == expected_output
    assert result.usage == expected_usage
    assert result.selected == runtime.ResolvedProvider(
        service=service_name, model=stage.model, effort=stage.effort
    )
    assert result.continuation is None
    assert len(harness.recorded_requests) == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id is None
    assert recorded_request.log_context is None
    assert recorded_request.command
    assert recorded_request.argv
    assert list((tmp_path / "logs").glob("*.log")) == []


def test_runtime_client_ephemeral_execution_remains_available_when_session_backed_support_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="ephemeral output",
            usage=runtime.ProviderUsage(input_tokens=3, output_tokens=2),
        ),
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_PORTABLE_CONTINUATION_PROVIDERS",
        frozenset({"claude"}),
    )

    with pytest.raises(RuntimeConfigurationError):
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="missing",
                        model="placeholder",
                        effort="placeholder",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )
    assert len(adapter.recorded_requests) == 0


def test_runtime_client_runs_resumed_opencode_session_through_built_in_provider_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="continued output",
            provider_session_id="persisted-session-2",
        ),
    )

    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    worktree.mkdir()
    provider_state_dir = harness.prepare_provider_state_dir(
        runtime_state_dir,
        service="opencode",
    )
    harness.prepare_opencode_provider_state(
        provider_state_dir,
        session_id="persisted-session-1",
    )
    continuation = harness.opencode_continuation(
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=worktree,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=continuation.tool_access,
        provider_resume_state={
            "provider_session_id": "persisted-session-2",
            "provider_state": {
                "session_id": "persisted-session-2",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": False,
        },
    )
    assert outcome.result is not None
    assert isinstance(outcome.result, prompt_runtime.RunResult)
    assert outcome.result.selected.model == "glm-5.2"
    assert outcome.result.selected.effort == "medium"
    assert harness.recorded_request_count == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.worktree == worktree
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "persisted-session-1"
    assert recorded_request.command


@pytest.mark.parametrize(
    "tool_access",
    [
        contracts_runtime.ToolAccess.no_tools(),
        contracts_runtime.ToolAccess.workspace_backed(
            Path("."), tool_policy=runtime.ToolPolicy.INSPECT_ONLY
        ),
        contracts_runtime.ToolAccess.workspace_backed(
            Path("."), tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION
        ),
        contracts_runtime.ToolAccess.workspace_backed(
            Path("."), tool_policy=runtime.ToolPolicy.UNRESTRICTED
        ),
    ],
)
def test_runtime_client_runs_codex_new_session_through_built_in_provider_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_access: contracts_runtime.ToolAccess,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "continued output",
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 3,
                            "cached_tokens": 1,
                            "output_tokens": 2,
                        },
                    }
                )
                + "\n",
            ),
        ),
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=(
                    tool_access
                    if tool_access.kind == "none"
                    else contracts_runtime.ToolAccess.workspace_backed(
                        tmp_path, tool_policy=tool_access.tool_policy
                    )
                ),
            )
        )
    )

    provider_state_dir_relpath = "implementer/main/codex/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=3,
        output_tokens=2,
        cache_read_input_tokens=1,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=(
            tool_access
            if tool_access.kind == "none"
            else contracts_runtime.ToolAccess.workspace_backed(
                tmp_path, tool_policy=tool_access.tool_policy
            )
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": False,
        },
    )
    assert len(harness.recorded_requests) == 1
    recorded_request = harness.recorded_request()
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id is None
    assert recorded_request.log_context is None
    assert recorded_request.command
    assert (provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
    )


def test_runtime_client_keeps_started_codex_new_session_continuation_from_provider_invocation_failure_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail="Usage limit reached (reset_time=None)",
            usage=runtime.ProviderUsage(
                input_tokens=3,
                output_tokens=1,
                cache_read_input_tokens=1,
            ),
            stdout_lines=(
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                json.dumps(
                    {
                        "type": "turn.failed",
                        "error": {
                            "message": (
                                "You've hit your usage limit. "
                                "Try again at January 2, 5pm (UTC)."
                            )
                        },
                    }
                )
                + "\n",
            ),
        ),
    )

    runtime_state_dir = harness.prepare_runtime_state_dir(tmp_path)
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time is None
    assert outcome.result.output == ""
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=3,
        output_tokens=1,
        cache_read_input_tokens=1,
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert len(harness.recorded_requests) == 1
    assert harness.recorded_request().log_context is None


def test_runtime_client_preserves_opencode_invalid_api_key_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 1,
                        "sessionID": "sess_123",
                        "error": {
                            "name": "AuthenticationError",
                            "data": {
                                "message": "invalid api key",
                                "statusCode": 401,
                                "isRetryable": False,
                            },
                        },
                    }
                )
                + "\n",
            )
        ),
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                        InternalStageSelection(
                            service="opencode",
                            model="kimi-k2.6",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(opencode_api_key="go-key"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert exc_info.value.service_name == "opencode"
    assert exc_info.value.classification == (
        "operator_actionable_agent_credential_failure"
    )
    assert str(exc_info.value) == "invalid api key"
    assert not hasattr(exc_info.value, "status_code")
    assert not hasattr(exc_info.value, "observations")


def test_runtime_client_maps_opencode_usage_limit_after_ignoring_malformed_and_non_text_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                '"not a dict"\n',
                "not json\n",
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1,
                        "sessionID": "sess_123",
                        "part": {
                            "type": "tool",
                            "text": "ignored",
                            "time": {"start": 1, "end": 2},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 2,
                        "sessionID": "sess_123",
                        "error": {
                            "name": "RateLimitError",
                            "data": {
                                "message": (
                                    "You have reached your OpenCode Go usage limit. "
                                    "Try again at Apr 28th, 2026 9:02 PM."
                                ),
                                "statusCode": 429,
                                "isRetryable": True,
                            },
                        },
                    }
                )
                + "\n",
            )
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time == datetime(2026, 4, 28, 21, 2, tzinfo=timezone.utc)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="kimi-k2.6", effort="medium"
    )
    assert outcome.result.continuation is None


def test_runtime_client_maps_codex_usage_limit_stream_to_usage_limited_and_logs_provider_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(monkeypatch, tmp_path)
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "turn.failed",
                        "error": {
                            "message": (
                                "You've hit your usage limit. "
                                "Try again at January 2, 5pm (UTC)."
                            )
                        },
                    }
                )
                + "\n",
            )
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time == datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.command.startswith(f"{_codex_executable()} exec")
    assert list((tmp_path / "logs").glob("*.log")) == []


def test_runtime_client_reused_after_usage_limited_ephemeral_call_still_invokes_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(monkeypatch, tmp_path)
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    adapter = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "turn.failed",
                        "error": {
                            "message": (
                                "You've hit your usage limit. "
                                "Try again at January 2, 5pm (UTC)."
                            )
                        },
                    }
                )
                + "\n",
            )
        ),
        provider_invocation_runtime.ProviderInvocationResult(
            output="completed on retry",
            usage=runtime.ProviderUsage(input_tokens=3, output_tokens=2),
        ),
    )
    request = prompt_runtime.EphemeralRunRequest(
        prompt="already rendered prompt",
        invocation_dir=tmp_path,
        provider_selection=InternalStageSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
        tool_access=contracts_runtime.ToolAccess.no_tools(),
    )

    client = runtime.RuntimeClient()
    first_outcome = asyncio.run(client.run_ephemeral(request))
    second_outcome = asyncio.run(client.run_ephemeral(request))

    assert isinstance(first_outcome.kind, prompt_runtime.UsageLimited)
    assert first_outcome.kind.reset_time == datetime(
        2026, 1, 2, 17, 0, tzinfo=timezone.utc
    )
    assert first_outcome.result.output == ""
    assert first_outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert first_outcome.result.continuation is None
    assert isinstance(second_outcome.kind, prompt_runtime.Completed)
    assert second_outcome.result.output == "completed on retry"
    assert second_outcome.result.usage == runtime.ProviderUsage(
        input_tokens=3, output_tokens=2
    )
    assert second_outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert second_outcome.result.continuation is None
    assert len(adapter.recorded_requests) == 2


def test_runtime_client_reports_selected_service_for_ephemeral_usage_limit_when_provider_omits_service_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_failure(
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail="Usage limit reached (reset_time=2026-01-02T17:00:00+00:00)",
            reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time == datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.continuation is None


def test_runtime_client_maps_opencode_missing_model_without_status_to_hard_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                "not json\n",
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1,
                        "sessionID": "sess_123",
                        "part": {
                            "type": "image",
                            "text": "ignored",
                            "time": {"start": 1, "end": 2},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 2,
                        "sessionID": "sess_123",
                        "error": {
                            "name": "UnknownError",
                            "data": {
                                "message": (
                                    "Model not found: "
                                    "opencode-go/deepseek-v4-flash. "
                                    "Did you mean: deepseek-v4-flash?"
                                )
                            },
                        },
                    }
                )
                + "\n",
            )
        ),
    )

    with pytest.raises(HardAgentError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                        InternalStageSelection(
                            service="opencode",
                            model="kimi-k2.6",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(opencode_api_key="go-key"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert str(exc_info.value) == (
        "Model not found: opencode-go/deepseek-v4-flash. "
        "Did you mean: deepseek-v4-flash?"
    )
    assert exc_info.value.service_name == "opencode"


def test_runtime_client_maps_opencode_transient_error_stream_to_transient_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 1,
                        "sessionID": "sess_123",
                        "error": {
                            "name": "InternalServerError",
                            "data": {
                                "message": "temporary backend failure",
                                "statusCode": 503,
                                "isRetryable": True,
                            },
                        },
                    }
                )
                + "\n",
            )
        ),
    )

    with pytest.raises(TransientAgentError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                        InternalStageSelection(
                            service="opencode",
                            model="kimi-k2.6",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(opencode_api_key="go-key"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert str(exc_info.value) == "temporary backend failure"
    assert exc_info.value.status_code == 503


def test_runtime_client_keeps_completed_opencode_result_after_idle_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1,
                        "sessionID": "sess_123",
                        "part": {
                            "id": "part_1",
                            "sessionID": "sess_123",
                            "messageID": "msg_1",
                            "type": "text",
                            "text": "completed answer",
                            "time": {"start": 1, "end": 2},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "session.status",
                        "timestamp": 2,
                        "sessionID": "sess_123",
                        "status": {"type": "idle"},
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 3,
                        "sessionID": "sess_123",
                        "error": {
                            "name": "InternalServerError",
                            "data": {
                                "message": "should be ignored after idle result",
                                "statusCode": 503,
                                "isRetryable": True,
                            },
                        },
                    }
                )
                + "\n",
            )
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "completed answer"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="kimi-k2.6", effort="medium"
    )
    assert outcome.result.continuation is None


def test_runtime_client_maps_claude_usage_limit_stream_to_usage_limited_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "api_error_status": 429,
                        "result": "Claude usage limit reached.",
                    }
                )
                + "\n",
            )
        ),
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time is None
    result = outcome.result
    assert result.output == ""
    assert result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )


def test_runtime_client_maps_claude_transient_error_stream_to_transient_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "api_error_status": 500,
                        "result": "temporary Claude failure",
                    }
                )
                + "\n",
            )
        ),
    )

    with pytest.raises(TransientAgentError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                        InternalStageSelection(
                            service="claude",
                            model="sonnet",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert "temporary Claude failure" in str(exc_info.value)
    assert exc_info.value.status_code == 500


def test_runtime_client_preserves_claude_usage_on_usage_limited_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "partial output"}],
                            "usage": {
                                "input_tokens": 3,
                                "cache_creation_input_tokens": 1,
                                "cache_read_input_tokens": 2,
                            },
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "api_error_status": 429,
                        "result": "Claude usage limit reached.",
                    }
                )
                + "\n",
            )
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time is None
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=3,
        output_tokens=None,
        cache_read_input_tokens=2,
        cache_creation_input_tokens=1,
        cost_usd=None,
        duration_seconds=None,
    )


def test_runtime_client_parses_claude_usage_limit_reset_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "api_error_status": 429,
                        "result": (
                            "Claude usage limit reached, resets Jan 2, 4pm (UTC)."
                        ),
                    }
                )
                + "\n",
            )
        ),
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time == datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)
    assert outcome.result.output == ""
    assert outcome.result.selected.service == "claude"


def test_runtime_client_keeps_runtime_reset_time_override_in_usage_limited_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reset_time = datetime(2026, 1, 2, 16, 0, tzinfo=timezone.utc)
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "api_error_status": 429,
                        "result": "Claude usage limit reached.",
                    }
                )
                + "\n",
            )
        ),
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        prompt_runtime,
        "_parse_claude_reset_time",
        lambda _text: reset_time,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time == reset_time
    assert outcome.result.output == ""
    assert outcome.result.selected.service == "claude"


def test_runtime_client_rejects_unsupported_selected_provider_for_ephemeral_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "final output"}) + "\n",
            )
        ),
    )

    with pytest.raises(RuntimeConfigurationError):
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=InternalStageSelection(
                        service="missing",
                        model="ignored",
                        effort="low",
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )


def test_runtime_client_preserves_claude_credential_failure_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    denial_message = "Disabled Claude subscription access for Claude Code."
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "api_error_status": 403,
                        "result": denial_message,
                    }
                )
                + "\n",
            )
        ),
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                        InternalStageSelection(
                            service="claude",
                            model="sonnet",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )

    assert denial_message in str(exc_info.value)
    assert exc_info.value.service_name == "claude"
    assert exc_info.value.classification is None
    assert not hasattr(exc_info.value, "status_code")
    assert not hasattr(exc_info.value, "observations")


def test_runtime_client_ephemeral_times_out_with_no_events_within_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A run with no Agent Events within the timeout window should abort with timed_out outcome."""
    _install_deterministic_timeout_watchdog(
        monkeypatch,
        timeout_check_numbers=(1,),
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "session.status", "status": {"type": "idle"}}),
            )
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=1,
                on_live_output=lambda _event: None,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.selected.service == "opencode"


def test_runtime_client_new_session_times_out_after_agent_event_and_preserves_observed_session_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )

    observed_events: list[prompt_runtime.AgentEvent] = []
    _install_deterministic_timeout_watchdog(
        monkeypatch,
        timeout_check_numbers=(2,),
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "intermediate"}],
                            "usage": {
                                "input_tokens": 5,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 1,
                            },
                        },
                    }
                )
                + "\n",
                _claude_result_output_line("final output"),
            ),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=1,
                on_live_output=observed_events.append,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=5,
        output_tokens=None,
        cache_read_input_tokens=1,
        cache_creation_input_tokens=0,
        cost_usd=None,
        duration_seconds=None,
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "session-uuid",
            "exact_transcript_match": False,
        },
    )
    assert [event.display_message for event in observed_events] == [
        "intermediate",
        "result",
    ]


def test_runtime_client_codex_new_session_timeout_preserves_observed_usage_and_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )

    observed_events: list[prompt_runtime.AgentEvent] = []
    _install_deterministic_timeout_watchdog(
        monkeypatch,
        timeout_check_numbers=(3,),
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "thread.started", "thread_id": "thread-123"})
                + "\n",
                _codex_assistant_output_line("initial output"),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 5,
                            "cached_tokens": 2,
                            "output_tokens": 1,
                        },
                    }
                )
                + "\n",
            ),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=harness.prepare_runtime_state_dir(tmp_path),
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=1,
                on_live_output=observed_events.append,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=5,
        output_tokens=1,
        cache_read_input_tokens=2,
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert [event.display_message for event in observed_events] == [
        "thread.started",
        "initial output",
        "turn.completed",
    ]


def test_runtime_client_codex_resumed_session_timeout_preserves_observed_usage_and_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )

    observed_events: list[prompt_runtime.AgentEvent] = []
    _install_deterministic_timeout_watchdog(
        monkeypatch,
        timeout_check_numbers=(3,),
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "thread.started", "thread_id": "thread-456"})
                + "\n",
                _codex_assistant_output_line("continued output"),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 8,
                            "cached_tokens": 3,
                            "output_tokens": 2,
                        },
                    }
                )
                + "\n",
            ),
        ),
    )

    continuation = harness.codex_continuation(provider_session_id="thread-123")

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
                timeout_seconds=1,
                on_live_output=observed_events.append,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=8,
        output_tokens=2,
        cache_read_input_tokens=3,
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-456",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert [event.display_message for event in observed_events] == [
        "thread.started",
        "continued output",
        "turn.completed",
    ]


def test_runtime_client_resumed_session_times_out_and_preserves_observed_session_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_events: list[prompt_runtime.AgentEvent] = []
    _install_deterministic_timeout_watchdog(
        monkeypatch,
        timeout_check_numbers=(2,),
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "continued"}],
                            "usage": {
                                "input_tokens": 7,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 1,
                            },
                        },
                    }
                )
                + "\n",
                _claude_result_output_line("final output"),
            ),
        ),
    )

    continuation = harness.claude_continuation()

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            harness.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                timeout_seconds=1,
                on_live_output=observed_events.append,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=7,
        output_tokens=None,
        cache_read_input_tokens=1,
        cache_creation_input_tokens=0,
        cost_usd=None,
        duration_seconds=None,
    )
    assert outcome.result.continuation == continuation
    assert [event.display_message for event in observed_events] == [
        "continued",
        "result",
    ]


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_runtime_client_ephemeral_disables_idle_timeout_for_non_positive_timeout_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timeout_seconds: int,
) -> None:
    observed_events: list[prompt_runtime.AgentEvent] = []
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "session.status", "status": {"type": "idle"}}),
            ),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=timeout_seconds,
                on_live_output=observed_events.append,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.selected.service == "opencode"
    assert observed_events == [
        prompt_runtime.AgentEvent(
            type="other",
            display_message="idle",
            raw_provider_output=(
                '{"type": "session.status", "status": {"type": "idle"}}'
            ),
        )
    ]


def test_idle_timeout_defaults_to_300_seconds_on_all_lifecycle_requests(
    tmp_path: Path,
) -> None:
    """All three lifecycle request types default to a 300s idle timeout."""
    import inspect

    ephemeral = prompt_runtime.EphemeralRunRequest(
        prompt="test",
        invocation_dir=tmp_path,
        provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
            InternalStageSelection(
                service="claude", model="claude-haiku-4-5", effort="low"
            ),
            runtime.ProviderAuth(claude_code_oauth_token="tok"),
        ),
        tool_policy=runtime.ToolPolicy.NONE,
    )
    assert ephemeral.timeout_seconds == 300

    new_session = prompt_runtime.NewSessionRunRequest(
        prompt="test",
        invocation_dir=tmp_path,
        provider_selection=RuntimeClientExecutionHarness.attach_provider_auth(
            InternalStageSelection(
                service="claude", model="claude-haiku-4-5", effort="low"
            ),
            runtime.ProviderAuth(claude_code_oauth_token="tok"),
        ),
        tool_policy=runtime.ToolPolicy.NONE,
    )
    assert new_session.timeout_seconds == 300

    assert (
        "timeout_seconds"
        in inspect.signature(prompt_runtime.ResumedSessionRunRequest).parameters
    )


def test_runtime_client_threads_timeout_seconds_into_provider_invocation_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(_claude_result_output_line("final output"),),
        ),
    )
    RuntimeClientExecutionHarness.install_generated_provider_session_id(
        monkeypatch,
        "session-uuid",
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            harness.start_session_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=17,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert harness.recorded_request().timeout_seconds == 17


def test_runtime_client_resumed_session_silent_invocation_timeout_preserves_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _TimedOutInvocationAdapter:
        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            error = provider_invocation_runtime.ProviderInvocationTimedOutError(
                "Provider subprocess exceeded the idle timeout."
            )
            setattr(error, "provider_session_id", request.provider_session_id)
            raise error

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: _TimedOutInvocationAdapter(),
    )

    continuation = RuntimeClientExecutionHarness.claude_continuation()

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                timeout_seconds=1,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.continuation == continuation


def test_runtime_client_new_session_invocation_timeout_preserves_observed_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "claude_partial_then_hang.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "import signal",
                "import sys",
                "import time",
                "",
                "signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))",
                "print(json.dumps({",
                "    'type': 'assistant',",
                "    'message': {",
                "        'content': [{'type': 'text', 'text': 'intermediate'}],",
                "        'usage': {",
                "            'input_tokens': 5,",
                "            'cache_creation_input_tokens': 0,",
                "            'cache_read_input_tokens': 1",
                "        }",
                "    }",
                "}), flush=True)",
                "while True:",
                "    time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        builtin_provider_rendering_runtime,
        "render_built_in_provider_invocation",
        lambda _request: (
            builtin_provider_rendering_runtime.BuiltInProviderRenderedInvocation(
                canonical_argv=(sys.executable, str(script_path)),
                legacy_command_text=None,
                environment={},
                prompt_path=None,
                prompt_cleanup_choice=builtin_provider_rendering_runtime.PromptCleanupChoice.KEEP,
                prompt_transport_preference=builtin_provider_rendering_runtime.PromptTransportPreference.STDIN,
                provider_session_id_placement=builtin_provider_rendering_runtime.ProviderSessionIdPlacement.NONE,
                prefer_argv=True,
            )
        ),
    )
    RuntimeClientExecutionHarness.install_generated_provider_session_id(
        monkeypatch,
        "session-uuid",
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            RuntimeClientExecutionHarness.start_session_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=1,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=5,
        output_tokens=None,
        cache_read_input_tokens=1,
        cache_creation_input_tokens=0,
        cost_usd=None,
        duration_seconds=None,
    )


def test_runtime_client_ephemeral_times_out_without_live_output_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Idle timeout fires even when no on_live_output callback is provided."""
    _install_deterministic_timeout_watchdog(
        monkeypatch,
        timeout_check_numbers=(1,),
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(_opencode_idle_output_line(),)
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=InternalStageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=1,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.selected.service == "opencode"
