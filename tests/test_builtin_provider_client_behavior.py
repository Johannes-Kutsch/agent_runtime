from __future__ import annotations

import asyncio
import json
import os
import subprocess
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Callable, cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.errors import (
    AgentCredentialFailureError,
    HardAgentError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    RuntimeConfigurationError,
    TransientAgentError,
)
from agent_runtime.session import RunKind
from agent_runtime.types import ProviderSelection as InternalStageSelection


_CURRENT_OPENCODE_GO_MODELS = [
    "glm-5.2",
    "glm-5.1",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "minimax-m3",
    "minimax-m2.7",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
]
_STALE_OPENCODE_GO_MODELS = [
    "glm-5",
    "qwen3.5-plus",
    "kimi-k2.5",
    "mimo-v2-omni",
    "mimo-v2-pro",
    "minimax-m2.5",
    "hy3-preview",
]


def _codex_executable() -> str:
    return "codex.cmd" if os.name == "nt" else "codex"


def _selection_with_auth(selection: Any, auth: Any) -> Any:
    return dataclasses.replace(selection, auth=auth)


def _write_codex_rollout(state_dir: Path, *thread_ids: str) -> None:
    rollout_dir = state_dir / "sessions" / "2026" / "05" / "30"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"type": "thread.started", "thread_id": thread_id})
        for thread_id in thread_ids
    ]
    (rollout_dir / "rollout-001.jsonl").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _install_in_memory_provider_invocation_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *prepared_invocations: (
        provider_invocation_runtime.ProviderInvocationResult
        | provider_invocation_runtime.ProviderInvocationFailure
        | provider_invocation_runtime.ProviderInvocationPreparedStream
    ),
) -> provider_invocation_runtime.InMemoryProviderInvocationAdapter:
    adapter = provider_invocation_runtime.InMemoryProviderInvocationAdapter(
        prepared_invocations=list(prepared_invocations)
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )
    return adapter


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


def _expected_opencode_permission(
    tool_policy: runtime.ToolPolicy | contracts_runtime.ToolPolicyProfile,
) -> dict[str, str] | str | None:
    profile = _normalize_tool_policy_profile(tool_policy)
    if profile == runtime.ToolPolicy.NONE.profile:
        return "deny"
    if profile == runtime.ToolPolicy.INSPECT_ONLY.profile:
        return {"edit": "deny", "bash": "deny"}
    if profile == runtime.ToolPolicy.NO_FILE_MUTATION.profile:
        return {"edit": "deny"}
    return None


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

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
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
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(hello_line, world_line),
        ),
    )
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
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
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(thread_started_line, message_line),
        ),
    )
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
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
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(tool_line, text_line, idle_line),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
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
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=_selection_with_auth(
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
    assert len(adapter.recorded_requests) == 1

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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "session-uuid"
    assert recorded_request.environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
        "CLAUDE_CONFIG_DIR": str(provider_state_dir),
    }
    assert "--session-id session-uuid" in recorded_request.command
    assert "--resume" not in recorded_request.command
    assert provider_state_dir.is_dir()


def test_runtime_client_new_session_without_runtime_state_dir_returns_meaningful_continuation_without_state_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-session-id",
    )
    _ = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="opencode-key"),
                ),
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
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)

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

    with pytest.raises(AgentCredentialFailureError):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
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
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="missing",
                            model="ignored",
                            effort="low",
                        ),
                        runtime.ProviderAuth(opencode_api_key="root-only-key"),
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

    assert len(adapter.recorded_requests) == 0


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


@pytest.mark.parametrize(
    ("tool_policy", "expected_flags"),
    [
        (runtime.ToolPolicy.NONE, ('--disallowedTools "all"',)),
        (
            runtime.ToolPolicy.INSPECT_ONLY,
            ("--tools 'Read Glob'",),
        ),
        (
            runtime.ToolPolicy.NO_FILE_MUTATION,
            ('--disallowedTools "Edit Write NotebookEdit"',),
        ),
        (runtime.ToolPolicy.UNRESTRICTED, tuple()),
    ],
)
def test_runtime_client_runs_claude_new_session_with_tool_policy_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_policy: runtime.ToolPolicy,
    expected_flags: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "intermediate"}],
                        },
                    }
                )
                + "\n",
                json.dumps({"type": "result", "result": "final output"}) + "\n",
            )
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    tool_access = (
        contracts_runtime.ToolAccess.no_tools()
        if tool_policy is runtime.ToolPolicy.NONE
        else contracts_runtime.ToolAccess.workspace_backed(
            tmp_path, tool_policy=tool_policy
        )
    )
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                session_namespace="main",
                tool_access=tool_access,
            )
        )
    )

    provider_state_dir_relpath = "implementer/main/claude/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "final output"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "session-uuid",
            "exact_transcript_match": False,
        },
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.provider_session_id == "session-uuid"
    assert recorded_request.environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
        "CLAUDE_CONFIG_DIR": str(provider_state_dir),
    }
    command = recorded_request.command
    if tool_policy is runtime.ToolPolicy.NONE:
        assert "--tools none" not in command
    if tool_policy is runtime.ToolPolicy.INSPECT_ONLY:
        assert '--disallowedTools "all"' not in command
    if tool_policy is runtime.ToolPolicy.NO_FILE_MUTATION:
        assert "--tools" not in command
    if tool_policy is runtime.ToolPolicy.UNRESTRICTED:
        assert "--tools" not in command
        assert "--disallowedTools" not in command
    for flag in expected_flags:
        assert flag in command


def test_runtime_client_runs_claude_new_session_and_returns_portable_continuation_for_resumption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                provider_selection=_selection_with_auth(
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
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                continuation=continuation,
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 2
    resumed_request = adapter.recorded_requests[1]
    assert resumed_request.environment == {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}
    assert "--resume session-uuid" in resumed_request.command
    assert "--session-id" not in resumed_request.command


def test_runtime_client_runs_claude_new_session_through_in_memory_provider_invocation_adapter(
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
        ]
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        provider_selection=_selection_with_auth(
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "session-uuid"


def test_runtime_client_runs_opencode_new_session_through_in_memory_provider_invocation_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-session-id",
    )
    adapter = provider_invocation_runtime.InMemoryProviderInvocationAdapter(
        prepared_invocations=[
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
        ]
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        provider_selection=_selection_with_auth(
            InternalStageSelection(
                service="opencode",
                model="glm-5.2",
                effort="medium",
            ),
            runtime.ProviderAuth(opencode_api_key="opencode-key"),
        ),
        session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
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
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-session-id",
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="opencode-key"),
                ),
                session_namespace="main",
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
    assert adapter.recorded_requests[0].provider_session_id == "prepared-session-id"
    provider_state_dir = runtime_state_dir / "implementer" / "main" / "opencode"
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "observed-session-id"
    )


@pytest.mark.parametrize(
    ("tool_policy", "expected_permission"),
    [
        (runtime.ToolPolicy.NONE, "deny"),
        (runtime.ToolPolicy.INSPECT_ONLY, {"edit": "deny", "bash": "deny"}),
        (runtime.ToolPolicy.NO_FILE_MUTATION, {"edit": "deny"}),
        (runtime.ToolPolicy.UNRESTRICTED, None),
    ],
)
def test_runtime_client_ephemeral_opencode_command_uses_tool_policy_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_policy: runtime.ToolPolicy,
    expected_permission: dict[str, str] | str | None,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "part": {
                            "type": "text",
                            "time": {"end": True},
                            "text": "hello from opencode",
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
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="opencode-key"),
                ),
                tool_access=_opencode_tool_access(tool_policy, tmp_path),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "hello from opencode"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="kimi-k2.6", effort="medium"
    )
    assert outcome.result.continuation is None

    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    config = json.loads(recorded_request.environment["OPENCODE_CONFIG_CONTENT"])
    if expected_permission is None:
        assert "permission" not in config
    else:
        assert config["permission"] == expected_permission


@pytest.mark.parametrize(
    "tool_policy",
    [
        pytest.param(runtime.ToolPolicy.NONE.profile, id="none-profile"),
        pytest.param(
            runtime.ToolPolicy.INSPECT_ONLY.profile,
            id="inspect-only-profile",
        ),
        pytest.param(
            runtime.ToolPolicy.NO_FILE_MUTATION.profile,
            id="no-file-mutation-profile",
        ),
        pytest.param(
            runtime.ToolPolicy.UNRESTRICTED.profile,
            id="unrestricted-profile",
        ),
    ],
)
def test_runtime_client_ephemeral_opencode_command_uses_equivalent_tool_policy_profile_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_policy: contracts_runtime.ToolPolicyProfile,
) -> None:
    expected_permission = _expected_opencode_permission(tool_policy)
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "part": {
                            "type": "text",
                            "time": {"end": True},
                            "text": "hello from opencode",
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
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="opencode-key"),
                ),
                tool_access=_opencode_tool_access(tool_policy, tmp_path),
            )
        )
    )

    assert outcome.result.output == "hello from opencode"
    recorded_request = adapter.recorded_requests[0]
    config = json.loads(recorded_request.environment["OPENCODE_CONFIG_CONTENT"])
    if expected_permission is None:
        assert "permission" not in config
    else:
        assert config["permission"] == expected_permission


@pytest.mark.parametrize(
    ("tool_policy", "expected_permission"),
    [
        (runtime.ToolPolicy.NONE, "deny"),
        (runtime.ToolPolicy.INSPECT_ONLY, {"edit": "deny", "bash": "deny"}),
        (runtime.ToolPolicy.NO_FILE_MUTATION, {"edit": "deny"}),
        (runtime.ToolPolicy.UNRESTRICTED, None),
    ],
)
def test_runtime_client_runs_opencode_new_session_with_tool_policy_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_policy: runtime.ToolPolicy,
    expected_permission: dict[str, str] | str | None,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-session-id",
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="opencode-key"),
                ),
                session_namespace="main",
                tool_access=_opencode_tool_access(tool_policy, tmp_path),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "final output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=7, output_tokens=3
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=_opencode_tool_access(tool_policy, tmp_path),
        provider_resume_state={
            "provider_session_id": "observed-session-id",
            "provider_state": {"session_id": "observed-session-id"},
            "exact_transcript_match": False,
        },
    )
    recorded_request = adapter.recorded_requests[0]
    config = json.loads(recorded_request.environment["OPENCODE_CONFIG_CONTENT"])
    if expected_permission is None:
        assert "permission" not in config
    else:
        assert config["permission"] == expected_permission


@pytest.mark.parametrize(
    ("tool_policy", "expected_permission"),
    [
        (runtime.ToolPolicy.NONE, "deny"),
        (runtime.ToolPolicy.INSPECT_ONLY, {"edit": "deny", "bash": "deny"}),
        (runtime.ToolPolicy.NO_FILE_MUTATION, {"edit": "deny"}),
        (runtime.ToolPolicy.UNRESTRICTED, None),
    ],
)
def test_runtime_client_runs_resumed_opencode_session_with_tool_policy_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_policy: runtime.ToolPolicy,
    expected_permission: dict[str, str] | str | None,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationResult(
            output="continued output",
            usage=runtime.ProviderUsage(
                input_tokens=7,
                output_tokens=2,
            ),
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "part": {
                            "type": "text",
                            "time": {"end": True},
                            "text": "continued output",
                        },
                    }
                )
                + "\n",
                json.dumps({"type": "session.status", "status": {"type": "idle"}})
                + "\n",
            ),
            provider_session_id="persisted-session-2",
        ),
    )

    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    provider_state_dir_relpath = "implementer/main/opencode/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    worktree.mkdir()
    provider_state_dir.mkdir(parents=True)
    (provider_state_dir / "resume.jsonl").write_text("[]\n", encoding="utf-8")
    (provider_state_dir / "session_id").write_text(
        "persisted-session-1\n",
        encoding="utf-8",
    )

    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=_opencode_tool_access(tool_policy, worktree),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": True,
        },
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=worktree,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=7, output_tokens=2
    )
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert outcome.result.continuation is not None
    assert outcome.result.continuation.provider_resume_state == {
        "provider_session_id": "persisted-session-2",
        "provider_state": {"session_id": "persisted-session-2"},
        "exact_transcript_match": False,
    }
    assert outcome.result.continuation.tool_access == _opencode_tool_access(
        tool_policy, worktree
    )

    recorded_request = adapter.recorded_requests[0]
    config = json.loads(recorded_request.environment["OPENCODE_CONFIG_CONTENT"])
    if expected_permission is None:
        assert "permission" not in config
    else:
        assert config["permission"] == expected_permission


def test_runtime_client_ephemeral_run_calls_live_output_observer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
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

    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    with pytest.raises(runtime.UsageLimitError):
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="codex",
                            model="gpt-5.4",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert observed == ["hello"]


def test_runtime_client_new_session_run_calls_live_output_observer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
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

    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("current invocation output"),
                json.dumps({"type": "turn.completed"}) + "\n",
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    _write_codex_rollout(provider_state_dir, "thread-123", "thread-123")

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].run_kind is RunKind.RESUME
    assert outcome.result.output == "current invocation output"
    assert observed == ["current invocation output"]


def test_runtime_client_new_session_run_forwards_live_output_observer_exceptions_as_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)
        raise runtime.UsageLimitError(service_name="codex")

    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _codex_assistant_output_line("hello"),
                _codex_assistant_output_line("world"),
            ),
        ),
    )
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    with pytest.raises(runtime.UsageLimitError):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="codex",
                            model="gpt-5.4",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert observed == ["hello"]


def test_runtime_client_resumed_session_run_calls_live_output_observer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                continuation=continuation,
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
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

    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    continuation=continuation,
                    on_live_output=on_live_output,
                )
            )
        )

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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE,
            detail=detail,
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                provider_selection=_selection_with_auth(
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

    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(_claude_assistant_output_line(("  hello ", "world")),),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer" / "main" / "claude"
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    (provider_state_dir / "session-state.json").write_text("{}", encoding="utf-8")

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _claude_assistant_output_line("intermediate"),
                _claude_result_output_line("final output"),
            ),
        ),
    )

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                ),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                on_live_output=on_live_output,
            )
        )
    )

    assert len(adapter.recorded_requests) == 1
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(tool_line, result_line),
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
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
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                    provider_selection=_selection_with_auth(
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
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="claude",
                            model="sonnet",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                    ),
                    session_namespace="main",
                    on_live_output=on_live_output,
                    tool_policy=runtime.ToolPolicy.NONE,
                )
            )
        )
    else:
        outcome = asyncio.run(
            client.run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_auth=runtime.ProviderAuth(
                        claude_code_oauth_token="oauth-token"
                    ),
                    continuation=prompt_runtime.Continuation(
                        selected_service="claude",
                        selected_model="sonnet",
                        selected_effort="medium",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state={
                            "run_kind": "resume",
                            "provider_session_id": "session-uuid",
                            "exact_transcript_match": False,
                        },
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
    assert len(adapter.recorded_requests) == 1


def test_runtime_client_new_session_run_propagates_claude_live_output_observer_failure_for_resumed_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer" / "main" / "claude"
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    (provider_state_dir / "session-state.json").write_text("{}", encoding="utf-8")

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(_claude_assistant_output_line("intermediate"),),
        ),
    )

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
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="claude",
                            model="sonnet",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                    ),
                    session_namespace="main",
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )

    assert len(adapter.recorded_requests) == 1


def test_runtime_client_new_opencode_session_calls_live_runtime_output_observer_once_per_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="opencode-key"),
                ),
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

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                    provider_selection=_selection_with_auth(
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
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="opencode",
                            model="kimi-k2.6",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(opencode_api_key="go-key"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                    on_live_output=on_live_output,
                )
            )
        )
    else:
        outcome = asyncio.run(
            client.run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                    continuation=prompt_runtime.Continuation(
                        selected_service="opencode",
                        selected_model="kimi-k2.6",
                        selected_effort="medium",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state={"provider_session_id": "sess_123"},
                    ),
                    on_live_output=on_live_output,
                )
            )
        )

    assert outcome.result.output == "hello\n\nsecond"
    assert observed == ["hello", "second"]
    assert len(adapter.recorded_requests) == 1


def test_runtime_client_opencode_live_runtime_output_stops_after_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def on_live_output(turn: runtime.AgentEvent) -> None:
        if turn.type == "agent_message":
            observed.append(turn.display_message)

    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                    provider_selection=_selection_with_auth(
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

    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
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
    adapter = provider_invocation_runtime.InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            provider_invocation_runtime.ProviderInvocationResult(
                output="continued output",
                usage=runtime.ProviderUsage(
                    input_tokens=7,
                    output_tokens=2,
                ),
                provider_session_id="observed-session",
            )
        ]
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )

    continuation = prompt_runtime.Continuation(
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

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                continuation=continuation,
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == tmp_path / ".provider_prompt"
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.environment == {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}
    assert "--resume claude-session-123" in recorded_request.command
    assert "--model sonnet" in recorded_request.command
    assert "--effort medium" in recorded_request.command


def test_runtime_client_runs_claude_resumed_session_from_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    continuation = prompt_runtime.Continuation(
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
    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                continuation=continuation,
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
    }
    assert "--resume claude-session-123" in recorded_request.command
    assert "--session-id" not in recorded_request.command
    assert "--model sonnet" in recorded_request.command
    assert "--effort medium" in recorded_request.command


@pytest.mark.parametrize(
    ("tool_policy", "expected_flags"),
    [
        (runtime.ToolPolicy.NONE, ('--disallowedTools "all"',)),
        (runtime.ToolPolicy.INSPECT_ONLY, ("--tools 'Read Glob'",)),
        (
            runtime.ToolPolicy.NO_FILE_MUTATION,
            ('--disallowedTools "Edit Write NotebookEdit"',),
        ),
        (runtime.ToolPolicy.UNRESTRICTED, tuple()),
    ],
)
def test_runtime_client_runs_claude_resumed_session_with_continuation_tool_policy_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_policy: runtime.ToolPolicy,
    expected_flags: tuple[str, ...],
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "continued output"}) + "\n",
            ),
        ),
    )

    tool_access = (
        contracts_runtime.ToolAccess.no_tools()
        if tool_policy is runtime.ToolPolicy.NONE
        else contracts_runtime.ToolAccess.workspace_backed(
            tmp_path, tool_policy=tool_policy
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                continuation=prompt_runtime.Continuation(
                    selected_service="claude",
                    selected_model="sonnet",
                    selected_effort="medium",
                    tool_access=tool_access,
                    provider_resume_state={
                        "run_kind": "resume",
                        "provider_session_id": "claude-session-123",
                        "exact_transcript_match": False,
                    },
                ),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "claude-session-123",
            "exact_transcript_match": False,
        },
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
    }
    command = recorded_request.command
    assert "--resume claude-session-123" in command
    if tool_policy is runtime.ToolPolicy.NONE:
        assert "--tools none" not in command
    elif tool_policy is runtime.ToolPolicy.INSPECT_ONLY:
        assert '--disallowedTools "all"' not in command
    elif tool_policy is runtime.ToolPolicy.NO_FILE_MUTATION:
        assert "--tools" not in command
    elif tool_policy is runtime.ToolPolicy.UNRESTRICTED:
        assert "--tools" not in command
        assert "--disallowedTools" not in command
    for flag in expected_flags:
        assert flag in command


@pytest.mark.parametrize(
    ("tool_access", "expected_flag"),
    [
        (
            contracts_runtime.ToolAccess.no_tools(),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.INSPECT_ONLY
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.UNRESTRICTED
            ),
            "--sandbox danger-full-access",
        ),
    ],
)
def test_runtime_client_runs_codex_resumed_session_through_built_in_provider_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_access: contracts_runtime.ToolAccess,
    expected_flag: str,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    _write_codex_rollout(provider_state_dir, "recovered-thread")
    continuation = prompt_runtime.Continuation(
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
            "provider_session_id": "selected-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == Path("/tmp/.provider_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "selected-thread"
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert recorded_request.command == (
        f"{_codex_executable()} exec resume selected-thread -m gpt-5.4 "
        f"-c model_reasoning_effort=medium -c approval_policy=never {expected_flag} "
        "--json"
    )
    assert recorded_request.prefer_argv is True
    assert recorded_request.argv[-3:] == (
        "--sandbox",
        expected_flag.removeprefix("--sandbox "),
        "--json",
    )
    assert (provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
    )


def test_runtime_client_resumes_codex_session_from_completed_new_session_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    new_outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
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
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                continuation=continuation,
                session_namespace="main",
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

    assert adapter.recorded_requests[0].command.startswith(
        f"{_codex_executable()} exec -m gpt-5.4"
    )
    assert adapter.recorded_requests[1].command == (
        f"{_codex_executable()} exec resume thread-123 -m gpt-5.4 -c model_reasoning_effort=medium -c approval_policy=never "
        "--sandbox read-only --json"
    )


def test_runtime_client_runs_codex_resumed_session_from_continuation_without_portable_state_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    continuation = prompt_runtime.Continuation(
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

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                continuation=continuation,
                session_namespace="main",
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
    assert adapter.recorded_requests[0].environment == {"TZ": "UTC"}
    assert adapter.recorded_requests[0].command == (
        f"{_codex_executable()} exec resume selected-thread -m gpt-5.4 -c model_reasoning_effort=medium -c approval_policy=never "
        "--sandbox read-only --json"
    )


def test_runtime_client_preserves_tool_policy_in_resumed_session_usage_limited_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    _write_codex_rollout(provider_state_dir, "recovered-thread")
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            tmp_path,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "selected-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )

    first = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
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
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=first.result.continuation,
                session_namespace="main",
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
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        failure,
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    _write_codex_rollout(provider_state_dir, "recovered-thread")
    if entrypoint == "new":
        outcome = asyncio.run(
            runtime.RuntimeClient().run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
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
        expected_recorded_provider_session_id = "recovered-thread"
    else:
        continuation = prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "selected-thread",
                "provider_state_dir_relpath": "implementer/main/codex/",
                "exact_transcript_match": False,
            },
        )

        outcome = asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                    session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.path == Path("/tmp/.provider_prompt")
    assert recorded_request.prompt.cleanup_path is True
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
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "generated output"}) + "\n",
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir_relpath = "implementer/main/claude/"
    continuation = prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": False,
        },
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "generated-session-id"
    assert recorded_request.environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
        "CLAUDE_CONFIG_DIR": str(runtime_state_dir / provider_state_dir_relpath),
    }
    assert "--session-id generated-session-id" in recorded_request.command
    assert "--resume" not in recorded_request.command


@pytest.mark.parametrize("create_state_dir", [False, True])
def test_runtime_client_runs_claude_resumed_session_fresh_when_provider_state_is_not_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    create_state_dir: bool,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    continuation = prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "claude-session-123",
            "provider_state_dir_relpath": provider_state_dir_relpath,
            "exact_transcript_match": False,
        },
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
        "CLAUDE_CONFIG_DIR": str(runtime_state_dir / provider_state_dir_relpath),
    }
    assert "--session-id claude-session-123" in recorded_request.command
    assert "--resume" not in recorded_request.command


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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=_selection_with_auth(
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
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                provider_selection=_selection_with_auth(
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
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                '{"type":"thread.started","thread_id":"thread-123"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"continued output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == Path("/tmp/.provider_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id is None
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert recorded_request.command == (
        f"{_codex_executable()} exec -m gpt-5.4 -c model_reasoning_effort=medium "
        "-c approval_policy=never --sandbox read-only --json"
    )
    assert recorded_request.prefer_argv is True
    assert recorded_request.argv == (
        _codex_executable(),
        "exec",
        "-m",
        "gpt-5.4",
        "-c",
        "model_reasoning_effort=medium",
        "-c",
        "approval_policy=never",
        "--sandbox",
        "read-only",
        "--json",
    )
    assert (provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
    )


def test_runtime_client_runs_codex_new_session_as_resume_for_deduplicated_rollout_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                '{"type":"item.completed","item":{"type":"agent_message","text":"continued output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    _write_codex_rollout(provider_state_dir, "thread-123", "thread-123")

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
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

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "continued output"
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.path == Path("/tmp/.provider_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "thread-123"
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert recorded_request.command == (
        f"{_codex_executable()} exec resume thread-123 -m gpt-5.4 "
        "-c model_reasoning_effort=medium -c approval_policy=never --sandbox read-only --json"
    )


def test_runtime_client_runs_codex_resumed_session_for_selected_continuation_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationResult(
            output="continued output",
            usage=runtime.ProviderUsage(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=1,
            ),
            stdout_lines=(
                '{"type":"item.completed","item":{"type":"agent_message","text":"continued output"}}\n',
                '{"type":"turn.completed","usage":{"input_tokens":3,"cached_tokens":1,"output_tokens":2}}\n',
            ),
            provider_session_id=None,
        ),
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "selected-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    _write_codex_rollout(provider_state_dir, "recovered-thread")

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
            )
        )
    )

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
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "selected-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.path == Path("/tmp/.provider_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "selected-thread"
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert recorded_request.command == (
        f"{_codex_executable()} exec resume selected-thread -m gpt-5.4 "
        "-c model_reasoning_effort=medium -c approval_policy=never --sandbox read-only --json"
    )


def test_runtime_client_keeps_started_codex_new_session_continuation_when_output_reduction_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail="Usage limit reached (reset_time=None)",
            stdout_lines=('{"type":"thread.started","thread_id":"thread-123"}\n',),
            provider_session_id=None,
        ),
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
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

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time is None
    assert outcome.result.output == ""
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
    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].provider_session_id is None


def test_runtime_client_keeps_started_codex_resumed_session_continuation_when_output_reduction_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail="Usage limit reached (reset_time=None)",
            stdout_lines=('{"type":"thread.started","thread_id":"thread-456"}\n',),
            provider_session_id=None,
        ),
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "selected-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    _write_codex_rollout(provider_state_dir, "recovered-thread")

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time is None
    assert outcome.result.output == ""
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
            "provider_session_id": "thread-456",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].provider_session_id == "selected-thread"


@pytest.mark.parametrize("entrypoint", ["new", "resumed"])
def test_runtime_client_session_backed_codex_outcome_includes_output_and_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    _ = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationResult(
            output="final output",
            usage=runtime.ProviderUsage(
                input_tokens=5,
                output_tokens=2,
            ),
            provider_session_id="thread-obs",
            stdout_lines=(),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "selected-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    if entrypoint == "new":
        outcome = asyncio.run(
            runtime.RuntimeClient().run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="codex",
                            model="gpt-5.4",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
                    ),
                    session_namespace="main",
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )
        expected_provider_session_id = "thread-obs"
    else:
        provider_state_dir = runtime_state_dir / "implementer/main/codex"
        _write_codex_rollout(provider_state_dir, "selected-thread")
        outcome = asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                    session_namespace="main",
                )
            )
        )
        expected_provider_session_id = "thread-obs"

    assert outcome.result.output == "final output"
    assert outcome.result is not None
    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=5,
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
            "provider_session_id": expected_provider_session_id,
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_rejects_codex_resumed_session_for_ambiguous_rollout_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "selected-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    _write_codex_rollout(provider_state_dir, "thread-a")
    (
        provider_state_dir / "sessions" / "2026" / "05" / "30" / "rollout-002.jsonl"
    ).write_text('{"type":"thread.started","thread_id":"thread-b"}\n', encoding="utf-8")

    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                    session_namespace="main",
                )
            )
        )

    assert str(exc_info.value) == (
        "Codex continuation is not recoverable from provider state."
    )
    assert adapter.recorded_requests == []


def test_runtime_client_rejects_codex_resumed_session_for_malformed_rollout_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )

    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "selected-thread",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
        },
    )
    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/codex"
    rollout_dir = provider_state_dir / "sessions" / "2026" / "05" / "30"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "rollout-001.jsonl").write_text(
        "\n".join(
            [
                "{not-json",
                "[]",
                '{"type":"turn.completed"}',
                '{"type":"thread.started","thread_id":"   "}',
                '{"type":"thread.started"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                    session_namespace="main",
                )
            )
        )

    assert str(exc_info.value) == (
        "Codex continuation is not recoverable from provider state."
    )
    assert adapter.recorded_requests == []


def test_runtime_client_rejects_codex_resumed_session_without_usable_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)

    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    continuation=prompt_runtime.Continuation(
                        selected_service="codex",
                        selected_model="gpt-5.4",
                        selected_effort="medium",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state={
                            "run_kind": "resume",
                            "provider_session_id": "   ",
                            "exact_transcript_match": False,
                        },
                    ),
                    session_namespace="main",
                )
            )
        )

    assert str(exc_info.value) == (
        "Codex continuation is missing `provider_session_id`."
    )
    assert adapter.recorded_requests == []


def test_runtime_client_rejects_resumed_session_with_non_object_portable_continuation_resume_state(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                    continuation=prompt_runtime.Continuation(
                        selected_service="codex",
                        selected_model="gpt-5.4",
                        selected_effort="medium",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state=["resume"],
                    ),
                )
            )
        )

    assert str(exc_info.value) == (
        "Continuation provider_resume_state must be a JSON object."
    )


def test_runtime_client_rejects_resumed_session_with_malformed_continuation_data(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    continuation=prompt_runtime.Continuation(serialized="{not-json"),
                )
            )
        )

    assert str(exc_info.value) == "Continuation data is not valid JSON."


def test_runtime_client_rejects_new_session_for_unsupported_session_backed_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_PORTABLE_CONTINUATION_PROVIDERS",
        frozenset({"claude"}),
    )

    with pytest.raises(RuntimeConfigurationError, match="Portable continuation"):
        asyncio.run(
            runtime.RuntimeClient().run_new_session(
                prompt_runtime.NewSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="opencode",
                            model="deepseek-v4-flash",
                            effort="medium",
                        ),
                        runtime.ProviderAuth(opencode_api_key="api-key"),
                    ),
                    session_namespace="main",
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )
    assert adapter.recorded_requests == []


def test_runtime_client_rejects_resumed_session_for_unsupported_session_backed_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_PORTABLE_CONTINUATION_PROVIDERS",
        frozenset({"claude"}),
    )

    with pytest.raises(RuntimeConfigurationError, match="Portable continuation"):
        asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                prompt_runtime.ResumedSessionRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                    continuation=prompt_runtime.Continuation(
                        selected_service="opencode",
                        selected_model="deepseek-v4-flash",
                        selected_effort="medium",
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                        provider_resume_state={
                            "run_kind": "resume",
                            "provider_session_id": "restored-session",
                            "exact_transcript_match": False,
                        },
                    ),
                    session_namespace="main",
                )
            )
        )
    assert adapter.recorded_requests == []


@pytest.mark.parametrize("entrypoint", ["new", "resumed"])
def test_runtime_client_requires_host_codex_auth_for_session_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: tmp_path / "missing-home",
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        if entrypoint == "new":
            asyncio.run(
                runtime.RuntimeClient().run_new_session(
                    prompt_runtime.NewSessionRunRequest(
                        prompt="already rendered prompt",
                        invocation_dir=tmp_path,
                        runtime_state_dir=runtime_state_dir,
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
        else:
            asyncio.run(
                runtime.RuntimeClient().run_resumed_session(
                    prompt_runtime.ResumedSessionRunRequest(
                        prompt="already rendered prompt",
                        invocation_dir=tmp_path,
                        runtime_state_dir=runtime_state_dir,
                        continuation=prompt_runtime.Continuation(
                            selected_service="codex",
                            selected_model="gpt-5.4",
                            selected_effort="medium",
                            tool_access=contracts_runtime.ToolAccess.no_tools(),
                            provider_resume_state={
                                "run_kind": "resume",
                                "provider_session_id": "selected-thread",
                                "provider_state_dir_relpath": "implementer/main/codex/",
                                "exact_transcript_match": False,
                            },
                        ),
                        session_namespace="main",
                    )
                )
            )

    assert str(exc_info.value) == (
        "Codex authentication missing: run `codex login` on the host."
    )
    assert exc_info.value.service_name == "codex"
    assert adapter.recorded_requests == []


def test_runtime_client_treats_nested_claude_provider_state_as_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "session-uuid",
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    assert "--resume session-uuid" in recorded_request.command
    assert "--session-id" not in recorded_request.command


@pytest.mark.parametrize(
    (
        "service_name",
        "stage",
        "auth",
        "prepared_invocation",
        "expected_output",
        "expected_usage",
        "expected_prompt_path",
        "expected_env",
        "expected_command_parts",
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
            lambda worktree: worktree / ".provider_prompt",
            {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"},
            ("claude", "--output-format stream-json", "--model sonnet"),
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
            lambda _worktree: Path("/tmp/.provider_prompt"),
            {"TZ": "UTC"},
            (
                f"{_codex_executable()} exec",
                "-m gpt-5.4",
                "-c model_reasoning_effort=medium",
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
            lambda _worktree: Path("/tmp/.provider_prompt"),
            {
                "TZ": "UTC",
                "OPENCODE_GO_API_KEY": "go-key",
            },
            (
                f"{prompt_runtime._opencode_command(model='kimi-k2.6', effort='medium')[0]} run",
                "--format json",
                "--model opencode-go/kimi-k2.6",
            ),
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
    expected_prompt_path: Callable[[Path], Path],
    expected_env: dict[str, str],
    expected_command_parts: tuple[str, ...],
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        prepared_invocation,
    )
    if service_name == "codex":
        host_home = tmp_path / "host-home"
        host_auth_path = host_home / ".codex" / "auth.json"
        host_auth_path.parent.mkdir(parents=True, exist_ok=True)
        host_auth_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            prompt_runtime._builtin_runtime_client_module.Path,
            "home",
            lambda: host_home,
        )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    stage,
                    auth,
                ),
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == expected_prompt_path(tmp_path)
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id is None
    assert recorded_request.log_context is None
    for key, value in expected_env.items():
        assert recorded_request.environment[key] == value
    if service_name == "claude":
        assert recorded_request.environment == expected_env
    if service_name == "opencode":
        config = json.loads(recorded_request.environment["OPENCODE_CONFIG_CONTENT"])
        provider = config["provider"]["opencode-go"]
        assert provider["options"] == {
            "baseURL": "https://opencode.ai/zen/go/v1",
            "apiKey": "{env:OPENCODE_GO_API_KEY}",
        }
        assert "kimi-k2.6" in provider["models"]
        assert "deepseek-v4-flash" in provider["models"]
    if service_name != "claude":
        assert recorded_request.prefer_argv is True
    for command_part in expected_command_parts:
        assert command_part in recorded_request.command
    assert list((tmp_path / "logs").glob("*.log")) == []


def test_opencode_command_uses_cmd_shim_on_windows() -> None:
    assert prompt_runtime._opencode_command(
        model="kimi-k2.6",
        effort="medium",
        os_name="nt",
    ) == (
        "opencode.cmd",
        "run",
        "--format",
        "json",
        "--model",
        "opencode-go/kimi-k2.6",
    )


def test_codex_command_uses_cmd_shim_on_windows() -> None:
    assert prompt_runtime._builtin_runtime_client_module._codex_command(
        model="gpt-5.4-mini",
        effort="low",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        os_name="nt",
    )[:2] == ("codex.cmd", "exec")


def test_opencode_env_includes_only_windows_process_launch_allowlist() -> None:
    env = prompt_runtime._opencode_env(
        auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        state_dir_container_path="state-dir",
        tool_policy=runtime.ToolPolicy.NONE,
        os_name="nt",
        environ={
            "PATH": "path-value",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "SystemRoot": "C:\\Windows",
            "ComSpec": "C:\\Windows\\System32\\cmd.exe",
            "WINDIR": "C:\\Windows",
            "SHOULD_NOT_LEAK": "host-value",
        },
    )

    assert env["PATH"] == "path-value"
    assert env["PATHEXT"] == ".COM;.EXE;.BAT;.CMD"
    assert env["SystemRoot"] == "C:\\Windows"
    assert env["ComSpec"] == "C:\\Windows\\System32\\cmd.exe"
    assert env["WINDIR"] == "C:\\Windows"
    assert env["TZ"] == "UTC"
    assert env["OPENCODE_HOME"] == "state-dir"
    assert env["OPENCODE_GO_API_KEY"] == "go-key"
    assert "OPENCODE_CONFIG_CONTENT" in env
    assert "SHOULD_NOT_LEAK" not in env


@pytest.mark.parametrize(
    ("service_name", "stage", "auth", "expected_argv"),
    [
        pytest.param(
            "codex",
            InternalStageSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            None,
            (
                _codex_executable(),
                "exec",
                "-m",
                "gpt-5.4",
                "-c",
                "model_reasoning_effort=medium",
                "-c",
                "approval_policy=never",
                "--sandbox",
                "read-only",
                "--json",
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
            prompt_runtime._opencode_command(model="kimi-k2.6", effort="medium"),
            id="opencode",
        ),
    ],
)
def test_runtime_client_ephemeral_non_claude_invocation_prefers_argv_prompt_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service_name: str,
    stage: InternalStageSelection,
    auth: runtime.ProviderAuth | None,
    expected_argv: tuple[str, ...],
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps({"type": "result", "result": "final output"}) + "\n",
            ),
        ),
    )
    if service_name == "codex":
        host_home = tmp_path / "host-home"
        host_auth_path = host_home / ".codex" / "auth.json"
        host_auth_path.parent.mkdir(parents=True, exist_ok=True)
        host_auth_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            prompt_runtime._builtin_runtime_client_module.Path,
            "home",
            lambda: host_home,
        )

    asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    stage,
                    auth,
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prefer_argv is True
    assert recorded_request.argv == expected_argv
    assert "< /tmp/.provider_prompt" not in recorded_request.command
    assert '"$(cat /tmp/.provider_prompt)"' not in recorded_request.command


def test_runtime_client_ephemeral_execution_remains_available_when_session_backed_support_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
    adapter = provider_invocation_runtime.InMemoryProviderInvocationAdapter(
        prepared_invocations=[
            provider_invocation_runtime.ProviderInvocationResult(
                output="continued output",
                provider_session_id="persisted-session-2",
            )
        ]
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )

    worktree = tmp_path / "worktree"
    runtime_state_dir = tmp_path / "runtime-state"
    provider_state_dir_relpath = "implementer/main/opencode/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath
    worktree.mkdir()
    provider_state_dir.mkdir(parents=True)
    (provider_state_dir / "resume.jsonl").write_text("[]", encoding="utf-8")
    (provider_state_dir / "session_id").write_text(
        "persisted-session-1\n",
        encoding="utf-8",
    )
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": True,
        },
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=worktree,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == worktree / ".provider_prompt"
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == worktree
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "persisted-session-1"
    assert Path(recorded_request.environment["OPENCODE_HOME"]).name == "opencode"
    assert recorded_request.environment["OPENCODE_GO_API_KEY"] == "go-key"
    assert "--session persisted-session-1" in recorded_request.command
    assert "--model opencode-go/glm-5.2" in recorded_request.command


def test_runtime_client_uses_observed_opencode_resumed_session_id_for_started_usage_limited_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1,
                        "sessionID": "observed-session-2",
                        "part": {
                            "type": "text",
                            "text": "started before failure",
                            "time": {"end": True},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 2,
                        "sessionID": "observed-session-2",
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
            ),
            provider_session_id="adapter-session-2",
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": True,
        },
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time == datetime(2026, 4, 28, 21, 2, tzinfo=timezone.utc)
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "observed-session-2",
            "provider_state": {
                "session_id": "observed-session-2",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": False,
        },
    )
    assert adapter.recorded_requests[0].provider_session_id == "persisted-session-1"
    provider_state_dir = runtime_state_dir / "implementer" / "main" / "opencode"
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "observed-session-2"
    )


def test_runtime_client_keeps_started_opencode_new_session_continuation_when_output_reduction_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_new_provider_session_id",
        lambda: "prepared-session-id",
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 1,
                        "sessionID": "provider-session-777",
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
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                session_namespace="main",
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time == datetime(2026, 4, 28, 21, 2, tzinfo=timezone.utc)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
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
    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].provider_session_id == "prepared-session-id"
    provider_state_dir = runtime_state_dir / "implementer" / "main" / "opencode"
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "provider-session-777"
    )


def test_runtime_client_passes_only_claude_specific_env_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SHOULD_NOT_LEAK", "host-value")
    captured: dict[str, Any] = {}

    class _ClaudeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [json.dumps({"type": "result", "result": "final output"}) + "\n"]
            )
            self.stderr = iter(())
            self.returncode = 0

        def wait(self) -> int:
            return 0

    def _fake_popen(
        command: str,
        *,
        shell: bool,
        cwd: Path,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        text: bool,
    ) -> _ClaudeProcess:
        captured["env"] = env
        return _ClaudeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
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

    assert captured["env"] == {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}


@pytest.mark.parametrize(
    ("tool_access", "expected_flag"),
    [
        (
            contracts_runtime.ToolAccess.no_tools(),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.INSPECT_ONLY
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION
            ),
            "--sandbox read-only",
        ),
        (
            contracts_runtime.ToolAccess.workspace_backed(
                Path("."), tool_policy=runtime.ToolPolicy.UNRESTRICTED
            ),
            "--sandbox danger-full-access",
        ),
    ],
)
def test_runtime_client_runs_codex_new_session_through_built_in_provider_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_access: contracts_runtime.ToolAccess,
    expected_flag: str,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                session_namespace="main",
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
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == Path("/tmp/.provider_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id is None
    assert recorded_request.log_context is None
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert expected_flag in recorded_request.command
    assert (provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
    )


def test_runtime_client_keeps_started_codex_new_session_continuation_from_provider_invocation_failure_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text('{"token":"host-auth"}\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
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
    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].log_context is None


def test_runtime_client_returns_run_result_for_session_run_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationResult(
            output="continued output",
            usage=runtime.ProviderUsage(
                input_tokens=5,
                output_tokens=2,
                cache_read_input_tokens=1,
                cache_creation_input_tokens=0,
            ),
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "session-123",
                        "part": {
                            "type": "text",
                            "text": "continued output",
                            "time": {"end": "2026-01-01T00:00:00Z"},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "session.status",
                        "status": {"type": "idle"},
                    }
                )
                + "\n",
            ),
            provider_session_id="session-123",
        ),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_new_session(
            prompt_runtime.NewSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome.result.output == "continued output"
    assert outcome.result.usage == runtime.ProviderUsage(
        input_tokens=5,
        output_tokens=2,
        cache_read_input_tokens=1,
        cache_creation_input_tokens=0,
    )
    assert isinstance(outcome.result, prompt_runtime.RunResult)
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "session-123",
            "provider_state": {"session_id": "session-123"},
            "exact_transcript_match": False,
        },
    )


def test_runtime_client_keeps_started_opencode_resumed_session_continuation_when_output_reduction_interrupts_without_new_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "text",
                        "timestamp": 1,
                        "part": {
                            "type": "text",
                            "text": "started before failure",
                            "time": {"end": True},
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 2,
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
            ),
        ),
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": True,
        },
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.UsageLimited)
    assert outcome.kind.reset_time == datetime(2026, 4, 28, 21, 2, tzinfo=timezone.utc)
    assert outcome.result.output == ""
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert outcome.result.continuation == prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            "exact_transcript_match": True,
        },
    )
    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].provider_session_id == "persisted-session-1"


def test_runtime_client_preserves_opencode_invalid_api_key_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                    provider_selection=_selection_with_auth(
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


def test_runtime_client_opencode_allowlist_accepts_current_models_and_rejects_stale_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for model in _CURRENT_OPENCODE_GO_MODELS:
        adapter = _install_in_memory_provider_invocation_adapter(
            monkeypatch,
            provider_invocation_runtime.ProviderInvocationResult(
                output=f"hello from {model}"
            ),
        )
        outcome = asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    worktree=tmp_path,
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="opencode",
                            model=model,
                            effort="medium",
                        ),
                        runtime.ProviderAuth(opencode_api_key="go-key"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )
        assert isinstance(outcome.kind, prompt_runtime.Completed)
        assert outcome.result.selected.model == model
        assert len(adapter.recorded_requests) == 1
        assert f"--model opencode-go/{model}" in adapter.recorded_requests[0].command

    for model in _STALE_OPENCODE_GO_MODELS:
        adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)
        with pytest.raises(
            RuntimeConfigurationError, match="Unsupported OpenCode model"
        ):
            asyncio.run(
                runtime.RuntimeClient().run_ephemeral(
                    prompt_runtime.EphemeralRunRequest(
                        prompt="already rendered prompt",
                        worktree=tmp_path,
                        provider_selection=_selection_with_auth(
                            InternalStageSelection(
                                service="opencode",
                                model=model,
                                effort="medium",
                            ),
                            runtime.ProviderAuth(opencode_api_key="go-key"),
                        ),
                        tool_access=contracts_runtime.ToolAccess.no_tools(),
                    )
                )
            )
        assert adapter.recorded_requests == []


def test_runtime_client_opencode_config_exposes_current_subscription_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current_models = set(_CURRENT_OPENCODE_GO_MODELS)
    stale_models = set(_STALE_OPENCODE_GO_MODELS)
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationResult(output="hello"),
    )

    asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="deepseek-v4-flash",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    config = json.loads(
        adapter.recorded_requests[0].environment["OPENCODE_CONFIG_CONTENT"]
    )
    configured_models = set(config["provider"]["opencode-go"]["models"])
    assert configured_models == current_models
    assert not configured_models.intersection(stale_models)


def test_runtime_client_opencode_command_uses_prefixed_model_reference_while_public_selection_stays_unprefixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = "glm-5.2"
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationResult(output="hello"),
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model=model,
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome.result.selected.model == model
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert f"--model opencode-go/{model}" in recorded_request.command
    assert f"--model {model}" not in recorded_request.command
    assert f"--model opencode-go/opencode-go/{model}" not in recorded_request.command


@pytest.mark.parametrize(
    ("model", "effort", "expected_message"),
    [
        ("not-a-real-model", "medium", "Unsupported OpenCode model"),
        ("kimi-k2.6", "high", "Unsupported OpenCode effort"),
    ],
)
def test_runtime_client_validates_opencode_model_allowlist_and_medium_effort(
    tmp_path: Path,
    model: str,
    effort: str,
    expected_message: str,
) -> None:
    with pytest.raises(RuntimeConfigurationError, match=expected_message):
        asyncio.run(
            runtime.RuntimeClient().run_ephemeral(
                prompt_runtime.EphemeralRunRequest(
                    prompt="already rendered prompt",
                    invocation_dir=tmp_path,
                    provider_selection=_selection_with_auth(
                        InternalStageSelection(
                            service="opencode",
                            model=model,
                            effort=effort,
                        ),
                        runtime.ProviderAuth(opencode_api_key="go-key"),
                    ),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        )


def test_runtime_client_maps_opencode_usage_limit_after_ignoring_malformed_and_non_text_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
    assert recorded_request.prompt.path == Path("/tmp/.provider_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.environment["TZ"] == "UTC"
    assert f"{_codex_executable()} exec" in recorded_request.command
    assert list((tmp_path / "logs").glob("*.log")) == []


def test_runtime_client_reused_after_usage_limited_ephemeral_call_still_invokes_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
    host_home = tmp_path / "host-home"
    host_auth_path = host_home / ".codex" / "auth.json"
    host_auth_path.parent.mkdir(parents=True, exist_ok=True)
    host_auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: host_home,
    )
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.USAGE_LIMITED,
            detail="Usage limit reached (reset_time=2026-01-02T17:00:00+00:00)",
            reset_time=datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc),
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
    assert outcome.result.continuation is None


def test_runtime_client_maps_opencode_missing_model_without_status_to_hard_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                    provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                    provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                    provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                provider_selection=_selection_with_auth(
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
    _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
                    provider_selection=_selection_with_auth(
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
    import time

    events_emitted: list[prompt_runtime.AgentEvent] = []

    def collect_events(event: prompt_runtime.AgentEvent) -> None:
        events_emitted.append(event)

    class SlowProviderAdapter(
        provider_invocation_runtime.InMemoryProviderInvocationAdapter
    ):
        """Provider adapter that emits output after the timeout window."""

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            self.recorded_requests.append(request)
            if not self.prepared_invocations:
                raise AssertionError("No prepared provider invocation remains.")
            prepared = self.prepared_invocations.pop(0)
            if isinstance(
                prepared, provider_invocation_runtime.ProviderInvocationPreparedStream
            ):
                stdout_lines = list(prepared.stdout_lines)
                # Delay before processing output, which triggers the timeout
                time.sleep(2)
                # Now emit the output - but timeout should trigger during this call
                provider_invocation_runtime._consume_new_stdout_lines(
                    request.output_hooks.reduce_output, stdout_lines
                )
                output, usage = request.output_hooks.reduce_output(stdout_lines)
                return provider_invocation_runtime.ProviderInvocationResult(
                    output=output,
                    usage=usage,
                    stdout_lines=tuple(stdout_lines),
                    provider_session_id=(
                        prepared.provider_session_id or request.provider_session_id
                    ),
                )
            assert isinstance(
                prepared, provider_invocation_runtime.ProviderInvocationResult
            )
            return prepared

    adapter = SlowProviderAdapter(
        prepared_invocations=[
            provider_invocation_runtime.ProviderInvocationPreparedStream(
                stdout_lines=(
                    json.dumps({"type": "session.status", "status": {"type": "idle"}}),
                )
            ),
        ]
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=1,
                on_live_output=collect_events,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.selected.service == "opencode"


def test_idle_timeout_defaults_to_300_seconds_on_all_lifecycle_requests(
    tmp_path: Path,
) -> None:
    """All three lifecycle request types default to a 300s idle timeout."""
    import inspect

    ephemeral = prompt_runtime.EphemeralRunRequest(
        prompt="test",
        invocation_dir=tmp_path,
        provider_selection=_selection_with_auth(
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
        provider_selection=_selection_with_auth(
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


def test_runtime_client_ephemeral_times_out_without_live_output_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Idle timeout fires even when no on_live_output callback is provided."""
    import time

    class SlowProvider(provider_invocation_runtime.InMemoryProviderInvocationAdapter):
        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            self.recorded_requests.append(request)
            if not self.prepared_invocations:
                raise AssertionError("No prepared provider invocation remains.")
            prepared = self.prepared_invocations.pop(0)
            assert isinstance(
                prepared, provider_invocation_runtime.ProviderInvocationPreparedStream
            )
            stdout_lines = list(prepared.stdout_lines)
            time.sleep(2)
            provider_invocation_runtime._consume_new_stdout_lines(
                request.output_hooks.reduce_output, stdout_lines
            )
            output, usage = request.output_hooks.reduce_output(stdout_lines)
            return provider_invocation_runtime.ProviderInvocationResult(
                output=output,
                usage=usage,
                stdout_lines=tuple(stdout_lines),
                provider_session_id=(
                    prepared.provider_session_id or request.provider_session_id
                ),
            )

    adapter = SlowProvider(
        prepared_invocations=[
            provider_invocation_runtime.ProviderInvocationPreparedStream(
                stdout_lines=(
                    json.dumps({"type": "session.status", "status": {"type": "idle"}}),
                )
            ),
        ]
    )
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                invocation_dir=tmp_path,
                provider_selection=_selection_with_auth(
                    InternalStageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                    ),
                    runtime.ProviderAuth(opencode_api_key="go-key"),
                ),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
                timeout_seconds=1,
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.TimedOut)
    assert outcome.result.selected.service == "opencode"
