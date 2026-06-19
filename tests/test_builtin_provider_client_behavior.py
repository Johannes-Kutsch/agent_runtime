from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

import agent_runtime as runtime
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.errors import (
    AgentCredentialFailureError,
    HardAgentError,
    RuntimeConfigurationError,
    TransientAgentError,
)
from agent_runtime.provider_errors import ProviderErrorObservation
from agent_runtime.roles import InvocationRole
from agent_runtime.session import RunKind


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
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    provider_state_dir_relpath = "implementer/main/claude/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="final output",
        result=prompt_runtime.SessionRunResult(
            output="final output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="claude",
                provider_session_id="session-uuid",
                run_kind=RunKind.FRESH,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="sonnet",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "session-uuid",
                    "provider_state_dir_relpath": provider_state_dir_relpath,
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=5,
            output_tokens=None,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cost_usd=None,
            duration_seconds=None,
        ),
    )
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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        worktree=tmp_path,
        runtime_state_dir=runtime_state_dir,
        stage=runtime.StageSelection(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        role=InvocationRole("implementer"),
        session_namespace="main",
        provider_auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        tool_access=runtime.ToolAccess.no_tools(),
    )
    outcome = prompt_runtime._run_builtin_session_outcome(
        lambda: prompt_runtime._builtin_runtime_client_module._run_builtin_new_session(
            request,
            provider_invocation_adapter=adapter,
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="final output",
        result=prompt_runtime.SessionRunResult(
            output="final output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="claude",
                provider_session_id="observed-session",
                run_kind=RunKind.FRESH,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="sonnet",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "observed-session",
                    "provider_state_dir_relpath": "implementer/main/claude/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=5,
            output_tokens=2,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.role == InvocationRole("implementer")
    assert recorded_request.usage_limit_scope is None
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
                    '{"type":"text","sessionID":"observed-session-id","part":{"type":"text","text":"thinking"}}\n',
                ),
                provider_session_id="observed-session-id",
            )
        ]
    )

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    request = prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        worktree=tmp_path,
        runtime_state_dir=runtime_state_dir,
        stage=runtime.StageSelection(
            service="opencode",
            model="glm-5",
            effort="medium",
        ),
        role=InvocationRole("implementer"),
        session_namespace="main",
        provider_auth=runtime.ProviderAuth(opencode_api_key="opencode-key"),
        tool_access=runtime.ToolAccess.no_tools(),
    )
    outcome = prompt_runtime._run_builtin_session_outcome(
        lambda: prompt_runtime._builtin_runtime_client_module._run_builtin_new_session(
            request,
            provider_invocation_adapter=adapter,
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="final output",
        result=prompt_runtime.SessionRunResult(
            output="final output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="opencode",
                provider_session_id="observed-session-id",
                run_kind=RunKind.FRESH,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="opencode",
                selected_model="glm-5",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "provider_session_id": "observed-session-id",
                    "provider_state_dir_relpath": "implementer/main/opencode/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=7,
            output_tokens=3,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.role == InvocationRole("implementer")
    assert recorded_request.usage_limit_scope is None
    assert recorded_request.provider_session_id == "prepared-session-id"
    provider_state_dir = runtime_state_dir / "implementer" / "main" / "opencode"
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "observed-session-id"
    )


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

    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/claude"
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    (provider_state_dir / "session.jsonl").write_text("{}\n", encoding="utf-8")
    continuation = prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "claude-session-123",
            "provider_state_dir_relpath": "implementer/main/claude/",
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
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                model="opus",
                effort="high",
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="continued output",
        result=prompt_runtime.SessionRunResult(
            output="continued output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="claude",
                provider_session_id="observed-session",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="opus",
                selected_effort="high",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "observed-session",
                    "provider_state_dir_relpath": "implementer/main/claude/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=7,
            output_tokens=2,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == tmp_path / ".pycastle_prompt"
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.role == InvocationRole("implementer")
    assert recorded_request.usage_limit_scope is None
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.environment["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"
    assert recorded_request.environment["CLAUDE_CONFIG_DIR"] == str(provider_state_dir)
    assert "--resume claude-session-123" in recorded_request.command
    assert "--model opus" in recorded_request.command
    assert "--effort high" in recorded_request.command


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
        tool_access=runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "claude-session-123",
            "provider_state_dir_relpath": "implementer/main/claude/",
            "exact_transcript_match": False,
        },
    )
    runtime_state_dir = tmp_path / ".agent-runtime" / "state"
    provider_state_dir = runtime_state_dir / "implementer/main/claude"
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    (provider_state_dir / "session.jsonl").write_text("{}\n", encoding="utf-8")

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            prompt_runtime.ResumedSessionRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                model="opus",
                effort="high",
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="continued output",
        result=prompt_runtime.SessionRunResult(
            output="continued output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="claude",
                provider_session_id="claude-session-123",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="opus",
                selected_effort="high",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "claude-session-123",
                    "provider_state_dir_relpath": "implementer/main/claude/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=7,
            output_tokens=None,
            cache_read_input_tokens=1,
            cache_creation_input_tokens=0,
            cost_usd=None,
            duration_seconds=None,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "claude-session-123"
    assert recorded_request.environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
        "CLAUDE_CONFIG_DIR": str(provider_state_dir),
    }
    assert "--resume claude-session-123" in recorded_request.command
    assert "--session-id" not in recorded_request.command
    assert "--model opus" in recorded_request.command
    assert "--effort high" in recorded_request.command


def test_runtime_client_runs_codex_resumed_session_through_built_in_provider_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        tool_access=runtime.ToolAccess.no_tools(),
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="continued output",
        result=prompt_runtime.SessionRunResult(
            output="continued output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="codex",
                provider_session_id="observed-thread",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "observed-thread",
                    "provider_state_dir_relpath": "implementer/main/codex/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=7,
            output_tokens=2,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == Path("/tmp/.pycastle_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.role == InvocationRole("implementer")
    assert recorded_request.usage_limit_scope is None
    assert recorded_request.provider_session_id == "selected-thread"
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert recorded_request.command == (
        "codex exec resume selected-thread -m gpt-5.4 "
        "-c model_reasoning_effort=medium -c approval_policy=never "
        "--sandbox danger-full-access --json < /tmp/.pycastle_prompt"
    )
    assert (provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
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
                error=runtime.UsageLimitError(
                    reset_time=None,
                    service_name="claude",
                    invocation_progress=runtime.InvocationProgress.STARTED,
                    usage=runtime.ProviderUsage(
                        input_tokens=3,
                        output_tokens=1,
                    ),
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
                worktree=tmp_path,
                runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=runtime.ToolAccess.no_tools(),
            ),
            provider_invocation_adapter=adapter,
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=None,
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "observed-session",
                "provider_state_dir_relpath": "implementer/main/claude/",
                "exact_transcript_match": False,
            },
        ),
        usage=runtime.ProviderUsage(
            input_tokens=3,
            output_tokens=1,
        ),
    )


def test_runtime_client_keeps_recoverable_codex_resumed_session_id_when_invocation_seam_has_no_observed_session_id(
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
            error=runtime.UsageLimitError(
                reset_time=None,
                service_name="codex",
                invocation_progress=runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(
                    input_tokens=3,
                    output_tokens=1,
                ),
            ),
            stdout_lines=(),
            provider_session_id=None,
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
        tool_access=runtime.ToolAccess.no_tools(),
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=None,
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "selected-thread",
                "provider_state_dir_relpath": "implementer/main/codex/",
                "exact_transcript_match": False,
            },
        ),
        usage=runtime.ProviderUsage(
            input_tokens=3,
            output_tokens=1,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.path == Path("/tmp/.pycastle_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.provider_session_id == "selected-thread"


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
        tool_access=runtime.ToolAccess.no_tools(),
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="generated output",
        result=prompt_runtime.SessionRunResult(
            output="generated output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="claude",
                provider_session_id="generated-session-id",
                run_kind=RunKind.FRESH,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="sonnet",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "generated-session-id",
                    "provider_state_dir_relpath": provider_state_dir_relpath,
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=None,
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
        tool_access=runtime.ToolAccess.no_tools(),
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="fresh output",
        result=prompt_runtime.SessionRunResult(
            output="fresh output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="claude",
                provider_session_id="claude-session-123",
                run_kind=RunKind.FRESH,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="claude",
                selected_model="sonnet",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "claude-session-123",
                    "provider_state_dir_relpath": provider_state_dir_relpath,
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=None,
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=None,
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "session-uuid",
                "provider_state_dir_relpath": "implementer/main/claude/",
                "exact_transcript_match": False,
            },
        ),
        usage=runtime.ProviderUsage(
            input_tokens=3,
            output_tokens=None,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cost_usd=None,
            duration_seconds=None,
        ),
    )


def test_runtime_client_omits_continuation_for_pre_start_claude_new_session_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ClaudeProcess:
        def __init__(self) -> None:
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
                worktree=tmp_path,
                runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="claude",
        reset_time=None,
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.NOT_STARTED,
    )


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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=runtime.ToolAccess.workspace_backed(
                    tmp_path,
                    tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
                ),
            )
        )
    )

    provider_state_dir_relpath = "implementer/main/codex/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="continued output",
        result=prompt_runtime.SessionRunResult(
            output="continued output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="codex",
                provider_session_id="thread-123",
                run_kind=RunKind.FRESH,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.workspace_backed(
                    tmp_path,
                    tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
                ),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "thread-123",
                    "provider_state_dir_relpath": provider_state_dir_relpath,
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=None,
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == Path("/tmp/.pycastle_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.provider_session_id is None
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert recorded_request.command == (
        "codex exec -m gpt-5.4 -c model_reasoning_effort=medium "
        "-c approval_policy=never --dangerously-bypass-approvals-and-sandbox "
        "--json < /tmp/.pycastle_prompt"
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="continued output",
        result=prompt_runtime.SessionRunResult(
            output="continued output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="codex",
                provider_session_id="thread-123",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "thread-123",
                    "provider_state_dir_relpath": "implementer/main/codex/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=None,
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.path == Path("/tmp/.pycastle_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "thread-123"
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert recorded_request.command == (
        "codex exec resume thread-123 -m gpt-5.4 "
        "-c model_reasoning_effort=medium -c approval_policy=never "
        "--sandbox danger-full-access --json < /tmp/.pycastle_prompt"
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
        tool_access=runtime.ToolAccess.no_tools(),
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="continued output",
        result=prompt_runtime.SessionRunResult(
            output="continued output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="codex",
                provider_session_id="selected-thread",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "selected-thread",
                    "provider_state_dir_relpath": "implementer/main/codex/",
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=3,
            output_tokens=2,
            cache_read_input_tokens=1,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.path == Path("/tmp/.pycastle_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "selected-thread"
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert recorded_request.command == (
        "codex exec resume selected-thread -m gpt-5.4 "
        "-c model_reasoning_effort=medium -c approval_policy=never "
        "--sandbox danger-full-access --json < /tmp/.pycastle_prompt"
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
            error=runtime.UsageLimitError(
                reset_time=None,
                service_name="codex",
                invocation_progress=runtime.InvocationProgress.STARTED,
            ),
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=None,
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "thread-123",
                "provider_state_dir_relpath": "implementer/main/codex/",
                "exact_transcript_match": False,
            },
        ),
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
            error=runtime.UsageLimitError(
                reset_time=None,
                service_name="codex",
                invocation_progress=runtime.InvocationProgress.STARTED,
            ),
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
        tool_access=runtime.ToolAccess.no_tools(),
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=None,
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "thread-456",
                "provider_state_dir_relpath": "implementer/main/codex/",
                "exact_transcript_match": False,
            },
        ),
    )
    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].provider_session_id == "selected-thread"


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
        tool_access=runtime.ToolAccess.no_tools(),
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
                    worktree=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                    role=InvocationRole("implementer"),
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
        tool_access=runtime.ToolAccess.no_tools(),
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
                    worktree=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                    role=InvocationRole("implementer"),
                    session_namespace="main",
                )
            )
        )

    assert str(exc_info.value) == (
        "Codex continuation is not recoverable from provider state."
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
                    worktree=tmp_path,
                    runtime_state_dir=tmp_path / ".agent-runtime" / "state",
                    continuation=prompt_runtime.Continuation(
                        selected_service="codex",
                        selected_model="gpt-5.4",
                        selected_effort="medium",
                        tool_access=runtime.ToolAccess.no_tools(),
                        provider_resume_state=["resume"],
                    ),
                    role=InvocationRole("implementer"),
                )
            )
        )

    assert str(exc_info.value) == (
        "Continuation provider_resume_state must be a JSON object."
    )


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
                        worktree=tmp_path,
                        runtime_state_dir=runtime_state_dir,
                        stage=runtime.StageSelection(
                            service="codex",
                            model="gpt-5.4",
                            effort="medium",
                        ),
                        role=InvocationRole("implementer"),
                        session_namespace="main",
                        tool_access=runtime.ToolAccess.no_tools(),
                    )
                )
            )
        else:
            asyncio.run(
                runtime.RuntimeClient().run_resumed_session(
                    prompt_runtime.ResumedSessionRunRequest(
                        prompt="already rendered prompt",
                        worktree=tmp_path,
                        runtime_state_dir=runtime_state_dir,
                        continuation=prompt_runtime.Continuation(
                            selected_service="codex",
                            selected_model="gpt-5.4",
                            selected_effort="medium",
                            tool_access=runtime.ToolAccess.no_tools(),
                            provider_resume_state={
                                "run_kind": "resume",
                                "provider_session_id": "selected-thread",
                                "provider_state_dir_relpath": "implementer/main/codex/",
                                "exact_transcript_match": False,
                            },
                        ),
                        role=InvocationRole("implementer"),
                        session_namespace="main",
                    )
                )
            )

    assert str(exc_info.value) == (
        "Codex authentication missing: run `codex login` on the host."
    )
    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 401
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="codex",
            raw_provider_text=(
                "Codex authentication missing: run `codex login` on the host."
            ),
            source_stream="pre-dispatch host check",
            status_code=401,
        ),
    )
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome.result is not None
    assert outcome.result.runtime_metadata.run_kind is RunKind.RESUME
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
            runtime.StageSelection(
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
            lambda worktree: worktree / ".pycastle_prompt",
            {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"},
            ("claude", "--output-format stream-json", "--model sonnet"),
            id="claude",
        ),
        pytest.param(
            "codex",
            runtime.StageSelection(
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
            lambda _worktree: Path("/tmp/.pycastle_prompt"),
            {"TZ": "UTC"},
            (
                "codex exec",
                "-m gpt-5.4",
                "-c model_reasoning_effort=medium",
                "< /tmp/.pycastle_prompt",
            ),
            id="codex",
        ),
        pytest.param(
            "opencode",
            runtime.StageSelection(
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
            lambda _worktree: Path("/tmp/.pycastle_prompt"),
            {
                "TZ": "UTC",
                "OPENCODE_GO_API_KEY": "go-key",
            },
            (
                "opencode run",
                "--format json",
                "--model opencode-go/kimi-k2.6",
                '"$(cat /tmp/.pycastle_prompt)"',
            ),
            id="opencode",
        ),
    ],
)
def test_runtime_client_runs_ephemeral_built_in_provider_through_invocation_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service_name: str,
    stage: runtime.StageSelection,
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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=stage,
            tool_access=runtime.ToolAccess.no_tools(),
            auth=auth,
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output=expected_output,
        result=prompt_runtime.EphemeralRunResult(
            output=expected_output,
            selected_service=service_name,
            selected_model=stage.model,
            selected_effort=stage.effort,
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=(service_name,),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
            usage=expected_usage,
        ),
        usage=expected_usage,
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == expected_prompt_path(tmp_path)
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.role == InvocationRole("implementer")
    assert recorded_request.usage_limit_scope is None
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
    for command_part in expected_command_parts:
        assert command_part in recorded_request.command
    assert list((tmp_path / "logs").glob("*.log")) == []


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
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(
            worktree,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
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
                worktree=worktree,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                role=InvocationRole("implementer"),
                session_namespace="main",
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="continued output",
        result=prompt_runtime.SessionRunResult(
            output="continued output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="opencode",
                provider_session_id="persisted-session-2",
                run_kind=RunKind.RESUME,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="opencode",
                selected_model="glm-5",
                selected_effort="medium",
                tool_access=continuation.tool_access,
                provider_resume_state={
                    "provider_session_id": "persisted-session-2",
                    "provider_state_dir_relpath": provider_state_dir_relpath,
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=None,
    )
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "persisted-session-2"
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == worktree / ".pycastle_prompt"
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == worktree
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.role == InvocationRole("implementer")
    assert recorded_request.usage_limit_scope is None
    assert recorded_request.provider_session_id == "persisted-session-1"
    assert recorded_request.environment["OPENCODE_HOME"] == str(provider_state_dir)
    assert recorded_request.environment["OPENCODE_GO_API_KEY"] == "go-key"
    assert "--session persisted-session-1" in recorded_request.command
    assert "--model opencode-go/glm-5" in recorded_request.command


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

    runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert captured["env"] == {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}


def test_runtime_client_reachable_opencode_stage_requires_api_key_without_falling_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="missing",
                    model="ignored",
                    effort="low",
                    fallback=runtime.StageSelection(
                        service="opencode",
                        model="kimi-k2.6",
                        effort="medium",
                        fallback=runtime.StageSelection(
                            service="codex",
                            model="gpt-5.4",
                            effort="medium",
                        ),
                    ),
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(),
            )
        )

    assert str(exc_info.value) == "Missing OpenCode API key."
    assert exc_info.value.service_name == "opencode"
    assert exc_info.value.classification == (
        "operator_actionable_agent_credential_failure"
    )
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="opencode",
            raw_provider_text="Missing OpenCode API key.",
            source_stream="pre-dispatch auth check",
            status_code=401,
        ),
    )
    assert adapter.recorded_requests == []


def test_runtime_client_reachable_codex_stage_requires_host_auth_without_falling_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module.Path,
        "home",
        lambda: tmp_path / "missing-home",
    )

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="missing",
                    model="ignored",
                    effort="low",
                    fallback=runtime.StageSelection(
                        service="codex",
                        model="gpt-5.4",
                        effort="medium",
                        fallback=runtime.StageSelection(
                            service="claude",
                            model="sonnet",
                            effort="medium",
                        ),
                    ),
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(),
            )
        )

    assert str(exc_info.value) == (
        "Codex authentication missing: run `codex login` on the host."
    )
    assert exc_info.value.service_name == "codex"
    assert exc_info.value.status_code == 401
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="codex",
            raw_provider_text=(
                "Codex authentication missing: run `codex login` on the host."
            ),
            source_stream="pre-dispatch host check",
            status_code=401,
        ),
    )
    assert adapter.recorded_requests == []


def test_runtime_client_runs_codex_new_session_through_built_in_provider_invocation_seam(
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                logs_dir=tmp_path / "logs",
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    provider_state_dir_relpath = "implementer/main/codex/"
    provider_state_dir = runtime_state_dir / provider_state_dir_relpath

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="continued output",
        result=prompt_runtime.SessionRunResult(
            output="continued output",
            runtime_metadata=prompt_runtime.SessionRuntimeMetadata(
                service_name="codex",
                provider_session_id="thread-123",
                run_kind=RunKind.FRESH,
                session_namespace="main",
                exact_transcript_match=False,
            ),
            continuation=prompt_runtime.Continuation(
                selected_service="codex",
                selected_model="gpt-5.4",
                selected_effort="medium",
                tool_access=runtime.ToolAccess.no_tools(),
                provider_resume_state={
                    "run_kind": "resume",
                    "provider_session_id": "thread-123",
                    "provider_state_dir_relpath": provider_state_dir_relpath,
                    "exact_transcript_match": False,
                },
            ),
        ),
        usage=runtime.ProviderUsage(
            input_tokens=3,
            output_tokens=2,
            cache_read_input_tokens=1,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == Path("/tmp/.pycastle_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.worktree == tmp_path
    assert recorded_request.run_kind is RunKind.FRESH
    assert recorded_request.role == InvocationRole("implementer")
    assert recorded_request.provider_session_id is None
    assert recorded_request.log_context is not None
    assert recorded_request.log_context.role == InvocationRole("implementer")
    assert recorded_request.environment == {
        "TZ": "UTC",
        "CODEX_HOME": str(provider_state_dir),
    }
    assert (provider_state_dir / "auth.json").read_text(encoding="utf-8") == (
        '{"token":"host-auth"}\n'
    )
    log_path = next((tmp_path / "logs").glob("*.log"))
    log_text = log_path.read_text(encoding="utf-8")
    assert '"prompt": "already rendered prompt"' in log_text
    assert '"thread_id":"thread-123"' in log_text
    assert '"text": "continued output"' in log_text


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
            error=runtime.UsageLimitError(
                reset_time=None,
                service_name="codex",
                invocation_progress=runtime.InvocationProgress.STARTED,
                usage=runtime.ProviderUsage(
                    input_tokens=3,
                    output_tokens=1,
                    cache_read_input_tokens=1,
                ),
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
                worktree=tmp_path,
                runtime_state_dir=runtime_state_dir,
                logs_dir=tmp_path / "logs",
                stage=runtime.StageSelection(
                    service="codex",
                    model="gpt-5.4",
                    effort="medium",
                ),
                role=InvocationRole("implementer"),
                session_namespace="main",
                tool_access=runtime.ToolAccess.no_tools(),
            )
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.usage_limited(
        output="",
        service_name="codex",
        reset_time=None,
        usage_limit_scope=None,
        invocation_progress=runtime.InvocationProgress.STARTED,
        continuation=prompt_runtime.Continuation(
            selected_service="codex",
            selected_model="gpt-5.4",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "thread-123",
                "provider_state_dir_relpath": "implementer/main/codex/",
                "exact_transcript_match": False,
            },
        ),
        usage=runtime.ProviderUsage(
            input_tokens=3,
            output_tokens=1,
            cache_read_input_tokens=1,
        ),
    )
    assert len(adapter.recorded_requests) == 1
    assert adapter.recorded_requests[0].log_context is not None


def test_runtime_client_preserves_opencode_invalid_api_key_observations(
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
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )

    assert exc_info.value.service_name == "opencode"
    assert exc_info.value.status_code == 401
    assert exc_info.value.classification == (
        "operator_actionable_agent_credential_failure"
    )
    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="opencode",
            raw_provider_text="invalid api key",
            source_stream="json_event.error",
            status_code=401,
            error_name="AuthenticationError",
        ),
    )


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
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="opencode",
                    model=model,
                    effort=effort,
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(opencode_api_key="go-key"),
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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="opencode",
                model="kimi-k2.6",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 4, 28, 21, 4, tzinfo=timezone.utc),
        usage_limit_scope=None,
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_maps_codex_usage_limit_stream_to_no_service_available_and_logs_provider_output(
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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 2, 17, 2, tzinfo=timezone.utc),
        usage_limit_scope=None,
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )
    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
    assert recorded_request.prompt.content == "already rendered prompt"
    assert recorded_request.prompt.path == Path("/tmp/.pycastle_prompt")
    assert recorded_request.prompt.cleanup_path is True
    assert recorded_request.environment["TZ"] == "UTC"
    assert "codex exec" in recorded_request.command
    assert list((tmp_path / "logs").glob("*.log")) == []


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
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )

    assert str(exc_info.value) == (
        "Model not found: opencode-go/deepseek-v4-flash. "
        "Did you mean: deepseek-v4-flash?"
    )
    assert exc_info.value.service_name == "opencode"
    assert exc_info.value.status_code == 400


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
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="opencode",
                    model="kimi-k2.6",
                    effort="medium",
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(opencode_api_key="go-key"),
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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="opencode",
                model="kimi-k2.6",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="completed answer",
        result=prompt_runtime.EphemeralRunResult(
            output="completed answer",
            selected_service="opencode",
            selected_model="kimi-k2.6",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=False,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("opencode",),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
        ),
    )


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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 1, 13, 2, tzinfo=timezone.utc),
        usage_limit_scope=None,
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_reachable_claude_stage_requires_token_without_falling_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _install_in_memory_provider_invocation_adapter(monkeypatch)

    with pytest.raises(AgentCredentialFailureError) as exc_info:
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="missing",
                    model="ignored",
                    effort="low",
                    fallback=runtime.StageSelection(
                        service="claude",
                        model="sonnet",
                        effort="medium",
                        fallback=runtime.StageSelection(
                            service="codex",
                            model="gpt-5",
                            effort="medium",
                        ),
                    ),
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(),
            )
        )

    assert exc_info.value.service_name == "claude"
    assert adapter.recorded_requests == []


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
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
            )
        )

    assert exc_info.value.status_code == 500


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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=datetime(2026, 1, 2, 16, 2, tzinfo=timezone.utc),
        usage_limit_scope=None,
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.no_service_available(
        output="",
        reset_time=reset_time + timedelta(minutes=2),
        usage_limit_scope=None,
        invocation_progress=prompt_runtime.InvocationProgress.NOT_STARTED,
    )


def test_runtime_client_reports_fallback_metadata_for_ephemeral_result(
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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="missing",
                model="ignored",
                effort="low",
                fallback=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome == prompt_runtime.RuntimeOutcome.completed(
        output="final output",
        result=prompt_runtime.EphemeralRunResult(
            output="final output",
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=runtime.ToolAccess.no_tools(),
            used_fallback=True,
            metadata=prompt_runtime.EphemeralResultMetadata(
                selected_service_path=("missing", "claude"),
                runtime=prompt_runtime.EphemeralRuntimeMetadata(
                    run_kind=RunKind.FRESH,
                ),
            ),
        ),
    )


def test_runtime_client_completed_ephemeral_result_hides_session_namespace_metadata(
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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome.runtime_metadata == prompt_runtime.EphemeralRuntimeMetadata(
        run_kind=RunKind.FRESH,
    )
    assert not hasattr(outcome.runtime_metadata, "session_namespace")


def test_runtime_client_ephemeral_usage_limit_outcome_hides_usage_limit_scope(
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

    outcome = runtime.RuntimeClient().run_ephemeral(
        prompt_runtime.EphemeralRunRequest(
            prompt="already rendered prompt",
            worktree=tmp_path,
            stage=runtime.StageSelection(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
            tool_access=runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert outcome.kind == "no_service_available"
    assert outcome.usage_limit_scope is None


def test_runtime_client_preserves_claude_credential_failure_observations(
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
        runtime.RuntimeClient().run_ephemeral(
            prompt_runtime.EphemeralRunRequest(
                prompt="already rendered prompt",
                worktree=tmp_path,
                stage=runtime.StageSelection(
                    service="claude",
                    model="sonnet",
                    effort="medium",
                ),
                tool_access=runtime.ToolAccess.no_tools(),
                auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
            )
        )

    assert exc_info.value.observations == (
        ProviderErrorObservation(
            service_name="claude",
            raw_provider_text=denial_message,
            source_stream="json_event.result",
            status_code=403,
        ),
    )
