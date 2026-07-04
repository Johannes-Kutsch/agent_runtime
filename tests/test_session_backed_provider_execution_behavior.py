from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

import agent_runtime as runtime
import agent_runtime._builtin_provider_rendering as built_in_provider_rendering
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime._session_backed_provider_execution as session_backed_execution
import agent_runtime._session_backed_provider_state_resolution as provider_state_resolution
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
from tests.runtime_client_execution_harness import RuntimeClientExecutionHarness
from agent_runtime._runtime_lifecycle import CancellationToken
from agent_runtime.errors import (
    AgentCancelledError,
    ContinuationUnrecoverableError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from agent_runtime.session import RunKind
from agent_runtime.types import ProviderSelection as InternalStageSelection


@pytest.mark.parametrize("entrypoint", ["new", "resumed"])
def test_session_backed_lifecycle_requires_session_store_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=('{"type":"result","output":"should not run"}\n',)
        )
    )

    with pytest.raises(RuntimeConfigurationError, match="session_store"):
        if entrypoint == "new":
            session_backed_execution._run_builtin_new_session(
                RuntimeClientExecutionHarness.start_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=None,
                    provider_selection=InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        else:
            session_backed_execution._run_builtin_resumed_session(
                RuntimeClientExecutionHarness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=None,
                    continuation=RuntimeClientExecutionHarness.opencode_continuation(),
                    provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                )
            )

    assert harness.recorded_request_count == 0


@pytest.mark.parametrize("entrypoint", ["new", "resumed"])
def test_session_backed_codex_completion_resolves_provider_session_id_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
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

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    continuation = RuntimeClientExecutionHarness.codex_continuation()

    if entrypoint == "new":
        result = session_backed_execution._run_builtin_new_session(
            RuntimeClientExecutionHarness.start_session_run_request(
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
    else:
        provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
            runtime_state_dir,
            service="codex",
        )
        RuntimeClientExecutionHarness.prepare_codex_rollout_state(
            provider_state_dir, "selected-thread"
        )
        result = session_backed_execution._run_builtin_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
            )
        )

    assert result.output == "final output"
    assert result.usage == runtime.ProviderUsage(
        input_tokens=5,
        output_tokens=2,
    )
    assert result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert result.continuation == RuntimeClientExecutionHarness.codex_continuation(
        provider_session_id="thread-obs",
    )
    assert harness.recorded_request_count == 1


def test_session_backed_codex_new_session_recovers_provider_state_through_module_interface(
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
                '{"type":"item.completed","item":{"type":"agent_message","text":"continued output"}}\n',
                '{"type":"turn.completed"}\n',
            ),
        ),
    )

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    RuntimeClientExecutionHarness.prepare_codex_rollout_state(
        provider_state_dir, "thread-123", "thread-123"
    )

    result = session_backed_execution._run_builtin_new_session(
        RuntimeClientExecutionHarness.start_session_run_request(
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

    assert result.output == "continued output"
    assert result.selected == runtime.ResolvedProvider(
        service="codex", model="gpt-5.4", effort="medium"
    )
    assert result.continuation == RuntimeClientExecutionHarness.codex_continuation(
        provider_session_id="thread-123",
    )
    recorded_request = harness.recorded_request()
    assert recorded_request.run_kind is RunKind.RESUME
    assert recorded_request.provider_session_id == "thread-123"
    assert recorded_request.command == (
        f"{'codex.cmd' if os.name == 'nt' else 'codex'} exec --sandbox read-only resume thread-123 -m gpt-5.4 "
        "-c model_reasoning_effort=medium -c approval_policy=never --json"
    )


@pytest.mark.parametrize(
    ("entrypoint", "run_kind", "provider_session_id"),
    [
        ("new", RunKind.FRESH, None),
        ("resumed", RunKind.RESUME, "selected-thread"),
    ],
)
def test_session_backed_codex_invocation_uses_built_in_provider_rendering_facts_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    run_kind: RunKind,
    provider_session_id: str | None,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
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

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    if entrypoint == "new":
        session_backed_execution._run_builtin_new_session(
            RuntimeClientExecutionHarness.start_session_run_request(
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
    else:
        RuntimeClientExecutionHarness.prepare_codex_rollout_state(
            provider_state_dir, "selected-thread"
        )
        continuation = RuntimeClientExecutionHarness.codex_continuation()
        session_backed_execution._run_builtin_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
            )
        )

    assert harness.recorded_request_count == 1
    recorded_request = harness.recorded_request()
    rendered = built_in_provider_rendering.render_built_in_provider_invocation(
        built_in_provider_rendering.BuiltInProviderRenderRequest(
            provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
                service="codex",
                model="gpt-5.4",
                effort="medium",
            ),
            run_kind=run_kind,
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            auth=None,
            invocation_dir=tmp_path,
            provider_state_dir=provider_state_dir,
            provider_session_id=provider_session_id,
        )
    )

    assert recorded_request.command == rendered.legacy_command_text
    assert recorded_request.argv == rendered.canonical_argv
    assert recorded_request.prefer_argv is rendered.prefer_argv
    assert recorded_request.environment == dict(rendered.environment)
    assert recorded_request.prompt.path == rendered.prompt_path
    assert recorded_request.prompt.cleanup_path is (
        rendered.prompt_cleanup_choice
        is built_in_provider_rendering.PromptCleanupChoice.DELETE_AFTER_INVOCATION
    )
    assert recorded_request.provider_session_id == provider_session_id
    assert recorded_request.run_kind is run_kind


@pytest.mark.parametrize(
    ("entrypoint", "expected_provider_session_id", "recorded_provider_session_id"),
    [
        ("new", "thread-123", None),
        ("resumed", "thread-456", "selected-thread"),
    ],
)
def test_session_backed_codex_expected_interruptions_keep_started_continuations_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    expected_provider_session_id: str,
    recorded_provider_session_id: str | None,
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
            stdout_lines=(
                json.dumps(
                    {
                        "type": "thread.started",
                        "thread_id": expected_provider_session_id,
                    }
                )
                + "\n",
            ),
            provider_session_id=None,
        ),
    )

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    continuation = RuntimeClientExecutionHarness.codex_continuation()

    with pytest.raises(UsageLimitError) as exc_info:
        if entrypoint == "new":
            session_backed_execution._run_builtin_new_session(
                RuntimeClientExecutionHarness.start_session_run_request(
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
        else:
            provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
                runtime_state_dir,
                service="codex",
            )
            RuntimeClientExecutionHarness.prepare_codex_rollout_state(
                provider_state_dir, "recovered-thread"
            )
            session_backed_execution._run_builtin_resumed_session(
                RuntimeClientExecutionHarness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                )
            )

    assert (
        exc_info.value.continuation
        == RuntimeClientExecutionHarness.codex_continuation(
            provider_session_id=expected_provider_session_id,
        )
    )
    assert (
        harness.recorded_request().provider_session_id == recorded_provider_session_id
    )


def test_session_backed_codex_resumed_session_surfaces_provider_state_resolution_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_codex_resumed_session_facts",
        lambda **_kwargs: (_ for _ in ()).throw(
            ContinuationUnrecoverableError(
                "Codex continuation is not recoverable from provider state.",
                service_name="codex",
            )
        ),
    )

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )

    with pytest.raises(ContinuationUnrecoverableError) as exc_info:
        session_backed_execution._run_builtin_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=RuntimeClientExecutionHarness.codex_continuation(),
            )
        )

    assert (
        str(exc_info.value)
        == "Codex continuation is not recoverable from provider state."
    )
    assert harness.recorded_request_count == 0


def test_session_backed_codex_resumed_session_uses_resolved_provider_session_id_for_invocation_through_module_interface(
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
            usage=runtime.ProviderUsage(input_tokens=7, output_tokens=2),
            stdout_lines=(
                json.dumps(
                    {
                        "type": "result",
                        "result": "continued output",
                    }
                )
                + "\n",
            ),
            provider_session_id=None,
        ),
    )
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_codex_resumed_session_facts",
        lambda **_kwargs: provider_state_resolution.CodexResumedSessionResolution(
            provider_state_dir=provider_state_dir,
            continuation_input_facts=provider_state_resolution.codex_continuation_input_facts(
                model="gpt-5.4",
                effort="medium",
                provider_state_dir=provider_state_dir,
                provider_state_dir_relpath="",
                provider_session_id="recovered-thread",
                recovered_provider_session_id=True,
                run_kind=RunKind.RESUME,
            ),
        ),
    )

    result = session_backed_execution._run_builtin_resumed_session(
        RuntimeClientExecutionHarness.resume_session_run_request(
            invocation_dir=tmp_path,
            runtime_state_dir=runtime_state_dir,
            continuation=RuntimeClientExecutionHarness.codex_continuation(
                provider_session_id="   "
            ),
        )
    )

    assert result.output == "continued output"
    assert result.usage == runtime.ProviderUsage(input_tokens=7, output_tokens=2)
    assert harness.recorded_request_count == 1
    assert harness.recorded_request().provider_session_id == "recovered-thread"


def test_session_backed_claude_completion_uses_observed_provider_session_id_in_completed_outcome_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    expected_provider_session_id = "observed-session-id"
    RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="final output",
            usage=runtime.ProviderUsage(
                input_tokens=5,
                output_tokens=2,
            ),
            provider_session_id=expected_provider_session_id,
            stdout_lines=(),
        ),
    )

    result = session_backed_execution._run_builtin_new_session(
        RuntimeClientExecutionHarness.start_session_run_request(
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
    )

    assert result.output == "final output"
    assert result.selected == runtime.ResolvedProvider(
        service="claude", model="sonnet", effort="medium"
    )
    assert result.continuation is not None
    assert (
        result.continuation.provider_resume_state["provider_session_id"]
        == expected_provider_session_id
    )


def test_session_backed_claude_resumed_session_uses_resolved_provider_state_dir_for_invocation_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _SnapshottingClaudeAdapter:
        def __init__(self) -> None:
            self.recorded_request_count = 0
            self.provider_state_dir: Path | None = None

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            self.recorded_request_count += 1
            self.provider_state_dir = Path(request.environment["CLAUDE_CONFIG_DIR"])
            assert request.run_kind is RunKind.RESUME
            return provider_invocation_runtime.ProviderInvocationResult(
                output="continued output",
                provider_session_id="resumed-session-1",
                usage=runtime.ProviderUsage(input_tokens=1, output_tokens=1),
                stdout_lines=(),
            )

    adapter = _SnapshottingClaudeAdapter()
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="claude",
    )
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    (provider_state_dir / "session_file").write_text(
        "continuation state", encoding="utf-8"
    )
    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_claude_resumed_session_facts",
        lambda **_kwargs: provider_state_resolution.ClaudeResumedSessionResolution(
            provider_state_dir=provider_state_dir,
            continuation_input_facts=provider_state_resolution.claude_continuation_input_facts(
                model="sonnet",
                effort="medium",
                provider_state_dir=provider_state_dir,
                provider_state_dir_relpath="",
                provider_session_id="resumed-session-1",
                run_kind=RunKind.RESUME,
            ),
        ),
    )
    continuation = RuntimeClientExecutionHarness.claude_continuation(
        provider_session_id="resumed-session-1",
        provider_state_dir_relpath="",
    )

    result = session_backed_execution._run_builtin_resumed_session(
        RuntimeClientExecutionHarness.resume_session_run_request(
            invocation_dir=tmp_path,
            runtime_state_dir=runtime_state_dir,
            continuation=continuation,
            provider_auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
        )
    )

    assert result.continuation == RuntimeClientExecutionHarness.claude_continuation(
        provider_session_id="resumed-session-1",
        provider_state_dir_relpath="",
    )
    assert adapter.recorded_request_count == 1
    assert adapter.provider_state_dir == provider_state_dir


def test_session_backed_opencode_completed_outcome_keeps_resolved_session_details_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _SnapshottingOpencodeAdapter:
        def __init__(self) -> None:
            self.recorded_requests: list[
                provider_invocation_runtime.ProviderInvocationRequest
            ] = []

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            self.recorded_requests.append(request)
            return provider_invocation_runtime.ProviderInvocationResult(
                output="continued output",
                stdout_lines=(
                    json.dumps({"type": "session.status", "status": {"type": "idle"}})
                    + "\n",
                ),
            )

    adapter = _SnapshottingOpencodeAdapter()
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )

    continuation = RuntimeClientExecutionHarness.opencode_continuation()
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    (runtime_state_dir / "resume.jsonl").write_text("", encoding="utf-8")

    result = session_backed_execution._run_builtin_resumed_session(
        RuntimeClientExecutionHarness.resume_session_run_request(
            invocation_dir=tmp_path,
            continuation=continuation,
            runtime_state_dir=runtime_state_dir,
            provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        )
    )

    assert result.output == "continued output"
    assert result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert result.continuation.provider_resume_state == {
        "provider_session_id": "persisted-session-1",
        "exact_transcript_match": False,
        "provider_state_dir_relpath": "",
    }
    assert adapter.recorded_requests[0].run_kind is RunKind.RESUME
    assert adapter.recorded_requests[0].provider_session_id == "persisted-session-1"


def test_session_backed_opencode_start_session_run_then_resume_session_run_reuses_saved_session_id_by_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="first output",
            provider_session_id="persisted-session-1",
            usage=runtime.ProviderUsage(input_tokens=1, output_tokens=1),
            stdout_lines=(),
        ),
        provider_invocation_runtime.ProviderInvocationResult(
            output="second output",
            provider_session_id="persisted-session-1",
            usage=runtime.ProviderUsage(input_tokens=1, output_tokens=1),
            stdout_lines=(),
        ),
    )

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )

    start_result = session_backed_execution._run_builtin_new_session(
        RuntimeClientExecutionHarness.start_session_run_request(
            invocation_dir=tmp_path,
            runtime_state_dir=runtime_state_dir,
            provider_selection=InternalStageSelection(
                service="opencode",
                model="glm-5.2",
                effort="medium",
            ),
            provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            session_namespace="main",
        )
    )
    continuation = start_result.continuation
    assert continuation is not None

    resume_request = RuntimeClientExecutionHarness.resume_session_run_request(
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        continuation=continuation,
        session_namespace="alt",
        provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
    )
    resume_result = session_backed_execution._run_builtin_resumed_session(
        resume_request
    )

    assert resume_result.output == "second output"
    assert resume_result.continuation is not None
    assert resume_result.continuation.provider_resume_state["provider_session_id"] == (
        "persisted-session-1"
    )
    assert harness.recorded_request(1).run_kind is RunKind.RESUME
    assert harness.recorded_request(1).provider_session_id == "persisted-session-1"


def test_session_backed_opencode_resumed_session_uses_resolved_provider_session_id_for_invocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="opencode",
    )
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    prepared_facts = provider_state_resolution.opencode_continuation_input_facts(
        model="glm-5.2",
        effort="medium",
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath="",
        provider_session_id="persisted-session-1",
        run_kind=RunKind.RESUME,
        exact_transcript_match=True,
    )
    active_session_ids: list[str | None] = []

    class _DetectingOpencodeAdapter:
        def __init__(self) -> None:
            self.recorded_requests: list[
                provider_invocation_runtime.ProviderInvocationRequest
            ] = []

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            self.recorded_requests.append(request)
            return provider_invocation_runtime.ProviderInvocationResult(
                output="continued output",
                provider_session_id="persisted-session-1",
                stdout_lines=(
                    json.dumps({"type": "session.status", "status": {"type": "idle"}})
                    + "\n",
                ),
            )

    adapter = _DetectingOpencodeAdapter()
    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: adapter,
    )

    def _resolve_opencode_resumed_session_facts(
        **_kwargs: object,
    ) -> provider_state_resolution.OpenCodeResumedSessionResolution:
        return provider_state_resolution.OpenCodeResumedSessionResolution(
            provider_state_dir=provider_state_dir,
            continuation_input_facts=prepared_facts,
        )

    def _resolve_opencode_active_session_facts(
        continuation_input_facts: provider_state_resolution.ContinuationInputFacts,
        *,
        provider_session_id: str | None,
    ) -> provider_state_resolution.ContinuationInputFacts:
        active_session_ids.append(provider_session_id)
        return continuation_input_facts

    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_opencode_resumed_session_facts",
        _resolve_opencode_resumed_session_facts,
    )
    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_opencode_active_session_facts",
        _resolve_opencode_active_session_facts,
    )

    forged_continuation = runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "persisted-session-1",
            "provider_state": {
                "session_id": "forged-session",
                "resume_jsonl": "forged",
            },
            "exact_transcript_match": True,
        },
    )
    result = session_backed_execution._run_builtin_resumed_session(
        RuntimeClientExecutionHarness.resume_session_run_request(
            invocation_dir=tmp_path,
            runtime_state_dir=runtime_state_dir,
            continuation=forged_continuation,
            provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        )
    )

    assert adapter.recorded_requests[0].run_kind is RunKind.RESUME
    assert adapter.recorded_requests[0].provider_session_id == "persisted-session-1"
    assert active_session_ids == ["persisted-session-1"]
    assert result.continuation.provider_resume_state == {
        "provider_session_id": "persisted-session-1",
        "exact_transcript_match": True,
        "provider_state_dir_relpath": "",
    }


@pytest.mark.parametrize(
    ("entrypoint", "run_kind", "provider_session_id"),
    [
        ("new", RunKind.FRESH, "prepared-session-id"),
        ("resumed", RunKind.RESUME, "persisted-session-1"),
    ],
)
def test_session_backed_opencode_invocation_uses_built_in_provider_rendering_facts_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    run_kind: RunKind,
    provider_session_id: str,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    if entrypoint == "new":
        RuntimeClientExecutionHarness.install_generated_provider_session_id(
            monkeypatch,
            "prepared-session-id",
        )
    harness.prepare(
        provider_invocation_runtime.ProviderInvocationResult(
            output="final output",
            usage=runtime.ProviderUsage(
                input_tokens=5,
                output_tokens=2,
            ),
            provider_session_id="provider-session-obs",
            stdout_lines=(),
        ),
    )

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="opencode",
    )
    if entrypoint == "new":
        session_backed_execution._run_builtin_new_session(
            RuntimeClientExecutionHarness.start_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                provider_selection=InternalStageSelection(
                    service="opencode",
                    model="glm-5.2",
                    effort="medium",
                ),
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            )
        )
    else:
        continuation = RuntimeClientExecutionHarness.opencode_continuation()
        (runtime_state_dir / "resume.jsonl").write_text("", encoding="utf-8")
        session_backed_execution._run_builtin_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )

    assert harness.recorded_request_count == 1
    recorded_request = harness.recorded_request()
    rendered = built_in_provider_rendering.render_built_in_provider_invocation(
        built_in_provider_rendering.BuiltInProviderRenderRequest(
            provider_selection=built_in_provider_rendering.BuiltInProviderSelectionFacts(
                service="opencode",
                model="glm-5.2",
                effort="medium",
            ),
            run_kind=run_kind,
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            invocation_dir=tmp_path,
            provider_state_dir=provider_state_dir,
            provider_session_id=provider_session_id,
        )
    )

    assert recorded_request.command == rendered.legacy_command_text
    assert recorded_request.argv == rendered.canonical_argv
    assert recorded_request.prefer_argv is rendered.prefer_argv
    assert recorded_request.environment == dict(rendered.environment)
    assert recorded_request.prompt.path == rendered.prompt_path
    assert recorded_request.prompt.cleanup_path is (
        rendered.prompt_cleanup_choice
        is built_in_provider_rendering.PromptCleanupChoice.DELETE_AFTER_INVOCATION
    )
    assert recorded_request.provider_session_id == rendered.provider_session_id
    assert recorded_request.run_kind is run_kind


@pytest.mark.parametrize(
    ("entrypoint", "expected_provider_session_id", "expected_exact_transcript_match"),
    [
        ("new", "provider-session-777", False),
        ("resumed", "persisted-session-1", False),
    ],
)
def test_session_backed_opencode_expected_interruptions_keep_started_continuations_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    expected_provider_session_id: str,
    expected_exact_transcript_match: bool,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    if entrypoint == "new":
        RuntimeClientExecutionHarness.install_generated_provider_session_id(
            monkeypatch,
            "prepared-session-id",
        )
    harness.prepare(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                json.dumps(
                    {
                        "type": "error",
                        "timestamp": 1,
                        **(
                            {"sessionID": "provider-session-777"}
                            if entrypoint == "new"
                            else {}
                        ),
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
            provider_session_id=(
                "adapter-session-2" if entrypoint == "resumed" else None
            ),
        ),
    )

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    continuation = RuntimeClientExecutionHarness.opencode_continuation()
    if entrypoint == "resumed":
        (runtime_state_dir / "resume.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(UsageLimitError) as exc_info:
        if entrypoint == "new":
            session_backed_execution._run_builtin_new_session(
                RuntimeClientExecutionHarness.start_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    provider_selection=InternalStageSelection(
                        service="opencode",
                        model="glm-5.2",
                        effort="medium",
                    ),
                    provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                    tool_access=contracts_runtime.ToolAccess.no_tools(),
                )
            )
        else:
            session_backed_execution._run_builtin_resumed_session(
                RuntimeClientExecutionHarness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=continuation,
                    provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
                )
            )

    assert exc_info.value.reset_time == datetime(
        2026, 4, 28, 21, 2, tzinfo=timezone.utc
    )
    assert exc_info.value.continuation is not None
    assert exc_info.value.continuation.provider_resume_state == {
        "provider_session_id": expected_provider_session_id,
        "exact_transcript_match": expected_exact_transcript_match,
        "provider_state_dir_relpath": "",
    }
    assert harness.recorded_request().provider_session_id == (
        "prepared-session-id" if entrypoint == "new" else "persisted-session-1"
    )


def test_session_backed_opencode_resumed_session_uses_observed_session_id_for_started_interruption_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
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

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="opencode",
    )
    provider_state_dir.mkdir(parents=True, exist_ok=True)
    prepared_facts = provider_state_resolution.opencode_continuation_input_facts(
        model="glm-5.2",
        effort="medium",
        provider_state_dir=provider_state_dir,
        provider_state_dir_relpath="",
        provider_session_id="persisted-session-1",
        run_kind=RunKind.RESUME,
        exact_transcript_match=True,
    )
    active_session_ids: list[str | None] = []

    def _resolve_opencode_resumed_session_facts(
        **_kwargs: object,
    ) -> provider_state_resolution.OpenCodeResumedSessionResolution:
        return provider_state_resolution.OpenCodeResumedSessionResolution(
            provider_state_dir=provider_state_dir,
            continuation_input_facts=prepared_facts,
        )

    def _resolve_opencode_active_session_facts(
        continuation_input_facts: provider_state_resolution.ContinuationInputFacts,
        *,
        provider_session_id: str | None,
    ) -> provider_state_resolution.ContinuationInputFacts:
        active_session_ids.append(provider_session_id)
        return provider_state_resolution.opencode_continuation_input_facts(
            model=continuation_input_facts.provider_identity.model,
            effort=continuation_input_facts.provider_identity.effort,
            provider_state_dir=continuation_input_facts.provider_state_directory.path,
            provider_state_dir_relpath=(
                continuation_input_facts.provider_state_relpath.value
                if continuation_input_facts.provider_state_relpath is not None
                else None
            ),
            provider_session_id=cast(str, provider_session_id),
            run_kind=continuation_input_facts.run_kind,
            exact_transcript_match=False,
        )

    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_opencode_resumed_session_facts",
        _resolve_opencode_resumed_session_facts,
    )
    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_opencode_active_session_facts",
        _resolve_opencode_active_session_facts,
    )
    continuation = RuntimeClientExecutionHarness.opencode_continuation()

    with pytest.raises(UsageLimitError) as exc_info:
        session_backed_execution._run_builtin_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )

    assert exc_info.value.continuation is not None
    assert exc_info.value.continuation.provider_resume_state == {
        "provider_session_id": "observed-session-2",
        "exact_transcript_match": False,
        "provider_state_dir_relpath": "",
    }
    assert active_session_ids == ["observed-session-2"]
    assert harness.recorded_request().provider_session_id == "persisted-session-1"


@pytest.mark.parametrize("entrypoint", ["new", "resumed"])
def test_session_backed_pre_cancelled_token_returns_cancelled_outcome_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    token = CancellationToken()
    token.cancel()

    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )

    if entrypoint == "new":
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
                    token=token,
                )
            )
        )
    else:
        outcome = asyncio.run(
            runtime.RuntimeClient().run_resumed_session(
                RuntimeClientExecutionHarness.resume_session_run_request(
                    invocation_dir=tmp_path,
                    runtime_state_dir=runtime_state_dir,
                    continuation=RuntimeClientExecutionHarness.claude_continuation(
                        provider_session_id="session-1",
                        provider_state_dir_relpath="",
                    ),
                    provider_auth=runtime.ProviderAuth(
                        claude_code_oauth_token="oauth-token"
                    ),
                    token=token,
                )
            )
        )

    assert isinstance(outcome.kind, runtime.Cancelled)
    assert outcome.result.continuation is None
    assert harness.recorded_request_count == 0


def test_session_backed_claude_new_session_cancellation_after_provider_started_returns_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_session_id = "prepared-session-1"

    class _CancelAfterStartedAdapter:
        recorded_request_count = 0

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
            argv_transform=None,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            _CancelAfterStartedAdapter.recorded_request_count += 1
            consume = getattr(
                request.output_hooks.reduce_output, "consume_stdout_lines", None
            )
            if callable(consume):
                consume(["some provider output"])
            raise AgentCancelledError()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: _CancelAfterStartedAdapter(),
    )
    RuntimeClientExecutionHarness.install_generated_provider_session_id(
        monkeypatch,
        expected_session_id,
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
                token=CancellationToken(),
            )
        )
    )

    assert isinstance(outcome.kind, runtime.Cancelled)
    assert outcome.result.continuation is not None
    assert (
        outcome.result.continuation.provider_resume_state["provider_session_id"]
        == expected_session_id
    )
    assert _CancelAfterStartedAdapter.recorded_request_count == 1


def test_session_backed_claude_new_session_cancellation_before_provider_output_returns_no_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _CancelBeforeStartedAdapter:
        recorded_request_count = 0

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
            argv_transform=None,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            _CancelBeforeStartedAdapter.recorded_request_count += 1
            raise AgentCancelledError()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: _CancelBeforeStartedAdapter(),
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
                token=CancellationToken(),
            )
        )
    )

    assert isinstance(outcome.kind, runtime.Cancelled)
    assert outcome.result.continuation is None
    assert _CancelBeforeStartedAdapter.recorded_request_count == 1


def test_session_backed_claude_resumed_session_cancellation_after_provider_started_returns_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_session_id = "resumed-session-1"

    class _CancelAfterStartedAdapter:
        recorded_request_count = 0

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
            argv_transform=None,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            _CancelAfterStartedAdapter.recorded_request_count += 1
            consume = getattr(
                request.output_hooks.reduce_output, "consume_stdout_lines", None
            )
            if callable(consume):
                consume(["some provider output"])
            raise AgentCancelledError()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: _CancelAfterStartedAdapter(),
    )
    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_claude_resumed_session_facts",
        lambda **_kwargs: provider_state_resolution.ClaudeResumedSessionResolution(
            provider_state_dir=tmp_path,
            continuation_input_facts=provider_state_resolution.claude_continuation_input_facts(
                model="sonnet",
                effort="medium",
                provider_state_dir=tmp_path,
                provider_state_dir_relpath="",
                provider_session_id=expected_session_id,
                run_kind=RunKind.RESUME,
            ),
        ),
    )
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    continuation = RuntimeClientExecutionHarness.claude_continuation(
        provider_session_id=expected_session_id,
        provider_state_dir_relpath="",
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                token=CancellationToken(),
            )
        )
    )

    assert isinstance(outcome.kind, runtime.Cancelled)
    assert outcome.result.continuation is not None
    assert (
        outcome.result.continuation.provider_resume_state["provider_session_id"]
        == expected_session_id
    )
    assert _CancelAfterStartedAdapter.recorded_request_count == 1


def test_session_backed_claude_resumed_session_cancellation_before_provider_output_uses_fallback_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _CancelBeforeStartedAdapter:
        recorded_request_count = 0

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
            argv_transform=None,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            _CancelBeforeStartedAdapter.recorded_request_count += 1
            raise AgentCancelledError()

    monkeypatch.setattr(
        prompt_runtime._builtin_runtime_client_module,
        "_default_provider_invocation_adapter",
        lambda: _CancelBeforeStartedAdapter(),
    )
    session_id = "resumed-session-1"
    monkeypatch.setattr(
        session_backed_execution._provider_state_resolution,
        "resolve_claude_resumed_session_facts",
        lambda **_kwargs: provider_state_resolution.ClaudeResumedSessionResolution(
            provider_state_dir=tmp_path,
            continuation_input_facts=provider_state_resolution.claude_continuation_input_facts(
                model="sonnet",
                effort="medium",
                provider_state_dir=tmp_path,
                provider_state_dir_relpath="",
                provider_session_id=session_id,
                run_kind=RunKind.RESUME,
            ),
        ),
    )
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    continuation = RuntimeClientExecutionHarness.claude_continuation(
        provider_session_id=session_id,
        provider_state_dir_relpath="",
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(
                    claude_code_oauth_token="oauth-token"
                ),
                token=CancellationToken(),
            )
        )
    )

    assert isinstance(outcome.kind, runtime.Cancelled)
    assert outcome.result.continuation == continuation
    assert _CancelBeforeStartedAdapter.recorded_request_count == 1
