from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import agent_runtime as runtime
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime.errors import ProviderUnavailableReason
from agent_runtime.errors import RuntimeConfigurationError
from agent_runtime.session import RunKind
from agent_runtime.types import ProviderSelection as InternalProviderSelection
from tests.runtime_client_execution_harness import RuntimeClientExecutionHarness


def _claude_assistant_output_line(text: str) -> str:
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
        )
        + "\n"
    )


def _claude_result_output_line(text: str) -> str:
    return json.dumps({"type": "result", "result": text}) + "\n"


def _claude_selection() -> InternalProviderSelection:
    return InternalProviderSelection(
        service="claude",
        model="sonnet",
        effort="medium",
        auth=runtime.ProviderAuth(claude_code_oauth_token="oauth-token"),
    )


def _new_session_request(
    harness: RuntimeClientExecutionHarness,
    tmp_path: Path,
) -> prompt_runtime.NewSessionRunRequest:
    return harness.start_session_run_request(
        invocation_dir=tmp_path,
        runtime_state_dir=harness.prepare_runtime_state_dir(tmp_path),
        provider_selection=_claude_selection(),
    )


def test_runtime_client_execution_harness_records_built_in_provider_invocation_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=(
                _claude_assistant_output_line("hello from claude"),
                _claude_result_output_line("final output"),
            ),
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=_claude_selection(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.Completed)
    assert outcome.result.output == "final output"
    assert harness.recorded_request_count == 1
    assert harness.recorded_request().worktree == tmp_path


def test_runtime_client_execution_harness_records_provider_invocation_request_without_logging_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_result(
        provider_invocation_runtime.ProviderInvocationResult(output="done")
    )

    asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=_claude_selection(),
            )
        )
    )

    assert tuple(harness.recorded_request().__dataclass_fields__) == (
        "worktree",
        "environment",
        "prompt",
        "run_kind",
        "provider_session_id",
        "output_hooks",
        "command",
        "argv",
        "prefer_argv",
        "timeout_seconds",
        "token",
    )


@pytest.mark.parametrize(
    ("prepare", "run_request", "expected_kind", "expected_output"),
    [
        (
            lambda harness: harness.prepare_result(
                provider_invocation_runtime.ProviderInvocationResult(
                    output="result output",
                    usage=runtime.ProviderUsage(output_tokens=2),
                )
            ),
            lambda harness, tmp_path: harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=_claude_selection(),
                tool_access=contracts_runtime.ToolAccess.no_tools(),
            ),
            prompt_runtime.Completed,
            "result output",
        ),
        (
            lambda harness: harness.prepare_failure(
                provider_invocation_runtime.ProviderInvocationFailure(
                    kind=(
                        provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE
                    ),
                    detail="provider unavailable",
                )
            ),
            _new_session_request,
            prompt_runtime.ProviderUnavailable,
            "",
        ),
        (
            lambda harness: harness.prepare_prepared_stream(
                provider_invocation_runtime.ProviderInvocationPreparedStream(
                    stdout_lines=(
                        _claude_assistant_output_line("hello from claude"),
                        _claude_result_output_line("stream output"),
                    ),
                )
            ),
            lambda harness, tmp_path: harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=_claude_selection(),
            ),
            prompt_runtime.Completed,
            "stream output",
        ),
    ],
)
def test_runtime_client_execution_harness_prepares_provider_invocation_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prepare,
    run_request,
    expected_kind: type[object],
    expected_output: str,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    prepare(harness)
    if expected_kind is prompt_runtime.ProviderUnavailable:
        outcome = asyncio.run(
            runtime.RuntimeClient().run_new_session(run_request(harness, tmp_path))
        )
    else:
        outcome = asyncio.run(
            runtime.RuntimeClient().run_ephemeral(run_request(harness, tmp_path))
        )

    assert isinstance(outcome.kind, expected_kind)
    assert harness.recorded_request_count == 1
    if isinstance(outcome.kind, prompt_runtime.ProviderUnavailable):
        assert outcome.kind.reason is ProviderUnavailableReason.TRANSIENT_API_ERROR
    else:
        assert outcome.result.output == expected_output


def test_runtime_client_ephemeral_run_returns_provider_unavailable_outcome_on_invocation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    failure_usage = runtime.ProviderUsage(input_tokens=2, output_tokens=1)
    harness.prepare_failure(
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE,
            detail="temporary provider failure",
            usage=failure_usage,
            stdout_lines=("ignored line\n",),
            provider_session_id="provider-session-123",
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=_claude_selection(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.ProviderUnavailable)
    assert outcome.kind.reason is ProviderUnavailableReason.TRANSIENT_API_ERROR
    assert outcome.kind.detail == "temporary provider failure"
    assert outcome.result.output == ""
    assert outcome.result.usage == failure_usage
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude",
        model="sonnet",
        effort="medium",
    )
    assert outcome.result.continuation is None


def test_runtime_client_ephemeral_run_returns_service_not_available_outcome_before_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_failure(
        provider_invocation_runtime.ProviderInvocationFailure(
            kind=provider_invocation_runtime.InvocationFailureKind.PROVIDER_UNAVAILABLE,
            detail="No configured service candidates are currently available.",
        )
    )

    outcome = asyncio.run(
        runtime.RuntimeClient().run_ephemeral(
            harness.ephemeral_run_request(
                invocation_dir=tmp_path,
                provider_selection=_claude_selection(),
            )
        )
    )

    assert isinstance(outcome.kind, prompt_runtime.ProviderUnavailable)
    assert outcome.kind.reason is ProviderUnavailableReason.SERVICE_NOT_AVAILABLE
    assert (
        outcome.kind.detail
        == "No configured service candidates are currently available."
    )
    assert outcome.result.output == ""
    assert outcome.result.usage is None
    assert outcome.result.selected == runtime.ResolvedProvider(
        service="claude",
        model="sonnet",
        effort="medium",
    )
    assert outcome.result.continuation is None


def test_runtime_session_requests_require_session_store_for_session_backed_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = RuntimeClientExecutionHarness.install(monkeypatch)
    harness.prepare_prepared_stream(
        provider_invocation_runtime.ProviderInvocationPreparedStream(
            stdout_lines=('{"type":"result","output":"should not run"}\n',)
        )
    )

    provider_selection = InternalProviderSelection(
        service="opencode",
        model="glm-5.2",
        effort="medium",
        auth=runtime.ProviderAuth(opencode_api_key="go-key"),
    )
    start_request = RuntimeClientExecutionHarness.start_session_run_request(
        invocation_dir=tmp_path,
        provider_selection=provider_selection,
        runtime_state_dir=None,
    )
    resume_request = RuntimeClientExecutionHarness.resume_session_run_request(
        invocation_dir=tmp_path,
        continuation=RuntimeClientExecutionHarness.opencode_continuation(),
        provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
    )

    with pytest.raises(RuntimeConfigurationError, match="session_store"):
        asyncio.run(runtime.RuntimeClient().run_new_session(start_request))

    with pytest.raises(RuntimeConfigurationError, match="session_store"):
        asyncio.run(runtime.RuntimeClient().run_resumed_session(resume_request))


def test_runtime_client_reuses_session_store_across_new_and_resumed_session_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        provider_invocation_runtime.ProviderInvocationResult(
            output="first output",
            provider_session_id="thread-123",
            usage=runtime.ProviderUsage(input_tokens=1, output_tokens=1),
            stdout_lines=(),
        ),
        provider_invocation_runtime.ProviderInvocationResult(
            output="second output",
            provider_session_id="thread-123",
            usage=runtime.ProviderUsage(input_tokens=1, output_tokens=1),
            stdout_lines=(),
        ),
    )

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    start_request = RuntimeClientExecutionHarness.start_session_run_request(
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        provider_selection=InternalProviderSelection(
            service="codex",
            model="gpt-5.4",
            effort="medium",
        ),
    )

    start_outcome = asyncio.run(runtime.RuntimeClient().run_new_session(start_request))
    continuation = start_outcome.result.continuation
    assert continuation is not None
    assert start_outcome.result.output == "first output"
    resumed_continuation = runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=continuation.tool_access,
        provider_resume_state={
            **continuation.provider_resume_state,
            "provider_session_id": "   ",
        },
    )

    RuntimeClientExecutionHarness.prepare_codex_rollout_state(
        RuntimeClientExecutionHarness.provider_state_dir(
            runtime_state_dir,
            service="codex",
        ),
        "thread-123",
    )
    resume_request = RuntimeClientExecutionHarness.resume_session_run_request(
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        continuation=resumed_continuation,
    )
    resume_outcome = asyncio.run(
        runtime.RuntimeClient().run_resumed_session(resume_request)
    )
    assert resume_outcome.result.output == "second output"
    assert harness.recorded_request_count == 2
    recorded_resume_request = harness.recorded_request(1)
    assert recorded_resume_request.run_kind is RunKind.RESUME
    assert recorded_resume_request.provider_session_id == "thread-123"


def test_runtime_client_execution_harness_builds_session_lifecycle_requests(
    tmp_path: Path,
) -> None:
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_auth = runtime.ProviderAuth(claude_code_oauth_token="override-token")
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        tmp_path,
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    start_request = RuntimeClientExecutionHarness.start_session_run_request(
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        provider_selection=_claude_selection(),
        provider_auth=provider_auth,
        tool_access=tool_access,
    )
    resume_request = RuntimeClientExecutionHarness.resume_session_run_request(
        invocation_dir=tmp_path,
        runtime_state_dir=runtime_state_dir,
        continuation=RuntimeClientExecutionHarness.codex_continuation(
            tool_access=tool_access,
        ),
        provider_auth=provider_auth,
    )

    assert isinstance(start_request, prompt_runtime.NewSessionRunRequest)
    assert start_request.invocation_dir == tmp_path
    assert start_request.provider_selection.auth == provider_auth
    assert start_request.tool_access == tool_access
    assert start_request._runtime_state_dir == runtime_state_dir
    assert isinstance(resume_request, prompt_runtime.ResumedSessionRunRequest)
    assert (
        resume_request.continuation
        == RuntimeClientExecutionHarness.codex_continuation(
            tool_access=tool_access,
        )
    )
    assert resume_request.provider_auth == provider_auth
    assert resume_request.tool_access == tool_access
    assert resume_request._runtime_state_dir == runtime_state_dir


def test_runtime_client_execution_harness_prepares_runtime_state_and_codex_rollout_state(
    tmp_path: Path,
) -> None:
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="codex",
    )

    rollout_path = RuntimeClientExecutionHarness.prepare_codex_rollout_state(
        provider_state_dir,
        "thread-1",
        "thread-2",
    )

    assert runtime_state_dir == tmp_path / ".agent-runtime" / "state"
    assert runtime_state_dir.is_dir()
    assert provider_state_dir == runtime_state_dir
    assert (
        rollout_path
        == provider_state_dir / "sessions" / "2026" / "05" / "30" / "rollout-001.jsonl"
    )
    assert rollout_path.read_text(encoding="utf-8").splitlines() == [
        json.dumps({"type": "session_meta", "payload": {"id": "thread-1"}}),
        json.dumps({"type": "session_meta", "payload": {"id": "thread-2"}}),
    ]


def test_runtime_client_execution_harness_writes_exact_codex_rollout_state_content(
    tmp_path: Path,
) -> None:
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    rollout_content = (
        "{not-json\n"
        '{"type":"session_meta","payload":{"id":"thread-a"}}\n'
        '{"type":"session_meta","payload":{"id":"   "}}\n'
    )

    rollout_path = RuntimeClientExecutionHarness.write_codex_rollout_state(
        provider_state_dir,
        rollout_content,
    )

    assert (
        rollout_path
        == provider_state_dir / "sessions" / "2026" / "05" / "30" / "rollout-001.jsonl"
    )
    assert rollout_path.read_text(encoding="utf-8") == rollout_content


def test_runtime_client_execution_harness_attaches_provider_auth_without_changing_selection_identity() -> (
    None
):
    selection = InternalProviderSelection(
        service="claude",
        model="sonnet",
        effort="medium",
        auth=runtime.ProviderAuth(claude_code_oauth_token="original-token"),
    )
    replacement_auth = runtime.ProviderAuth(claude_code_oauth_token="replacement-token")

    attached = RuntimeClientExecutionHarness.attach_provider_auth(
        selection,
        replacement_auth,
    )

    assert attached == InternalProviderSelection(
        service="claude",
        model="sonnet",
        effort="medium",
        auth=replacement_auth,
    )
    assert selection.auth == runtime.ProviderAuth(
        claude_code_oauth_token="original-token"
    )


def test_runtime_client_execution_harness_installs_local_codex_host_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_auth_path = RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )

    assert host_auth_path == tmp_path / "host-home" / ".codex" / "auth.json"
    assert host_auth_path.read_text(encoding="utf-8") == '{"token":"host-auth"}\n'
    assert prompt_runtime._builtin_runtime_client_module.Path.home() == (
        tmp_path / "host-home"
    )


def test_runtime_client_execution_harness_opencode_continuation_no_provider_state() -> (
    None
):
    continuation = RuntimeClientExecutionHarness.opencode_continuation(
        provider_session_id="persisted-session-1",
    )

    assert continuation.provider_resume_state == {
        "provider_session_id": "persisted-session-1",
        "exact_transcript_match": True,
    }
