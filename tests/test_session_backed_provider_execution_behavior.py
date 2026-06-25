from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import agent_runtime as runtime
import agent_runtime._builtin_provider_rendering as built_in_provider_rendering
import agent_runtime._provider_invocation as provider_invocation_runtime
import agent_runtime._session_backed_provider_execution as session_backed_execution
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime
from tests.runtime_client_execution_harness import RuntimeClientExecutionHarness
from agent_runtime.errors import RuntimeConfigurationError, UsageLimitError
from agent_runtime.session import RunKind
from agent_runtime.types import ProviderSelection as InternalStageSelection


def _selection_with_auth(selection: Any, auth: Any) -> Any:
    return RuntimeClientExecutionHarness.attach_provider_auth(selection, auth)


def _install_in_memory_provider_invocation_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *prepared_invocations: (
        provider_invocation_runtime.ProviderInvocationResult
        | provider_invocation_runtime.ProviderInvocationFailure
        | provider_invocation_runtime.ProviderInvocationPreparedStream
    ),
) -> RuntimeClientExecutionHarness:
    return RuntimeClientExecutionHarness.install(monkeypatch).prepare_all(
        *prepared_invocations
    )


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
    harness = _install_in_memory_provider_invocation_adapter(
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
    harness = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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
        f"{'codex.cmd' if os.name == 'nt' else 'codex'} exec resume thread-123 -m gpt-5.4 "
        "-c model_reasoning_effort=medium -c approval_policy=never --sandbox read-only --json"
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
    harness = _install_in_memory_provider_invocation_adapter(
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
    harness = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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


@pytest.mark.parametrize(
    "rollout_lines",
    [
        ['{"type":"thread.started","thread_id":"thread-a"}\n'],
        [
            "{not-json\n",
            "[]\n",
            '{"type":"turn.completed"}\n',
            '{"type":"thread.started","thread_id":"   "}\n',
            '{"type":"thread.started"}\n',
        ],
    ],
)
def test_session_backed_codex_resumed_session_requires_recoverable_provider_state_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rollout_lines: list[str],
) -> None:
    RuntimeClientExecutionHarness.install_local_codex_host_auth(
        monkeypatch,
        tmp_path,
        auth_file_content='{"token":"host-auth"}\n',
    )
    harness = _install_in_memory_provider_invocation_adapter(monkeypatch)

    continuation = RuntimeClientExecutionHarness.codex_continuation()
    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
    )
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="codex",
    )
    if len(rollout_lines) == 1:
        RuntimeClientExecutionHarness.write_codex_rollout_state(
            provider_state_dir,
            rollout_lines[0] + '{"type":"thread.started","thread_id":"thread-b"}\n',
        )
    else:
        RuntimeClientExecutionHarness.write_codex_rollout_state(
            provider_state_dir,
            "".join(rollout_lines),
        )

    with pytest.raises(RuntimeConfigurationError) as exc_info:
        session_backed_execution._run_builtin_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
            )
        )

    assert str(exc_info.value) == (
        "Codex continuation is not recoverable from provider state."
    )
    assert harness.recorded_request_count == 0


def test_session_backed_opencode_completion_restores_portable_provider_state_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _SnapshottingOpencodeAdapter:
        def __init__(self) -> None:
            self.recorded_requests: list[
                provider_invocation_runtime.ProviderInvocationRequest
            ] = []
            self.state_dir: Path | None = None
            self.session_id_contents: str | None = None
            self.resume_jsonl_contents: str | None = None

        def execute(
            self,
            request: provider_invocation_runtime.ProviderInvocationRequest,
        ) -> provider_invocation_runtime.ProviderInvocationResult:
            self.recorded_requests.append(request)
            self.state_dir = Path(request.environment["OPENCODE_HOME"])
            self.session_id_contents = (self.state_dir / "session_id").read_text(
                encoding="utf-8"
            )
            self.resume_jsonl_contents = (self.state_dir / "resume.jsonl").read_text(
                encoding="utf-8"
            )
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

    result = session_backed_execution._run_builtin_resumed_session(
        RuntimeClientExecutionHarness.resume_session_run_request(
            invocation_dir=tmp_path,
            continuation=continuation,
            provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
        )
    )

    assert result.output == "continued output"
    assert result.selected == runtime.ResolvedProvider(
        service="opencode", model="glm-5.2", effort="medium"
    )
    assert result.continuation == RuntimeClientExecutionHarness.opencode_continuation()
    assert adapter.recorded_requests[0].run_kind is RunKind.RESUME
    assert adapter.recorded_requests[0].provider_session_id == "persisted-session-1"
    assert adapter.session_id_contents == "persisted-session-1\n"
    assert adapter.resume_jsonl_contents == "[]"
    assert adapter.state_dir is not None
    assert adapter.state_dir.exists() is False


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
    if entrypoint == "new":
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
    else:
        continuation = RuntimeClientExecutionHarness.opencode_continuation()
        session_backed_execution._run_builtin_resumed_session(
            RuntimeClientExecutionHarness.resume_session_run_request(
                invocation_dir=tmp_path,
                runtime_state_dir=runtime_state_dir,
                continuation=continuation,
                provider_auth=runtime.ProviderAuth(opencode_api_key="go-key"),
            )
        )

    assert len(adapter.recorded_requests) == 1
    recorded_request = adapter.recorded_requests[0]
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
    (
        "entrypoint",
        "expected_provider_session_id",
        "expected_provider_state",
        "expected_exact_transcript_match",
    ),
    [
        ("new", "provider-session-777", {"session_id": "provider-session-777"}, False),
        (
            "resumed",
            "persisted-session-1",
            {
                "session_id": "persisted-session-1",
                "resume_jsonl": "[]",
            },
            True,
        ),
    ],
)
def test_session_backed_opencode_expected_interruptions_keep_started_continuations_through_module_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    expected_provider_session_id: str,
    expected_provider_state: dict[str, str],
    expected_exact_transcript_match: bool,
) -> None:
    monkeypatch.setattr(
        prompt_runtime._time_module,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    if entrypoint == "new":
        monkeypatch.setattr(
            prompt_runtime._builtin_runtime_client_module,
            "_new_provider_session_id",
            lambda: "prepared-session-id",
        )
    adapter = _install_in_memory_provider_invocation_adapter(
        monkeypatch,
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

    with pytest.raises(UsageLimitError) as exc_info:
        if entrypoint == "new":
            session_backed_execution._run_builtin_new_session(
                RuntimeClientExecutionHarness.start_session_run_request(
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
    assert (
        exc_info.value.continuation
        == RuntimeClientExecutionHarness.opencode_continuation(
            provider_session_id=expected_provider_session_id,
            provider_state=expected_provider_state,
            exact_transcript_match=expected_exact_transcript_match,
        )
    )
    assert adapter.recorded_requests[0].provider_session_id == (
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

    runtime_state_dir = RuntimeClientExecutionHarness.prepare_runtime_state_dir(
        tmp_path
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

    assert (
        exc_info.value.continuation
        == RuntimeClientExecutionHarness.opencode_continuation(
            provider_session_id="observed-session-2",
            provider_state={
                "session_id": "observed-session-2",
                "resume_jsonl": "[]",
            },
            exact_transcript_match=False,
        )
    )
    assert adapter.recorded_requests[0].provider_session_id == "persisted-session-1"
    provider_state_dir = RuntimeClientExecutionHarness.provider_state_dir(
        runtime_state_dir,
        service="opencode",
    )
    assert (provider_state_dir / "session_id").read_text(encoding="utf-8").strip() == (
        "observed-session-2"
    )
