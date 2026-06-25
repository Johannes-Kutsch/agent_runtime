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
    _harness: RuntimeClientExecutionHarness,
    tmp_path: Path,
) -> prompt_runtime.NewSessionRunRequest:
    return prompt_runtime.NewSessionRunRequest(
        prompt="already rendered prompt",
        invocation_dir=tmp_path,
        runtime_state_dir=tmp_path / ".agent-runtime" / "state",
        provider_selection=_claude_selection(),
        session_namespace="main",
        tool_policy=runtime.ToolPolicy.NONE,
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
    assert len(harness.recorded_requests) == 1
    assert harness.recorded_requests[0].worktree == tmp_path


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
    assert len(harness.recorded_requests) == 1
    if isinstance(outcome.kind, prompt_runtime.ProviderUnavailable):
        assert outcome.kind.reason is ProviderUnavailableReason.TRANSIENT_API_ERROR
    else:
        assert outcome.result.output == expected_output
