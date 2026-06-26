from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

import agent_runtime as runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime._live_runtime_output_exceptions import (
    mark_live_runtime_output_exception,
)
from agent_runtime.errors import (
    AgentCancelledError,
    AgentTimeoutError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    RuntimeConfigurationError,
    UsageLimitError,
)
from agent_runtime.provider_usage import ProviderUsage
from agent_runtime.types import ResolvedProvider


@pytest.mark.parametrize(
    ("raised", "expected_kind", "expected_selected"),
    [
        pytest.param(
            AgentCancelledError(
                usage=ProviderUsage(input_tokens=1, output_tokens=2),
            ),
            runtime.Cancelled(),
            ResolvedProvider(service="claude", model="sonnet", effort="medium"),
            id="cancelled",
        ),
        pytest.param(
            AgentTimeoutError(
                "timed out",
                usage=ProviderUsage(input_tokens=3, output_tokens=4),
            ),
            runtime.TimedOut(),
            ResolvedProvider(service="claude", model="sonnet", effort="medium"),
            id="timed-out",
        ),
        pytest.param(
            ProviderUnavailableError(
                message="provider unavailable",
                reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
                service_name="codex",
                usage=ProviderUsage(input_tokens=5, output_tokens=6),
            ),
            runtime.ProviderUnavailable(
                reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
                detail="provider unavailable",
            ),
            ResolvedProvider(service="codex", model="sonnet", effort="medium"),
            id="provider-unavailable",
        ),
        pytest.param(
            UsageLimitError(
                reset_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
                service_name="claude",
                usage=ProviderUsage(input_tokens=7, output_tokens=8),
            ),
            runtime.UsageLimited(reset_time=datetime(2026, 1, 2, tzinfo=timezone.utc)),
            ResolvedProvider(service="claude", model="sonnet", effort="medium"),
            id="usage-limited",
        ),
    ],
)
def test_runtime_outcome_folding_maps_expected_interruptions_to_runtime_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raised: BaseException,
    expected_kind: object,
    expected_selected: ResolvedProvider,
) -> None:
    def _run_builtin_ephemeral(
        *_args: object,
        **_kwargs: object,
    ) -> runtime.RunResult:
        raise raised

    monkeypatch.setattr(
        prompt_runtime, "_run_builtin_ephemeral", _run_builtin_ephemeral
    )

    request = prompt_runtime.EphemeralRunRequest(
        prompt="hello",
        invocation_dir=tmp_path / "invocation",
        provider_selection=runtime.ProviderSelection(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        tool_policy=runtime.ToolPolicy.NONE,
    )

    outcome = asyncio.run(runtime.RuntimeClient().run_ephemeral(request))

    assert outcome.kind == expected_kind
    assert outcome.result.output == ""
    assert outcome.result.usage == getattr(raised, "usage")
    assert outcome.result.continuation is None
    assert outcome.result.selected == expected_selected


def test_runtime_outcome_folding_accepts_selected_provider_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_by_backend = ResolvedProvider(
        service="opencode",
        model="glm-5.2",
        effort="high",
    )
    continuation = runtime.Continuation(
        serialized='{"runtime_version":"1.0","provider_resume_state":"unit-test"}'
    )

    def _run_builtin_new_session(
        *_args: object,
        **_kwargs: object,
    ) -> runtime.RunResult:
        return runtime.RunResult(
            output="final output",
            usage=ProviderUsage(input_tokens=11, output_tokens=12),
            continuation=continuation,
            selected=selected_by_backend,
        )

    monkeypatch.setattr(
        prompt_runtime, "_run_builtin_new_session", _run_builtin_new_session
    )

    request = prompt_runtime.NewSessionRunRequest(
        prompt="hello",
        invocation_dir=tmp_path / "new-session",
        session_store=tmp_path / "session-store",
        provider_selection=runtime.ProviderSelection(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        tool_policy=runtime.ToolPolicy.NONE,
    )

    outcome = asyncio.run(runtime.RuntimeClient().run_new_session(request))

    assert outcome == runtime.RuntimeOutcome(
        kind=runtime.Completed(),
        result=runtime.RunResult(
            output="final output",
            usage=ProviderUsage(input_tokens=11, output_tokens=12),
            continuation=continuation,
            selected=selected_by_backend,
        ),
    )


@pytest.mark.parametrize(
    "delegate_error",
    [
        pytest.param(
            AgentCancelledError(usage=ProviderUsage(input_tokens=13, output_tokens=14)),
            id="cancelled",
        ),
    ],
)
def test_runtime_outcome_folding_propagates_live_runtime_output_callback_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    delegate_error: BaseException,
) -> None:
    observer_failure = mark_live_runtime_output_exception(delegate_error)

    def _run_builtin_ephemeral(
        *_args: object,
        **_kwargs: object,
    ) -> runtime.RunResult:
        raise observer_failure

    monkeypatch.setattr(
        prompt_runtime, "_run_builtin_ephemeral", _run_builtin_ephemeral
    )

    request = prompt_runtime.EphemeralRunRequest(
        prompt="hello",
        invocation_dir=tmp_path / "invocation",
        provider_selection=runtime.ProviderSelection(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        tool_policy=runtime.ToolPolicy.NONE,
    )

    with pytest.raises(AgentCancelledError) as exc_info:
        asyncio.run(runtime.RuntimeClient().run_ephemeral(request))

    assert exc_info.value is observer_failure


def test_runtime_outcome_folding_leaves_runtime_configuration_failures_exceptional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _run_builtin_ephemeral(
        *_args: object,
        **_kwargs: object,
    ) -> runtime.RunResult:
        raise RuntimeConfigurationError("misconfigured runtime")

    monkeypatch.setattr(
        prompt_runtime, "_run_builtin_ephemeral", _run_builtin_ephemeral
    )

    request = prompt_runtime.EphemeralRunRequest(
        prompt="hello",
        invocation_dir=tmp_path / "invocation",
        provider_selection=runtime.ProviderSelection(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        tool_policy=runtime.ToolPolicy.NONE,
    )

    with pytest.raises(RuntimeConfigurationError, match="misconfigured runtime"):
        asyncio.run(runtime.RuntimeClient().run_ephemeral(request))
