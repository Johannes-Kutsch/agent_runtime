from __future__ import annotations

from datetime import datetime, timezone

import pytest

import agent_runtime as runtime
from agent_runtime._live_runtime_output_exceptions import (
    mark_live_runtime_output_exception,
)
from agent_runtime._runtime_outcome_folding import _fold_runtime_outcome
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
    raised: BaseException,
    expected_kind: object,
    expected_selected: ResolvedProvider,
) -> None:
    selected_provider_calls = 0

    def selected_provider() -> ResolvedProvider:
        nonlocal selected_provider_calls
        selected_provider_calls += 1
        return ResolvedProvider(service="claude", model="sonnet", effort="medium")

    outcome = _fold_runtime_outcome(
        lambda: (_ for _ in ()).throw(raised),
        selected_provider=selected_provider,
    )

    assert outcome.kind == expected_kind
    assert outcome.result.output == ""
    assert outcome.result.usage == getattr(raised, "usage")
    assert outcome.result.continuation is None
    assert outcome.result.selected == expected_selected
    assert selected_provider_calls == 1


def test_runtime_outcome_folding_accepts_selected_provider_facts() -> None:
    selected = ResolvedProvider(service="opencode", model="glm-5.2", effort="high")
    result = runtime.RunResult(
        output="final output",
        usage=ProviderUsage(input_tokens=11, output_tokens=12),
        continuation=None,
        selected=selected,
    )

    outcome = _fold_runtime_outcome(
        lambda: result,
        selected_provider=selected,
    )

    assert outcome == runtime.RuntimeOutcome(
        kind=runtime.Completed(),
        result=result,
    )


def test_runtime_outcome_folding_propagates_live_runtime_output_callback_failures() -> (
    None
):
    observer_failure = mark_live_runtime_output_exception(
        AgentCancelledError(
            usage=ProviderUsage(input_tokens=13, output_tokens=14),
        )
    )

    with pytest.raises(AgentCancelledError) as exc_info:
        _fold_runtime_outcome(
            lambda: (_ for _ in ()).throw(observer_failure),
            selected_provider=ResolvedProvider(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
        )

    assert exc_info.value is observer_failure


def test_runtime_outcome_folding_leaves_runtime_configuration_failures_exceptional() -> (
    None
):
    with pytest.raises(RuntimeConfigurationError, match="misconfigured runtime"):
        _fold_runtime_outcome(
            lambda: (_ for _ in ()).throw(
                RuntimeConfigurationError("misconfigured runtime")
            ),
            selected_provider=ResolvedProvider(
                service="claude",
                model="sonnet",
                effort="medium",
            ),
        )
