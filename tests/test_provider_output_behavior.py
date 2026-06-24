from __future__ import annotations


import pytest

from agent_runtime.invocation_progress import InvocationProgress as _InvocationProgress
from agent_runtime.contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    ModelActivity,
    PromptTokens,
    Result,
    TransientError,
    UnsupportedTokens,
    UsageLimit,
)
from agent_runtime.errors import (
    AgentCredentialFailureError,
    HardAgentError,
    RetryableProviderFailureError,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime.provider_output import reduce_text_output_events


def test_provider_output_reduction_returns_result() -> None:
    turns: list[str] = []

    result = reduce_text_output_events(
        [
            PromptTokens(2),
            UnsupportedTokens(3, "source"),
            AssistantTurn("hello"),
            Result("done"),
        ],
        lambda turn, _raw: turns.append(turn),
        provider="codex",
    )

    assert result == "done"
    assert turns == ["hello"]


def test_provider_output_reduction_reports_prompt_tokens() -> None:
    token_counts: list[int] = []

    result = reduce_text_output_events(
        [PromptTokens(2)],
        lambda _turn, _raw: None,
        token_counts.append,
        provider="codex",
    )

    assert result == ""
    assert token_counts == [2]


def test_provider_output_reduction_maps_usage_limit() -> None:
    with pytest.raises(UsageLimitError) as exc_info:
        reduce_text_output_events(
            [UsageLimit(reset_time=None)],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.service_name == "codex"
    assert exc_info.value.reset_time is None


def test_provider_output_reduction_accepts_explicit_model_activity_for_usage_limit() -> (
    None
):
    with pytest.raises(UsageLimitError) as exc_info:
        reduce_text_output_events(
            [ModelActivity(), UsageLimit(reset_time=None)],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.invocation_progress is _InvocationProgress.STARTED


def test_provider_output_reduction_keeps_unknown_activity_usage_limits_not_started() -> (
    None
):
    with pytest.raises(UsageLimitError) as exc_info:
        reduce_text_output_events(
            [
                PromptTokens(2),
                UnsupportedTokens(3, "source"),
                UsageLimit(reset_time=None),
            ],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.invocation_progress is _InvocationProgress.NOT_STARTED


def test_provider_output_reduction_maps_transient_error() -> None:
    with pytest.raises(TransientAgentError) as exc_info:
        reduce_text_output_events(
            [TransientError(status_code=503, raw_message="retry")],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.status_code == 503
    assert str(exc_info.value) == "retry"


def test_provider_output_reduction_maps_retryable_provider_failure() -> None:
    with pytest.raises(RetryableProviderFailureError) as exc_info:
        reduce_text_output_events(
            [
                TransientError(
                    status_code=503,
                    raw_message="retry",
                    classification="retryable",
                )
            ],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.service_name == "codex"
    assert exc_info.value.invocation_progress is _InvocationProgress.NOT_STARTED
    assert str(exc_info.value) == "retry"
    assert not hasattr(exc_info.value, "status_code")
    assert not hasattr(exc_info.value, "observations")


def test_provider_output_reduction_reports_started_progress_for_retryable_provider_failure() -> (
    None
):
    with pytest.raises(RetryableProviderFailureError) as exc_info:
        reduce_text_output_events(
            [
                AssistantTurn("hello"),
                TransientError(
                    status_code=503,
                    raw_message="retry",
                    classification="retryable",
                ),
            ],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.invocation_progress is _InvocationProgress.STARTED


def test_provider_output_reduction_accepts_explicit_model_activity_for_retryable_provider_failure() -> (
    None
):
    with pytest.raises(RetryableProviderFailureError) as exc_info:
        reduce_text_output_events(
            [
                ModelActivity(),
                TransientError(
                    status_code=503,
                    raw_message="retry",
                    classification="retryable",
                ),
            ],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.invocation_progress is _InvocationProgress.STARTED


def test_provider_output_reduction_maps_hard_error() -> None:
    with pytest.raises(HardAgentError) as exc_info:
        reduce_text_output_events(
            [HardError(status_code=400, raw_message="bad")],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.service_name == "codex"
    assert str(exc_info.value) == "bad"
    assert not hasattr(exc_info.value, "status_code")
    assert not hasattr(exc_info.value, "observations")


def test_provider_output_reduction_maps_credential_failure() -> None:
    with pytest.raises(AgentCredentialFailureError) as exc_info:
        reduce_text_output_events(
            [
                CredentialFailure(
                    raw_message="missing auth",
                    service_name="codex",
                    classification="operator_actionable_agent_credential_failure",
                )
            ],
            lambda _turn, _raw: None,
            provider="codex",
        )

    assert exc_info.value.service_name == "codex"
    assert exc_info.value.classification == (
        "operator_actionable_agent_credential_failure"
    )
    assert str(exc_info.value) == "missing auth"
    assert not hasattr(exc_info.value, "status_code")
    assert not hasattr(exc_info.value, "observations")


def test_provider_output_reduction_joins_assistant_turns_without_result() -> None:
    turns: list[str] = []

    result = reduce_text_output_events(
        [
            PromptTokens(2),
            AssistantTurn("hello"),
            UnsupportedTokens(3, "source"),
            AssistantTurn("world"),
        ],
        lambda turn, _raw: turns.append(turn),
        provider="codex",
    )

    assert result == "hello\nworld"
    assert turns == ["hello", "world"]


def test_provider_output_reduction_stops_after_result() -> None:
    turns: list[str] = []
    token_counts: list[int] = []

    result = reduce_text_output_events(
        [
            AssistantTurn("hello"),
            Result("done"),
            PromptTokens(99),
            AssistantTurn("ignored"),
        ],
        lambda turn, _raw: turns.append(turn),
        token_counts.append,
        provider="codex",
    )

    assert result == "done"
    assert turns == ["hello"]
    assert token_counts == []
