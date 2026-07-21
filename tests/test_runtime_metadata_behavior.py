from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.errors import (
    AgentCredentialFailureError,
    AgentRuntimeError,
    AgentTimeoutError,
    HardAgentError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    UsageLimitError,
)


def test_agent_timeout_error_exposes_invocation_role_metadata() -> None:
    timeout = AgentTimeoutError(
        "timed out",
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
    )

    assert timeout.invocation_role == "reviewer"


@pytest.mark.parametrize("service_name", [" ", "a/b", "../escape"])
def test_hard_agent_error_rejects_unsafe_runtime_service_labels_before_recording_diagnostics(
    service_name: str,
) -> None:
    with pytest.raises(ValueError):
        HardAgentError("hard", service_name=service_name)


def test_agent_timeout_error_is_an_agent_runtime_error() -> None:
    timeout = AgentTimeoutError("timed out")

    assert isinstance(timeout, AgentRuntimeError)


def test_usage_limit_error_defaults_service_name_metadata_to_none() -> None:
    usage_limit = UsageLimitError(reset_time=None)

    assert usage_limit.service_name is None


def test_hard_agent_error_exposes_service_name_metadata() -> None:
    hard = HardAgentError("hard", service_name="codex")

    assert hard.service_name == "codex"


def test_provider_unavailable_error_omits_provider_diagnostic_metadata() -> None:
    provider_unavailable = ProviderUnavailableError(
        "retry",
        reason=ProviderUnavailableReason.TRANSIENT_API_ERROR,
        service_name="codex",
    )

    assert not hasattr(provider_unavailable, "status_code")
    assert not hasattr(provider_unavailable, "observations")
    assert not hasattr(provider_unavailable, "reset_time")
    assert not hasattr(provider_unavailable, "classification")


def test_hard_agent_error_omits_provider_diagnostic_metadata() -> None:
    hard = HardAgentError("hard", service_name="codex")

    assert not hasattr(hard, "status_code")
    assert not hasattr(hard, "observations")


def test_agent_credential_failure_error_omits_provider_diagnostic_metadata() -> None:
    credential_failure = AgentCredentialFailureError(
        "bad token",
        service_name="codex",
    )

    assert not hasattr(credential_failure, "status_code")
    assert not hasattr(credential_failure, "observations")
