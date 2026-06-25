from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.errors import (
    AgentFailedError,
    AgentRuntimeError,
    AgentCredentialFailureError,
    AgentTimeoutError,
    HardAgentError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    TransientAgentError,
    UsageLimitError,
)


def test_agent_failed_error_rejects_unsafe_session_namespace_before_building_diagnostics() -> (
    None
):
    with pytest.raises(ValueError):
        AgentFailedError(
            invocation_role="implementer",
            worktree_path=Path("."),
            namespace="../escape",
        )


def test_agent_failed_error_rejects_unsafe_service_name_before_building_diagnostics() -> (
    None
):
    with pytest.raises(ValueError):
        AgentFailedError(
            invocation_role="implementer",
            worktree_path=Path("."),
            service_name="a/b",
        )


def test_agent_timeout_error_exposes_invocation_role_metadata() -> None:
    timeout = AgentTimeoutError(
        "timed out",
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
    )

    assert timeout.invocation_role == "reviewer"


def test_agent_failed_error_exposes_invocation_role_metadata() -> None:
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
    )

    assert failed.invocation_role == "reviewer"


def test_agent_failed_error_builds_session_dir_from_namespace_metadata() -> None:
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
        namespace="main",
    )

    assert failed.session_dir == "reviewer/main"


def test_agent_failed_error_builds_session_dir_from_namespace_and_service_name_metadata() -> (
    None
):
    failed = AgentFailedError(
        invocation_role="reviewer",
        worktree_path=Path("worktree"),
        namespace="main",
        service_name="codex",
    )

    assert failed.session_dir == "reviewer/main/codex"


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


def test_transient_agent_error_exposes_status_code_metadata() -> None:
    transient = TransientAgentError("transient", status_code=502)

    assert transient.status_code == 502


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


def test_agent_failed_error_builds_session_dir_from_service_name_metadata() -> None:
    failed = AgentFailedError("implementer", Path("worktree"), service_name="codex")

    assert failed.session_dir == "implementer/codex"
