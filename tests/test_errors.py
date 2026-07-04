import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
from agent_runtime.errors import (
    AgentRuntimeError,
    ContinuationUnrecoverableError,
    ModelNotAvailableError,
    RuntimeConfigurationError,
)


def test_model_not_available_error_is_agent_runtime_error() -> None:
    err = ModelNotAvailableError(
        "codex model not available for this account.",
        service_name="codex",
        raw_message="model not available on free tier",
    )
    assert isinstance(err, AgentRuntimeError)
    assert not isinstance(err, RuntimeConfigurationError)
    assert str(err) == "codex model not available for this account."
    assert err.service_name == "codex"
    assert err.raw_message == "model not available on free tier"


def test_model_not_available_error_defaults() -> None:
    err = ModelNotAvailableError(service_name="codex")
    assert err.raw_message is None
    assert err.invocation_progress.name == "NOT_STARTED"
    assert err.continuation is None
    assert err.usage is None


def test_model_not_available_error_rejects_invalid_service_name() -> None:
    with pytest.raises(ValueError):
        ModelNotAvailableError(service_name="INVALID SERVICE NAME")


def test_model_not_available_error_is_not_on_runtime_public_surface() -> None:
    assert not hasattr(runtime, "ModelNotAvailableError")
    assert "ModelNotAvailableError" not in runtime.__all__


def test_model_unavailable_contract_carries_service_name_and_raw_message() -> None:
    event = contracts_runtime.ModelUnavailable(
        service_name="codex",
        raw_message="model not available",
    )
    assert event.service_name == "codex"
    assert event.raw_message == "model not available"
    assert "ModelUnavailable" in contracts_runtime.__all__


def test_continuation_unrecoverable_error_is_agent_runtime_error() -> None:
    err = ContinuationUnrecoverableError(
        "Codex continuation is not recoverable from provider state.",
        service_name="codex",
    )
    assert isinstance(err, AgentRuntimeError)
    assert not isinstance(err, RuntimeConfigurationError)
    assert str(err) == "Codex continuation is not recoverable from provider state."
    assert err.service_name == "codex"


def test_continuation_unrecoverable_error_defaults_optional_fields() -> None:
    err = ContinuationUnrecoverableError(service_name="codex")
    assert err.classification is None
    assert err.raw_message is None


def test_continuation_unrecoverable_error_carries_classification_and_raw_message() -> (
    None
):
    err = ContinuationUnrecoverableError(
        "Claude session not found.",
        service_name="claude",
        classification="session_not_found",
        raw_message="Session abc123 does not exist",
    )
    assert err.classification == "session_not_found"
    assert err.raw_message == "Session abc123 does not exist"
    assert err.service_name == "claude"
    assert str(err) == "Claude session not found."


def test_continuation_unrecoverable_error_rejects_invalid_service_name() -> None:
    with pytest.raises(ValueError):
        ContinuationUnrecoverableError(service_name="INVALID SERVICE NAME")
