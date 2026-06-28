import pytest

from agent_runtime.errors import (
    AgentRuntimeError,
    ContinuationUnrecoverableError,
    RuntimeConfigurationError,
)


def test_continuation_unrecoverable_error_is_agent_runtime_error() -> None:
    err = ContinuationUnrecoverableError(
        "Codex continuation is not recoverable from provider state.",
        service_name="codex",
    )
    assert isinstance(err, AgentRuntimeError)
    assert not isinstance(err, RuntimeConfigurationError)
    assert str(err) == "Codex continuation is not recoverable from provider state."
    assert err.service_name == "codex"


def test_continuation_unrecoverable_error_rejects_invalid_service_name() -> None:
    with pytest.raises(ValueError):
        ContinuationUnrecoverableError(service_name="INVALID SERVICE NAME")
