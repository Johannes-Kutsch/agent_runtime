from __future__ import annotations

import json

import pytest

from agent_runtime._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
)
from agent_runtime._session_backed_provider_lifecycle_policy import policy_for_service
from agent_runtime.errors import (
    ContinuationUnrecoverableError,
    RuntimeConfigurationError,
    UsageLimitError,
)


def test_policy_for_service_returns_claude_stream_interpretation() -> None:
    result = policy_for_service("claude").stream_interpretation()
    assert isinstance(result, BuiltInProviderStreamInterpretation)


def test_policy_for_service_returns_codex_stream_interpretation() -> None:
    result = policy_for_service("codex").stream_interpretation()
    assert isinstance(result, BuiltInProviderStreamInterpretation)


def test_policy_for_service_returns_opencode_stream_interpretation() -> None:
    result = policy_for_service("opencode").stream_interpretation()
    assert isinstance(result, BuiltInProviderStreamInterpretation)


def test_policy_for_service_claude_stream_interpretation_attributes_errors_to_claude_service() -> (
    None
):
    interpretation = policy_for_service("claude").stream_interpretation()
    line = (
        json.dumps(
            {
                "type": "result",
                "is_error": True,
                "errors": [
                    {"message": "No conversation found with session ID abc-123"}
                ],
            }
        )
        + "\n"
    )
    with pytest.raises(ContinuationUnrecoverableError) as exc_info:
        interpretation.reduce_output([line])
    assert exc_info.value.service_name == "claude"


def test_policy_for_service_codex_stream_interpretation_raises_usage_limit_with_codex_service_name() -> (
    None
):
    interpretation = policy_for_service("codex").stream_interpretation()
    with pytest.raises(UsageLimitError) as exc_info:
        interpretation.reduce_output(
            [
                json.dumps({"type": "error", "message": "You've hit your usage limit."})
                + "\n"
            ]
        )
    assert exc_info.value.service_name == "codex"


def test_policy_for_service_raises_runtime_configuration_error_for_unknown_service() -> (
    None
):
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        policy_for_service("unknown")
    assert str(exc_info.value) == (
        "RuntimeClient session-backed execution is only implemented for Claude, Codex, and OpenCode."
    )


@pytest.mark.parametrize("service_name", ["", "CLAUDE", "Claude", "gpt", "gemini"])
def test_policy_for_service_raises_for_any_unrecognized_service_name(
    service_name: str,
) -> None:
    with pytest.raises(RuntimeConfigurationError):
        policy_for_service(service_name)
