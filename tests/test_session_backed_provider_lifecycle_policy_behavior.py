from __future__ import annotations

import json

import pytest

from agent_runtime._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
)
from agent_runtime._runtime_lifecycle import ProviderAuth
from agent_runtime._session_backed_provider_lifecycle_policy import policy_for_service
from agent_runtime.errors import (
    AgentCredentialFailureError,
    ContinuationUnrecoverableError,
    RuntimeConfigurationError,
    UsageLimitError,
)
from agent_runtime.types import ProviderSelection


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


# validate_stage tests


def test_claude_policy_validate_stage_raises_for_unsupported_model() -> None:
    selection = ProviderSelection(service="claude", model="gpt-5.5", effort="medium")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported Claude model"):
        policy_for_service("claude").validate_stage(selection)


def test_claude_policy_validate_stage_raises_for_unsupported_effort() -> None:
    selection = ProviderSelection(service="claude", model="sonnet", effort="turbo")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported Claude effort"):
        policy_for_service("claude").validate_stage(selection)


def test_claude_policy_validate_stage_passes_for_valid_selection() -> None:
    selection = ProviderSelection(service="claude", model="sonnet", effort="medium")
    policy_for_service("claude").validate_stage(selection)


def test_codex_policy_validate_stage_raises_for_unsupported_model() -> None:
    selection = ProviderSelection(
        service="codex", model="claude-sonnet", effort="medium"
    )
    with pytest.raises(RuntimeConfigurationError, match="Unsupported Codex model"):
        policy_for_service("codex").validate_stage(selection)


def test_codex_policy_validate_stage_raises_for_unsupported_effort() -> None:
    selection = ProviderSelection(service="codex", model="gpt-5.5", effort="max")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported Codex effort"):
        policy_for_service("codex").validate_stage(selection)


def test_codex_policy_validate_stage_passes_for_valid_selection() -> None:
    selection = ProviderSelection(service="codex", model="gpt-5.5", effort="medium")
    policy_for_service("codex").validate_stage(selection)


def test_opencode_policy_validate_stage_raises_for_unsupported_model() -> None:
    selection = ProviderSelection(service="opencode", model="gpt-5.5", effort="medium")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported OpenCode model"):
        policy_for_service("opencode").validate_stage(selection)


def test_opencode_policy_validate_stage_raises_for_unsupported_effort() -> None:
    selection = ProviderSelection(service="opencode", model="kimi-k2.6", effort="high")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported OpenCode effort"):
        policy_for_service("opencode").validate_stage(selection)


def test_opencode_policy_validate_stage_passes_for_valid_selection() -> None:
    selection = ProviderSelection(
        service="opencode", model="kimi-k2.6", effort="medium"
    )
    policy_for_service("opencode").validate_stage(selection)


# require_auth tests


def test_claude_policy_require_auth_raises_when_auth_is_none() -> None:
    with pytest.raises(
        AgentCredentialFailureError, match="Missing Claude Code OAuth token"
    ) as exc_info:
        policy_for_service("claude").require_auth(None)
    assert exc_info.value.service_name == "claude"


def test_claude_policy_require_auth_raises_when_token_is_missing() -> None:
    with pytest.raises(
        AgentCredentialFailureError, match="Missing Claude Code OAuth token"
    ):
        policy_for_service("claude").require_auth(
            ProviderAuth(claude_code_oauth_token=None)
        )


def test_claude_policy_require_auth_passes_with_valid_token() -> None:
    policy_for_service("claude").require_auth(
        ProviderAuth(claude_code_oauth_token="tok-abc")
    )


def test_codex_policy_require_auth_is_noop_for_none() -> None:
    policy_for_service("codex").require_auth(None)


def test_codex_policy_require_auth_is_noop_for_any_auth_value() -> None:
    policy_for_service("codex").require_auth(ProviderAuth())


def test_opencode_policy_require_auth_raises_when_auth_is_none() -> None:
    with pytest.raises(
        AgentCredentialFailureError, match="Missing OpenCode API key"
    ) as exc_info:
        policy_for_service("opencode").require_auth(None)
    assert exc_info.value.service_name == "opencode"


def test_opencode_policy_require_auth_raises_when_key_is_missing() -> None:
    with pytest.raises(AgentCredentialFailureError, match="Missing OpenCode API key"):
        policy_for_service("opencode").require_auth(ProviderAuth(opencode_api_key=None))


def test_opencode_policy_require_auth_passes_with_valid_key() -> None:
    policy_for_service("opencode").require_auth(
        ProviderAuth(opencode_api_key="key-xyz")
    )
