from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent_runtime._builtin_provider_parsed_output import (
    parse_claude_event,
    parse_claude_usage,
)
from agent_runtime.contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    PromptTokens,
    Result,
    SessionGone,
    TransientError,
    UsageLimit,
)
from agent_runtime.provider_usage import ProviderUsage
from agent_runtime import _time as time_runtime


def _line(event: dict) -> str:
    return json.dumps(event)


# --- Assistant text ---


def test_claude_assistant_text_line_produces_assistant_turn() -> None:
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello world"}],
                "usage": {},
            },
        }
    )

    result = parse_claude_event(line)

    assert result == [AssistantTurn(text="Hello world")]


def test_claude_assistant_text_line_with_multiple_text_blocks_joins_them() -> None:
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ],
                "usage": {},
            },
        }
    )

    result = parse_claude_event(line)

    assert result == [AssistantTurn(text="First part\n\nSecond part")]


def test_claude_assistant_line_with_only_whitespace_text_produces_no_assistant_turn() -> (
    None
):
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "   "}],
                "usage": {},
            },
        }
    )

    result = parse_claude_event(line)

    assert result == []


# --- Final result text ---


def test_claude_result_line_produces_result_event() -> None:
    line = _line({"type": "result", "result": "Final answer", "is_error": False})

    result = parse_claude_event(line)

    assert result == [Result(text="Final answer")]


def test_claude_error_result_line_does_not_produce_result_event() -> None:
    line = _line({"type": "result", "result": "error text", "is_error": True})

    result = parse_claude_event(line)

    assert not any(isinstance(e, Result) for e in result)


# --- Token counts / ProviderUsage ---


def test_claude_assistant_line_with_token_counts_produces_provider_usage() -> None:
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [],
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                },
            },
        }
    )

    usage = parse_claude_usage(line)

    assert usage == ProviderUsage(
        input_tokens=100,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=30,
    )


def test_claude_assistant_line_with_partial_token_counts_produces_provider_usage() -> (
    None
):
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [],
                "usage": {"input_tokens": 50},
            },
        }
    )

    usage = parse_claude_usage(line)

    assert usage == ProviderUsage(input_tokens=50)


def test_claude_assistant_line_without_token_counts_produces_no_provider_usage() -> (
    None
):
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello"}],
                "usage": {},
            },
        }
    )

    usage = parse_claude_usage(line)

    assert usage is None


def test_claude_assistant_line_with_tokens_produces_prompt_tokens_event() -> None:
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [],
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 0,
                },
            },
        }
    )

    result = parse_claude_event(line)

    assert PromptTokens(count=15) in result


# --- Usage limit ---


def test_claude_usage_limit_line_produces_usage_limit_with_reset_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        time_runtime,
        "now_local",
        lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    line = _line(
        {
            "api_error_status": 429,
            "result": "You've exceeded your usage limit. It resets 5pm (UTC).",
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, UsageLimit)
    assert event.reset_time == datetime(2026, 1, 1, 17, 0, tzinfo=timezone.utc)
    assert event.raw_message is None


def test_claude_usage_limit_line_without_parseable_reset_time_preserves_raw_message() -> (
    None
):
    line = _line({"api_error_status": 429, "result": "Rate limited."})

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, UsageLimit)
    assert event.reset_time is None
    assert event.raw_message == line


def test_claude_usage_limit_line_with_month_day_reset_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        time_runtime,
        "now_local",
        lambda: datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
    )
    line = _line(
        {
            "api_error_status": 429,
            "result": "Usage limit reached. Resets March 2, 9am (UTC).",
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, UsageLimit)
    assert event.reset_time == datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)


# --- Credential failure ---


def test_claude_subscription_denial_line_produces_credential_failure() -> None:
    line = _line(
        {
            "is_error": True,
            "api_error_status": 403,
            "result": "Your organization has disabled Claude subscription access for Claude Code.",
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, CredentialFailure)
    assert event.service_name == "claude"
    assert event.status_code == 403


# --- Transient error ---


def test_claude_error_result_with_5xx_status_produces_transient_error() -> None:
    line = _line(
        {"type": "result", "is_error": True, "api_error_status": 500, "result": "oops"}
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, TransientError)
    assert event.status_code == 500


def test_claude_error_result_with_no_status_produces_transient_error() -> None:
    line = _line({"type": "result", "is_error": True, "result": "unknown error"})

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, TransientError)
    assert event.status_code is None


def test_claude_transient_error_is_distinct_from_hard_error() -> None:
    transient_line = _line(
        {"type": "result", "is_error": True, "api_error_status": 503, "result": "err"}
    )
    hard_line = _line(
        {"type": "result", "is_error": True, "api_error_status": 400, "result": "err"}
    )

    transient_result = parse_claude_event(transient_line)
    hard_result = parse_claude_event(hard_line)

    assert len(transient_result) == 1 and isinstance(
        transient_result[0], TransientError
    )
    assert len(hard_result) == 1 and isinstance(hard_result[0], HardError)


# --- Hard error ---


def test_claude_error_result_with_4xx_status_produces_hard_error() -> None:
    line = _line(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 400,
            "result": "bad request",
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, HardError)
    assert event.status_code == 400


def test_claude_error_result_with_401_produces_hard_error() -> None:
    line = _line(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 401,
            "result": "unauthorized",
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, HardError)
    assert event.status_code == 401


# --- SessionGone ---


def test_claude_session_gone_line_produces_session_gone() -> None:
    session_id = "abc-123"
    message = f"No conversation found with session id {session_id}"
    line = _line(
        {
            "type": "result",
            "is_error": True,
            "errors": [{"message": message}],
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, SessionGone)
    assert event.raw_message == message
    assert event.classification == "session_not_found"


def test_claude_session_gone_detection_is_case_insensitive() -> None:
    message = "NO CONVERSATION FOUND WITH SESSION ID abc-999"
    line = _line(
        {
            "type": "result",
            "is_error": True,
            "errors": [{"message": message}],
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    assert isinstance(result[0], SessionGone)


def test_claude_non_session_gone_error_with_500_produces_transient_error() -> None:
    line = _line(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 500,
            "errors": [{"message": "Something went wrong"}],
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    assert isinstance(result[0], TransientError)


# --- Combined text and token counts ---


def test_claude_assistant_line_with_text_and_tokens_produces_both_events() -> None:
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello"}],
                "usage": {"input_tokens": 10},
            },
        }
    )

    result = parse_claude_event(line)

    assert PromptTokens(count=10) in result
    assert AssistantTurn(text="Hello") in result


def test_claude_assistant_line_prompt_tokens_precede_assistant_turn() -> None:
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {"input_tokens": 5},
            },
        }
    )

    result = parse_claude_event(line)

    token_idx = next(i for i, e in enumerate(result) if isinstance(e, PromptTokens))
    turn_idx = next(i for i, e in enumerate(result) if isinstance(e, AssistantTurn))
    assert token_idx < turn_idx


# --- Non-text content blocks ---


def test_claude_assistant_line_with_only_tool_use_blocks_produces_no_assistant_turn() -> (
    None
):
    line = _line(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "tool_use", "name": "bash", "input": {}}],
                "usage": {},
            },
        }
    )

    result = parse_claude_event(line)

    assert not any(isinstance(e, AssistantTurn) for e in result)


# --- parse_claude_usage edge cases ---


def test_parse_claude_usage_returns_none_for_non_assistant_type() -> None:
    line = _line({"type": "result", "result": "done", "is_error": False})

    assert parse_claude_usage(line) is None


def test_parse_claude_usage_returns_none_for_missing_message_field() -> None:
    line = _line({"type": "assistant"})

    assert parse_claude_usage(line) is None


# --- Usage limit year rollover ---


def test_claude_usage_limit_month_day_reset_time_rolls_over_to_next_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        time_runtime,
        "now_local",
        lambda: datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )
    line = _line(
        {
            "api_error_status": 429,
            "result": "Usage limit reached. Resets March 15, 9am (UTC).",
        }
    )

    result = parse_claude_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, UsageLimit)
    assert event.reset_time == datetime(2027, 3, 15, 9, 0, tzinfo=timezone.utc)


# --- Invalid / unrecognized lines ---


def test_invalid_json_line_produces_no_events() -> None:
    result = parse_claude_event("not valid json")

    assert result == []


def test_unrecognized_event_type_produces_no_events() -> None:
    line = _line({"type": "unknown", "data": "whatever"})

    result = parse_claude_event(line)

    assert result == []
