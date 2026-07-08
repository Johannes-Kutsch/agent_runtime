from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent_runtime._builtin_provider_parsed_output import (
    classify_codex_invocation_progress,
    extract_codex_provider_session_id,
    parse_codex_event,
    parse_codex_usage,
)
from agent_runtime.contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    ModelUnavailable,
    TransientError,
    UsageLimit,
)
from agent_runtime.invocation_progress import InvocationProgress
from agent_runtime.provider_usage import ProviderUsage
from agent_runtime import _time as time_runtime


def _line(event: dict) -> str:
    return json.dumps(event)


# --- Assistant text ---


def test_codex_item_completed_agent_message_produces_assistant_turn() -> None:
    line = _line(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Hello from Codex"},
        }
    )

    result = parse_codex_event(line)

    assert result == [AssistantTurn(text="Hello from Codex")]


def test_codex_item_completed_agent_message_with_content_field_produces_assistant_turn() -> (
    None
):
    line = _line(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "content": "Hello via content"},
        }
    )

    result = parse_codex_event(line)

    assert result == [AssistantTurn(text="Hello via content")]


def test_codex_item_completed_non_agent_message_produces_no_events() -> None:
    line = _line(
        {
            "type": "item.completed",
            "item": {"type": "tool_call", "text": "some tool"},
        }
    )

    result = parse_codex_event(line)

    assert result == []


# --- Token counts / ProviderUsage ---


def test_codex_turn_completed_with_usage_produces_provider_usage() -> None:
    line = _line(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_tokens": 20,
                "output_tokens": 50,
            },
        }
    )

    result = parse_codex_usage(line)

    assert result == ProviderUsage(
        input_tokens=100,
        cache_read_input_tokens=20,
        output_tokens=50,
    )


def test_codex_turn_completed_with_partial_usage_produces_provider_usage() -> None:
    line = _line(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 75},
        }
    )

    result = parse_codex_usage(line)

    assert result == ProviderUsage(input_tokens=75)


def test_codex_non_turn_completed_event_produces_no_usage() -> None:
    line = _line(
        {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}
    )

    result = parse_codex_usage(line)

    assert result is None


# --- Usage-limit without reset date ---


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_usage_limit_line_without_reset_date_produces_usage_limit_with_no_reset_time(
    event_type: str,
) -> None:
    message = "You've hit your usage limit."
    if event_type == "error":
        line = _line({"type": "error", "message": message})
    else:
        line = _line({"type": "turn.failed", "error": {"message": message}})

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, UsageLimit)
    assert event.reset_time is None


# --- Usage-limit with reset date ---


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_usage_limit_line_with_reset_date_produces_usage_limit_with_reset_time(
    event_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        time_runtime,
        "now_local",
        lambda: datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc),
    )
    message = "You've hit your usage limit. Resets at 5pm (UTC)."
    if event_type == "error":
        line = _line({"type": "error", "message": message})
    else:
        line = _line({"type": "turn.failed", "error": {"message": message}})

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, UsageLimit)
    assert event.reset_time == datetime(2026, 7, 8, 17, 0, tzinfo=timezone.utc)


# --- At-capacity lines ---


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_at_capacity_line_produces_transient_error(event_type: str) -> None:
    message = "The selected model is at capacity. Please try again later."
    if event_type == "error":
        line = _line({"type": "error", "message": message})
    else:
        line = _line({"type": "turn.failed", "error": {"message": message}})

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, TransientError)


# --- Auth lineage error lines ---


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_refresh_token_reused_produces_credential_failure(
    event_type: str,
) -> None:
    message = "Auth failed: refresh_token_reused - your session has expired"
    if event_type == "error":
        line = _line({"type": "error", "message": message})
    else:
        line = _line({"type": "turn.failed", "error": {"message": message}})

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, CredentialFailure)
    assert event.classification == "codex_auth_lineage_exhausted"


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_access_token_refresh_failed_with_revoked_token_produces_credential_failure(
    event_type: str,
) -> None:
    message = "access token could not be refreshed: refresh token was revoked"
    if event_type == "error":
        line = _line({"type": "error", "message": message})
    else:
        line = _line({"type": "turn.failed", "error": {"message": message}})

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, CredentialFailure)
    assert event.classification == "codex_auth_lineage_exhausted"


# --- Generic auth error lines ---


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_generic_auth_error_produces_hard_error(event_type: str) -> None:
    message = "Request failed with 401: unauthorized access"
    if event_type == "error":
        line = _line({"type": "error", "message": message})
    else:
        line = _line({"type": "turn.failed", "error": {"message": message}})

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, HardError)
    assert event.status_code == 401


# --- Account model restriction lines ---


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_model_restriction_error_produces_model_unavailable(
    event_type: str,
) -> None:
    message = json.dumps(
        {
            "status": 400,
            "error": {
                "type": "invalid_request_error",
                "message": "The model is not available for your account",
            },
        }
    )
    if event_type == "error":
        line = _line({"type": "error", "message": message})
    else:
        line = _line({"type": "turn.failed", "error": {"message": message}})

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, ModelUnavailable)
    assert event.service_name == "codex"


# --- Thread-id lines ---


def test_codex_thread_started_line_produces_provider_session_id() -> None:
    lines = [_line({"type": "thread.started", "thread_id": "thread-abc123"})]

    result = extract_codex_provider_session_id(lines)

    assert result == "thread-abc123"


def test_codex_multiple_lines_with_single_thread_id_returns_that_thread_id() -> None:
    lines = [
        _line(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}
        ),
        _line({"type": "thread.started", "thread_id": "thread-xyz"}),
        _line({"type": "turn.completed", "usage": {"output_tokens": 5}}),
    ]

    result = extract_codex_provider_session_id(lines)

    assert result == "thread-xyz"


def test_codex_no_thread_started_line_returns_none() -> None:
    lines = [
        _line(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}
        )
    ]

    result = extract_codex_provider_session_id(lines)

    assert result is None


# --- Progress-start lines ---


def test_codex_lines_with_assistant_turn_produce_invocation_progress_started() -> None:
    lines = [
        _line(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Hello"},
            }
        )
    ]

    result = classify_codex_invocation_progress(lines)

    assert result == InvocationProgress.STARTED


def test_codex_lines_without_assistant_turn_or_result_produce_invocation_progress_not_started() -> (
    None
):
    lines = [
        _line({"type": "thread.started", "thread_id": "thread-abc"}),
        _line({"type": "turn.completed", "usage": {"output_tokens": 10}}),
    ]

    result = classify_codex_invocation_progress(lines)

    assert result == InvocationProgress.NOT_STARTED


def test_codex_lines_with_only_error_facts_produce_invocation_progress_not_started() -> (
    None
):
    lines = [
        _line(
            {
                "type": "error",
                "message": "The selected model is at capacity. Please try again later.",
            }
        )
    ]

    result = classify_codex_invocation_progress(lines)

    assert result == InvocationProgress.NOT_STARTED


# --- Edge cases ---


def test_codex_invalid_json_line_produces_no_events() -> None:
    result = parse_codex_event("not valid json {{{")

    assert result == []


def test_codex_turn_completed_without_usage_field_produces_no_usage() -> None:
    line = _line({"type": "turn.completed"})

    result = parse_codex_usage(line)

    assert result is None


def test_codex_turn_completed_with_empty_usage_dict_produces_no_usage() -> None:
    line = _line({"type": "turn.completed", "usage": {}})

    result = parse_codex_usage(line)

    assert result is None


def test_codex_multiple_different_thread_ids_produces_no_session_id() -> None:
    lines = [
        _line({"type": "thread.started", "thread_id": "thread-aaa"}),
        _line({"type": "thread.started", "thread_id": "thread-bbb"}),
    ]

    result = extract_codex_provider_session_id(lines)

    assert result is None


def test_codex_unrecognized_error_without_http_status_produces_hard_error_500() -> None:
    line = _line(
        {"type": "error", "message": "Something went wrong with no status code"}
    )

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, HardError)
    assert event.status_code == 500


def test_codex_server_error_with_5xx_status_produces_transient_error() -> None:
    line = _line(
        {
            "type": "error",
            "message": "Request failed with status 503 service unavailable",
        }
    )

    result = parse_codex_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, TransientError)
    assert event.status_code == 503
