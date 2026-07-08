from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent_runtime._builtin_provider_parsed_output import (
    classify_opencode_invocation_progress,
    classify_opencode_output_line,
    extract_opencode_provider_session_id,
    parse_opencode_event,
    parse_opencode_events,
    parse_opencode_reset_time,
)
from agent_runtime.contracts import (
    AssistantTurn,
    CredentialFailure,
    HardError,
    Result,
    TransientError,
    UsageLimit,
)
from agent_runtime.invocation_progress import InvocationProgress


def _line(event: dict) -> str:
    return json.dumps(event)


# --- Assistant text ---


def test_opencode_completed_text_part_produces_assistant_turn() -> None:
    line = _line(
        {
            "type": "text",
            "part": {"type": "text", "text": "Hello from OpenCode", "time": {"end": 1}},
        }
    )

    result = parse_opencode_event(line)

    assert result == [AssistantTurn(text="Hello from OpenCode")]


def test_opencode_text_part_strips_surrounding_whitespace() -> None:
    line = _line(
        {
            "type": "text",
            "part": {"type": "text", "text": "  padded  ", "time": {"end": 1}},
        }
    )

    result = parse_opencode_event(line)

    assert result == [AssistantTurn(text="padded")]


def test_opencode_text_part_without_end_time_produces_no_events() -> None:
    line = _line(
        {
            "type": "text",
            "part": {"type": "text", "text": "in progress", "time": {"start": 1}},
        }
    )

    result = parse_opencode_event(line)

    assert result == []


def test_opencode_text_part_with_null_end_time_produces_no_events() -> None:
    line = _line(
        {
            "type": "text",
            "part": {"type": "text", "text": "in progress", "time": {"end": None}},
        }
    )

    result = parse_opencode_event(line)

    assert result == []


def test_opencode_non_text_part_type_produces_no_events() -> None:
    line = _line(
        {
            "type": "text",
            "part": {"type": "tool", "name": "Read", "input": {}},
        }
    )

    result = parse_opencode_event(line)

    assert result == []


def test_opencode_whitespace_only_text_produces_no_events() -> None:
    line = _line(
        {
            "type": "text",
            "part": {"type": "text", "text": "   \n  ", "time": {"end": 1}},
        }
    )

    result = parse_opencode_event(line)

    assert result == []


# --- Idle termination / Result ---


def test_opencode_idle_status_after_text_produces_result() -> None:
    lines = [
        _line(
            {
                "type": "text",
                "part": {"type": "text", "text": "first part", "time": {"end": 1}},
            }
        ),
        _line(
            {
                "type": "text",
                "part": {"type": "text", "text": "second part", "time": {"end": 2}},
            }
        ),
        _line({"type": "session.status", "status": {"type": "idle"}}),
    ]

    result = parse_opencode_events(lines)

    assert AssistantTurn(text="first part") in result
    assert AssistantTurn(text="second part") in result
    result_events = [e for e in result if isinstance(e, Result)]
    assert len(result_events) == 1
    assert result_events[0].text == "first part\n\nsecond part"


def test_opencode_idle_status_without_prior_text_produces_no_result() -> None:
    lines = [
        _line({"type": "session.status", "status": {"type": "idle"}}),
    ]

    result = parse_opencode_events(lines)

    assert not any(isinstance(e, Result) for e in result)


# --- Trailing-error suppression after idle ---


def test_opencode_error_after_idle_is_suppressed() -> None:
    lines = [
        _line(
            {
                "type": "text",
                "part": {"type": "text", "text": "done", "time": {"end": 1}},
            }
        ),
        _line({"type": "session.status", "status": {"type": "idle"}}),
        _line(
            {
                "type": "error",
                "error": {
                    "name": "InternalServerError",
                    "data": {"message": "ignored trailing error", "statusCode": 503},
                },
            }
        ),
    ]

    result = parse_opencode_events(lines)

    assert not any(isinstance(e, (TransientError, HardError)) for e in result)
    result_events = [e for e in result if isinstance(e, Result)]
    assert len(result_events) == 1
    assert result_events[0].text == "done"


# --- Usage-limit without reset date ---


def test_opencode_rate_limit_error_without_parseable_reset_time_produces_usage_limit() -> (
    None
):
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "RateLimitError",
                "data": {
                    "message": "You have reached your OpenCode Go usage limit.",
                    "statusCode": 429,
                },
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, UsageLimit)
    assert event.reset_time is None


# --- Usage-limit with reset date ---


def test_opencode_rate_limit_error_with_reset_date_produces_usage_limit_with_reset_time() -> (
    None
):
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "RateLimitError",
                "data": {
                    "message": (
                        "You have reached your OpenCode Go usage limit. "
                        "Try again at Apr 28th, 2026 9:02 PM."
                    ),
                    "statusCode": 429,
                },
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, UsageLimit)
    assert event.reset_time == datetime(2026, 4, 28, 21, 2, tzinfo=timezone.utc)
    assert event.raw_message is None


# --- Credential failure ---


def test_opencode_invalid_api_key_authentication_error_produces_credential_failure() -> (
    None
):
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "AuthenticationError",
                "data": {
                    "message": "invalid api key",
                    "statusCode": 401,
                },
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, CredentialFailure)
    assert event.service_name == "opencode"
    assert event.classification == "operator_actionable_agent_credential_failure"
    assert event.status_code == 401


def test_opencode_generic_401_without_authentication_error_name_produces_hard_error() -> (
    None
):
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "SomeOtherError",
                "data": {
                    "message": "unauthorized",
                    "statusCode": 401,
                },
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, HardError)
    assert event.status_code == 401


# --- Transient vs hard error by status code ---


def test_opencode_5xx_error_produces_transient_error() -> None:
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "InternalServerError",
                "data": {"message": "temporary backend failure", "statusCode": 503},
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, TransientError)
    assert event.status_code == 503


def test_opencode_4xx_error_not_rate_limit_or_auth_produces_hard_error() -> None:
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "BadRequestError",
                "data": {"message": "bad request parameter", "statusCode": 400},
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, HardError)
    assert event.status_code == 400


def test_opencode_model_not_found_without_status_code_produces_hard_error() -> None:
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "UnknownError",
                "data": {
                    "message": "model not found: opencode-go/deepseek-v4-flash. Did you mean: deepseek-v4-flash?"
                },
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, HardError)
    assert event.status_code == 400


def test_opencode_error_without_status_code_and_non_model_message_produces_transient_error() -> (
    None
):
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "NetworkError",
                "data": {"message": "connection reset by peer"},
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    event = result[0]
    assert isinstance(event, TransientError)
    assert event.status_code is None


# --- Provider session id ---


def test_opencode_line_with_session_id_produces_provider_session_id() -> None:
    lines = [
        _line({"type": "text", "sessionID": "sess-abc123", "part": {"type": "tool"}}),
    ]

    result = extract_opencode_provider_session_id(lines)

    assert result == "sess-abc123"


def test_opencode_lines_without_session_id_produces_none() -> None:
    lines = [
        _line(
            {
                "type": "text",
                "part": {"type": "text", "text": "hi", "time": {"end": 1}},
            }
        )
    ]

    result = extract_opencode_provider_session_id(lines)

    assert result is None


def test_opencode_multiple_lines_with_same_session_id_produces_that_id() -> None:
    lines = [
        _line({"type": "text", "sessionID": "sess-xyz", "part": {"type": "tool"}}),
        _line(
            {
                "type": "session.status",
                "sessionID": "sess-xyz",
                "status": {"type": "running"},
            }
        ),
    ]

    result = extract_opencode_provider_session_id(lines)

    assert result == "sess-xyz"


# --- Invocation progress ---


def test_opencode_lines_with_assistant_turn_produce_invocation_progress_started() -> (
    None
):
    lines = [
        _line(
            {
                "type": "text",
                "part": {"type": "text", "text": "hello", "time": {"end": 1}},
            }
        )
    ]

    result = classify_opencode_invocation_progress(lines)

    assert result == InvocationProgress.STARTED


def test_opencode_lines_without_assistant_turn_produce_invocation_progress_not_started() -> (
    None
):
    lines = [
        _line(
            {
                "type": "session.status",
                "sessionID": "sess-abc",
                "status": {"type": "running"},
            }
        ),
    ]

    result = classify_opencode_invocation_progress(lines)

    assert result == InvocationProgress.NOT_STARTED


def test_opencode_lines_with_only_error_facts_produce_invocation_progress_not_started() -> (
    None
):
    lines = [
        _line(
            {
                "type": "error",
                "error": {
                    "name": "InternalServerError",
                    "data": {"message": "server error", "statusCode": 503},
                },
            }
        )
    ]

    result = classify_opencode_invocation_progress(lines)

    assert result == InvocationProgress.NOT_STARTED


# --- Reset time parsing ---


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Try again at Apr 28th, 2026 9:02 PM.",
            datetime(2026, 4, 28, 21, 2, tzinfo=timezone.utc),
        ),
        (
            "Try again at January 1st, 2027 12:00 AM.",
            datetime(2027, 1, 1, 0, 0, tzinfo=timezone.utc),
        ),
        (
            "Try again at December 31st, 2026 12:00 PM.",
            datetime(2026, 12, 31, 12, 0, tzinfo=timezone.utc),
        ),
        (
            "You have reached your limit. Try again at Mar 2nd, 2026 9:00 AM.",
            datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc),
        ),
    ],
)
def test_opencode_reset_time_parser_extracts_datetime_from_message(
    text: str, expected: datetime
) -> None:
    result = parse_opencode_reset_time(text)

    assert result == expected


def test_opencode_reset_time_parser_returns_none_for_non_string() -> None:
    assert parse_opencode_reset_time(None) is None
    assert parse_opencode_reset_time(42) is None


def test_opencode_reset_time_parser_returns_none_when_pattern_absent() -> None:
    assert parse_opencode_reset_time("You have reached your usage limit.") is None


# --- Edge cases ---


def test_opencode_invalid_json_line_produces_no_events() -> None:
    result = parse_opencode_event("not valid json {{{")

    assert result == []


def test_opencode_non_dict_json_produces_no_events() -> None:
    result = parse_opencode_event(json.dumps([1, 2, 3]))

    assert result == []


def test_opencode_error_event_with_missing_data_field_produces_no_events() -> None:
    line = _line(
        {
            "type": "error",
            "error": {"name": "UnknownError"},
        }
    )

    result = parse_opencode_event(line)

    assert result == []


def test_opencode_unrecognized_event_type_produces_no_events() -> None:
    line = _line({"type": "step_finish", "step": {"tokens": {}}})

    result = parse_opencode_event(line)

    assert result == []


def test_opencode_authentication_error_with_non_api_key_message_produces_hard_error() -> (
    None
):
    line = _line(
        {
            "type": "error",
            "error": {
                "name": "AuthenticationError",
                "data": {
                    "message": "wrong credentials",
                    "statusCode": 401,
                },
            },
        }
    )

    result = parse_opencode_event(line)

    assert len(result) == 1
    assert isinstance(result[0], HardError)
    assert result[0].status_code == 401


def test_opencode_events_after_error_are_not_processed() -> None:
    lines = [
        _line(
            {
                "type": "error",
                "error": {
                    "name": "InternalServerError",
                    "data": {"message": "server failure", "statusCode": 503},
                },
            }
        ),
        _line(
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "unreachable text",
                    "time": {"end": 1},
                },
            }
        ),
    ]

    result = parse_opencode_events(lines)

    assert len(result) == 1
    assert isinstance(result[0], TransientError)
    assert not any(isinstance(e, AssistantTurn) for e in result)


def test_opencode_non_idle_session_status_does_not_stop_parsing() -> None:
    lines = [
        _line({"type": "session.status", "status": {"type": "running"}}),
        _line(
            {
                "type": "text",
                "part": {"type": "text", "text": "hello", "time": {"end": 1}},
            }
        ),
        _line({"type": "session.status", "status": {"type": "idle"}}),
    ]

    result = parse_opencode_events(lines)

    assert AssistantTurn(text="hello") in result
    result_events = [e for e in result if isinstance(e, Result)]
    assert len(result_events) == 1
    assert result_events[0].text == "hello"


# --- classify_opencode_output_line ---


def test_classify_opencode_output_line_invalid_json_is_not_json_object() -> None:
    session_id, is_terminal, is_json_object = classify_opencode_output_line(
        "not valid json {{{"
    )

    assert session_id is None
    assert is_terminal is False
    assert is_json_object is False


def test_classify_opencode_output_line_non_dict_json_is_not_json_object() -> None:
    session_id, is_terminal, is_json_object = classify_opencode_output_line(
        json.dumps([1, 2, 3])
    )

    assert session_id is None
    assert is_terminal is False
    assert is_json_object is False


def test_classify_opencode_output_line_idle_status_is_terminal_json_object() -> None:
    line = _line(
        {
            "type": "session.status",
            "sessionID": "sess_abc",
            "status": {"type": "idle"},
        }
    )

    session_id, is_terminal, is_json_object = classify_opencode_output_line(line)

    assert session_id == "sess_abc"
    assert is_terminal is True
    assert is_json_object is True


def test_classify_opencode_output_line_non_idle_status_is_not_terminal() -> None:
    line = _line(
        {
            "type": "session.status",
            "sessionID": "sess_abc",
            "status": {"type": "running"},
        }
    )

    session_id, is_terminal, is_json_object = classify_opencode_output_line(line)

    assert session_id == "sess_abc"
    assert is_terminal is False
    assert is_json_object is True


def test_classify_opencode_output_line_error_event_is_terminal_json_object() -> None:
    line = _line(
        {
            "type": "error",
            "sessionID": "sess_abc",
            "error": {"name": "InternalServerError", "data": {"statusCode": 503}},
        }
    )

    session_id, is_terminal, is_json_object = classify_opencode_output_line(line)

    assert session_id == "sess_abc"
    assert is_terminal is True
    assert is_json_object is True


def test_classify_opencode_output_line_regular_event_is_json_object_not_terminal() -> (
    None
):
    line = _line(
        {
            "type": "text",
            "sessionID": "sess_abc",
            "part": {"type": "text", "text": "hello", "time": {"end": 1}},
        }
    )

    session_id, is_terminal, is_json_object = classify_opencode_output_line(line)

    assert session_id == "sess_abc"
    assert is_terminal is False
    assert is_json_object is True


def test_classify_opencode_output_line_missing_session_id_yields_none() -> None:
    line = _line({"type": "text", "part": {"type": "text", "text": "hi"}})

    session_id, is_terminal, is_json_object = classify_opencode_output_line(line)

    assert session_id is None
    assert is_json_object is True
