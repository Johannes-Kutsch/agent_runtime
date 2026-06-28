from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable, cast

import pytest

from agent_runtime import _time as time_runtime
from agent_runtime._builtin_provider_agent_event_building import (
    build_claude_agent_event,
    build_codex_agent_event,
    build_opencode_agent_event,
)
from agent_runtime._builtin_provider_stream_interpretation import (
    BuiltInProviderStreamInterpretation,
    classify_built_in_provider_invocation_progress,
    claude_built_in_provider_stream_interpretation,
    codex_built_in_provider_stream_interpretation,
    observe_opencode_output,
    opencode_built_in_provider_stream_interpretation,
    reduce_opencode_stream,
)
from agent_runtime._runtime_lifecycle import AgentEvent, ProviderUsage
from agent_runtime.errors import (
    AgentCredentialFailureError,
    HardAgentError,
    ProviderUnavailableError,
    ProviderUnavailableReason,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime.invocation_progress import InvocationProgress


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_built_in_provider_stream_interpretation_maps_usage_limit_with_dateless_reset_time(
    event_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        time_runtime,
        "now_local",
        lambda: datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc),
    )
    interpretation = codex_built_in_provider_stream_interpretation()

    with pytest.raises(UsageLimitError) as exc_info:
        interpretation.reduce_output(
            [
                json.dumps(
                    {"type": event_type}
                    | (
                        {
                            "message": (
                                "You've hit your usage limit. Try again at 5pm (UTC)."
                            )
                        }
                        if event_type == "error"
                        else {
                            "error": {
                                "message": (
                                    "You've hit your usage limit. "
                                    "Try again at 5pm (UTC)."
                                )
                            }
                        }
                    )
                )
                + "\n"
            ]
        )

    assert exc_info.value.service_name == "codex"
    assert exc_info.value.reset_time == datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc)
    assert exc_info.value.invocation_progress is InvocationProgress.NOT_STARTED


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_built_in_provider_stream_interpretation_maps_selected_model_at_capacity_to_retryable_failure(
    event_type: str,
) -> None:
    interpretation = codex_built_in_provider_stream_interpretation()
    message = "Selected model is at capacity. Please try a different model."

    with pytest.raises(ProviderUnavailableError) as exc_info:
        interpretation.reduce_output(
            [
                json.dumps(
                    {"type": event_type}
                    | (
                        {"message": message}
                        if event_type == "error"
                        else {"error": {"message": message}}
                    )
                )
                + "\n"
            ]
        )

    assert exc_info.value.reason is ProviderUnavailableReason.TRANSIENT_API_ERROR
    assert exc_info.value.service_name == "codex"
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("event_type", "message", "expected_exception"),
    [
        ("error", "upstream status 503", TransientAgentError),
        ("turn.failed", "upstream status 503", TransientAgentError),
        ("error", "basic authentication failed", HardAgentError),
        ("turn.failed", "refresh_token_reused", AgentCredentialFailureError),
        (
            "error",
            "Your access token could not be refreshed because your refresh token was revoked. Please log out and sign in again.",
            AgentCredentialFailureError,
        ),
        (
            "turn.failed",
            "Your access token could not be refreshed because your refresh token was revoked. Please log out and sign in again.",
            AgentCredentialFailureError,
        ),
    ],
)
def test_codex_built_in_provider_stream_interpretation_preserves_error_classification(
    event_type: str,
    message: str,
    expected_exception: type[Exception],
) -> None:
    interpretation = codex_built_in_provider_stream_interpretation()
    line = json.dumps(
        {
            "type": event_type,
            **(
                {"message": message}
                if event_type == "error"
                else {"error": {"message": message}}
            ),
        }
    )

    with pytest.raises(expected_exception) as exc_info:
        interpretation.reduce_output([line + "\n"])

    assert str(exc_info.value) == message
    if isinstance(exc_info.value, (AgentCredentialFailureError, HardAgentError)):
        assert exc_info.value.service_name == "codex"


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_built_in_provider_stream_interpretation_treats_unrecognized_message_as_hard_failure(
    event_type: str,
) -> None:
    interpretation = codex_built_in_provider_stream_interpretation()
    message = "The codex model service is temporarily in maintenance mode."

    with pytest.raises(HardAgentError) as exc_info:
        interpretation.reduce_output(
            [
                json.dumps(
                    {"type": event_type}
                    | (
                        {"message": message}
                        if event_type == "error"
                        else {"error": {"message": message}}
                    )
                )
                + "\n"
            ]
        )

    assert exc_info.value.service_name == "codex"
    assert str(exc_info.value) == message


def test_codex_built_in_provider_stream_interpretation_uses_codex_event_builder() -> (
    None
):
    interpretation = codex_built_in_provider_stream_interpretation()
    line = (
        json.dumps(
            {
                "type": "item.started",
                "item": {
                    "type": "shell",
                    "name": "shell",
                    "arguments": {"command": "pwd"},
                },
            }
        )
        + "\n"
    )

    event = interpretation.build_agent_event(line)

    assert event == build_codex_agent_event(line)


def test_claude_built_in_provider_stream_interpretation_reduce_output_preserves_live_agent_event_values() -> (
    None
):
    interpretation = claude_built_in_provider_stream_interpretation()
    reduce_output = cast(
        Callable[
            [list[str], Callable[[AgentEvent], None]],
            tuple[str, ProviderUsage | None],
        ],
        interpretation.reduce_output,
    )
    observed: list[AgentEvent] = []
    assistant_line = (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "hello from claude"}],
                    "usage": {"input_tokens": 12},
                },
            }
        )
        + "\n"
    )
    tool_line = (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    system_line = (
        json.dumps(
            {
                "type": "system",
                "subtype": "system.init",
                "cwd": "/workspace/project",
            }
        )
        + "\n"
    )
    result_line = (
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "final output",
            }
        )
        + "\n"
    )
    non_object_line = '"text"\n'
    raw_text_line = "  Reading prompt from stdin...  \n"

    output, usage = reduce_output(
        [
            assistant_line,
            tool_line,
            system_line,
            result_line,
            non_object_line,
            raw_text_line,
        ],
        observed.append,
    )

    assert output == "final output"
    assert usage is not None
    assert usage.input_tokens == 12
    assert observed == [
        build_claude_agent_event(assistant_line),
        build_claude_agent_event(tool_line),
        build_claude_agent_event(system_line),
        build_claude_agent_event(result_line),
        build_claude_agent_event(non_object_line),
        build_claude_agent_event(raw_text_line),
    ]


def test_codex_built_in_provider_stream_interpretation_extracts_provider_session_id_and_started_progress() -> (
    None
):
    interpretation = codex_built_in_provider_stream_interpretation()
    lines = ['{"type":"thread.started","thread_id":"thread-123"}\n']

    assert interpretation.extract_provider_session_id is not None
    assert interpretation.extract_provider_session_id(lines) == "thread-123"
    assert (
        classify_built_in_provider_invocation_progress(interpretation, lines)
        is InvocationProgress.STARTED
    )


@pytest.mark.parametrize(
    ("error_data", "expected_exception", "expected_message"),
    [
        (
            {
                "name": "AuthenticationError",
                "data": {"message": "invalid api key", "statusCode": 401},
            },
            AgentCredentialFailureError,
            "invalid api key",
        ),
        (
            {
                "name": "UnknownError",
                "data": {
                    "message": (
                        "Model not found: opencode-go/deepseek-v4-flash. "
                        "Did you mean: deepseek-v4-flash?"
                    )
                },
            },
            HardAgentError,
            (
                "Model not found: opencode-go/deepseek-v4-flash. "
                "Did you mean: deepseek-v4-flash?"
            ),
        ),
        (
            {
                "name": "InternalServerError",
                "data": {
                    "message": "temporary backend failure",
                    "statusCode": 503,
                },
            },
            TransientAgentError,
            "temporary backend failure",
        ),
    ],
)
def test_opencode_built_in_provider_stream_interpretation_preserves_error_classification(
    error_data: dict[str, object],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    interpretation = opencode_built_in_provider_stream_interpretation()
    line = json.dumps({"type": "error", "error": error_data}) + "\n"

    with pytest.raises(expected_exception) as exc_info:
        interpretation.reduce_output([line])

    assert str(exc_info.value) == expected_message
    if isinstance(exc_info.value, (AgentCredentialFailureError, HardAgentError)):
        assert exc_info.value.service_name == "opencode"


def test_opencode_built_in_provider_stream_interpretation_maps_usage_limit_and_extracts_provider_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        time_runtime,
        "now_local",
        lambda: datetime(2026, 4, 28, 20, 0, tzinfo=timezone.utc),
    )
    interpretation = opencode_built_in_provider_stream_interpretation()
    lines = [
        json.dumps(
            {
                "type": "error",
                "sessionID": "sess_123",
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
        + "\n"
    ]

    assert interpretation.extract_provider_session_id is not None
    assert interpretation.extract_provider_session_id(lines) == "sess_123"

    with pytest.raises(UsageLimitError) as exc_info:
        interpretation.reduce_output(lines)

    assert exc_info.value.service_name == "opencode"
    assert exc_info.value.reset_time == datetime(
        2026, 4, 28, 21, 2, tzinfo=timezone.utc
    )
    assert exc_info.value.invocation_progress is InvocationProgress.STARTED


def test_opencode_built_in_provider_stream_interpretation_keeps_completed_result_after_idle_status() -> (
    None
):
    interpretation = opencode_built_in_provider_stream_interpretation()
    lines = [
        json.dumps(
            {
                "type": "text",
                "sessionID": "sess_123",
                "part": {
                    "type": "text",
                    "text": "completed answer",
                    "time": {"start": 1, "end": 2},
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "session.status",
                "sessionID": "sess_123",
                "status": {"type": "idle"},
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "error",
                "sessionID": "sess_123",
                "error": {
                    "name": "InternalServerError",
                    "data": {
                        "message": "should be ignored after idle result",
                        "statusCode": 503,
                    },
                },
            }
        )
        + "\n",
    ]

    output, usage = interpretation.reduce_output(lines)

    assert output == "completed answer"
    assert usage is None


def test_opencode_built_in_provider_stream_interpretation_uses_opencode_event_builder() -> (
    None
):
    interpretation = opencode_built_in_provider_stream_interpretation()
    line = (
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "tool",
                    "name": "Read",
                    "input": {"path": "README.md"},
                },
            }
        )
        + "\n"
    )

    event = interpretation.build_agent_event(line)

    assert event == build_opencode_agent_event(line)


def test_reduce_opencode_stream_preserves_live_agent_event_values() -> None:
    interpretation = opencode_built_in_provider_stream_interpretation()
    observed: list[AgentEvent] = []
    text_line = (
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "hello from opencode",
                    "time": {"end": True},
                },
            }
        )
        + "\n"
    )
    tool_line = (
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "tool",
                    "name": "Read",
                    "input": {"path": "README.md"},
                },
            }
        )
        + "\n"
    )
    summary_line = (
        json.dumps(
            {
                "type": "step_finish",
                "step": {
                    "tokens": {
                        "input": 120,
                        "output": 45,
                        "reasoning": 12,
                        "cache": {"read": 30, "write": 8},
                    },
                    "cost": 0.0123,
                },
            }
        )
        + "\n"
    )
    idle_line = (
        json.dumps({"type": "session.status", "status": {"type": "idle"}}) + "\n"
    )
    non_object_line = '"text"\n'
    raw_text_line = "  not json  \n"
    lines = [
        text_line,
        tool_line,
        summary_line,
        idle_line,
        non_object_line,
        raw_text_line,
    ]

    output, usage = reduce_opencode_stream(lines, observed.append)

    assert output == "hello from opencode"
    assert usage is None
    assert observed == [interpretation.build_agent_event(line) for line in lines]


def test_opencode_observation_emits_live_agent_events_and_tracks_provider_session_id_until_idle() -> (
    None
):
    interpretation = opencode_built_in_provider_stream_interpretation()
    observed: list[AgentEvent] = []
    observed_provider_session_ids: list[str] = []
    observe_output = observe_opencode_output(
        stream_interpretation=interpretation,
        on_live_output=observed.append,
        on_provider_session_id=observed_provider_session_ids.append,
    )
    text_line = (
        json.dumps(
            {
                "type": "text",
                "sessionID": "sess_123",
                "part": {
                    "type": "text",
                    "text": "hello from opencode",
                    "time": {"end": True},
                },
            }
        )
        + "\n"
    )
    idle_line = (
        json.dumps(
            {
                "type": "session.status",
                "sessionID": "sess_123",
                "status": {"type": "idle"},
            }
        )
        + "\n"
    )
    trailing_error_line = (
        json.dumps(
            {
                "type": "error",
                "sessionID": "sess_456",
                "error": {
                    "name": "InternalServerError",
                    "data": {"message": "ignored after idle", "statusCode": 503},
                },
            }
        )
        + "\n"
    )

    observe_output([text_line, idle_line, trailing_error_line])

    assert [event.type for event in observed] == ["agent_message", "other"]
    assert [event.display_message for event in observed] == [
        "hello from opencode",
        "idle",
    ]
    assert observed_provider_session_ids == ["sess_123"]


def test_opencode_observation_uses_stream_interpretation_builder_for_live_events() -> (
    None
):
    observed: list[AgentEvent] = []
    observed_provider_session_ids: list[str] = []
    interpretation = BuiltInProviderStreamInterpretation(
        reduce_output=lambda _lines: ("", None),
        build_agent_event=lambda line: AgentEvent(
            type="other",
            display_message=f"observed:{line.strip()}",
            raw_provider_output=f"wrapped:{line}",
        ),
        classify_invocation_progress=lambda _lines: InvocationProgress.NOT_STARTED,
    )
    observe_output = observe_opencode_output(
        stream_interpretation=interpretation,
        on_live_output=observed.append,
        on_provider_session_id=observed_provider_session_ids.append,
    )
    text_line = (
        json.dumps(
            {
                "type": "text",
                "sessionID": "sess_123",
                "part": {
                    "type": "text",
                    "text": "hello from opencode",
                    "time": {"end": True},
                },
            }
        )
        + "\n"
    )
    idle_line = (
        json.dumps(
            {
                "type": "session.status",
                "sessionID": "sess_123",
                "status": {"type": "idle"},
            }
        )
        + "\n"
    )
    trailing_error_line = (
        json.dumps(
            {
                "type": "error",
                "sessionID": "sess_456",
                "error": {
                    "name": "InternalServerError",
                    "data": {"message": "ignored after idle", "statusCode": 503},
                },
            }
        )
        + "\n"
    )

    observe_output([text_line, idle_line, trailing_error_line])

    assert observed == [
        AgentEvent(
            type="other",
            display_message=f"observed:{text_line.strip()}",
            raw_provider_output=f"wrapped:{text_line}",
        ),
        AgentEvent(
            type="other",
            display_message=f"observed:{idle_line.strip()}",
            raw_provider_output=f"wrapped:{idle_line}",
        ),
    ]
    assert observed_provider_session_ids == ["sess_123"]
