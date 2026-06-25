from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent_runtime import _time as time_runtime
from agent_runtime._builtin_provider_stream_interpretation import (
    classify_built_in_provider_invocation_progress,
    codex_built_in_provider_stream_interpretation,
)
from agent_runtime.errors import (
    AgentCredentialFailureError,
    HardAgentError,
    TransientAgentError,
    UsageLimitError,
)
from agent_runtime.invocation_progress import InvocationProgress


def test_codex_built_in_provider_stream_interpretation_maps_error_usage_limit_with_dateless_reset_time(
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
                    {
                        "type": "error",
                        "message": (
                            "You've hit your usage limit. Try again at 5pm (UTC)."
                        ),
                    }
                )
                + "\n"
            ]
        )

    assert exc_info.value.service_name == "codex"
    assert exc_info.value.reset_time == datetime(2026, 1, 2, 17, 0, tzinfo=timezone.utc)
    assert exc_info.value.invocation_progress is InvocationProgress.NOT_STARTED


@pytest.mark.parametrize(
    ("event_type", "message", "expected_exception"),
    [
        ("error", "upstream status 503", TransientAgentError),
        ("turn.failed", "upstream status 503", TransientAgentError),
        ("error", "basic authentication failed", HardAgentError),
        ("turn.failed", "refresh_token_reused", AgentCredentialFailureError),
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


def test_codex_built_in_provider_stream_interpretation_builds_tool_call_event_from_item_started() -> (
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

    assert event.type == "agent_tool_call"
    assert event.display_message == 'shell({"command":"pwd"})'
    assert event.raw_provider_output == line


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
