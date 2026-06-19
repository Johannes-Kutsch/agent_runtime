from __future__ import annotations

from pathlib import Path

import pytest

import agent_runtime as runtime
import agent_runtime.runtime as prompt_runtime
from agent_runtime._portable_continuation_payload import (
    create_portable_continuation_payload,
    read_portable_continuation_payload,
)


def test_portable_continuation_payload_round_trips_current_continuation_contents() -> (
    None
):
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.workspace_backed(
            Path("/repo"),
            tool_policy=runtime.ToolPolicy.PARTIAL,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
            "metadata": {"attempt": 1, "notes": None},
        },
    )

    payload = read_portable_continuation_payload(continuation)

    assert payload.service_name == "codex"
    assert payload.model == "gpt-5.4"
    assert payload.effort == "medium"
    assert payload.tool_access == continuation.tool_access
    assert payload.provider_resume_state == continuation.provider_resume_state
    assert payload.to_continuation() == continuation


def test_portable_continuation_payload_create_keeps_current_continuation_schema() -> (
    None
):
    tool_access = runtime.ToolAccess.no_tools()

    payload = create_portable_continuation_payload(
        service_name="claude",
        model="sonnet",
        effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "session-123",
            "provider_state_dir_relpath": "implementer/main/claude/",
            "exact_transcript_match": False,
        },
    )

    assert payload.to_continuation() == prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "session-123",
            "provider_state_dir_relpath": "implementer/main/claude/",
            "exact_transcript_match": False,
        },
    )


def test_portable_continuation_payload_rejects_non_object_provider_resume_state() -> (
    None
):
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5",
        selected_effort="medium",
        tool_access=runtime.ToolAccess.no_tools(),
        provider_resume_state=["resume"],
    )

    with pytest.raises(TypeError) as exc_info:
        read_portable_continuation_payload(continuation)

    assert str(exc_info.value) == (
        "Continuation provider_resume_state must be a JSON object."
    )
