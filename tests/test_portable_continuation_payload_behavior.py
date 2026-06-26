from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
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
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            Path("/repo"),
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
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


def test_continuation_exposes_portable_resume_facts_through_module_interface() -> None:
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=tool_access,
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
        },
    )

    assert continuation.service_name == "codex"
    assert continuation.model == "gpt-5.4"
    assert continuation.effort == "medium"
    assert continuation.tool_access == tool_access
    assert continuation.provider_resume_state == {
        "run_kind": "resume",
        "provider_session_id": "thread-123",
        "provider_state_dir_relpath": "implementer/main/codex/",
    }


def test_portable_continuation_payload_create_keeps_current_continuation_schema() -> (
    None
):
    tool_access = contracts_runtime.ToolAccess.no_tools()

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


def test_portable_continuation_payload_serialization_omits_provider_selection_and_credentials() -> (
    None
):
    payload = create_portable_continuation_payload(
        service_name="opencode",
        model="glm-5.2",
        effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "session-123",
            "auth": "provider-owned token marker",
        },
    )

    serialized_payload = json.loads(payload.serialized)

    assert serialized_payload == {
        "service_name": "opencode",
        "model": "glm-5.2",
        "effort": "medium",
        "tool_access": {
            "kind": "none",
            "workspace": None,
            "tool_policy": {"kind": "tool_policy", "value": "none"},
        },
        "provider_resume_state": {
            "provider_session_id": "session-123",
            "auth": "provider-owned token marker",
        },
    }
    assert "provider_selection" not in serialized_payload
    assert "provider_auth" not in serialized_payload


def test_portable_continuation_payload_rejects_non_object_provider_resume_state() -> (
    None
):
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state=["resume"],
    )

    with pytest.raises(TypeError) as exc_info:
        read_portable_continuation_payload(continuation)

    assert str(exc_info.value) == (
        "Continuation provider_resume_state must be a JSON object."
    )


def test_portable_continuation_payload_rejects_legacy_inspect_only_tool_policy() -> (
    None
):
    raw_payload = json.dumps(
        {
            "service_name": "codex",
            "model": "gpt-5.4",
            "effort": "low",
            "tool_access": {
                "kind": "none",
                "workspace": None,
                "tool_policy": {
                    "kind": "tool_policy",
                    "value": "inspect_only",
                },
            },
            "provider_resume_state": {},
        }
    )
    continuation = prompt_runtime.Continuation(serialized=raw_payload)

    with pytest.raises(TypeError, match="legacy tool-policy value `inspect_only`"):
        read_portable_continuation_payload(continuation)


@pytest.mark.parametrize(
    "attribute", ["service_name", "model", "effort", "tool_access"]
)
def test_continuation_properties_preserve_malformed_token_errors(
    attribute: str,
) -> None:
    continuation = prompt_runtime.Continuation(serialized="{not-json")

    with pytest.raises(TypeError, match="Continuation data is not valid JSON."):
        getattr(continuation, attribute)


@pytest.mark.parametrize("attribute", ["service_name", "tool_access"])
def test_continuation_properties_reject_legacy_inspect_only_tool_policy(
    attribute: str,
) -> None:
    continuation = prompt_runtime.Continuation(
        serialized=json.dumps(
            {
                "service_name": "codex",
                "model": "gpt-5.4",
                "effort": "low",
                "tool_access": {
                    "kind": "none",
                    "workspace": None,
                    "tool_policy": {
                        "kind": "tool_policy",
                        "value": "inspect_only",
                    },
                },
                "provider_resume_state": {},
            }
        )
    )

    with pytest.raises(TypeError, match="legacy tool-policy value `inspect_only`"):
        getattr(continuation, attribute)


def test_portable_continuation_payload_rejects_unsupported_tool_policy_value() -> None:
    raw_payload = json.dumps(
        {
            "service_name": "codex",
            "model": "gpt-5.4",
            "effort": "low",
            "tool_access": {
                "kind": "none",
                "workspace": None,
                "tool_policy": {
                    "kind": "tool_policy",
                    "value": "workspace_write_only",
                },
            },
            "provider_resume_state": {},
        }
    )
    continuation = prompt_runtime.Continuation(serialized=raw_payload)

    with pytest.raises(
        TypeError,
        match="unsupported tool-policy value 'workspace_write_only'",
    ):
        read_portable_continuation_payload(continuation)
