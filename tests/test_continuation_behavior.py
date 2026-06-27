from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import cast

import pytest

import agent_runtime as runtime
import agent_runtime.contracts as contracts_runtime
import agent_runtime.runtime as prompt_runtime


def test_continuation_serialization_keeps_current_resume_token_schema() -> None:
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

    assert json.loads(continuation.serialized) == {
        "service_name": "codex",
        "model": "gpt-5.4",
        "effort": "medium",
        "tool_access": {
            "kind": "workspace_backed",
            "workspace": "/repo",
            "tool_policy": {
                "kind": "tool_policy",
                "value": "no_file_mutation",
            },
        },
        "provider_resume_state": {
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
            "exact_transcript_match": False,
            "metadata": {"attempt": 1, "notes": None},
        },
    }
    assert continuation.service_name == "codex"
    assert continuation.model == "gpt-5.4"
    assert continuation.effort == "medium"
    assert continuation.tool_access == contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )
    assert continuation.provider_resume_state == {
        "run_kind": "resume",
        "provider_session_id": "thread-123",
        "provider_state_dir_relpath": "implementer/main/codex/",
        "exact_transcript_match": False,
        "metadata": {"attempt": 1, "notes": None},
    }
    assert (
        prompt_runtime.Continuation(serialized=continuation.serialized) == continuation
    )


def test_continuation_serialization_normalizes_workspace_path_to_posix() -> None:
    workspace = cast(Path, PureWindowsPath(r"C:\repo\agent-runtime"))
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.workspace_backed(
            workspace,
            tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
        ),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "thread-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
        },
    )

    assert json.loads(continuation.serialized)["tool_access"]["workspace"] == (
        "C:/repo/agent-runtime"
    )


def test_continuation_exposes_resume_facts_through_interface() -> None:
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


@pytest.mark.parametrize(
    (
        "service",
        "model",
        "effort",
        "provider_session_id",
        "provider_state_dir_relpath",
        "exact_transcript_match",
        "run_kind",
        "expected_provider_resume_state",
    ),
    [
        (
            "codex",
            "gpt-5.4",
            "medium",
            "thread-123",
            "implementer/main/codex/",
            False,
            "resume",
            {
                "run_kind": "resume",
                "provider_session_id": "thread-123",
                "provider_state_dir_relpath": "implementer/main/codex/",
                "exact_transcript_match": False,
            },
        ),
        (
            "claude",
            "sonnet",
            "high",
            "claude-session-1",
            "implementer/main/claude/",
            False,
            "resume",
            {
                "run_kind": "resume",
                "provider_session_id": "claude-session-1",
                "provider_state_dir_relpath": "implementer/main/claude/",
                "exact_transcript_match": False,
            },
        ),
        (
            "opencode",
            "glm-5.2",
            "medium",
            "persisted-session-1",
            "implementer/main/opencode/",
            True,
            None,
            {
                "provider_session_id": "persisted-session-1",
                "provider_state_dir_relpath": "implementer/main/opencode/",
                "exact_transcript_match": True,
            },
        ),
    ],
)
def test_continuation_builds_session_backed_provider_resume_facts_through_module_interface(
    service: str,
    model: str,
    effort: str,
    provider_session_id: str,
    provider_state_dir_relpath: str,
    exact_transcript_match: bool,
    run_kind: str | None,
    expected_provider_resume_state: dict[str, object],
) -> None:
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=runtime.ToolPolicy.NO_FILE_MUTATION,
    )

    continuation = prompt_runtime.Continuation.for_session_backed_provider(
        selected_service=service,
        selected_model=model,
        selected_effort=effort,
        tool_access=tool_access,
        provider_session_id=provider_session_id,
        provider_state_dir_relpath=provider_state_dir_relpath,
        exact_transcript_match=exact_transcript_match,
        run_kind=run_kind,
    )

    assert continuation.provider_resume_state == expected_provider_resume_state
    assert continuation == prompt_runtime.Continuation(
        selected_service=service,
        selected_model=model,
        selected_effort=effort,
        tool_access=tool_access,
        provider_resume_state=expected_provider_resume_state,
    )
    assert continuation.session_backed_facts.selected == runtime.ResolvedProvider(
        service=service,
        model=model,
        effort=effort,
    )
    assert continuation.session_backed_facts.tool_access == tool_access
    assert (
        continuation.session_backed_facts.provider_resume_state
        == expected_provider_resume_state
    )
    assert continuation.session_backed_facts.provider_session_id == provider_session_id
    assert (
        continuation.session_backed_facts.provider_state_dir_relpath
        == provider_state_dir_relpath
    )
    assert (
        continuation.session_backed_facts.exact_transcript_match
        == exact_transcript_match
    )
    assert continuation.session_backed_facts.run_kind == run_kind


def test_continuation_serialized_round_trip_keeps_current_resume_token_schema() -> None:
    continuation = prompt_runtime.Continuation(
        selected_service="claude",
        selected_model="sonnet",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "run_kind": "resume",
            "provider_session_id": "session-123",
            "provider_state_dir_relpath": "implementer/main/claude/",
            "exact_transcript_match": False,
        },
    )

    assert prompt_runtime.Continuation(serialized=continuation.serialized) == (
        prompt_runtime.Continuation(
            selected_service="claude",
            selected_model="sonnet",
            selected_effort="medium",
            tool_access=contracts_runtime.ToolAccess.no_tools(),
            provider_resume_state={
                "run_kind": "resume",
                "provider_session_id": "session-123",
                "provider_state_dir_relpath": "implementer/main/claude/",
                "exact_transcript_match": False,
            },
        )
    )


def test_continuation_serialization_omits_provider_selection_and_credentials() -> None:
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state={
            "provider_session_id": "session-123",
            "auth": "provider-owned token marker",
        },
    )

    serialized_payload = json.loads(continuation.serialized)

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


def test_continuation_round_trips_tool_policy_profile_through_serialized_token() -> (
    None
):
    tool_access = contracts_runtime.ToolAccess.workspace_backed(
        Path("/repo"),
        tool_policy=contracts_runtime.ToolPolicyProfile(
            allowed_tools=("Read", "Glob"),
            disallowed_tools=("Edit",),
            strict_mcp_config=False,
        ),
    )
    continuation = prompt_runtime.Continuation(
        selected_service="codex",
        selected_model="gpt-5.4",
        selected_effort="high",
        tool_access=tool_access,
        provider_resume_state={
            "provider_session_id": "session-123",
            "provider_state_dir_relpath": "implementer/main/codex/",
        },
    )

    assert json.loads(continuation.serialized)["tool_access"]["tool_policy"] == {
        "kind": "tool_policy_profile",
        "allowed_tools": ["Read", "Glob"],
        "disallowed_tools": ["Edit"],
        "strict_mcp_config": False,
    }
    assert prompt_runtime.Continuation(
        serialized=continuation.serialized
    ).tool_access == (tool_access)


def test_continuation_resume_facts_reject_non_object_provider_resume_state() -> None:
    continuation = prompt_runtime.Continuation(
        selected_service="opencode",
        selected_model="glm-5.2",
        selected_effort="medium",
        tool_access=contracts_runtime.ToolAccess.no_tools(),
        provider_resume_state=["resume"],
    )

    with pytest.raises(TypeError) as exc_info:
        _ = continuation.resume_facts

    assert str(exc_info.value) == (
        "Continuation provider_resume_state must be a JSON object."
    )


def test_continuation_session_backed_facts_reject_non_object_provider_resume_state() -> (
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
        _ = continuation.session_backed_facts

    assert str(exc_info.value) == (
        "Continuation provider_resume_state must be a JSON object."
    )


def test_continuation_resume_facts_reject_legacy_inspect_only_tool_policy() -> None:
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
        _ = continuation.resume_facts


@pytest.mark.parametrize(
    "attribute", ["service_name", "model", "effort", "tool_access"]
)
def test_continuation_properties_preserve_malformed_token_errors(
    attribute: str,
) -> None:
    continuation = prompt_runtime.Continuation(serialized="{not-json")

    with pytest.raises(TypeError, match="Continuation data is not valid JSON."):
        getattr(continuation, attribute)


@pytest.mark.parametrize(
    ("field_name", "field_value", "attribute"),
    [
        ("service_name", 7, "service_name"),
        ("model", None, "model"),
        ("effort", [], "effort"),
    ],
)
def test_continuation_properties_reject_non_string_resume_token_fields(
    field_name: str,
    field_value: object,
    attribute: str,
) -> None:
    payload: dict[str, object] = {
        "service_name": "codex",
        "model": "gpt-5.4",
        "effort": "low",
        "tool_access": {
            "kind": "none",
            "workspace": None,
            "tool_policy": {
                "kind": "tool_policy",
                "value": "none",
            },
        },
        "provider_resume_state": {},
    }
    payload[field_name] = field_value
    continuation = prompt_runtime.Continuation(serialized=json.dumps(payload))

    with pytest.raises(TypeError, match="Continuation data is malformed."):
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


def test_continuation_resume_facts_reject_unsupported_tool_policy_value() -> None:
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
        _ = continuation.resume_facts
